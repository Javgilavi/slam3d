"""Geometry contracts: spherical projection, seam handling, fisheye models, transforms,
timestamp association. Uses the real Hilti 2026 calibration when present."""
from pathlib import Path

import numpy as np
import pytest

from slam3d.geometry import fisheye, sphere, transforms as tf

ROOT = Path(__file__).resolve().parents[1]
KALIBR = ROOT / "data/hilti/refs/kalibr_imucam_chain.yaml"


def test_equirect_roundtrip_pixels():
    W, H = 1920, 960
    rng = np.random.default_rng(1)
    u = rng.uniform(-0.5, W - 0.5, 5000)
    v = rng.uniform(0, H - 1, 5000)
    b = sphere.pixel_to_bearing(u, v, W, H)
    assert np.allclose(np.linalg.norm(b, axis=-1), 1.0)
    u2, v2 = sphere.bearing_to_pixel(b * rng.uniform(0.1, 50, (5000, 1)), W, H)
    du = np.abs(sphere.wrap_u(u2, W) - sphere.wrap_u(u, W))
    du = np.minimum(du, W - du)
    assert du.max() < 1e-6 and np.abs(v2 - v).max() < 1e-6


def test_equirect_axes_convention():
    W, H = 2000, 1000
    centre = sphere.pixel_to_bearing(W / 2 - 0.5, H / 2 - 0.5, W, H)
    assert np.allclose(centre, [0, 0, 1], atol=1e-9)
    right = sphere.pixel_to_bearing(0.75 * W - 0.5, H / 2 - 0.5, W, H)
    assert np.allclose(right, [1, 0, 0], atol=1e-9)
    top = sphere.pixel_to_bearing(W / 2 - 0.5, -0.5, W, H)  # top edge
    assert np.allclose(top, [0, -1, 0], atol=1e-9)


def test_stella_convention_matches_up_to_half_pixel():
    W, H = 1920, 960
    x, y = 123.0, 456.0  # stella keypoint coordinates
    lon = (x / W - 0.5) * 2 * np.pi
    lat = -(y / H - 0.5) * np.pi
    b_stella = np.array([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)])
    b_ours = sphere.pixel_to_bearing(x - 0.5, y - 0.5, W, H)
    assert np.allclose(b_stella, b_ours, atol=1e-12)


def test_seam_sampling_wraps():
    W, H = 64, 32
    img = np.zeros((H, W), np.float32)
    img[:, 0] = 1.0
    img[:, W - 1] = 3.0
    # halfway between last and first column must blend across the seam
    val = sphere.sample_equirect(img, np.array([[W - 0.5]]), np.array([[10.0]]))
    assert abs(float(val[0, 0]) - 2.0) < 1e-5
    val = sphere.sample_equirect(img, np.array([[-0.5]]), np.array([[10.0]]))
    assert abs(float(val[0, 0]) - 2.0) < 1e-5


def test_radial_vs_zdepth():
    W, H = 400, 200
    radial = np.full((H, W), 5.0)
    pts = sphere.radial_to_points(radial)
    assert np.allclose(np.linalg.norm(pts, axis=-1), 5.0)
    b = sphere.bearing_grid(W, H, np.float64)
    z = sphere.radial_to_zdepth(radial, b)
    assert np.allclose(z, pts[..., 2])
    assert z.max() <= 5.0 + 1e-9 and z.min() < 0  # rear hemisphere has negative z


def test_perspective_view_shares_centre_and_maps_back():
    W, H = 1024, 512
    rng = np.random.default_rng(0)
    img = rng.random((H, W, 3)).astype(np.float32)
    R = sphere.perspective_view_rotation(np.deg2rad(170), np.deg2rad(-20))  # crosses the seam
    K = sphere.perspective_intrinsics(90, 256)
    view, uvmap = sphere.equirect_to_perspective(img, R, K, 256)
    ray = np.linalg.inv(K) @ np.array([100, 60, 1.0])
    b = R @ ray
    u, v = sphere.bearing_to_pixel(b, W, H)
    assert np.allclose(sphere.wrap_u(uvmap[60, 100, 0], W), sphere.wrap_u(u, W), atol=1e-6)
    assert np.allclose(uvmap[60, 100, 1], v, atol=1e-6)
    # yaw positive turns towards +x
    fwd = sphere.perspective_view_rotation(np.deg2rad(90), 0) @ np.array([0, 0, 1.0])
    assert np.allclose(fwd, [1, 0, 0], atol=1e-12)


def test_kb4_roundtrip():
    cam = fisheye.KannalaBrandt4(465.3, 465.3, 730.0, 720.1, np.array([0.0258, -0.0109, -0.0017, 0.00015]))
    rng = np.random.default_rng(2)
    d = rng.normal(size=(2000, 3))
    d[:, 2] = np.abs(d[:, 2]) + 0.05
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    uv, ok = cam.project(d)
    back = cam.unproject(uv)
    assert np.allclose(back, d, atol=1e-8)


def test_eucm_roundtrip():
    cam = fisheye.EUCM(0.69, 0.89, 465.3, 465.3, 730.0, 720.1)
    rng = np.random.default_rng(3)
    d = rng.normal(size=(2000, 3))
    d[:, 2] = np.abs(d[:, 2]) + 0.05
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    uv, ok = cam.project(d)
    assert ok.all()
    back = cam.unproject(uv)
    assert np.allclose(back, d, atol=1e-8)


@pytest.mark.skipif(not KALIBR.exists(), reason="Hilti calibration not downloaded")
def test_hilti_kb4_fit_agrees_with_eucm():
    """The released KB4 intrinsics were fitted to EUCM; both must agree to ~1 px inside 95 deg."""
    kb = fisheye.load_kalibr_chain(str(KALIBR), "kb4")
    eu = fisheye.load_kalibr_chain(str(KALIBR), "eucm")
    for cam in ("cam0", "cam1"):
        th = np.deg2rad(np.linspace(0, 95, 40))
        ph = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        T, P = np.meshgrid(th, ph)
        d = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3)
        uk, _ = kb[cam]["model"].project(d)
        ue, _ = eu[cam]["model"].project(d)
        err = np.linalg.norm(uk - ue, axis=1)
        assert np.median(err) < 1.0 and err.max() < 3.0, (cam, np.median(err), err.max())
        assert np.allclose(kb[cam]["T_cam_imu"], eu[cam]["T_cam_imu"])


def test_umeyama_recovers_sim3_and_sim2():
    rng = np.random.default_rng(4)
    R = tf.quat_to_R(rng.normal(size=4))
    s, t = 2.7, np.array([1.0, -3.0, 0.5])
    src = rng.normal(size=(50, 3))
    dst = s * src @ R.T + t
    s2, R2, t2 = tf.umeyama(src, dst)
    assert np.isclose(s2, s) and np.allclose(R2, R) and np.allclose(t2, t)
    S = tf.sim2_from_params(0.01, 0.7, 30.0, 12.0)
    p = rng.normal(size=(20, 2)) * 100
    q = tf.apply_sim2(S, p)
    s3, R3, t3 = tf.umeyama(p, q)
    assert np.isclose(s3, 0.01) and np.isclose(np.arctan2(R3[1, 0], R3[0, 0]), 0.7) and np.allclose(t3, [30, 12])


def test_ransac_umeyama_rejects_outliers():
    rng = np.random.default_rng(5)
    src = rng.uniform(-10, 10, (40, 2))
    S = tf.sim2_from_params(1.5, -0.3, 2.0, 1.0)
    dst = tf.apply_sim2(S, src)
    dst[:8] += rng.uniform(5, 20, (8, 2))
    (s, R, t), inl = tf.ransac_umeyama(src, dst, 0.1)
    assert inl[8:].all() and not inl[:8].any() and np.isclose(s, 1.5)


def test_interpolation_and_association():
    t_ref = np.array([0.0, 1.0, 2.0])
    T = np.tile(np.eye(4), (3, 1, 1))
    T[:, 0, 3] = [0, 1, 2]
    T[1, :3, :3] = tf.quat_to_R([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])  # 90deg yaw
    out, ok = tf.interpolate_poses([0.5, 2.5], t_ref, T)
    assert ok.tolist() == [True, False]
    assert np.isclose(out[0, 0, 3], 0.5)
    assert np.isclose(np.rad2deg(np.arccos((np.trace(out[0, :3, :3]) - 1) / 2)), 45.0)
    ia, ib = tf.associate_timestamps(np.array([0.0, 0.034, 0.066, 1.0]), np.array([0.001, 0.033, 0.5]), 0.01)
    assert ia.tolist() == [0, 1] and ib.tolist() == [0, 1]


def test_apply_sim3_to_poses_preserves_relative_geometry():
    rng = np.random.default_rng(6)
    T = np.tile(np.eye(4), (5, 1, 1))
    T[:, :3, 3] = rng.normal(size=(5, 3))
    R = tf.quat_to_R([0.1, 0.2, 0.3, 0.9])
    out = tf.apply_sim3_to_poses(2.0, R, np.array([1, 2, 3.0]), T)
    d0 = np.linalg.norm(T[1, :3, 3] - T[0, :3, 3])
    d1 = np.linalg.norm(out[1, :3, 3] - out[0, :3, 3])
    assert np.isclose(d1, 2 * d0)
    assert np.allclose(out[:, :3, :3] @ out[:, :3, :3].transpose(0, 2, 1), np.eye(3), atol=1e-12)
