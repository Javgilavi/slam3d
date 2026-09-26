"""Contracts for scale/floor estimation and calibrated stitching."""
from pathlib import Path

import numpy as np
import pytest

from slam3d.geometry import sphere
from slam3d.geometry.gravity import G, camera_height_units, scale_from_imu
from slam3d.geometry.transforms import quat_to_R

ROOT = Path(__file__).resolve().parents[1]
KALIBR = ROOT / "data/hilti/refs/kalibr_imucam_chain.yaml"


def test_floor_layer_ignores_reflections_and_clutter():
    rng = np.random.default_rng(0)
    cams = np.column_stack([rng.uniform(0, 20, 300), rng.uniform(0, 10, 300), np.full(300, 1.5)])
    floor = np.column_stack([rng.uniform(-2, 22, 4000), rng.uniform(-2, 12, 4000), rng.normal(0.0, 0.01, 4000)])
    # glossy-floor reflections: MORE points, but concentrated (window/light reflections), below the floor
    refl = np.column_stack([rng.normal(10, 1.0, 6000), rng.normal(5, 0.5, 6000), rng.normal(-1.2, 0.02, 6000)])
    clutter = np.column_stack([rng.normal(5, 0.3, 800), rng.normal(5, 0.3, 800), rng.normal(0.45, 0.02, 800)])
    walls = np.column_stack([rng.uniform(0, 20, 3000), np.full(3000, 11.0), rng.uniform(0, 3, 3000)])
    pts = np.vstack([floor, refl, clutter, walls])
    h, info = camera_height_units(pts, cams, bin_size=0.05, cell=0.5)
    assert abs(h - 1.5) < 0.08, info


def test_imu_scale_recovers_known_scale():
    rng = np.random.default_rng(1)
    fs_imu, dur = 400.0, 60.0
    t = np.arange(0, dur, 1 / fs_imu)
    # smooth random 3D walk with sway (metres)
    freqs = [0.07, 0.13, 0.31, 0.9, 1.7]
    p = sum(np.column_stack([np.sin(2 * np.pi * f * t + ph) for ph in rng.uniform(0, 6, 3)]) * a
            for f, a in zip(freqs, [3.0, 1.5, 0.3, 0.05, 0.02]))
    acc = np.gradient(np.gradient(p, 1 / fs_imu, axis=0), 1 / fs_imu, axis=0)
    yaw = 0.3 * np.sin(2 * np.pi * 0.05 * t)
    R = np.stack([quat_to_R([0, 0, np.sin(y / 2), np.cos(y / 2)]) for y in yaw])  # T_grav_pano rotations
    R_pano_imu = quat_to_R([0.1, -0.2, 0.3, 0.9])
    f_body = np.einsum("nji,nj->ni", R @ R_pano_imu, acc + [0, 0, G]) + rng.normal(0, 0.05, acc.shape)
    imu = np.column_stack([t, np.zeros((len(t), 3)), f_body])
    s_true = 2.5  # metres per SLAM unit
    cam_t = t[:: int(fs_imu / 30)]
    T = np.tile(np.eye(4), (len(cam_t), 1, 1))
    T[:, :3, :3] = R[:: int(fs_imu / 30)]
    T[:, :3, 3] = p[:: int(fs_imu / 30)] / s_true + rng.normal(0, 0.002 / s_true, (len(cam_t), 3))
    s, info = scale_from_imu(imu, cam_t, T, R_pano_imu)
    assert abs(s / s_true - 1) < 0.1, info
    assert info["corr"] > 0.9


@pytest.mark.skipif(not KALIBR.exists(), reason="Hilti calibration not downloaded")
def test_stitch_maps_follow_calibration():
    from slam3d.ingest.stitch import build_maps

    W, H = 960, 480
    maps = build_maps(str(KALIBR), W, H, sphere_radius_m=3.0)
    assert maps.valid.mean() > 0.999
    # panorama centre (pano +z) == cam0 optical axis -> near cam0 principal point, fully from cam0
    c = (H // 2, W // 2)
    assert maps.w0[c] > 0.99
    assert np.hypot(maps.map0[c][0] - 730.0, maps.map0[c][1] - 720.1) < 5.0
    # rear direction comes from cam1
    assert maps.w1[H // 2, 5] > 0.99
    # upright: pano "up" (-y) is cam0 +y (sensor mounted upside down) -> top rows sample cam0 rows below cy
    b_top = sphere.pixel_to_bearing(W // 2, int(0.3 * H), W, H)
    assert b_top[1] < 0 and maps.map0[int(0.3 * H), W // 2][1] > 720.1
    # weights are a partition of unity
    assert np.allclose((maps.w0 + maps.w1)[maps.valid], 1.0, atol=1e-5)
