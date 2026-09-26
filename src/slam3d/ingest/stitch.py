"""Calibrated dual-fisheye -> equirectangular stitching (central-camera approximation).

Panorama frame definition (recorded in stitch.json):
  * origin = cam0 optical centre;
  * orientation R_cam0_pano (default diag(-1,-1,1): the Insta360 ONE RS 1-inch sensors are mounted
    upside down, so rotating 180 deg about the optical axis yields an upright panorama whose
    centre column looks along cam0's optical axis). Verified against IMU gravity in tests/validation.
  * cam1 rays use the calibrated cam1<-cam0 extrinsics. The ~cm lens baseline is handled by
    intersecting rays with a sphere of radius `sphere_radius_m`; content at other depths gets
    stitching parallax near the seam (documented limitation of stitched consumer panoramas).
  * blending: feathering by angle from each optical axis inside the overlap band; pixels covered by
    the static fisheye masks (white = invalid: outside the image circle) get zero weight.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from slam3d.geometry import sphere
from slam3d.geometry.fisheye import load_kalibr_chain
from slam3d.geometry.transforms import inv_T

R_CAM0_PANO_INSTA360_RS1 = np.diag([-1.0, -1.0, 1.0])


@dataclass
class StitchMaps:
    map0: np.ndarray  # (H, W, 2) float32 pixel coords in cam0 image
    map1: np.ndarray
    w0: np.ndarray  # (H, W) float32 blend weights
    w1: np.ndarray
    valid: np.ndarray  # (H, W) bool: any lens contributes
    meta: dict


def build_maps(calib_yaml: str, width: int, height: int, sphere_radius_m: float = 3.0,
               R_cam0_pano: np.ndarray = R_CAM0_PANO_INSTA360_RS1, model: str = "kb4",
               mask0: np.ndarray | None = None, mask1: np.ndarray | None = None,
               blend_lo_deg: float = 86.0, blend_hi_deg: float = 96.0) -> StitchMaps:
    cams = load_kalibr_chain(calib_yaml, model)
    c0, c1 = cams["cam0"], cams["cam1"]
    T_c1_c0 = c1["T_cam_imu"] @ inv_T(c0["T_cam_imu"])
    b_pano = sphere.bearing_grid(width, height, np.float64).reshape(-1, 3)
    b_c0 = b_pano @ R_cam0_pano.T
    p_c1 = (sphere_radius_m * b_c0) @ T_c1_c0[:3, :3].T + T_c1_c0[:3, 3]

    uv0, ok0 = c0["model"].project(b_c0)
    uv1, ok1 = c1["model"].project(p_c1)

    def theta(v):
        return np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1))

    th0, th1 = np.rad2deg(theta(b_c0)), np.rad2deg(theta(p_c1))

    def feather(th):
        return np.clip((blend_hi_deg - th) / (blend_hi_deg - blend_lo_deg), 0.0, 1.0)

    w0 = feather(th0) * ok0
    w1 = feather(th1) * ok1

    def apply_mask(w, uv, m):
        if m is None:
            return w
        mu = np.clip(np.round(uv[:, 0]).astype(int), 0, m.shape[1] - 1)
        mv = np.clip(np.round(uv[:, 1]).astype(int), 0, m.shape[0] - 1)
        return w * (m[mv, mu] < 128)

    w0 = apply_mask(w0, uv0, mask0)
    w1 = apply_mask(w1, uv1, mask1)
    s = w0 + w1
    valid = s > 1e-6
    w0 = np.where(valid, w0 / np.maximum(s, 1e-6), 0)
    w1 = np.where(valid, w1 / np.maximum(s, 1e-6), 0)
    H, W = height, width
    meta = {
        "projection": "equirectangular",
        "width": W, "height": H,
        "pano_frame": "origin=cam0 centre; axes x right, y down, z forward (centre column)",
        "R_cam0_pano": R_cam0_pano.tolist(),
        "T_cam1_cam0": T_c1_c0.tolist(),
        "T_cam0_imu": c0["T_cam_imu"].tolist(),
        "timeshift_cam_imu": c0["timeshift_cam_imu"],
        "sphere_radius_m": sphere_radius_m,
        "fisheye_model": model,
        "blend_deg": [blend_lo_deg, blend_hi_deg],
        "central_camera_approximation": True,
        "lens_baseline_m": float(np.linalg.norm(T_c1_c0[:3, 3])),
        "valid_fraction": float(valid.mean()),
    }
    return StitchMaps(uv0.reshape(H, W, 2).astype(np.float32), uv1.reshape(H, W, 2).astype(np.float32),
                      w0.reshape(H, W).astype(np.float32), w1.reshape(H, W).astype(np.float32),
                      valid.reshape(H, W), meta)


def stitch(img0: np.ndarray, img1: np.ndarray, maps: StitchMaps) -> np.ndarray:
    r0 = cv2.remap(img0, maps.map0[..., 0], maps.map0[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    r1 = cv2.remap(img1, maps.map1[..., 0], maps.map1[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if r0.ndim == 3:
        out = r0 * maps.w0[..., None] + r1 * maps.w1[..., None]
    else:
        out = r0 * maps.w0 + r1 * maps.w1
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------
# Parallel bag -> equirect frames
# ----------------------------------------------------------------------------------------------
_MAPS: StitchMaps | None = None
_OUT: Path | None = None
_Q: int = 92


def _init_worker(calib_yaml, width, height, radius, mask0_path, mask1_path, out_dir, quality):
    global _MAPS, _OUT, _Q
    cv2.setNumThreads(1)
    m0 = cv2.imread(mask0_path, cv2.IMREAD_GRAYSCALE) if mask0_path else None
    m1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE) if mask1_path else None
    _MAPS = build_maps(calib_yaml, width, height, radius, mask0=m0, mask1=m1)
    _OUT = Path(out_dir)
    _Q = quality


def _work(args):
    idx, j0, j1 = args
    a = cv2.imdecode(np.frombuffer(j0, np.uint8), cv2.IMREAD_COLOR)
    b = cv2.imdecode(np.frombuffer(j1, np.uint8), cv2.IMREAD_COLOR)
    pano = stitch(a, b, _MAPS)
    name = f"{idx:06d}.jpg"
    cv2.imwrite(str(_OUT / name), pano, [cv2.IMWRITE_JPEG_QUALITY, _Q])
    return idx, name, float(pano.mean())


def stitch_bag(bag_dir: str, calib_yaml: str, out_dir: str, width: int = 1920, radius: float = 3.0,
               topic0="/cam0/image_raw/compressed", topic1="/cam1/image_raw/compressed",
               mask0_path: str | None = None, mask1_path: str | None = None, workers: int = 16,
               quality: int = 92, max_frames: int | None = None, log=print) -> dict:
    """Stitch every synchronised pair in a bag to JPEG frames + frames.csv (idx,timestamp_s,path)."""
    from multiprocessing import get_context

    from slam3d.io.rosbag_hilti import iter_synced_pairs

    height = width // 2
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    m0 = cv2.imread(mask0_path, cv2.IMREAD_GRAYSCALE) if mask0_path else None
    m1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE) if mask1_path else None
    maps = build_maps(calib_yaml, width, height, radius, mask0=m0, mask1=m1)
    cv2.imwrite(str(out / "pano_valid_mask.png"), (maps.valid * 255).astype(np.uint8))
    stamps = []
    ctx = get_context("fork")
    results = []

    def gen():
        for i, (s0, s1, j0, j1) in enumerate(iter_synced_pairs(bag_dir, topic0, topic1)):
            if max_frames and i >= max_frames:
                break
            stamps.append((i, s0, s1))
            yield i, j0, j1

    with ctx.Pool(workers, _init_worker, (calib_yaml, width, height, radius, mask0_path, mask1_path,
                                          str(out / "frames"), quality)) as pool:
        for k, res in enumerate(pool.imap(_work, gen(), chunksize=4)):
            results.append(res)
            if k % 500 == 0:
                log(f"[stitch] {k} frames")
    brightness = {i: br for i, _, br in results}
    with open(out / "frames.csv", "w") as f:
        f.write("idx,timestamp_s,path,cam1_dt_ms,mean_intensity\n")
        for i, s0, s1 in stamps:
            f.write(f"{i},{s0 * 1e-9:.9f},frames/{i:06d}.jpg,{(s1 - s0) * 1e-6:.3f},{brightness.get(i, float('nan')):.1f}\n")
    meta = dict(maps.meta, n_frames=len(stamps), source_bag=str(bag_dir), calib=str(calib_yaml),
                timestamp_source="cam0 header stamp", masks=[mask0_path, mask1_path])
    (out / "stitch.json").write_text(json.dumps(meta, indent=2))
    return meta
