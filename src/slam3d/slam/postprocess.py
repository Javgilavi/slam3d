"""Post-process SLAM output into gravity-aligned, scale-annotated, quality-checked geometry.

Outputs in <run>/geometry/:
  trajectory_grav.tum     final (globally optimised) T_grav_pano, SLAM units
  online_grav.tum         online poses (as estimated causally), SLAM units
  keyframes_grav.csv      keyframe id, t, T_grav_pano (12 values)
  landmarks_grav.ply      coloured landmarks (SLAM units)
  geometry.json           frames, gravity estimate, scale estimate, tracking & reprojection statistics
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from slam3d.geometry import sphere
from slam3d.geometry.gravity import camera_height_units, gravity_rotation, up_from_imu, up_from_level_horizon
from slam3d.io.ply import write_ply
from slam3d.io.tum import read_tum, write_tum
from slam3d.slam import stella


def _segments(mask, t):
    """Contiguous True segments -> [(t_start, t_end, n)]."""
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((float(t[i]), float(t[j]), int(j - i + 1)))
            i = j + 1
        else:
            i += 1
    return out


def tracking_stats(on: stella.OnlinePoses) -> dict:
    tracked = on.valid & (on.state == "Tracking")
    lost = ~on.valid
    first = int(np.argmax(tracked)) if tracked.any() else None
    after = np.arange(len(tracked)) >= (first or 0)
    return {
        "n_frames": int(len(on.t)),
        "tracked_frac": float(tracked.mean()),
        "tracked_frac_after_init": float(tracked[after].mean()) if first is not None else 0.0,
        "init_time_s": float(on.t[first] - on.t[0]) if first is not None else None,
        "lost_segments_after_init": [s for s in _segments(lost & after, on.t) if s[2] >= 3],
        "loop_ba_frames": int(on.loop_ba.sum()),
        "track_ms_median": float(np.median(on.track_ms)),
        "track_ms_p95": float(np.percentile(on.track_ms, 95)),
        "n_tracked_median": float(np.median(on.n_tracked[tracked])) if tracked.any() else 0.0,
    }


def low_parallax_sections(t, P, landmarks_xyz, window_s=2.0, ratio_thresh=0.02):
    """Flag windows whose baseline is tiny relative to the median scene depth (rotation-only motion)."""
    from scipy.spatial import cKDTree

    if len(P) < 10 or len(landmarks_xyz) < 50:
        return []
    tree = cKDTree(landmarks_xyz)
    flags = np.zeros(len(t), bool)
    j = 0
    for i in range(len(t)):
        while t[j] < t[i] - window_s:
            j += 1
        base = np.linalg.norm(P[i] - P[j])
        dists, _ = tree.query(P[i], k=min(200, len(landmarks_xyz)))
        depth = float(np.median(dists))
        flags[i] = (t[i] - t[j] > 0.5 * window_s) and base / max(depth, 1e-9) < ratio_thresh
    return [s for s in _segments(flags, t) if s[1] - s[0] >= 1.0]


def reprojection_stats(kfs, lms, kf_obs, width, height):
    id2i = {int(k): i for i, k in enumerate(kfs["id"])}
    lm2i = {int(k): i for i, k in enumerate(lms["id"])}
    ki = np.array([id2i.get(int(k), -1) for k in kf_obs["kf"]])
    li = np.array([lm2i.get(int(k), -1) for k in kf_obs["lm"]])
    ok = (ki >= 0) & (li >= 0)
    if not ok.any():
        return {"n": 0}
    T = kfs["T"][ki[ok]]
    pw = lms["xyz"][li[ok]]
    pc = np.einsum("nji,nj->ni", T[:, :3, :3], pw - T[:, :3, 3])  # R^T (p - t)
    uv = kf_obs["uv"][ok].astype(np.float64)
    b_obs = sphere.pixel_to_bearing(uv[:, 0] - 0.5, uv[:, 1] - 0.5, width, height)  # stella px -> ours
    ang = np.degrees(sphere.angular_distance(pc, b_obs))
    return {"n": int(ok.sum()), "angle_deg_median": float(np.median(ang)), "angle_deg_p90": float(np.percentile(ang, 90)),
            "px_equiv_median": float(np.median(ang) / 360.0 * width), "frac_over_1deg": float((ang > 1).mean())}


def landmark_colors(kfs, lms, kf_obs, frames_t, frame_paths, max_kf=400):
    """Colour each landmark from one observing keyframe image (nearest frame by timestamp)."""
    col = np.full((len(lms["id"]), 3), 160, np.uint8)
    lm2i = {int(k): i for i, k in enumerate(lms["id"])}
    order = np.argsort(kf_obs["kf"], kind="stable")
    obs = kf_obs[order]
    kf_ids, starts = np.unique(obs["kf"], return_index=True)
    ends = np.append(starts[1:], len(obs))
    kid2t = dict(zip(kfs["id"].tolist(), kfs["t"].tolist()))
    pick = np.linspace(0, len(kf_ids) - 1, min(max_kf, len(kf_ids))).astype(int)
    done = np.zeros(len(col), bool)
    for p in pick:
        kid = int(kf_ids[p])
        if kid not in kid2t:
            continue
        fi = int(np.argmin(np.abs(frames_t - kid2t[kid])))
        img = cv2.imread(frame_paths[fi])
        if img is None:
            continue
        o = obs[starts[p]:ends[p]]
        idx = np.array([lm2i.get(int(l), -1) for l in o["lm"]])
        sel = (idx >= 0)
        idx = idx[sel]
        uv = np.round(o["uv"][sel]).astype(int)
        uv[:, 0] = np.clip(uv[:, 0], 0, img.shape[1] - 1)
        uv[:, 1] = np.clip(uv[:, 1], 0, img.shape[0] - 1)
        new = ~done[idx]
        col[idx[new]] = img[uv[new, 1], uv[new, 0], ::-1]
        done[idx[new]] = True
    return col, float(done.mean())


def postprocess(slam_dir: Path, ingest_dir: Path, out_dir: Path, gravity_method: str = "auto",
                camera_height_m: float | None = 1.3, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_in = json.loads((ingest_dir / "stitch.json").read_text())
    W, H = int(meta_in["width"]), int(meta_in["height"])
    on = stella.load_online_poses(slam_dir / "online_poses.txt")
    t_fin, T_fin = read_tum(slam_dir / "frame_trajectory.txt")
    kfs = stella.load_keyframes(slam_dir / "keyframes.csv")
    lms = stella.load_landmarks(slam_dir / "landmarks.bin")
    kf_obs = stella.load_kf_obs(slam_dir / "kf_obs.bin")
    good = lms["nobs"] >= 3
    rep = {"tracking": tracking_stats(on), "n_keyframes": int(len(kfs["id"])), "n_landmarks": int(len(lms["id"])),
           "n_landmarks_nobs_ge3": int(good.sum()), "n_final_frames": int(len(t_fin))}

    # ---- gravity
    imu_csv = ingest_dir / "imu.csv"
    up, ginfo = None, {}
    if gravity_method in ("auto", "imu") and imu_csv.exists() and "T_cam0_imu" in meta_in:
        imu = np.loadtxt(imu_csv, delimiter=",", comments="#")
        R_c0_imu = np.array(meta_in["T_cam0_imu"])[:3, :3]
        R_pano_imu = np.array(meta_in["R_cam0_pano"]).T @ R_c0_imu
        up, ginfo = up_from_imu(imu, t_fin, T_fin, R_pano_imu, meta_in.get("timeshift_cam_imu", 0.0))
    if up is None:
        if gravity_method == "imu":
            raise RuntimeError(f"IMU gravity failed: {ginfo}")
        up, ginfo = up_from_level_horizon(T_fin)
    R = gravity_rotation(up)
    origin = R @ kfs["T"][0, :3, 3]

    def to_grav(T):
        O = np.array(T, copy=True)
        O[:, :3, :3] = R @ O[:, :3, :3]
        O[:, :3, 3] = O[:, :3, 3] @ R.T - origin
        return O

    Tg = to_grav(T_fin)
    Tog = on.T.copy()
    Tog[on.valid] = to_grav(on.T[on.valid])
    kfT = to_grav(kfs["T"])
    Lg = lms["xyz"] @ R.T - origin
    write_tum(out_dir / "trajectory_grav.tum", t_fin, Tg, "final T_grav_pano, SLAM units")
    write_tum(out_dir / "online_grav.tum", on.t[on.valid], Tog[on.valid], "online T_grav_pano, SLAM units")
    with open(out_dir / "keyframes_grav.csv", "w") as f:
        f.write("id,t," + ",".join(f"T{i}{j}" for i in range(3) for j in range(4)) + "\n")
        for k in range(len(kfs["id"])):
            f.write(f"{kfs['id'][k]},{kfs['t'][k]:.6f}," + ",".join(f"{v:.6f}" for v in kfT[k, :3, :].ravel()) + "\n")

    # ---- scale: IMU accelerations (if available) and camera-height prior
    # all lengths below are expressed relative to the scene extent so they work for any SLAM scale
    ext = float(np.percentile(np.linalg.norm(Lg[good] - np.median(Lg[good], 0), axis=1), 50))
    h_units, hinfo = camera_height_units(Lg[good], Tg[:, :3, 3], bin_size=0.004 * ext)
    scale = {"camera_height_units": h_units, "camera_height_fit": hinfo, "scene_extent_units": ext}
    imu_s, imu_info = None, None
    if imu_csv.exists() and "T_cam0_imu" in meta_in:
        from slam3d.geometry.gravity import scale_from_imu

        imu = np.loadtxt(imu_csv, delimiter=",", comments="#")
        R_pano_imu = np.array(meta_in["R_cam0_pano"]).T @ np.array(meta_in["T_cam0_imu"])[:3, :3]
        imu_s, imu_info = scale_from_imu(imu, t_fin, Tg, R_pano_imu, meta_in.get("timeshift_cam_imu", 0.0))
        scale["imu"] = imu_info
    height_s = camera_height_m / h_units if (h_units and camera_height_m) else None
    scale["camera_height_prior"] = {"assumed_camera_height_m": camera_height_m, "metres_per_unit": height_s}
    if imu_s and imu_info.get("corr", 0) > 0.8:
        scale.update(method="imu", metres_per_unit=imu_s, status="estimated (IMU accelerations, coarse prior ~ +-25%)")
        if h_units:
            scale["implied_camera_height_m"] = imu_s * h_units
    elif height_s:
        scale.update(method="camera_height_prior", metres_per_unit=height_s, status="estimated (camera-height prior)")
    else:
        scale.update(method=None, metres_per_unit=None, status="arbitrary")

    # ---- quality
    frames = np.genfromtxt(ingest_dir / "frames.csv", delimiter=",", skip_header=1, dtype=str)
    frames_t = frames[:, 1].astype(float)
    frame_paths = [str(ingest_dir / r[2]) for r in frames]
    rep["reprojection"] = reprojection_stats(kfs, lms, kf_obs, W, H)
    rep["low_parallax_sections"] = low_parallax_sections(t_fin[::5], Tg[::5, :3, 3], Lg[good][::3])
    online_vs_final = None
    common, ia, ib = np.intersect1d(np.round(on.t[on.valid], 6), np.round(t_fin, 6), return_indices=True)
    if len(common) > 10:
        d = np.linalg.norm(Tog[on.valid][ia, :3, 3] - Tg[ib, :3, 3], axis=1)
        online_vs_final = {"median_units": float(np.median(d)), "p95_units": float(np.percentile(d, 95))}
    rep["online_vs_final_position_diff"] = online_vs_final

    cols, cfrac = landmark_colors(kfs, lms, kf_obs, frames_t, frame_paths)
    write_ply(out_dir / "landmarks_grav.ply", Lg[good], cols[good])
    geo = {
        "frame": "grav: SLAM world rotated so +z is up; origin at first keyframe; units = SLAM units",
        "R_grav_slam": R.tolist(), "origin_offset_in_rotated_slam": origin.tolist(),
        "gravity": {"up_in_slam": up.tolist(), **ginfo},
        "scale": scale, "landmark_color_coverage": cfrac, **rep,
    }
    (out_dir / "geometry.json").write_text(json.dumps(geo, indent=2, default=float))
    log(f"[geometry] tracked {rep['tracking']['tracked_frac']:.3f}, kf {rep['n_keyframes']}, lms {rep['n_landmarks']}, "
        f"gravity {ginfo.get('method')}, h_units {h_units}, reproj {rep['reprojection'].get('angle_deg_median')}")
    return geo
