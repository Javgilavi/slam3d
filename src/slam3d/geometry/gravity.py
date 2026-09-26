"""Gravity alignment and scale estimation for monocular 360 SLAM output.

World frames:
  * `slam`: arbitrary frame/scale from SLAM.
  * `grav`: slam frame rotated so +z is up (gravity opposite), same origin, same scale.

Scale sources (explicitly labelled downstream):
  * imu:            metres/unit from SLAM vs IMU linear accelerations (estimated metric scale);
  * camera_height:  metres/unit from a prior camera height and the floor observed below the camera;
  * plan:           metres/unit from floor-plan similarity registration (externally calibrated).
"""
from __future__ import annotations

import numpy as np

from slam3d.geometry.transforms import interpolate_poses, rotation_between

G = 9.81


def up_from_imu(imu: np.ndarray, t_pose: np.ndarray, T_world_pano: np.ndarray, R_pano_imu: np.ndarray,
                time_shift_cam_imu: float = 0.0, gyro_max: float = 0.3, acc_tol: float = 0.6):
    """Estimate the world up vector from accelerometer specific force.

    imu: (N,7) t, gx,gy,gz, ax,ay,az (IMU frame). A resting accelerometer measures +g upward.
    Uses quasi-static samples (|gyro| < gyro_max rad/s and | |a|-9.81 | < acc_tol).
    Kalibr convention: t_imu = t_cam + timeshift_cam_imu  =>  t_cam = t_imu - shift."""
    t = imu[:, 0] - time_shift_cam_imu
    g = np.linalg.norm(imu[:, 1:4], axis=1)
    a = imu[:, 4:7]
    an = np.linalg.norm(a, axis=1)
    sel = (g < gyro_max) & (np.abs(an - G) < acc_tol) & (t >= t_pose.min()) & (t <= t_pose.max())
    if sel.sum() < 50:
        return None, {"n_samples": int(sel.sum())}
    sel_idx = np.nonzero(sel)[0][:: max(1, sel.sum() // 20000)]
    T, ok = interpolate_poses(t[sel_idx], t_pose, T_world_pano, max_gap=0.2)
    R_wi = T[:, :3, :3] @ R_pano_imu
    up = np.einsum("nij,nj->ni", R_wi[ok], a[sel_idx][ok] / an[sel_idx][ok, None])
    mean = up.mean(0)
    up_hat = mean / np.linalg.norm(mean)
    spread = np.degrees(np.arccos(np.clip(up @ up_hat, -1, 1)))
    return up_hat, {"n_samples": int(ok.sum()), "angular_spread_deg_median": float(np.median(spread)),
                    "method": "imu"}


def up_from_level_horizon(T_world_pano: np.ndarray):
    """Assume a horizon-levelled panorama (typical for stabilised consumer 360 video): pano -y is up."""
    ups = T_world_pano[:, :3, :3] @ np.array([0.0, -1.0, 0.0])
    mean = ups.mean(0)
    up_hat = mean / np.linalg.norm(mean)
    spread = np.degrees(np.arccos(np.clip(ups @ up_hat, -1, 1)))
    return up_hat, {"method": "level_horizon", "angular_spread_deg_median": float(np.median(spread)),
                    "angular_spread_deg_p95": float(np.percentile(spread, 95))}


def gravity_rotation(up_world: np.ndarray) -> np.ndarray:
    """R_grav_slam such that R @ up_world = +z."""
    return rotation_between(up_world, np.array([0.0, 0.0, 1.0]))


def camera_height_units(points_grav: np.ndarray, cam_pos_grav: np.ndarray, bin_size: float, rel_peak: float = 0.5,
                        cell: float | None = None):
    """Camera height above the floor in SLAM units, for a single-storey walkthrough.

    Landmarks below the median camera height are histogrammed globally by height; each bin's score is the
    number of distinct horizontal cells it covers (a floor is a large horizontal layer; clutter tops are
    small). Among bins scoring >= rel_peak * max, the HIGHEST one is chosen: on glossy construction floors
    reflections are triangulated as an additional layer BELOW the real floor.
    Developed on Hilti floor_2_2025-12-03_run_1 (per-camera peak variants were 25-36% off; this was 8%)."""
    cz = float(np.median(cam_pos_grav[:, 2]))
    below = points_grav[points_grav[:, 2] < cz - 2 * bin_size]
    if len(below) < 100:
        return None, {"n": int(len(below))}
    cell = cell or 20 * bin_size
    edges = np.arange(np.percentile(below[:, 2], 0.5), cz, bin_size)
    if len(edges) < 4:
        return None, {"n": int(len(below))}
    k = np.clip(np.digitize(below[:, 2], edges) - 1, 0, len(edges) - 2)
    ij = np.floor(below[:, :2] / cell).astype(np.int64)
    key = np.unique(np.column_stack([k, ij]), axis=0)
    cov = np.bincount(key[:, 0], minlength=len(edges) - 1).astype(float)
    covs = np.convolve(cov, [0.25, 0.5, 0.25], mode="same")
    cand = np.nonzero(covs >= rel_peak * covs.max())[0]
    kb = int(cand.max())
    floor_z = float((edges[kb] + edges[kb + 1]) / 2)
    return cz - floor_z, {"method": "global_floor_layer_by_horizontal_coverage", "floor_z_units": floor_z,
                          "n_points_below": int(len(below)), "peak_coverage_cells": int(cov[kb]), "n_candidate_bins": int(len(cand))}


def scale_from_imu(imu: np.ndarray, t_pose: np.ndarray, T_grav_pano: np.ndarray, R_pano_imu: np.ndarray,
                   time_shift_cam_imu: float = 0.0, lags_s=(0.5, 1.0, 2.0), fs: float = 30.0, fs_imu_grid: float = 300.0):
    """Metric scale (metres per SLAM unit) from an EXACT kinematic identity, avoiding filter mismatch:

        p(t+tau) - 2 p(t) + p(t-tau) = integral_{-tau}^{tau} (tau - |u|) a(t+u) du

    Left: second difference of the SLAM trajectory (units). Right: triangular-kernel integral of the
    gravity-compensated IMU acceleration in the gravity frame (metres). A constant accelerometer bias b
    contributes b * tau^2 and is estimated jointly. Several lags are evaluated; the best-correlated lag is
    kept and the spread across lags is reported. (An earlier Savitzky-Golay derivative/smoothing
    comparison was biased by ~20 %: the two filters do not have matched frequency responses.)"""
    from scipy.optimize import least_squares

    t0, t1 = t_pose[0] + max(lags_s) + 0.5, t_pose[-1] - max(lags_s) - 0.5
    if t1 - t0 < 10:
        return None, {"reason": "trajectory too short"}
    tg = np.arange(t_pose[0], t_pose[-1], 1.0 / fs_imu_grid)
    Tg_, okg = interpolate_poses(tg, t_pose, T_grav_pano, max_gap=0.25)
    t_imu = imu[:, 0] - time_shift_cam_imu
    m = (t_imu >= tg[0]) & (t_imu <= tg[-1])
    bins = np.clip(np.round((t_imu[m] - tg[0]) * fs_imu_grid).astype(int), 0, len(tg) - 1)
    cnt = np.bincount(bins, minlength=len(tg))
    f_b = np.stack([np.bincount(bins, weights=imu[m, 4 + k], minlength=len(tg)) for k in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    have = cnt > 0
    if not have.all():  # fill empty bins by interpolation
        idx = np.arange(len(tg))
        f_b = np.stack([np.interp(idx, idx[have], f_b[have, k]) for k in range(3)], 1)
    a_w = np.einsum("nij,nj->ni", Tg_[:, :3, :3] @ R_pano_imu, f_b) - np.array([0.0, 0.0, G])

    tc = np.arange(t0, t1, 1.0 / fs)
    Pc, okc = interpolate_poses(tc, t_pose, T_grav_pano, max_gap=0.25)
    res = []
    for lag in lags_s:
        Pp, okp = interpolate_poses(tc + lag, t_pose, T_grav_pano, max_gap=0.25)
        Pm, okm = interpolate_poses(tc - lag, t_pose, T_grav_pano, max_gap=0.25)
        A = Pp[:, :3, 3] - 2 * Pc[:, :3, 3] + Pm[:, :3, 3]
        L = int(round(lag * fs_imu_grid))
        u = np.arange(-L, L + 1) / fs_imu_grid
        kern = (lag - np.abs(u)) / fs_imu_grid  # du
        conv = np.stack([np.convolve(a_w[:, k], kern[::-1], mode="same") for k in range(3)], 1)
        ci = np.clip(np.round((tc - tg[0]) * fs_imu_grid).astype(int), 0, len(tg) - 1)
        B = conv[ci]
        valid = okc & okp & okm & okg[ci]
        if valid.sum() < 5 * fs:
            continue
        Av, Bv = A[valid], B[valid]

        def r(x):
            return (Bv - (x[0] * Av + x[1:4] * lag ** 2)).ravel()

        s0 = float(np.sum(Av * Bv) / max(np.sum(Av * Av), 1e-12))
        sol = least_squares(r, np.array([s0, 0, 0, 0]), loss="soft_l1", f_scale=0.05 * lag ** 2 + 1e-3)
        pred = sol.x[0] * Av + sol.x[1:4] * lag ** 2
        corr = float(np.corrcoef(pred.ravel(), Bv.ravel())[0, 1])
        res.append({"lag_s": lag, "s": float(sol.x[0]), "corr": corr, "bias_mps2": sol.x[1:4].tolist(), "n": int(valid.sum())})
    if not res:
        return None, {"reason": "no lag succeeded"}
    best = max(res, key=lambda d: d["corr"])
    ss = np.array([d["s"] for d in res])
    return best["s"], {"method": "imu_second_difference_identity", "metres_per_unit": best["s"], "corr": best["corr"],
                       "lag_s": best["lag_s"], "bias_mps2": best["bias_mps2"], "lags": res,
                       "lag_rel_spread": float((ss.max() - ss.min()) / 2 / np.mean(ss))}


def _scale_from_imu_window(imu: np.ndarray, t_pose: np.ndarray, T_grav_pano: np.ndarray, R_pano_imu: np.ndarray,
                           time_shift_cam_imu: float = 0.0, fs: float = 30.0, win_s: float = 0.8, min_motion: float = 0.3):
    """Metric scale (metres per SLAM unit) by comparing trajectory accelerations with IMU accelerations.

    a_imu_world(t) = R_grav_imu(t) f(t) - [0, 0, g]      (f: specific force, IMU frame)
    a_slam(t)      = d2/dt2 p_grav(t)                     (SLAM units / s^2, Savitzky-Golay)
    Both are filtered with the same Savitzky-Golay window; solve a_imu ~= s * a_slam + b (per-axis bias)
    with a robust loss on samples with significant motion. Rotation errors leaking gravity into the
    horizontal axes are absorbed by the bias (constant part) and down-weighted by the robust loss."""
    from scipy.optimize import least_squares
    from scipy.signal import savgol_filter

    t0, t1 = t_pose[0] + 1.0, t_pose[-1] - 1.0
    tu = np.arange(t0, t1, 1.0 / fs)
    if len(tu) < 5 * fs:
        return None, {"reason": "trajectory too short"}
    T, ok = interpolate_poses(tu, t_pose, T_grav_pano, max_gap=0.25)
    n = int(round(win_s * fs)) | 1
    a_v = savgol_filter(T[:, :3, 3], n, 3, deriv=2, delta=1.0 / fs, axis=0)

    t_imu = imu[:, 0] - time_shift_cam_imu
    m = (t_imu >= t0 - 0.5) & (t_imu <= t1 + 0.5)
    t_imu, raw = t_imu[m], imu[m]
    Ti, oki = interpolate_poses(t_imu, t_pose, T_grav_pano, max_gap=0.25)
    f_w = np.einsum("nij,nj->ni", Ti[:, :3, :3] @ R_pano_imu, raw[:, 4:7])
    a_w = f_w - np.array([0.0, 0.0, G])
    # average IMU samples into the uniform bins, then apply the identical smoothing
    bins = np.clip(np.round((t_imu - t0) * fs).astype(int), 0, len(tu) - 1)
    cnt = np.bincount(bins, minlength=len(tu))
    a_i = np.stack([np.bincount(bins, weights=a_w[:, k], minlength=len(tu)) for k in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    a_i = savgol_filter(a_i, n, 3, deriv=0, axis=0)
    valid = ok & (cnt > 0)
    # edges of the filter window are unreliable
    valid[: n // 2] = False
    valid[-(n // 2):] = False
    mag = np.linalg.norm(a_i, axis=1)
    use = valid & (mag > min_motion)
    if use.sum() < 3 * fs:
        return None, {"reason": "not enough motion", "n": int(use.sum())}
    A, B = a_v[use], a_i[use]

    def res(x):
        return (B - (x[0] * A + x[1:4])).ravel()

    s0 = float(np.sum(A * B) / max(np.sum(A * A), 1e-12))
    r = least_squares(res, np.array([s0, 0, 0, 0]), loss="huber", f_scale=0.3)
    s, b = float(r.x[0]), r.x[1:4]
    pred = s * A + b
    corr = float(np.corrcoef(pred.ravel(), B.ravel())[0, 1])
    resid = np.linalg.norm(B - pred, axis=1)
    # bootstrap over 10 s blocks for an uncertainty estimate
    rng = np.random.default_rng(0)
    blk = np.arange(len(A)) // int(10 * fs)
    ub = np.unique(blk)
    boots = []
    for _ in range(50):
        pick = rng.choice(ub, len(ub))
        sel = np.concatenate([np.nonzero(blk == k)[0] for k in pick])
        Ab, Bb = A[sel], B[sel] - b
        boots.append(float(np.sum(Ab * Bb) / max(np.sum(Ab * Ab), 1e-12)))
    return s, {"method": "imu_acceleration", "metres_per_unit": s, "bias_mps2": b.tolist(), "corr": corr,
               "residual_mps2_median": float(np.median(resid)), "n_samples": int(use.sum()),
               "bootstrap_rel_std": float(np.std(boots) / abs(s)) if s else None, "filter_window_s": win_s}
