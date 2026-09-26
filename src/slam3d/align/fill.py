"""Batch coverage completion for a finished recording (NOT used by causal localization).

Every output pose carries a flag so metrics can be reported per class:
  0 tracked            SLAM pose (globally optimised, plan-aligned)
  1 static_fill        before initialisation / after the end, only where the IMU shows the camera was
                       stationary (then the first/last tracked pose is correct up to sensor noise)
  2 gap_interpolated   inside a tracking gap: linear position / slerp rotation between the bounding poses
                       (low confidence; long gaps can be metres off)
"""
from __future__ import annotations

import numpy as np

from slam3d.geometry.transforms import interpolate_poses

FLAGS = {0: "tracked", 1: "static_fill", 2: "gap_interpolated"}


def stationary(imu: np.ndarray | None, t0: float, t1: float, gyro_p95_max=0.08, acc_std_max=0.25) -> bool:
    if imu is None or t1 <= t0:
        return False
    m = (imu[:, 0] >= t0) & (imu[:, 0] <= t1)
    if m.sum() < 50:
        return False
    g = np.linalg.norm(imu[m, 1:4], axis=1)
    a = np.linalg.norm(imu[m, 4:7], axis=1)
    return bool(np.percentile(g, 95) < gyro_p95_max and np.std(a) < acc_std_max)


def fill_trajectory(t_frames: np.ndarray, t_traj: np.ndarray, T: np.ndarray, imu: np.ndarray | None = None,
                    gap_min_s: float = 0.2, max_gap_s: float = 60.0):
    t_frames = np.asarray(t_frames, float)
    out_t, out_T, flags = [], [], []
    first, last = t_traj[0], t_traj[-1]
    info = {"static_prefill": False, "static_postfill": False, "gaps": []}
    pre = t_frames[t_frames < first - 1e-6]
    if len(pre) and stationary(imu, pre[0], first):
        info["static_prefill"] = True
        out_t += pre.tolist()
        out_T += [T[0]] * len(pre)
        flags += [1] * len(pre)
    inside = t_frames[(t_frames >= first - 1e-6) & (t_frames <= last + 1e-6)]
    Ti, ok = interpolate_poses(inside, t_traj, T)
    idx = np.searchsorted(t_traj, inside)
    idx = np.clip(idx, 1, len(t_traj) - 1)
    gap = t_traj[idx] - t_traj[idx - 1]
    exact = np.min(np.abs(t_traj[np.clip(idx - 1, 0, None)][:, None] - inside[:, None]), axis=1) < 1e-4
    exact |= np.abs(t_traj[idx] - inside) < 1e-4
    for k, tt in enumerate(inside):
        if exact[k] or gap[k] <= gap_min_s:
            flags.append(0)
        elif gap[k] <= max_gap_s:
            flags.append(2)
        else:
            continue
        out_t.append(tt)
        out_T.append(Ti[k])
    for i in range(1, len(t_traj)):
        if t_traj[i] - t_traj[i - 1] > gap_min_s:
            info["gaps"].append([float(t_traj[i - 1]), float(t_traj[i])])
    post = t_frames[t_frames > last + 1e-6]
    if len(post) and stationary(imu, last, post[-1]):
        info["static_postfill"] = True
        out_t += post.tolist()
        out_T += [T[-1]] * len(post)
        flags += [1] * len(post)
    order = np.argsort(out_t)
    return np.asarray(out_t)[order], np.asarray(out_T)[order], np.asarray(flags)[order], info
