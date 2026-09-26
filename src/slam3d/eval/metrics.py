"""Trajectory metrics: ATE (SE3 / Sim3 / none), 2D plan-frame error, RPE and the Hilti 2026 score."""
from __future__ import annotations

import numpy as np

from slam3d.geometry.transforms import associate_timestamps, umeyama

HILTI_K = 0.46051701859880917


def hilti_score(err_m: np.ndarray, coverage: float, min_coverage: float = 0.99) -> float:
    if coverage < min_coverage or len(err_m) == 0:
        return 0.0
    return float(min(100.0, np.mean(100.0 * np.exp(-HILTI_K * err_m))))


def _stats(e):
    if len(e) == 0:
        return {"n": 0}
    return {"n": int(len(e)), "rmse": float(np.sqrt(np.mean(e ** 2))), "mean": float(np.mean(e)),
            "median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "max": float(np.max(e))}


def ate(t_est, p_est, t_gt, p_gt, align: str = "se3", max_dt: float = 0.02, dims: int = 3,
        skip_first_s: float = 0.0):
    """Absolute trajectory error on positions.

    align: 'none' (already in GT frame), 'se3' (rigid), 'sim3' (similarity; for arbitrary-scale SLAM).
    dims: 3 for full error, 2 to ignore z (floor-plan localization).
    Coverage = fraction of GT poses (after skip) that have an associated estimate."""
    t_gt = np.asarray(t_gt)
    keep = t_gt >= t_gt[0] + skip_first_s
    t_gt, p_gt = t_gt[keep], np.asarray(p_gt)[keep]
    ia, ib = associate_timestamps(np.asarray(t_gt), np.asarray(t_est), max_dt)
    coverage = len(ia) / max(len(t_gt), 1)
    G = p_gt[ia, :dims]
    E = np.asarray(p_est)[ib, :dims]
    s, R, t = 1.0, np.eye(dims), np.zeros(dims)
    if align in ("se3", "sim3") and len(ia) >= 3:
        s, R, t = umeyama(E, G, with_scale=(align == "sim3"))
    E_al = s * E @ R.T + t
    err = np.linalg.norm(G - E_al, axis=1)
    res = {"align": align, "dims": dims, "coverage": coverage, "scale": float(s), **_stats(err),
           "hilti_score": hilti_score(err, coverage),
           "hilti_score_ignoring_coverage_rule": hilti_score(err, 1.0)}
    return res, err, (s, R, t), (ia, ib)


def rpe(t_est, T_est, t_gt, T_gt, delta_m: float = 1.0, max_dt: float = 0.02):
    """Relative translation error over GT path segments of ~delta_m (scale-free ratio also reported)."""
    ia, ib = associate_timestamps(np.asarray(t_gt), np.asarray(t_est), max_dt)
    Pg = np.asarray(T_gt)[ia, :3, 3]
    Pe = np.asarray(T_est)[ib, :3, 3]
    if len(Pg) < 3:
        return {"n": 0}
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Pg, axis=0), axis=1))])
    j = np.searchsorted(d, d + delta_m)
    ok = j < len(d)
    i0, i1 = np.nonzero(ok)[0], j[ok]
    lg = np.linalg.norm(Pg[i1] - Pg[i0], axis=1)
    le = np.linalg.norm(Pe[i1] - Pe[i0], axis=1)
    ratio = le / np.maximum(lg, 1e-9)
    med = np.median(ratio)
    return {"delta_m": delta_m, "n": int(len(i0)), "segment_length_ratio_median": float(med),
            "segment_length_ratio_rel_spread_p90": float(np.percentile(np.abs(ratio / med - 1), 90))}
