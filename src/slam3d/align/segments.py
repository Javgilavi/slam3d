"""Segment-wise structural refinement of the plan alignment (local drift correction).

A single similarity cannot remove residual SLAM drift. Here the recording is split into overlapping time
windows; each window's wall evidence (landmarks first observed inside the window) is registered to the plan
starting from the global transform, with a Gaussian prior that keeps it close to the global solution:

    minimise  sum_i rho(dist_plan(S_k q_i))  +  ((theta_k - theta)/s_theta)^2 + ((t_k - t)/s_t)^2 + ((log s_k - log s)/s_s)^2

A window correction is accepted only with enough evidence and a clear fit improvement; otherwise it keeps
the global transform (the plan is an imperfect prior: unbuilt or extra walls must not pull the map).
Per-pose transforms are blended with triangular time weights over the accepted windows, so the result is
continuous. The globally optimised SLAM trajectory is otherwise unchanged (no pose-graph re-optimisation).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from slam3d.align.bev import wall_evidence
from slam3d.align.register import evaluate_fit
from slam3d.floorplan.prepare import FloorPlan
from slam3d.geometry.transforms import apply_sim2, sim2_from_params, sim2_params


def _refine_with_prior(plan, S_glob, xy, w, path, sig_t=0.5, sig_th=np.deg2rad(2.0), sig_s=0.02, trunc=1.0, huber=0.15):
    s0, th0, tx0, ty0 = sim2_params(S_glob)
    x0 = np.array([np.log(s0), th0, tx0, ty0])
    sw = np.sqrt(w / w.mean())
    n_eff = np.sqrt(len(xy))

    def res(x):
        S = sim2_from_params(np.exp(x[0]), x[1], x[2], x[3])
        d = np.minimum(plan.dist_at(apply_sim2(S, xy), outside=trunc), trunc)
        prior = n_eff * np.array([(x[0] - x0[0]) / sig_s, (x[1] - th0) / sig_th, (x[2] - tx0) / sig_t, (x[3] - ty0) / sig_t]) * huber
        return np.concatenate([sw * d, prior])

    r = least_squares(res, x0, loss="huber", f_scale=huber, x_scale="jac", max_nfev=100)
    return sim2_from_params(np.exp(r.x[0]), r.x[1], r.x[2], r.x[3])


def refine_segments(plan: FloorPlan, S_glob, t, T_grav, L_grav, L_time, cam_height_units, window_s=30.0, step_s=15.0,
                    min_wall_cells=30, min_gain=0.03, bev_kw=None):
    """Returns (per-pose 3x3 transforms (N,3,3), segment report list)."""
    bev_kw = bev_kw or {}
    segs = []
    starts = np.arange(t[0], max(t[0] + 1e-6, t[-1] - window_s + step_s), step_s)
    for a in starts:
        b = a + window_s
        m_pose = (t >= a) & (t <= b)
        m_pts = (L_time >= a) & (L_time <= b)
        rec = {"t0": float(a), "t1": float(b), "accepted": False}
        if m_pose.sum() < 30 or m_pts.sum() < 50:
            rec["reason"] = "too few poses/points"
            segs.append((rec, S_glob))
            continue
        cams = T_grav[m_pose, :3, 3]
        bev = wall_evidence(L_grav[m_pts], cams, cam_height_units, **bev_kw)
        rec["n_wall_cells"] = int(len(bev.xy))
        if len(bev.xy) < min_wall_cells:
            rec["reason"] = "insufficient wall evidence"
            segs.append((rec, S_glob))
            continue
        path = cams[:: max(1, len(cams) // 100), :2]
        fit0 = evaluate_fit(plan, S_glob, bev.xy, path)
        S_k = _refine_with_prior(plan, S_glob, bev.xy, bev.weight, path)
        fit1 = evaluate_fit(plan, S_k, bev.xy, path)
        gain = fit1["wall_inlier_frac"] - fit0["wall_inlier_frac"]
        shift = float(np.median(np.linalg.norm(apply_sim2(S_k, path) - apply_sim2(S_glob, path), axis=1)))
        rec.update(inliers_global=fit0["wall_inlier_frac"], inliers_segment=fit1["wall_inlier_frac"], gain=gain, median_shift_m=shift)
        ok = gain >= min_gain and fit1["path_exterior_frac"] <= fit0["path_exterior_frac"] and fit1["path_in_structure_frac"] <= fit0["path_in_structure_frac"] + 0.01
        rec["accepted"] = bool(ok)
        if not ok:
            rec["reason"] = "no clear improvement"
        segs.append((rec, S_k if ok else S_glob))
    # blend parameters with triangular weights
    params_g = np.array(sim2_params(S_glob))
    P = np.zeros((len(t), 4))
    Wsum = np.zeros(len(t))
    for rec, S_k in segs:
        c = 0.5 * (rec["t0"] + rec["t1"])
        half = 0.5 * (rec["t1"] - rec["t0"])
        wgt = np.clip(1.0 - np.abs(t - c) / half, 0.0, 1.0)
        pk = np.array(sim2_params(S_k))
        pk[1] = params_g[1] + (pk[1] - params_g[1] + np.pi) % (2 * np.pi) - np.pi
        pk[0] = np.log(pk[0])
        P += wgt[:, None] * pk
        Wsum += wgt
    base = params_g.copy()
    base[0] = np.log(base[0])
    P = np.where(Wsum[:, None] > 1e-9, P / np.maximum(Wsum, 1e-9)[:, None], base)
    S_pose = np.stack([sim2_from_params(np.exp(p[0]), p[1], p[2], p[3]) for p in P])
    return S_pose, [s[0] for s in segs]
