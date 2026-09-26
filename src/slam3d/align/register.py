"""2D registration of structural BEV evidence (reconstruction units) to a floor plan (metres).

Model:   p_plan = s * R(theta) * q + t        (similarity; s fixed -> rigid when scale is known)

Cost (robust, plan treated as an imperfect prior):
  * wall term:  rho(dist_plan(p_i)) weighted by evidence weight (Huber / truncated);
    unbuilt/temporary walls simply produce truncated residuals instead of forcing a fit;
  * path term:  camera positions must lie in interior free space (penalise structure/exterior).

Stages: manual correspondences (Umeyama) and/or global coarse search (FFT correlation over
yaw x scale, translation by correlation) -> local robust refinement -> confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from slam3d.floorplan.prepare import FloorPlan
from slam3d.geometry.transforms import apply_sim2, ransac_umeyama, sim2_from_params, sim2_params


@dataclass
class Alignment:
    S: np.ndarray  # 3x3 similarity: plan_xy = S @ [q, 1]
    score: float
    confidence: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_json(self):
        s, th, tx, ty = sim2_params(self.S)
        return {"S_plan_from_grav": self.S.tolist(), "scale": s, "yaw_rad": th, "tx": tx, "ty": ty,
                "score": self.score, "confidence": self.confidence, **self.meta}


def from_correspondences(src_xy, plan_xy, with_scale=True, thresh_m=0.5):
    src_xy = np.asarray(src_xy, float)
    plan_xy = np.asarray(plan_xy, float)
    if len(src_xy) < 2:
        raise ValueError("need >= 2 correspondences")
    if len(src_xy) <= 3:
        from slam3d.geometry.transforms import umeyama

        s, R, t = umeyama(src_xy, plan_xy, with_scale)
        inl = np.ones(len(src_xy), bool)
    else:
        (s, R, t), inl = ransac_umeyama(src_xy, plan_xy, thresh_m, with_scale)
    S = np.eye(3)
    S[:2, :2] = s * R
    S[:2, 2] = t
    res = np.linalg.norm(apply_sim2(S, src_xy) - plan_xy, axis=1)
    return Alignment(S, float(-np.mean(res[inl])), {"residual_m": res.tolist(), "inliers": inl.tolist()},
                     {"method": "manual_correspondences", "n": int(len(src_xy))})


def _score_maps(plan: FloorPlan, sigma_m: float):
    wall = np.exp(-0.5 * (plan.dist / sigma_m) ** 2).astype(np.float32)
    interior_free = (~plan.exterior) & (plan.dist > 0.25)
    path = np.where(interior_free, 1.0, -1.0).astype(np.float32)
    return wall, path


def _rasterize(q_xy, weights, S_no_t, res, pad):
    """Rasterise rotated/scaled points into a template (row 0 = max y). Returns template and the plan
    coordinates of template pixel (0,0) centre relative to translation."""
    p = q_xy @ S_no_t[:2, :2].T
    lo, hi = p.min(0) - pad, p.max(0) + pad
    W = int(np.ceil((hi[0] - lo[0]) / res)) + 1
    H = int(np.ceil((hi[1] - lo[1]) / res)) + 1
    tpl = np.zeros((H, W), np.float32)
    c = np.round((p[:, 0] - lo[0]) / res).astype(int)
    r = np.round((hi[1] - p[:, 1]) / res).astype(int)
    np.add.at(tpl, (r, c), weights)
    return tpl, lo, hi


def global_search(plan: FloorPlan, wall_xy, wall_w, path_xy, scales, yaws_deg=None, coarse_res=0.2,
                  sigma_m=0.3, path_weight=0.5, top_k=8, min_sep_m=3.0, min_sep_deg=20.0):
    """Exhaustive search. Returns list of hypotheses [(score, S)] sorted desc, distinct by min_sep."""
    yaws = np.deg2rad(np.arange(0, 360, 2.0) if yaws_deg is None else np.asarray(yaws_deg))
    f = plan.res / coarse_res
    wall_map, path_map = _score_maps(plan, sigma_m)
    wall_c = cv2.resize(wall_map, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    path_c = cv2.resize(path_map, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    Hc = wall_c.shape[0]
    ww = wall_w / max(wall_w.sum(), 1e-9)
    pw = np.full(len(path_xy), path_weight / max(len(path_xy), 1))
    cand = []
    for s in scales:
        for th in yaws:
            S = sim2_from_params(s, th, 0, 0)
            pts = np.vstack([wall_xy, path_xy])
            p = pts @ S[:2, :2].T
            lo, hi = p.min(0), p.max(0)
            ext = hi - lo
            if ext[0] > (wall_c.shape[1] - 2) * coarse_res or ext[1] > (Hc - 2) * coarse_res:
                continue
            tw, lo_w, hi_w = _rasterize(wall_xy, ww, S, coarse_res, 0.0)
            tp, lo_p, hi_p = _rasterize(path_xy, pw, S, coarse_res, 0.0)
            # common template frame: pad both to shared bbox
            lo_all = np.minimum(lo_w, lo_p)
            hi_all = np.maximum(hi_w, hi_p)
            Wt = int(np.ceil((hi_all[0] - lo_all[0]) / coarse_res)) + 1
            Ht = int(np.ceil((hi_all[1] - lo_all[1]) / coarse_res)) + 1
            if Wt >= wall_c.shape[1] or Ht >= Hc:
                continue
            T1 = np.zeros((Ht, Wt), np.float32)
            T2 = np.zeros((Ht, Wt), np.float32)
            ow = (int(round((lo_w[0] - lo_all[0]) / coarse_res)), int(round((hi_all[1] - hi_w[1]) / coarse_res)))
            op = (int(round((lo_p[0] - lo_all[0]) / coarse_res)), int(round((hi_all[1] - hi_p[1]) / coarse_res)))
            T1[ow[1]:ow[1] + tw.shape[0], ow[0]:ow[0] + tw.shape[1]] += tw[:Ht - ow[1], :Wt - ow[0]]
            T2[op[1]:op[1] + tp.shape[0], op[0]:op[0] + tp.shape[1]] += tp[:Ht - op[1], :Wt - op[0]]
            sc = cv2.matchTemplate(wall_c, T1, cv2.TM_CCORR) + cv2.matchTemplate(path_c, T2, cv2.TM_CCORR)
            # take several local maxima per (s, theta)
            k = 3
            flat = np.argpartition(sc.ravel(), -k)[-k:]
            for fidx in flat:
                r0, c0 = divmod(int(fidx), sc.shape[1])
                # template pixel (0,0) centre in plan: col c0 -> x, row r0 -> y (rows from top)
                x_tl = c0 * coarse_res
                y_tl = (Hc - 1 - r0) * coarse_res
                tx = x_tl - lo_all[0]
                ty = y_tl - hi_all[1]
                cand.append((float(sc.ravel()[fidx]), sim2_from_params(s, th, tx, ty), s, th))
    cand.sort(key=lambda c: -c[0])
    distinct = []
    for c in cand:
        ok = True
        for d in distinct:
            dt = np.hypot(c[1][0, 2] - d[1][0, 2], c[1][1, 2] - d[1][1, 2])
            dth = np.degrees(abs((c[3] - d[3] + np.pi) % (2 * np.pi) - np.pi))
            if dt < min_sep_m and dth < min_sep_deg and abs(np.log(c[2] / d[2])) < 0.1:
                ok = False
                break
        if ok:
            distinct.append(c)
        if len(distinct) >= top_k:
            break
    return [(c[0], c[1]) for c in distinct]


def _residuals(x, plan, wall_xy, wall_sqrt_w, path_xy, fixed_scale, path_weight, trunc_m):
    if fixed_scale is None:
        s = np.exp(x[0]); th, tx, ty = x[1:]
    else:
        s = fixed_scale; th, tx, ty = x
    S = sim2_from_params(s, th, tx, ty)
    pw = apply_sim2(S, wall_xy)
    d = np.minimum(plan.dist_at(pw, outside=trunc_m), trunc_m)
    r_wall = wall_sqrt_w * d
    pp = apply_sim2(S, path_xy)
    dp = plan.dist_at(pp, outside=0.0)
    ext = plan.lookup(plan.exterior.astype(np.float32), pp, 1.0)
    r_path = np.sqrt(path_weight / max(len(path_xy), 1)) * (np.maximum(0.3 - dp, 0) + ext * 1.0) * np.sqrt(len(wall_xy))
    return np.concatenate([r_wall, r_path])


def refine(plan: FloorPlan, S0, wall_xy, wall_w, path_xy, fixed_scale=None, path_weight=0.5,
           trunc_m=1.0, huber_m=0.15):
    s0, th0, tx0, ty0 = sim2_params(S0)
    x0 = np.array([th0, tx0, ty0]) if fixed_scale is not None else np.array([np.log(s0), th0, tx0, ty0])
    sw = np.sqrt(wall_w / wall_w.mean())
    r = least_squares(_residuals, x0, loss="huber", f_scale=huber_m, x_scale="jac", max_nfev=200,
                      args=(plan, wall_xy, sw, path_xy, fixed_scale, path_weight, trunc_m))
    x = r.x
    s, th, tx, ty = (fixed_scale, *x) if fixed_scale is not None else (np.exp(x[0]), *x[1:])
    S = sim2_from_params(s, th, tx, ty)
    return S, float(r.cost), r


def evaluate_fit(plan: FloorPlan, S, wall_xy, path_xy, inlier_m=0.2):
    d = plan.dist_at(apply_sim2(S, wall_xy), outside=10.0)
    dp = plan.dist_at(apply_sim2(S, path_xy), outside=0.0)
    ext = plan.lookup(plan.exterior.astype(np.float32), apply_sim2(S, path_xy), 1.0)
    return {"wall_inlier_frac": float((d < inlier_m).mean()) if len(d) else 0.0,
            "wall_median_dist_m": float(np.median(d)) if len(d) else None,
            "path_in_structure_frac": float((dp < 0.1).mean()),
            "path_exterior_frac": float((ext > 0.5).mean())}


def align(plan: FloorPlan, wall_xy, wall_w, path_xy, scale_prior=None, scale_range=(0.5, 2.0), n_scales=15,
          fixed_scale=None, init_S=None, min_confident_cells=30, log=print):
    """Full automatic alignment. scale_prior: expected metres per reconstruction unit (e.g. from
    camera height). Returns best Alignment with confidence based on the runner-up hypothesis."""
    hyps = []
    if init_S is not None:
        hyps.append((np.inf, init_S))
    else:
        if fixed_scale is not None:
            scales = [fixed_scale]
        else:
            base = scale_prior if scale_prior else 1.0
            scales = base * np.exp(np.linspace(np.log(scale_range[0]), np.log(scale_range[1]), n_scales))
        hyps = global_search(plan, wall_xy, wall_w, path_xy, scales)
        log(f"[align] global search: {len(hyps)} distinct hypotheses")
    results = []
    for sc, S0 in hyps:
        S, cost, _ = refine(plan, S0, wall_xy, wall_w, path_xy, fixed_scale=fixed_scale)
        fit = evaluate_fit(plan, S, wall_xy, path_xy)
        quality = fit["wall_inlier_frac"] - 2.0 * fit["path_exterior_frac"] - 1.0 * fit["path_in_structure_frac"]
        results.append((quality, S, cost, fit, sc))
    results.sort(key=lambda r: -r[0])
    best = results[0]
    runner = None
    for r in results[1:]:  # best DISTINCT alternative: refinements from different seeds often converge to the same pose
        if np.median(np.linalg.norm(apply_sim2(r[1], path_xy) - apply_sim2(best[1], path_xy), axis=1)) > 1.0:
            runner = r
            break
    conf = dict(best[3], quality=best[0], coarse_score=best[4] if np.isfinite(best[4]) else None,
                runner_up_quality=runner[0] if runner else None,
                margin=(best[0] - runner[0]) if runner else None,
                n_hypotheses=len(results))
    if runner is not None:
        s1, th1, tx1, ty1 = sim2_params(best[1])
        s2, th2, tx2, ty2 = sim2_params(runner[1])
        conf["runner_up_offset_m"] = float(np.hypot(tx1 - tx2, ty1 - ty2))
        conf["runner_up_yaw_diff_deg"] = float(np.degrees(abs((th1 - th2 + np.pi) % (2 * np.pi) - np.pi)))
    conf["n_wall_cells"] = int(len(wall_xy))
    if len(wall_xy) < min_confident_cells:
        conf["status"] = "insufficient_evidence"  # a handful of cells can fit a plan well by chance
    else:
        ok = best[0] > 0.4 and (runner is None or best[0] - runner[0] > 0.1)
        conf["status"] = "confident" if ok else "ambiguous_or_weak"
    return Alignment(best[1], float(best[0]), conf, {"method": "auto" if init_S is None else "refined_from_init",
                                                     "fixed_scale": fixed_scale})
