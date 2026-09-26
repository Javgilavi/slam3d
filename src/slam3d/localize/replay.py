"""Sequential (causal) replay of recorded SLAM output through the plan particle filter.

Only information available at time t is used at step t: the ONLINE SLAM pose logged when frame t
was processed, and landmarks tracked in frames <= t (their positions as known at that time).
Not used: final optimised trajectory, future frames, ground truth (except for evaluation and the
optional known start pose, which is reported explicitly).

Global initialisation and recovery use structural registration of the CAUSAL local map (last
`reg_window_s` seconds of online landmarks + online camera path) against the plan; its distinct top
hypotheses seed a particle mixture, so repetitive layouts keep several modes until evidence
disambiguates them.

Gravity rotation and the scale prior are taken from the geometry stage. This is a documented
assumption: in a live system they come from the IMU (gravity, coarse scale) and the first seconds
of mapping; they are not derived from the plan or ground truth.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from slam3d.floorplan.prepare import FloorPlan
from slam3d.geometry.transforms import apply_sim2, interpolate_poses, sim2_params
from slam3d.localize.pf import PFConfig, PlanParticleFilter, _wrap
from slam3d.slam import stella


def _yaw_grav(R_grav_pano):
    f = R_grav_pano[:, 2]  # pano forward axis
    return float(np.arctan2(f[1], f[0]))


def structural_hypotheses(plan: FloorPlan, pts: np.ndarray, cams: np.ndarray, h_units: float, s_prior: float,
                          cfg: PFConfig, cur_cam_xy: np.ndarray):
    """Register a causal local map to the plan. Returns [(weight, x, y, phi, s)] for the CURRENT camera,
    plus diagnostics. Hypotheses are distinct (> 1.5 m or > 15 deg apart)."""
    from slam3d.align import register
    from slam3d.align.bev import wall_evidence

    bev = wall_evidence(pts, cams, h_units, cell_frac=0.08, min_zbins=2, n_zbins=6)
    if len(bev.xy) < cfg.reg_min_wall_cells:
        return [], {"n_wall_cells": int(len(bev.xy))}
    path = cams[:: max(1, len(cams) // 150), :2]
    scales = s_prior * np.exp(np.linspace(np.log(1 - cfg.reg_scale_rel), np.log(1 + cfg.reg_scale_rel), cfg.reg_n_scales))
    coarse = register.global_search(plan, bev.xy, bev.weight, path, scales, top_k=cfg.reg_top_k, min_sep_m=2.0, min_sep_deg=15.0)
    cand = []
    for _, S0 in coarse:
        S, _, _ = register.refine(plan, S0, bev.xy, bev.weight, path)
        fit = register.evaluate_fit(plan, S, bev.xy, path)
        q = fit["wall_inlier_frac"] - 2.0 * fit["path_exterior_frac"] - 1.0 * fit["path_in_structure_frac"]
        s, th, _, _ = sim2_params(S)
        xy = apply_sim2(S, cur_cam_xy[None])[0]
        if q <= 0 or not (0.5 * s_prior < s < 2.0 * s_prior):
            continue
        dup = any(np.hypot(*(xy - np.array(c[1:3]))) < 1.5 and abs(_wrap(th - c[3])) < np.deg2rad(15) for c in cand)
        if not dup:
            cand.append((q, xy[0], xy[1], th, s))
    if not cand:
        return [], {"n_wall_cells": int(len(bev.xy)), "n_coarse": len(coarse)}
    q = np.array([c[0] for c in cand])
    w = np.exp((q - q.max()) / 0.05)  # sharpen by fit quality
    hyps = [(float(wi), *c[1:]) for wi, c in zip(w, cand)]
    return hyps, {"n_wall_cells": int(len(bev.xy)), "qualities": q.round(3).tolist()}


def replay(slam_dir: Path, geometry_json: Path, plan: FloorPlan, out_dir: Path, cfg: PFConfig = PFConfig(),
           init: dict | None = None, kidnap_at_s: float | None = None, gt: tuple | None = None,
           snapshot_every: int = 2, save_debug: bool = False, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    debug = [] if save_debug else None
    geo = json.loads(Path(geometry_json).read_text())
    Rg = np.array(geo["R_grav_slam"])
    h_units = geo["scale"]["camera_height_units"]
    s_prior = geo["scale"]["metres_per_unit"] or 1.0
    on = stella.load_online_poses(slam_dir / "online_poses.txt")
    obs = stella.load_local_obs(slam_dir / "local_obs.bin")
    obs_by_idx = {o[0]: o for o in obs}
    init = init or {"mode": "global"}
    known = init.get("mode") == "known"

    pf = PlanParticleFilter(plan, h_units, cfg)
    t0 = on.t[on.valid][0] if on.valid.any() else on.t[0]
    initialised = False
    window: OrderedDict = OrderedDict()  # lm_id -> (t, p_grav_online) for the observation window
    causal_map: OrderedDict = OrderedDict()  # lm_id -> (t_last, p_grav_online) for structural registration
    cam_hist: list = []  # (t, c_grav_online)
    prev_c = prev_yaw = None
    step_hist: list = []
    last_update_t = -np.inf
    prev_used_ids: set = set()
    track, snaps, events = [], [], []
    kidnapped = False
    prev_loop = False
    low_since, last_reg_t = None, -np.inf
    t_track_start = None
    last_valid_t = None
    map_reset = False

    def window_map(t_now, span):
        while causal_map and next(iter(causal_map.values()))[0] < t_now - max(span, cfg.reg_window_s):
            causal_map.popitem(last=False)
        pts = np.array([v[1] for v in causal_map.values() if v[0] >= t_now - span])
        cams = np.array([c for tc, c in cam_hist if tc >= t_now - span])
        return pts, cams

    for k in range(len(on.t)):
        t = on.t[k]
        if not on.valid[k] or on.state[k] != "Tracking":
            # Tracking lost: stella keeps the map and tries to RELOCALISE; the odometry frame stays valid.
            # Only a return to "Initializing" means a new map (unrelated coordinates).
            window.clear()
            if on.state[k] == "Initializing" and last_valid_t is not None:
                map_reset = True
            if initialised:
                track.append((t, int(on.idx[k]), *[np.nan] * 6, "lost", 0, 0.0, 0))
            continue
        Tw = on.T[k]
        c = Rg @ Tw[:3, 3]
        R_gp = Rg @ Tw[:3, :3]
        yaw = _yaw_grav(R_gp)
        t_track_start = t if t_track_start is None else t_track_start
        gap_s = (t - last_valid_t) if last_valid_t is not None else 0.0
        resumed = gap_s > 0.2 and not map_reset
        last_valid_t = t

        # ---- motion (online odometry), with jump detection
        kind = "predict"
        if prev_c is None or map_reset:
            prev_c, prev_yaw = c, yaw
        dp = (c - prev_c)[:2]
        step = float(np.linalg.norm(c - prev_c))
        med = float(np.median(step_hist[-60:])) if len(step_hist) >= 10 else None
        loop_edge = bool(on.loop_ba[k]) != prev_loop
        prev_loop = bool(on.loop_ba[k])
        # physical speed bound: hand-held / helmet capture never exceeds ~8 m/s, while pose-graph corrections
        # teleport the online pose (relative step-size rules misfire under aggressive motion)
        dt_step = gap_s if gap_s > 0 else 1.0 / 30.0
        speed_mps = step * s_prior / max(dt_step, 1e-3)
        is_jump = map_reset or loop_edge or (not resumed and speed_mps > cfg.jump_speed_mps)
        if is_jump:
            window.clear()
            causal_map.clear()
            cam_hist.clear()
            kind = "jump"
            if map_reset:
                events.append({"t": float(t - t0), "event": "slam_map_reset"})
                low_since, last_reg_t = -np.inf, -np.inf  # request structural recovery as soon as possible
                map_reset = False
        elif resumed:
            kind = "resume"
            events.append({"t": float(t - t0), "event": "tracking_resumed_after_gap", "gap_s": float(gap_s),
                           "displacement_units": step})
        else:
            step_hist.append(step)
        cam_hist.append((t, c))

        # ---- causal local structure (observation window + registration map)
        if on.idx[k] in obs_by_idx:
            _, _, ids, pc = obs_by_idx[on.idx[k]]
            pw = (pc @ Tw[:3, :3].T + Tw[:3, 3]) @ Rg.T  # online world position, gravity axes
            for i, lid in enumerate(ids.tolist()):
                window[lid] = (t, pw[i])
                window.move_to_end(lid)
                causal_map[lid] = (t, pw[i])
                causal_map.move_to_end(lid)
        while window and next(iter(window.values()))[0] < t - cfg.window_s:
            window.popitem(last=False)
        while cam_hist and cam_hist[0][0] < t - cfg.reg_window_s:
            cam_hist.pop(0)

        # ---- initialisation
        if not initialised:
            if known:
                if t < init.get("t", -np.inf):
                    prev_c, prev_yaw = c, yaw
                    continue
                phi = _wrap(init["yaw"] - yaw)
                pf.init_gaussian(init["x"], init["y"], phi, s_prior, init.get("sigma_xy", 0.5),
                                 np.deg2rad(init.get("sigma_yaw_deg", 10.0)), init.get("sigma_log_s", 0.15))
                initialised = True
                log(f"[pf] initialised (known start pose) at t={t - t0:.1f}s, N={len(pf.P)}")
            elif t - t_track_start >= cfg.global_init_after_s and t - last_reg_t >= 2.0:
                pts, cams = window_map(t, cfg.reg_window_s)
                last_reg_t = t
                if len(pts) > 50 and len(cams) > 30:
                    hyps, info = structural_hypotheses(plan, pts, cams, h_units, s_prior, cfg, c[:2])
                    events.append({"t": float(t - t0), "event": "global_init_attempt", "n_hypotheses": len(hyps), **info})
                    if hyps:
                        pf.init_hypotheses(hyps)
                        initialised = True
                        log(f"[pf] initialised (structural hypotheses: {len(hyps)}) at t={t - t0:.1f}s")
            if not initialised:
                prev_c, prev_yaw = c, yaw
                track.append((t, int(on.idx[k]), *[np.nan] * 6, "not_localized", 0, 0.0, 0))
                continue
            prev_c, prev_yaw = c, yaw
            dp, is_jump = np.zeros(2), False

        if is_jump:
            pf.predict(np.zeros(2), 0.0, jump=True)
        elif resumed:
            # relocalised in the same map: the displacement across the gap is valid but less certain
            pf.predict(dp, _wrap(yaw - prev_yaw))
            extra = 0.2 + 0.05 * step * s_prior
            pf.P[:, 0:2] += pf.rng.normal(0, extra, (len(pf.P), 2))
            pf.P[:, 2] = _wrap(pf.P[:, 2] + pf.rng.normal(0, np.deg2rad(3.0), len(pf.P)))
        else:
            pf.predict(dp, _wrap(yaw - prev_yaw))
        prev_c, prev_yaw = c, yaw

        if kidnap_at_s is not None and not kidnapped and t - t0 >= kidnap_at_s:
            pf.P[:, 0] += 8.0
            pf.P[:, 2] = _wrap(pf.P[:, 2] + np.pi / 2)
            kidnapped = True
            kind = "kidnap"
            events.append({"t": float(t - t0), "event": "kidnap", "shift_m": 8.0, "rot_deg": 90})
            log(f"[pf] kidnapped particles at t={t - t0:.1f}s")

        n_pts, beta, n_rand = 0, 0.0, 0
        if t - last_update_t >= cfg.update_every_s and len(window) > 30:
            ids = np.fromiter(window.keys(), np.int64)
            P = np.array([v[1] for v in window.values()])
            xy, w = pf.local_wall_points(P - c)
            if len(xy) >= cfg.min_wall_cells:
                id_set = set(ids.tolist())
                new_frac = len(id_set - prev_used_ids) / max(len(id_set), 1)
                beta = cfg.beta0 * new_frac
                if beta > 0.05:
                    if debug is not None:
                        debug.append({"t": float(t), "yaw_grav": yaw, "xy": xy.copy(), "w": w.copy(), "beta": beta,
                                      "est_before": pf.estimate(yaw)[0].tolist()})
                    pf.update(xy, w, beta)
                    prev_used_ids = id_set
                    last_update_t = t
                    n_pts = len(xy)
                    kind = "update" if kind == "predict" else kind
                    if pf.ess() < cfg.ess_frac * len(pf.P):
                        n_rand = pf.resample(cfg.n_particles, s_prior, 0.2)
                    # ---- recovery: sustained poor fit -> structural hypotheses from the recent causal map
                    ratio = pf.w_fast / max(pf.w_slow, 1e-12) if pf.w_slow else 1.0
                    low_since = (low_since or t) if ratio < cfg.recovery_ratio else None
                    if low_since is not None and t - low_since >= cfg.recovery_trigger_s and t - last_reg_t >= cfg.recovery_cooldown_s:
                        pts, cams = window_map(t, cfg.reg_window_s)
                        last_reg_t = t
                        if len(pts) > 50 and len(cams) > 30:
                            hyps, info = structural_hypotheses(plan, pts, cams, h_units, s_prior, cfg, c[:2])
                            n_inj = pf.inject_hypotheses(hyps, cfg.recovery_inject_frac)
                            events.append({"t": float(t - t0), "event": "recovery_injection", "n_hypotheses": len(hyps), "n_injected": n_inj, **info})
                            if n_inj:
                                kind = "recovery"
                                low_since = None
        est, cov = pf.estimate(yaw)
        modes = pf.modes() if kind in ("update", "jump", "kidnap", "recovery") else []
        track.append((t, int(on.idx[k]), est[0], est[1], est[2], est[3], float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1])),
                      kind, n_pts, beta, n_rand))
        if kind != "predict" and (len(track) % snapshot_every == 0 or kind != "update"):
            sub = pf.P[np.random.default_rng(len(track)).choice(len(pf.P), min(400, len(pf.P)), replace=False)]
            snaps.append((t, sub[:, 0], sub[:, 1], _wrap(sub[:, 2] + yaw), pf.weights().max(), modes))

    # ---- write
    with open(out_dir / "pf_track.csv", "w") as f:
        f.write("t,frame_idx,x,y,yaw,metres_per_unit,std_x,std_y,kind,n_wall_cells,beta,n_random\n")
        for r in track:
            f.write(",".join(str(v) if not isinstance(v, float) else f"{v:.6f}" for v in r) + "\n")
    M = max((len(s[1]) for s in snaps), default=0)

    def _pad(i):
        return np.array([np.pad(np.asarray(s[i], np.float32), (0, M - len(s[i])), constant_values=np.nan) for s in snaps],
                        np.float32).reshape(len(snaps), M)

    np.savez_compressed(out_dir / "pf_particles.npz", t=np.array([s[0] for s in snaps]), x=_pad(1), y=_pad(2), yaw=_pad(3))
    (out_dir / "pf_modes.json").write_text(json.dumps([{"t": s[0], "modes": s[5]} for s in snaps if s[5]]))
    if debug is not None:
        import pickle

        with open(out_dir / "pf_debug_updates.pkl", "wb") as f:
            pickle.dump(debug, f)
    arr = np.array([r[:8] for r in track], float)
    kinds = [r[8] for r in track]
    report = {"n_steps": len(track), "n_updates": int(sum(k == "update" for k in kinds)),
              "n_jumps": int(sum(k == "jump" for k in kinds)), "n_lost": int(sum(k == "lost" for k in kinds)),
              "frac_steps_localized": float(np.mean([k not in ("not_localized", "lost") for k in kinds])) if kinds else 0.0,
              "events": events, "init": init, "kidnap_at_s": kidnap_at_s, "config": cfg.__dict__,
              "assumptions": ["gravity + scale prior from geometry stage (IMU / camera-height), not from plan or GT",
                              "start pose from init (reported)" if known else "global: structural registration of causal local map"]}
    if gt is not None and len(arr):
        tg, Tg = gt
        ok = np.isfinite(arr[:, 2])
        Tq, inside = interpolate_poses(arr[ok, 0], tg, Tg, max_gap=0.2)
        e = np.linalg.norm(arr[ok][inside, 2:4] - Tq[inside, :2, 3], axis=1)
        te = arr[ok][inside, 0] - t0
        yaw_gt = np.arctan2(Tq[inside, 1, 2], Tq[inside, 0, 2])
        eyaw = np.degrees(np.abs(_wrap(arr[ok][inside, 4] - yaw_gt)))

        def first_stable(err, tt, thr=1.0, n=150):
            for i in range(len(err) - n + 1):
                if np.all(err[i:i + n] < thr):
                    return float(tt[i])
            return None

        rep = {"n_eval": int(len(e)), "err_median_m": float(np.median(e)) if len(e) else None,
               "err_rmse_m": float(np.sqrt(np.mean(e ** 2))) if len(e) else None,
               "err_p90_m": float(np.percentile(e, 90)) if len(e) else None,
               "frac_below_0.5m": float((e < 0.5).mean()) if len(e) else 0.0, "frac_below_1m": float((e < 1.0).mean()) if len(e) else 0.0,
               "yaw_err_median_deg": float(np.median(eyaw)) if len(e) else None,
               "converged_below_1m_at_s": first_stable(e, te),
               "gt_poses_total": int(len(tg))}
        conv = rep["converged_below_1m_at_s"]
        if conv is not None:
            after = te >= conv
            rep["after_convergence"] = {"err_median_m": float(np.median(e[after])), "frac_below_1m": float((e[after] < 1).mean())}
        if kidnap_at_s is not None:
            after = te >= kidnap_at_s
            rep["recovery_time_after_kidnap_s"] = (lambda r: None if r is None else r - kidnap_at_s)(first_stable(e[after], te[after]))
            rep["err_median_before_kidnap_m"] = float(np.median(e[~after])) if (~after).any() else None
        report["evaluation_vs_gt"] = rep
        np.savetxt(out_dir / "pf_error.csv", np.column_stack([te, e, eyaw]), delimiter=",", header="t_rel_s,err_xy_m,err_yaw_deg", fmt="%.4f")
    (out_dir / "pf_report.json").write_text(json.dumps(report, indent=2, default=float))
    log(f"[pf] done: {json.dumps(report.get('evaluation_vs_gt', {}))}")
    return report
