"""Config-driven, resumable processing pipeline.

Stages (in order): ingest -> mask -> plan -> slam -> geometry -> align -> localize -> objects -> dense -> report
Each stage reads earlier stage outputs from the run directory and is skipped when already done with
an identical config section (use --force to recompute).
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from slam3d.config import REPO, Status, resolve, run_dir

STAGES = ["ingest", "mask", "plan", "slam", "geometry", "dense", "align", "discrepancy", "localize", "objects", "report"]


def _log_to(rdir: Path):
    logf = open(rdir / "pipeline.log", "a")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    return log


def _frames(rdir):
    ing = rdir / "ingest"
    rows = np.genfromtxt(ing / "frames.csv", delimiter=",", skip_header=1, dtype=str)
    return rows[:, 1].astype(float), [str(ing / r[2]) for r in rows]


def _gt(cfg):
    p = resolve(cfg.get("evaluation", {}).get("gt_tum"))
    if p and p.exists():
        from slam3d.io.tum import read_tum

        return read_tum(p)
    return None


# ------------------------------------------------------------------------------------------ stages
def stage_ingest(cfg, rdir, log):
    inp, ic = cfg["input"], cfg.get("ingest", {})
    out = rdir / "ingest"
    if inp["type"] == "hilti_bag":
        from slam3d.ingest.stitch import stitch_bag
        from slam3d.io.rosbag_hilti import read_imu

        masks = inp.get("fisheye_masks") or [None, None]
        meta = stitch_bag(str(resolve(inp["bag"])), str(resolve(inp["calib"])), str(out), width=ic.get("width", 1920),
                          radius=ic.get("stitch_radius_m", 3.0), mask0_path=str(resolve(masks[0])) if masks[0] else None,
                          mask1_path=str(resolve(masks[1])) if masks[1] else None, workers=ic.get("workers", 12), log=log)
        imu = read_imu(str(resolve(inp["bag"])))
        np.savetxt(out / "imu.csv", imu, delimiter=",", fmt="%.9f", header="t_s,gx,gy,gz,ax,ay,az (IMU frame)")
    elif inp["type"] == "equirect_video":
        from slam3d.ingest.video import ingest_video

        meta = ingest_video(str(resolve(inp["video"])), str(out), width=ic.get("width", 1920), every_n=ic.get("every_n", 1),
                            start_s=ic.get("start_s", 0.0), end_s=ic.get("end_s"), time_offset_s=ic.get("time_offset_s", 0.0), log=log)
        if inp.get("imu_csv") and inp.get("R_pano_imu"):
            shutil.copy(resolve(inp["imu_csv"]), out / "imu.csv")
            meta.update(R_cam0_pano=np.eye(3).tolist(), T_cam0_imu=np.pad(np.array(inp["R_pano_imu"]), ((0, 1), (0, 1))).tolist(),
                        timeshift_cam_imu=inp.get("timeshift_cam_imu", 0.0))
            (out / "stitch.json").write_text(json.dumps(meta, indent=2))
    else:
        raise ValueError(f"unknown input type {inp['type']}")
    return {"n_frames": meta["n_frames"]}


def stage_mask(cfg, rdir, log):
    from slam3d.ingest.masks import build_static_mask

    mc = cfg.get("mask", {})
    if not mc.get("enabled", True):
        return {"enabled": False}
    kw = {k: v for k, v in mc.items() if k in ("std_rel_thresh", "min_row_frac", "dilate_px", "bottom_band_frac", "n_samples")}
    mask, info, _ = build_static_mask(rdir / "ingest", **kw)
    log(f"[mask] masked fraction {info['masked_fraction']:.3f}")
    return info


def stage_plan(cfg, rdir, log):
    from slam3d.floorplan import prepare

    fc = cfg["floorplan"]
    out = rdir / "floorplan"
    kind = fc.get("type", "mask")
    common = {"res": fc.get("res_m", 0.05)}
    if kind == "mask":
        plan = prepare.from_mask(str(resolve(fc["path"])), fc["src_res_m_per_px"], structure_is_dark=fc.get("structure_is_dark", True), **common)
    elif kind in ("drawing", "pdf"):
        plan = prepare.from_drawing(str(resolve(fc["path"])), fc["src_res_m_per_px"], save_editable_mask=str(out / "editable_structure_mask.png"),
                                    **common, **fc.get("extraction", {}))
    elif kind == "dxf":
        plan = prepare.from_dxf(str(resolve(fc["path"])), layer_regex=fc.get("layer_regex", ".*"), **common)
    else:
        raise ValueError(kind)
    plan.save(out)
    if fc.get("image"):
        img = cv2.imread(str(resolve(fc["image"])))
        if img is not None:  # background drawing for the viewer, same raster as the source mask
            s = 3000 / max(img.shape[:2])
            cv2.imwrite(str(out / "drawing.jpg"), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
            (out / "drawing.json").write_text(json.dumps({"src_res_m_per_px": fc["src_res_m_per_px"], "scale": s,
                                                          "width": img.shape[1], "height": img.shape[0]}))
    return {k: plan.meta[k] for k in ("n_columns", "structure_fraction", "exterior_fraction")}


def stage_slam(cfg, rdir, log):
    from slam3d.slam import stella

    sc = cfg.get("slam", {})
    ing = rdir / "ingest"
    meta = json.loads((ing / "stitch.json").read_text())
    t, _ = _frames(rdir)
    fps = float(1.0 / np.median(np.diff(t)))
    out = rdir / "slam"
    out.mkdir(parents=True, exist_ok=True)
    stella.make_config(meta["width"], meta["height"], fps, out / "stella_config.yaml", sc.get("config_overrides"))
    mask = ing / "static_mask.png"
    rc = stella.run(ing / "frames.csv", out, out / "stella_config.yaml", resolve(sc.get("vocab", "third_party/stella/orb_vocab.fbow")),
                    mask=mask if mask.exists() else None, obs_every=sc.get("obs_every", 3), max_frames=sc.get("max_frames"),
                    log_path=out / "stella.log", container_name=f"slam3d-{cfg['run_name']}"[:60])
    if rc != 0 or not (out / "frame_trajectory.txt").exists():
        raise RuntimeError(f"stella driver failed (rc={rc}); see {out / 'stella.log'}")
    return {"fps": fps, "rc": rc}


def stage_geometry(cfg, rdir, log):
    from slam3d.slam.postprocess import postprocess

    gc = cfg.get("geometry", {})
    geo = postprocess(rdir / "slam", rdir / "ingest", rdir / "geometry", gc.get("gravity", "auto"),
                      gc.get("camera_height_m", 1.3), log=log)
    res = {"tracking": geo["tracking"], "scale": geo["scale"], "reprojection": geo["reprojection"]}
    gt = _gt(cfg)
    if gt is not None:
        from slam3d.eval.metrics import ate, rpe
        from slam3d.io.tum import read_tum

        t, T = read_tum(rdir / "geometry/trajectory_grav.tum")
        r_sim3, _, (s, _, _), _ = ate(t, T[:, :3, 3], gt[0], gt[1][:, :3, 3], align="sim3")
        r_rpe = rpe(t, T, gt[0], gt[1], delta_m=1.0)
        geo_eval = {"slam_ate_sim3": r_sim3, "slam_rpe_1m": r_rpe,
                    "camera_height_scale_vs_gt_sim3_ratio": (geo["scale"]["metres_per_unit"] / s) if geo["scale"]["metres_per_unit"] else None}
        ton, Ton = read_tum(rdir / "geometry/online_grav.tum")
        geo_eval["online_ate_sim3"] = ate(ton, Ton[:, :3, 3], gt[0], gt[1][:, :3, 3], align="sim3")[0]
        (rdir / "geometry/eval.json").write_text(json.dumps(geo_eval, indent=2))
        log(f"[geometry] SLAM ATE(sim3) rmse {r_sim3['rmse']:.3f} m coverage {r_sim3['coverage']:.3f}; "
            f"height-prior scale / GT scale = {geo_eval['camera_height_scale_vs_gt_sim3_ratio']}")
        res["eval"] = geo_eval
    return res


def _pose_plan(S, floor_z, R_grav_pano, p_grav):
    s = float(np.hypot(S[0, 0], S[1, 0]))
    th = float(np.arctan2(S[1, 0], S[0, 0]))
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    T = np.tile(np.eye(4), (len(p_grav), 1, 1))
    T[:, :3, :3] = Rz @ R_grav_pano
    T[:, :2, 3] = p_grav[:, :2] @ S[:2, :2].T + S[:2, 2]
    T[:, 2, 3] = s * (p_grav[:, 2] - floor_z)
    return T


def stage_align(cfg, rdir, log):
    from slam3d.align import register
    from slam3d.align.bev import wall_evidence
    from slam3d.floorplan.prepare import FloorPlan
    from slam3d.geometry.transforms import apply_sim2, interpolate_poses
    from slam3d.io.ply import read_ply, write_ply
    from slam3d.io.tum import read_tum, write_tum

    ac = cfg.get("align", {})
    out = rdir / "align"
    out.mkdir(parents=True, exist_ok=True)
    plan = FloorPlan.load(rdir / "floorplan")
    geo = json.loads((rdir / "geometry/geometry.json").read_text())
    L, cols = read_ply(rdir / "geometry/landmarks_grav.ply")
    t, T = read_tum(rdir / "geometry/trajectory_grav.tum")
    h = geo["scale"]["camera_height_units"]
    variants = [("sparse_landmarks", L)]
    dense_ply = rdir / "dense/dense_grav.ply"
    if ac.get("use_dense", True) and dense_ply.exists():
        D, _ = read_ply(dense_ply)
        variants.append(("sparse_landmarks+panovggt_dense", np.vstack([L, D])))
    path = T[:: max(1, len(T) // 400), :2, 3]
    init_S = None
    corr_file = resolve(ac.get("correspondences"))
    used_corr_t = []
    if corr_file and corr_file.exists():
        corr = json.loads(corr_file.read_text())["pairs"]
        ts = np.array([c["t"] for c in corr])
        Tq, ok = interpolate_poses(ts, t, T)
        src = Tq[ok, :2, 3]
        dst = np.array([c["plan_xy"] for c in corr])[ok]
        man = register.from_correspondences(src, dst, with_scale=True)
        init_S = man.S
        used_corr_t = ts[ok].tolist()
        (out / "manual_alignment.json").write_text(json.dumps(man.to_json(), indent=2))
        log(f"[align] manual init from {len(src)} correspondences, residuals {np.round(man.confidence['residual_m'], 2)}")
    mode = ac.get("mode", "auto")
    sp = geo["scale"]["metres_per_unit"]
    rng = tuple(ac.get("scale_range", (0.6, 1.6) if sp else (0.05, 20.0)))
    results = []
    for vname, pts in variants:
        bev_v = wall_evidence(pts, T[:, :3, 3], h, **ac.get("bev", {}))
        bev_v.meta["evidence_source"] = vname
        log(f"[align] wall evidence ({vname}): {bev_v.meta}")
        if len(bev_v.xy) < 10:
            continue
        if mode == "manual_only" and init_S is not None:
            fit = register.evaluate_fit(plan, init_S, bev_v.xy, path)
            alg_v = register.Alignment(init_S, 0.0, dict(fit, quality=fit["wall_inlier_frac"], status="manual"), {"method": "manual_only"})
        else:
            alg_v = register.align(plan, bev_v.xy, bev_v.weight, path, scale_prior=sp, scale_range=rng,
                                   n_scales=ac.get("n_scales", 15 if sp else 40), init_S=init_S, log=log)
        log(f"[align] variant {vname}: {alg_v.confidence.get('status')} quality {alg_v.confidence.get('quality'):.3f}")
        results.append((vname, bev_v, alg_v))
    if not results:
        raise RuntimeError("no usable structural evidence for floor-plan alignment (need wall-like reconstruction)")
    # automatic selection WITHOUT ground truth: a confident registration beats an ambiguous one; then prefer
    # variants with enough evidence; then fit quality
    def _rank(r):
        conf = r[2].confidence.get("status") in ("confident", "manual")
        enough = len(r[1].xy) >= ac.get("min_wall_cells", 50)
        return (conf and enough, conf, enough, r[2].confidence.get("quality", 0.0))

    _, bev, alg = max(results, key=_rank)
    S = alg.S
    R_gp = T[:, :3, :3]
    Tp_global = _pose_plan(S, bev.floor_z, R_gp, T[:, :3, 3])
    write_tum(out / "trajectory_plan_global.tum", t, Tp_global, "T_plan_pano with one global similarity (plan metres)")
    Tp, seg_report = Tp_global, None
    sc = ac.get("segment_refine", {})
    if sc.get("enabled", True):
        from slam3d.align.segments import refine_segments
        from slam3d.slam import stella

        lms = stella.load_landmarks(rdir / "slam/landmarks.bin")
        kfr = stella.load_keyframes(rdir / "slam/keyframes.csv")
        Rg_ = np.array(geo["R_grav_slam"])
        L_all = lms["xyz"] @ Rg_.T - np.array(geo["origin_offset_in_rotated_slam"])
        k2t = dict(zip(kfr["id"].tolist(), kfr["t"].tolist()))
        L_time = np.array([k2t.get(int(k), np.nan) for k in lms["first_kf"]])
        good_l = (lms["nobs"] >= 3) & np.isfinite(L_time)
        S_pose, seg_report = refine_segments(plan, S, t, T, L_all[good_l], L_time[good_l], h,
                                             window_s=sc.get("window_s", 45.0), step_s=sc.get("step_s", 22.5))
        s_p = np.hypot(S_pose[:, 0, 0], S_pose[:, 1, 0])
        th_p = np.arctan2(S_pose[:, 1, 0], S_pose[:, 0, 0])
        Rz = np.zeros((len(t), 3, 3))
        Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = np.cos(th_p), -np.sin(th_p), np.sin(th_p), np.cos(th_p), 1.0
        Tp = Tp_global.copy()
        Tp[:, :3, :3] = Rz @ R_gp
        Tp[:, :2, 3] = np.einsum("nij,nj->ni", S_pose[:, :2, :2], T[:, :2, 3]) + S_pose[:, :2, 2]
        Tp[:, 2, 3] = s_p * (T[:, 2, 3] - bev.floor_z)
        log(f"[align] segment refinement: {sum(r['accepted'] for r in seg_report)}/{len(seg_report)} windows accepted")
    write_tum(out / "trajectory_plan.tum", t, Tp, "T_plan_pano: plan metres (z above estimated floor); segment-refined where accepted")
    s = float(np.hypot(S[0, 0], S[1, 0]))
    Lp = np.column_stack([apply_sim2(S, L[:, :2]), s * (L[:, 2] - bev.floor_z)])
    write_ply(out / "landmarks_plan.ply", Lp, cols)
    wall_plan = apply_sim2(S, bev.xy)
    np.savetxt(out / "wall_evidence_plan.csv", np.column_stack([wall_plan, bev.weight, plan.dist_at(wall_plan, 10.0)]),
               delimiter=",", header="x,y,weight,dist_to_plan_m", fmt="%.3f")
    info = dict(alg.to_json(), floor_z_units=bev.floor_z, bev=bev.meta, used_correspondence_times=used_corr_t,
                scale_status="externally calibrated by plan registration (similarity)",
                variants=[{"evidence": n, "n_wall_cells": int(len(b.xy)), "S_plan_from_grav": a.S.tolist(),
                           "confidence": a.confidence} for n, b, a in results],
                selected_evidence=bev.meta["evidence_source"], segment_refinement=seg_report)
    # observed-vs-planned discrepancy candidates: strong wall evidence far from any planned structure
    d = plan.dist_at(wall_plan, 10.0)
    info["unplanned_structure_candidates"] = int(((d > 0.5) & (bev.weight > np.percentile(bev.weight, 75))).sum())
    (out / "alignment.json").write_text(json.dumps(info, indent=2, default=float))
    # overlay
    img = plan.preview(max_side=10**6)
    def px(xy):
        return plan.world_to_px(xy).round().astype(int)
    for (x, y), dd in zip(px(wall_plan), d):
        cv2.circle(img, (int(x), int(y)), 1, (0, 170, 0) if dd < 0.2 else (0, 140, 255), -1)
    cv2.polylines(img, [px(Tp[:, :2, 3]).reshape(-1, 1, 2)], False, (255, 0, 0), 2)
    gt = _gt(cfg)
    res = {"alignment": info["confidence"], "scale": s}
    if gt is not None:
        from slam3d.eval.metrics import ate

        cv2.polylines(img, [px(gt[1][:, :2, 3]).reshape(-1, 1, 2)], False, (0, 0, 255), 1)
        r2d, err, _, (ia, ib) = ate(t, Tp[:, :3, 3], gt[0], gt[1][:, :3, 3], align="none", dims=2)
        ev = {"plan_frame_2d_error_no_extra_alignment": r2d,
              "plan_frame_2d_error_global_transform_only": ate(t, Tp_global[:, :3, 3], gt[0], gt[1][:, :3, 3], align="none", dims=2)[0]}
        if used_corr_t:
            far = np.min(np.abs(gt[0][ia][:, None] - np.array(used_corr_t)[None]), axis=1) > 5.0
            ev["held_out_frames_2d_error_median_m"] = float(np.median(err[far])) if far.any() else None
        # scale accuracy of the registration (vs GT similarity of the SLAM trajectory)
        r_sim, _, (sg, _, _), _ = ate(t, T[:, :3, 3], gt[0], gt[1][:, :3, 3], align="sim3")
        ev["registration_scale_over_gt_scale"] = s / sg
        for vname, vbev, valg in results:  # ablation record only; selection above never sees GT
            Tv = _pose_plan(valg.S, vbev.floor_z, R_gp, T[:, :3, 3])
            rv = ate(t, Tv[:, :3, 3], gt[0], gt[1][:, :3, 3], align="none", dims=2)[0]
            ev.setdefault("variants", {})[vname] = {"median": rv["median"], "rmse": rv["rmse"], "max": rv["max"],
                                                    "status": valg.confidence.get("status"), "n_wall_cells": int(len(vbev.xy))}
        (out / "eval.json").write_text(json.dumps(ev, indent=2, default=float))
        res["eval"] = ev
        log(f"[align] plan-frame 2D error: median {r2d['median']:.3f} m, rmse {r2d['rmse']:.3f} m, score {r2d['hilti_score']:.1f}, coverage {r2d['coverage']:.3f}")
    cv2.imwrite(str(out / "overlay.png"), img)

    # ---- batch coverage completion (flagged; never used by causal localization)
    from slam3d.align.fill import FLAGS, fill_trajectory

    ft, _ = _frames(rdir)
    imu_p = rdir / "ingest/imu.csv"
    imu = np.loadtxt(imu_p, delimiter=",", comments="#") if imu_p.exists() else None
    tf_, Tf_, flags, finfo = fill_trajectory(ft, t, Tp, imu)
    write_tum(out / "trajectory_plan_filled.tum", tf_, Tf_, "T_plan_pano incl. flagged fills; see trajectory_plan_filled_flags.csv")
    np.savetxt(out / "trajectory_plan_filled_flags.csv", np.column_stack([tf_, flags]), delimiter=",", fmt=["%.6f", "%d"],
               header="t,flag (0 tracked, 1 static_fill, 2 gap_interpolated)")
    info["coverage_fill"] = dict(finfo, counts={FLAGS[k]: int((flags == k).sum()) for k in FLAGS})
    if gt is not None:
        rf, ef, _, (iaf, ibf) = ate(tf_, Tf_[:, :3, 3], gt[0], gt[1][:, :3, 3], align="none", dims=2)
        per = {FLAGS[k]: float(np.median(ef[flags[ibf] == k])) if (flags[ibf] == k).any() else None for k in FLAGS}
        ev["filled_trajectory_2d"] = dict(rf, median_error_by_flag=per)
        (out / "eval.json").write_text(json.dumps(ev, indent=2, default=float))
        log(f"[align] filled trajectory: coverage {rf['coverage']:.3f}, median {rf['median']:.3f} m, Hilti score {rf['hilti_score']:.1f}, by flag {per}")
    (out / "alignment.json").write_text(json.dumps(info, indent=2, default=float))
    return res


def stage_discrepancy(cfg, rdir, log):
    from slam3d.align.bev import wall_evidence
    from slam3d.align.discrepancy import analyse
    from slam3d.floorplan.prepare import FloorPlan
    from slam3d.geometry.transforms import apply_sim2
    from slam3d.io.ply import read_ply
    from slam3d.io.tum import read_tum

    dc = cfg.get("discrepancy", {})
    a = json.loads((rdir / "align/alignment.json").read_text())
    geo = json.loads((rdir / "geometry/geometry.json").read_text())
    plan = FloorPlan.load(rdir / "floorplan")
    t, T = read_tum(rdir / "geometry/trajectory_grav.tum")
    pts, _ = read_ply(rdir / "geometry/landmarks_grav.ply")
    if (rdir / "dense/dense_grav.ply").exists():
        D, _ = read_ply(rdir / "dense/dense_grav.ply")
        pts = np.vstack([pts, D])
    bev = wall_evidence(pts, T[:, :3, 3], geo["scale"]["camera_height_units"])
    S = np.array(a["S_plan_from_grav"])
    tp, Tp = read_tum(rdir / "align/trajectory_plan.tum")
    cams = Tp[:: max(1, len(Tp) // 300), :2, 3]
    rep = analyse(plan, apply_sim2(S, bev.xy), bev.weight, cams, a["confidence"].get("status", ""), rdir / "discrepancy", **dc)
    log(f"[discrepancy] {rep.get('status')}: {rep.get('planned_boundary_length_m')}, extra clusters {len(rep.get('observed_not_in_plan_clusters', []))}")
    return {k: v for k, v in rep.items() if k != "observed_not_in_plan_clusters"}


def _init_from_cfg(cfg):
    lc = cfg.get("localize", {})
    ini = dict(lc.get("init", {"mode": "global"}))
    if ini.get("mode") == "known" and ini.get("hilti_init_csv"):
        from slam3d.geometry.transforms import quat_to_R

        rows = [r.strip().split(",") for r in open(resolve(ini["hilti_init_csv"])) if not r.startswith("#")]
        row = next(r for r in rows if r[0] == ini["sequence"])
        tt, x, y, z, qx, qy, qz, qw = (float(v) for v in row[2:10])
        R = quat_to_R([qx, qy, qz, qw])  # T_map_cam0; pano forward == cam0 +z
        ini.update(t=tt, x=x, y=y, yaw=float(np.arctan2(R[1, 2], R[0, 2])), source="Hilti init_gt_poses.csv (start pose only)")
    return ini


def stage_localize(cfg, rdir, log):
    from slam3d.floorplan.prepare import FloorPlan
    from slam3d.localize.pf import PFConfig
    from slam3d.localize.replay import replay

    lc = cfg.get("localize", {})
    plan = FloorPlan.load(rdir / "floorplan")
    pcfg = PFConfig(**lc.get("pf", {}))
    ini = _init_from_cfg(cfg)
    gt = _gt(cfg)
    results = {}
    exps = lc.get("experiments", [{"name": "main"}])
    for ex in exps:
        e_ini = dict(ini, **ex.get("init", {}))
        rep = replay(rdir / "slam", rdir / "geometry/geometry.json", plan, rdir / "localize" / ex["name"], pcfg,
                     init=e_ini, kidnap_at_s=ex.get("kidnap_at_s"), gt=gt, log=log)
        results[ex["name"]] = rep.get("evaluation_vs_gt", {"n_updates": rep["n_updates"]})
    return results


def stage_objects(cfg, rdir, log):
    from slam3d.semantics.pipeline import run_objects

    return run_objects(cfg, rdir, log)


def stage_dense(cfg, rdir, log):
    dc = cfg.get("dense", {})
    if not dc.get("enabled", False):
        return {"enabled": False}
    from slam3d.dense.panovggt_windows import run_dense

    return run_dense(cfg, rdir, log)


def stage_report(cfg, rdir, log):
    from slam3d.eval.report import write_report

    return write_report(cfg, rdir)


FUNCS = {n: globals()[f"stage_{n}"] for n in STAGES}


def run(cfg, stages=None, force=False):
    rdir = run_dir(cfg)
    (rdir / "config_used.yaml").write_text(Path(cfg["_config_path"]).read_text())
    log = _log_to(rdir)
    status = Status(rdir)
    for name in stages or STAGES:
        part = {"stage": name, "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")}}
        if not force and status.is_done(name, part):
            log(f"[{name}] already done (same config) -> skip")
            continue
        with status.stage(name, part, log) as rec:
            rec["result"] = FUNCS[name](cfg, rdir, log)
            status.save()
    if (rdir / "ingest/frames.csv").exists():
        stage_report(cfg, rdir, log)  # refresh with final stage states/timings
    return rdir
