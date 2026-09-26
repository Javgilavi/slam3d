"""Collect stage status and metrics of a run into report.json and REPORT.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_report(cfg, rdir: Path) -> dict:
    status = _load(rdir / "status.json") or {}
    frames = np.genfromtxt(rdir / "ingest/frames.csv", delimiter=",", skip_header=1, usecols=(1,))
    duration = float(frames[-1] - frames[0]) if len(frames) > 1 else None
    stage_times = {k: v.get("runtime_s") for k, v in status.items() if isinstance(v, dict) and v.get("state") == "done"}
    total = sum(t for k, t in stage_times.items() if t and k != "report")
    rep = {
        "run": cfg["run_name"], "video_duration_s": duration, "n_frames": int(len(frames)),
        "processing_time_s": total, "processing_to_video_ratio": total / duration if duration else None,
        "stages": {k: {kk: v.get(kk) for kk in ("state", "runtime_s", "peak_rss_mb", "peak_gpu_used_mb", "peak_docker_mem_mb", "error")}
                   for k, v in status.items() if isinstance(v, dict)},
        "geometry": _load(rdir / "geometry/geometry.json"),
        "geometry_eval": _load(rdir / "geometry/eval.json"),
        "alignment": _load(rdir / "align/alignment.json"),
        "alignment_eval": _load(rdir / "align/eval.json"),
        "localization": {p.parent.name: _load(p) for p in sorted((rdir / "localize").glob("*/pf_report.json"))},
        "objects": _load(rdir / "objects/objects_report.json"),
        "discrepancy": _load(rdir / "discrepancy/discrepancies.json"),
        "dense": _load(rdir / "dense/dense_report.json"),
        "viewer_validation": _load(rdir / "viewer/validation.json"),
    }
    (rdir / "report.json").write_text(json.dumps(rep, indent=2, default=float))

    L = [f"# Run report: {cfg['run_name']}", ""]
    L.append(f"Video duration {_fmt(duration, 1)} s, {len(frames)} frames. Total processing {_fmt(total, 1)} s "
             f"(ratio {_fmt(rep['processing_to_video_ratio'], 2)}x real time).")
    L += ["", "| stage | state | runtime s | peak RSS MB | peak GPU MB (device) | peak docker MB |", "|---|---|---|---|---|---|"]
    for k, v in rep["stages"].items():
        L.append(f"| {k} | {v['state']} | {_fmt(v['runtime_s'], 1)} | {_fmt(v['peak_rss_mb'], 0)} | {_fmt(v['peak_gpu_used_mb'])} | {_fmt(v['peak_docker_mem_mb'], 0)} |")
    g = rep["geometry"]
    if g:
        tr = g["tracking"]
        L += ["", "## SLAM", f"- tracked fraction {_fmt(tr['tracked_frac'])} (after init {_fmt(tr['tracked_frac_after_init'])}), init after {_fmt(tr['init_time_s'], 1)} s",
              f"- lost segments after init: {tr['lost_segments_after_init']}",
              f"- low-parallax sections: {g['low_parallax_sections']}",
              f"- keyframes {g['n_keyframes']}, landmarks {g['n_landmarks']} ({g['n_landmarks_nobs_ge3']} with >=3 obs)",
              f"- keyframe reprojection: median {_fmt(g['reprojection'].get('angle_deg_median'))} deg, p90 {_fmt(g['reprojection'].get('angle_deg_p90'))} deg",
              f"- gravity: {g['gravity'].get('method')} (spread median {_fmt(g['gravity'].get('angular_spread_deg_median'), 2)} deg)",
              f"- scale: {g['scale'].get('status')}, metres/unit {_fmt(g['scale'].get('metres_per_unit'), 4)}"]
    ge = rep["geometry_eval"]
    if ge:
        a = ge["slam_ate_sim3"]
        L += [f"- vs GT (Sim3-aligned, final trajectory): ATE RMSE {_fmt(a['rmse'])} m, median {_fmt(a['median'])} m, coverage {_fmt(a['coverage'])}",
              f"- vs GT (Sim3-aligned, online poses): ATE RMSE {_fmt(ge['online_ate_sim3']['rmse'])} m",
              f"- camera-height scale prior / GT scale: {_fmt(ge['camera_height_scale_vs_gt_sim3_ratio'])}"]
    al, ae = rep["alignment"], rep["alignment_eval"]
    if al:
        c = al["confidence"]
        L += ["", "## Floor-plan alignment", f"- method {al.get('method')}, status **{c.get('status')}**, wall inliers {_fmt(c.get('wall_inlier_frac'))}, margin to runner-up {_fmt(c.get('margin'))}",
              f"- scale {_fmt(al['scale'], 4)} m/unit, yaw {_fmt(np.degrees(al['yaw_rad']), 1)} deg"]
    if ae:
        e = ae["plan_frame_2d_error_no_extra_alignment"]
        g = ae.get("plan_frame_2d_error_global_transform_only")
        fl = ae.get("filled_trajectory_2d")
        L += [f"- vs GT in plan frame (2D, no further alignment): median {_fmt(e['median'])} m, RMSE {_fmt(e['rmse'])} m, p90 {_fmt(e['p90'])} m, coverage {_fmt(e['coverage'])}",
              f"- registration scale / GT scale: {_fmt(ae['registration_scale_over_gt_scale'])}"]
        if g:
            L.append(f"- single global transform only: median {_fmt(g['median'])} m, RMSE {_fmt(g['rmse'])} m")
        if fl:
            L.append(f"- coverage-completed (flagged fills): coverage {_fmt(fl['coverage'])}, median {_fmt(fl['median'])} m, Hilti score {_fmt(fl['hilti_score'], 1)}/100, by flag {fl.get('median_error_by_flag')}")
    dc = rep.get("discrepancy")
    if dc:
        L += ["", "## As-built vs plan", f"- status: {dc.get('status')}"]
        if dc.get("status") == "ok":
            L += [f"- planned wall boundary length m: {dc.get('planned_boundary_length_m')}, confirmed fraction of seen {_fmt(dc.get('confirmed_fraction_of_seen'))}",
                  f"- observed-not-in-plan clusters: {len(dc.get('observed_not_in_plan_clusters', []))}"]
    vv = rep.get("viewer_validation")
    if vv:
        pb = vv.get("playback", {})
        L += ["", "## Viewer validation", f"- all checks passed: {vv.get('all_passed')} ({vv.get('webgl_renderer')})",
              f"- render {_fmt(pb.get('render_fps_median'), 0)} fps, playback {_fmt(pb.get('video_fps_effective'), 2)} fps, sync max {_fmt(pb.get('pose_sync_ms_max'), 1)} ms, seek {_fmt(vv.get('seek_latency_ms'), 1)} ms"]
    if rep["localization"]:
        L += ["", "## Particle-filter localization (causal replay)", "| experiment | median err m | RMSE m | <1 m | yaw err deg | converged <1 m at s | recovery s |", "|---|---|---|---|---|---|---|"]
        for name, r in rep["localization"].items():
            ev = (r or {}).get("evaluation_vs_gt", {})
            L.append(f"| {name} | {_fmt(ev.get('err_median_m'))} | {_fmt(ev.get('err_rmse_m'))} | {_fmt(ev.get('frac_below_1m'))} | {_fmt(ev.get('yaw_err_median_deg'), 1)} | {_fmt(ev.get('converged_below_1m_at_s'), 1)} | {_fmt(ev.get('recovery_time_after_kidnap_s'), 1)} |")
    ob = rep["objects"]
    if ob:
        L += ["", "## Objects", "```", json.dumps({k: ob[k] for k in ob if k != "per_class"}, indent=1, default=float)[:1500], "```"]
    (rdir / "REPORT.md").write_text("\n".join(L) + "\n")
    return {"processing_time_s": total, "ratio": rep["processing_to_video_ratio"]}
