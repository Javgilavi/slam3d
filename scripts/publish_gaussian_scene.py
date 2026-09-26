"""Publish a trained Gaussian scene as the run's `gaussians/` world (used by `slam3d viewer` -> /world).

Preserves existing work: an existing `gaussians/` directory is RENAMED to `gaussians_<suffix>/` (never deleted
or overwritten); refuses to run if that destination already exists.

The published report keeps every metric of the training report and adds the fields the world viewer reads:
metres_per_slam_unit (same scale the trainer used, so trajectory/objects/splats share one metric frame),
units, total_s, objects (static and dynamic, positions/extents in metres, gravity frame).

usage: python scripts/publish_gaussian_scene.py outputs/<run> <trained_scene_dir> [--previous-suffix cpu_baseline]
"""
from __future__ import annotations

import argparse
import json
import shutil
import hashlib
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("scene_dir", type=Path)
    p.add_argument("--previous-suffix", default="cpu_baseline")
    p.add_argument("--browser-max", type=int, default=400000)
    a = p.parse_args()
    rd, src = a.run_dir.resolve(), a.scene_dir.resolve()
    rep = json.loads((src / "report.json").read_text())
    for f in ("scene.bin", "scene.npz", "scene.ply", "report.json"):
        if not (src / f).exists():
            raise SystemExit(f"missing {src / f}")
    dst = rd / "gaussians"
    prev = rd / f"gaussians_{a.previous_suffix}"
    if dst.exists() and prev.exists():
        raise SystemExit(f"{prev} already exists; choose another --previous-suffix (nothing was moved)")
    final_dst = dst
    dst = rd / f"gaussians_staging_{a.previous_suffix}"
    if dst.exists():
        raise SystemExit(f"{dst} already exists; inspect the previous staging attempt")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("checkpoint_*.pt"))
    # Browser level of detail: world.js depth-sorts every Gaussian in JavaScript on camera motion, so the browser file
    # keeps the most visually important Gaussians (opacity x projected footprint). The complete scene stays in
    # scene_full.bin / scene.ply / scene.npz (used by gsplat renders and external 3DGS viewers).
    import numpy as np

    full = np.fromfile(dst / "scene.bin", dtype="<f4").reshape(-1, 14)
    (dst / "scene.bin").rename(dst / "scene_full.bin")
    if len(full) > a.browser_max:
        cov_diag = full[:, [7, 10, 12]].clip(min=0)
        footprint = np.sqrt(np.sort(cov_diag, axis=1)[:, 1:].prod(axis=1))  # ~ area of the two largest axes
        importance = full[:, 6] * footprint
        keep = np.argsort(-importance)[: a.browser_max]
        browser = full[np.sort(keep)]
    else:
        browser = full
        keep = np.arange(len(full))
    keep = np.sort(keep)
    browser.astype("<f4").tofile(dst / "scene.bin")
    lod = {"browser_gaussians": int(len(browser)), "full_gaussians": int(len(full)),
           "browser_selection": "top-k by opacity x sqrt(product of two largest covariance diagonal terms)",
           "browser_renders_degree0_colour_only": True}
    scene = np.load(dst / "scene.npz")
    if "shN" in scene and int(rep.get("sh_degree", 0)) == 3:
        sh = np.zeros((len(keep), 16, 4), dtype="<f4")
        sh[:, 0, :3] = scene["sh0"][keep]
        sh[:, 1:, :3] = scene["shN"][keep]
        sh.tofile(dst / "scene.sh.bin")
        lod["browser_renders_degree0_colour_only"] = False
        rep["browser_sh_degree"] = 3
    mpu = float(rep["scale_metres_per_unit"])
    objs = json.loads((rd / "objects/objects.json").read_text()) if (rd / "objects/objects.json").exists() else []
    world_objects = []
    for o in objs:
        w = {k: o[k] for k in ("id", "label", "labels", "status", "confidence", "n_observations", "observed_extent_m", "note") if k in o}
        w["position"] = [float(v) * mpu for v in o["centroid_grav_units"]]
        w["observations"] = [{"t": ob["t"], "label": ob["label"], "conf": ob["conf"], "src": ob["src"]} for ob in o["observations"]]
        sem = next((s for s in rep.get("semantic_objects", []) if s["id"] == o["id"]), None)
        w["n_gaussians"] = sem["n_gaussians"] if sem else 0
        world_objects.append(w)
    rep.update(metres_per_slam_unit=mpu, units=f"estimated metres ({rep.get('scale_source', 'plan')} scale), gravity-aligned frame, z up",
               total_s=float(rep["total_seconds"]), objects=world_objects, published_from=str(src),
               checkpoints_kept_in=str(src), level_of_detail=lod, gaussians=lod["browser_gaussians"],
               gaussians_full=lod["full_gaussians"], scene_full_sha256=rep["scene_sha256"],
               scene_sha256=hashlib.sha256((dst / "scene.bin").read_bytes()).hexdigest())
    (dst / "report.json").write_text(json.dumps(rep, indent=2))
    (dst / "status.json").write_text(json.dumps({"state": "done", "total_s": rep["total_s"]}))
    if final_dst.exists():
        final_dst.rename(prev)
        print(f"preserved previous scene: {final_dst} -> {prev}")
    try:
        dst.rename(final_dst)
    except OSError:
        if prev.exists() and not final_dst.exists():
            prev.rename(final_dst)
        raise
    dst = final_dst
    print(f"published {src} -> {dst} ({rep['gaussians']} Gaussians, held-out PSNR {rep['after']['held_out']['psnr_db']:.2f} dB)")


if __name__ == "__main__":
    main()
