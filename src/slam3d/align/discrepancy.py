"""As-built vs planned comparison with explicit observation coverage.

Observed and planned geometry are kept separate; the comparison is only reported when the plan alignment
is confident (or manual). "Not observed" never implies "missing":

planned structure cells
  confirmed                 observed wall evidence within `match_m`
  unobserved_in_view        visible (line of sight through planned free space) from >= `min_views` path
                            positions within `max_range_m`, yet no evidence  -> candidate unbuilt / deviation
                            (or simply not reconstructed: low texture, glass, occlusion by unplanned objects)
  not_covered               not visible from the recorded path -> unknown

observed structure not in plan
  clusters of strong wall evidence farther than `extra_m` from any planned structure -> candidate
  temporary partitions, unplanned walls, large equipment, or alignment residuals.

Visibility uses the PLANNED structure as occluder (conservative where planned walls are not yet built).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from slam3d.floorplan.prepare import FloorPlan


def visibility_counts(plan: FloorPlan, cams_xy: np.ndarray, max_range_m=8.0, n_rays=720, step_m=None):
    """Count, per planned-structure cell, how many camera positions see it first along a ray."""
    st = plan.structure
    H, W = st.shape
    step = step_m or plan.res * 0.8
    r = np.arange(step, max_range_m, step)
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
    counts = np.zeros((H, W), np.int32)
    for c in cams_xy:
        pts = c[None, None, :] + dirs[:, None, :] * r[None, :, None]  # (rays, steps, 2)
        cr = plan.world_to_px(pts)
        col = np.round(cr[..., 0]).astype(int)
        row = np.round(cr[..., 1]).astype(int)
        inside = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        hit = np.zeros(inside.shape, bool)
        hit[inside] = st[row[inside], col[inside]]
        any_hit = hit.any(1)
        first = hit.argmax(1)
        rr = row[np.arange(n_rays), first][any_hit]
        cc = col[np.arange(n_rays), first][any_hit]
        seen = np.zeros((H, W), bool)
        seen[rr, cc] = True
        counts += seen
    return counts


def analyse(plan: FloorPlan, wall_xy_plan: np.ndarray, wall_w: np.ndarray, cams_xy: np.ndarray, alignment_status: str,
            out_dir: Path, match_m=0.3, extra_m=0.6, min_views=3, max_range_m=8.0, min_cluster_cells=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = {"alignment_status": alignment_status, "params": {"match_m": match_m, "extra_m": extra_m, "min_views": min_views,
                                                             "max_range_m": max_range_m}}
    if alignment_status not in ("confident", "manual"):
        rep["status"] = "suppressed: floor-plan alignment is not confident"
        (out_dir / "discrepancies.json").write_text(json.dumps(rep, indent=2))
        return rep
    H, W = plan.shape
    ev = np.zeros((H, W), bool)
    cr = np.round(plan.world_to_px(wall_xy_plan)).astype(int)
    ok = (cr[:, 0] >= 0) & (cr[:, 0] < W) & (cr[:, 1] >= 0) & (cr[:, 1] < H)
    ev[cr[ok, 1], cr[ok, 0]] = True
    ev_dist = ndimage.distance_transform_edt(~ev) * plan.res
    vis = visibility_counts(plan, cams_xy, max_range_m=max_range_m)
    st = plan.structure
    # only the planned-structure boundary faces the camera; interior cells of thick walls are never "seen"
    boundary = st & ~ndimage.binary_erosion(st)
    seen = boundary & (vis >= min_views)
    confirmed = seen & (ev_dist <= match_m)
    unobs = seen & (ev_dist > match_m)
    not_cov = boundary & (vis < min_views)
    cell_len = plan.res
    rep.update(planned_boundary_length_m={"confirmed": float(confirmed.sum() * cell_len), "unobserved_in_view": float(unobs.sum() * cell_len),
                                          "not_covered": float(not_cov.sum() * cell_len)})
    rep["confirmed_fraction_of_seen"] = float(confirmed.sum() / max(seen.sum(), 1))
    # observed-not-planned clusters
    d_plan = plan.dist_at(wall_xy_plan, outside=10.0)
    strong = (d_plan > extra_m) & (wall_w >= np.median(wall_w))
    extra = np.zeros((H, W), np.uint8)
    extra[cr[ok & strong, 1], cr[ok & strong, 0]] = 1
    extra = cv2.dilate(extra, np.ones((5, 5), np.uint8))
    lab, n = ndimage.label(extra)
    clusters = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        m = lab[sl] == i + 1
        ncells = int((ev[sl] & m).sum())
        if ncells < min_cluster_cells:
            continue
        rr, cc = np.nonzero(m)
        xy = plan.px_to_world(np.stack([cc + sl[1].start, rr + sl[0].start], 1).astype(float))
        ext = xy.max(0) - xy.min(0)
        clusters.append({"centroid_xy": xy.mean(0).round(2).tolist(), "extent_m": ext.round(2).tolist(), "n_evidence_cells": ncells,
                         "min_dist_to_plan_m": float(plan.dist_at(xy.mean(0)[None], 10.0)[0])})
    rep["observed_not_in_plan_clusters"] = sorted(clusters, key=lambda c: -c["n_evidence_cells"])
    rep["status"] = "ok"
    # RGBA layer on the plan raster: green confirmed, orange unobserved-in-view, grey not covered, magenta extra
    img = np.zeros((H, W, 4), np.uint8)
    img[not_cov] = (150, 150, 150, 160)
    img[unobs] = (0, 140, 255, 255)
    img[confirmed] = (60, 180, 60, 255)
    img[(lab > 0) & ev] = (255, 0, 200, 255)
    img = cv2.dilate(img, np.ones((3, 3), np.uint8))
    cv2.imwrite(str(out_dir / "discrepancy_layer.png"), img)
    (out_dir / "discrepancies.json").write_text(json.dumps(rep, indent=2))
    return rep
