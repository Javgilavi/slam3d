"""Structural bird's-eye-view evidence from a gravity-aligned sparse/dense reconstruction.

Wall-like evidence = BEV cells whose points span several distinct height bins between
`floor + low*h` and `floor + high*h` (h = camera height above floor in reconstruction units).
Floors and ceilings are excluded by the height band; low clutter (buckets, pallets) and isolated
noise fail the vertical-extent test. Dynamic objects are removed upstream (only multi-view
triangulated landmarks / static masks are used).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BEVEvidence:
    xy: np.ndarray  # (N,2) cell centres in reconstruction units (grav frame)
    weight: np.ndarray  # (N,)
    cell: float  # cell size in reconstruction units
    floor_z: float
    cam_height: float
    meta: dict


def wall_evidence(points_grav: np.ndarray, cam_pos_grav: np.ndarray, cam_height: float,
                  cell_frac: float = 0.08, low: float = 0.35, high: float = 1.6, n_zbins: int = 8,
                  min_zbins: int = 3, max_range_h: float = 12.0, min_points: int = 3) -> BEVEvidence:
    """cam_height: camera height above floor (reconstruction units).
    cell = cell_frac * cam_height (~10 cm for a 1.3 m camera height)."""
    floor_z = float(np.median(cam_pos_grav[:, 2]) - cam_height)
    z0, z1 = floor_z + low * cam_height, floor_z + high * cam_height
    p = points_grav[(points_grav[:, 2] > z0) & (points_grav[:, 2] < z1)]
    # keep points reasonably close to the travelled path
    from scipy.spatial import cKDTree

    d, _ = cKDTree(cam_pos_grav[:, :2]).query(p[:, :2])
    p = p[d < max_range_h * cam_height]
    cell = cell_frac * cam_height
    if len(p) == 0:
        return BEVEvidence(np.zeros((0, 2)), np.zeros(0), cell, floor_z, cam_height, {"n_in_band": 0})
    ij = np.floor(p[:, :2] / cell).astype(np.int64)
    zb = np.clip(((p[:, 2] - z0) / (z1 - z0) * n_zbins).astype(int), 0, n_zbins - 1)
    key = (ij[:, 0] << 32) ^ (ij[:, 1] & 0xFFFFFFFF)
    order = np.argsort(key)
    key, ij, zb = key[order], ij[order], zb[order]
    uniq, start, counts = np.unique(key, return_index=True, return_counts=True)
    xy, w = [], []
    for s, c in zip(start, counts):
        if c < min_points:
            continue
        nz = len(np.unique(zb[s:s + c]))
        if nz >= min_zbins:
            xy.append((ij[s] + 0.5) * cell)
            w.append(np.log1p(c) * nz / n_zbins)
    xy = np.asarray(xy, float).reshape(-1, 2)
    w = np.asarray(w, float)
    return BEVEvidence(xy, w, cell, floor_z, cam_height,
                       {"n_in_band": int(len(p)), "n_cells": int(len(uniq)), "n_wall_cells": int(len(xy)),
                        "band": [low, high], "min_zbins": min_zbins, "cell_frac": cell_frac})
