"""Static (camera-rigid) region masks for 360 walkthroughs.

Rigs, helmets, poles, hands holding a monopod and LiDAR domes move WITH the camera, so their pixels
have low temporal variance across a long walkthrough while the scene varies. Features on them
break SLAM (they look like points at infinity-like constant bearing) and pollute object maps.

Output convention: uint8 PNG, 255 = usable, 0 = masked (stella_vslam ignores keypoints on 0).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def static_mask_from_temporal_std(frame_paths: list[str], out_size: tuple[int, int], n_samples: int = 150,
                                  work_width: int = 480, std_rel_thresh: float = 0.4, min_row_frac: float = 0.45,
                                  dilate_px: int = 4, bottom_band_frac: float = 0.04, min_area_frac: float = 1e-3):
    """Returns (mask uint8 at out_size (W,H), info dict, std image).

    Threshold is relative (std < std_rel_thresh * median std over the image) so exposure changes,
    low light and aggressive motion do not break it. Only rows below `min_row_frac` of the height
    are eligible (the zenith row is low-variance simply because it is a single stretched point, and
    upper regions are never rigidly attached in hand-held/helmet captures)."""
    sel = np.linspace(0, len(frame_paths) - 1, min(n_samples, len(frame_paths))).astype(int)
    wh = (work_width, work_width // 2)
    stack = np.stack([cv2.resize(cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE), wh, interpolation=cv2.INTER_AREA)
                      .astype(np.float32) for i in sel])
    std = stack.std(0)
    H, W = std.shape
    std_thresh = float(std_rel_thresh * np.median(std))
    cand = (std < std_thresh)
    cand[: int(min_row_frac * H)] = False
    # remove speckles; wrap-aware closing across the seam
    pad = 8
    c = cv2.copyMakeBorder(cand.astype(np.uint8), 0, 0, pad, pad, cv2.BORDER_WRAP)
    c = cv2.morphologyEx(c, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    if dilate_px:
        c = cv2.dilate(c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)))
    c = c[:, pad:-pad]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(c)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_frac * H * W
    masked = keep[lab]
    if bottom_band_frac > 0:
        masked[int((1 - bottom_band_frac) * H):] = True
    mask = np.where(masked, 0, 255).astype(np.uint8)
    mask = cv2.resize(mask, out_size, interpolation=cv2.INTER_NEAREST)
    info = {"method": "temporal_std", "n_samples": int(len(sel)), "std_thresh": std_thresh,
            "std_rel_thresh": std_rel_thresh, "bottom_band_frac": bottom_band_frac,
            "min_row_frac": min_row_frac, "dilate_px_at_work_res": dilate_px, "work_width": work_width,
            "masked_fraction": float((mask == 0).mean()), "convention": "255 usable, 0 masked"}
    return mask, info, std


def build_static_mask(ingest_dir: str | Path, **kw):
    ingest_dir = Path(ingest_dir)
    rows = np.genfromtxt(ingest_dir / "frames.csv", delimiter=",", skip_header=1, dtype=str)
    paths = [str(ingest_dir / r[2]) for r in rows]
    meta = json.loads((ingest_dir / "stitch.json").read_text()) if (ingest_dir / "stitch.json").exists() else {}
    size = (int(meta.get("width", 1920)), int(meta.get("height", 960)))
    mask, info, std = static_mask_from_temporal_std(paths, size, **kw)
    cv2.imwrite(str(ingest_dir / "static_mask.png"), mask)
    (ingest_dir / "static_mask.json").write_text(json.dumps(info, indent=2))
    return mask, info, std
