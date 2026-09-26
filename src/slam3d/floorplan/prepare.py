"""Floor-plan ingestion -> structural map + distance field.

Plan frame (all outputs): metres, x to the right, y up, z up out of the drawing.
The CENTRE of the bottom-left pixel of the source raster is (0, 0) (Hilti 2026 convention), so
  x = col * res,   y = (H - 1 - row) * res.

Supported inputs:
  * binary / greyscale structure mask (PNG/JPG). Recommended, and user-editable.
  * rendered architectural drawing (PNG/JPG/PDF page): automatic thick-stroke extraction. Text,
    dimension lines, furniture and hatching are suppressed by morphological opening. This is a
    heuristic, so review the saved editable mask.
  * DXF: vector entities rasterised from selected layers (regex).
  * IFC/BIM: not supported directly; export a plan view to DXF/PDF (documented limitation).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class FloorPlan:
    structure: np.ndarray  # (H, W) bool, planned structure at `res`
    res: float  # metres per pixel
    exterior: np.ndarray  # (H, W) bool, outside building footprint / unknown
    dist: np.ndarray  # (H, W) float32 metres to nearest structure (0 on structure)
    columns: list = field(default_factory=list)  # [{x, y, area_m2}]
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.structure.shape

    def world_to_px(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        H = self.structure.shape[0]
        return np.stack([xy[..., 0] / self.res, (H - 1) - xy[..., 1] / self.res], -1)

    def px_to_world(self, cr):
        cr = np.asarray(cr, dtype=np.float64)
        H = self.structure.shape[0]
        return np.stack([cr[..., 0] * self.res, ((H - 1) - cr[..., 1]) * self.res], -1)

    def extent(self):
        H, W = self.structure.shape
        return 0.0, (W - 1) * self.res, 0.0, (H - 1) * self.res

    def lookup(self, grid: np.ndarray, xy, outside: float = np.nan):
        """Bilinear lookup of a (H, W) grid at world xy; `outside` for points off the raster."""
        cr = self.world_to_px(xy)
        c, r = cr[..., 0], cr[..., 1]
        H, W = grid.shape
        inside = (c >= 0) & (c <= W - 1) & (r >= 0) & (r <= H - 1)
        c0 = np.clip(np.floor(c).astype(int), 0, W - 2)
        r0 = np.clip(np.floor(r).astype(int), 0, H - 2)
        fc = np.clip(c - c0, 0, 1)
        fr = np.clip(r - r0, 0, 1)
        g = grid.astype(np.float32)
        v = (g[r0, c0] * (1 - fc) * (1 - fr) + g[r0, c0 + 1] * fc * (1 - fr)
             + g[r0 + 1, c0] * (1 - fc) * fr + g[r0 + 1, c0 + 1] * fc * fr)
        return np.where(inside, v, outside)

    def dist_at(self, xy, outside: float = 5.0):
        return self.lookup(self.dist, xy, outside)

    # ------------------------------------------------------------------ io
    def save(self, out_dir: str | Path):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "structure.png"), np.where(self.structure, 0, 255).astype(np.uint8))
        cv2.imwrite(str(out / "exterior.png"), (self.exterior * 255).astype(np.uint8))
        np.save(out / "dist_m.npy", self.dist.astype(np.float32))
        meta = dict(self.meta, res_m_per_px=self.res, width_px=int(self.shape[1]), height_px=int(self.shape[0]),
                    frame="plan: metres, x right, y up; centre of bottom-left pixel = (0,0)",
                    columns=self.columns)
        (out / "plan.json").write_text(json.dumps(meta, indent=2))
        cv2.imwrite(str(out / "preview.png"), self.preview())

    @staticmethod
    def load(out_dir: str | Path) -> "FloorPlan":
        out = Path(out_dir)
        meta = json.loads((out / "plan.json").read_text())
        st = cv2.imread(str(out / "structure.png"), cv2.IMREAD_GRAYSCALE) < 128
        ext = cv2.imread(str(out / "exterior.png"), cv2.IMREAD_GRAYSCALE) > 127
        dist = np.load(out / "dist_m.npy")
        return FloorPlan(st, float(meta["res_m_per_px"]), ext, dist, meta.get("columns", []), meta)

    def preview(self, max_side: int = 2000):
        img = np.full(self.shape + (3,), 255, np.uint8)
        img[self.exterior] = (225, 225, 225)
        img[self.structure] = (0, 0, 0)
        for c in self.columns:
            cr = self.world_to_px([c["x"], c["y"]])
            cv2.circle(img, (int(cr[0]), int(cr[1])), max(2, int(0.4 / self.res)), (0, 0, 255), 2)
        s = min(1.0, max_side / max(self.shape))
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


# ---------------------------------------------------------------------------------------------
def _resample_structure(struct_src: np.ndarray, src_res: float, res: float) -> np.ndarray:
    """Downsample keeping thin walls: a coarse cell is structure if any fine pixel is structure.
    Keeps the bottom-left pixel centre as origin (rows counted from the bottom)."""
    if abs(src_res - res) < 1e-12:
        return struct_src.copy()
    f = res / src_res
    if f < 1:
        raise ValueError("upsampling a plan is not supported; choose res >= source resolution")
    k = int(round(f))
    H, W = struct_src.shape
    Hc, Wc = (H + k - 1) // k, (W + k - 1) // k
    flipped = struct_src[::-1]  # bottom row first
    pad = np.zeros((Hc * k, Wc * k), bool)
    pad[:H, :W] = flipped
    coarse = pad.reshape(Hc, k, Wc, k).any(axis=(1, 3))
    return coarse[::-1]


def _finish(structure: np.ndarray, res: float, meta: dict, close_m: float = 2.0,
            column_max_area_m2: float = 1.2) -> FloorPlan:
    structure = structure.astype(bool)
    # exterior: close facade gaps (windows/doors) then flood-fill free space from the raster border
    k = max(3, int(round(close_m / res)) | 1)
    closed = cv2.morphologyEx(structure.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    free = ~closed
    lab, _ = ndimage.label(free)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    exterior = np.isin(lab, border[border > 0]) & ~structure
    dist = ndimage.distance_transform_edt(~structure).astype(np.float32) * res
    # columns: compact small components
    lab_s, n = ndimage.label(structure)
    cols = []
    if n:
        objs = ndimage.find_objects(lab_s)
        areas = ndimage.sum(np.ones_like(lab_s), lab_s, index=np.arange(1, n + 1)) * res * res
        for i, sl in enumerate(objs):
            h = (sl[0].stop - sl[0].start) * res
            w = (sl[1].stop - sl[1].start) * res
            if areas[i] <= column_max_area_m2 and max(h, w) / max(min(h, w), res) < 3.0 and max(h, w) >= 0.15:
                rr, cc = np.nonzero(lab_s[sl] == i + 1)
                r = rr.mean() + sl[0].start
                c = cc.mean() + sl[1].start
                H = structure.shape[0]
                cols.append({"x": float(c * res), "y": float((H - 1 - r) * res), "area_m2": float(areas[i])})
    meta = dict(meta, n_columns=len(cols), structure_fraction=float(structure.mean()),
                exterior_fraction=float(exterior.mean()), exterior_close_m=close_m)
    return FloorPlan(structure, res, exterior, dist, cols, meta)


def from_mask(path: str, src_res: float, res: float = 0.05, structure_is_dark: bool = True, **kw) -> FloorPlan:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    st = img < 128 if structure_is_dark else img >= 128
    return _finish(_resample_structure(st, src_res, res), res,
                   {"source": str(path), "source_type": "mask", "source_res_m_per_px": src_res}, **kw)


def extract_structure_from_drawing(img_bgr: np.ndarray, src_res: float, min_wall_m: float = 0.12,
                                   dark_thresh: int = 90, min_area_m2: float = 0.05,
                                   fill_colors_bgr: list | None = None, color_tol: int = 40) -> np.ndarray:
    """Heuristic structure extraction from a rendered architectural drawing.

    Keeps dark strokes (and optional solid/hatched fill colours) that survive an opening with a disk
    of diameter `min_wall_m`: text, dimension lines, thin furniture strokes and door arcs vanish;
    walls/columns thicker than min_wall_m remain."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cand = gray < dark_thresh
    for col in fill_colors_bgr or []:
        d = np.abs(img_bgr.astype(int) - np.array(col)[None, None]).max(-1)
        fill = d < color_tol
        # hatch patterns: close small holes before opening
        fill = cv2.morphologyEx(fill.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
        cand |= fill
    k = max(3, int(round(min_wall_m / src_res)) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, se)
    lab, n = ndimage.label(opened)
    if n:
        areas = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * src_res * src_res
        keep = np.zeros(n + 1, bool)
        keep[1:] = areas >= min_area_m2
        opened = keep[lab]
    return opened.astype(bool)


def from_drawing(path: str, src_res: float, res: float = 0.05, save_editable_mask: str | None = None,
                 page: int = 0, dpi: float | None = None, **kw) -> FloorPlan:
    p = str(path)
    if p.lower().endswith(".pdf"):
        import fitz  # pymupdf

        doc = fitz.open(p)
        pg = doc[page]
        pix = pg.get_pixmap(dpi=dpi or 150)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3][..., ::-1].copy()
    else:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
    ex_kw = {k: kw.pop(k) for k in list(kw) if k in ("min_wall_m", "dark_thresh", "min_area_m2", "fill_colors_bgr", "color_tol")}
    st = extract_structure_from_drawing(img, src_res, **ex_kw)
    if save_editable_mask:
        cv2.imwrite(save_editable_mask, np.where(st, 0, 255).astype(np.uint8))
    return _finish(_resample_structure(st, src_res, res), res,
                   {"source": p, "source_type": "drawing_auto", "source_res_m_per_px": src_res, "extraction": ex_kw}, **kw)


def from_dxf(path: str, res: float = 0.05, layer_regex: str = ".*", wall_thickness_m: float = 0.15,
             units_to_m: float | None = None, **kw) -> FloorPlan:
    """Rasterise LINE / LWPOLYLINE / POLYLINE / ARC / CIRCLE / HATCH-boundary entities of matching layers.
    The DXF's own origin becomes the plan origin only up to the raster bounding box (recorded in meta)."""
    import ezdxf

    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    if units_to_m is None:
        ins = doc.header.get("$INSUNITS", 6)
        units_to_m = {1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0}.get(ins, 1.0)
    rx = re.compile(layer_regex)
    polylines = []
    for e in msp:
        if not rx.match(e.dxf.layer):
            continue
        t = e.dxftype()
        try:
            if t == "LINE":
                polylines.append(np.array([e.dxf.start[:2], e.dxf.end[:2]]))
            elif t in ("LWPOLYLINE", "POLYLINE"):
                pts = np.array([p[:2] for p in (e.get_points("xy") if t == "LWPOLYLINE" else e.points())])
                if len(pts) >= 2:
                    if getattr(e, "closed", False) or getattr(e, "is_closed", False):
                        pts = np.vstack([pts, pts[:1]])
                    polylines.append(pts)
            elif t in ("ARC", "CIRCLE"):
                c = np.array(e.dxf.center[:2])
                r = e.dxf.radius
                a0, a1 = (np.deg2rad(e.dxf.start_angle), np.deg2rad(e.dxf.end_angle)) if t == "ARC" else (0, 2 * np.pi)
                if a1 < a0:
                    a1 += 2 * np.pi
                a = np.linspace(a0, a1, 32)
                polylines.append(c + r * np.stack([np.cos(a), np.sin(a)], 1))
        except Exception:
            continue
    if not polylines:
        raise ValueError(f"no drawable entities on layers matching {layer_regex!r}")
    allp = np.vstack(polylines) * units_to_m
    lo, hi = allp.min(0), allp.max(0)
    W = int(np.ceil((hi[0] - lo[0]) / res)) + 1
    H = int(np.ceil((hi[1] - lo[1]) / res)) + 1
    img = np.zeros((H, W), np.uint8)
    th = max(1, int(round(wall_thickness_m / res)))
    for pl in polylines:
        q = (pl * units_to_m - lo) / res
        pts = np.stack([q[:, 0], (H - 1) - q[:, 1]], 1).round().astype(np.int32)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, 255, th)
    meta = {"source": path, "source_type": "dxf", "layer_regex": layer_regex, "units_to_m": units_to_m,
            "dxf_origin_offset_m": lo.tolist(), "note": "plan (0,0) = DXF point dxf_origin_offset_m"}
    return _finish(img > 0, res, meta, **kw)
