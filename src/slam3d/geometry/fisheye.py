"""Fisheye camera models used by calibrated dual-fisheye 360 cameras.

* KannalaBrandt4 ("pinhole-equidistant" in Kalibr): theta_d = theta (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)
* EUCM (Enhanced Unified Camera Model, Khomutenko et al. 2016), Kalibr parameter order
  [alpha, beta, fx, fy, cx, cy].

Camera frame: x right, y down, z optical axis (Kalibr/OpenCV).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KannalaBrandt4:
    fx: float
    fy: float
    cx: float
    cy: float
    k: np.ndarray = field(default_factory=lambda: np.zeros(4))
    width: int = 0
    height: int = 0
    max_theta: float = np.deg2rad(100.0)

    def project(self, p):
        p = np.asarray(p, dtype=np.float64)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        r = np.hypot(x, y)
        theta = np.arctan2(r, z)
        t2 = theta * theta
        k1, k2, k3, k4 = self.k
        theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        scale = np.where(r > 1e-12, theta_d / np.maximum(r, 1e-12), 1.0 / np.maximum(z, 1e-12))
        u = self.fx * x * scale + self.cx
        v = self.fy * y * scale + self.cy
        valid = theta <= self.max_theta
        if self.width:
            valid &= (u >= 0) & (u <= self.width - 1) & (v >= 0) & (v <= self.height - 1)
        return np.stack([u, v], -1), valid

    def unproject(self, uv, iters: int = 20):
        uv = np.asarray(uv, dtype=np.float64)
        mx = (uv[..., 0] - self.cx) / self.fx
        my = (uv[..., 1] - self.cy) / self.fy
        theta_d = np.hypot(mx, my)
        theta = theta_d.copy()
        k1, k2, k3, k4 = self.k
        for _ in range(iters):  # Newton iterations on f(theta) = theta_d(theta) - theta_d
            t2 = theta * theta
            f = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d
            df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + 9 * k4 * t2)))
            theta = theta - f / df
        s = np.where(theta_d > 1e-12, np.sin(theta) / np.maximum(theta_d, 1e-12), 1.0)
        b = np.stack([mx * s, my * s, np.cos(theta)], -1)
        return b / np.linalg.norm(b, axis=-1, keepdims=True)


@dataclass
class EUCM:
    alpha: float
    beta: float
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 0
    height: int = 0
    max_theta: float = np.deg2rad(100.0)

    def project(self, p):
        p = np.asarray(p, dtype=np.float64)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        d = np.sqrt(self.beta * (x * x + y * y) + z * z)
        denom = self.alpha * d + (1.0 - self.alpha) * z
        denom_safe = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        u = self.fx * x / denom_safe + self.cx
        v = self.fy * y / denom_safe + self.cy
        theta = np.arctan2(np.hypot(x, y), z)
        valid = (denom > 1e-9) & (theta <= self.max_theta)
        if self.alpha > 0.5:
            w = (1.0 - self.alpha) / (2.0 * self.alpha - 1.0) if self.alpha != 0.5 else np.inf
            valid &= z > -w * d
        if self.width:
            valid &= (u >= 0) & (u <= self.width - 1) & (v >= 0) & (v <= self.height - 1)
        return np.stack([u, v], -1), valid

    def unproject(self, uv):
        uv = np.asarray(uv, dtype=np.float64)
        mx = (uv[..., 0] - self.cx) / self.fx
        my = (uv[..., 1] - self.cy) / self.fy
        r2 = mx * mx + my * my
        a, b = self.alpha, self.beta
        inside = 1.0 - (2.0 * a - 1.0) * b * r2
        mz = (1.0 - b * a * a * r2) / (a * np.sqrt(np.maximum(inside, 0.0)) + (1.0 - a))
        v = np.stack([mx, my, mz], -1)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def load_kalibr_chain(path: str, model: str = "kb4"):
    """Parse a Kalibr imu-cam chain YAML (supports the commented EUCM block used by the
    Hilti 2026 release). Returns dict cam_name -> {model, T_cam_imu, timeshift_cam_imu}."""
    import yaml

    text = open(path, encoding="utf-8", errors="replace").read()
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("%YAML")]
    if model == "eucm":
        # the EUCM block is commented out after the "EUCM" banner; uncomment it
        idx = next(i for i, ln in enumerate(lines) if "EUCM" in ln)
        block = [ln[2:] if ln.startswith("# ") else ln for ln in lines[idx + 2:]]
        data = yaml.safe_load("\n".join(ln for ln in block if not ln.startswith("#")))
    else:
        data = yaml.safe_load("\n".join(lines))
    cams = {}
    for name in ("cam0", "cam1", "cam2", "cam3"):
        if not data or name not in data:
            continue
        c = data[name]
        w, h = c["resolution"]
        intr = c["intrinsics"]
        if c["camera_model"] == "eucm":
            cam = EUCM(*[float(v) for v in intr], width=w, height=h)
        else:
            cam = KannalaBrandt4(*[float(v) for v in intr], k=np.array(c["distortion_coeffs"], float), width=w, height=h)
        cams[name] = {
            "model": cam,
            "T_cam_imu": np.array(c["T_cam_imu"], dtype=np.float64),
            "timeshift_cam_imu": float(c.get("timeshift_cam_imu", 0.0)),
            "rostopic": c.get("rostopic"),
        }
    return cams
