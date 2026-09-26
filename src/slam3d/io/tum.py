"""TUM trajectory I/O: `timestamp tx ty tz qx qy qz qw`, pose = T_world_cam."""
from __future__ import annotations

import numpy as np

from slam3d.geometry.transforms import R_to_quat, quat_to_R


def read_tum(path):
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            v = s.replace(",", " ").split()
            rows.append([float(x) for x in v[:8]])
    a = np.asarray(rows, dtype=np.float64).reshape(-1, 8)
    ok = np.isfinite(a).all(1)
    a = a[ok]
    T = np.tile(np.eye(4), (len(a), 1, 1))
    if len(a):
        T[:, :3, :3] = quat_to_R(a[:, 4:8])
        T[:, :3, 3] = a[:, 1:4]
    return a[:, 0], T


def write_tum(path, t, T, header: str | None = None):
    t = np.asarray(t)
    q = R_to_quat(np.asarray(T)[:, :3, :3]) if len(t) else np.zeros((0, 4))
    with open(path, "w") as f:
        f.write(f"# {header}\n" if header else "# timestamp tx ty tz qx qy qz qw\n")
        for i in range(len(t)):
            p = T[i][:3, 3]
            f.write(f"{t[i]:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[i,0]:.9f} {q[i,1]:.9f} {q[i,2]:.9f} {q[i,3]:.9f}\n")
