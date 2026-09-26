"""Minimal binary PLY writer/reader for coloured point clouds."""
from __future__ import annotations

import numpy as np


def write_ply(path, xyz, rgb=None):
    xyz = np.asarray(xyz, np.float32)
    n = len(xyz)
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if rgb is not None:
        dt += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    a = np.empty(n, dt)
    a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None:
        rgb = np.asarray(rgb, np.uint8)
        a["red"], a["green"], a["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + "".join(
        f"property {'float' if t == '<f4' else 'uchar'} {name}\n" for name, t in dt) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(a.tobytes())


def read_ply(path):
    with open(path, "rb") as f:
        n, props = 0, []
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, t, name = line.split()
                props.append((name, "<f4" if t == "float" else "u1"))
            elif line == "end_header":
                break
        a = np.frombuffer(f.read(), dtype=props, count=n)
    xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
    rgb = np.stack([a["red"], a["green"], a["blue"]], 1) if "red" in a.dtype.names else None
    return xyz, rgb
