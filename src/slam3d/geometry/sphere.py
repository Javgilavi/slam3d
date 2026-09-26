"""Equirectangular (spherical) camera model.

Conventions (see docs/COORDINATES.md):
  * Panorama camera frame: x right, y down, z forward (OpenCV-like). The image centre
    column looks along +z, the top row looks along -y.
  * Pixel (u, v) refers to pixel centres at integer coordinates, image size W x H with W == 2H.
      lon = ((u + 0.5) / W - 0.5) * 2*pi      in [-pi, pi)
      lat = ((v + 0.5) / H - 0.5) * pi        in [-pi/2, pi/2], positive = downward
      bearing = (cos(lat) sin(lon), sin(lat), cos(lat) cos(lon))
  * This matches stella_vslam's equirectangular model up to a half-pixel offset
    (stella uses u/W without +0.5), and the Hilti stitching script exactly.
  * Depth for spherical images is RADIAL range ||p||, never perspective z.
"""
from __future__ import annotations

import numpy as np


def pixel_to_lonlat(u, v, width: int, height: int):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    lon = ((u + 0.5) / width - 0.5) * 2.0 * np.pi
    lat = ((v + 0.5) / height - 0.5) * np.pi
    return lon, lat


def lonlat_to_pixel(lon, lat, width: int, height: int):
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    u = (lon / (2.0 * np.pi) + 0.5) * width - 0.5
    v = (lat / np.pi + 0.5) * height - 0.5
    return u, v


def lonlat_to_bearing(lon, lat):
    cl = np.cos(lat)
    return np.stack([cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)], axis=-1)


def bearing_to_lonlat(b):
    b = np.asarray(b, dtype=np.float64)
    n = np.linalg.norm(b, axis=-1, keepdims=True)
    b = b / np.maximum(n, 1e-12)
    lon = np.arctan2(b[..., 0], b[..., 2])
    lat = np.arcsin(np.clip(b[..., 1], -1.0, 1.0))
    return lon, lat


def pixel_to_bearing(u, v, width: int, height: int):
    return lonlat_to_bearing(*pixel_to_lonlat(u, v, width, height))


def bearing_to_pixel(b, width: int, height: int):
    """Project 3D directions/points (any norm) to equirect pixel coordinates.

    The returned u lies in [-0.5, W - 0.5); callers sampling images must wrap u modulo W
    across the seam (see wrap_u)."""
    return lonlat_to_pixel(*bearing_to_lonlat(b), width, height)


def wrap_u(u, width: int):
    """Wrap horizontal pixel coordinate across the 360-degree seam."""
    return np.mod(np.asarray(u, dtype=np.float64) + 0.5, width) - 0.5


def bearing_grid(width: int, height: int, dtype=np.float32):
    """(H, W, 3) unit bearings for every pixel centre."""
    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    return pixel_to_bearing(uu, vv, width, height).astype(dtype)


def radial_to_points(radial_depth, width: int | None = None, height: int | None = None):
    """Back-project a radial range image (H, W) to (H, W, 3) points in the panorama frame."""
    h, w = radial_depth.shape[:2]
    grid = bearing_grid(w, h, dtype=np.float64)
    return grid * radial_depth[..., None]


def points_to_radial(points):
    return np.linalg.norm(points, axis=-1)


def radial_to_zdepth(radial, bearings):
    """Perspective z-depth along +z for a radial range and unit bearing (only meaningful for z>0)."""
    return radial * bearings[..., 2]


def angular_distance(b1, b2):
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.arccos(np.clip((b1 * b2).sum(-1), -1.0, 1.0))


def sample_equirect(img, u, v, interp: str = "bilinear"):
    """Sample an equirect image at float pixel coords with horizontal wrap-around at the seam
    and clamping at the poles."""
    import cv2

    h, w = img.shape[:2]
    pad = 2
    padded = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_WRAP)
    mapx = (wrap_u(u, w) + pad).astype(np.float32)
    mapy = np.clip(np.asarray(v, dtype=np.float64), 0, h - 1).astype(np.float32)
    flag = cv2.INTER_LINEAR if interp == "bilinear" else cv2.INTER_NEAREST
    return cv2.remap(padded, mapx, mapy, flag, borderMode=cv2.BORDER_REPLICATE)


def perspective_view_rotation(yaw: float, pitch: float, roll: float = 0.0):
    """Rotation R_pano_view mapping perspective-view camera coordinates into the panorama frame.

    yaw rotates about panorama -y (up) so that positive yaw turns right (towards +x);
    pitch positive looks up. Views share the panorama camera centre (pure rotation)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    r_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    r_pitch = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    r_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return r_yaw @ r_pitch @ r_roll


def perspective_intrinsics(fov_deg: float, size: int):
    f = 0.5 * size / np.tan(np.deg2rad(fov_deg) / 2.0)
    c = (size - 1) / 2.0
    return np.array([[f, 0, c], [0, f, c], [0, 0, 1.0]])


def equirect_to_perspective(img, R_pano_view, K, size: int, interp: str = "bilinear"):
    """Render a size x size pinhole view that shares the panorama centre.

    Returns the image and the (size, size, 2) float map of panorama pixel coordinates used,
    so detections in the view can be mapped back exactly."""
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
    rays = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones_like(xs)], -1)
    rays_pano = rays @ R_pano_view.T
    u, v = bearing_to_pixel(rays_pano, w, h)
    view = sample_equirect(img, u, v, interp)
    return view, np.stack([u, v], -1)
