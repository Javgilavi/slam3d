"""Rigid / similarity transforms, quaternions, trajectory alignment and interpolation.

Pose convention: T_world_cam (a.k.a. twc) maps camera coordinates to world coordinates.
Quaternions are stored (qx, qy, qz, qw) as in TUM trajectories.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def quat_to_R(q_xyzw):
    return Rotation.from_quat(np.asarray(q_xyzw, dtype=np.float64)).as_matrix()


def R_to_quat(R):
    return Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    Ti = np.eye(4)
    R = T[:3, :3]
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


def transform_points(T, p):
    p = np.asarray(p, dtype=np.float64)
    return p @ T[:3, :3].T + T[:3, 3]


def sim3_matrix(s, R, t):
    S = np.eye(4)
    S[:3, :3] = s * R
    S[:3, 3] = t
    return S


def apply_sim3_to_poses(s, R, t, T_poses):
    """Apply world similarity x' = s R x + t to camera-to-world poses (keeps rotations orthonormal)."""
    out = np.array(T_poses, dtype=np.float64, copy=True)
    out[:, :3, :3] = R @ out[:, :3, :3]
    out[:, :3, 3] = (s * (R @ out[:, :3, 3].T)).T + t
    return out


def umeyama(src, dst, with_scale: bool = True):
    """Least-squares similarity (s, R, t) minimising ||dst - (s R src + t)|| (Umeyama 1991).
    Works in any dimension (2D or 3D)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.shape[0] >= src.shape[1]
    n, d = src.shape
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ransac_umeyama(src, dst, thresh: float, with_scale=True, iters=500, min_samples=None, rng=None):
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n, d = src.shape
    m = min_samples or (d if d == 2 else 3)
    rng = rng or np.random.default_rng(0)
    best = None
    best_inl = np.zeros(n, bool)
    if n < m:
        raise ValueError("not enough correspondences")
    for _ in range(iters):
        idx = rng.choice(n, m, replace=False)
        try:
            s, R, t = umeyama(src[idx], dst[idx], with_scale)
        except np.linalg.LinAlgError:
            continue
        res = np.linalg.norm(dst - (s * src @ R.T + t), axis=1)
        inl = res < thresh
        if inl.sum() > best_inl.sum():
            best_inl, best = inl, (s, R, t)
    if best_inl.sum() >= m:
        best = umeyama(src[best_inl], dst[best_inl], with_scale)
    return best, best_inl


def sim2_from_params(s, theta, tx, ty):
    c, si = np.cos(theta), np.sin(theta)
    return np.array([[s * c, -s * si, tx], [s * si, s * c, ty], [0, 0, 1.0]])


def sim2_params(S):
    s = float(np.hypot(S[0, 0], S[1, 0]))
    theta = float(np.arctan2(S[1, 0], S[0, 0]))
    return s, theta, float(S[0, 2]), float(S[1, 2])


def apply_sim2(S, xy):
    xy = np.asarray(xy, dtype=np.float64)
    return xy @ S[:2, :2].T + S[:2, 2]


def interpolate_poses(t_query, t_ref, T_ref, max_gap: float | None = None):
    """Interpolate camera-to-world poses (N,4,4) at t_query (linear translation, slerp rotation).
    Returns poses (M,4,4) and a validity mask (inside range and gap <= max_gap)."""
    t_query = np.asarray(t_query, dtype=np.float64)
    t_ref = np.asarray(t_ref, dtype=np.float64)
    order = np.argsort(t_ref)
    t_ref, T_ref = t_ref[order], np.asarray(T_ref)[order]
    valid = (t_query >= t_ref[0]) & (t_query <= t_ref[-1])
    tq = np.clip(t_query, t_ref[0], t_ref[-1])
    idx = np.clip(np.searchsorted(t_ref, tq, side="right") - 1, 0, len(t_ref) - 2)
    if max_gap is not None:
        valid &= (t_ref[idx + 1] - t_ref[idx]) <= max_gap
    slerp = Slerp(t_ref, Rotation.from_matrix(T_ref[:, :3, :3]))
    R = slerp(tq).as_matrix()
    denom = np.maximum(t_ref[idx + 1] - t_ref[idx], 1e-12)
    a = ((tq - t_ref[idx]) / denom)[:, None]
    p = (1 - a) * T_ref[idx, :3, 3] + a * T_ref[idx + 1, :3, 3]
    out = np.tile(np.eye(4), (len(tq), 1, 1))
    out[:, :3, :3] = R
    out[:, :3, 3] = p
    return out, valid


def associate_timestamps(t_a, t_b, max_diff: float):
    """Greedy one-to-one nearest association of sorted timestamp arrays. Returns index pairs."""
    t_a = np.asarray(t_a, float)
    t_b = np.asarray(t_b, float)
    j = np.clip(np.searchsorted(t_b, t_a), 1, max(len(t_b) - 1, 1))
    cand = np.stack([j - 1, np.minimum(j, len(t_b) - 1)], 1)
    d = np.abs(t_b[cand] - t_a[:, None])
    best = cand[np.arange(len(t_a)), d.argmin(1)]
    ok = np.abs(t_b[best] - t_a) <= max_diff
    ia = np.nonzero(ok)[0]
    ib = best[ok]
    # enforce one-to-one: keep closest for duplicate b indices
    order = np.argsort(np.abs(t_b[ib] - t_a[ia]))
    _, keep = np.unique(ib[order], return_index=True)
    sel = np.sort(order[keep])
    return ia[sel], ib[sel]


def yaw_from_R(R):
    """Heading of the camera forward axis (+z) projected on the world XY plane (world z up)."""
    f = R[:3, 2]
    return float(np.arctan2(f[1], f[0]))


def rotation_between(a, b):
    """Minimal rotation mapping unit vector a onto unit vector b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        return Rotation.from_rotvec(np.pi * axis / np.linalg.norm(axis)).as_matrix()
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
