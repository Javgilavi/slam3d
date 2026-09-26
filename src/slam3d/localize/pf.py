"""Particle-filter localization on a floor plan from sequential 360 SLAM odometry + local structure.

State per particle: (x, y, phi, log_s)
  x, y   camera position in the plan frame [m]
  phi    yaw rotating the SLAM gravity-aligned frame into the plan frame [rad]
  log_s  log metres per SLAM unit (explicit scale uncertainty; monocular SLAM is scale-free)

Causality: step k uses only the online (not later loop-corrected) SLAM pose at k and landmarks
tracked in frames <= k, expressed relative to the current camera via online relative poses.

Motion model: plan displacement = s R(phi) * delta_p_grav (SLAM units), with noise proportional to
travelled distance; phi and log_s follow slow random walks. Online pose jumps (loop-BA corrections,
relocalisation, tracking loss) are detected and turned into diffusion instead of motion.

Observation model: BEV wall-like points from a short causal window of tracked landmarks (vertical
extent test), mapped by each particle, scored with a robust mixture on the plan distance field:
  p(d) = (1-eps) N(d; 0, sigma) + eps * U(0, d_max)
The per-update log-likelihood is tempered by beta = beta0 * (fraction of landmark IDs not used in
the previous update) and divided by an effective point count cap, to avoid counting overlapping
reconstructed evidence as independent observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from slam3d.floorplan.prepare import FloorPlan


@dataclass
class PFConfig:
    n_particles: int = 4000
    n_particles_global: int = 30000
    # defaults selected by a 16-point sweep on Hilti floor_2_2025-12-03_run_1 (development sequence);
    # the held-out sequence is evaluated with these values unchanged.
    sigma_d: float = 0.45  # m, wall distance noise
    eps_outlier: float = 0.4
    d_max: float = 3.0
    beta0: float = 1.0
    eff_points_cap: int = 12  # evidence counted as at most this many independent points per update
    trans_noise_frac: float = 0.08  # std of displacement noise / distance
    trans_noise_min: float = 0.02  # m per step
    phi_noise_per_m: float = np.deg2rad(1.0)
    phi_noise_per_rad: float = 0.02
    log_s_noise_per_m: float = 0.003
    jump_speed_mps: float = 8.0  # online pose moving faster than this (metres/s via scale prior) => pose jump
    jump_diffuse_m: float = 1.0
    window_s: float = 4.0
    update_every_s: float = 0.5
    min_wall_cells: int = 8
    cell_frac: float = 0.08
    band: tuple = (0.35, 1.6)
    min_zbins: int = 2
    n_zbins: int = 6
    free_space_penalty: float = 4.0
    ess_frac: float = 0.5
    recovery_alpha_slow: float = 0.02
    recovery_alpha_fast: float = 0.3
    recovery_ratio: float = 0.5  # inject only when short-term fit < ratio * long-term fit
    max_random_frac: float = 0.03
    rough_xy: float = 0.05  # m jitter after resampling (prevents sample impoverishment)
    rough_yaw_deg: float = 0.5
    rough_log_s: float = 0.005
    # structural (re-)initialisation from causal local-map registration against the plan
    global_init_after_s: float = 20.0
    reg_window_s: float = 30.0
    reg_min_wall_cells: int = 25
    reg_scale_rel: float = 0.3
    reg_n_scales: int = 7
    reg_top_k: int = 6
    reg_sigma_xy: float = 0.5
    reg_sigma_yaw_deg: float = 3.0
    reg_sigma_log_s: float = 0.05
    recovery_trigger_s: float = 3.0
    recovery_inject_frac: float = 0.3
    recovery_cooldown_s: float = 10.0
    seed: int = 0


@dataclass
class PFStep:
    t: float
    frame_idx: int
    mean: np.ndarray  # x, y, yaw_plan(camera heading), s
    cov_xy: np.ndarray
    ess: float
    n_points: int
    beta: float
    kind: str  # 'predict' | 'update' | 'jump' | 'lost'
    modes: list = field(default_factory=list)
    random_injected: int = 0
    particles: np.ndarray | None = None  # subsample (M, 4) for visualisation


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class PlanParticleFilter:
    def __init__(self, plan: FloorPlan, cam_height_units: float, cfg: PFConfig = PFConfig()):
        self.plan = plan
        self.h = cam_height_units
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.P = None
        self.logw = None
        self.w_slow = None
        self.w_fast = None
        self._free_cells = np.argwhere((~plan.exterior) & (plan.dist > 0.3))
        self._prev_ids: set = set()

    # ---------------------------------------------------------------- init
    def init_gaussian(self, x, y, phi, s, sxy=0.5, sphi=np.deg2rad(10), s_rel=0.1, n=None):
        n = n or self.cfg.n_particles
        self.P = np.column_stack([
            self.rng.normal(x, sxy, n), self.rng.normal(y, sxy, n),
            _wrap(self.rng.normal(phi, sphi, n)), self.rng.normal(np.log(s), s_rel, n)])
        self.logw = np.zeros(n)

    def _sample_uniform(self, n, s_prior, s_rel):
        cells = self._free_cells[self.rng.integers(0, len(self._free_cells), n)]
        xy = self.plan.px_to_world(cells[:, ::-1] + self.rng.uniform(-0.5, 0.5, (n, 2)))
        return np.column_stack([xy, self.rng.uniform(-np.pi, np.pi, n), self.rng.normal(np.log(s_prior), s_rel, n)])

    def init_global(self, s_prior, s_rel=0.2, n=None):
        n = n or self.cfg.n_particles_global
        self.P = self._sample_uniform(n, s_prior, s_rel)
        self.logw = np.zeros(n)
        self.s_prior, self.s_rel = s_prior, s_rel

    def _sample_hypotheses(self, hyps, n):
        """hyps: list of (weight, x, y, phi, s). Gaussian mixture samples (n, 4)."""
        c = self.cfg
        w = np.array([h[0] for h in hyps], float)
        w = w / w.sum()
        k = self.rng.choice(len(hyps), n, p=w)
        H = np.array([[h[1], h[2], h[3], np.log(h[4])] for h in hyps])[k]
        H[:, 0:2] += self.rng.normal(0, c.reg_sigma_xy, (n, 2))
        H[:, 2] = _wrap(H[:, 2] + self.rng.normal(0, np.deg2rad(c.reg_sigma_yaw_deg), n))
        H[:, 3] += self.rng.normal(0, c.reg_sigma_log_s, n)
        return H

    def init_hypotheses(self, hyps, n=None):
        n = n or self.cfg.n_particles
        self.P = self._sample_hypotheses(hyps, n)
        self.logw = np.zeros(n)
        self.w_slow = self.w_fast = None

    def inject_hypotheses(self, hyps, frac):
        """Replace the lowest-weight fraction of particles by samples around structural hypotheses."""
        n = int(frac * len(self.P))
        if n <= 0 or not hyps:
            return 0
        idx = np.argsort(self.logw)[:n]
        self.P[idx] = self._sample_hypotheses(hyps, n)
        self.logw[idx] = np.median(self.logw)
        self.w_fast = self.w_slow
        return n

    # ---------------------------------------------------------------- motion
    def predict(self, dp_grav_xy: np.ndarray, dyaw: float, jump: bool = False):
        c = self.cfg
        n = len(self.P)
        s = np.exp(self.P[:, 3])
        dist_units = float(np.linalg.norm(dp_grav_xy))
        if jump:
            self.P[:, 0:2] += self.rng.normal(0, c.jump_diffuse_m, (n, 2))
            self.P[:, 2] = _wrap(self.P[:, 2] + self.rng.normal(0, np.deg2rad(5), n))
            return
        cp, sp = np.cos(self.P[:, 2]), np.sin(self.P[:, 2])
        dx = s * (cp * dp_grav_xy[0] - sp * dp_grav_xy[1])
        dy = s * (sp * dp_grav_xy[0] + cp * dp_grav_xy[1])
        dm = s * dist_units
        sig = np.maximum(c.trans_noise_frac * dm, c.trans_noise_min)
        self.P[:, 0] += dx + self.rng.normal(0, 1, n) * sig
        self.P[:, 1] += dy + self.rng.normal(0, 1, n) * sig
        self.P[:, 2] = _wrap(self.P[:, 2] + self.rng.normal(0, 1, n) * (c.phi_noise_per_m * dm + c.phi_noise_per_rad * abs(dyaw) + 1e-4))
        self.P[:, 3] += self.rng.normal(0, 1, n) * (c.log_s_noise_per_m * dm + 1e-5)

    # ---------------------------------------------------------------- observation
    def local_wall_points(self, pts_rel_grav: np.ndarray):
        """pts_rel_grav: (N,3) landmark positions relative to the current camera, gravity-aligned axes,
        SLAM units. Returns (M,2) BEV wall-cell centres (SLAM units) and weights."""
        c = self.cfg
        h = self.h
        z0, z1 = -h + c.band[0] * h, -h + c.band[1] * h
        p = pts_rel_grav[(pts_rel_grav[:, 2] > z0) & (pts_rel_grav[:, 2] < z1)]
        if len(p) == 0:
            return np.zeros((0, 2)), np.zeros(0)
        cell = c.cell_frac * h
        ij = np.floor(p[:, :2] / cell).astype(np.int64)
        zb = np.clip(((p[:, 2] - z0) / (z1 - z0) * c.n_zbins).astype(int), 0, c.n_zbins - 1)
        key = ((ij[:, 0] + 100000) << 24) + (ij[:, 1] + 100000)
        uk, nz = np.unique(key, return_counts=True)  # points per cell
        pairs = np.unique(key * c.n_zbins + zb)  # distinct (cell, height-bin) pairs
        _, nzb = np.unique(pairs // c.n_zbins, return_counts=True)  # height bins per cell, same order as uk
        keep = nzb >= c.min_zbins
        cells_ij = np.stack([(uk >> 24) - 100000, (uk & ((1 << 24) - 1)) - 100000], 1)
        xy = (cells_ij[keep] + 0.5) * cell
        w = np.log1p(nz[keep])
        return xy, w

    def update(self, wall_xy_units: np.ndarray, wall_w: np.ndarray, beta: float):
        c = self.cfg
        n = len(self.P)
        s = np.exp(self.P[:, 3])
        cp, sp = np.cos(self.P[:, 2]), np.sin(self.P[:, 2])
        m = len(wall_xy_units)
        # plan points for all particles: (n, m, 2)
        qx, qy = wall_xy_units[:, 0], wall_xy_units[:, 1]
        px = self.P[:, 0:1] + s[:, None] * (cp[:, None] * qx[None] - sp[:, None] * qy[None])
        py = self.P[:, 1:2] + s[:, None] * (sp[:, None] * qx[None] + cp[:, None] * qy[None])
        d = self.plan.dist_at(np.stack([px, py], -1), outside=c.d_max)
        d = np.minimum(d, c.d_max)
        lik = (1 - c.eps_outlier) * np.exp(-0.5 * (d / c.sigma_d) ** 2) / (np.sqrt(2 * np.pi) * c.sigma_d) + c.eps_outlier / c.d_max
        ww = wall_w / wall_w.sum()
        eff = min(m, c.eff_points_cap)
        ll = eff * (np.log(lik) * ww[None]).sum(1)  # weighted mean log-lik scaled to capped count
        # free-space prior on the particle position itself
        dpos = self.plan.dist_at(self.P[:, :2], outside=0.0)
        ext = self.plan.lookup(self.plan.exterior.astype(np.float32), self.P[:, :2], 1.0)
        prior = -c.free_space_penalty * ((dpos < 0.15) | (ext > 0.5))
        inc = beta * ll + prior
        self.logw += inc
        # recovery statistics (augmented MCL) on average per-point likelihood
        from scipy.special import logsumexp

        mll = ll / max(eff, 1)  # per-point mean log-likelihood of each particle
        avg = float(np.exp(logsumexp(mll) - np.log(len(mll))))
        if self.w_slow is None:
            self.w_slow = self.w_fast = avg
        self.w_slow += c.recovery_alpha_slow * (avg - self.w_slow)
        self.w_fast += c.recovery_alpha_fast * (avg - self.w_fast)
        return inc

    # ---------------------------------------------------------------- resampling
    def weights(self):
        lw = self.logw - self.logw.max()
        w = np.exp(lw)
        return w / w.sum()

    def ess(self):
        w = self.weights()
        return float(1.0 / np.sum(w ** 2))

    def resample(self, n_out=None, s_prior=None, s_rel=0.2):
        c = self.cfg
        w = self.weights()
        n_out = n_out or c.n_particles
        n_rand = 0
        if self.w_slow and self.w_fast is not None and s_prior is not None:
            ratio = self.w_fast / max(self.w_slow, 1e-12)
            if ratio < c.recovery_ratio:
                n_rand = int(min(c.max_random_frac, 1.0 - ratio) * n_out)
        u = (self.rng.random() + np.arange(n_out - n_rand)) / (n_out - n_rand)
        idx = np.searchsorted(np.cumsum(w), u)
        idx = np.minimum(idx, len(w) - 1)
        P = self.P[idx].copy()
        m = len(P)
        P[:, 0:2] += self.rng.normal(0, c.rough_xy, (m, 2))
        P[:, 2] = _wrap(P[:, 2] + self.rng.normal(0, np.deg2rad(c.rough_yaw_deg), m))
        P[:, 3] += self.rng.normal(0, c.rough_log_s, m)
        if n_rand:
            P = np.vstack([P, self._sample_uniform(n_rand, s_prior, s_rel)])
        self.P = P
        self.logw = np.zeros(len(P))
        return n_rand

    # ---------------------------------------------------------------- estimates
    def estimate(self, yaw_grav: float):
        w = self.weights()
        mx = np.sum(w * self.P[:, 0])
        my = np.sum(w * self.P[:, 1])
        phi = np.arctan2(np.sum(w * np.sin(self.P[:, 2])), np.sum(w * np.cos(self.P[:, 2])))
        s = float(np.exp(np.sum(w * self.P[:, 3])))
        d = self.P[:, :2] - [mx, my]
        cov = (w[:, None] * d).T @ d
        return np.array([mx, my, _wrap(phi + yaw_grav), s]), cov

    def modes(self, cell_m=2.0, max_modes=5):
        w = self.weights()
        ij = np.floor(self.P[:, :2] / cell_m).astype(int)
        key = ij[:, 0] * 100000 + ij[:, 1]
        uk, inv = np.unique(key, return_inverse=True)
        mass = np.bincount(inv, weights=w)
        order = np.argsort(-mass)[:max_modes]
        out = []
        for k in order:
            sel = inv == k
            ws = w[sel] / w[sel].sum()
            out.append({"x": float(ws @ self.P[sel, 0]), "y": float(ws @ self.P[sel, 1]), "mass": float(mass[k])})
        return out
