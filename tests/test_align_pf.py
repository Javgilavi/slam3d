"""Code-correctness tests for plan registration and the particle filter on a synthetic plan with a
known transform. These verify implementation contracts only; quality is evaluated on real data."""
import numpy as np
import pytest

from slam3d.align import register
from slam3d.floorplan.prepare import _finish
from slam3d.geometry.transforms import apply_sim2, sim2_from_params, sim2_params
from slam3d.localize.pf import PFConfig, PlanParticleFilter


@pytest.fixture(scope="module")
def plan():
    res = 0.05
    H, W = 500, 700  # 25 m x 35 m
    st = np.zeros((H, W), bool)
    st[40:46, 40:660] = True  # outer walls
    st[454:460, 40:660] = True
    st[40:460, 40:46] = True
    st[40:460, 654:660] = True
    st[40:300, 300:306] = True  # asymmetric interior walls
    st[200:206, 460:660] = True
    for c in (150, 520):  # columns
        st[350:360, c:c + 10] = True
    return _finish(st, res, {"source": "synthetic"}, close_m=1.0)


def _wall_samples(plan, rng, n=1500):
    rc = np.argwhere(plan.structure)
    rc = rc[rng.choice(len(rc), n)]
    return plan.px_to_world(rc[:, ::-1].astype(float))


def test_plan_pixel_world_contract(plan):
    H = plan.shape[0]
    assert np.allclose(plan.px_to_world([0, H - 1]), [0, 0])
    assert np.allclose(plan.world_to_px([1.0, 2.0]), [20, H - 1 - 40])
    assert plan.dist_at(np.array([[5.0, 5.0]]))[0] > 1.0
    assert not plan.exterior[250, 350] and plan.exterior[5, 5]


def test_manual_correspondences_recover_similarity():
    S_true = sim2_from_params(0.8, 1.1, 12.0, -3.0)
    src = np.array([[0, 0], [10, 0], [0, 7], [4, 4.0]])
    alg = register.from_correspondences(src, apply_sim2(S_true, src))
    assert np.allclose(alg.S, S_true, atol=1e-9)


def test_auto_alignment_recovers_known_transform(plan):
    rng = np.random.default_rng(0)
    walls_plan = _wall_samples(plan, rng)
    path_plan = np.column_stack([np.linspace(4, 30, 200), 12 + 5 * np.sin(np.linspace(0, 6, 200))])
    S_true = sim2_from_params(1.6, np.deg2rad(37.0), 5.0, 8.0)  # plan <- recon
    S_inv = np.linalg.inv(S_true)
    walls_rec = apply_sim2(S_inv, walls_plan) + rng.normal(0, 0.03, walls_plan.shape)
    path_rec = apply_sim2(S_inv, path_plan)
    # 20 % clutter points that are not on planned walls
    clutter = apply_sim2(S_inv, rng.uniform([5, 5], [30, 20], (300, 2)))
    wall_xy = np.vstack([walls_rec, clutter])
    w = np.ones(len(wall_xy))
    alg = register.align(plan, wall_xy, w, path_rec, scale_prior=1.5, scale_range=(0.7, 1.4), n_scales=9, log=lambda *a: None)
    s, th, tx, ty = sim2_params(alg.S)
    err = np.linalg.norm(apply_sim2(alg.S, path_rec) - path_plan, axis=1)
    assert np.max(err) < 0.3, (s, np.degrees(th), tx, ty, alg.confidence)
    assert alg.confidence["wall_inlier_frac"] > 0.6


def test_particle_filter_tracks_with_odometry_and_structure(plan):
    rng = np.random.default_rng(1)
    s_true, phi_true = 0.5, np.deg2rad(-60)  # metres per unit, grav->plan yaw
    h_units = 1.3 / s_true
    R = np.array([[np.cos(phi_true), -np.sin(phi_true)], [np.sin(phi_true), np.cos(phi_true)]])
    path_plan = np.column_stack([np.linspace(6, 28, 120), np.full(120, 15.0)])
    walls3d = []
    rc = np.argwhere(plan.structure)
    for r, c in rc[rng.choice(len(rc), 4000)]:
        xy = plan.px_to_world(np.array([c, r], float))
        walls3d.append([*xy, rng.uniform(0.2, 2.8)])
    walls3d = np.array(walls3d)
    pf = PlanParticleFilter(plan, h_units, PFConfig(n_particles=1500, seed=3))
    pf.init_gaussian(path_plan[0, 0] + 1.0, path_plan[0, 1] - 1.0, phi_true + np.deg2rad(15), s_true * 1.15, sxy=1.0, sphi=np.deg2rad(15), s_rel=0.15)
    errs = []
    for k in range(1, len(path_plan)):
        dp_plan = path_plan[k] - path_plan[k - 1]
        dp_grav = R.T @ dp_plan / s_true
        pf.predict(dp_grav, 0.0)
        # local observation: walls within 8 m, relative to camera at height 1.3 m, in grav axes & units
        cam = path_plan[k]
        near = walls3d[np.linalg.norm(walls3d[:, :2] - cam, axis=1) < 8.0]
        rel_xy = (R.T @ (near[:, :2] - cam).T).T / s_true
        rel_z = (near[:, 2] - 1.3) / s_true
        xy, w = pf.local_wall_points(np.column_stack([rel_xy, rel_z]))
        if len(xy) >= 5:
            pf.update(xy, w, beta=1.0)
            if pf.ess() < 0.5 * len(pf.P):
                pf.resample()
        est, _ = pf.estimate(0.0)
        errs.append(np.linalg.norm(est[:2] - cam))
    assert np.median(errs[-40:]) < 0.35, np.median(errs[-40:])
