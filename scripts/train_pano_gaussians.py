"""360-degree Gaussian splatting of a processed walkthrough (gsplat, CUDA).

Cameras (shared-centre multi-view adapter):
  every selected panorama is cut into `--views` exact pinhole crops that are pure rotations about the
  panorama centre (slam3d.geometry.sphere.equirect_to_perspective). Crop pose:
      T_grav_view = T_grav_pano(stella final, metres) @ [R_pano_view | 0]
  OpenCV camera axes (x right, y down, z forward) == slam3d panorama axes, as gsplat expects.
Poses/scale: stella_vslam globally optimised trajectory (validated vs LiDAR GT at 0.13 m Sim3 RMSE on this
  sequence) scaled to metres with the floor-plan registration scale (1.006x GT). Gravity frame, z up.
Initialisation: PanoVGGT dense cloud + SLAM landmarks (gravity frame), voxel-downsampled.
Masks: the camera-rigid rig mask is mapped into every crop; masked pixels never enter loss or metrics.
Held-out: every `--holdout-every`-th selected panorama with ALL its crops; held-out panoramas are excluded
  from the loss (they still influenced SLAM poses, PanoVGGT geometry and initial point colours, so this
  measures appearance/novel-view interpolation, not independent-sequence reconstruction).
Appearance: degree-3 spherical harmonics. Optional per-panorama colour affine (auto-exposure) is learned for
  training panoramas only; held-out metrics are reported raw and, separately labelled, after a per-image
  colour affine fitted to the held-out target (appearance-compensated).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import DefaultStrategy, rasterization
from gsplat.optimizers import SelectiveAdam
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from slam3d.geometry import sphere
from slam3d.io.ply import read_ply
from slam3d.io.tum import read_tum

C0 = 0.28209479177387814


def ssim_map(x, y):
    """x, y: (H, W, 3) in [0,1] -> per-pixel SSIM (H, W)."""
    x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]
    pool = lambda z: F.avg_pool2d(z, 11, 1, 5)
    u, v = pool(x), pool(y)
    a, b, c = pool(x * x) - u * u, pool(y * y) - v * v, pool(x * y) - u * v
    s = ((2 * u * v + 0.01 ** 2) * (2 * c + 0.03 ** 2)) / ((u * u + v * v + 0.01 ** 2) * (a + b + 0.03 ** 2))
    return s[0].mean(0)


def consistent_dense(rd: Path, D: np.ndarray, n_check: int = 3, tol: float = 0.05, min_agree: int = 2):
    """Multi-view consistency filter for the fused PanoVGGT cloud (gravity frame, SLAM units).

    Each point is projected into its `n_check` nearest keyframes' scaled radial depth maps
    (dense/depth/kf_<id>.npy). agree: |r - r_map| < tol * r_map. free-space violation: r < (1 - 2 tol) r_map
    (a keyframe sees THROUGH the point, i.e. a floater). Keep: agree >= min_agree and no violation."""
    kc = np.genfromtxt(rd / "geometry/keyframes_grav.csv", delimiter=",", skip_header=1)
    kid = kc[:, 0].astype(int)
    KT = np.tile(np.eye(4), (len(kid), 1, 1))
    KT[:, :3, :] = kc[:, 2:14].reshape(-1, 3, 4)
    avail = [j for j, k in enumerate(kid) if (rd / f"dense/depth/kf_{k}.npy").exists()]
    centres = KT[avail, :3, 3]
    _, nn = cKDTree(centres).query(D, k=n_check, workers=8)
    agree = np.zeros(len(D), np.int16)
    viol = np.zeros(len(D), np.int16)
    for jj, j in enumerate(avail):
        dm = np.load(rd / f"dense/depth/kf_{kid[j]}.npy").astype(np.float32)
        h, w = dm.shape
        for col in range(n_check):
            idx = np.nonzero(nn[:, col] == jj)[0]
            if len(idx) == 0:
                continue
            pc = (D[idx] - KT[j, :3, 3]) @ KT[j, :3, :3]
            r = np.linalg.norm(pc, axis=1)
            u, v = sphere.bearing_to_pixel(pc, w, h)
            ui = np.clip(np.round(sphere.wrap_u(u, w)).astype(int), 0, w - 1)
            vi = np.clip(np.round(v).astype(int), 0, h - 1)
            rm = dm[vi, ui]
            okm = rm > 0
            agree[idx] += (okm & (np.abs(r - rm) < tol * rm)).astype(np.int16)
            viol[idx] += (okm & (r < (1 - 2 * tol) * rm)).astype(np.int16)
    keep = (agree >= min_agree) & (viol == 0)
    return keep, {"consistency_tol": tol, "n_check": n_check, "min_agree": min_agree, "dense_kept": int(keep.sum()),
                  "dense_kept_frac": float(keep.mean()), "free_space_violations_frac": float((viol > 0).mean())}


def view_rotations(ring: int, up: bool = True, down: bool = False):
    """Horizontal ring of `ring` views + optional zenith/nadir. Returns list of (name, R_pano_view).
    The nadir view is off by default: on hand-held rigs it mostly shows the (masked) rig and operator."""
    out = [(f"yaw{int(360 * k / ring):03d}", sphere.perspective_view_rotation(2 * np.pi * k / ring, 0.0)) for k in range(ring)]
    if up:
        out.append(("up", sphere.perspective_view_rotation(0.0, -np.pi / 2)))
    if down:
        out.append(("down", sphere.perspective_view_rotation(0.0, np.pi / 2)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--panorama-step", type=int, default=30, help="use every k-th frame (30 = 1 fps)")
    p.add_argument("--views", type=int, default=6, help="horizontal ring views per panorama")
    p.add_argument("--no-up", action="store_true")
    p.add_argument("--down", action="store_true")
    p.add_argument("--max-scale-m", type=float, default=0.3)
    p.add_argument("--means-lr-scale", type=float, default=1.0, help="multiplier on the standard 1.6e-4*extent position LR")
    p.add_argument("--no-densify", action="store_true")
    p.add_argument("--camera-clearance-m", type=float, default=0.3, help="remove Gaussians closer than this to any camera centre")
    p.add_argument("--fov", type=float, default=100.0)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--holdout-every", type=int, default=8)
    p.add_argument("--init", choices=["dense", "landmarks", "dense_consistent"], default="dense_consistent")
    p.add_argument("--consistency-tol", type=float, default=0.05)
    p.add_argument("--init-voxel-m", type=float, default=0.02)
    p.add_argument("--max-init", type=int, default=600000)
    p.add_argument("--max-gaussians", type=int, default=2500000)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--exposure", action="store_true", help="learn per-panorama colour affine for training panoramas")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale-source", choices=["imu", "plan"], default="imu")
    p.add_argument("--visible-adam", action="store_true", help="update only Gaussians visible in the current crop")
    p.add_argument("--color-lr", type=float, default=0.0025)
    p.add_argument("--scale-lr", type=float, default=0.005)
    p.add_argument("--appearance-only", action="store_true")
    p.add_argument("--bounded-rgb", action="store_true", help="bounded degree-zero RGB instead of spherical harmonics")
    p.add_argument("--keyframe-stride", type=int, default=0, help="train on final bundle-adjusted keyframe cameras")
    p.add_argument("--ssim-weight", type=float, default=.2)
    a = p.parse_args()
    if a.bounded_rgb:
        a.sh_degree = 0
    if a.appearance_only:
        a.no_densify = True
    a.output.mkdir(parents=True, exist_ok=a.resume is not None)
    (a.output / "arguments.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, indent=2))
    torch.set_num_threads(6)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    t_start = time.perf_counter()
    rd = a.run_dir

    # ------------------------------------------------------------------ poses, scale, frames
    geo = json.loads((rd / "geometry/geometry.json").read_text())
    mpu = float(geo["scale"]["metres_per_unit"]) if a.scale_source == "imu" else float(json.loads((rd / "align/alignment.json").read_text())["scale"])
    t_traj, T_traj = read_tum(rd / "geometry/trajectory_grav.tum")
    frames = np.genfromtxt(rd / "ingest/frames.csv", delimiter=",", skip_header=1, dtype=str)
    f_t = frames[:, 1].astype(float)
    pose_of = {round(t, 6): T for t, T in zip(t_traj, T_traj)}
    sel = [i for i in range(0, len(frames), a.panorama_step) if round(f_t[i], 6) in pose_of]
    if a.keyframe_stride:
        keyframes = np.atleast_2d(np.loadtxt(rd / "geometry/keyframes_grav.csv", delimiter=",", skiprows=1))
        sel = []
        for row in keyframes[::a.keyframe_stride]:
            i = int(np.argmin(abs(f_t-row[1])))
            T = np.eye(4)
            T[:3] = row[2:14].reshape(3,4)
            pose_of[round(f_t[i],6)] = T
            sel.append(i)
    held_pano = set(sel[a.holdout_every // 2 :: a.holdout_every])
    static = cv2.imread(str(rd / "ingest/static_mask.png"), cv2.IMREAD_GRAYSCALE)
    views = view_rotations(a.views, up=not a.no_up, down=a.down)
    K = sphere.perspective_intrinsics(a.fov, a.size)

    # ------------------------------------------------------------------ crops (CPU, uint8)
    cams = []  # dict(pano, view, t, held, V (w2c 4x4), img uint8 (H,W,3) RGB, mask bool)
    t0 = time.perf_counter()
    for i in sel:
        pano = cv2.imread(str(rd / "ingest" / frames[i][2]))
        T = pose_of[round(f_t[i], 6)].copy()
        T[:3, 3] *= mpu
        for name, Rv in views:
            crop, uv = sphere.equirect_to_perspective(pano, Rv, K, a.size)
            m = sphere.sample_equirect(static, uv[..., 0], uv[..., 1], interp="nearest") > 127
            c2w = np.eye(4)
            c2w[:3, :3] = T[:3, :3] @ Rv
            c2w[:3, 3] = T[:3, 3]
            cams.append(dict(pano=int(i), view=name, t=float(f_t[i]), held=i in held_pano, c2w=c2w,
                             V=np.linalg.inv(c2w), img=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), mask=m))
    train_ids = [k for k, c in enumerate(cams) if not c["held"]]
    held_ids = [k for k, c in enumerate(cams) if c["held"]]
    pano_index = {pno: j for j, pno in enumerate(sorted({c["pano"] for c in cams}))}
    print(f"[data] {len(sel)} panoramas x {len(views)} views = {len(cams)} crops ({len(train_ids)} train / {len(held_ids)} held-out) "
          f"in {time.perf_counter() - t0:.1f}s; metres/unit {mpu:.4f}", flush=True)

    # ------------------------------------------------------------------ initial Gaussians
    geo = json.loads((rd / "geometry/geometry.json").read_text())
    L, Lc = read_ply(rd / "geometry/landmarks_grav.ply")
    D, Dc = read_ply(rd / "dense/dense_grav.ply")
    init_info = {"mode": a.init, "landmarks": int(len(L)), "dense_in": int(len(D))}
    if a.init == "dense_consistent":
        keep_d, stats = consistent_dense(rd, D, tol=a.consistency_tol)
        D, Dc = D[keep_d], Dc[keep_d]
        init_info.update(stats)
    if a.init == "landmarks":
        X, Xc = L * mpu, Lc.astype(np.float64) / 255.0
    else:
        X = np.vstack([L, D]) * mpu
        Xc = np.vstack([Lc, Dc]).astype(np.float64) / 255.0
    init_info["dense_used"] = int(len(D)) if a.init != "landmarks" else 0
    print(f"[init] {init_info}", flush=True)
    # keep points seen from training panoramas' neighbourhood: drop points only near held-out camera centres? no,
    # geometry is shared; appearance of held-out views is never fitted. Voxel downsample:
    key = np.floor(X / a.init_voxel_m).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    X, Xc = X[first], Xc[first]
    if len(X) > a.max_init:
        pick = rng.choice(len(X), a.max_init, replace=False)
        X, Xc = X[pick], Xc[pick]
    cam_centres = np.array([c["c2w"][:3, 3] for c in cams])
    extent = float(np.linalg.norm(cam_centres.max(0) - cam_centres.min(0))) * 0.55 + 1.0
    # remove initial points too close to the camera path (rig, operator, floor right under the lens)
    cam_c = np.unique(np.round(np.array([c["c2w"][:3, 3] for c in cams]), 4), axis=0)
    dcam, _ = cKDTree(cam_c).query(X, workers=8)
    far = dcam >= a.camera_clearance_m
    init_info["removed_near_camera"] = int((~far).sum())
    X, Xc = X[far], Xc[far]
    d3 = cKDTree(X).query(X, k=4, workers=8)[0][:, 1:].mean(1)
    d3 = np.clip(d3, 0.002, 0.2)
    tt = lambda v: torch.tensor(v, device=dev, dtype=torch.float32)
    n_sh = (a.sh_degree + 1) ** 2
    shs = np.zeros((len(X), n_sh, 3))
    shs[:, 0] = (Xc - 0.5) / C0
    if a.bounded_rgb:
        rgb_init = np.clip(Xc, .001, .999)
        shs[:, 0] = np.log(rgb_init / (1-rgb_init))
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(tt(X)),
        "scales": torch.nn.Parameter(tt(np.log(d3)[:, None].repeat(3, 1))),
        "quats": torch.nn.Parameter(tt(np.tile([1.0, 0, 0, 0], (len(X), 1)))),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((len(X),), 0.1, device=dev))),
        "sh0": torch.nn.Parameter(tt(shs[:, :1])),
        "shN": torch.nn.Parameter(tt(shs[:, 1:])),
    })
    lrs = {"means": 1.6e-4 * extent * a.means_lr_scale, "scales": a.scale_lr, "quats": 1e-3, "opacities": 5e-2, "sh0": a.color_lr, "shN": a.color_lr / 20}
    if a.appearance_only:
        for k in ["means", "scales", "quats"]:
            lrs[k] = 0.0
        a.no_densify = True
    def make_optimizer(k):
        if a.visible_adam:
            return SelectiveAdam([{"params": [params[k]], "lr": lrs[k]}], eps=1e-15, betas=(.9,.999))
        return torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15)
    optimizers = {k: make_optimizer(k) for k in lrs}
    if a.bounded_rgb:
        params["shN"] = torch.nn.Parameter(torch.zeros((len(X), 1, 3), device=dev))
        optimizers["shN"] = make_optimizer("shN")
    exposure = torch.nn.Parameter(torch.zeros(len(pano_index), 6, device=dev)) if a.exposure else None  # gain(3) bias(3)
    exp_opt = torch.optim.Adam([exposure], lr=1e-3) if a.exposure else None
    # schedule scales with run length: gsplat only prunes oversized Gaussians when step > reset_every, so at least
    # two reset cycles must fall inside the refinement window
    reset_every = max(500, a.steps // 10)
    strategy = DefaultStrategy(refine_start_iter=min(500, a.steps // 20), refine_stop_iter=int(a.steps * 0.5), refine_every=100,
                               reset_every=reset_every, prune_opa=0.005, grow_grad2d=0.0002, verbose=False)
    from gsplat.strategy.ops import remove as gs_remove

    cam_t = tt(cam_c)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=extent)
    start_step = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        for k in params:
            params[k] = torch.nn.Parameter(ck["params"][k])
        optimizers = {k: make_optimizer(k) for k in lrs}
        for k in optimizers:
            optimizers[k].load_state_dict(ck["optimizers"][k])
        state = ck["strategy_state"]
        start_step = ck["step"]
        if exposure is not None and ck.get("exposure") is not None:
            exposure.data.copy_(ck["exposure"])
        print(f"[resume] from step {start_step}", flush=True)
    Kt = tt(K)[None]
    print(f"[init] {len(X)} Gaussians, extent {extent:.1f} m, SH degree {a.sh_degree}", flush=True)

    def gpu_sample(k):
        c = cams[k]
        img = torch.from_numpy(c["img"]).to(dev, non_blocking=True).float() / 255.0
        return img, torch.from_numpy(c["mask"]).to(dev), tt(c["V"])[None]

    def render(Vw2c, sh_deg):
        colors = params["sh0"][:, 0].sigmoid() if a.bounded_rgb else torch.cat([params["sh0"], params["shN"]], 1)
        out, alpha, info = rasterization(params["means"], F.normalize(params["quats"], dim=-1), params["scales"].exp(),
                                         params["opacities"].sigmoid(), colors, Vw2c, Kt, a.size, a.size, sh_degree=None if a.bounded_rgb else sh_deg,
                                         near_plane=0.05, far_plane=200.0, packed=False)  # default black background
        return out[0], alpha[0, ..., 0], info  # unclamped: clamping would block gradients of over-bright Gaussians

    def affine_fit(pred, gt, m):
        """Per-channel least-squares gain/bias mapping pred->gt on mask (appearance compensation for evaluation)."""
        P = pred[m]
        G = gt[m]
        A = torch.stack([P, torch.ones_like(P)], -1)  # (n,3,2)
        sol = torch.linalg.lstsq(A.transpose(0, 1), G.transpose(0, 1).unsqueeze(-1)).solution  # (3,2,1)
        return (pred * sol[:, 0, 0] + sol[:, 1, 0]).clamp(0, 1)

    @torch.no_grad()
    def evaluate(tag, ids, max_views=None, save=4):
        ids = list(ids)
        if max_views and len(ids) > max_views:
            ids = [ids[j] for j in np.linspace(0, len(ids) - 1, max_views).astype(int)]
        rows, panels = [], []
        for j, k in enumerate(ids):
            img, m, V = gpu_sample(k)
            pred, alpha, _ = render(V, a.sh_degree)
            pred = pred.clamp(0, 1)
            valid = m & torch.isfinite(img).all(-1)
            mse = ((pred - img) ** 2)[valid].mean()
            comp = affine_fit(pred, img, valid)
            mse_c = ((comp - img) ** 2)[valid].mean()
            s = ssim_map(pred, img)[valid].mean()
            rows.append(dict(crop=int(k), pano=cams[k]["pano"], view=cams[k]["view"], psnr_db=float(-10 * torch.log10(mse)),
                             psnr_color_compensated_db=float(-10 * torch.log10(mse_c)), ssim=float(s),
                             coverage=float((alpha > 0.5)[valid].float().mean())))
            if j in set(np.linspace(0, len(ids)-1, min(save, len(ids))).astype(int)):
                pair = torch.cat([img, pred], 1).cpu().numpy()
                im = (pair * 255).astype(np.uint8)
                cv2.putText(im, f"{tag} {cams[k]['view']} pano {cams[k]['pano']}: {rows[-1]['psnr_db']:.2f} dB", (6, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 60, 30), 1)
                panels.append(im)
        if panels:
            cv2.imwrite(str(a.output / f"{tag}_contact.jpg"), cv2.cvtColor(np.concatenate(panels, 0), cv2.COLOR_RGB2BGR))
        agg = lambda key: float(np.mean([r[key] for r in rows]))
        return dict(n_views=len(rows), psnr_db=agg("psnr_db"), psnr_color_compensated_db=agg("psnr_color_compensated_db"),
                    ssim=agg("ssim"), coverage=agg("coverage"), views=rows)

    torch.cuda.reset_peak_memory_stats()
    before = dict(held_out=evaluate("before_held_out", held_ids, max_views=48), train=evaluate("before_train", train_ids, max_views=48))
    print(f"[before] held-out PSNR {before['held_out']['psnr_db']:.2f} dB SSIM {before['held_out']['ssim']:.3f}", flush=True)
    history = []
    t_train = time.perf_counter()
    order = rng.permutation(train_ids)
    for step in range(start_step, a.steps):
        k = int(order[step % len(order)])
        if step % len(order) == len(order) - 1:
            order = rng.permutation(train_ids)
        img, m, V = gpu_sample(k)
        sh_deg = min(a.sh_degree, step // 1000)
        for o in optimizers.values():
            o.zero_grad(set_to_none=True)
        pred, alpha, info = render(V, sh_deg)
        if exposure is not None:
            e = exposure[pano_index[cams[k]["pano"]]]
            pred = pred * (1 + e[:3]) + e[3:]
            exp_opt.zero_grad(set_to_none=True)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        mf = m.float()[..., None]
        l1 = ((pred - img).abs() * mf).sum() / (mf.sum() * 3 + 1e-6)
        sm = ssim_map(pred * mf + img * (1 - mf), img)
        loss = (1-a.ssim_weight) * l1 + a.ssim_weight * (1 - sm[m].mean())
        loss.backward()
        visible = (info["radii"] > 0).all(dim=-1).any(dim=0)
        for o in optimizers.values():
            o.step(visibility=visible) if a.visible_adam else o.step()
        if exp_opt is not None:
            exp_opt.step()
        if len(params["means"]) < a.max_gaussians and not a.no_densify:
            strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        with torch.no_grad():
            params["scales"].clamp_(np.log(0.001), np.log(a.max_scale_m))
            if step % 500 == 0 and step > 0:
                dmin = torch.cat([torch.cdist(chunk, cam_t).min(1).values for chunk in params["means"].split(262144)])
                near = dmin < a.camera_clearance_m
                if near.any():
                    gs_remove(params=params, optimizers=optimizers, state=state, mask=near)
        # exponential decay of the position learning rate (standard 3DGS schedule)
        optimizers["means"].param_groups[0]["lr"] = lrs["means"] * (0.01 ** (step / max(a.steps - 1, 1)))
        if step % 500 == 0:
            row = dict(step=step, loss=float(loss.detach()), l1=float(l1.detach()), gaussians=int(len(params["means"])),
                       seconds=time.perf_counter() - t_train, peak_vram_mb=torch.cuda.max_memory_allocated() / 2 ** 20)
            history.append(row)
            print(row, flush=True)
        if (step + 1) % 5000 == 0 or step + 1 == a.steps:
            torch.save(dict(step=step + 1, params=params.state_dict(), optimizers={kk: v.state_dict() for kk, v in optimizers.items()},
                            strategy_state=state, exposure=None if exposure is None else exposure.detach()),
                       a.output / "checkpoint_latest.pt")
            interim = evaluate(f"step_{step+1}_held_out", held_ids, max_views=24)
            (a.output / f"metrics_{step+1}.json").write_text(json.dumps(interim, indent=2))
            print(f"[validation {step+1}] PSNR {interim['psnr_db']:.2f} SSIM {interim['ssim']:.3f}", flush=True)
    train_s = time.perf_counter() - t_train
    after = dict(held_out=evaluate("after_held_out", held_ids, save=6), train=evaluate("after_train", train_ids, max_views=64))
    print(f"[after] held-out PSNR {after['held_out']['psnr_db']:.2f} dB (colour-compensated {after['held_out']['psnr_color_compensated_db']:.2f}) "
          f"SSIM {after['held_out']['ssim']:.3f}; train PSNR {after['train']['psnr_db']:.2f}", flush=True)

    # ------------------------------------------------------------------ export
    with torch.no_grad():
        xyz = params["means"].detach().cpu().numpy()
        opacity = params["opacities"].sigmoid().detach().cpu().numpy()
        sc = params["scales"].exp().detach().cpu().numpy()
        q = F.normalize(params["quats"], dim=-1).detach().cpu().numpy()
        sh0 = params["sh0"].detach().cpu().numpy()[:, 0]
        shN = params["shN"].detach().cpu().numpy()
        if a.bounded_rgb:
            sh0 = (torch.sigmoid(params["sh0"]).detach().cpu().numpy()[:, 0] - .5) / C0
            shN = np.zeros((len(xyz), 0, 3), dtype=np.float32)
        rgb = np.clip(sh0 * C0 + 0.5, 0, 1)
        R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        cov = (R * sc[:, None, :] ** 2) @ R.transpose(0, 2, 1)
        keep = (opacity > 0.02) & np.isfinite(xyz).all(1)
        # semantic transfer: static objects' observed extents (gravity frame, SLAM units -> metres)
        obj_id = np.full(len(xyz), -1.0)
        objs = json.loads((rd / "objects/objects.json").read_text()) if (rd / "objects/objects.json").exists() else []
        semantic = []
        tree = cKDTree(xyz)
        for o in objs:
            if o["status"] == "dynamic":
                continue
            c = np.array(o["centroid_grav_units"]) * mpu
            half = np.clip(np.array(o["observed_extent_m"]) / 2 + 0.1, 0.15, 1.5)
            idx = tree.query_ball_point(c, float(np.linalg.norm(half)))
            idx = [i for i in idx if np.all(np.abs(xyz[i] - c) <= half) and obj_id[i] < 0]
            obj_id[idx] = o["id"]
            semantic.append(dict(id=o["id"], label=o["label"], status=o["status"], confidence=o["confidence"], n_gaussians=len(idx),
                                 centre_m=c.tolist(), half_extent_m=half.tolist()))
        np.savez_compressed(a.output / "scene.npz", xyz=xyz[keep], rgb=rgb[keep], opacity=opacity[keep], covariance=cov[keep],
                            object_id=obj_id[keep], sh0=sh0[keep], shN=shN[keep], scales=sc[keep], quats=q[keep])
        packed = np.column_stack((xyz, rgb, opacity, cov[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]], obj_id)).astype("<f4")[keep]
        packed.tofile(a.output / "scene.bin")
        names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(3 * (n_sh - 1))] + \
                ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        rest = shN.transpose(0, 2, 1).reshape(len(xyz), -1)
        rows = np.column_stack((xyz, np.zeros_like(xyz), sh0, rest, params["opacities"].detach().cpu().numpy(), np.log(sc), q)).astype("<f4")[keep]
        with (a.output / "scene.ply").open("wb") as f:
            f.write(("ply\nformat binary_little_endian 1.0\nelement vertex " + str(len(rows)) + "\n" +
                     "".join(f"property float {n}\n" for n in names) + "end_header\n").encode())
            f.write(rows.tobytes())

        # novel views: along the path, 0.6 m to the side of the recorded centre, looking around (not recorded poses)
        writer = cv2.VideoWriter(str(a.output / "novel_flythrough.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12, (a.size, a.size))
        novel, t_render = [], time.perf_counter()
        path_ids = sorted({c["pano"] for c in cams})
        nframes = 240
        for j in range(nframes):
            pano = path_ids[int(j / nframes * (len(path_ids) - 1))]
            c2w = next(c["c2w"] for c in cams if c["pano"] == pano and c["view"].startswith("yaw000")).copy()
            yaw = 2 * np.pi * j / 80
            Ry = sphere.perspective_view_rotation(yaw, -0.15)
            pano_c2w = c2w.copy()  # yaw000 view has R_pano_view = I
            c2w[:3, :3] = pano_c2w[:3, :3] @ Ry
            side = pano_c2w[:3, 0] * 0.6 * np.sin(2 * np.pi * j / nframes)
            c2w[:3, 3] = pano_c2w[:3, 3] + side
            out, alpha, _ = render(tt(np.linalg.inv(c2w))[None], a.sh_degree)
            im = (out.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            writer.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            if j in (30, 90, 150, 210):
                cv2.imwrite(str(a.output / f"novel_{j:03d}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            novel.append(dict(j=j, pano=int(pano), side_offset_m=float(np.linalg.norm(side)), yaw_deg=float(np.degrees(yaw) % 360),
                              coverage=float((alpha > 0.5).float().mean())))
        writer.release()
        torch.cuda.synchronize()
        render_fps = nframes / (time.perf_counter() - t_render)

    appearance = "bounded RGB" if a.bounded_rgb else f"degree-{a.sh_degree} spherical harmonics"
    method = f"gsplat 1.5.3 CUDA: {'fixed geometry' if a.appearance_only else 'joint geometry'} and {appearance}; {'no' if a.no_densify else 'adaptive'} densification"
    report = dict(state="done", method=method,
                  poses=f"stella_vslam final trajectory; scale source: {a.scale_source}", scale_source=a.scale_source,
                  scale_metres_per_unit=mpu, frame="gravity-aligned SLAM frame (z up), metres; origin at first keyframe",
                  camera_adapter=dict(fov_deg=a.fov, size=a.size, views=[n for n, _ in views], shared_centre=True),
                  panoramas=len(sel), crops=len(cams), train_crops=len(train_ids), held_out_crops=len(held_ids),
                  held_out_panoramas=sorted(held_pano), init_gaussians=int(len(X)), init=init_info, gaussians=int(keep.sum()),
                  steps=a.steps, sh_degree=a.sh_degree, exposure_compensation_training=bool(a.exposure),
                  before=before, after=after, loss_history=history, train_seconds=train_s,
                  total_seconds=time.perf_counter() - t_start, peak_vram_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                  novel_views=novel, novel_render_fps_gsplat=render_fps, semantic_objects=semantic,
                  scene_sha256=hashlib.sha256((a.output / "scene.bin").read_bytes()).hexdigest(),
                  limitations=["Held-out panoramas influenced SLAM poses, PanoVGGT geometry and initial point colours: appearance interpolation, not an independent sequence.",
                               "Stitched-panorama crops inherit the central-camera approximation (4 cm lens baseline).",
                               "Workers and moving equipment are not masked (only the camera rig); they appear as ghosts.",
                               "Unobserved surfaces are incomplete; renders far from the walked path degrade.",
                               "Semantic labels are spatial transfers of static object extents, not per-Gaussian mask fusion.",
                               "Offline reconstruction; rendering FPS is not reconstruction throughput."])
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ["gaussians", "train_seconds", "total_seconds", "peak_vram_mb", "novel_render_fps_gsplat"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
