"""Dense panoramic geometry with PanoVGGT on short temporal keyframe windows, fused with SLAM poses.

For keyframe k, PanoVGGT receives [k-1, k, k+1] (equirect, 518x1036) and predicts per-frame radial depth /
local points in each camera frame. PanoVGGT's ERP ray convention (x right, y down, z forward, pixel
centres) equals slam3d's panorama convention, so local points are used directly.

PanoVGGT poses are NOT used; geometry is placed with the globally optimised SLAM keyframe pose. The
per-keyframe scale between PanoVGGT depth and SLAM units is estimated robustly from the radial
ranges of SLAM landmarks observed in that keyframe (median ratio with MAD inlier selection); frames
without enough support are skipped and reported. Depth consistency = relative deviation of scaled
PanoVGGT depth from those landmark ranges (inliers and all).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from slam3d.config import REPO


def load_panovggt(device="cuda"):
    import torch
    from omegaconf import OmegaConf

    root = REPO / "third_party/panovggt"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from panovggt.models.panovggt_model import PanoVGGTModel

    cfg = OmegaConf.load(root / "training/config/default.yaml")
    mc = cfg.model
    model = PanoVGGTModel(img_size=cfg.img_size, patch_size=cfg.patch_size, embed_dim=cfg.embed_dim,
                          enable_camera=False, enable_depth=mc.enable_depth, enable_point=mc.enable_point,
                          aggregator=OmegaConf.to_container(mc.aggregator, resolve=True))
    ckpt = torch.load(REPO / "third_party/weights/panovggt/model.pt", map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "model", "state_dict"):
        if key in ckpt:
            ckpt = ckpt[key]
            break
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del ckpt, sd
    # upstream inference keeps fp32 weights and runs under bf16 autocast (pure bf16 mixes dtypes in the
    # positional-embedding path)
    model = model.to(device=device).eval()
    return model, {"missing_keys": len(missing), "unexpected_keys": len(unexpected)}


def run_dense(cfg, rdir: Path, log=print) -> dict:
    import torch

    from slam3d.geometry import sphere
    from slam3d.io.ply import write_ply
    from slam3d.slam import stella

    dc = cfg.get("dense", {})
    out = rdir / "dense"
    out.mkdir(parents=True, exist_ok=True)
    geo = json.loads((rdir / "geometry/geometry.json").read_text())
    meta = json.loads((rdir / "ingest/stitch.json").read_text())
    W, H = int(meta["width"]), int(meta["height"])
    frames = np.genfromtxt(rdir / "ingest/frames.csv", delimiter=",", skip_header=1, dtype=str)
    ft = frames[:, 1].astype(float)
    static = cv2.imread(str(rdir / "ingest/static_mask.png"), cv2.IMREAD_GRAYSCALE)
    kcsv = np.genfromtxt(rdir / "geometry/keyframes_grav.csv", delimiter=",", skip_header=1)
    kid, kt = kcsv[:, 0].astype(int), kcsv[:, 1]
    KT = np.tile(np.eye(4), (len(kid), 1, 1))
    KT[:, :3, :] = kcsv[:, 2:14].reshape(-1, 3, 4)
    lms = stella.load_landmarks(rdir / "slam/landmarks.bin")
    Rg = np.array(geo["R_grav_slam"])
    L = lms["xyz"] @ Rg.T - np.array(geo["origin_offset_in_rotated_slam"])
    lm2i = {int(k): i for i, k in enumerate(lms["id"])}
    kobs = stella.load_kf_obs(rdir / "slam/kf_obs.bin")
    order = np.argsort(kobs["kf"], kind="stable")
    kobs = kobs[order]
    ukf, starts = np.unique(kobs["kf"], return_index=True)
    ends = np.append(starts[1:], len(kobs))
    kf_slice = {int(k): (s, e) for k, s, e in zip(ukf, starts, ends)}
    mpu = geo["scale"].get("metres_per_unit") or 1.0

    t_load = time.time()
    torch.cuda.reset_peak_memory_stats()
    model, load_info = load_panovggt()
    t_load = time.time() - t_load
    h_in, w_in = 518, 1036
    step = int(dc.get("keyframe_step", 1))
    stride = int(dc.get("pixel_stride", 3))
    max_range_m = float(dc.get("max_range_m", 12.0))
    sel = np.arange(0, len(kid), step)
    if dc.get("max_keyframes"):
        sel = sel[: int(dc["max_keyframes"])]

    def load_img(i):
        fi = int(np.argmin(np.abs(ft - kt[i])))
        bgr = cv2.imread(str(rdir / "ingest" / frames[fi][2]))
        rgb = cv2.cvtColor(cv2.resize(bgr, (w_in, h_in), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        return rgb

    vmask = cv2.resize(static, (w_in, h_in), interpolation=cv2.INTER_NEAREST) > 0 if static is not None else np.ones((h_in, w_in), bool)
    vv, uu = np.mgrid[0:h_in:stride, 0:w_in:stride]
    vv, uu = vv.ravel(), uu.ravel()
    keep_px = vmask[vv, uu]
    vv, uu = vv[keep_px], uu[keep_px]
    all_p, all_c, rows = [], [], []
    t_inf = 0.0
    for n, i in enumerate(sel):
        win = [j for j in (i - 1, i, i + 1) if 0 <= j < len(kid)]
        imgs = [load_img(j) for j in win]
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div(255).unsqueeze(0).cuda()
        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x)
        torch.cuda.synchronize()
        t_inf += time.time() - t0
        c = win.index(i)
        lp = pred["local_points"][0, c].float().cpu().numpy()  # (h, w, 3)
        r_pvg = np.linalg.norm(lp, axis=-1)
        # landmark support for scale
        s_, e_ = kf_slice.get(int(kid[i]), (0, 0))
        ob = kobs[s_:e_]
        li = np.array([lm2i.get(int(l), -1) for l in ob["lm"]])
        ok = li >= 0
        Tw = KT[i]
        pc = (L[li[ok]] - Tw[:3, 3]) @ Tw[:3, :3]
        r_slam = np.linalg.norm(pc, axis=1)
        uv = ob["uv"][ok].astype(np.float64)
        su = np.clip((uv[:, 0] / W) * w_in - 0.5, 0, w_in - 1).astype(np.float32)
        sv = np.clip((uv[:, 1] / H) * h_in - 0.5, 0, h_in - 1).astype(np.float32)
        rp = cv2.remap(r_pvg.astype(np.float32), su[None], sv[None], cv2.INTER_LINEAR)[0]
        good = (rp > 1e-3) & (r_slam > 0)
        row = {"kf": int(kid[i]), "t": float(kt[i]), "n_landmarks": int(good.sum())}
        if good.sum() < int(dc.get("min_landmarks", 20)):
            row["status"] = "skipped_low_support"
            rows.append(row)
            continue
        ratio = r_slam[good] / rp[good]
        med = np.median(ratio)
        mad = np.median(np.abs(ratio - med)) * 1.4826
        inl = np.abs(ratio - med) < max(3 * mad, 0.05 * med)
        s_k = float(np.median(ratio[inl]))
        rel = np.abs(s_k * rp[good] - r_slam[good]) / r_slam[good]
        row.update(status="ok", scale_units_per_pvg=s_k, inlier_frac=float(inl.mean()), depth_rel_err_median_inliers=float(np.median(rel[inl])),
                   depth_rel_err_median_all=float(np.median(rel)))
        rows.append(row)
        # scaled radial depth (SLAM units) for mask lifting in the objects stage
        (out / "depth").mkdir(exist_ok=True)
        np.save(out / "depth" / f"kf_{int(kid[i])}.npy", (r_pvg * s_k).astype(np.float16))
        P = lp[vv, uu] * s_k
        rng_m = np.linalg.norm(P, axis=1) * mpu
        m = (rng_m > 0.3) & (rng_m < max_range_m)
        Pw = P[m] @ Tw[:3, :3].T + Tw[:3, 3]
        all_p.append(Pw.astype(np.float32))
        all_c.append(imgs[c][vv[m], uu[m]])
        if n % 20 == 0:
            log(f"[dense] keyframe {n}/{len(sel)} scale {s_k:.4f} depth rel err (inliers) {row['depth_rel_err_median_inliers']:.3f}")
    peak = torch.cuda.max_memory_allocated() / 2**20
    del model
    torch.cuda.empty_cache()
    P = np.concatenate(all_p) if all_p else np.zeros((0, 3), np.float32)
    C = np.concatenate(all_c) if all_c else np.zeros((0, 3), np.uint8)
    vox = float(dc.get("voxel_m", 0.03)) / mpu
    if len(P):
        key = np.floor(P / vox).astype(np.int64)
        _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
        keep = cnt >= int(dc.get("min_voxel_support", 2))
        P, C = P[idx[keep]], C[idx[keep]]
    write_ply(out / "dense_grav.ply", P, C)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    with open(out / "keyframe_scales.csv", "w") as f:
        f.write("kf,t,status,n_landmarks,scale,inlier_frac,depth_rel_err_inliers,depth_rel_err_all\n")
        for r in rows:
            f.write(f"{r['kf']},{r['t']:.6f},{r['status']},{r['n_landmarks']},{r.get('scale_units_per_pvg', '')},{r.get('inlier_frac', '')},"
                    f"{r.get('depth_rel_err_median_inliers', '')},{r.get('depth_rel_err_median_all', '')}\n")
    sc = np.array([r["scale_units_per_pvg"] for r in ok_rows]) if ok_rows else np.array([np.nan])
    rep = {"model": "PanoVGGT model.pt (CVPR 2026, commit 556bb7d)", "load_info": load_info, "load_s": t_load,
           "inference_s": t_inf, "keyframes": int(len(sel)), "keyframes_ok": len(ok_rows), "window": 3, "input": [h_in, w_in],
           "peak_torch_gpu_mb": peak, "points": int(len(P)), "voxel_m": dc.get("voxel_m", 0.03),
           "depth_rel_err_median_inliers": float(np.median([r["depth_rel_err_median_inliers"] for r in ok_rows])) if ok_rows else None,
           "depth_rel_err_median_all": float(np.median([r["depth_rel_err_median_all"] for r in ok_rows])) if ok_rows else None,
           "scale_rel_spread_across_keyframes": float(np.std(sc) / np.mean(sc)) if ok_rows else None,
           "note": "PanoVGGT scale drifts between windows; each keyframe is scaled independently to SLAM landmarks"}
    (out / "dense_report.json").write_text(json.dumps(rep, indent=2, default=float))
    log(f"[dense] {rep['keyframes_ok']}/{rep['keyframes']} keyframes, {rep['points']} points, peak GPU {peak:.0f} MB, "
        f"depth rel err {rep['depth_rel_err_median_inliers']}")
    return rep
