"""Bounded CPU appearance fitting of fixed anisotropic Gaussian geometry.

This is offline, uses final SLAM poses, and is not a full densifying 3DGS trainer.
Training uses exact rotated pinhole crops of panoramas, a shared camera centre,
projected 3D covariance, and depth-ordered front-to-back alpha compositing.
"""
from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from slam3d.geometry.sphere import equirect_to_perspective, perspective_intrinsics, perspective_view_rotation
from slam3d.io.ply import read_ply


def projected_covariance(p, cov, focal):
    """Perspective Jacobian J Sigma J^T, camera axes x right/y down/z forward."""
    z = np.maximum(p[:, 2], 1e-5)
    J = np.zeros((len(p), 2, 3))
    J[:, 0, 0] = focal / z
    J[:, 1, 1] = focal / z
    J[:, 0, 2] = -focal * p[:, 0] / z**2
    J[:, 1, 2] = -focal * p[:, 1] / z**2
    return J @ cov @ J.transpose(0, 2, 1) + np.eye(2) * 0.3


def raster_table(xyz, covariance, T, size=96, layers=24):
    """Fixed geometry sparse pixel support table, nearest layers first.

    3-sigma footprints; truncated to `layers` contributors/pixel and a maximum
    footprint radius of 16 pixels. These approximations are reported explicitly.
    """
    R, c = T[:3, :3], T[:3, 3]
    p = (xyz - c) @ R
    f = size / 2
    uv = f * p[:, :2] / np.maximum(p[:, 2:], 1e-5) + (size - 1) / 2
    cov = R.T @ covariance @ R
    C = projected_covariance(p, cov, f)
    rad = np.minimum(np.ceil(3 * np.sqrt(np.linalg.eigvalsh(C)[:, 1])), 16).astype(int)
    visible = (p[:, 2] > 0.15) & (uv[:, 0] > -rad) & (uv[:, 0] < size + rad) & (uv[:, 1] > -rad) & (uv[:, 1] < size + rad)
    pixels, ids, weights = [], [], []
    for i in np.flatnonzero(visible):
        x0, y0 = np.maximum(np.floor(uv[i] - rad[i]), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(uv[i] + rad[i]) + 1, size).astype(int)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.stack([xx.ravel(), yy.ravel()], 1) - uv[i]
        q = np.einsum('ni,ij,nj->n', d, np.linalg.inv(C[i]), d)
        keep = q < 9
        pixels.append((yy.ravel() * size + xx.ravel())[keep])
        ids.append(np.full(keep.sum(), i, np.int32))
        weights.append(np.exp(-0.5 * q[keep]))
    table = np.zeros((size * size, layers), np.int64)
    weight = np.zeros_like(table, np.float32)
    if not pixels:
        return table, weight, 0
    pix, ids, weights = np.concatenate(pixels), np.concatenate(ids), np.concatenate(weights)
    order = np.lexsort((p[ids, 2], pix))
    pix, ids, weights = pix[order], ids[order], weights[order]
    starts = np.maximum.accumulate(np.where(np.r_[True, np.diff(pix) != 0], np.arange(len(pix)), 0))
    rank = np.arange(len(pix)) - starts
    keep = rank < layers
    table[pix[keep], rank[keep]] = ids[keep]
    weight[pix[keep], rank[keep]] = weights[keep]
    return table, weight, int((~keep).sum())


def composite(rgb, opacity, ids, weights):
    import torch
    alpha = (opacity[ids] * weights).clamp(0, 0.995)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], 1), 1)
    return ((alpha * trans)[..., None] * rgb[ids]).sum(1), (alpha * trans).sum(1)


def run_gaussians(rdir: Path, max_points=45000, views=12, steps=120, size=96, seed=7):
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rdir = Path(rdir).resolve()
    out = rdir / 'gaussians'
    out.mkdir(exist_ok=True)
    started = time.perf_counter()
    params = dict(max_points=max_points, views=views, steps=steps, size=size, seed=seed)
    if max_points < 100 or views < 3 or steps < 1 or size < 16:
        raise ValueError('need >=100 points, >=3 views, >=1 step, size>=16')
    (out / 'status.json').write_text(json.dumps({'state': 'running', 'parameters': params}))
    geo = json.loads((rdir / 'geometry/geometry.json').read_text())
    scale = geo['scale'].get('metres_per_unit') or 1.0
    xyz, rgb = read_ply(rdir / 'dense/dense_grav.ply')
    xyz = xyz * scale
    good = np.isfinite(xyz).all(1)
    xyz, rgb = xyz[good], rgb[good]
    # Coverage preserving voxel selection, then bounded deterministic sampling.
    _, ix = np.unique(np.floor(xyz / 0.09).astype(np.int64), axis=0, return_index=True)
    if len(ix) > max_points:
        ix = rng.choice(ix, max_points, replace=False)
    xyz, rgb = xyz[ix], rgb[ix].astype(np.float32) / 255
    dist, nn = cKDTree(xyz).query(xyz, k=9)
    delta = xyz[nn[:, 1:]] - xyz[:, None]
    local = np.einsum('nki,nkj->nij', delta, delta) / 8
    eigen, basis = np.linalg.eigh(local)
    # Thin normal axis and two tangent axes. Geometry is frozen during fitting.
    sigma = np.sqrt(np.maximum(eigen, 1e-6)) * 0.8
    sigma[:, 0] = np.minimum(sigma[:, 0], np.maximum(dist[:, 1] * 0.25, 0.008))
    sigma = np.clip(sigma, 0.008, 0.22)
    basis[:, :, 0] *= np.linalg.det(basis)[:, None]
    covariance = (basis * sigma[:, None, :]**2) @ basis.transpose(0, 2, 1)
    kf = np.atleast_2d(np.genfromtxt(rdir / 'geometry/keyframes_grav.csv', delimiter=',', skip_header=1))
    frames = np.atleast_2d(np.genfromtxt(rdir / 'ingest/frames.csv', delimiter=',', skip_header=1, dtype=str))
    ft = frames[:, 1].astype(float)
    static = cv2.imread(str(rdir / 'ingest/static_mask.png'), 0)
    objects = json.loads((rdir / 'objects/objects.json').read_text()) if (rdir / 'objects/objects.json').exists() else []
    samples, frame_rows, truncated = [], [], 0
    # Interleaved held-out *camera centres*, no crops from those centres in fitting.
    selection = np.unique(np.linspace(0, len(kf)-1, views).astype(int))
    for n, ki in enumerate(selection):
        Tw = np.eye(4)
        Tw[:3] = kf[ki, 2:14].reshape(3, 4)
        Tw[:3, 3] *= scale
        fi = int(np.argmin(abs(ft - kf[ki, 1])))
        img = cv2.cvtColor(cv2.imread(str(rdir / 'ingest' / frames[fi, 2])), cv2.COLOR_BGR2RGB)
        valid = static.copy() if static is not None else np.full(img.shape[:2], 255, np.uint8)
        # Exclude detector-supported dynamic rectangles, including seam wrapping.
        for ob in objects:
            if ob['status'] != 'dynamic':
                continue
            for obs in ob['observations']:
                if abs(obs['t'] - kf[ki, 1]) < 0.1:
                    x0,y0,x1,y1 = np.asarray(obs['bbox_pano_uv'], int)
                    y0,y1 = max(0,y0), min(valid.shape[0],y1+1)
                    if x1 >= x0:
                        valid[y0:y1, max(0,x0):min(valid.shape[1],x1+1)] = 0
                    else:
                        valid[y0:y1, :x1+1] = 0
                        valid[y0:y1, x0:] = 0
        for direction in range(4):
            R = perspective_view_rotation(direction * np.pi / 2, 0)
            T = Tw.copy(); T[:3, :3] = Tw[:3, :3] @ R
            crop, _ = equirect_to_perspective(img, R, perspective_intrinsics(90, size), size)
            mask, _ = equirect_to_perspective(valid, R, perspective_intrinsics(90, size), size, interp='nearest')
            ids, weights, dropped = raster_table(xyz, covariance, T, size)
            truncated += dropped
            samples.append((torch.from_numpy(ids), torch.from_numpy(weights), torch.from_numpy(crop.reshape(-1,3).astype(np.float32)/255), torch.from_numpy(mask.ravel()>0), n % 4 == 2))
            frame_rows.append({'keyframe_id':int(kf[ki,0]), 'frame_idx':fi, 't':float(kf[ki,1]), 'yaw':direction*np.pi/2, 'held_out':n%4==2, 'T_world_view':T.tolist()})
        print(f'[gaussians] prepared camera {n+1}/{len(selection)}', flush=True)
    initial_rgb = torch.from_numpy(rgb)
    color = torch.nn.Parameter(torch.logit(initial_rgb.clamp(.01,.99)))
    opacity = torch.nn.Parameter(torch.full((len(xyz),), 1.5))
    optim = torch.optim.Adam([color, opacity], lr=0.04)
    train = [s for s in samples if not s[-1]]
    def evaluate():
        values = {'train': [], 'held_out': []}
        with torch.no_grad():
            for ids, w, target, mask, holdout in samples:
                rendered, alpha = composite(color.sigmoid(), opacity.sigmoid(), ids, w)
                mse = ((rendered[mask]-target[mask])**2).mean().item()
                values['held_out' if holdout else 'train'].append({'mse':mse,'coverage':float((alpha[mask]>.5).float().mean())})
        return {k:{'psnr_db':float(-10*np.log10(np.mean([v['mse'] for v in val]))), 'alpha_coverage':float(np.mean([v['coverage'] for v in val])), 'views':len(val)} for k,val in values.items() if val}
    before = evaluate()
    prep_s = time.perf_counter()-started
    history=[]
    for step in range(steps):
        ids,w,target,mask,_ = train[step % len(train)]
        optim.zero_grad()
        pred, alpha = composite(color.sigmoid(),opacity.sigmoid(),ids,w)
        loss = (pred[mask]-target[mask]).abs().mean() + 0.001*(color.sigmoid()-initial_rgb).square().mean()
        loss.backward(); optim.step()
        history.append(float(loss.detach()))
        if step % 20 == 0:
            print(f'[gaussians] fit {step}/{steps} L1={history[-1]:.4f}', flush=True)
    after = evaluate()
    fit_s = time.perf_counter()-started-prep_s
    colors, alpha = color.sigmoid().detach().numpy(), opacity.sigmoid().detach().numpy()
    # Semantic IDs are conservative spatial associations to existing static detections.
    # They are not per-pixel segmentation or new semantic inference.
    labels = np.full(len(xyz), -1, np.int32)
    score = np.full(len(xyz), np.inf)
    world_objects=[]
    for o in objects:
        ow = dict(o)
        ow['position'] = (np.asarray(o['centroid_grav_units'])*scale).tolist()
        world_objects.append(ow)
        if o['status']=='dynamic':
            continue
        extent = np.maximum(np.asarray(o['observed_extent_m']), .12)
        delta = abs(xyz - np.asarray(ow['position'])) / (extent/2+.08)
        d = np.linalg.norm(delta,axis=1)
        hit = (delta<=1).all(1) & (d<score)
        labels[hit], score[hit] = o['id'], d[hit]
    covariance6 = covariance[:, [0,0,0,1,1,2], [0,1,2,1,2,2]]
    # float32 xyz, rgb, opacity, covariance(xx,xy,xz,yy,yz,zz), object ID.
    packed = np.column_stack([xyz, colors, alpha, covariance6, labels]).astype('<f4')
    packed.tofile(out/'scene.bin')
    np.savez_compressed(out/'scene.npz', xyz=xyz, rgb=colors, opacity=alpha, covariance=covariance, object_id=labels)
    # Standard degree-zero 3DGS PLY: log-scales, opacity logits, wxyz quaternion.
    names=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    quat=Rotation.from_matrix(basis).as_quat()[:,[3,0,1,2]]
    rows=np.column_stack([xyz,np.zeros_like(xyz),(colors-.5)/.28209479177387814,opacity.detach().numpy(),np.log(sigma),quat]).astype('<f4')
    with open(out/'scene.ply','wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(xyz))+'\n'+''.join('property float '+n+'\n' for n in names)+'end_header\n').encode())
        f.write(rows.tobytes())
    for split in [False,True]:
        j=next(i for i,s in enumerate(samples) if s[-1]==split)
        ids,w,target,mask,_=samples[j]
        with torch.no_grad(): rendered,_=composite(color.sigmoid(),opacity.sigmoid(),ids,w)
        pair=np.concatenate([target.numpy().reshape(size,size,3),rendered.numpy().reshape(size,size,3)],1)
        cv2.imwrite(str(out/('held_out.jpg' if split else 'train.jpg')), cv2.cvtColor((pair.clip(0,1)*255).astype('uint8'),cv2.COLOR_RGB2BGR))
    total=time.perf_counter()-started
    report={'state':'done','method':'fixed anisotropic geometry, optimized RGB and opacity; degree-zero Gaussian splatting', 'frame':'gravity aligned, +z up', 'units':'estimated metres' if geo['scale'].get('metres_per_unit') else 'arbitrary units', 'metres_per_slam_unit':scale, 'scale_source':geo['scale'], 'parameters':params,'gaussians':len(xyz),'semantic_gaussians':int((labels>=0).sum()),'semantic_method':'static observed-box spatial association; no instance-mask fusion', 'objects':world_objects,'before':before,'after':after,'prepare_s':prep_s,'fit_and_eval_s':fit_s,'total_s':total,'source_video_s':float(ft[-1]-ft[0]),'additional_stage_to_video_ratio':total/(ft[-1]-ft[0]),'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'cuda_available':torch.cuda.is_available(),'loss_history':history,'training_views':frame_rows,'max_layers_per_pixel':24,'truncated_contributors':truncated,'evaluation_condition':'held-out appearance views only; geometry/SLAM used entire recording, not an independent reconstruction test','online':False,'source_run':rdir.name,'source_versions':json.loads((rdir.parents[1]/'third_party/VERSIONS.json').read_text()),'scene_sha256':hashlib.sha256((out/'scene.bin').read_bytes()).hexdigest(),'warnings':['Geometry is fixed, no densification, no view-dependent appearance.','Depth and final SLAM poses are offline and use future frames.','Dynamic appearance may remain in existing dense geometry; moving object boxes are excluded from fitting where observed.','Unseen surfaces remain incomplete. Free-view coordinates are not live camera localization.','Training footprints are capped at 16px radius and 24 contributors per pixel.']}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'status.json').write_text(json.dumps({'state':'done','total_s':total}))
    print(json.dumps({k:report[k] for k in ['gaussians','before','after','total_s','peak_rss_mb']},indent=2))
    return report
