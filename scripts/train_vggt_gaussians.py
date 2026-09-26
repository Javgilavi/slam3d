"""CUDA Gaussian geometry/appearance optimization of a VGGT-SLAM export.

Appearance holdouts never initialize Gaussians or enter the training loss.
SLAM still uses their images: this measures appearance interpolation, not an
independent-sequence reconstruction. Coordinates retain arbitrary VGGT scale.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from gsplat import rasterization, DefaultStrategy


def ssim(x, y):
    x, y = x.permute(2,0,1)[None], y.permute(2,0,1)[None]
    pool = lambda z: F.avg_pool2d(z, 7, 1, 3)
    u, v = pool(x), pool(y)
    a, b, c = pool(x*x)-u*u, pool(y*y)-v*v, pool(x*y)-u*v
    return (((2*u*v+.01**2)*(2*c+.03**2))/((u*u+v*v+.01**2)*(a+b+.03**2))).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('reconstruction', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--size', type=int, default=384)
    p.add_argument('--max-points', type=int, default=160000)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(6)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    source = json.loads((a.reconstruction/'reconstruction.json').read_text())
    frames = source['frames']
    held = [i for i in range(len(frames)) if i % 8 == 4]
    if not held:
        held = [len(frames)//2]
    train = [i for i in range(len(frames)) if i not in held]
    points, colors, samples = [], [], []
    for i, frame in enumerate(frames):
        d = np.load(a.reconstruction/frame['data'])
        rgb, xyz, valid = d['rgb'], d['xyz'], d['valid']
        h,w = rgb.shape[:2]
        T = d['c2w']
        K = d['K'].copy()
        K[:2] *= a.size/w
        target = cv2.resize(rgb, (a.size,a.size), interpolation=cv2.INTER_AREA)/255.
        depth = ((xyz-T[:3,3]) @ T[:3,:3])[...,2]
        depth = cv2.resize(depth, (a.size,a.size), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(valid.astype(np.uint8), (a.size,a.size), interpolation=cv2.INTER_NEAREST)>0
        mask &= np.isfinite(depth) & (depth>0)
        samples.append((torch.tensor(target,device='cuda',dtype=torch.float32), torch.tensor(np.linalg.inv(T),device='cuda',dtype=torch.float32), torch.tensor(K,device='cuda',dtype=torch.float32), torch.tensor(depth,device='cuda',dtype=torch.float32), torch.tensor(mask,device='cuda')))
        if i in train:
            keep = valid & np.isfinite(xyz).all(-1)
            points.append(xyz[keep][::6])
            colors.append(rgb[keep][::6]/255.)
    xyz, rgb = np.concatenate(points), np.concatenate(colors)
    if len(xyz)>a.max_points:
        ids = rng.choice(len(xyz), a.max_points, replace=False)
        xyz, rgb = xyz[ids], rgb[ids]
    distance = cKDTree(xyz).query(xyz,k=4,workers=6)[0][:,1:].mean(1)
    extent = float(np.linalg.norm(np.percentile(xyz,95,axis=0)-np.percentile(xyz,5,axis=0)))
    distance = np.clip(distance,extent*1e-5,extent*.02)
    tensor = lambda v: torch.tensor(v,device='cuda',dtype=torch.float32)
    params = torch.nn.ParameterDict({
        'means':torch.nn.Parameter(tensor(xyz)),
        'scales':torch.nn.Parameter(tensor(np.log(distance[:,None]*np.ones((1,3))))),
        'quats':torch.nn.Parameter(tensor(np.tile([1,0,0,0],(len(xyz),1)))),
        'colors':torch.nn.Parameter(tensor(np.log(np.clip(rgb,.001,.999)/(1-np.clip(rgb,.001,.999))))),
        'opacities':torch.nn.Parameter(torch.full((len(xyz),),-1.,device='cuda')),
    })
    optimizers = {k:torch.optim.Adam([params[k]],lr=lr,eps=1e-15) for k,lr in [('means',extent*1e-4),('scales',.004),('quats',.001),('colors',.025),('opacities',.025)]}
    strategy = DefaultStrategy(refine_start_iter=500,refine_stop_iter=min(a.steps-500,6000),refine_every=200,reset_every=100000,verbose=True)
    strategy.check_sanity(params,optimizers)
    strategy_state = strategy.initialize_state(scene_scale=extent)
    def render(i, view=None):
        target,V,K,depth,mask = samples[i]
        result, alpha, info = rasterization(params['means'],F.normalize(params['quats'],dim=-1),params['scales'].exp(),params['opacities'].sigmoid(),params['colors'].sigmoid(),V[None] if view is None else view[None],K[None],a.size,a.size,render_mode='RGB+ED',packed=False,near_plane=extent*1e-4,far_plane=extent*100)
        return result[0], alpha[0,...,0], info
    @torch.no_grad()
    def evaluate(tag):
        metrics = {}
        panels = []
        for label,indices in [('train',train),('held_out',held)]:
            rows = []
            # Every held-out view and a bounded, evenly spaced training sample.
            ids = indices if label=='held_out' else list(np.array(indices)[np.linspace(0,len(indices)-1,min(8,len(indices))).astype(int)])
            for j,i in enumerate(ids):
                rendered,alpha,_ = render(i)
                target = samples[i][0]
                mse = (rendered[...,:3]-target).square().mean()
                rows.append(dict(frame=int(i),psnr_db=float(-10*torch.log10(mse)),ssim=float(ssim(rendered[...,:3],target)),coverage=float((alpha>.5).float().mean())))
                if j<4:
                    pair = torch.cat((target,rendered[...,:3]),dim=1).clamp(0,1).cpu().numpy()
                    image=(pair*255).astype(np.uint8)
                    cv2.putText(image,f'{tag} {label} {i}: {rows[-1]["psnr_db"]:.2f} dB',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,80,30),1)
                    cv2.imwrite(str(a.output/f'{tag}_{label}_{i:04d}.jpg'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                    if label=='held_out': panels.append(image)
            metrics[label] = dict(psnr_db=float(np.mean([r['psnr_db'] for r in rows])),ssim=float(np.mean([r['ssim'] for r in rows])),coverage=float(np.mean([r['coverage'] for r in rows])),views=rows)
        if panels:
            cv2.imwrite(str(a.output/f'{tag}_held_out_contact.jpg'),cv2.cvtColor(np.concatenate(panels,axis=0),cv2.COLOR_RGB2BGR))
        return metrics
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    before = evaluate('before')
    history = []
    for step in range(a.steps):
        i = int(rng.choice(train))
        target,V,K,depth,mask = samples[i]
        for optimizer in optimizers.values():optimizer.zero_grad(set_to_none=True)
        result, alpha, info = render(i)
        strategy.step_pre_backward(params,optimizers,strategy_state,step,info)
        loss_rgb = .8*(result[...,:3]-target).abs().mean()+.2*(1-ssim(result[...,:3],target))
        valid = mask & (alpha.detach()>.5)
        loss_depth = ((result[...,3][valid]-depth[valid]).abs()/depth[valid].clamp_min(extent*.001)).clamp_max(1).mean() if valid.any() else loss_rgb*0
        loss = loss_rgb+.02*loss_depth+.001*params['opacities'].sigmoid().mean()
        loss.backward()
        for optimizer in optimizers.values():optimizer.step()
        if len(params['means'])<a.max_points*2:
            strategy.step_post_backward(params,optimizers,strategy_state,step,info,packed=False)
        with torch.no_grad():
            params['scales'].clamp_(np.log(extent*1e-5),np.log(extent*.03))
            params['opacities'].clamp_(-8,8)
        optimizers['means'].param_groups[0]['lr'] = extent*1e-4*(.1**(step/max(a.steps-1,1)))
        if step%100==0:
            row = dict(step=step,loss=float(loss.detach()),rgb=float(loss_rgb.detach()),depth=float(loss_depth.detach()))
            history.append(row)
            print(row,flush=True)
        if (step+1)%1000==0:
            torch.save(dict(step=step+1,params=params.state_dict(),optimizers={k:v.state_dict() for k,v in optimizers.items()},strategy_state=strategy_state),a.output/f'checkpoint_{step+1}.pt')
    after = evaluate('after')
    torch.save(dict(step=a.steps,params=params.state_dict()),a.output/'checkpoint_final.pt')
    with torch.no_grad():
        arrays={k:v.detach().cpu().numpy() for k,v in params.items()}
        xyz=arrays['means']; rgb=torch.sigmoid(params['colors']).detach().cpu().numpy(); opacity=torch.sigmoid(params['opacities']).detach().cpu().numpy()
        q=F.normalize(params['quats'],dim=-1).detach().cpu().numpy()
        R=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
        covariance=(R*np.exp(arrays['scales'])[:,None,:]**2)@R.transpose(0,2,1)
        keep=opacity>.03
        np.savez_compressed(a.output/'scene.npz',xyz=xyz[keep],rgb=rgb[keep],opacity=opacity[keep],covariance=covariance[keep],object_id=np.full(keep.sum(),-1))
        packed=np.column_stack((xyz,rgb,opacity,covariance[:,[0,0,0,1,1,2],[0,1,2,1,2,2]],np.full(len(xyz),-1))).astype('<f4')
        packed[keep].tofile(a.output/'scene.bin')
        names=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
        rows=np.column_stack((xyz,np.zeros_like(xyz),(rgb-.5)/.28209479177387814,arrays['opacities'],arrays['scales'],q)).astype('<f4')[keep]
        with (a.output/'scene.ply').open('wb') as f:
            f.write(('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(rows))+'\n'+''.join('property float '+n+'\n' for n in names)+'end_header\n').encode()); f.write(rows.tobytes())
        writer=cv2.VideoWriter(str(a.output/'flythrough.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8,(a.size,a.size))
        novel=[]
        for j,i in enumerate(np.linspace(0,len(frames)-1,48).astype(int)):
            T=np.array(frames[i]['c2w'])
            shift=extent*.002*np.sin(j/47*np.pi*2)
            T[:3,3]+=T[:3,0]*shift
            result,alpha,_=render(int(i),tensor(np.linalg.inv(T)))
            image=(result[...,:3].clamp(0,1).cpu().numpy()*255).astype(np.uint8)
            writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            if j in [8,24,40]:cv2.imwrite(str(a.output/f'novel_{j:02d}.jpg'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            novel.append(dict(frame=int(i),c2w=T.tolist(),offset_arbitrary_units=float(shift)))
        writer.release()
    report=dict(state='done',method='gsplat CUDA: joint position, scale, rotation, RGB and opacity optimization with adaptive densification; degree zero',gaussians=int(keep.sum()),before=before,after=after,total_s=time.perf_counter()-start,peak_vram_bytes=torch.cuda.max_memory_allocated(),train_indices=train,held_out_indices=held,steps=a.steps,size=a.size,source=str(a.reconstruction),coordinate_system=source['coordinate_system'],units='arbitrary VGGT units',online=False,semantic_gaussians=0,objects=[],loss_history=history,novel_views=novel,scene_sha256=hashlib.sha256((a.output/'scene.bin').read_bytes()).hexdigest(),limitations=['One fixed-direction perspective stream; no 360 coverage.','Appearance holdouts only: all frames influenced SLAM geometry and camera estimates.','Adaptive densification is bounded; degree-zero appearance has no spherical harmonics; unseen surfaces remain incomplete.','No semantic transfer: old map has different coordinates.','No independent-sequence validation; no real-time reconstruction claim.'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['gaussians','before','after','total_s','peak_vram_bytes']},indent=2),flush=True)


if __name__=='__main__':main()
