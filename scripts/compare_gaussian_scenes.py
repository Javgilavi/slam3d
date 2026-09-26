"""Compare exported scenes on identical recorded crops and rig masks.

No GT poses enter rendering. Tests the NPZ export, not in-memory training state.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization
from scipy.spatial.transform import Rotation

from slam3d.geometry import sphere
from slam3d.io.tum import read_tum
from train_pano_gaussians import ssim_map, view_rotations


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--baseline',type=Path);a=p.parse_args()
    baseline=a.baseline or a.run/'gaussians'
    rep=json.loads((a.candidate/'report.json').read_text())
    args=json.loads((a.candidate/'arguments.json').read_text())
    out=a.candidate/'comparison';out.mkdir(exist_ok=True)
    torch.set_num_threads(6)
    tt=lambda x:torch.tensor(x,device='cuda',dtype=torch.float32)
    models={}
    for name,path in [('baseline',baseline),('candidate',a.candidate)]:
        r=json.loads((path/'report.json').read_text());d=np.load(path/'scene.npz')
        scale=float(rep['scale_metres_per_unit'])/float(r.get('scale_metres_per_unit',r.get('metres_per_slam_unit',1)))
        vals,vecs=np.linalg.eigh(d['covariance']);vecs[:,:,0]*=np.linalg.det(vecs)[:,None]
        q=Rotation.from_matrix(vecs).as_quat()[:,[3,0,1,2]]
        colors=np.concatenate([d['sh0'][:,None,:],d['shN']],axis=1) if 'shN' in d else d['rgb']
        models[name]=dict(means=tt(d['xyz']*scale),quats=tt(q),scales=tt(np.sqrt(vals)*scale),opacities=tt(d['opacity']),colors=tt(colors),sh_degree=r.get('sh_degree') if 'shN' in d else None)
    ts,poses=read_tum(a.run/'geometry/trajectory_grav.tum');pose_of={round(t,6):T for t,T in zip(ts,poses)}
    frames=np.genfromtxt(a.run/'ingest/frames.csv',delimiter=',',skip_header=1,dtype=str)
    if args.get('keyframe_stride'):
        for row in np.atleast_2d(np.loadtxt(a.run/'geometry/keyframes_grav.csv',delimiter=',',skiprows=1))[::args['keyframe_stride']]:
            i=int(np.argmin(abs(frames[:,1].astype(float)-row[1])));T=np.eye(4);T[:3]=row[2:14].reshape(3,4);pose_of[round(float(frames[i,1]),6)]=T
    static=cv2.imread(str(a.run/'ingest/static_mask.png'),0)
    size=args['size'];K=sphere.perspective_intrinsics(args['fov'],size)
    results={k:[] for k in models};panels=[]
    held=rep['held_out_panoramas'];save_ids=set(np.linspace(0,len(held)-1,min(6,len(held))).astype(int))
    with torch.no_grad():
        for j,index in enumerate(held):
            pano=cv2.imread(str(a.run/'ingest'/frames[index,2]));T=pose_of[round(float(frames[index,1]),6)].copy();T[:3,3]*=rep['scale_metres_per_unit']
            for view,R in view_rotations(args['views'],up=not args['no_up'],down=args['down']):
                crop,uv=sphere.equirect_to_perspective(pano,R,K,size)
                target=tt(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)/255.)
                mask=torch.tensor(sphere.sample_equirect(static,uv[...,0],uv[...,1],interp='nearest')>127,device='cuda')
                C=T.copy();C[:3,:3]=T[:3,:3]@R
                row=[target.cpu().numpy()]
                for name,model in models.items():
                    rendered,alpha,_=rasterization(**model,viewmats=tt(np.linalg.inv(C))[None],Ks=tt(K)[None],width=size,height=size,near_plane=.05,far_plane=200.,packed=False)
                    image=rendered[0].clamp(0,1);mse=(image-target).square()[mask].mean()
                    results[name].append(dict(pano=index,view=view,psnr_db=float(-10*torch.log10(mse)),ssim=float(ssim_map(image,target)[mask].mean()),coverage=float((alpha[0,...,0][mask]>.5).float().mean())))
                    row.append(image.cpu().numpy())
                if j in save_ids and view=='yaw000':
                    pair=(np.concatenate(row,axis=1)*255).astype(np.uint8)
                    cv2.putText(pair,f'Pano {index}: recorded | baseline | candidate',(6,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,80,30),1)
                    cv2.imwrite(str(out/f'pano_{index:06d}.jpg'),cv2.cvtColor(pair,cv2.COLOR_RGB2BGR));panels.append(pair)
    summary={name:{k:float(np.mean([row[k] for row in rows])) for k in ['psnr_db','ssim','coverage']} for name,rows in results.items()}
    report=dict(summary=summary,views=results,conditions=dict(size=size,fov=args['fov'],held_out_panoramas=held,mask='static rig mask',renderer='gsplat CUDA; exported NPZ; same crops and camera poses',baseline=str(baseline),candidate=str(a.candidate),limitation='Appearance holdouts; both scenes use geometry estimated from the sequence. Initial point colors can contain held-out observations.'))
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))
    cv2.imwrite(str(out/'contact.jpg'),cv2.cvtColor(np.concatenate(panels),cv2.COLOR_RGB2BGR))


if __name__=='__main__':main()
