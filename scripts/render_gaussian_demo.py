"""Render an actual novel-camera Gaussian preview on CPU, no browser/GPU.

This is an offline render, not a real-time mapping benchmark. Uses the same
fixed-geometry alpha compositor used for fitting, at higher resolution.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from slam3d.dense.gaussian import raster_table, composite
from slam3d.geometry.sphere import perspective_view_rotation
from slam3d.io.tum import read_tum


def main():
    p=argparse.ArgumentParser();p.add_argument('run_dir');p.add_argument('--frames',type=int,default=36);p.add_argument('--size',type=int,default=192);a=p.parse_args()
    torch.set_num_threads(4)
    rdir=Path(a.run_dir).resolve();out=rdir/'gaussians';model=np.load(out/'scene.npz')
    report=json.loads((out/'report.json').read_text())
    times,T=read_tum(rdir/'geometry/trajectory_grav.tum');T[:,:3,3]*=report['metres_per_slam_unit']
    start=max(times[0],report['training_views'][0]['t']+20)
    end=min(times[-1],start+6)
    ts=np.linspace(start,end,a.frames)
    rotations=Slerp(times,Rotation.from_matrix(T[:,:3,:3]))(ts).as_matrix()
    xyz=np.column_stack([np.interp(ts,times,T[:,j,3]) for j in range(3)])
    rgb=torch.from_numpy(model['rgb']);opacity=torch.from_numpy(model['opacity'])
    cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{a.size}x{a.size}','-r','6','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(out/'flythrough.mp4')]
    started=time.perf_counter()
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    poses=[]
    try:
        for i,t in enumerate(ts):
            P=np.eye(4);P[:3,:3]=rotations[i]@perspective_view_rotation(.25,0)
            # Smooth offset from the actual trajectory proves novel-view rendering.
            P[:3,3]=xyz[i]+rotations[i][:,0]*(.2*np.sin(np.pi*i/max(1,a.frames-1)))
            ids,w,_=raster_table(model['xyz'],model['covariance'],P,a.size,layers=24)
            with torch.no_grad(): image,alpha=composite(rgb,opacity,torch.from_numpy(ids),torch.from_numpy(w))
            arr=(image.numpy().reshape(a.size,a.size,3).clip(0,1)*255).astype('uint8')
            cv2.putText(arr,'GAUSSIAN / NOVEL CAMERA',(6,13),cv2.FONT_HERSHEY_SIMPLEX,.30,(100,255,220),1,cv2.LINE_AA)
            cv2.putText(arr,f'Offline render  {t-ts[0]:.1f}s',(6,a.size-8),cv2.FONT_HERSHEY_SIMPLEX,.30,(255,255,255),1,cv2.LINE_AA)
            if i==a.frames//2:cv2.imwrite(str(out/'novel_view.jpg'),cv2.cvtColor(arr,cv2.COLOR_RGB2BGR))
            proc.stdin.write(arr.tobytes());poses.append(P.tolist())
            if i%6==0:print(f'[render] {i}/{a.frames}',flush=True)
    finally:
        proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
    seconds=time.perf_counter()-started
    (out/'flythrough_report.json').write_text(json.dumps({'render_s':seconds,'frames':a.frames,'size':a.size,'offline_render_fps':a.frames/seconds,'encoded_playback_fps':6,'poses_camera_to_world':poses,'source_times':ts.tolist(),'novel_camera_offset_max_m':.2,'note':'Offline CPU render from saved model, not real-time reconstruction or original video.'},indent=2))
    print(f'{out / "flythrough.mp4"}: {seconds:.2f}s, {a.frames/seconds:.2f} offline rendered fps')


if __name__=='__main__':main()
