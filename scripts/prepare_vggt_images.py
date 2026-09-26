"""Prepare one temporally ordered, fixed-direction perspective stream from ERP.

This is an input adapter, not SLAM. It never interleaves cube directions or treats
them as stereo. Camera centres are unchanged. VGGT estimates geometry itself.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from slam3d.geometry.sphere import equirect_to_perspective,perspective_intrinsics,perspective_view_rotation


def main():
    p=argparse.ArgumentParser();p.add_argument('run_dir');p.add_argument('--fps',type=float,default=2);p.add_argument('--size',type=int,default=518);p.add_argument('--fov',type=float,default=90);p.add_argument('--yaw',type=float,default=0);a=p.parse_args()
    if not 0<a.fps<=30 or not 0<a.fov<160 or a.size<64:raise ValueError('Invalid fps, fov or size')
    rdir=Path(a.run_dir).resolve();out=rdir/'vggt_input';out.mkdir(exist_ok=True)
    if (out/'images').exists():raise FileExistsError(f'{out}/images already exists; preserve or move it before creating a new input')
    (out/'images').mkdir()
    rows=np.atleast_2d(np.genfromtxt(rdir/'ingest/frames.csv',delimiter=',',skip_header=1,dtype=str));times=rows[:,1].astype(float)
    selected=[];last=-np.inf
    for i,t in enumerate(times):
        if t-last>=1/a.fps-1e-4:selected.append(i);last=t
    R=perspective_view_rotation(np.deg2rad(a.yaw),0);K=perspective_intrinsics(a.fov,a.size);frames=[]
    for n,i in enumerate(selected):
        img=cv2.imread(str(rdir/'ingest'/rows[i,2]))
        if img is None:raise FileNotFoundError(rows[i,2])
        crop,_=equirect_to_perspective(img,R,K,a.size)
        name=f'{n:06d}.jpg'
        if not cv2.imwrite(str(out/'images'/name),crop,[cv2.IMWRITE_JPEG_QUALITY,95]):raise IOError(name)
        frames.append({'file':name,'source_frame_index':int(i),'timestamp':float(times[i])})
    metadata={'source':str(rdir),'camera_model':'pinhole','size':[a.size,a.size],'K':K.tolist(),'R_pano_view':R.tolist(),'translation_pano_view':[0,0,0],'requested_fps':a.fps,'frames':frames,'limitations':['Single fixed panorama direction; does not reconstruct full 360-degree coverage.','Consumer stitching is approximately central.','No poses or ground truth supplied to VGGT-SLAM.','Native VGGT-SLAM multi-camera support is not assumed.'],'state':'input_prepared_not_reconstructed'}
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2));print(f'{len(frames)} images: {out / "images"}')


if __name__=='__main__':main()
