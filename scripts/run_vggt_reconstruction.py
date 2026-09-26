"""Run pinned upstream VGGT-SLAM and export optimized cameras and dense geometry.

Uses a separate output directory; preserves per-submap predictions for inspection.
Camera matrices are decomposed after SL(4) optimization, including updated K.
"""
import argparse
import gc
import json
import time
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('images', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--submap-size', type=int, default=4)
    p.add_argument('--min-disparity', type=float, default=25)
    p.add_argument('--max-loops', type=int, default=1)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    from vggt.models.vggt import VGGT
    from vggt_slam.solver import Solver
    from vggt_slam.slam_utils import decompose_camera
    torch.set_num_threads(6)
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    np.random.seed(0)
    solver = Solver(init_conf_threshold=35, lc_thres=0.95, vis_voxel_size=0.01)
    original_get_frames = solver.map.get_frames_from_loops
    solver.map.get_frames_from_loops = lambda loops: [f.cuda() for f in original_get_frames(loops)]
    model = VGGT()
    model.load_state_dict(torch.load('third_party/weights/vggt_1b.pt', map_location='cpu', weights_only=True, mmap=True))
    model.eval().to(dtype=torch.bfloat16, device='cuda')
    files = sorted(a.images.glob('*.jpg'))
    if a.limit:
        files = files[:a.limit]
    selected = [str(f) for f in files if solver.flow_tracker.compute_disparity(cv2.imread(str(f)), a.min_disparity)]
    if str(files[-1]) != selected[-1]:
        selected.append(str(files[-1]))
    (a.output/'selected.json').write_text(json.dumps(selected, indent=2))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    timings = []
    for offset in range(0, len(selected)-1, a.submap_size):
        group = selected[offset:offset+a.submap_size+1]
        t = time.perf_counter()
        predictions = solver.run_predictions(group, model, a.max_loops, None, None)
        saved = {k: v for k, v in predictions.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(a.output/f'prediction_{offset:04d}.npz', **saved, files=np.array(group))
        solver.add_points(predictions)
        solver.graph.optimize()
        # Keep old images/descriptors on CPU so GPU memory is bounded by a window.
        for sub in solver.map.get_submaps():
            sub.frames = sub.frames.cpu()
        del predictions, saved
        gc.collect()
        torch.cuda.empty_cache()
        timings.append(time.perf_counter()-t)
        print('CHECKPOINT', offset, 'seconds', timings[-1], 'peak VRAM', torch.cuda.max_memory_allocated(), flush=True)
    records, xyz, rgb = [], [], []
    seen = set()
    for sub in solver.map.get_submaps():
        if sub.get_lc_status():
            continue
        points, ids, masks = sub.get_points_list_in_world_frame(solver.graph)
        projections = sub.get_all_poses_world(solver.graph, give_camera_mat=True)
        for i, name in enumerate(sub.img_names):
            if name in seen:
                continue
            seen.add(name)
            K, R, c, scale = decompose_camera(projections[i])
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, c
            mask = masks[i] & np.isfinite(points[i]).all(-1)
            # Store per-frame maps for holdout-safe initialization and depth checks.
            frame_path = a.output/f'frame_{len(records):04d}.npz'
            np.savez_compressed(frame_path, xyz=points[i].astype(np.float32), rgb=sub.colors[i], valid=mask, K=K, c2w=T)
            records.append(dict(image=name, data=frame_path.name, K=K.tolist(), c2w=T.tolist()))
            xyz.append(points[i][mask][::8])
            rgb.append(sub.colors[i][mask][::8])
    np.savez_compressed(a.output/'map.npz', xyz=np.concatenate(xyz).astype(np.float32), rgb=np.concatenate(rgb))
    report = dict(frames=records, input_count=len(files), selected_count=len(selected), submaps=solver.map.get_num_submaps(), loops=solver.graph.get_num_loops(), seconds=time.perf_counter()-start, submap_seconds=timings, peak_vram_bytes=torch.cuda.max_memory_allocated(), arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}, commits={n:subprocess.check_output(['git','-C','third_party/'+n,'rev-parse','HEAD'], text=True).strip() for n in ['vggt_slam','vggt_spark','salad']}, coordinate_system='VGGT arbitrary scale; OpenCV camera axes; no GT alignment')
    (a.output/'reconstruction.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='frames'}, indent=2), flush=True)
    solver.viewer.server.stop()


if __name__ == '__main__':
    main()
