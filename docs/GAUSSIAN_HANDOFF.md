# 2026-09-16 resumed results — selected improved baseline

## Current outcome

The default `outputs/hilti_floor_2_2025-12-03_run_1/gaussians/` now contains the improved CUDA-trained scene from `gaussians_keyframes_v5`. The original scene is preserved as `gaussians_cpu_baseline_20260916/`; the training checkpoint remains in `gaussians_keyframes_v5/`. All VGGT runs and unsuccessful training experiments are retained. Claude's source/handoff snapshot is in `outputs/development_snapshot_20260916/`.

**This improves recorded-neighbourhood rendering but does not yet deliver high-quality arbitrary free-view reconstruction.** Visually inspected novel frames at offsets up to 0.6 m have severe floaters, blur, and occlusions (particularly novel_090 and novel_150). Do not hide these failures behind the appearance metrics. See `gaussians/quality_assessment.json`.

## Independently verified

- CUDA matmul passes on RTX 5070 Laptop, torch 2.9.1+cu128 (`outputs/gpu_preflight_20260916.json`). gsplat forward/backward executes on CUDA.
- VGGT-SLAM results reproduced from exported cameras against the GT timestamps, with Sim3 alignment for evaluation only: submap-4 RMSE **10.926 m**, submap-8 **3.811 m**, stella at the same 197 timestamps **0.130 m**. Record: `reconstruction_comparison.json`. Neither VGGT map was selected for Gaussian fitting.
- Selected training uses final **bundle-adjusted stella keyframe cameras**, four horizontal 100-degree crops at each shared centre, **IMU-derived scale** (3.702537 units-to-metres), and PanoVGGT/landmark initialization. It does not use floor-plan alignment or GT to fit the scene.
- 83 panoramas, 332 crops, 40 appearance holdout crops from 10 centres. Geometry and initial point colours used the sequence, including holdout observations; these are not independent reconstruction tests.
- 366,398 exported Gaussians; joint position/scale/rotation/opacity/bounded RGB optimization, adaptive densification, degree-zero appearance. 6,000 updates: **52.48 s training**, **71.15 s additional stage**, peak allocated CUDA memory **268.79 MiB**. These times exclude ingestion, SLAM and dense reconstruction and are not end-to-end real-time claims.

### Same-camera exported-artifact comparison

`scripts/compare_gaussian_scenes.py` renders both exported NPZ scenes with gsplat at identical cameras, 320×320 resolution, crops and rig masks (40 views):

| | Old baseline | Selected scene |
|---|---:|---:|
| PSNR dB | 14.50 | 17.53 |
| SSIM | 0.467 | 0.520 |
| Alpha coverage >0.5 | 96.33% | 97.71% |

Report and recorded/baseline/candidate panels: `gaussians/comparison/`. These supersede comparisons between differently sized old/new training metrics.

### Browser validation

Served `/world` passes loading, movement, playback, object inspector and console-error checks. Intel Mesa / ANGLE EGL renderer, 1440×900 browser viewport, render pixel ratio 0.65 for this scene: **21.2 FPS overview**, **18.9 FPS walking**, playback advanced 2.0 s in the 2-second check. Full 366,398 splats are shown (no subset). The regenerated standalone `demo.html` also passes artifact and browser checks (20.0 FPS measured; `gaussians/validation.json`). These are browser rendering rates, not mapping throughput. See `gaussians/world_validation.json` and actual `world_*.png` screenshots. 71 object entries remain selectable; labels use static spatial association, not mask/visibility fusion.

## Fixes and development state

- Up/down crop optical axes corrected in the trainer; the underlying sphere helper's pitch sign disagrees with its prose documentation. A regression test checks the actual zenith/nadir rays.
- Viewer now accepts old reports without SSIM, uses elapsed wall time for playback, performs bucket depth ordering and avoids per-splat temporary arrays.
- Browser covariance projection clamps off-axis slopes like the CUDA rasterizer and culls off-screen footprints. Without this, nearby off-screen splats covered the entire image with grey and walking ran below 1 FPS. Actual screenshots caught this despite valid artifact files.
- Optional degree-3 SH texture rendering and endpoint added; **the selected scene is degree zero**, so SH rendering is not exercised by its validation.
- Publishing stages files before swapping directories, preserves the old scene, and records the actual browser-buffer hash.
- Trainer records arguments and intermediate held-out metrics, supports bounded RGB, exact keyframe cameras, visibility-aware Adam experiments and L1-only fitting. Several full-scene SH/SSIM experiments degraded and were not selected. **The successful recipe changes both camera selection and loss; the causal contribution of each has not been isolated.**
- Existing tests plus the camera regression: **25 passed** in `.venv-vggt`.

## Reproduce selected run in a NEW directory

```bash
PATH="$PWD/.venv-vggt/bin:$PATH" PYTHONPATH=src TORCH_CUDA_ARCH_LIST=12.0 \
.venv-vggt/bin/python scripts/train_pano_gaussians.py \
  outputs/hilti_floor_2_2025-12-03_run_1 NEW_OUTPUT \
  --keyframe-stride 2 --views 4 --no-up --size 320 --steps 6000 \
  --max-init 300000 --max-gaussians 800000 --means-lr-scale .05 \
  --color-lr .025 --scale-lr .001 --max-scale-m .15 --bounded-rgb --ssim-weight 0
```

Use the explicit validated recipe above; experimental trainer defaults are not a quality guarantee. Launch the selected scene with `.venv/bin/slam3d viewer outputs/hilti_floor_2_2025-12-03_run_1`, then `/world`, or open `gaussians/demo.html` directly. The standalone file contains nearest-keyframe reference previews, not a full embedded recording.

---

# Gaussian scene work

## 2026-09-16: continuation (Claude Code session, after Codex stopped on a usage limit)

Codex's last turn ended 00:02 with "You've hit your usage limit"; its full VGGT-SLAM run finished 00:03. Its files are preserved unchanged: `scripts/{run_vggt_reconstruction,train_vggt_gaussians,export_vggt_demo,validate_vggt_scene,prepare_vggt_images,gpu_preflight}.py`, `.venv-vggt`, `third_party/{vggt_slam,vggt_spark,salad}`, `outputs/<run>/{vggt_input,vggt_pilot,vggt_full,gaussians,gaussians_vggt_pilot}`.

### Verified in this session

**CUDA.** Works in this environment: RTX 5070 Laptop, torch 2.9.1+cu128, real CUDA matmul. gsplat 1.5.3 kernels are cached in `~/.cache/torch_extensions/*/gsplat_cuda`.
- `.venv-vggt` needs `PATH=.venv-vggt/bin:$PATH`: gsplat's loader checks for `ninja`, and without it Codex's `gaussians_vggt_full_v0` training crashed before any step.
- It also needs `PYTHONPATH=src` for slam3d modules.

**VGGT-SLAM on the prepared 306 forward-looking crops (199 selected).** Its camera centres were matched to the LiDAR GT via manifest timestamps and aligned with Sim3.

| run | submaps / loops | ATE RMSE / max | 2 m segment-scale spread p90 | recovered fx (true 259) | peak VRAM |
|---|---|---|---|---|---|
| `vggt_full` (submap 4, Codex) | 55 / 5 | **10.9 m / 29.0 m** | 5.39 | 23–1506 px | 4.8 GB |
| `vggt_full_s8` (submap 8, this session) | 25 / 0 | **3.8 m / 9.0 m** | 0.43 | 219–451 px | 6.0 GB (8 GB card limit) |
| stella_vslam at the same 197 timestamps | – | **0.13 m / 0.37 m** | – | fixed | CPU |

Top-down maps (scratch) show the submap-4 path collapsing and streaking, and the submap-8 path bent with doubled walls. **Conclusion:** VGGT-SLAM poses are not usable for Gaussian training on this sequence and this GPU. The Gaussian world uses stella poses.

**Pose conventions for Gaussian training.** SLAM landmarks reproject into their keyframes through `geometry/keyframes_grav.csv` at 1.2–1.3 px median (1920 px panorama). Conventions are correct.

**PanoVGGT dense cloud registration.** Landmark → nearest dense point: median 0.10 m, p75 0.25 m, p90 0.57 m. The fused cloud has thick, offset surfaces and floaters.
- New multi-view consistency filter: keep a point only if ≥ 2 of its 3 nearest keyframes' depth maps agree within 5% and no keyframe sees through it.
- The filter keeps 83% of points; 4.6% violate free space.

**First gsplat trainer bug (fixed).** Gaussians exploded into huge blobs: 4000 steps took held-out PSNR from 11.95 to 8.14 dB and training PSNR to 8.97 dB. Causes:
- no scale clamp;
- gsplat only prunes oversized Gaussians when `step > reset_every`, which fell after `refine_stop_iter`;
- unsupervised near-camera Gaussians (rig, operator);
- a nadir view that is mostly masked rig;
- the render clamped before the loss.

Fixes:
- scale clamp 1 mm – 0.3 m;
- camera-clearance pruning at 0.3 m, also at initialisation;
- schedule proportional to the step count;
- no nadir view by default;
- unclamped loss.

### New scripts (this session)
- `scripts/train_pano_gaussians.py`: 360° shared-centre multi-view crops from stitched panoramas; stella poses in metres (plan-registration scale); PanoVGGT/landmark initialisation (`--init dense|landmarks|dense_consistent`); rig mask; SH degree 3; densification; held-out panoramas; raw and colour-compensated PSNR, SSIM; novel side-offset renders; exports; checkpoints/resume.
- `scripts/publish_gaussian_scene.py`: renames the existing `gaussians/` to `gaussians_<suffix>/` (never deletes), publishes the trained scene with a browser level-of-detail subset and viewer-compatible report/objects.
- `scripts/validate_world_page.py`: served `/world` page in GPU Chromium; fps, interactions, screenshots.
- `viewer/static/world.js|html`: metrics/scope text read from the report's method and limitations (old reports still render as before).

### Exact environment for the new scripts
```bash
export PATH="$HOME/slam3d/.venv-vggt/bin:$PATH" PYTHONPATH=$HOME/slam3d/src
.venv-vggt/bin/python scripts/run_vggt_reconstruction.py outputs/hilti_floor_2_2025-12-03_run_1/vggt_input/images outputs/hilti_floor_2_2025-12-03_run_1/vggt_full_s8 --submap-size 8
.venv-vggt/bin/python scripts/train_pano_gaussians.py outputs/hilti_floor_2_2025-12-03_run_1 <out> --init dense_consistent
```

### Status
Initialisation comparison is running. Full training, publishing and browser validation are pending; results will be appended here.

---

# Gaussian scene work — 2026-09-15

## Latest request: GPU diagnosis and VGGT-SLAM replacement

The user rejected the visual quality and explicitly asked to make CUDA work and use VGGT-SLAM to reconstruct a better map. No improved VGGT-SLAM map has been generated yet; execution is blocked at the device boundary described below.

### New diagnosis, directly verified

- `/proc/driver/nvidia/gpus/*/information` identifies **NVIDIA GeForce RTX 5070 Laptop GPU**, not excluded.
- NVIDIA open kernel driver **580.178.04** is loaded; `nvidia`, `nvidia_uvm`, `nvidia_modeset`, `nvidia_drm` modules are present.
- `nvcc --version` succeeds: CUDA toolkit **12.9**.
- Project torch **2.9.1+cu128** imports successfully.
- **There are no `/dev/nvidia*` or `/dev/dri` entries visible to this process.** CUDA initialization fails with `No CUDA GPUs are available`.
- This establishes a missing hardware-access prerequisite in the execution environment. It does not prove that the host driver has no other issues, but changing torch/toolkit versions cannot remedy invisible device nodes.
- The active managed environment forbids escalation and restricts networking. Do not attempt device-node creation, alternate host processes, container sockets or other routes to bypass that boundary. Driver installation/removal was not attempted.
- `git ls-remote https://github.com/MIT-SPARK/VGGT-SLAM.git HEAD` fails with DNS resolution (the existing git configuration routes it through SSH). Upstream has not been cloned or pinned; do not claim otherwise.

Reproducible diagnostic: `.venv/bin/python scripts/gpu_preflight.py` writes `outputs/gpu_preflight.json` and returns **2** when real CUDA execution fails. It tests a CUDA matrix multiplication and verifies its values if a GPU is accessible. This was run and failed on missing GPU visibility, not model memory usage.

### VGGT-SLAM research and preparation

[Official repository](https://github.com/MIT-SPARK/VGGT-SLAM/blob/main/README.md), reviewed this turn: main is VGGT-SLAM 2.0; image-folder mapping uses `python main.py --image_folder ... --max_loops 1 --vis_map`. It incrementally visualizes dense point maps. Open-set detection is optional (`--run_os`) and brings Perception Encoder/SAM3 dependencies and their separate terms. Do not describe its point-map viewer as trained Gaussian rendering.

[Paper, submitted January 27, 2026](https://arxiv.org/abs/2601.19887): revised factor-graph alignment of VGGT submaps, reduced projective ambiguity/drift and attention-based loop verification. Published real-time demonstrations use Jetson Thor; those results are not measurements on this laptop. The next experiment should use the actual upstream mapper before deciding whether to train a Gaussian renderer on its output.

Prepared **306 real perspective images** covering the development recording, sampled at approximately 2 FPS, 518×518 and 90° FOV. Command:

```bash
.venv/bin/python scripts/prepare_vggt_images.py outputs/hilti_floor_2_2025-12-03_run_1 --fps 2 --size 518
```

Files: `outputs/hilti_floor_2_2025-12-03_run_1/vggt_input/images/` and `manifest.json`. The manifest preserves source timestamps, intrinsic matrix, exact fixed rotation and zero translation from the panorama centre. It is **one fixed-direction temporal stream**, not interleaved cube faces and not full 360° coverage. No SLAM or GT poses are passed as input. The adapter preserves an existing images directory by refusing to overwrite it. Compilation and every output image's shape were checked.

### Next required execution conditions

Resume with GPU devices exposed to the task and permitted upstream downloads. First run `scripts/gpu_preflight.py` until an actual CUDA operation passes. Then inspect/pin upstream installation, use an isolated environment, run a short bounded submap configuration on these prepared images, measure peak VRAM and inspect map quality before scaling up. Integrate full 360 geometry through an explicit shared-centre multi-view adapter rather than concatenating camera directions. Only after that should joint Gaussian geometry/appearance optimization and visibility-gated semantic fusion replace the existing weak baseline.

## User's current objective

Continue this repository. Build a freely navigable Gaussian reconstruction of the actual construction video with semantic detections. Building-plan work is deferred. Distinguish processing at video speed from interactive playback of a processed scene.

## Verified starting state

- Existing real Hilti recordings, stella_vslam outputs, PanoVGGT dense radial depths, objects and local three.js viewer are present. Preserve these files.
- Existing reports describe earlier GPU runs; this session independently finds `torch.cuda.is_available() == False` and `nvidia-smi` cannot contact the driver. No drivers/system changes authorized or attempted.
- Shell download attempt fails DNS resolution. Existing `.venv` imports torch 2.9.1+cu128, numpy, scipy, OpenCV and Playwright. Browser binaries and three.js are cached. `gsplat` is not installed.
- Original viewer renders points, not Gaussian splats. Original PanoVGGT is a geometry estimator, not a Gaussian renderer.
- No new reconstruction-quality measurements have yet been made in this session. Prior `docs/RESULTS.md` numbers belong to the existing work.

## Selected feasible implementation

Use saved gravity-frame PanoVGGT geometry and SLAM poses (no floor-plan transform). Initialize anisotropic Gaussians from local surface covariance; fit appearance against recorded panoramas with a bounded CPU differentiable alpha compositor. Export covariance, RGB, opacity, semantics and provenance. Add a dedicated Gaussian world view using locally cached three.js, a projected-covariance shader, depth sorting, free movement, recorded-view playback, object selection and top-down current-position display.

This baseline fixes geometry and fits appearance; it is not the full 3DGS optimizer (no position/covariance optimization, densification or spherical harmonics). It is not online SLAM: poses and depth come from completed offline runs. Live GPU mapping remains blocked by CUDA availability and missing gsplat.

## Research sources

- [PanoVGGT official implementation](https://github.com/YijingGuo-June/PanoVGGT): panoramic geometry, existing pinned code/weights in `third_party/VERSIONS.json`.
- [gsplat official implementation](https://github.com/nerfstudio-project/gsplat): Apache-2.0 CUDA rasterizer; candidate production training backend when CUDA and downloads are available.
- [gsplat camera API](https://docs.gsplat.studio/main/apis/rasterization.html): supports pinhole/fisheye and other models; do not feed equirectangular images as pinhole images. Exact rotated perspective crops must retain shared centres.
- [Original 3DGS paper](https://arxiv.org/abs/2308.04079): real-time rendering follows optimization; rendering FPS is not reconstruction throughput.

## Completed checkpoint

All changes are in this repository; the original untracked files and previous results were preserved. Before the user supplied this path, an empty isolated environment was created at `~/site-slam/.venv`; no project implementation was placed there. Sub-agent attempts failed due to the service usage limit; implementation here was done by the main agent.

### Source files

- `src/slam3d/dense/gaussian.py`: covariance initialization, spherical-to-pinhole view preparation, dynamic rectangle masking, fixed footprint tables, differentiable alpha compositing, RGB/opacity fitting, appearance holdouts, semantic association, standard degree-zero Gaussian PLY and binary/NPZ exports.
- `src/slam3d/cli.py`: `gaussians` command and failure status.
- `src/slam3d/viewer/server.py`: `/world`, `/api/gaussians`, `/api/gaussians.bin`, all in gravity coordinates. No plan dependency in these handlers.
- `viewer/static/world.html`, `world.js`: projected-covariance Gaussian rendering, view-dependent depth sort, orbit/walk/follow, original frame images, path visit selection, top-down virtual position, static object inspector and semantic colors. Existing viewer links to it.
- `scripts/export_gaussian_demo.py`: self-contained HTML with embedded libraries/data and **keyframe-only image previews**; no sockets/network required when opened by the user.
- `scripts/render_gaussian_demo.py`: actual novel-pose CPU rendering and encoded clip.
- `scripts/validate_gaussian_world.py`: direct route/artifact checks plus browser interaction checks when browser startup is possible.
- `tests/test_gaussian.py`: analytical-vs-numerical projection Jacobian, ordered alpha composition and gradients, depth order/behind-camera rejection, shared-world-transform invariance.

### Exact commands run

```bash
cd ~/slam3d
.venv/bin/slam3d gaussians outputs/hilti_floor_2_2025-12-03_run_1 --max-points 45000 --views 12 --steps 120 --size 96
.venv/bin/python scripts/export_gaussian_demo.py outputs/hilti_floor_2_2025-12-03_run_1
.venv/bin/python scripts/render_gaussian_demo.py outputs/hilti_floor_2_2025-12-03_run_1 --frames 36 --size 192
.venv/bin/python -m pytest -q tests
node --input-type=module --check < viewer/static/world.js
.venv/bin/python scripts/validate_gaussian_world.py outputs/hilti_floor_2_2025-12-03_run_1 --no-browser
ffprobe -v error -show_entries format=duration,size -show_entries stream=width,height,nb_frames -of json outputs/hilti_floor_2_2025-12-03_run_1/gaussians/flythrough.mp4
```

### Measured new results

`outputs/hilti_floor_2_2025-12-03_run_1/gaussians/report.json` is the machine-readable record, including parameters, all train/holdout camera transforms, source model references, scene checksum, limitations and loss history.

| Quantity | Measured result / condition |
|---|---|
| Scene | 45,000 anisotropic Gaussians from cached real Hilti dense reconstruction |
| Fitting cameras | 12 distinct panorama centres × 4 pure-rotation 90° views, 96×96 pixels |
| Appearance train / held out | 36 / 12 views; 3 held-out centres; no fitting images from these centres |
| Train PSNR before → after | 12.87 → 15.12 dB |
| Held-out appearance PSNR before → after | 13.48 → 13.87 dB |
| Held-out opacity coverage | 95.90% pixels alpha >0.5 (does not imply correct geometry) |
| Preparation | 39.55 s |
| Fit and final evaluation | 2.12 s, 120 optimizer updates, torch CPU 4 threads |
| Whole additional Gaussian stage | 42.16 s; source video 152.92 s; ratio 0.276 |
| Peak process RSS | 864.79 MiB |
| Spatially associated semantic Gaussians | 1,522; remainder unknown |
| Novel-view CPU clip | 36 frames at 192×192, 29.43 s rendering = 1.22 actual FPS; encoded at 6 FPS for a 6-second clip |
| Automated contracts | 24 passed (20 existing + 4 new), 5.32 s |
| Artifact checks | finite values, positive covariance eigenvalues, valid opacity, handler return files, trajectory lengths/monotonic timestamps pass |
| New browser rendering FPS / responsiveness | **Unmeasured — browser launch blocked** |

The Gaussian-stage ratio excludes all existing SLAM, dense geometry, semantic inference and ingestion time. It must not be reported as end-to-end real-time mapping. Appearance PSNR excludes static rig masks and detector-supported dynamic rectangles; geometry/SLAM used the whole sequence, including held-out centres. There is no independent new sequence validation for this addition. Training caps Gaussian support at 16 pixels radius and 24 contributors per pixel; the browser uses full 3-sigma projected ellipses and a global depth sort, so its result need not be pixel-identical to the bounded training renderer.

### Delivered artifacts and launch

In `outputs/hilti_floor_2_2025-12-03_run_1/gaussians/`:

- `demo.html`: 9.34 MiB standalone navigable world; open directly in a modern browser. No server dependency. Image previews are nearest saved keyframes, not frame-exact video.
- `scene.ply`: standard 3DGS degree-zero attributes (log scale, opacity logit, wxyz rotations).
- `scene.bin`: little-endian float32 rows of 14: xyz, RGB, alpha, covariance xx/xy/xz/yy/yz/zz, object ID (-1 unknown).
- `scene.npz`: full named arrays, no pickle.
- `report.json`, `status.json`, `validation.json`: measured provenance/results.
- `train.jpg`, `held_out.jpg`: left recorded crop / right fitted render.
- `flythrough.mp4`, `novel_view.jpg`, `flythrough_report.json`: actual novel camera rendering, source times and exact camera poses. Camera positions deviate from the recorded path by up to 0.2 estimated metres. The preview is visibly blurry and incomplete; do not imply photorealism.

Full viewer from a normal local terminal:

```bash
cd ~/slam3d
.venv/bin/slam3d viewer outputs/hilti_floor_2_2025-12-03_run_1
# http://127.0.0.1:8765/world
.venv/bin/python scripts/validate_gaussian_world.py outputs/hilti_floor_2_2025-12-03_run_1
```

### Concrete blockers and remaining work

1. **Browser validation:** Chromium aborts at `sandbox_host_linux.cc:41`, `shutdown: Operation not permitted`. A plain Python `socket.socket()` raises `PermissionError [Errno 1]`. The ASGI transport smoke attempt also stalled and was interrupted; it is not counted as passed. Direct route handlers were checked instead. Run the browser validator from an environment that permits normal browser IPC; inspect shaders, rendering, movement, controls, and capture `world_screenshot.png`. Do not claim the new UI is browser-tested yet.
2. **GPU availability:** both `nvidia-smi` and torch CUDA checks fail here. Earlier cached runs used an RTX 5070. Restore access through the execution environment/user-managed setup; do not silently change drivers.
3. **Production Gaussian training:** gsplat is absent and shell DNS downloads are blocked. With CUDA/network available, integrate a pinned gsplat backend using exact perspective crop rotations and shared camera centres, optimize geometry/covariance/opacity/SH, add densification and depth regularization, then measure held-out quality. The CPU baseline only optimizes color and opacity.
4. **True live pipeline:** currently PanoVGGT uses [k−1,k,k+1] windows and final SLAM poses. Implement bounded causal windows, incremental Gaussian insertion/pruning, timestamped semantic updates and pose-correction propagation before claiming online mapping. Real-time playback of a saved model is different.
5. **Semantics:** current Gaussian IDs use static observed boxes; original masks are not fused into per-Gaussian labels. Add mask/depth/visibility-gated voting, retain uncertainty/unknown, reject dynamic geometry from the dense scene itself, validate association failures with annotations.
6. **Localization:** moving in the viewer updates virtual map coordinates; no physical-camera relocalization anywhere in the map was added. Existing offline localization work remains separate.
7. **User recording:** no supplied recording has been processed in this session. Use the existing template ingestion/SLAM/dense/objects workflow, then `slam3d gaussians NEW_RUN_DIR`. Do not represent the public-data demo as validation on user footage.
8. Gaussian generation currently regenerates the Gaussian output on rerun; fitting checkpoints/resume beyond existing pipeline outputs remain to be added. Existing prior-stage caches are reused.
