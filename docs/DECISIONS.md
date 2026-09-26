# Decision record

Date: 2026-09-15. Machine: Ubuntu 24.04, RTX 5070 Laptop (8 GB, sm_120), 24 threads, 31 GB RAM, no sudo, Docker with NVIDIA runtime.

## Data

| Role | Choice | Why |
|---|---|---|
| User recording | none found | `~/Downloads/*.mp4` are ordinary 16:9 / 9:16 videos (no 2:1 equirect, no spherical metadata). User equirect-video ingestion is implemented (`configs/template_equirect_video.yaml`) but has not been validated on a user recording. |
| Public sample, development | Hilti-Trimble-Oxford 2026 `floor_2_2025-12-03_run_1` | Real construction site, Insta360 ONE RS 1-inch dual fisheye + 1 kHz IMU, Kalibr calibration, floor plans at 1 cm/px, LiDAR GT **in the floor-plan frame**. [paper](https://arxiv.org/abs/2607.06464) · [repo](https://github.com/Hilti-Research/hilti-trimble-slam-challenge-2026) · [data](https://huggingface.co/datasets/Hilti-Research/hilti-trimble-slam-challenge-2026). CC BY-NC-SA (non-commercial). |
| Held-out #1 | `floor_2_2025-10-28_run_2` | Same floor, low light + clutter + aggressive motion. It exposed two generic bugs (relative jump rule; lost-gap odometry), which were fixed. No parameters were tuned on it. |
| Held-out #2 | `floor_1_2025-05-05_run_1` | Different floor plan, early construction phase. Untouched during development. |

## SLAM backend

**Selected: [stella_vslam](https://github.com/stella-cv/stella_vslam) `e445b54`, equirectangular monocular, in Docker, driven by slam3d's C++ driver.**
- Native equirect camera model, loop closure + global BA, relocalisation.
- CPU-only: 34–41 ms median tracking per 1920×960 frame here, leaving the GPU free.
- The challenge ships an equirect config for this camera.
- It accepts exactly what a stitched consumer 360 video provides.
- The driver (`cpp/stella_driver`) adds:
  - true per-frame timestamps (the stock example synthesises 1/fps);
  - online (causal) pose logging with tracking state and loop-BA flags;
  - per-frame tracked landmarks in the camera frame (causal PF observations);
  - keyframe, landmark and observation exports.
- Scale is monocular (arbitrary). The upstream library is built unmodified.

**Documented alternative: [OKVIS2-X](https://github.com/ethz-mrl/OKVIS2-X) (BSD-3)** for raw dual-fisheye + IMU.
- The 2026 challenge winners built on it (SLAM 0.089 m average RMSE; localization 0.238 m). Those numbers are from the dataset paper and were not reproduced here.
- Not integrated: multi-camera + IMU configuration effort. The target input is stitched video.

**Considered, not selected:**
- [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM): perspective model, CC BY-NC-SA, VRAM.
- [VGGT-SLAM 2.0](https://github.com/MIT-SPARK/VGGT-SLAM): perspective input.
- [ODGS-SLAM](https://github.com/odgs-slam/odgs-slam): equirect 3DGS SLAM, but non-commercial academic license and per-frame optimisation; not benchmarked.

## Dense geometry

**[PanoVGGT](https://github.com/YijingGuo-June/PanoVGGT) (CVPR 2026, MIT code, `556bb7d`), used as a dense module rather than as SLAM.**
- Runs on [k−1, k, k+1] keyframe windows at 518×1036.
- Its ERP ray convention equals slam3d's panorama convention (verified in source).
- **fp32 weights + bf16 autocast** as upstream; pure bf16 fails with a dtype mix. Peak 5.8 GB, ~1 s per window on the RTX 5070.
- PanoVGGT poses are not used. Each keyframe's depth is scaled robustly to the radial ranges of its SLAM landmarks, because PanoVGGT scale varies ~12% between windows.

## Metric scale

1. **IMU, exact second-difference identity** `p(t+τ) − 2p(t) + p(t−τ) = ∫(τ−|u|)a(t+u)du`.
   - Result: 0.985× / 1.028× GT on the two floor-2 runs.
   - It replaced a Savitzky–Golay derivative-vs-smoothing comparison that was biased ~20%, confirmed by a synthetic contract test.
   - Used as the search prior.
2. **Camera-height prior** (no IMU): global floor layer by horizontal coverage, taking the highest significant layer. Glossy-floor reflections form a denser layer *below* the floor; per-camera peak methods were 25–36% off on development data.
3. **Plan registration** gives the externally calibrated scale (1.002–1.006× GT on the development sequence).

## Floor-plan alignment

[Z-FLoc](https://arxiv.org/abs/2606.04788) code is not released, so slam3d implements structural BEV registration:
- **Wall evidence:** BEV cells whose points span several height bins inside a band relative to the detected floor and camera height. This rejects floor, ceiling and low clutter.
- **Automatic search:**
  - exhaustive yaw × scale search, with translation by FFT correlation against a plan distance field plus a free-space term for the camera path;
  - robust refinement of the top hypotheses;
  - confidence = fit quality and margin to the best **distinct** alternative (> 1 m path difference);
  - **at least 30 wall cells** are required for `confident`.
- **Evidence variants:** sparse landmarks, and sparse + PanoVGGT dense. They are registered separately, and one is selected **without GT**: confident with enough evidence, then confident, then enough evidence, then quality. Per-variant GT errors are stored only as an ablation.
- **Manual correspondences:** from the viewer, Umeyama initialisation followed by refinement.
- **Local drift:** segment refinement over 45 s windows (22.5 s step).
  - Each window is registered with a Gaussian prior toward the global transform, accepted only on a clear inlier gain, and blended with triangular weights.
  - The window length was selected on the development sequence; on the held-out runs no window passed the gate, so they are unchanged.
  - This is not a full pose-graph re-optimisation (documented limitation).
- **Coverage completion** for the finished recording is flagged: tracked / static (IMU-verified) / gap-interpolated. It is never used by causal localization.
- **As-built comparison:**
  - planned boundary cells are classified as confirmed / unobserved-in-view / not covered, using ray-cast visibility from the path; observed-not-in-plan clusters are listed;
  - it is reported only when alignment is confident;
  - "not observed" never means "missing".

## Objects

- **Detection:** Ultralytics YOLOE-11s-seg with text prompts (open vocabulary; AGPL-3.0, local use) on 6 overlapping 100° perspective views per keyframe, with exact pixel maps back to the panorama.
- **3D lifting:** SLAM landmarks inside the eroded mask, then PanoVGGT dense depth, then a floor-ray fallback, each with background-leakage rejection.
- **Camera-rigid filter:** detections at a constant camera-frame position across ≥ 15% of keyframes are the operator, helmet or rig, not scene objects.
- **Association:** 3D gate + HSV histogram; never merges two observations from the same keyframe. Status is confirmed / tentative / dynamic.
- Extents are observed-point extents only.

## Localization

Custom particle filter over (x, y, φ, log s). φ is the gravity-frame→plan yaw, and scale uncertainty is explicit.
- **Motion:** online SLAM odometry.
  - Pose jumps are detected by a **physical speed bound** (8 m/s); the earlier relative rule fired 551 times under aggressive motion.
  - After a same-map relocalisation, the displacement across the gap is applied with inflated noise.
  - A map reset triggers structural recovery.
- **Observation:** a causal 4 s window of tracked landmarks, turned into BEV wall cells and scored with a robust mixture likelihood on the plan distance field.
  - Evidence is counted as at most 12 independent points per update and tempered by the fraction of new landmark IDs (no double counting of overlapping evidence).
  - Parameters come from a 16-point sweep on the development sequence.
- **Global init / recovery:** the same structural registration runs on the causal local map (last 30 s), and its distinct hypotheses seed a particle mixture (multi-modal in repetitive layouts).
- **Evaluation:** strictly causal replay; errors are computed only where the filter reports an estimate.
