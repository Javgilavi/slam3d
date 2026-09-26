# Progress notes and checkpoints

All stage results are resumable: `slam3d run <config>` skips stages whose `status.json` entry is `done` with an identical config hash; `--stages a,b --force` recomputes selected stages. Measured numbers live in the generated [RESULTS.md](RESULTS.md).

## 2026-09-15: session 1

**Environment**
- Ubuntu 24.04, RTX 5070 Laptop 8 GB (sm_120), no sudo.
- PyTorch 2.9.1+cu128 in `.venv` (uv), pinned in `requirements.lock.txt`.
- stella_vslam `e445b54` in Docker `slam3d-stella:e445b54`; C++ driver compiled against it.
- The project folder was deleted at the user's request mid-session and then restored at their request. Code was rewritten from the session record, and data/weights were re-downloaded.

**Data**
- Hilti-Trimble-Oxford 2026: `floor_2_2025-12-03_run_1` (development), `floor_2_2025-10-28_run_2` (held-out #1), `floor_1_2025-05-05_run_1` (held-out #2, untouched during development).
- No user 360 recording on the machine.

**Verified contracts**
- `pytest`: 20 tests.
  - Sphere, seam and perspective maps; KB4/EUCM, including the real calibration.
  - Umeyama/RANSAC, pose interpolation, timestamp association.
  - Floor layer with a reflection layer below and clutter above; IMU scale on a synthetic known-scale trajectory.
  - Stitch maps vs calibration; registration and PF on a synthetic plan.
- Real data:
  - IMU gravity vs GT gravity in the pano frame: 1.16° median;
  - IMU↔GT time offset −4 ms;
  - GT poses fall 100% in plan free space.

**Development findings and fixes (in order)**
1. **Scale.** A 1.3 m camera-height prior made the scale 0.64× GT, so registration failed (correctly flagged ambiguous). Fixes:
   - global floor layer by coverage;
   - IMU second-difference identity (the first IMU version was 20% biased by filter mismatch, which the synthetic test caught);
   - scale search 0.6–1.6×.
2. **PanoVGGT** needs fp32 weights + bf16 autocast (pure bf16 dtype mix). Per-keyframe scale is required (12% spread across windows).
3. **PF overconfidence.**
   - Oracle test: 65% of observed wall cells lie on planned structure at the GT pose, but single-update likelihood peaks are 0.79 m median from GT.
   - Sweep selected cap 12, σ 0.45 m, 4 s window.
4. **Alignment confidence.** The runner-up is now the best *distinct* solution. A 30-wall-cell minimum is required for `confident`. Selection without GT: confident & enough evidence first.
5. **Held-out #1 exposed:**
   - a relative jump rule firing 551× under aggressive motion → physical 8 m/s bound;
   - odometry dropped after a 28 s relocalisation gap → displacement applied with inflated noise.
   Both are generic fixes; no parameters were tuned on it.
6. **Added features:**
   - structural PF global init and recovery from the causal local map;
   - segment refinement (45 s windows, selected on development);
   - flagged coverage completion;
   - as-built discrepancy analysis;
   - camera-rigid object filter (removed 32 operator/rig observations on development).
7. **Viewer.** Headless Chromium hung with Vulkan flags; ANGLE/EGL works (Intel iGPU). A PF snapshot format bug gave HTTP 500 → fixed.

**Final state:** all three runs were re-processed with the final code (align, discrepancy, localize, report); objects re-run with the camera-rigid filter. Viewer validation ran on all runs (see RESULTS.md).

**Open / next**
- Causal dense (PanoVGGT on [k−2, k−1, k]) observations in the PF: global init/recovery on low-evidence floors (held-out #2 has 4–23 causal wall cells).
- OKVIS2-X (raw fisheye + IMU) as the alternative backend for hard lighting/motion.
- Pose-graph optimisation with plan constraints (beyond windowed similarity blending).
- Validation on a user recording.
