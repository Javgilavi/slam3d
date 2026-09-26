# Frames, units, timestamps and conventions

All machine-readable outputs restate their frame in their header or JSON (`frame`, `units`, `status`).

## Panorama camera (`pano`)
- **Axes:** x right, y down, z forward, where forward is the centre column of the equirect image.
- **Pixel ↔ bearing:**
  - Pixel (u, v) is a pixel centre.
  - `lon = ((u+0.5)/W − 0.5)·2π` (positive = right), `lat = ((v+0.5)/H − 0.5)·π` (positive = down).
  - `bearing = (cos lat sin lon, sin lat, cos lat cos lon)`.
- **Other tools:**
  - stella_vslam uses the same axes with u/W (no +0.5); converted in `slam/postprocess.py` and `semantics/pipeline.py`.
  - PanoVGGT `local_points` use exactly this convention (verified in its `_get_direction_vectors`).
- **Depth:** radial range ‖p‖, never perspective z (`geometry/sphere.py: radial_to_zdepth` exists only for explicit conversions).
- **Seam:** u wraps modulo W (`sphere.wrap_u`, `sample_equirect` pads with BORDER_WRAP). Perspective views are pure rotations about the shared panorama centre (`equirect_to_perspective` returns the exact pixel map back to the panorama).
- **Hilti stitching** (`ingest/stitch.py`):
  - The origin is the cam0 optical centre; `R_cam0_pano = diag(−1, −1, 1)` because the sensors are mounted upside down.
  - cam1 rays use Kalibr `T_cam1_cam0` and meet a 3 m sphere. The lens baseline is 4.0 cm, so seam parallax is a documented central-camera approximation.
  - Verified: IMU gravity vs GT gravity in the pano frame agree to 1.16° median over 30 windows; IMU↔GT time offset −4 ms (Kalibr: −6.6 ms).

## Fisheye (`cam0`, `cam1`)
- Kalibr frames (x right, y down, z optical axis). Models: KB4 (equidistant) and EUCM `[alpha, beta, fx, fy, cx, cy]`. Released KB4 vs EUCM agree to <1 px median inside 95° (test).

## SLAM world (`slam`)
- stella_vslam map frame, **arbitrary scale and orientation**. Trajectories are `T_slam_pano` (camera-to-world), TUM `t tx ty tz qx qy qz qw`.
- `slam/online_poses.txt`: pose returned when each frame was processed (causal). `slam/frame_trajectory.txt`: after loop closure / global BA (non-causal).

## Gravity-aligned (`grav`)
- `p_grav = R_grav_slam · p_slam − origin_offset` (first keyframe at the origin, +z up), still in SLAM units.
- Gravity comes from the IMU when available (quasi-static accelerometer samples), otherwise from a level-horizon assumption for stabilised consumer videos. The method and angular spread are recorded in `geometry/geometry.json`.

## Floor plan (`plan`)
- **Units and axes:** metres, x right, y up, z up out of the drawing. The **centre of the bottom-left pixel** of the source raster is (0, 0) (Hilti convention): `x = col·res`, `y = (H−1−row)·res`.
- **Hilti GT:** expressed as `T_plan_cam0` in this frame.
- **Registration:** `plan_xy = S · [grav_xy, 1]` with `S = [[s·cosθ, −s·sinθ, tx], [s·sinθ, s·cosθ, ty]]`, and `z_plan = s·(z_grav − floor_z_grav)` (height above the estimated floor), stored in `align/alignment.json`.
- **Rotation:** `R_plan_pano = Rz(θ) · R_grav_pano`.
- DXF: plan origin = DXF point `dxf_origin_offset_m` (recorded).

## Scale status (always reported)
| status | meaning |
|---|---|
| `arbitrary` | monocular SLAM units only |
| `estimated (IMU accelerations, coarse prior ~ +-25%)` | from IMU vs SLAM accelerations; used as a search prior |
| `estimated (camera-height prior)` | assumed lens height and floor layer detected below the cameras |
| `externally calibrated by plan registration (similarity)` | from floor-plan registration (`align/alignment.json: scale`) |

## Timestamps
- **Seconds:** all timestamps are float seconds on the recording clock.
- **Hilti bags:** header stamps, shifted by +1e4 s by the dataset; image stamp = cam0 header stamp.
- **User videos:** ffprobe `best_effort_timestamp_time` + `time_offset_s`.
- **Frame index:** `ingest/frames.csv` row i ↔ `viewer/pano.mp4` frame i ↔ `frames/%06d.jpg`.
- **Kalibr:** `t_imu = t_cam + timeshift_cam_imu`.
- **Associations:** trajectory ↔ GT nearest timestamp within 20 ms (one-to-one); interpolation uses slerp/linear with a max gap of 0.2–0.25 s.
