"""Ingest a stitched equirectangular 360 video (e.g. Insta360/GoPro/Ricoh export).

* Frame timestamps come from each frame's presentation timestamp (ffprobe best_effort_timestamp),
  so variable frame rate and dropped frames are preserved; an optional `time_offset_s` maps video
  time to an external clock (e.g. IMU).
* Spherical metadata (Google spherical-video V2 / "projection=equirectangular") is reported; a 2:1
  aspect ratio is required.
* Panorama orientation is kept as exported. If the camera app stabilised / horizon-levelled the
  video, the SLAM gravity step can use `level_horizon`; otherwise provide IMU or rely on structure.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


def probe(video: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", video],
                         capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    side = vs.get("side_data_list", [])
    spherical = next((sd for sd in side if "spherical" in sd.get("side_data_type", "").lower()), None)
    num, den = (int(x) for x in vs.get("avg_frame_rate", "0/1").split("/"))
    return {
        "width": int(vs["width"]), "height": int(vs["height"]),
        "fps_avg": num / den if den else None,
        "duration_s": float(info["format"].get("duration", "nan")),
        "codec": vs.get("codec_name"),
        "spherical_side_data": spherical,
        "is_2to1": int(vs["width"]) == 2 * int(vs["height"]),
        "tags": {**info["format"].get("tags", {}), **vs.get("tags", {})},
    }


def frame_timestamps(video: str) -> np.ndarray:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "frame=best_effort_timestamp_time", "-of", "csv=p=0", video],
                         capture_output=True, text=True, check=True).stdout
    return np.array([float(x.split(",")[0]) for x in out.split() if x and x.split(",")[0] not in ("N/A", "")])


def ingest_video(video: str, out_dir: str, width: int = 1920, every_n: int = 1, start_s: float = 0.0,
                 end_s: float | None = None, time_offset_s: float = 0.0, quality: int = 3, log=print) -> dict:
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    info = probe(video)
    if not info["is_2to1"]:
        raise ValueError(f"{video} is {info['width']}x{info['height']}; an equirectangular 2:1 video is required")
    ts = frame_timestamps(video)
    sel = np.nonzero((ts >= start_s) & ((ts <= end_s) if end_s is not None else True))[0][::every_n]
    height = width // 2
    # decode all frames in range once; select by index to keep exact timestamps
    vf = f"select='not(mod(n\\,{every_n}))',scale={width}:{height}:flags=area" if every_n > 1 else f"scale={width}:{height}:flags=area"
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start_s > 0:
        cmd += ["-ss", f"{ts[sel[0]]:.6f}"]
    cmd += ["-i", video]
    if end_s is not None:
        cmd += ["-to", f"{end_s - (ts[sel[0]] if start_s > 0 else 0):.6f}"]
    cmd += ["-vf", vf, "-vsync", "0", "-q:v", str(quality), "-start_number", "0", str(out / "frames/%06d.jpg")]
    log("[ingest] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    files = sorted((out / "frames").glob("*.jpg"))
    n = min(len(files), len(sel))
    if len(files) != len(sel):
        log(f"[ingest] warning: decoded {len(files)} frames, expected {len(sel)}; using first {n}")
    with open(out / "frames.csv", "w") as f:
        f.write("idx,timestamp_s,path,video_pts_s\n")
        for i in range(n):
            f.write(f"{i},{ts[sel[i]] + time_offset_s:.6f},frames/{i:06d}.jpg,{ts[sel[i]]:.6f}\n")
    meta = {"projection": "equirectangular", "width": width, "height": height, "n_frames": n, "source_video": str(video),
            "probe": info, "every_n": every_n, "start_s": start_s, "end_s": end_s, "time_offset_s": time_offset_s,
            "timestamp_source": "ffprobe best_effort_timestamp_time (+offset)",
            "pano_frame": "as exported: x right, y down, z = centre column; central-camera approximation of stitched video"}
    (out / "stitch.json").write_text(json.dumps(meta, indent=2))
    return meta
