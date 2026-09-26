"""slam3d command line.

  slam3d doctor                              hardware / dependency diagnostics
  slam3d download-sample [--sequence S ...]  public Hilti-Trimble-Oxford 2026 sample (non-commercial license)
  slam3d build-driver                        build docker image + C++ stella driver
  slam3d run CONFIG [--stages a,b] [--force] process a recording
  slam3d viewer RUN_DIR [--port 8765]        launch the synchronized web viewer
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from slam3d.config import REPO

HF = "https://huggingface.co/datasets/Hilti-Research/hilti-trimble-slam-challenge-2026/resolve/main"
GH = "https://raw.githubusercontent.com/Hilti-Research/hilti-trimble-slam-challenge-2026/main"


def _curl(url, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {url} -> {dst}")
    subprocess.run(["curl", "-fL", "-C", "-", "--retry", "5", "-s", "-o", str(dst), url], check=True)


def download_sample(sequences, floor=None, with_bags=True):
    d = REPO / "data/hilti"
    for f in ["config/hilti_openvins/kalibr_imucam_chain.yaml", "config/hilti_openvins/kalibr_imu_chain.yaml",
              "config/hilti_openvins/mask_cam0.png", "config/hilti_openvins/mask_cam1.png", "SEQUENCES.md", "LICENSE"]:
        _curl(f"{GH}/{f}", d / "refs" / Path(f).name)
    _curl(f"{GH}/config/hilti_stella_vslam/orb_vocab.fbow", REPO / "third_party/stella/orb_vocab.fbow")
    _curl(f"{HF}/groundtruth/init_gt_poses.csv", d / "groundtruth/init_gt_poses.csv")
    floors = set()
    for s in sequences:
        fl, date, run = s.rsplit("_", 3)[0], s.rsplit("_", 3)[1], "_".join(s.rsplit("_", 3)[2:])
        floors.add(fl.replace("floor_", ""))
        _curl(f"{HF}/groundtruth/{s}.txt", d / f"groundtruth/{s}.txt")
        if with_bags:
            for name in ("metadata.yaml", "rosbag.db3"):
                _curl(f"{HF}/data/{fl}/{date}/{run}/rosbag/{name}", d / f"bags/{s}/{name}")
    og = {"1": "1OG", "2": "2OG", "3": "3OG", "4": "4OG", "5": "5OG", "6": "6OG", "7": "7OG", "EG": "EG", "UG1": "1UG"}
    for fl in floors:
        _curl(f"{HF}/floorplans/png_format/floor_{fl}.png", d / f"floorplans/floor_{fl}.png")
        _curl(f"{HF}/floorplans/binary_masks/masks_no-window/BuchsIT_{og[fl]}_mask_nowindows.png", d / f"floorplans/floor_{fl}_mask_nowindows.png")
        _curl(f"{HF}/floorplans/dxf_format/floor_{fl}.dxf", d / f"floorplans/floor_{fl}.dxf")
    print("done. License: CC BY-NC-SA (non-commercial). See data/hilti/refs/LICENSE")


def main(argv=None):
    p = argparse.ArgumentParser(prog="slam3d")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    d = sub.add_parser("download-sample")
    d.add_argument("--sequence", nargs="+", default=["floor_2_2025-12-03_run_1", "floor_2_2025-10-28_run_2"])
    d.add_argument("--no-bags", action="store_true")
    sub.add_parser("build-driver")
    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--stages", default=None, help="comma separated subset, e.g. slam,geometry,align")
    r.add_argument("--force", action="store_true")
    v = sub.add_parser("viewer")
    v.add_argument("run_dir")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--host", default="127.0.0.1")
    g = sub.add_parser("gaussians", help="fit an offline Gaussian appearance scene from an existing dense run")
    g.add_argument("run_dir")
    g.add_argument("--max-points", type=int, default=45000)
    g.add_argument("--views", type=int, default=12)
    g.add_argument("--steps", type=int, default=120)
    g.add_argument("--size", type=int, default=96)
    a = p.parse_args(argv)

    if a.cmd == "gaussians":
        from slam3d.dense.gaussian import run_gaussians
        try:
            run_gaussians(Path(a.run_dir), a.max_points, a.views, a.steps, a.size)
        except Exception as exc:
            import json
            status = Path(a.run_dir) / "gaussians/status.json"
            if status.parent.exists():
                status.write_text(json.dumps({"state": "failed", "error": str(exc)}, indent=2))
            raise
    elif a.cmd == "doctor":
        from slam3d.doctor import main as doctor

        doctor(REPO)
    elif a.cmd == "download-sample":
        download_sample(a.sequence, with_bags=not a.no_bags)
    elif a.cmd == "build-driver":
        subprocess.run(["docker", "build", "-t", "slam3d-stella:e445b54", str(REPO / "docker/stella")], check=True)
        from slam3d.slam import stella

        print("driver:", stella.build_driver(REPO / "outputs/driver_build.log"))
    elif a.cmd == "run":
        from slam3d.config import load_config
        from slam3d.pipeline import run

        cfg = load_config(a.config)
        rdir = run(cfg, a.stages.split(",") if a.stages else None, a.force)
        print(f"outputs: {rdir}")
    elif a.cmd == "viewer":
        from slam3d.viewer.server import serve

        serve(Path(a.run_dir).resolve(), a.host, a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
