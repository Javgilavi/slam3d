"""`slam3d doctor`: hardware and dependency diagnostics (no changes are made to the system)."""
from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
from pathlib import Path


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def diagnose(repo: Path) -> dict:
    import psutil

    rep = {
        "os": platform.platform(), "python": platform.python_version(),
        "cpu_threads": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "disk_free_gb": round(shutil.disk_usage(repo).free / 2**30, 1),
        "gpu": _run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version", "--format=csv,noheader"]),
        "ffmpeg": _run(["ffmpeg", "-version"]).split("\n")[0],
        "docker": _run(["docker", "--version"]),
    }
    pk = {}
    for m in ["numpy", "scipy", "cv2", "torch", "torchvision", "ultralytics", "rosbags", "open3d", "ezdxf", "fitz", "fastapi"]:
        try:
            mod = importlib.import_module(m)
            pk[m] = getattr(mod, "__version__", "ok")
        except Exception as e:  # noqa: BLE001
            pk[m] = f"MISSING ({type(e).__name__})"
    rep["python_packages"] = pk
    try:
        import torch

        rep["torch_cuda"] = {"available": torch.cuda.is_available(),
                             "cuda": torch.version.cuda,
                             "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                             "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else None}
    except Exception as e:  # noqa: BLE001
        rep["torch_cuda"] = f"ERR {e}"
    img = _run(["docker", "images", "-q", "slam3d-stella:e445b54"])
    rep["stella_image"] = bool(img) and not img.startswith("ERR")
    rep["stella_driver_built"] = (repo / "cpp/stella_driver/build/slam3d_stella_driver").exists()
    rep["orb_vocab"] = (repo / "third_party/stella/orb_vocab.fbow").exists()
    rep["panovggt_weights"] = (repo / "third_party/weights/panovggt/model.pt").exists()
    return rep


def main(repo: Path):
    rep = diagnose(repo)
    print(json.dumps(rep, indent=2))
    problems = [k for k, v in rep["python_packages"].items() if str(v).startswith("MISSING")]
    if not rep["stella_image"]:
        problems.append("docker image slam3d-stella (run scripts/bootstrap.sh)")
    if not rep["stella_driver_built"]:
        problems.append("stella driver (run: slam3d build-driver)")
    print("\nstatus:", "OK" if not problems else "missing -> " + ", ".join(problems))
    return rep
