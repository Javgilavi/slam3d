#!/usr/bin/env bash
# slam3d bootstrap: isolated Python env, Dockerised stella_vslam, C++ driver, third-party code and weights.
# No sudo, no driver or system changes. Re-runnable (skips what exists).
#   scripts/bootstrap.sh            # env + docker + driver + PanoVGGT/YOLOE weights
#   scripts/bootstrap.sh --sample   # additionally download the public Hilti sample (~6.7 GB, CC BY-NC-SA)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 1; }
command -v docker >/dev/null || { echo "docker is required for the SLAM backend"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg is required"; exit 1; }

echo "== python env (.venv, pinned by requirements.lock.txt)"
[ -d .venv ] || uv venv --python 3.12 .venv
. .venv/bin/activate
if [ -f requirements.lock.txt ]; then
  uv pip install --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple -r requirements.lock.txt
else
  uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1
  uv pip install numpy scipy opencv-python-headless pillow rosbags pyyaml tqdm msgpack psutil nvidia-ml-py matplotlib \
    scikit-image shapely ezdxf pymupdf plyfile fastapi "uvicorn[standard]" pytest einops huggingface_hub safetensors \
    open3d ultralytics omegaconf trimesh kornia regex playwright
fi
uv pip install -e . --no-deps
python -m playwright install chromium || echo "playwright browser install failed (viewer e2e test only)"

echo "== stella_vslam docker image + driver"
docker image inspect slam3d-stella:e445b54 >/dev/null 2>&1 || docker build -t slam3d-stella:e445b54 docker/stella
[ -x cpp/stella_driver/build/slam3d_stella_driver ] || python -c "from slam3d.slam import stella; print(stella.build_driver())"
mkdir -p third_party/stella
[ -f third_party/stella/orb_vocab.fbow ] || curl -fL -o third_party/stella/orb_vocab.fbow \
  https://raw.githubusercontent.com/Hilti-Research/hilti-trimble-slam-challenge-2026/main/config/hilti_stella_vslam/orb_vocab.fbow

echo "== PanoVGGT (MIT code; weights from Hugging Face, 3.9 GB)"
[ -d third_party/panovggt ] || { git clone -q https://github.com/YijingGuo-June/PanoVGGT.git third_party/panovggt && git -C third_party/panovggt checkout -q 556bb7d2ec2d02bd3ee4ed535542e74290ba22cf; }
mkdir -p third_party/weights/panovggt third_party/weights/yoloe
[ -f third_party/weights/panovggt/model.pt ] || curl -fL -C - -o third_party/weights/panovggt/model.pt https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt

echo "== YOLOE weights (downloaded by ultralytics into third_party/weights/yoloe)"
(cd third_party/weights/yoloe && YOLO_AUTOINSTALL=False python -c "from ultralytics import YOLOE; m=YOLOE('yoloe-11s-seg.pt'); m.set_classes(['person'], m.get_text_pe(['person']))")

echo "== three.js for the offline viewer"
mkdir -p viewer/static/vendor
[ -f viewer/static/vendor/three.module.js ] || curl -fL -o viewer/static/vendor/three.module.js https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js
[ -f viewer/static/vendor/OrbitControls.js ] || curl -fL -o viewer/static/vendor/OrbitControls.js https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js

if [[ "${1:-}" == "--sample" ]]; then
  echo "== public sample data (Hilti-Trimble-Oxford 2026, CC BY-NC-SA)"
  slam3d download-sample
fi
slam3d doctor
