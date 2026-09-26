# Showcase asset notes

The [10-second film](showcase.mp4) is a 1280 × 720, 24 fps MP4 with an original synthesized soundtrack. The [animated preview](showcase-preview.gif) links to it from the project README.

| Time | Visual | Status |
|---|---|---|
| 0–2.3 s | Perspective conversion of the public Hilti walkthrough panorama | Real sample footage |
| 2.3–4.3 s | Screenshot of the synchronized slam3d viewer | Current prototype |
| 4.3–6.6 s | `vision-site.png` | Illustrative concept |
| 6.6–8.7 s | `vision-dashboard.png` | Illustrative concept |
| 8.7–10 s | Brand end card over `vision-site.png` | Illustrative concept |

The first two shots and the existing `viewer.png` and `gaussian-world.png` screenshots derive from the [Hilti–Trimble–Oxford 2026 dataset](https://huggingface.co/datasets/Hilti-Research/hilti-trimble-slam-challenge-2026) by Centanni et al. The dataset is [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). The dataset-derived screenshots and composed video/GIF are shared under those noncommercial share-alike terms. The two `vision-*.png` source frames were generated for this promo and portray a possible product direction rather than measured reconstruction or released UI.

To rebuild locally after processing the sample sequence, run `python scripts/build_showcase.py`. It reads `outputs/hilti_floor_2_2025-12-03_run_1/viewer/pano.mp4` for the opening shot when available. Without that video, it falls back to the committed viewer screenshot. The script uses ffmpeg, numpy, Pillow, and OpenCV.
