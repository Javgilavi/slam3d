"""Build the 10-second README promo from project imagery and optional sample footage.

Requires ffmpeg, numpy, Pillow, and OpenCV. With a processed Hilti sample, the
opening shot uses outputs/<run>/viewer/pano.mp4; otherwise it uses the committed
viewer screenshot. The published video was rendered with the sample footage.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
PANO = ROOT / "outputs/hilti_floor_2_2025-12-03_run_1/viewer/pano.mp4"
SIZE = (1280, 720)
FPS = 24
SECONDS = 10
FRAMES = FPS * SECONDS
CYAN = (51, 229, 224)
WHITE = (246, 250, 252)
FONT = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"
FONT_MED = "/usr/share/fonts/opentype/inter/InterDisplay-Medium.otf"


def font(size: int, medium: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_MED if medium else FONT
    if not Path(path).exists():
        path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size)


def ease(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def still(path: Path, u: float, zoom: float = 0.055) -> Image.Image:
    source = Image.open(path).convert("RGB")
    width, height = SIZE
    scale = max(width / source.width, height / source.height) * (1.035 + zoom * u)
    scaled = source.resize((math.ceil(source.width * scale), math.ceil(source.height * scale)), Image.Resampling.LANCZOS)
    x = int((scaled.width - width) * (0.35 + 0.25 * u))
    y = int((scaled.height - height) * (0.45 + 0.08 * u))
    return scaled.crop((x, y, x + width, y + height))


def real_frames(tmp: Path) -> list[Image.Image]:
    if not PANO.exists():
        return []
    clip = tmp / "real-view.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", "29.8", "-t", "2.5", "-i", str(PANO),
            "-vf", "v360=input=equirect:output=flat:yaw=0:h_fov=100:v_fov=65,scale=1280:720,fps=24",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", str(clip),
        ],
        check=True,
    )
    cap = cv2.VideoCapture(str(clip))
    frames: list[Image.Image] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def overlay() -> Image.Image:
    w, h = SIZE
    x = np.arange(w, dtype=np.float32)[None, :]
    y = np.arange(h, dtype=np.float32)[:, None]
    left = 205 * (1 - x / w) ** 2.3
    bottom = 80 * (y / h) ** 3
    alpha = np.clip(left + bottom, 0, 228).astype(np.uint8)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = (4, 12, 19)
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, "RGBA")


SHADE = overlay()
SCENES = [
    dict(start=0.0, end=2.3, kind="real", tag="REAL WALKTHROUGH", title="See the site.", sub="A 360° record of every step"),
    dict(start=2.3, end=4.3, kind="viewer", tag="CURRENT PROTOTYPE", title="Map the change.", sub="Video  •  floor plan  •  3D, in sync"),
    dict(start=4.3, end=6.6, kind="site", tag="VISION CONCEPT", title="Reveal what matters.", sub="Spatial context for every decision"),
    dict(start=6.6, end=8.7, kind="dashboard", tag="VISION CONCEPT", title="One view. Every decision.", sub="A construction intelligence workspace"),
    dict(start=8.7, end=10.0, kind="outro", tag="VISION CONCEPT", title="slam3d", sub="From walkthrough to foresight."),
]


def background(scene: dict, t: float, footage: list[Image.Image]) -> Image.Image:
    u = max(0.0, min(1.0, (t - scene["start"]) / (scene["end"] - scene["start"])))
    kind = scene["kind"]
    if kind == "real" and footage:
        return footage[min(int(u * (len(footage) - 1)), len(footage) - 1)].copy()
    paths = {
        "real": ASSETS / "viewer.png",
        "viewer": ASSETS / "viewer.png",
        "site": ASSETS / "vision-site.png",
        "dashboard": ASSETS / "vision-dashboard.png",
        "outro": ASSETS / "vision-site.png",
    }
    return still(paths[kind], u)


def scene_index(t: float) -> int:
    return next((i for i, s in enumerate(SCENES) if t < s["end"]), len(SCENES) - 1)


def draw_frame(n: int, footage: list[Image.Image]) -> Image.Image:
    t = n / FPS
    idx = scene_index(t)
    scene = SCENES[idx]
    bg = background(scene, t, footage)
    if idx and t - scene["start"] < 0.28:
        previous = background(SCENES[idx - 1], scene["start"] - 0.01, footage)
        bg = Image.blend(previous, bg, ease((t - scene["start"]) / 0.28))
    image = bg.convert("RGBA")
    image.alpha_composite(SHADE)
    if scene["kind"] == "outro":
        image.alpha_composite(Image.new("RGBA", SIZE, (1, 9, 15, 122)))

    layer = Image.new("RGBA", SIZE)
    draw = ImageDraw.Draw(layer)
    w, h = SIZE
    draw.rectangle((0, 0, w, 5), fill=(*CYAN, 185))
    draw.rectangle((0, h - 5, int(w * min(1, (t + 0.05) / SECONDS)), h), fill=(*CYAN, 230))
    draw.text((70, 46), "slam3d", font=font(27), fill=(*WHITE, 240))
    draw.rectangle((70, 92, 152, 96), fill=(*CYAN, 250))

    entry = ease((t - scene["start"] - 0.12) / 0.40)
    exit_alpha = ease((scene["end"] - t) / 0.22) if idx < len(SCENES) - 1 else 1.0
    a = int(255 * entry * exit_alpha)
    if idx == 0:
        a = int(a * ease(t / 0.28))
    if idx == len(SCENES) - 1:
        title_size = 91
        title_y = 258
        sub_y = 375
    else:
        title_size = 62 if idx != 3 else 58
        title_y = 280
        sub_y = 367
    draw.rounded_rectangle((70, 203, 70 + 15 * len(scene["tag"]) + 29, 243), radius=8, fill=(11, 42, 48, int(a * 0.66)))
    draw.text((85, 214), scene["tag"], font=font(17, True), fill=(*CYAN, a))
    draw.text((70, title_y), scene["title"], font=font(title_size), fill=(*WHITE, a), stroke_width=1, stroke_fill=(2, 10, 15, int(a * 0.2)))
    draw.text((73, sub_y), scene["sub"], font=font(26, True), fill=(219, 238, 242, a))
    if idx >= 2:
        draw.text((70, 648), "Illustrative product vision", font=font(17, True), fill=(213, 229, 232, int(a * 0.74)))
    else:
        draw.text((70, 648), "Hilti–Trimble–Oxford 2026 sample", font=font(16, True), fill=(216, 231, 234, int(a * 0.76)))
    draw.text((w - 120, 646), f"0{idx + 1} / 05", font=font(16, True), fill=(213, 230, 234, 190))
    image.alpha_composite(layer)
    if idx == 0:
        image.alpha_composite(Image.new("RGBA", SIZE, (0, 0, 0, int(255 * (1 - ease(t / 0.25))))))
    if t > 9.72:
        image.alpha_composite(Image.new("RGBA", SIZE, (0, 0, 0, int(230 * ease((t - 9.72) / 0.28)))))
    return image.convert("RGB")


def soundtrack(path: Path) -> None:
    rate = 44100
    t = np.arange(rate * SECONDS, dtype=np.float64) / rate
    rng = np.random.default_rng(4302)
    audio = np.zeros_like(t)
    for hz, level in [(110.0, 0.12), (164.81, 0.08), (220.0, 0.07), (293.66, 0.04)]:
        audio += level * (np.sin(2 * np.pi * hz * t) + 0.2 * np.sin(2 * np.pi * hz * 2.01 * t))
    audio *= 0.69 + 0.31 * np.sin(2 * np.pi * 0.16 * t)
    for beat in np.arange(0.0, 10.0, 0.63):
        dt = t - beat
        mask = (dt >= 0) & (dt < 0.55)
        audio[mask] += 0.25 * np.sin(2 * np.pi * (92 * dt[mask] - 28 * dt[mask] ** 2)) * np.exp(-10 * dt[mask])
    for mark in [0.0, 2.3, 4.3, 6.6, 8.7]:
        dt = t - mark
        mask = (dt >= 0) & (dt < 0.65)
        audio[mask] += 0.065 * np.sin(2 * np.pi * 880 * dt[mask]) * np.exp(-8 * dt[mask])
        noise = rng.standard_normal(mask.sum())
        audio[mask] += 0.027 * noise * np.exp(-15 * dt[mask])
    envelope = np.minimum(1, t / 0.35) * np.minimum(1, (SECONDS - t) / 0.75)
    audio = np.clip(audio * envelope, -0.85, 0.85)
    left = audio * (0.96 + 0.04 * np.sin(2 * np.pi * 0.11 * t))
    right = audio * (0.96 - 0.04 * np.sin(2 * np.pi * 0.11 * t))
    pcm = (np.stack([left, right], axis=1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="slam3d-showcase-") as temp:
        tmp = Path(temp)
        footage = real_frames(tmp)
        silent = tmp / "silent.mp4"
        encoder = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", str(FPS), "-i", "-",
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
                str(silent),
            ],
            stdin=subprocess.PIPE,
        )
        assert encoder.stdin is not None
        for n in range(FRAMES):
            frame = draw_frame(n, footage)
            if n == 180:
                frame.save(ASSETS / "showcase-poster.jpg", quality=92)
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg video encode failed")
        audio = tmp / "soundtrack.wav"
        soundtrack(audio)
        video = ASSETS / "showcase.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(silent), "-i", str(audio), "-c:v", "copy",
                "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(video),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                "-vf", "fps=8,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                "-loop", "0", str(ASSETS / "showcase-preview.gif"),
            ],
            check=True,
        )
    print(f"Built {video} and README preview")


if __name__ == "__main__":
    main()
