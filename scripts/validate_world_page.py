"""Validate the served Gaussian world page (/world) in a real GPU-accelerated browser.

Starts the slam3d viewer app in-process, opens /world in headless Chromium (ANGLE/EGL) and checks:
load time, Gaussians shown, WebGL renderer, console errors, measured render FPS (requestAnimationFrame count,
overview and walking), walk movement, follow-video playback, object selection/inspection, semantic colouring,
overview button. Screenshots are saved; browser image quality is NOT numerically compared (the browser renderer is
degree-0 colour with a global depth sort, unlike the gsplat renderer used for the held-out metrics).

usage: python scripts/validate_world_page.py outputs/<run> [--port 8830]
writes outputs/<run>/gaussians/world_validation.json and world_*.png
"""
from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np


def serve(rdir: Path, port: int):
    import uvicorn

    from slam3d.viewer.server import create_app

    srv = uvicorn.Server(uvicorn.Config(create_app(rdir), host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            return srv
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("server did not start")


FPS_JS = """ms => new Promise(res => { let n = 0; const t0 = performance.now();
  function f(now) { n++; if (now - t0 < ms) requestAnimationFrame(f); else res(n * 1000 / (now - t0)); }
  requestAnimationFrame(f); })"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--port", type=int, default=8830)
    a = ap.parse_args()
    rdir = Path(a.run_dir).resolve()
    out = rdir / "gaussians"
    srv = serve(rdir, a.port)
    from playwright.sync_api import sync_playwright

    res = {"run": rdir.name, "checks": {}, "console_errors": []}
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True, args=["--enable-gpu", "--ignore-gpu-blocklist", "--use-gl=angle", "--use-angle=gl-egl"])
        page = b.new_page(viewport={"width": 1440, "height": 900})
        page.on("console", lambda m: res["console_errors"].append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: res["console_errors"].append(str(e)))
        t0 = time.time()
        page.goto(f"http://127.0.0.1:{a.port}/world")
        page.wait_for_function("window.__world && window.__world.ready", timeout=180000)
        res["load_s"] = round(time.time() - t0, 2)
        res["gaussians_shown"] = page.evaluate("window.__world.gaussians")
        res["webgl_renderer"] = page.evaluate("""() => { const c = document.createElement('canvas').getContext('webgl2');
            const d = c && c.getExtension('WEBGL_debug_renderer_info'); return d ? c.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'unknown'; }""")
        res["checks"]["loaded"] = bool(res["gaussians_shown"])
        page.wait_for_timeout(1500)
        res["fps_overview"] = round(page.evaluate(FPS_JS, 3000), 1)
        page.screenshot(path=str(out / "world_overview.png"))

        # recorded pose (video-synchronised viewpoint)
        page.click("#jump")
        page.wait_for_timeout(1200)
        page.screenshot(path=str(out / "world_recorded_pose.png"))

        # walk mode: movement + fps while moving
        page.click("#walk")
        before = page.evaluate("window.__world.camera.position.toArray()")
        page.keyboard.down("w")
        fps_walk = page.evaluate(FPS_JS, 1500)
        page.keyboard.up("w")
        after = page.evaluate("window.__world.camera.position.toArray()")
        res["fps_walking"] = round(fps_walk, 1)
        res["walk_displacement_m"] = float(np.linalg.norm(np.array(after) - np.array(before)))
        res["checks"]["walk_moves_camera"] = res["walk_displacement_m"] > 0.2
        page.screenshot(path=str(out / "world_walk.png"))
        page.click("#walk")

        # follow video + play: time must advance and the camera follows the recorded path
        page.click("#follow")
        page.click("#play")
        tp0 = page.evaluate("window.__world.time")
        page.wait_for_timeout(2000)
        tp1 = page.evaluate("window.__world.time")
        page.click("#play")
        res["playback_advance_s"] = round(tp1 - tp0, 2)
        res["checks"]["playback_advances"] = tp1 > tp0 + 1.0
        page.screenshot(path=str(out / "world_follow.png"))

        # objects + semantic colouring
        n_obj = page.evaluate("document.querySelectorAll('#objects option').length - 1")
        res["objects_listed"] = n_obj
        if n_obj > 0:
            page.select_option("#objects", index=1)
            page.wait_for_timeout(300)
            res["checks"]["object_inspector"] = len(page.locator("#inspect").inner_text()) > 0
        page.select_option("#appearance", "semantic")
        page.wait_for_timeout(800)
        page.screenshot(path=str(out / "world_semantic.png"))
        page.select_option("#appearance", "rgb")
        page.click("#home")
        page.wait_for_timeout(800)
        res["checks"]["no_console_errors"] = len(res["console_errors"]) == 0
        b.close()
    srv.should_exit = True
    res["all_passed"] = all(res["checks"].values())
    res["note"] = "FPS measured with requestAnimationFrame on the named renderer; not reconstruction throughput."
    (out / "world_validation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
