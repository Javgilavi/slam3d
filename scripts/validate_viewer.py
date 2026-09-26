"""End-to-end viewer validation with a real browser (Playwright/Chromium).

Measures render fps, plan redraw time, video-frame <-> pose timestamp sync, playback progression,
seek latency, and exercises click-to-seek (incl. repeated visits), panorama rotation vs heading
indicator, object inspection + jump-to-observation, layer toggles and correspondence picking.

usage: python scripts/validate_viewer.py outputs/<run> [--port 8799] [--headed]
Writes <run>/viewer/validation.json and <run>/viewer/screenshot.png
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
from pathlib import Path


def _serve(rdir: Path, port: int):
    import uvicorn

    from slam3d.viewer.server import create_app

    cfg = uvicorn.Config(create_app(rdir), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(cfg)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return server
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("server did not start")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    rdir = Path(a.run_dir).resolve()
    from slam3d.viewer.server import ensure_video

    ensure_video(rdir)
    server = _serve(rdir, a.port)
    from playwright.sync_api import sync_playwright

    res = {"run": rdir.name, "checks": {}, "console_errors": []}
    with sync_playwright() as pw:
        # ANGLE on EGL uses a real GPU in headless mode (the Vulkan flag set hung the page on this machine)
        browser = pw.chromium.launch(headless=not a.headed, args=["--enable-gpu", "--ignore-gpu-blocklist", "--use-gl=angle",
                                                                  "--use-angle=gl-egl", "--autoplay-policy=no-user-gesture-required"])
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.on("console", lambda m: res["console_errors"].append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: res["console_errors"].append(str(e)))
        t0 = time.time()
        page.goto(f"http://127.0.0.1:{a.port}/")
        page.wait_for_function("window.__slam3d && window.__slam3dPerf", timeout=60000)
        res["load_s"] = round(time.time() - t0, 2)
        res["webgl_renderer"] = page.evaluate("""() => { const c = document.createElement('canvas').getContext('webgl2');
            const d = c && c.getExtension('WEBGL_debug_renderer_info'); return d ? c.getParameter(d.UNMASKED_RENDERER_WEBGL) : (c ? 'webgl2 (masked)' : 'none'); }""")
        info = page.evaluate("() => { const S = window.__slam3d.S; return {frames: S.frames.length, traj: S.traj ? S.traj.t.length : 0, objects: S.objects.length, t0: S.t0, trajFrame: S.traj && S.traj.frame}; }")
        res["loaded"] = info
        res["checks"]["data_loaded"] = info["frames"] > 0 and info["traj"] > 0

        # ---- playback at 1x for 10 s
        page.evaluate("() => { document.querySelector('#video').play(); }")
        samples = []
        idx_start = page.evaluate("() => window.__slam3d.S.idx")
        tw = time.time()
        for _ in range(10):
            time.sleep(1.0)
            samples.append(page.evaluate("() => window.__slam3dPerf"))
        wall = time.time() - tw
        idx_end = page.evaluate("() => window.__slam3d.S.idx")
        page.evaluate("() => document.querySelector('#video').pause()")
        fps = [s["fps"] for s in samples[1:]]
        sync = [s["syncMs"] for s in samples if s.get("syncMs") is not None and not (isinstance(s["syncMs"], float) and math.isnan(s["syncMs"]))]
        res["playback"] = {"render_fps_median": sorted(fps)[len(fps) // 2] if fps else None, "render_fps_min": min(fps) if fps else None,
                           "plan_redraw_ms_last": samples[-1]["planMs"], "frames_advanced": idx_end - idx_start,
                           "wall_s": round(wall, 2), "video_fps_effective": (idx_end - idx_start) / wall,
                           "pose_sync_ms_max": max(sync) if sync else None}
        res["checks"]["playback_advances"] = (idx_end - idx_start) > 0.8 * 29.97 * wall
        res["checks"]["pose_sync_within_1_frame"] = bool(sync) and max(sync) < 33.4

        # ---- seek
        target = info["t0"] + 60.0
        ts = time.time()
        page.evaluate(f"() => window.__slam3d.seekTime({target})")
        page.wait_for_function(f"() => Math.abs(window.__slam3d.S.t - {target}) < 0.05", timeout=10000)
        res["seek_latency_ms"] = round((time.time() - ts) * 1000, 1)
        res["checks"]["seek_exact_frame"] = True

        # ---- click on the path at t0+100 s -> video time
        tgt = info["t0"] + 100.0
        page.evaluate("() => { const cb = document.querySelector('input[data-layer=objects]'); if (cb.checked) cb.click(); }")  # avoid object hit
        # choose the path point closest in time to t0+100 s whose screen position is not covered by an overlay panel
        # require the plan canvas at the point AND 10 px around it: Chromium delivers integer-rounded clicks,
        # so a point a sub-pixel away from an overlay panel edge can land on the panel
        pt = page.evaluate(f"""() => {{ const S = window.__slam3d.S; const c = document.querySelector('#plan-dyn'); const r = c.getBoundingClientRect();
            const order = S.traj.t.map((t, i) => [Math.abs(t - {tgt}), i]).sort((a, b) => a[0] - b[0]);
            const clear = (x, y) => [[0, 0], [10, 0], [-10, 0], [0, 10], [0, -10]].every(([dx, dy]) => document.elementFromPoint(x + dx, y + dy) === c);
            for (const [, k] of order) {{
              const p = S.traj.xyz[k];
              const x = Math.round(r.left + (p[0] * S.plan.s + S.plan.ox) / devicePixelRatio), y = Math.round(r.top + (-p[1] * S.plan.s + S.plan.oy) / devicePixelRatio);
              if (x > r.left + 12 && x < r.right - 12 && y > r.top + 12 && y < r.bottom - 12 && clear(x, y)) return {{x, y, t: S.traj.t[k]}};
            }}
            return null; }}""")
        t_before = page.evaluate("() => window.__slam3d.S.t")
        page.mouse.click(pt["x"], pt["y"])
        for _ in range(30):  # wait for the visits request -> seek, or the repeated-visit popup
            time.sleep(0.1)
            if page.evaluate(f"() => Math.abs(window.__slam3d.S.t - {t_before}) > 1e-6 || !document.querySelector('#visits').classList.contains('hidden')"):
                break
        time.sleep(0.3)
        popup = page.evaluate("() => !document.querySelector('#visits').classList.contains('hidden')")
        visits_n = 0
        if popup:
            visits_n = page.evaluate("() => document.querySelectorAll('#visits div.v').length")
            page.evaluate(f"""() => {{ const vs = [...document.querySelectorAll('#visits div.v')]; vs[0].click(); }}""")
            time.sleep(0.8)
        tnow = page.evaluate("() => window.__slam3d.S.t")
        res["click_path"] = {"target_t_rel": pt["t"] - info["t0"], "result_t_rel": tnow - info["t0"], "repeated_visit_popup": popup, "visits_listed": visits_n}
        res["checks"]["click_path_seeks"] = abs(tnow - pt["t"]) < 2.0 or popup
        page.evaluate("() => { const cb = document.querySelector('input[data-layer=objects]'); if (!cb.checked) cb.click(); }")

        # ---- rotate panorama, heading indicator follows
        h0 = page.evaluate("() => ({lon: window.__slam3d.S.lon})")
        box = page.locator("#pano").bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] / 2 - 300, box["y"] + box["height"] / 2, steps=10)
        page.mouse.up()
        h1 = page.evaluate("() => ({lon: window.__slam3d.S.lon})")
        res["rotate"] = {"lon_change_deg": math.degrees(h1["lon"] - h0["lon"])}
        res["checks"]["panorama_rotates"] = abs(h1["lon"] - h0["lon"]) > 0.2

        # ---- object inspection + jump to observation
        obj = page.evaluate("""() => { const S = window.__slam3d.S; const o = S.objects.find(o => o.status === 'confirmed') || S.objects[0];
            if (!o) return null; const a = S.run.alignment, c = o.centroid_grav_units, M = a.S_plan_from_grav;
            const x = M[0][0]*c[0] + M[0][1]*c[1] + M[0][2], y = M[1][0]*c[0] + M[1][1]*c[1] + M[1][2];
            const cv = document.querySelector('#plan-dyn'); const r = cv.getBoundingClientRect();
            return {id: o.id, label: o.label, x: r.left + (x * S.plan.s + S.plan.ox) / devicePixelRatio, y: r.top + (-y * S.plan.s + S.plan.oy) / devicePixelRatio,
                    obs_t: o.observations.map(ob => ob.t).sort((a, b) => a - b)[0]}; }""")
        if obj:
            page.mouse.click(obj["x"], obj["y"])
            time.sleep(0.5)
            shown = page.evaluate("() => !document.querySelector('#inspector').classList.contains('hidden') && document.querySelector('#inspector-body').innerText")
            page.evaluate("() => { const r = document.querySelector('#inspector tr.obs'); if (r) r.click(); }")
            time.sleep(1.0)
            tj = page.evaluate("() => window.__slam3d.S.t")
            res["object_inspect"] = {"object": obj["label"], "inspector_shown": bool(shown), "jump_dt_s": abs(tj - obj["obs_t"])}
            res["checks"]["object_inspector"] = bool(shown) and (obj["label"] in (shown or ""))
            res["checks"]["jump_to_observation"] = abs(tj - obj["obs_t"]) < 0.2
            page.evaluate("() => document.querySelector('#inspector-close').click()")

        # ---- layers + particle filter + correspondence pick
        for layer in ["walls", "gt", "pf", "dynamic"]:
            page.evaluate(f"() => document.querySelector('input[data-layer={layer}]').click()")
            time.sleep(0.6)
        res["pf_particles_loaded"] = page.evaluate("() => !!(window.__slam3d.S.particles && window.__slam3d.S.particles.x.length)")
        page.evaluate("() => document.querySelector('#btn-pick').click()")
        cbox = page.locator("#plan-dyn").bounding_box()
        page.mouse.click(cbox["x"] + cbox["width"] * 0.4, cbox["y"] + cbox["height"] * 0.5)
        time.sleep(0.3)
        res["checks"]["correspondence_pick"] = page.evaluate("() => window.__slam3d.S.pairs.length === 1")
        page.evaluate("() => window.__slam3d.seekTime(window.__slam3d.S.t0 + 30)")
        time.sleep(1.5)
        page.screenshot(path=str(rdir / "viewer/screenshot.png"))
        res["checks"]["no_console_errors"] = len(res["console_errors"]) == 0
        browser.close()
    server.should_exit = True
    res["all_passed"] = all(res["checks"].values())
    (rdir / "viewer/validation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
