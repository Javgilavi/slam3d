"""Local synchronized viewer: FastAPI backend serving one run directory + a static three.js frontend.

Recorded-path browsing only: the panorama shown is always a recorded frame; the 3D view is a sparse
reconstruction explorer and does not synthesise unseen surfaces.
"""
from __future__ import annotations

import io
import json
import subprocess
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from slam3d.config import REPO

STATIC = REPO / "viewer/static"


def _read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)


def ensure_video(rdir: Path, log=print) -> Path:
    """H.264 video of the stitched panoramas with a short GOP for fast seeking; frame i == frames.csv row i."""
    vid = rdir / "viewer/pano.mp4"
    if vid.exists():
        return vid
    vid.parent.mkdir(parents=True, exist_ok=True)
    t = np.genfromtxt(rdir / "ingest/frames.csv", delimiter=",", skip_header=1, usecols=(1,))
    fps = 1.0 / float(np.median(np.diff(t)))
    cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", f"{fps:.5f}", "-i", str(rdir / "ingest/frames/%06d.jpg"),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-g", "15", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(vid)]
    log("[viewer] encoding " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    (rdir / "viewer/pano.json").write_text(json.dumps({"fps_encoded": fps, "n_frames": int(len(t))}))
    return vid


def create_app(rdir: Path) -> FastAPI:
    app = FastAPI(title="slam3d viewer")
    state = {"realign": None}

    def j(path):
        p = rdir / path
        return json.loads(p.read_text()) if p.exists() else None

    @app.get("/api/run")
    def run_info():
        frames = np.genfromtxt(rdir / "ingest/frames.csv", delimiter=",", skip_header=1, usecols=(1,))
        pano = j("viewer/pano.json") or {}
        exps = sorted(p.parent.name for p in (rdir / "localize").glob("*/pf_track.csv")
                      if not p.parent.name.startswith("_")) if (rdir / "localize").exists() else []  # "_" = dev runs
        return {"run": rdir.name, "stitch": j("ingest/stitch.json"), "n_frames": int(len(frames)),
                "t0": float(frames[0]), "t_end": float(frames[-1]), "fps_encoded": pano.get("fps_encoded"),
                "has_video": (rdir / "viewer/pano.mp4").exists(), "plan": j("floorplan/plan.json"),
                "drawing": j("floorplan/drawing.json"), "alignment": j("align/alignment.json"),
                "alignment_eval": j("align/eval.json"), "geometry": j("geometry/geometry.json"),
                "pf_experiments": exps, "has_gt": (rdir / "viewer/gt_plan.json").exists() or bool(j("align/eval.json"))}

    @app.get("/api/status")
    def status():
        return j("status.json") or {}

    @app.get("/world")
    def gaussian_world():
        return FileResponse(STATIC / "world.html")

    @app.get("/api/gaussians")
    def gaussian_info():
        report = j("gaussians/report.json")
        if report is None:
            raise HTTPException(404, "Run slam3d gaussians RUN_DIR first")
        from slam3d.io.tum import read_tum
        t, T = read_tum(rdir / "geometry/trajectory_grav.tum")
        scale = report["metres_per_slam_unit"]
        report["trajectory"] = {"t": t.tolist(), "xyz": (T[:, :3, 3] * scale).tolist(), "R": T[:, :3, :3].tolist()}
        ft = np.atleast_1d(np.genfromtxt(rdir / "ingest/frames.csv", delimiter=",", skip_header=1, usecols=(1,)))
        report["frame_times"] = ft.tolist()
        return report

    @app.get("/api/gaussians.bin")
    def gaussian_binary():
        p = rdir / "gaussians/scene.bin"
        if not p.exists():
            raise HTTPException(404, "Gaussian scene has not been fitted")
        return FileResponse(p, media_type="application/octet-stream")

    @app.get("/api/gaussians.sh.bin")
    def gaussian_sh_binary():
        p = rdir / "gaussians/scene.sh.bin"
        if not p.exists():
            raise HTTPException(404, "View-dependent appearance is unavailable for this scene")
        return FileResponse(p, media_type="application/octet-stream")

    @app.get("/api/frames")
    def frames():
        t = np.genfromtxt(rdir / "ingest/frames.csv", delimiter=",", skip_header=1, usecols=(1,))
        return {"t": np.round(t, 6).tolist()}

    @app.get("/api/trajectory")
    def trajectory():
        from slam3d.io.tum import read_tum

        p = rdir / "align/trajectory_plan.tum"
        frame = "plan"
        if not p.exists():
            p, frame = rdir / "geometry/trajectory_grav.tum", "grav"
        if not p.exists():
            raise HTTPException(404, "no trajectory yet")
        t, T = read_tum(p)
        # per-frame heading basis: pano x and z axes projected on the ground plane
        return {"frame": frame, "t": np.round(t, 6).tolist(), "xyz": np.round(T[:, :3, 3], 4).tolist(),
                "xaxis": np.round(T[:, :2, 0], 4).tolist(), "zaxis": np.round(T[:, :2, 2], 4).tolist()}

    @app.get("/api/gt")
    def gt():
        cfg = rdir / "config_used.yaml"
        import yaml

        c = yaml.safe_load(cfg.read_text()) if cfg.exists() else {}
        g = (c.get("evaluation") or {}).get("gt_tum")
        if not g:
            return {"t": [], "xyz": []}
        from slam3d.io.tum import read_tum

        t, T = read_tum(REPO / g if not Path(g).is_absolute() else g)
        return {"t": np.round(t[::3], 4).tolist(), "xyz": np.round(T[::3, :3, 3], 3).tolist()}

    @app.get("/api/pointcloud.bin")
    def pointcloud(voxel: float = 0.05):
        from slam3d.io.ply import read_ply

        p = rdir / "align/landmarks_plan.ply"
        if not p.exists():
            p = rdir / "geometry/landmarks_grav.ply"
        xyz, rgb = read_ply(p)
        if voxel > 0:
            key = np.floor(xyz / voxel).astype(np.int64)
            _, idx = np.unique(key, axis=0, return_index=True)
            xyz, rgb = xyz[idx], rgb[idx]
        buf = io.BytesIO()
        buf.write(np.array([len(xyz)], "<u4").tobytes())
        buf.write(xyz.astype("<f4").tobytes())
        buf.write((rgb if rgb is not None else np.full((len(xyz), 3), 180, np.uint8)).astype(np.uint8).tobytes())
        return Response(buf.getvalue(), media_type="application/octet-stream")

    @app.get("/api/objects")
    def objects():
        objs = j("objects/objects.json") or []
        for o in objs:
            o["thumb"] = (rdir / f"objects/thumbs/{o['id']}.jpg").exists()
        return objs

    @app.get("/api/thumb/{oid}.jpg")
    def thumb(oid: int):
        p = rdir / f"objects/thumbs/{oid}.jpg"
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/api/wall_evidence")
    def walls():
        p = rdir / "align/wall_evidence_plan.csv"
        if not p.exists():
            return {"xy": [], "d": []}
        a = np.loadtxt(p, delimiter=",", comments="#").reshape(-1, 4)
        return {"xy": np.round(a[:, :2], 3).tolist(), "d": np.round(a[:, 3], 3).tolist()}

    @app.get("/api/pf/{exp}")
    def pf(exp: str):
        p = rdir / f"localize/{exp}/pf_track.csv"
        if not p.exists():
            raise HTTPException(404)
        a = _read_csv(p)
        ok = np.isfinite(a["x"])
        err = None
        pe = rdir / f"localize/{exp}/pf_error.csv"
        return {"t": a["t"][ok].round(4).tolist(), "x": a["x"][ok].round(3).tolist(), "y": a["y"][ok].round(3).tolist(),
                "std": np.maximum(a["std_x"][ok], a["std_y"][ok]).round(3).tolist(), "kind": a["kind"][ok].tolist(),
                "report": j(f"localize/{exp}/pf_report.json")}

    @app.get("/api/particles/{exp}")
    def particles(exp: str, t: float):
        p = rdir / f"localize/{exp}/pf_particles.npz"
        if not p.exists():
            raise HTTPException(404)
        z = np.load(p, allow_pickle=True)
        ts = z["t"]
        if len(ts) == 0:
            return {"x": [], "y": []}
        i = int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1))  # latest snapshot <= t (causal)
        x = np.asarray(z["x"][i], dtype=float)
        y = np.asarray(z["y"][i], dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)  # snapshots are NaN-padded
        return {"t": float(ts[i]), "x": np.round(x[keep], 2).tolist(), "y": np.round(y[keep], 2).tolist()}

    @app.get("/api/visits")
    def visits(x: float, y: float, radius: float = 1.0):
        """All separate passes of the recorded path near (x, y) -> list of (t, distance)."""
        from slam3d.io.tum import read_tum

        p = rdir / "align/trajectory_plan.tum"
        if not p.exists():
            return []
        t, T = read_tum(p)
        d = np.hypot(T[:, 0, 3] - x, T[:, 1, 3] - y)
        near = d < radius
        out, i = [], 0
        while i < len(t):
            if near[i]:
                k = i
                while k + 1 < len(t) and near[k + 1]:
                    k += 1
                b = i + int(np.argmin(d[i:k + 1]))
                out.append({"t": float(t[b]), "dist": float(d[b])})
                i = k + 1
            else:
                i += 1
        return sorted(out, key=lambda v: v["dist"])

    @app.post("/api/correspondences")
    async def save_corr(req: Request):
        body = await req.json()
        pairs = body.get("pairs", [])
        (rdir / "align").mkdir(exist_ok=True)
        (rdir / "align/correspondences.json").write_text(json.dumps({"pairs": pairs}, indent=2))
        return {"saved": len(pairs)}

    @app.post("/api/realign")
    def realign():
        if state["realign"] and state["realign"].is_alive():
            return {"state": "running"}

        def work():
            import yaml

            from slam3d.config import load_config
            from slam3d.pipeline import run

            cfg = load_config(rdir / "config_used.yaml")
            cfg.setdefault("align", {})["correspondences"] = str(rdir / "align/correspondences.json")
            cfg["run_name"] = rdir.name
            cfg["output_root"] = str(rdir.parent)
            tmp = rdir / "viewer/realign_config.yaml"
            tmp.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}))
            cfg["_config_path"] = str(tmp)
            try:
                run(cfg, ["align"], force=True)
            except Exception as e:  # noqa: BLE001  (failure recorded in status.json)
                print("realign failed", e)

        state["realign"] = threading.Thread(target=work, daemon=True)
        state["realign"].start()
        return {"state": "started"}

    @app.get("/api/frame/{idx}.jpg")
    def frame(idx: int):
        p = rdir / f"ingest/frames/{idx:06d}.jpg"
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/media/pano.mp4")
    def video():
        p = rdir / "viewer/pano.mp4"
        if not p.exists():
            raise HTTPException(404, "video not encoded")
        return FileResponse(p, media_type="video/mp4")

    @app.get("/media/plan_structure.png")
    def plan_png():
        return FileResponse(rdir / "floorplan/structure.png")

    @app.get("/media/discrepancy.png")
    def discrepancy_png():
        p = rdir / "discrepancy/discrepancy_layer.png"
        if not p.exists():  # suppressed (alignment not confident) or not computed: transparent placeholder
            import base64

            png1x1 = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
            return Response(png1x1, media_type="image/png")
        return FileResponse(p)

    @app.get("/api/discrepancy")
    def discrepancy():
        return j("discrepancy/discrepancies.json") or {"status": "not computed"}

    @app.get("/media/plan_drawing.jpg")
    def plan_drawing():
        p = rdir / "floorplan/drawing.jpg"
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def serve(rdir: Path, host="127.0.0.1", port=8765):
    import uvicorn

    ensure_video(rdir)
    print(f"slam3d viewer for {rdir} -> http://{host}:{port}/")
    uvicorn.run(create_app(rdir), host=host, port=port, log_level="warning")
