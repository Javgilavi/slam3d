"""Config loading, run directories and resumable stage bookkeeping.

Every stage writes `<run>/<stage>/` outputs and records in `<run>/status.json`:
  state (running|done|failed|skipped), start/end time, runtime, peak process RSS (incl. children),
  peak GPU memory used (nvidia-smi, whole device), error message, config hash.
A stage whose status is `done` with the same config hash is skipped unless forced.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def resolve(p: str | None) -> Path | None:
    if p is None:
        return None
    q = Path(p).expanduser()
    return q if q.is_absolute() else (REPO / q)


def run_dir(cfg: dict) -> Path:
    d = resolve(cfg.get("output_root", "outputs")) / cfg["run_name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


class _Sampler(threading.Thread):
    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        self.period = period
        self.peak_rss = 0
        self.peak_gpu = 0
        self.peak_docker = 0.0
        self._stop_evt = threading.Event()  # NOT `_stop`: that name shadows Thread._stop and breaks os.fork()

    def run(self):
        import subprocess

        import psutil

        me = psutil.Process()
        while not self._stop_evt.is_set():
            try:
                rss = me.memory_info().rss + sum(c.memory_info().rss for c in me.children(recursive=True))
                self.peak_rss = max(self.peak_rss, rss)
            except Exception:  # noqa: BLE001
                pass
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                self.peak_gpu = max(self.peak_gpu, int(out.split()[0]))
            except Exception:  # noqa: BLE001
                pass
            try:
                out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.MemUsage}}"],
                                     capture_output=True, text=True, timeout=5).stdout
                for line in out.splitlines():
                    if line.startswith("slam3d-"):
                        v = line.split()[1]
                        mult = {"GiB": 1024, "MiB": 1, "KiB": 1 / 1024}
                        for k, m in mult.items():
                            if v.endswith(k):
                                self.peak_docker = max(self.peak_docker, float(v[:-len(k)]) * m)
            except Exception:  # noqa: BLE001
                pass
            self._stop_evt.wait(self.period)

    def stop(self):
        self._stop_evt.set()


class Status:
    def __init__(self, rdir: Path):
        self.path = rdir / "status.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2))

    def is_done(self, stage: str, cfg_part) -> bool:
        s = self.data.get(stage)
        return bool(s and s.get("state") == "done" and s.get("config_hash") == _hash(cfg_part))

    @contextmanager
    def stage(self, stage: str, cfg_part, log=print):
        sampler = _Sampler()
        sampler.start()
        t0 = time.time()
        self.data[stage] = {"state": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "config_hash": _hash(cfg_part)}
        self.save()
        log(f"[{stage}] started")
        try:
            yield self.data[stage]
        except BaseException as e:
            sampler.stop()
            self.data[stage].update(state="failed", runtime_s=time.time() - t0, error=f"{type(e).__name__}: {e}",
                                    traceback=traceback.format_exc()[-4000:])
            self.save()
            log(f"[{stage}] FAILED: {e}")
            raise
        sampler.stop()
        self.data[stage].update(state="done", runtime_s=round(time.time() - t0, 2),
                                finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                                peak_rss_mb=round(sampler.peak_rss / 2**20, 1),
                                peak_gpu_used_mb=sampler.peak_gpu,
                                peak_docker_mem_mb=round(sampler.peak_docker, 1))
        self.save()
        log(f"[{stage}] done in {self.data[stage]['runtime_s']:.1f}s")
