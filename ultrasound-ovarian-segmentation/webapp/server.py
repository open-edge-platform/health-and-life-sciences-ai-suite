"""Flask control plane for the ovarian ultrasound AI app.

Serves the static dashboard, enumerates OpenVINO devices + use-cases, and
exposes start/stop/reset plus an SSE stream of live pipeline metrics. The
inference video is shown in a separate OpenCV pop-up window (see
pipeline_worker.py), not streamed to the browser.
"""
from __future__ import annotations
import json
import os
import platform as _platform
import shutil
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from flask import Flask, Response, jsonify, request, send_from_directory
import openvino as ov

ROOT = Path(__file__).resolve().parents[1]  # app/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .registry import USE_CASES  # noqa: E402
from .pipeline_worker import PipelineWorker  # noqa: E402

# Hardware-utilization monitoring (CPU/GPU/NPU/power/memory), ported from the
# Intel Education AI Suite. The Windows collectors require win32pdh (pywin32)
# and pandas, so the import is guarded to keep the server importable on dev
# hosts where those are absent (metrics then fall back to psutil CPU+memory).
_MON_LOGS = ROOT / "cache" / "monitoring_logs" / "utilization_logs"
try:
    from monitoring.monitor import (  # noqa: E402
        start_monitoring as _mon_start,
        stop_monitoring as _mon_stop,
        get_metrics as _mon_get,
        is_monitoring_active as _mon_active,
    )
    _MON_OK = True
    _MON_ERR = ""
except Exception as _exc:  # platform / dependency dependent
    _mon_start = _mon_stop = _mon_get = _mon_active = None  # type: ignore
    _MON_OK = False
    _MON_ERR = f"{type(_exc).__name__}: {_exc}"

STATIC = Path(__file__).resolve().parent / "static"
UI_DIST = Path(__file__).resolve().parent / "ui" / "dist"
DEVICE_LABELS = {"CPU": "CPU", "GPU": "Intel iGPU", "NPU": "Intel NPU"}

app = Flask(__name__, static_folder=None)
worker = PipelineWorker()
_core = ov.Core()

# The model/use-case is chosen at launch on the CLI (run_webapp.ps1 -UseCase ->
# WEBAPP_USECASE); the UI no longer selects it — it only shows the active model
# and the Start button runs it. Default to the first registered use-case.
_ACTIVE_USECASE = (os.environ.get("WEBAPP_USECASE", "").strip()
                   or next(iter(USE_CASES)))
if _ACTIVE_USECASE not in USE_CASES:
    raise SystemExit(f"[webapp] unknown WEBAPP_USECASE '{_ACTIVE_USECASE}'. "
                     f"Choose one of: {', '.join(USE_CASES)}")
_METRICS_HISTORY: deque = deque(maxlen=60)


def available_devices() -> list[dict]:
    seen: dict[str, dict] = {}
    for d in _core.available_devices:
        base = d.split(".")[0]
        if base in DEVICE_LABELS and base not in seen:
            try:
                name = _core.get_property(d, "FULL_DEVICE_NAME")
            except Exception:
                name = d
            seen[base] = {"id": base, "label": DEVICE_LABELS[base], "name": name}
    # Stable order CPU, GPU, NPU
    return [seen[k] for k in ("CPU", "GPU", "NPU") if k in seen]


def _platform_util() -> dict:
    out = {"cpu_pct": None, "gpu_pct": None, "npu_pct": None}
    try:
        import psutil
        out["cpu_pct"] = psutil.cpu_percent(interval=None)
    except Exception:
        pass
    return out


def _reset_mon_logs() -> None:
    """Remove the previous run's utilization CSVs so each run starts fresh."""
    try:
        _MON_LOGS.mkdir(parents=True, exist_ok=True)
        for f in _MON_LOGS.glob("*.csv"):
            try:
                f.unlink()
            except Exception:
                pass
    except Exception:
        pass


def _adapt_metrics(m: dict) -> dict:
    """Map monitoring.get_metrics() output to the dashboard's ResourceMetrics.

    The GPU collector emits a rich row [ts, mem_total_mb, ded_mb, shared_mb,
    3D%, VideoEncode%, VideoDecode%, VideoProcessing%, Copy%, Compute%]; the UI
    reads index [1] as the GPU %, so collapse the six engine columns into one
    overall utilization (the busiest engine). CPU/NPU/memory/power already match.
    """
    gpu = []
    for r in m.get("gpu_utilization", []) or []:
        engines = r[4:10] if len(r) >= 10 else r[1:]
        try:
            pct = max((float(x) for x in engines), default=0.0)
        except Exception:
            pct = 0.0
        gpu.append([r[0], round(pct, 1)])
    return {
        "cpu_utilization": m.get("cpu_utilization", []) or [],
        "gpu_utilization": gpu,
        "npu_utilization": m.get("npu_utilization", []) or [],
        "memory": m.get("memory", []) or [],
        "power": m.get("power", []) or [],
    }


@app.get("/")
def index() -> Response:
    if (UI_DIST / "index.html").exists():
        return send_from_directory(UI_DIST, "index.html")
    return Response(
        "Ovarian UI is not built yet. Run:\n"
        "  cd webapp/ui && npm install && npm run build\n",
        mimetype="text/plain", status=503)


@app.get("/assets/<path:filename>")
def ui_assets(filename: str) -> Response:
    return send_from_directory(UI_DIST / "assets", filename)


@app.get("/vite.svg")
def ui_favicon() -> Response:
    return send_from_directory(UI_DIST, "vite.svg")


@app.get("/static/<path:filename>")
def static_files(filename: str) -> Response:
    return send_from_directory(STATIC, filename)


@app.get("/api/readiness")
def readiness():
    return jsonify({"ready": True, "devices": available_devices()})


@app.get("/api/devices")
def devices():
    return jsonify(available_devices())


@app.get("/api/usecases")
def usecases():
    present = {d["id"] for d in available_devices()}
    out = []
    for uc in USE_CASES.values():
        if uc.key != _ACTIVE_USECASE:
            continue
        out.append({
            "key": uc.key,
            "label": uc.label,
            "modality": uc.modality,
            "task": uc.task,
            "model": uc.model_name,
            "input_shape": list(uc.input_shape),
            "precisions": list(uc.precisions),
            "devices": [d for d in uc.devices if d in present],
            "ir_available": {p: uc.ir_path(p).exists() for p in uc.precisions},
            "kpi_latency_ms": uc.kpi_latency_ms,
            "kpi_fps": uc.kpi_fps,
            "enabled": uc.enabled,
            "note": uc.note,
        })
    return jsonify(out)


@app.post("/api/start")
def start():
    body = request.get_json(force=True, silent=True) or {}
    try:
        # The model/use-case is fixed at launch (WEBAPP_USECASE); the UI only
        # supplies device / precision / source.
        worker.start(_ACTIVE_USECASE, body["device"],
                     body.get("precision", "fp16"), body.get("video"))
        if _MON_OK:
            try:
                _reset_mon_logs()
                _mon_start(str(_MON_LOGS))
            except Exception as exc:
                print(f"[webapp] monitoring start failed: {exc}", file=sys.stderr)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/stop")
def stop():
    worker.stop()
    if _MON_OK:
        try:
            _mon_stop()
        except Exception:
            pass
    return jsonify({"ok": True})


@app.post("/api/reset")
def reset():
    worker.stop()
    if _MON_OK:
        try:
            _mon_stop()
        except Exception:
            pass
    worker.state = "idle"
    worker.config = None
    return jsonify({"ok": True})


@app.get("/api/status")
def status():
    snap = worker.snapshot()
    snap["platform"] = _platform_util()
    snap["lifecycle"] = snap.get("state")
    return jsonify(snap)


@app.get("/api/events")
def events():
    def gen():
        while True:
            snap = worker.snapshot()
            snap["platform"] = _platform_util()
            snap["lifecycle"] = snap.get("state")
            yield f"data: {json.dumps(snap)}\n\n"
            time.sleep(0.5)
    return Response(gen(), mimetype="text/event-stream")


def _device_name(dev: str) -> str:
    try:
        return _core.get_property(dev, "FULL_DEVICE_NAME")
    except Exception:
        return "N/A"


@app.get("/api/platform-info")
def platform_info():
    present = {d["id"] for d in available_devices()}
    try:
        import psutil
        mem_gb = f"{psutil.virtual_memory().total / (1024 ** 3):.0f} GB"
    except Exception:
        mem_gb = "N/A"
    try:
        storage = f"{shutil.disk_usage(str(ROOT)).total / (1024 ** 3):.0f} GB"
    except Exception:
        storage = "N/A"
    return jsonify({
        "Processor": _device_name("CPU"),
        "iGPU": _device_name("GPU") if "GPU" in present else "N/A",
        "NPU": _device_name("NPU") if "NPU" in present else "N/A",
        "Memory": mem_gb,
        "Storage": storage,
        "OS": f"{_platform.system()} {_platform.release()}",
    })


@app.get("/api/hardware-metrics")
def hardware_metrics():
    # Prefer the full hardware monitor (CPU/GPU/NPU/power/memory) while a run is
    # active; otherwise fall back to psutil CPU+memory so the panel still moves.
    if _MON_OK and _mon_active():
        try:
            return jsonify(_adapt_metrics(_mon_get(str(_MON_LOGS))))
        except Exception as exc:
            print(f"[webapp] hardware-metrics read failed: {exc}", file=sys.stderr)
    ts = datetime.now().isoformat()
    cpu_pct = 0.0
    mem = [0, 0, 0, 0.0]
    try:
        import psutil
        cpu_pct = float(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        mem = [int(vm.used), int(vm.total), int(vm.available), round(float(vm.percent), 1)]
    except Exception:
        pass
    _METRICS_HISTORY.append({"ts": ts, "cpu": round(cpu_pct, 1), "mem": mem})
    hist = list(_METRICS_HISTORY)
    return jsonify({
        "cpu_utilization": [[h["ts"], h["cpu"]] for h in hist],
        "gpu_utilization": [],
        "npu_utilization": [],
        "memory": [[h["ts"], *h["mem"]] for h in hist],
        "power": [],
    })


@app.post("/api/start-monitoring")
def start_monitoring_endpoint():
    """Reference-parity endpoint: begin hardware-utilization sampling."""
    if not _MON_OK:
        return jsonify({"status": "unavailable", "message": _MON_ERR})
    _reset_mon_logs()
    _mon_start(str(_MON_LOGS))
    return jsonify({"status": "success", "message": "Monitoring started"})


@app.get("/api/metrics")
def metrics_endpoint():
    """Reference-parity endpoint: raw get_metrics() payload."""
    if not _MON_OK:
        return jsonify({"cpu_utilization": [], "gpu_utilization": [],
                        "npu_utilization": [], "memory": [], "power": []})
    return jsonify(_mon_get(str(_MON_LOGS)))


@app.post("/api/stop-monitoring")
def stop_monitoring_endpoint():
    """Reference-parity endpoint: stop hardware-utilization sampling."""
    if _MON_OK:
        _mon_stop()
    return jsonify({"status": "success", "message": "Monitoring stopped"})


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


def main() -> None:
    host = "127.0.0.1"
    port = int(os.environ.get("WEBAPP_PORT", "8080"))
    print(f"[webapp] Ovarian Ultrasound AI app -> http://{host}:{port}")
    print(f"[webapp] Devices: {[d['id'] for d in available_devices()]}")
    if _MON_OK:
        print("[webapp] HW monitoring: available (CPU/GPU/NPU/power/memory)")
    else:
        print(f"[webapp] HW monitoring: DISABLED -> {_MON_ERR}")
    # Run Flask on a background thread so the process's MAIN thread stays free.
    # OpenCV's pop-up (created by the pipeline worker) only renders reliably on
    # Windows when the main thread is not blocked inside the WSGI accept loop --
    # this mirrors the standalone run, where the window shows correctly.
    server = threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True,
                                use_reloader=False),
        name="flask-server", daemon=True)
    server.start()
    print(f"[webapp] serving on http://{host}:{port}  (Ctrl+C to quit)")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[webapp] shutting down")


if __name__ == "__main__":
    main()
