"""Background pipeline worker for the ovarian ultrasound web app.

This worker drives the repository's own ``src/`` segmentation pipeline (the same
DS2Net / prompted SAM-256 backend the CLI uses) so the web UI and ``run.ps1``
share a single inference path. A daemon thread owns the OpenCV pop-up window and
pumps per-inference latency into :class:`RollingStats`; the dashboard polls
:meth:`snapshot` (over SSE) for live latency / FPS and the governor's iGPU KPI.
"""
from __future__ import annotations
import logging
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

# The repository's src/ pipeline is the single source of truth for inference.
# webapp/ is a sibling of src/, so put the app root on sys.path (idempotent).
_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from src.config import parse_config  # noqa: E402
from src.display import Presenter  # noqa: E402
from src.gpu_governor import GpuGovernor  # noqa: E402
from src.segmenters import create_segmenter  # noqa: E402
from src.sources import create_source  # noqa: E402
from src import app as _pipeline  # noqa: E402  (capture_loop, Rate, LatestResult, shutdown)

from .registry import USE_CASES

log = logging.getLogger("webapp.pipeline")


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class RollingStats:
    """Thread-safe rolling window of inference latencies for the dashboard."""

    def __init__(self, maxlen: int = 200):
        self._lats: deque[float] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._total = 0

    def add(self, ms: float) -> None:
        with self._lock:
            self._lats.append(ms)
            self._total += 1

    def snapshot(self) -> dict:
        with self._lock:
            xs = list(self._lats)
            total = self._total
        if not xs:
            return {"count": total, "window": 0, "mean_ms": 0.0, "p50_ms": 0.0,
                    "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "fps": 0.0}
        mean = sum(xs) / len(xs)
        return {
            "count": total,
            "window": len(xs),
            "mean_ms": round(mean, 3),
            "p50_ms": round(_percentile(xs, 0.50), 3),
            "p90_ms": round(_percentile(xs, 0.90), 3),
            "p95_ms": round(_percentile(xs, 0.95), 3),
            "p99_ms": round(_percentile(xs, 0.99), 3),
            "fps": round(1000.0 / mean, 2) if mean > 0 else 0.0,
        }


class PipelineWorker:
    """Single-pipeline lifecycle wrapper around the ``src/`` backend."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.state = "idle"          # idle | loading | running | stopped | error
        self.error: str | None = None
        self.config: dict | None = None
        self.model_info: dict = {}
        self.stats = RollingStats()
        self._governor = None
        self._gpu_cap = 80.0

    # ---- control -------------------------------------------------------
    def start(self, usecase_key: str, device: str, precision: str = "fp16",
              video: str | None = None) -> None:
        with self._lock:
            if self.state in ("loading", "running"):
                raise RuntimeError("A pipeline is already running \u2014 stop it first.")
            if usecase_key not in USE_CASES:
                raise KeyError(f"Unknown use-case: {usecase_key}")
            uc = USE_CASES[usecase_key]
            if not uc.enabled:
                raise RuntimeError(f"{uc.label} is not wired up yet.")
            if device not in uc.devices:
                raise ValueError(f"{uc.label} does not support device {device}.")
            if precision not in uc.precisions:
                raise ValueError(f"{uc.label} does not support precision {precision}.")

            cfg = self._build_config(uc, device, precision, video)

            self._stop.clear()
            self.error = None
            self.stats = RollingStats()
            self.state = "loading"
            self.config = {"usecase": uc.key, "label": uc.label, "device": device,
                           "precision": precision, "source": cfg.source,
                           "input": cfg.input, "arch": cfg.model_arch}
            self.model_info = {"model": uc.model_name, "arch": cfg.model_arch,
                               "device": device, "precision": precision,
                               "resolution": str(cfg.res)}
            self._gpu_cap = cfg.gpu_cap
            self._thread = threading.Thread(target=self._run, args=(cfg, uc),
                                            name="pipeline", daemon=True)
            self._thread.start()

    def _build_config(self, uc, device: str, precision: str, video: str | None):
        """Translate a UI request into the src/ CLI Config (one inference path)."""
        inp = (video or "").strip() or str(_APP_ROOT / uc.default_video)
        p = Path(inp)
        if not p.exists():
            raise FileNotFoundError(
                f"input source not found: {inp}. Type a file or folder path in "
                f"the Source box, or prepare the dataset so the default path "
                f"({uc.default_video}) exists - see "
                f"docs/user-guide/get-started/model-preparation.md")
        source = "folder" if p.is_dir() else "file"
        arch = "sam" if uc.task == "ovarian_sam" else "ds2net"
        argv = ["--source", source, "--input", str(p), "--device", device,
                "--model-arch", arch, "--view", "side_by_side"]
        if arch == "ds2net":
            ir = uc.ir_path(precision)
            if not ir.exists():
                raise FileNotFoundError(
                    f"model IR not built on this box: {ir}. Build it with  "
                    f".\\prepare_model.ps1 -Verify  (export a checkpoint) or  "
                    f".\\prepare_model.ps1 -Train -Verify  (train first), then Start "
                    f"again - see docs/user-guide/get-started/model-preparation.md")
            argv += ["--model", str(ir)]
        else:
            enc = uc.ir_path(precision)
            dec = uc.decoder_ir_path(precision)
            yolo = uc.bbox_ir_path(precision)
            for label, ir in (("encoder", enc), ("decoder", dec), ("yolo bbox", yolo)):
                if not ir.exists():
                    raise FileNotFoundError(
                        f"SAM {label} IR not built on this box: {ir}. Build the SAM "
                        f"models with  .\\prepare_model.ps1 -Arch sam , then Start "
                        f"again - see docs/user-guide/get-started/model-preparation.md")
            argv += ["--sam-encoder", str(enc), "--sam-decoder", str(dec),
                     "--sam-yolo", str(yolo)]
        if os.environ.get("WEBAPP_HEADLESS", "0") != "0":
            argv.append("--headless")
        return parse_config(argv)

    # ---- worker body ---------------------------------------------------
    def _run(self, cfg, uc) -> None:
        _pipeline.shutdown.clear()
        governor = None
        presenter = None
        src = None
        threads: list[threading.Thread] = []
        try:
            src = create_source(cfg)
            seg = create_segmenter(cfg)
            governor = GpuGovernor(cap_pct=cfg.gpu_cap, enabled=not cfg.no_governor)
            governor.start()
            self._governor = governor

            display_q: queue.Queue = queue.Queue(maxsize=1)
            infer_q: queue.Queue = queue.Queue(maxsize=1)
            latest = _pipeline.LatestResult()
            cap_rate, inf_rate = _pipeline.Rate(), _pipeline.Rate()

            threads = [
                threading.Thread(target=_pipeline.capture_loop,
                                 args=(cfg, src, display_q, infer_q, cap_rate),
                                 name="capture", daemon=True),
                threading.Thread(target=self._infer_loop,
                                 args=(seg, governor, infer_q, latest, inf_rate),
                                 name="inference", daemon=True),
            ]
            for t in threads:
                t.start()

            presenter = Presenter(cfg.window, cfg.display_scale, cfg.alpha,
                                  cfg.mask_color, headless=cfg.headless,
                                  record=cfg.record, src=src, view=cfg.view)
            self.state = "running"
            last_frame = None
            while not _pipeline.shutdown.is_set() and not self._stop.is_set():
                try:
                    last_frame = display_q.get(timeout=0.1)
                except queue.Empty:
                    if last_frame is None:
                        continue
                result, _ = latest.get()
                gov = governor.snapshot()
                hud = {
                    "arch": cfg.model_arch,
                    "inf_fps": inf_rate.fps,
                    "inf_ms": inf_rate.latency_ms,
                    "gpu_peak": gov.get("peak_pct", 0.0),
                    "gpu_mean": gov.get("mean_pct", 0.0),
                    "gpu_cap": gov.get("cap_pct", cfg.gpu_cap),
                    "gpu_available": gov.get("available", False),
                    "throttle_ms": gov.get("throttle_delay_ms", 0.0),
                    "within_cap": gov.get("within_cap", True),
                }
                key = presenter.show(last_frame, result, hud)
                if key in (27, ord("q")):
                    break
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.state = "error"
            log.exception("pipeline failed: %s", exc)
        finally:
            _pipeline.shutdown.set()
            if governor is not None:
                governor.stop()
            if presenter is not None:
                presenter.close()
            if src is not None:
                src.close()
            for t in threads:
                t.join(timeout=1.0)
            if self.state != "error":
                self.state = "stopped"

    def _infer_loop(self, seg, governor, infer_q, latest, inf_rate) -> None:
        """Mirror src.app.inference_loop, also feeding the dashboard stats."""
        while not _pipeline.shutdown.is_set():
            try:
                frame = infer_q.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            try:
                result = seg.infer(frame)
            except Exception as exc:  # noqa: BLE001
                log.warning("inference error: %s", exc)
                continue
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latest.set(result, time.perf_counter_ns())
            inf_rate.tick(dt_ms)
            self.stats.add(dt_ms)
            d = governor.delay()
            if d > 0:
                time.sleep(d)

    def stop(self) -> None:
        self._stop.set()
        _pipeline.shutdown.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=5)
        if self.state not in ("error",):
            self.state = "idle"

    def snapshot(self) -> dict:
        snap = {
            "state": self.state,
            "error": self.error,
            "config": self.config,
            "model_info": self.model_info,
            "metrics": self.stats.snapshot(),
        }
        gov = self._governor
        if gov is not None:
            snap["gpu_cap"] = gov.snapshot()
        return snap
