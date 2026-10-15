# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ultrasound ovarian tumor segmentation — live OpenVINO pipeline (Windows).

Decoupled capture / inference / display design (the same proven architecture as
the surgical-instrument sample), specialised for a promptless SegFormer-B5
(DS2Net) model that segments ovarian tumors in ultrasound frames:

  * Capture thread   : pulls the newest frame from the source, feeds the display
                       and (every Nth frame) the inference queue. Never blocks.
  * Inference thread : OpenVINO segmentation on the latest frame -> shared mask,
                       then self-throttles via the GPU governor's per-frame delay.
  * Display (main)   : draws every captured frame with the latest mask overlaid,
                       so display FPS is independent of inference FPS.

The GPU governor holds the Intel iGPU busiest-engine utilization under the 80%
KPI ceiling and records the peak/mean so the KPI can be proven after a run.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import parse_config  # noqa: E402
from src.display import Presenter  # noqa: E402
from src.gpu_governor import GpuGovernor  # noqa: E402
from src.segmenters import Result, create_segmenter  # noqa: E402
from src.sources import create_source  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s: %(message)s")
log = logging.getLogger("app")

shutdown = threading.Event()


class Rate:
    """Rolling FPS / latency counter."""

    def __init__(self) -> None:
        self._t = time.monotonic()
        self._n = 0
        self.fps = 0.0
        self._lat_ms = 0.0
        self._lat_sum = 0.0
        self._lat_count = 0

    def tick(self, latency_ms: float | None = None) -> None:
        self._n += 1
        if latency_ms is not None:
            self._lat_ms = 0.9 * self._lat_ms + 0.1 * latency_ms
            self._lat_sum += latency_ms
            self._lat_count += 1
        now = time.monotonic()
        if now - self._t >= 1.0:
            self.fps = self._n / (now - self._t)
            self._t, self._n = now, 0

    @property
    def latency_ms(self) -> float:
        return self._lat_ms

    @property
    def mean_latency_ms(self) -> float:
        return self._lat_sum / self._lat_count if self._lat_count else 0.0

    @property
    def count(self) -> int:
        return self._lat_count


class LatestResult:
    """Thread-safe most-recent segmentation result + timestamp."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result: Result | None = None
        self._ts_ns = 0

    def set(self, result: Result, ts_ns: int) -> None:
        with self._lock:
            self._result, self._ts_ns = result, ts_ns

    def get(self) -> tuple[Result | None, int]:
        with self._lock:
            return self._result, self._ts_ns


def _put_latest(q: "queue.Queue", item) -> None:
    """Non-blocking put that keeps only the newest item."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


def capture_loop(cfg, src, display_q, infer_q, cap_rate) -> None:
    frame_interval = 1.0 / src.fps if (not src.is_live and src.fps > 0) else 0.0
    n = 0
    next_t = time.monotonic()
    while not shutdown.is_set():
        frame = src.read()
        if frame is None:
            if not src.is_live:
                shutdown.set()
                break
            continue
        n += 1
        cap_rate.tick()
        _put_latest(display_q, frame)
        if n % cfg.frame_skip == 0:
            _put_latest(infer_q, frame)
        if frame_interval:
            next_t += frame_interval
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()


def inference_loop(seg, governor, infer_q, latest, inf_rate) -> None:
    while not shutdown.is_set():
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
        latest.set(result, time.perf_counter_ns())
        inf_rate.tick((time.perf_counter() - t0) * 1000.0)
        # Self-throttle: if iGPU utilization approaches the cap, the governor
        # returns a positive per-frame delay that paces this loop down.
        d = governor.delay()
        if d > 0:
            time.sleep(d)


def _attach_log_file(path: str) -> None:
    """Opt-in: also write the run log (incl. KPI verdicts) to a file.

    Off by default; enabled only via --log-file / LOG_FILE. Each record is
    flushed, so the KPI lines survive even an abrupt Ctrl-C. The file is the
    user's to keep or delete (reports/ is git-ignored); the app never rotates
    or auto-removes it.
    """
    fpath = Path(path)
    if fpath.parent and not fpath.parent.exists():
        fpath.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(fpath, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    log.info("writing run log to %s", fpath)


def _log_kpis(cfg, governor, inf_rate) -> None:
    """Emit the GPU + segmentation KPI verdicts.

    Called from main()'s finally block so the verdicts are logged on ANY exit
    (clean quit, window close, or Ctrl-C), not only a graceful one.
    """
    gov = governor.snapshot()
    if gov.get("available"):
        verdict = "PASS" if gov["within_cap"] else "OVER"
        log.info("GPU KPI (< %.0f%%): %s — peak busiest-engine %.1f%% mean %.1f%% "
                 "(samples=%d, throttle_engaged=%s)",
                 gov["cap_pct"], verdict, gov["peak_pct"], gov["mean_pct"],
                 gov["samples"], gov["throttle_engaged"])

    lat = inf_rate.mean_latency_ms
    if lat > 0:
        seg_fps = 1000.0 / lat
        lat_v = "PASS" if lat < 30.0 else "OVER"
        fps_v = "PASS" if seg_fps > 30.0 else "UNDER"
        log.info("Segmentation KPI (device=%s): latency %.1f ms (< 30 ms: %s) | "
                 "throughput %.1f FPS (> 30 FPS: %s) over %d inferences",
                 cfg.device, lat, lat_v, seg_fps, fps_v, inf_rate.count)


def main() -> int:
    cfg = parse_config()
    if cfg.log_file:
        _attach_log_file(cfg.log_file)
    log.info("config: %s", cfg)

    src = create_source(cfg)
    log.info("source: %s (%dx%d @ %.1f fps, live=%s)",
             src.name, src.width, src.height, src.fps, src.is_live)

    seg = create_segmenter(cfg)

    governor = GpuGovernor(cap_pct=cfg.gpu_cap, enabled=not cfg.no_governor)
    governor.start()
    if governor.enabled:
        log.info("GPU governor active: cap=%.0f%% target=%.0f%% (throttles only if GPU%% climbs)",
                 governor.cap_pct, governor.target_pct)
    else:
        log.info("GPU governor inactive (%s)", governor.snapshot().get("note") or "disabled")

    display_q: queue.Queue = queue.Queue(maxsize=1)
    infer_q: queue.Queue = queue.Queue(maxsize=1)
    latest = LatestResult()
    cap_rate, inf_rate = Rate(), Rate()

    threads = [
        threading.Thread(target=capture_loop, args=(cfg, src, display_q, infer_q, cap_rate),
                         name="capture", daemon=True),
        threading.Thread(target=inference_loop, args=(seg, governor, infer_q, latest, inf_rate),
                         name="inference", daemon=True),
    ]
    for t in threads:
        t.start()

    presenter = Presenter(cfg.window, cfg.display_scale, cfg.alpha, cfg.mask_color,
                          headless=cfg.headless, record=cfg.record, src=src, view=cfg.view)
    last_frame = None
    try:
        while not shutdown.is_set():
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
    except KeyboardInterrupt:
        pass
    finally:
        shutdown.set()
        # Log the KPI verdicts first thing on the way out, so they are emitted
        # (and flushed to --log-file) on ANY exit — clean quit, window close, or
        # Ctrl-C — before the slower governor/thread teardown below.
        _log_kpis(cfg, governor, inf_rate)
        governor.stop()
        presenter.close()
        src.close()
        for t in threads:
            t.join(timeout=1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
