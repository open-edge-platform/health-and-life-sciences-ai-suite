# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive GPU-utilization governor: keep the app's iGPU utilization under a
hard KPI ceiling (default 80%) by pacing the inference loop.

A background sampler reads the Windows GPU-Engine perf counters
(monitoring.windows.collect_gpu.get_gpu_utilization) and tracks the
busiest-engine %. A light proportional controller grows a per-frame delay as
utilization approaches the ceiling and releases it when the GPU is idle, so the
pipeline self-throttles instead of ever exceeding the KPI. It also records the
peak / mean busiest-engine % for the run so callers can prove the KPI was met.

On non-Windows / no-pywin32 hosts the counter import fails; the governor then
degrades to a no-op (zero delay, "unavailable" snapshot) so the app still runs
on dev boxes. Windows-only enforcement; the controller itself is pure Python
and unit-testable via ``simulate`` / ``python -m src.gpu_governor``.
"""
from __future__ import annotations

import threading
import time

# KPI: the app's iGPU busiest-engine utilization must stay under this ceiling.
DEFAULT_CAP_PCT = 80.0
# Start easing off below the ceiling so the ~1 s sample cadence never overshoots.
DEFAULT_TARGET_PCT = 70.0

try:
    from monitoring.windows.collect_gpu import get_gpu_utilization
    _SAMPLER_OK = True
    _SAMPLER_ERR = ""
except Exception as _exc:  # pragma: no cover - platform dependent (needs win32pdh)
    get_gpu_utilization = None  # type: ignore
    _SAMPLER_OK = False
    _SAMPLER_ERR = f"{type(_exc).__name__}: {_exc}"


def _busiest(engine_totals: dict | None) -> float:
    """Busiest-engine % = max across the six GPU-Engine columns (Task-Manager
    equivalent)."""
    if not engine_totals:
        return 0.0
    try:
        return max(float(v) for v in engine_totals.values())
    except Exception:
        return 0.0


class GpuGovernor:
    """Keep busiest-engine GPU% under ``cap_pct`` by pacing the caller's loop."""

    def __init__(self, cap_pct: float = DEFAULT_CAP_PCT,
                 target_pct: float = DEFAULT_TARGET_PCT,
                 interval: float = 1.0, max_delay: float = 1.0,
                 gain: float = 0.5, enabled: bool = True) -> None:
        self.cap_pct = float(cap_pct)
        # Keep the soft target a sensible margin below the hard ceiling.
        self.target_pct = float(min(target_pct, cap_pct - 5.0))
        self.interval = float(interval)
        self.max_delay = float(max_delay)
        self.gain = float(gain)
        self.enabled = bool(enabled) and _SAMPLER_OK

        self._delay = 0.0
        self._peak = 0.0
        self._sum = 0.0
        self._n = 0
        self._engaged = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- control -------------------------------------------------------
    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop,
                                        name="gpu-governor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None

    def delay(self) -> float:
        """Current extra per-frame delay in seconds (cheap; call every frame)."""
        with self._lock:
            return self._delay

    # ---- controller ----------------------------------------------------
    def _update(self, busiest: float) -> float:
        """Fold one utilization sample into the controller; return the new delay.

        A pure function of the sample plus current state, so the throttle
        response can be unit-tested without any GPU (see ``simulate``).
        """
        with self._lock:
            self._peak = max(self._peak, busiest)
            self._sum += busiest
            self._n += 1
            error = busiest - self.target_pct
            delay = self._delay + self.gain * error / 100.0
            if busiest > self.cap_pct:
                delay += 0.1            # hard kick if we ever cross the ceiling
            delay = max(0.0, min(self.max_delay, delay))
            self._delay = delay
            if delay > 0.0:
                self._engaged = True
            return delay

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            busiest = _busiest(get_gpu_utilization()) if get_gpu_utilization else 0.0
            self._update(busiest)
            # get_gpu_utilization already blocks ~0.2 s; pad out to `interval`.
            self._stop.wait(max(0.0, self.interval - (time.perf_counter() - t0)))

    # ---- reporting -----------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            mean = (self._sum / self._n) if self._n else 0.0
            peak = self._peak
            samples = self._n
            engaged = self._engaged
            delay_ms = round(self._delay * 1000.0, 1)
        return {
            "available": _SAMPLER_OK,
            "enabled": self.enabled,
            "cap_pct": round(self.cap_pct, 1),
            "target_pct": round(self.target_pct, 1),
            "peak_pct": round(peak, 1),
            "mean_pct": round(mean, 1),
            "samples": samples,
            "throttle_engaged": engaged,
            "throttle_delay_ms": delay_ms,
            "within_cap": bool(peak < self.cap_pct),
            "note": "" if _SAMPLER_OK else f"GPU counters unavailable ({_SAMPLER_ERR})",
        }

    # ---- self-test -----------------------------------------------------
    @classmethod
    def simulate(cls, util_series, **kw) -> list:
        """Feed a list of busiest% samples through the controller (no GPU).

        Returns the per-sample delay (seconds) so the throttle response can be
        verified on any host. Used by ``python -m src.gpu_governor``.
        """
        gov = cls(enabled=False, **kw)   # no sampler thread; drive it by hand
        return [gov._update(float(u)) for u in util_series]


def _selftest() -> None:
    # Idle -> ramps toward saturation -> back to idle.
    series = [5, 10, 20, 30, 55, 72, 85, 92, 88, 78, 60, 30, 10]
    delays = GpuGovernor.simulate(series)
    print(f"GPU governor self-test (cap={DEFAULT_CAP_PCT:.0f}%, "
          f"target={DEFAULT_TARGET_PCT:.0f}%)")
    print(f"{'busiest%':>9} {'delay_ms':>9}")
    for u, d in zip(series, delays):
        flag = "  <- throttling" if d > 0 else ""
        print(f"{u:9.0f} {d * 1000:9.1f}{flag}")
    engaged_at = next((u for u, d in zip(series, delays) if d > 0), None)
    print(f"\nthrottle first engaged at busiest={engaged_at}% ; "
          f"max simulated busiest={max(series)}%")
    print(f"sampler available on this host: {_SAMPLER_OK} {_SAMPLER_ERR}".rstrip())


if __name__ == "__main__":
    _selftest()
