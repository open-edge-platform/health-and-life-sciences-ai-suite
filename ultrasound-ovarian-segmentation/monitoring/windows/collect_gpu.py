# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Windows GPU-Engine / GPU-Adapter-Memory perf-counter reader (win32pdh).

Returns the per-engine utilization percentages (the same counters the Windows
Task Manager GPU view uses) and total GPU memory. The adaptive GPU governor
collapses the per-engine dict to its busiest value to enforce the iGPU KPI.

Windows-only: needs ``pywin32`` (``win32pdh``). On other hosts the import fails
and the governor degrades to a no-op.
"""
import logging
import re
import time
from collections import defaultdict

import win32pdh

logger = logging.getLogger(__name__)


def get_gpu_memory_total():
    """(total_mb, dedicated_mb, shared_mb) across all GPU adapters."""
    try:
        query = win32pdh.OpenQuery()
        counters_dedicated = []
        counters_shared = []

        instances = win32pdh.EnumObjectItems(
            None, None, "GPU Adapter Memory", win32pdh.PERF_DETAIL_WIZARD
        )[1]
        for inst in instances:
            counters_dedicated.append(
                win32pdh.AddCounter(query, f"\\GPU Adapter Memory({inst})\\Dedicated Usage")
            )
            counters_shared.append(
                win32pdh.AddCounter(query, f"\\GPU Adapter Memory({inst})\\Shared Usage")
            )

        win32pdh.CollectQueryData(query)

        total_dedicated = 0
        total_shared = 0
        for c in counters_dedicated:
            _, val = win32pdh.GetFormattedCounterValue(c, win32pdh.PDH_FMT_LARGE)
            total_dedicated += val
        for c in counters_shared:
            _, val = win32pdh.GetFormattedCounterValue(c, win32pdh.PDH_FMT_LARGE)
            total_shared += val

        win32pdh.CloseQuery(query)

        dedicated_mb = total_dedicated / (1024 * 1024)
        shared_mb = total_shared / (1024 * 1024)
        return dedicated_mb + shared_mb, dedicated_mb, shared_mb
    except Exception as exc:  # noqa: BLE001
        logger.error("GPU memory read error: %s", exc)
        return None, None, None


def get_gpu_utilization():
    """Per-engine utilization %% as {engine_type: percent} across all adapters.

    Two CollectQueryData calls 0.2 s apart are required for PDH rate counters to
    produce a non-zero delta, so this call blocks ~0.2 s.
    """
    query = win32pdh.OpenQuery()
    counters = []

    _, instances = win32pdh.EnumObjectItems(None, None, "GPU Engine", win32pdh.PERF_DETAIL_WIZARD)
    engine_types = ["engtype_3D", "engtype_VideoEncode", "engtype_VideoDecode",
                    "engtype_VideoProcessing", "engtype_Copy", "engtype_Compute"]

    for inst in instances:
        for engine_type in engine_types:
            if re.search(engine_type, inst, re.IGNORECASE):
                try:
                    c = win32pdh.AddCounter(query, f"\\GPU Engine({inst})\\Utilization Percentage")
                    counters.append((c, engine_type))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("skipping %s: %s", inst, exc)

    win32pdh.CollectQueryData(query)
    time.sleep(0.2)
    win32pdh.CollectQueryData(query)

    engine_totals = defaultdict(float)
    for c, engine_type in counters:
        try:
            _, val = win32pdh.GetFormattedCounterValue(c, win32pdh.PDH_FMT_DOUBLE)
            engine_totals[engine_type] += val
        except Exception:  # noqa: BLE001
            pass

    win32pdh.CloseQuery(query)
    return engine_totals
