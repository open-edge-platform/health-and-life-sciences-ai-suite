# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware utilization monitoring for the ovarian ultrasound app.

Two consumers share this package:
  * the GPU governor polls ``monitoring.windows.collect_gpu.get_gpu_utilization``
    to keep the iGPU under its utilization cap (used by the CLI and the web app);
  * the web dashboard uses ``monitoring.monitor`` (``start_monitoring`` /
    ``stop_monitoring`` / ``get_metrics``) to record the CPU/GPU/NPU/power/memory
    time series shown in the browser.

Ported from the Intel Education AI Suite. The Windows collectors require
``pywin32`` + ``pandas`` and the vendor tools under ``monitoring/tools/``
(SoCWatch / Level Zero, staged separately); absent those, the dashboard falls
back to psutil CPU + memory.
"""
