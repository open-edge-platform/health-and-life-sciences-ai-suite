"""Local configuration for the bundled hardware-monitoring package.

Mirrors the ``config.monitoring.*`` attributes the Education AI Suite collectors
expect, so the upstream collector code runs unchanged. Values match the
reference ``config.yaml`` (interval: 2s); all paths are resolved relative to this
package / the app so they work regardless of the process working directory.
"""
import os
from types import SimpleNamespace

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))   # app/monitoring
_APP_DIR = os.path.dirname(_PKG_DIR)                     # app
_LOGS_DIR = os.path.join(_APP_DIR, "cache", "monitoring_logs")

monitoring = SimpleNamespace(
    # Sampling interval in seconds (matches reference config.yaml).
    interval=2,
    # Root directory for the per-metric CSVs.
    logs_dir=_LOGS_DIR,
    # Absolute path to the bundled Intel SoCWatch binary (Windows, power).
    socwatch_path=os.path.join(
        _PKG_DIR, "tools", "socwatch_windows_v2026.0", "64", "socwatch.exe"
    ),
    # Scratch directory for SoCWatch output files.
    execution_logs=os.path.join(_LOGS_DIR, "executionlogs"),
    # NPU utilization exe, relative to this package root. collect_npu joins it to
    # <this file>/../.. so it resolves under app/monitoring/tools/...
    npu_exe_path=os.path.join(
        "tools", "level-zero", "build", "bin", "Release", "npu_utilization.exe"
    ),
)

config = SimpleNamespace(monitoring=monitoring)
