# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin OpenVINO runner: compile an IR to a device and run sync inference.

Ported verbatim (bar the cache path) from the ovarian POC so the SAM encoder /
decoder / detector run with the exact same compile settings that produced the
customer-accepted Dice and latency numbers.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import openvino as ov

# Compile cache root — saved under the app so iGPU/NPU re-compiles are skipped on
# subsequent runs (avoids the first-compile cost on every restart).
_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "ov_compile_cache"


class OVRunner:
    def __init__(self, model_xml: str | Path, device: str, hint: str = "LATENCY",
                 inference_precision: str | None = None):
        self.core = ov.Core()
        available = self.core.available_devices
        if device not in available:
            raise RuntimeError(f"Device {device!r} not available. Present: {available}")
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.core.set_property({"CACHE_DIR": str(_CACHE_DIR)})
        cfg: dict[str, str] = {"PERFORMANCE_HINT": hint}
        if inference_precision:
            cfg["INFERENCE_PRECISION_HINT"] = inference_precision
        self.model = self.core.read_model(str(model_xml))
        self.compiled = self.core.compile_model(self.model, device, cfg)
        self.input = self.compiled.inputs[0]
        self.outputs = self.compiled.outputs
        self.device = device
        self.model_xml = str(model_xml)

    def infer(self, x: np.ndarray | Mapping[str, np.ndarray] | Sequence[np.ndarray]) -> list[np.ndarray]:
        if isinstance(x, Mapping):
            r = self.compiled(x)
        elif isinstance(x, Sequence) and not isinstance(x, np.ndarray):
            r = self.compiled(list(x))
        else:
            r = self.compiled([x])
        return [r[o] for o in self.outputs]

    def device_info(self) -> str:
        try:
            return self.core.get_property(self.device, "FULL_DEVICE_NAME")
        except Exception:
            return self.device
