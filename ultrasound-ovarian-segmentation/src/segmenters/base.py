# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Segmenter abstraction: a typed result plus a factory that selects the model.

The app supports two interchangeable ovarian-tumor models behind one interface:

  * ``ds2net`` — promptless SegFormer-B5 semantic segmentation (default).
  * ``sam``    — YOLOv8 bbox prompt -> SAM-256 encoder/decoder instance mask.

Both return a :class:`Result`, so the capture/inference/display pipeline in
``app.py`` is model-agnostic; only the overlay branches on ``Result.kind``.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("segmenter")


@dataclass
class Result:
    """One frame's segmentation output, model-agnostic.

    mask  : native-resolution boolean tumor mask ``(H, W)``, or ``None``.
    boxes : instance boxes to draw as ``(x0, y0, x1, y1)``; empty for promptless
            semantic models (DS2Net), populated by the SAM pipeline.
    kind  : ``"semantic"`` (fill only) or ``"instance"`` (fill + box); selects
            how :class:`~src.display.Presenter` draws the overlay.
    """

    mask: np.ndarray | None
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    kind: str = "semantic"


class BaseSegmenter(ABC):
    """Turn a BGR frame into a :class:`Result`."""

    @abstractmethod
    def infer(self, frame_bgr: np.ndarray) -> Result:
        ...


def resolve_device(device: str, available: list[str], model_label: str) -> str:
    """Map the requested device onto one this product supports.

    These OpenVINO models target the Intel iGPU (GPU). The NPU is not supported;
    fall back to GPU (or CPU) with a clear message instead of failing. CPU is a
    development / sanity fallback only.
    """
    dev = device.upper()
    if dev == "NPU":
        dev = "GPU" if "GPU" in available else "CPU"
        log.warning("NPU is not supported for %s; using %s instead "
                    "(supported: GPU recommended, CPU fallback).", model_label, dev)
    if dev not in available and dev != "AUTO":
        log.warning("device %s not available %s; falling back to CPU", dev, available)
        dev = "CPU"
    return dev


def create_segmenter(cfg) -> BaseSegmenter:
    """Build the segmenter selected by ``cfg.model_arch`` (``ds2net`` | ``sam``)."""
    arch = getattr(cfg, "model_arch", "ds2net").lower()
    if arch == "ds2net":
        from src.segmenters.ds2net import Ds2NetSegmenter
        return Ds2NetSegmenter(cfg.model, device=cfg.device, res=cfg.res)
    if arch == "sam":
        from src.segmenters.sam import SamSegmenter
        return SamSegmenter(
            encoder_xml=cfg.sam_encoder,
            decoder_xml=cfg.sam_decoder,
            yolo_xml=cfg.sam_yolo,
            device=cfg.device,
            enc_res=cfg.res,
        )
    raise ValueError(f"unknown --model-arch {arch!r} (expected 'ds2net' or 'sam')")
