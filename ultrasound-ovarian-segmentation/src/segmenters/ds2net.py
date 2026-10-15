# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenVINO DS2Net segmenter — promptless SegFormer-B5 ovarian-tumor segmentation.

Takes a BGR numpy frame and returns a boolean tumor mask at the frame's native
resolution. Preprocessing matches the trainer / evaluator exactly (RGB, resize
to the network resolution, /255, ImageNet normalisation, NCHW) so the live
overlay reflects the same model that was scored on the MMOTU-2D val split
(Dice 0.8657 / IoU 0.7913). Pure OpenVINO — no PyTorch, no prompts, no detector.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from src.segmenters.base import BaseSegmenter, Result, resolve_device

log = logging.getLogger("segmenter")

# ImageNet normalisation, matching train_segformer_mmotu.py / eval_seg_dice_segformer.py.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class Ds2NetSegmenter(BaseSegmenter):
    """Compile a SegFormer IR to a device and turn frames into tumor masks."""

    def __init__(self, model_xml: str, device: str = "GPU", res: int = 256) -> None:
        self.res = int(res)
        self._path = Path(model_xml)
        if not self._path.exists():
            raise FileNotFoundError(
                f"model IR not found: {self._path} — run model-prep or copy an exported IR "
                f"into models/ (see docs/user-guide/get-started/model-preparation.md)"
            )

        core = ov.Core()
        available = core.available_devices
        self.device = resolve_device(device, available, "this DS2Net SegFormer-B5 model")

        cfg: dict[str, str] = {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "ALLOW_AUTO_BATCHING": "NO",
        }
        model = core.read_model(str(self._path))
        self._check_input_res(model)
        self._compiled = core.compile_model(model, self.device, cfg)
        self._out = self._compiled.outputs[0]
        try:
            name = core.get_property(self.device, "FULL_DEVICE_NAME")
        except Exception:
            name = self.device
        log.info("segmenter ready: device=%s (%s) ir=%s res=%d", self.device, name, self._path.name, self.res)
        self._warmup()

    def _check_input_res(self, model) -> None:
        """Fail fast if --res disagrees with the IR's static input resolution.

        A mismatch otherwise makes every inference silently return a blank mask
        (the warmup throws, then ``infer`` falls back to zeros), so validate here.
        """
        try:
            ps = model.inputs[0].partial_shape
            if len(ps) != 4 or ps[2].is_dynamic or ps[3].is_dynamic:
                return  # dynamic input tolerates any --res
            ih, iw = ps[2].get_length(), ps[3].get_length()
        except Exception:  # noqa: BLE001
            return  # unknown layout; don't block on the check itself
        if ih != iw:
            raise ValueError(
                f"IR input is non-square ({ih}x{iw}); this app expects a square IR "
                f"({self._path.name}). Re-export with backend.bootstrap.export --res <N>."
            )
        if ih != self.res:
            raise ValueError(
                f"--res {self.res} does not match the IR input resolution {ih} "
                f"({self._path.name}). Re-run with --res {ih}, or re-export the IR at "
                f"{self.res} (backend.bootstrap.export --from-pretrained <ckpt> --res {self.res})."
            )

    def _warmup(self) -> None:
        try:
            self._compiled([np.zeros((1, 3, self.res, self.res), np.float32)])
            log.info("warmup inference done")
        except Exception as exc:  # noqa: BLE001
            log.warning("warmup failed: %s", exc)

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        r = cv2.resize(rgb, (self.res, self.res), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        r = (r - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(r.transpose(2, 0, 1)[None])  # [1,3,res,res]

    def infer(self, frame_bgr: np.ndarray) -> Result:
        """Return a boolean tumor mask at the frame's native (H, W)."""
        h0, w0 = frame_bgr.shape[:2]
        blob = self._preprocess(frame_bgr)
        logits = np.squeeze(np.asarray(self._compiled([blob])[self._out]))  # [2,h,w]
        if logits.ndim != 3 or logits.shape[0] < 2:
            return Result(mask=np.zeros((h0, w0), bool), kind="semantic")
        # Upsample the 2 class logits to the network resolution, argmax, then
        # nearest-resize the binary mask to the native frame size.
        bg = cv2.resize(logits[0], (self.res, self.res), interpolation=cv2.INTER_LINEAR)
        fg = cv2.resize(logits[1], (self.res, self.res), interpolation=cv2.INTER_LINEAR)
        mask_res = (fg > bg).astype(np.uint8)
        mask = cv2.resize(mask_res, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
        return Result(mask=mask, kind="semantic")
