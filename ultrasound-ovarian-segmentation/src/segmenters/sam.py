# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenVINO SAM-256 segmenter — YOLOv8 bbox prompt -> SAM encoder/decoder mask.

Replicates, verbatim, the ovarian POC pipeline that was benchmarked and accepted
by the customer (ITEP-91363): a YOLOv8n@320 detector proposes a tumor box, a SAM
ViT-B image encoder (stock Meta SAM fine-tuned on MMOTU-2D @256²) embeds the
frame, and Meta's stock SAM multi-mask decoder turns the box prompt into three
candidate masks; the highest predicted-IoU candidate is kept (Dice ~0.82).

Model licensing: the SAM encoder/decoder path is Apache-2.0 (stock Meta SAM +
MMOTU-2D). The YOLOv8/Ultralytics detector used for the box prompt is AGPL-3.0 —
a separate obligation that applies only to this architecture. See release-notes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from src.segmenters.base import BaseSegmenter, Result, resolve_device
from src.segmenters.ov_runner import OVRunner
from src.segmenters.yolo import YoloBBox

log = logging.getLogger("segmenter")


def preprocess_encoder(image_bgr: np.ndarray, enc_res: int) -> np.ndarray:
    """Resize to the encoder resolution, BGR->RGB, /255, NCHW. No ImageNet norm."""
    im = cv2.resize(image_bgr, (enc_res, enc_res))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(im, (2, 0, 1))[None].astype(np.float32)


def bbox_to_decoder_inputs(bbox: list[float], image_shape: tuple[int, ...],
                           embed: np.ndarray, enc_res: int) -> dict[str, np.ndarray]:
    """Turn an ``[x0, y0, w, h]`` box into SAM decoder inputs (box prompt).

    The two box corners become point prompts with SAM's box labels (2 = top-left,
    3 = bottom-right), scaled from native pixels into the encoder's grid.
    """
    x0, y0, bw, bh = bbox
    ih, iw = image_shape[:2]
    sx, sy = float(enc_res) / iw, float(enc_res) / ih
    coords = np.array([[
        [x0 * sx, y0 * sy],
        [(x0 + bw) * sx, (y0 + bh) * sy],
    ]], dtype=np.float32)
    labels = np.array([[2, 3]], dtype=np.float32)
    mask_in = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask = np.array([0.0], dtype=np.float32)
    orig_sz = np.array([ih, iw], dtype=np.int32)
    embed_in = embed if embed.ndim == 4 else embed[np.newaxis]
    return {
        "image_embeddings": embed_in,
        "point_coords": coords,
        "point_labels": labels,
        "mask_input": mask_in,
        "has_mask_input": has_mask,
        "orig_im_size": orig_sz,
    }


def decode_multimask(decoder: OVRunner, embed: np.ndarray, bbox: list[float],
                     image_shape: tuple[int, ...], enc_res: int) -> tuple[np.ndarray, np.ndarray]:
    """Run the multi-mask decoder: return (low-res masks [3,H,W], iou scores [3])."""
    dec_out = decoder.infer(bbox_to_decoder_inputs(bbox, image_shape, embed, enc_res))
    outs = list(dec_out.values()) if isinstance(dec_out, dict) else list(dec_out)
    masks = np.array(outs[0]).squeeze(0)      # [1,3,H,W] -> [3,H,W]
    iou_scores = np.array(outs[1]).squeeze(0)  # [1,3] -> [3]
    return masks, iou_scores


def pick_best_mask(low_res_3xHxW: np.ndarray, iou_scores_3: np.ndarray,
                   image_shape: tuple[int, ...]) -> tuple[np.ndarray, int, float]:
    """Argmax over predicted-IoU, threshold at 0, nearest-resize to native."""
    idx = int(np.argmax(iou_scores_3))
    chosen = low_res_3xHxW[idx]
    mask_small = (chosen > 0).astype(np.uint8)
    ih, iw = image_shape[:2]
    mask = cv2.resize(mask_small, (iw, ih), interpolation=cv2.INTER_NEAREST)
    return mask, idx, float(iou_scores_3[idx])


class SamSegmenter(BaseSegmenter):
    """YOLO bbox -> SAM encoder -> SAM multi-mask decoder -> best-IoU tumor mask."""

    def __init__(self, encoder_xml: str, decoder_xml: str, yolo_xml: str,
                 device: str = "GPU", enc_res: int = 256) -> None:
        self.enc_res = int(enc_res)
        self._enc_path = Path(encoder_xml)
        self._dec_path = Path(decoder_xml)
        self._yolo_path = Path(yolo_xml)
        for label, path in (("SAM encoder", self._enc_path),
                            ("SAM decoder", self._dec_path),
                            ("YOLO detector", self._yolo_path)):
            if not path.exists():
                raise FileNotFoundError(
                    f"{label} IR not found: {path} — run model-prep for the SAM arch or copy the "
                    f"exported IRs into models/ (see docs/user-guide/get-started/model-preparation.md)"
                )

        available = ov.Core().available_devices
        self.device = resolve_device(device, available, "the SAM-256 model")

        self.encoder = OVRunner(self._enc_path, self.device)
        self.decoder = OVRunner(self._dec_path, self.device)
        self._check_encoder_res(self.encoder)
        self.yolo = YoloBBox(self._yolo_path, self.device)
        log.info("segmenter ready: device=%s (%s) enc=%s dec=%s yolo=%s res=%d",
                 self.device, self.encoder.device_info(), self._enc_path.name,
                 self._dec_path.name, self._yolo_path.name, self.enc_res)
        self._warmup()

    def _check_encoder_res(self, encoder: OVRunner) -> None:
        """Fail fast if --res disagrees with the encoder IR's static input resolution."""
        try:
            ps = encoder.input.get_partial_shape()
            if len(ps) != 4 or ps[2].is_dynamic or ps[3].is_dynamic:
                return
            ih, iw = ps[2].get_length(), ps[3].get_length()
        except Exception:  # noqa: BLE001
            return
        if ih != iw:
            raise ValueError(
                f"SAM encoder IR input is non-square ({ih}x{iw}); expected a square IR "
                f"({self._enc_path.name})."
            )
        if ih != self.enc_res:
            raise ValueError(
                f"--res {self.enc_res} does not match the SAM encoder IR input resolution {ih} "
                f"({self._enc_path.name}). Re-run with --res {ih}, or re-export the encoder at "
                f"{self.enc_res}."
            )

    def _warmup(self) -> None:
        try:
            self.infer(np.zeros((self.enc_res, self.enc_res, 3), np.uint8))
            log.info("warmup inference done")
        except Exception as exc:  # noqa: BLE001
            log.warning("warmup failed: %s", exc)

    def infer(self, frame_bgr: np.ndarray) -> Result:
        """Return a boolean tumor mask + the prompt box at the frame's native (H, W)."""
        # 1. YOLO bbox prompt ([x0, y0, w, h] in native pixels).
        bbox = self.yolo.detect(frame_bgr)
        # 2. Encode the frame at the encoder resolution.
        embed, = self.encoder.infer(preprocess_encoder(frame_bgr, self.enc_res))
        # 3-4. Multi-mask decode the box prompt and keep the highest predicted-IoU mask.
        masks3, iou3 = decode_multimask(self.decoder, embed, bbox, frame_bgr.shape, self.enc_res)
        mask, _idx, _score = pick_best_mask(masks3, iou3, frame_bgr.shape)
        x0, y0, bw, bh = (int(round(v)) for v in bbox)
        boxes = [(x0, y0, x0 + bw, y0 + bh)]
        return Result(mask=mask.astype(bool), boxes=boxes, kind="instance")
