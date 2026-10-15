# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""YOLOv8 bbox prompt for the SAM ovarian pipeline.

Ported verbatim from the ovarian POC (bbox_yolo.py). Letterboxes the frame to
the detector's square input, runs the OpenVINO YOLOv8 IR, decodes + NMS-filters
the output, and returns the highest-scoring box as ``[x0, y0, w, h]`` in native
frame coordinates (a center-box fallback keeps the pipeline alive when the
detector finds nothing).

NOTE: the Ultralytics/YOLOv8 detector weights are licensed AGPL-3.0. This is a
separate obligation from the SAM model (Apache-2.0) and applies only to the
``sam`` architecture's bbox prompt. See third_party_programs / release-notes.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from src.segmenters.ov_runner import OVRunner


def _center_fallback(width: int, height: int) -> list[float]:
    fallback_w = int(width * 0.5)
    fallback_h = int(height * 0.5)
    fallback_x = (width - fallback_w) // 2
    fallback_y = (height - fallback_h) // 2
    return [float(fallback_x), float(fallback_y), float(fallback_w), float(fallback_h)]


class YoloBBox:
    def __init__(self, ir_xml: str | Path, device: str, imgsz: int | None = None,
                 conf: float = 0.25, iou: float = 0.45,
                 nms_impl: str = "cv2"):
        """YOLO OpenVINO wrapper for ovarian ultrasound bbox prompt.

        Parameters
        ----------
        nms_impl : {"cv2", "python"}
            Which NMS implementation to run inside ``detect()``.
            - ``"cv2"``   : cv2.dnn.NMSBoxes (C++ backend, default).
            - ``"python"``: pure-Python ``while``-loop fallback.
        """
        self.ir_xml = Path(ir_xml)
        if not self.ir_xml.exists():
            raise FileNotFoundError(f"Missing YOLO IR: {self.ir_xml}")
        self.device = device
        self.conf = float(conf)
        self.iou = float(iou)
        self.last_ms = -1.0

        nms_impl = str(nms_impl).lower()
        if nms_impl not in ("cv2", "python"):
            raise ValueError(f"nms_impl must be 'cv2' or 'python', got {nms_impl!r}")
        self.nms_impl = nms_impl

        self.runner = OVRunner(self.ir_xml, device)

        # Auto-detect imgsz from the compiled model's input shape [1,3,H,W]
        # so callers don't have to know whether the IR was exported at 640, 320, etc.
        try:
            in_shape = self.runner.input.get_partial_shape()
            h = in_shape[2].get_length() if in_shape[2].is_static else None
            w = in_shape[3].get_length() if in_shape[3].is_static else None
            if h and w and h == w:
                detected = int(h)
            else:
                detected = None
        except Exception:
            detected = None

        if imgsz is None:
            self.imgsz = detected if detected else 640
        else:
            self.imgsz = int(imgsz)

    def _letterbox(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        height, width = image_bgr.shape[:2]
        scale = min(self.imgsz / width, self.imgsz / height)
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = self.imgsz - new_w
        pad_h = self.imgsz - new_h
        left = pad_w // 2
        top = pad_h // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas, scale, left, top

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
        """Vectorised NMS via cv2.dnn.NMSBoxes (C++ backend).

        Returns indices into ``boxes`` / ``scores`` in descending score order,
        so ``keep[0]`` is the highest-scoring survivor.
        """
        n = int(boxes.shape[0])
        if n == 0:
            return []
        if n == 1:
            return [0]

        # cv2.dnn.NMSBoxes expects [x, y, w, h] (top-left + width/height).
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        w = np.clip(boxes[:, 2] - x1, a_min=0.0, a_max=None)
        h = np.clip(boxes[:, 3] - y1, a_min=0.0, a_max=None)
        boxes_ltwh = np.stack([x1, y1, w, h], axis=1).astype(np.float32)

        try:
            indices = cv2.dnn.NMSBoxes(
                bboxes=boxes_ltwh.tolist(),
                scores=scores.astype(np.float32).tolist(),
                score_threshold=0.0,  # upstream filter already applied self.conf
                nms_threshold=float(iou_thres),
            )
        except Exception:
            return YoloBBox._nms_python(boxes, scores, iou_thres)

        if indices is None:
            return []
        indices = np.asarray(indices).reshape(-1)
        if indices.size == 0:
            return []
        return indices.astype(int).tolist()

    @staticmethod
    def _nms_python(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
        """Pure-Python NMS (fallback for the cv2 path)."""
        if boxes.size == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter + 1e-9
            iou = inter / union
            order = order[1:][iou <= iou_thres]
        return keep

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def detect(self, image_bgr: np.ndarray) -> list[float]:
        height, width = image_bgr.shape[:2]
        letterboxed, scale, pad_x, pad_y = self._letterbox(image_bgr)
        image = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))[None].astype(np.float32)

        t0 = time.perf_counter()
        outputs = self.runner.infer(image)
        self.last_ms = (time.perf_counter() - t0) * 1000.0

        if not outputs:
            return _center_fallback(width, height)

        pred = np.array(outputs[0])
        if pred.ndim == 3:
            pred = pred[0]
        if pred.ndim != 2:
            return _center_fallback(width, height)

        if pred.shape[0] < pred.shape[1] and pred.shape[0] in (5, 6, 7, 84, 85):
            pred = pred.T

        if pred.shape[1] < 5:
            return _center_fallback(width, height)

        boxes_xywh = pred[:, :4]
        tail = pred[:, 4:]
        if tail.shape[1] == 1:
            scores = self._sigmoid(tail[:, 0])
        elif tail.shape[1] == 2:
            scores = self._sigmoid(tail[:, 0]) * self._sigmoid(tail[:, 1])
        elif tail.shape[1] > 2:
            objectness = self._sigmoid(tail[:, 0])
            class_scores = self._sigmoid(tail[:, 1:]).max(axis=1)
            scores = objectness * class_scores
        else:
            scores = np.zeros((pred.shape[0],), dtype=np.float32)
        keep_mask = scores >= self.conf
        if not np.any(keep_mask):
            return _center_fallback(width, height)

        boxes_xywh = boxes_xywh[keep_mask]
        scores = scores[keep_mask]

        cx = boxes_xywh[:, 0]
        cy = boxes_xywh[:, 1]
        bw = boxes_xywh[:, 2]
        bh = boxes_xywh[:, 3]
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        if self.nms_impl == "python":
            keep = self._nms_python(boxes_xyxy, scores, self.iou)
        else:
            keep = self._nms(boxes_xyxy, scores, self.iou)
        if not keep:
            return _center_fallback(width, height)

        best = keep[0]
        x1, y1, x2, y2 = boxes_xyxy[best]
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            return _center_fallback(width, height)
        return [x1, y1, x2 - x1, y2 - y1]
