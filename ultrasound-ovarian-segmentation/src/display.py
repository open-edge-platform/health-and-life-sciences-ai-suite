# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cv2 pop-up presenter — overlay the tumor mask and a GPU/FPS HUD on each frame.

Draws the latest segmentation mask (semi-transparent fill + contour) on every
displayed frame and a heads-up panel showing display/inference FPS and the live
iGPU utilization vs. the governor's cap, so the KPI status is visible while the
app runs. Optionally writes an annotated .mp4. Windows pop-up window via OpenCV.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

log = logging.getLogger("display")


class _Rate:
    def __init__(self) -> None:
        self._t = time.monotonic()
        self._n = 0
        self.fps = 0.0

    def tick(self) -> None:
        self._n += 1
        now = time.monotonic()
        if now - self._t >= 1.0:
            self.fps = self._n / (now - self._t)
            self._t, self._n = now, 0


class Presenter:
    def __init__(self, window: str, scale: float, alpha: float,
                 color: tuple[int, int, int], headless: bool = False,
                 record: str | None = None, src=None,
                 view: str = "side_by_side") -> None:
        self.window = window
        self.scale = float(scale)
        self.alpha = float(alpha)
        self.color = tuple(int(c) for c in color)
        self.box_color = (0, 255, 0)  # bbox prompt overlay (SAM instance kind)
        self.headless = headless
        self.view = view if view in ("overlay", "side_by_side") else "side_by_side"
        self.side_by_side = self.view == "side_by_side"
        self._rate = _Rate()
        # The writer is opened lazily on the first frame, once the composed
        # display size is known (side-by-side doubles the frame width).
        self._record_path = record or None
        self._record_fps = src.fps if (src is not None and src.fps > 0) else 30.0
        self._writer = None
        self._writer_size = (0, 0)
        if not headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def _overlay(self, frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        if mask is None:
            return frame.copy()
        m = mask.astype(bool)
        if m.shape[:2] != frame.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (frame.shape[1], frame.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        out = frame.copy()
        fill = np.zeros_like(frame)
        fill[m] = self.color
        out = cv2.addWeighted(out, 1.0, fill, self.alpha, 0.0)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, self.color, 2)
        return out

    def _hud(self, frame: np.ndarray, hud: dict) -> np.ndarray:
        self._rate.tick()
        arch = {"ds2net": "DS2Net", "sam": "SAM-256"}.get(hud.get("arch", ""), hud.get("arch", ""))
        head = f"[{arch}]  " if arch else ""
        lines = [f"{head}display {self._rate.fps:4.1f} fps   inference {hud['inf_fps']:4.1f} fps "
                 f"({hud['inf_ms']:4.0f} ms)"]
        if hud.get("gpu_available"):
            verdict = "PASS" if hud["within_cap"] else "OVER"
            status = "THROTTLING" if hud["throttle_ms"] > 0 else "ok"
            lines.append(f"iGPU busiest  peak {hud['gpu_peak']:4.1f}%  mean {hud['gpu_mean']:4.1f}%  "
                         f"/ cap {hud['gpu_cap']:.0f}%  [{verdict}]")
            lines.append(f"governor {status}  (+{hud['throttle_ms']:.0f} ms/frame)")
        else:
            lines.append("iGPU governor: counters unavailable (non-Windows host)")
        y = 26
        for ln in lines:
            cv2.putText(frame, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y += 26
        return frame

    def _draw_boxes(self, frame: np.ndarray, boxes) -> None:
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), self.box_color, 2)

    def _caption(self, img: np.ndarray, text: str) -> None:
        """Small bottom-left caption on a pane (used to label side-by-side panes)."""
        y = img.shape[0] - 12
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def _compose(self, raw: np.ndarray, seg: np.ndarray) -> np.ndarray:
        """Build the display image: 'Original | Segmentation' or just the segmented frame."""
        if not self.side_by_side:
            return seg
        left = raw.copy()
        right = seg
        if left.shape[0] != right.shape[0]:
            right = cv2.resize(right, (right.shape[1], left.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        self._caption(left, "Original")
        self._caption(right, "Segmentation")
        sep = np.full((left.shape[0], 2, 3), (60, 60, 60), dtype=np.uint8)
        return np.hstack([left, sep, right])

    def _record_frame(self, img: np.ndarray) -> None:
        """Open the writer lazily (sized to the composed frame) and write it."""
        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer_size = (img.shape[1], img.shape[0])
            self._writer = cv2.VideoWriter(self._record_path, fourcc,
                                           self._record_fps, self._writer_size)
            log.info("recording annotated output -> %s", self._record_path)
        rec = img
        if (img.shape[1], img.shape[0]) != self._writer_size:
            rec = cv2.resize(img, self._writer_size, interpolation=cv2.INTER_AREA)
        self._writer.write(rec)

    def show(self, frame: np.ndarray, result, hud: dict) -> int:
        """Draw the view + HUD, present, and return the pressed key (-1 if none).

        ``result`` is a :class:`~src.segmenters.Result` (or ``None``). The mask is
        filled for every model; instance models (SAM) also get their prompt box
        drawn. In the default side-by-side view the original frame is shown next
        to the segmented frame; ``--view overlay`` draws the mask on the frame.
        Returns 27 (Esc) if the user closed the window via the [X] button, so the
        caller stops the run instead of inferring against a dead window.
        """
        mask = result.mask if result is not None else None
        boxes = result.boxes if result is not None else []
        seg = self._overlay(frame, mask)
        if boxes:
            self._draw_boxes(seg, boxes)
        disp = self._compose(frame, seg)
        disp = self._hud(disp, hud)
        if self._record_path is not None:
            self._record_frame(disp)
        if self.headless:
            return -1
        if self.scale != 1.0:
            disp = cv2.resize(disp, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_LINEAR)
        cv2.imshow(self.window, disp)
        key = cv2.waitKey(1) & 0xFF
        try:
            if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                return 27  # window closed via the [X] button
        except cv2.error:
            return 27
        return key

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
        if not self.headless:
            try:
                cv2.destroyWindow(self.window)
            except Exception:  # noqa: BLE001
                pass
