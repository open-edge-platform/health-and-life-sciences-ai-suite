# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frame sources — one interface for a video file, a USB/webcam, or a folder of
ultrasound stills.

Every source yields BGR numpy frames via ``read()`` so the rest of the app is
source-agnostic: the capture/inference/display pipeline works identically for a
recorded clip, a live probe feed, or a directory of exported frames.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

log = logging.getLogger("sources")

_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class Source(Protocol):
    name: str
    width: int
    height: int
    fps: float
    is_live: bool

    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...


class FileSource:
    """Video file via OpenCV. Loops on EOF so demos run continuously."""

    is_live = False

    def __init__(self, path: str, loop: bool = True) -> None:
        self.name = f"file:{path}"
        self._loop = loop
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video file: {path}")
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok:
            if not self._loop:
                return None
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()


class WebcamSource:
    """Generic USB / webcam via OpenCV (device index). Newest-frame-wins."""

    is_live = True

    def __init__(self, index: str | int, width: int, height: int, fps: float) -> None:
        self.name = f"webcam:{index}"
        dev = int(index) if str(index).isdigit() else index
        self._cap = cv2.VideoCapture(dev)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open webcam: {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # newest-frame-wins
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or fps

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()


class FolderSource:
    """A directory of ultrasound stills, read in sorted order and looped."""

    is_live = False

    def __init__(self, folder: str, fps: float, loop: bool = True) -> None:
        self.name = f"folder:{folder}"
        self._loop = loop
        self._i = 0
        self._files = sorted(
            p for p in Path(folder).iterdir() if p.suffix.lower() in _IMG_EXT
        )
        if not self._files:
            raise RuntimeError(f"no images found in folder: {folder}")
        first = cv2.imread(str(self._files[0]))
        if first is None:
            raise RuntimeError(f"cannot read image: {self._files[0]}")
        self.height, self.width = first.shape[:2]
        self.fps = float(fps) if fps > 0 else 10.0

    def read(self) -> np.ndarray | None:
        n = len(self._files)
        for _ in range(n):  # bounded scan: skip unreadable images without recursing
            if self._i >= n:
                if not self._loop:
                    return None
                self._i = 0
            path = self._files[self._i]
            self._i += 1
            img = cv2.imread(str(path))
            if img is None:
                log.warning("skipping unreadable image: %s", path)
                continue
            return img
        log.error("no readable images in folder (all %d entries failed to decode)", n)
        return None

    def close(self) -> None:
        return None


def create_source(cfg) -> Source:
    s = cfg.source.lower()
    if s == "file":
        if not cfg.input:
            raise ValueError("--input <video> is required for --source file")
        return FileSource(cfg.input, loop=cfg.loop)
    if s == "webcam":
        return WebcamSource(cfg.input or "0", cfg.width, cfg.height, cfg.target_fps)
    if s == "folder":
        if not cfg.input:
            raise ValueError("--input <folder> is required for --source folder")
        return FolderSource(cfg.input, cfg.target_fps, loop=cfg.loop)
    raise ValueError(f"unknown source: {cfg.source}")
