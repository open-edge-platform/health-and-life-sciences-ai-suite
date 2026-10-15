# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pluggable ovarian-tumor segmenters (DS2Net semantic | SAM-256 instance)."""
from __future__ import annotations

from src.segmenters.base import BaseSegmenter, Result, create_segmenter, resolve_device

__all__ = ["BaseSegmenter", "Result", "create_segmenter", "resolve_device"]
