# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime configuration for the ovarian segmentation app.

Every knob is exposed via both a CLI flag and an environment variable (CLI
wins), so the app can be driven from ``run.ps1`` or a plain ``python -m src.app``
invocation. Windows-only runtime; no platform-specific knobs are required.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _color(spec: str) -> tuple[int, int, int]:
    """Parse a ``B,G,R`` triplet (OpenCV order); fall back to red on error."""
    try:
        b, g, r = (int(x) for x in spec.split(","))
        return (b, g, r)
    except Exception:
        return (0, 0, 255)


# In-tree IRs produced by model-prep (backend/bootstrap) or copied from an export.
_DEFAULT_MODEL = os.path.join("models", "ds2net_segformer_b5_256", "model.xml")
_DEFAULT_SAM_ENCODER = os.path.join("models", "sam_encoder_256_stockft", "encoder.xml")
_DEFAULT_SAM_DECODER = os.path.join("models", "sam_decoder_256_multimask", "decoder.xml")
_DEFAULT_SAM_YOLO = os.path.join("models", "yolo_mmotu_320", "yolov8n_mmotu.xml")


@dataclass
class Config:
    # source
    source: str          # file | webcam | folder
    input: str           # video path | webcam index | folder of stills
    loop: bool           # loop file / folder sources on EOF
    width: int
    height: int
    target_fps: float    # pacing for folder / webcam sources

    # model / inference
    device: str          # CPU | GPU | NPU | AUTO
    model_arch: str      # ds2net (promptless SegFormer) | sam (YOLO bbox + SAM-256)
    model: str           # ds2net: path to model.xml
    sam_encoder: str     # sam: SAM encoder IR
    sam_decoder: str     # sam: SAM multimask decoder IR
    sam_yolo: str        # sam: YOLOv8 detector IR
    res: int             # network input resolution (square)
    frame_skip: int      # run inference every Nth captured frame

    # GPU utilization cap
    gpu_cap: float       # hard iGPU busiest-engine ceiling the governor enforces
    no_governor: bool    # disable the governor (measure uncapped utilization)

    # display / output
    alpha: float                 # mask overlay opacity
    mask_color: tuple[int, int, int]
    display_scale: float
    window: str
    headless: bool
    record: str | None           # path to write an annotated .mp4
    view: str                    # overlay | side_by_side (original | segmented)
    log_file: str | None         # opt-in: also write the run log (incl. KPIs) here


def parse_config(argv: list[str] | None = None) -> Config:
    p = argparse.ArgumentParser(
        description="Ultrasound ovarian tumor segmentation (DS2Net SegFormer-B5 or SAM-256, OpenVINO)."
    )

    # source
    p.add_argument("--source", default=_env("SOURCE", "file"),
                   choices=["file", "webcam", "folder"])
    p.add_argument("--input", default=_env("INPUT", ""),
                   help="video file path | webcam index | folder of ultrasound stills")
    p.add_argument("--no-loop", dest="loop", action="store_false",
                   default=_env("LOOP", "1") != "0")
    p.add_argument("--width", type=int, default=int(_env("WIDTH", "1280")))
    p.add_argument("--height", type=int, default=int(_env("HEIGHT", "720")))
    p.add_argument("--target-fps", type=float, default=float(_env("TARGET_FPS", "30")),
                   help="playback/capture pacing for folder and webcam sources")

    # model / inference
    p.add_argument("--device", default=_env("DEVICE", "GPU").upper(),
                   choices=["CPU", "GPU", "NPU", "AUTO"])
    p.add_argument("--model-arch", default=_env("MODEL_ARCH", "ds2net"),
                   choices=["ds2net", "sam"],
                   help="segmentation model: 'ds2net' (promptless SegFormer-B5, default) or "
                        "'sam' (YOLOv8 bbox prompt + SAM-256 encoder/decoder)")
    p.add_argument("--model", default=_env("MODEL", _DEFAULT_MODEL),
                   help="ds2net arch: path to the OpenVINO IR model.xml")
    p.add_argument("--sam-encoder", default=_env("SAM_ENCODER", _DEFAULT_SAM_ENCODER),
                   help="sam arch: path to the SAM encoder IR (encoder.xml)")
    p.add_argument("--sam-decoder", default=_env("SAM_DECODER", _DEFAULT_SAM_DECODER),
                   help="sam arch: path to the SAM multimask decoder IR (decoder.xml)")
    p.add_argument("--sam-yolo", default=_env("SAM_YOLO", _DEFAULT_SAM_YOLO),
                   help="sam arch: path to the YOLOv8 detector IR (yolov8n_mmotu.xml)")
    p.add_argument("--res", type=int, default=int(_env("RES", "256")),
                   help="network input resolution (must match the exported IR; 256 for both archs)")
    p.add_argument("--frame-skip", type=int, default=int(_env("FRAME_SKIP", "1")),
                   help="run inference every Nth captured frame (1 = every frame)")

    # GPU utilization cap
    p.add_argument("--gpu-cap", type=float, default=float(_env("GPU_CAP", "80")),
                   help="hard iGPU busiest-engine utilization ceiling (KPI, default 80%%)")
    p.add_argument("--no-governor", action="store_true",
                   default=_env("NO_GOVERNOR", "0") != "0",
                   help="disable the GPU governor (measure uncapped utilization)")

    # display / output
    p.add_argument("--alpha", type=float, default=float(_env("ALPHA", "0.45")),
                   help="tumor mask overlay opacity (0-1)")
    p.add_argument("--mask-color", default=_env("MASK_COLOR", "0,0,255"),
                   help="overlay colour as B,G,R (default red = 0,0,255)")
    p.add_argument("--display-scale", type=float, default=float(_env("DISPLAY_SCALE", "1.0")))
    p.add_argument("--window", default=_env("WINDOW", "Ovarian Tumor Segmentation"))
    p.add_argument("--view", default=_env("VIEW", "side_by_side"),
                   choices=["overlay", "side_by_side"],
                   help="pop-up layout: 'side_by_side' (original | segmented, default) or "
                        "'overlay' (mask drawn directly on the frame)")
    p.add_argument("--headless", action="store_true", default=_env("HEADLESS", "0") != "0",
                   help="no window (useful for recording or soak tests)")
    p.add_argument("--record", default=_env("RECORD", ""),
                   help="path to write an annotated .mp4 of the output")
    p.add_argument("--log-file", default=_env("LOG_FILE", ""),
                   help="also write the run log (incl. KPI verdicts) to this file; "
                        "off by default (stderr only). e.g. reports\\run.log")

    a = p.parse_args(argv)
    return Config(
        source=a.source,
        input=a.input,
        loop=a.loop,
        width=a.width,
        height=a.height,
        target_fps=a.target_fps,
        device=a.device,
        model_arch=a.model_arch,
        model=a.model,
        sam_encoder=a.sam_encoder,
        sam_decoder=a.sam_decoder,
        sam_yolo=a.sam_yolo,
        res=a.res,
        frame_skip=max(1, a.frame_skip),
        gpu_cap=a.gpu_cap,
        no_governor=a.no_governor,
        alpha=a.alpha,
        mask_color=_color(a.mask_color),
        display_scale=a.display_scale,
        window=a.window,
        headless=a.headless,
        record=a.record or None,
        view=a.view,
        log_file=a.log_file or None,
    )
