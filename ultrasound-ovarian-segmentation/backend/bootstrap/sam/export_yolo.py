# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export the trained YOLOv8n MMOTU-2D detector to a fp16 OpenVINO IR.

Exports at ``imgsz 320`` — the shipped operating point for the bbox stage
(~5 ms on the iGPU) — and stages the result as
``models/yolo_mmotu_320/yolov8n_mmotu.xml``, exactly the default the app loads
(see ``src/config.py`` ``--sam-yolo``). The runtime reads this IR directly with
OpenVINO; ultralytics is not imported at runtime.

LICENCE: ``ultralytics`` (YOLOv8) is AGPL-3.0 and used BUILD-TIME ONLY — see
``train_yolo.py``.

Usage (Windows, inside the app venv with backend/requirements-sam.txt installed):

    python -m backend.bootstrap.sam.export_yolo
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _stage_export(src: Path | str, dst_dir: Path, stem: str = "yolov8n_mmotu") -> Path:
    """Copy an ultralytics OpenVINO export dir to ``dst_dir`` renaming model.* -> stem.*."""
    src_path = Path(src)
    src_dir = src_path if src_path.is_dir() else src_path.parent
    if not list(src_dir.rglob("*.xml")):
        raise FileNotFoundError(f"No .xml file found under export directory: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for child in dst_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    for file in src_dir.iterdir():
        if not file.is_file():
            continue
        target = file.name
        if file.name == "model.xml":
            target = f"{stem}.xml"
        elif file.name == "model.bin":
            target = f"{stem}.bin"
        shutil.copy2(file, dst_dir / target)
    return dst_dir / f"{stem}.xml"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export trained YOLOv8n MMOTU-2D detector to a fp16 OpenVINO IR."
    )
    ap.add_argument("--weights", default="models/yolov8n_mmotu.pt",
                    help="Trained YOLO checkpoint from train_yolo.py.")
    ap.add_argument("--imgsz", type=int, default=320,
                    help="Export resolution (shipped operating point = 320).")
    ap.add_argument("--out", default="models/yolo_mmotu_320",
                    help="Output IR dir (default: models/yolo_mmotu_320).")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Missing YOLO weights: {weights}\n"
            "Run: python -m backend.bootstrap.sam.train_yolo"
        )

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - env guard
        raise SystemExit(
            "[export-yolo] ultralytics not installed. Inside the app venv run:\n"
            "    pip install -r backend/requirements-sam.txt\n"
            f"(import error: {type(exc).__name__}: {exc})"
        )

    model = YOLO(str(weights))
    print(f"[export-yolo] exporting fp16 OpenVINO IR at imgsz {args.imgsz} ...")
    result = model.export(format="openvino", half=True, imgsz=args.imgsz, dynamic=False)
    xml = _stage_export(result, Path(args.out))
    print(f"[export-yolo] IR -> {xml}")
    print("[export-yolo] next: python -m backend.bootstrap.sam.export_encoder "
          "and export_decoder, then eval.")


if __name__ == "__main__":
    main()
