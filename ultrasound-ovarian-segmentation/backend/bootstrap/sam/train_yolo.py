# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fine-tune YOLOv8n on MMOTU-2D to provide the ovarian bbox prompt.

The SAM-256 arch prompts the mask decoder with a bounding box. This stage
fine-tunes YOLOv8n on the detection labels produced by ``prepare_yolo_data.py``
and copies the best checkpoint to ``models/yolov8n_mmotu.pt`` for export.

LICENCE: ``ultralytics`` (YOLOv8) is AGPL-3.0. It is used here BUILD-TIME ONLY
to train and export the detector. The runtime app loads the exported OpenVINO
IR with OpenCV/OpenVINO and never imports ultralytics, so the shipped app is not
a derivative work of ultralytics. If AGPL is unacceptable for your deployment,
substitute any box detector that emits an ``[x, y, w, h]`` lesion box.

Usage (Windows, inside the app venv with backend/requirements-sam.txt installed):

    python -m backend.bootstrap.sam.train_yolo --device cpu

Reference: ~30 epochs at imgsz 640 reaches val mAP50 >= 0.85 on MMOTU-2D, which
is the localisation accuracy behind the reference end-to-end Dice 0.8207.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n on MMOTU-2D for the ovarian bbox prompt."
    )
    ap.add_argument("--data-yaml", default="data/yolo_mmotu/mmotu.yaml",
                    help="Dataset yaml from prepare_yolo_data.py.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Training resolution. YOLO is resolution-flexible; the IR is "
                         "exported at 320 (the shipped operating point) by export_yolo.py.")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu",
                    help="ultralytics device string ('cpu', '0' for CUDA, 'xpu' for Intel GPU).")
    ap.add_argument("--out", default="models/yolov8n_mmotu.pt",
                    help="Where to copy the best checkpoint (consumed by export_yolo.py).")
    args = ap.parse_args()

    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Missing dataset yaml: {data_yaml}\n"
            "Run: python -m backend.bootstrap.sam.prepare_yolo_data"
        )

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - env guard
        raise SystemExit(
            "[train-yolo] ultralytics not installed. Inside the app venv run:\n"
            "    pip install -r backend/requirements-sam.txt\n"
            f"(import error: {type(exc).__name__}: {exc})"
        )

    project_dir = Path("reports") / "yolo_mmotu"
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")
    t0 = time.perf_counter()
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=8,
        device=args.device,
        project=str(project_dir),
        name="train",
        exist_ok=True,
        plots=False,
        verbose=True,
        amp=False,      # disable AMP + AMP pre-check (unstable on Intel XPU / iGPU)
        workers=0,      # Windows-safe dataloader (avoid multiprocessing spawn deadlocks)
    )
    elapsed = time.perf_counter() - t0

    trainer = getattr(model, "trainer", None)
    best = getattr(trainer, "best", None) if trainer is not None else None
    best_path = Path(best) if best else None
    if best_path is None or not best_path.exists():
        candidates = list(project_dir.rglob("best.pt"))
        if not candidates:
            raise FileNotFoundError("Training finished but best.pt was not found")
        best_path = max(candidates, key=lambda p: p.stat().st_mtime)

    out_weights = Path(args.out)
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, out_weights)
    print(f"[train-yolo] saved best weights to {out_weights}")

    metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device, verbose=False)
    map50 = getattr(getattr(metrics, "box", None), "map50", None)
    map5095 = getattr(getattr(metrics, "box", None), "map", None)
    print(f"[train-yolo] val mAP50={map50}  mAP50-95={map5095}")
    print(f"[train-yolo] elapsed={elapsed / 60.0:.1f} min")
    print("[train-yolo] next: python -m backend.bootstrap.sam.export_yolo")


if __name__ == "__main__":
    main()
