# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert MMOTU-2D masks into YOLOv8 detection labels for the ovarian bbox model.

Reads the same ``data/OTU_2d`` layout the DS2Net stages use
(``images/{stem}.JPG``, ``annotations/{stem}_binary.PNG``, ``train.txt`` /
``val.txt``) and writes a YOLOv8 detection dataset under ``data/yolo_mmotu`` —
one ``lesion`` box per image, derived from the largest mask contour — plus the
``mmotu.yaml`` the trainer consumes. Images with no mask contour get a
center-box fallback so every frame has a label.

Usage (Windows, inside the app venv with backend/requirements-sam.txt installed):

    python -m backend.bootstrap.sam.prepare_yolo_data --data data/OTU_2d
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for e in (".JPG", ".jpg", ".jpeg", ".png", ".PNG", ".bmp"):
        p = images_dir / f"{stem}{e}"
        if p.exists():
            return p
    return None


def _find_mask(masks_dir: Path, stem: str) -> Path | None:
    for name in (f"{stem}_binary.PNG", f"{stem}_binary.png", f"{stem}.png", f"{stem}.PNG"):
        p = masks_dir / name
        if p.exists():
            return p
    return None


def _mask_to_bbox(mask_path: Path) -> tuple[int, int, int, int] | None:
    """Largest-contour bounding box (x, y, w, h) in pixels, or None if empty."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w <= 0 or h <= 0:
        return None
    return int(x), int(y), int(w), int(h)


def _write_split(data: Path, dst: Path, split: str) -> int:
    images_dir = data / "images"
    masks_dir = data / "annotations"
    list_file = data / f"{split}.txt"
    if not list_file.exists():
        raise FileNotFoundError(f"Missing split list {list_file}")
    stems = [Path(ln.strip()).stem for ln in list_file.read_text().splitlines() if ln.strip()]

    img_out = dst / "images" / split
    lbl_out = dst / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n = 0
    for stem in stems:
        ip = _find_image(images_dir, stem)
        mp = _find_mask(masks_dir, stem)
        if ip is None or mp is None:
            continue
        img = cv2.imread(str(ip))
        if img is None:
            continue
        height, width = img.shape[:2]
        shutil.copy2(ip, img_out / ip.name)

        bbox = _mask_to_bbox(mp)
        label = lbl_out / f"{stem}.txt"
        if bbox is None:
            label.write_text("0 0.500000 0.500000 0.500000 0.500000\n")
        else:
            x, y, w, h = bbox
            cx = (x + w / 2.0) / width
            cy = (y + h / 2.0) / height
            label.write_text(f"0 {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}\n")
        n += 1
    print(f"[prepare] {split}: {n} images -> {img_out}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MMOTU-2D masks -> YOLOv8 detection labels for the ovarian bbox model."
    )
    ap.add_argument("--data", default="data/OTU_2d",
                    help="MMOTU-2D root (images/, annotations/, train.txt, val.txt).")
    ap.add_argument("--out", default="data/yolo_mmotu",
                    help="Output YOLO dataset dir (default: data/yolo_mmotu).")
    args = ap.parse_args()

    data = Path(args.data)
    if not (data / "images").exists() or not (data / "annotations").exists():
        raise FileNotFoundError(
            f"Expected images/ and annotations/ under {data}. Download OTU_2d from "
            "https://github.com/cv516Buaa/MMOTU_DS2Net and extract it there."
        )

    dst = Path(args.out)
    total = _write_split(data, dst, "train") + _write_split(data, dst, "val")

    yaml_path = dst / "mmotu.yaml"
    yaml_path.write_text(
        f"path: {dst.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: lesion\n"
    )
    print(f"[prepare] wrote {yaml_path}")
    print(f"[prepare] total images: {total}")
    print("[prepare] next: python -m backend.bootstrap.sam.train_yolo")


if __name__ == "__main__":
    main()
