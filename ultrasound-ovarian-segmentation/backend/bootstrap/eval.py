# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dice / IoU of an exported SegFormer IR on the MMOTU-2D val split.

Verifies the IR produced by ``export.py`` before it is used in a demo. Promptless:
the IR takes the whole image, no bbox / detector. The per-image pipeline mirrors
the app segmenter (``src/segmenters/ds2net.py``) exactly, so the Dice reported
here is what the live overlay reflects:

    PIL RGB -> resize(res,res) -> ImageNet-normalise -> IR -> logits[1,2,res/4,res/4]
    -> bilinear upsample logits to (res,res) -> argmax -> mask@res
    -> NEAREST resize to native -> Dice / IoU vs native GT ({stem}_binary.PNG).

Reference result at res 256: Dice mean 0.8657 / median 0.922, IoU mean 0.7913
over the 469-image val split. An IR exported from random weights scores ~0.

Usage (Windows, inside the app venv):

    python -m backend.bootstrap.eval ^
        --ir models/ds2net_segformer_b5_256/model.xml --res 256 ^
        --data data/OTU_2d --device GPU
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image

# ImageNet normalisation, matching train.py / export.py / the app segmenter.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def dice_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> tuple[float, float]:
    """(Dice, IoU) for two binary masks: 2*inter/(|p|+|g|), inter/union."""
    p = pred_bin.astype(bool)
    g = gt_bin.astype(bool)
    inter = int(np.logical_and(p, g).sum())
    ps = int(p.sum())
    gs = int(g.sum())
    union = ps + gs - inter
    dice = (2.0 * inter) / (ps + gs) if (ps + gs) > 0 else 0.0
    iou = inter / union if union > 0 else 0.0
    return float(dice), float(iou)


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


def summarize(vals: list[float]) -> dict:
    if not vals:
        return dict(n=0, mean=-1, median=-1, std=-1, p10=-1, p90=-1, min=-1, max=-1)
    xs = sorted(vals)
    n = len(xs)

    def pct(p: float) -> float:
        k = (n - 1) * p
        lo = int(k)
        hi = min(lo + 1, n - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    return dict(n=n, mean=float(np.mean(xs)), median=xs[n // 2], std=float(np.std(xs)),
                p10=pct(0.10), p90=pct(0.90), min=xs[0], max=xs[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Promptless SegFormer Dice/IoU eval on MMOTU-2D val.")
    ap.add_argument("--ir", required=True, help="SegFormer IR model.xml (static [1,3,res,res]).")
    ap.add_argument("--device", default="GPU", choices=["CPU", "GPU", "NPU"])
    ap.add_argument("--data", default="data/OTU_2d")
    ap.add_argument("--val-list", default=None, help="Defaults to <data>/val.txt.")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="If >0, eval at most this many images (smoke).")
    ap.add_argument("--out-csv", default="reports/dice_segformer.csv")
    args = ap.parse_args()

    ir = Path(args.ir)
    if not ir.exists():
        raise FileNotFoundError(f"IR missing: {ir}")
    data = Path(args.data)
    images_dir = data / "images"
    masks_dir = data / "annotations"
    val_list = Path(args.val_list) if args.val_list else (data / "val.txt")
    if not val_list.exists():
        raise FileNotFoundError(f"val list missing: {val_list}")
    stems = [ln.strip() for ln in val_list.read_text().splitlines() if ln.strip()]
    if args.limit > 0:
        stems = stems[: args.limit]

    core = ov.Core()
    model = core.read_model(str(ir))
    compiled = core.compile_model(model, args.device)
    out_port = compiled.outputs[0]
    print(f"[eval] device={args.device} ir={ir}")
    print(f"[eval] val stems={len(stems)} data={data}")

    res = args.res
    dices: list[float] = []
    ious: list[float] = []
    rows: list[tuple[str, float, float]] = []
    missing = 0
    for i, stem in enumerate(stems):
        ip = _find_image(images_dir, stem)
        mp = _find_mask(masks_dir, stem)
        if ip is None or mp is None:
            missing += 1
            continue
        img = Image.open(ip).convert("RGB")
        gt_bin = (np.asarray(Image.open(mp)) > 0).astype(np.uint8)
        H, W = gt_bin.shape[:2]

        x = img.resize((res, res), Image.BILINEAR)
        arr = np.asarray(x, np.float32).transpose(2, 0, 1) / 255.0   # [3,res,res]
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        logits = np.asarray(compiled(arr[None])[out_port])[0]        # [2,h,w]

        up = np.empty((2, res, res), np.float32)
        for c in range(2):
            up[c] = np.asarray(
                Image.fromarray(logits[c], mode="F").resize((res, res), Image.BILINEAR),
                np.float32,
            )
        mask_res = (up[1] > up[0]).astype(np.uint8)                  # [res,res] {0,1}
        pred_native = np.asarray(
            Image.fromarray(mask_res * 255).resize((W, H), Image.NEAREST)
        ) > 127

        d, j = dice_iou(pred_native, gt_bin)
        dices.append(d)
        ious.append(j)
        rows.append((stem, d, j))
        if (i + 1) % 50 == 0:
            print(f"[eval] {i + 1}/{len(stems)} running mean Dice={np.mean(dices):.4f}")

    ds = summarize(dices)
    js = summarize(ious)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("stem,dice,iou\n")
        for stem, d, j in rows:
            f.write(f"{stem},{d:.6f},{j:.6f}\n")

    print(f"[eval] evaluated={ds['n']} missing={missing}")
    print(f"[eval] Dice mean={ds['mean']:.4f} std={ds['std']:.4f} "
          f"p10={ds['p10']:.4f} p90={ds['p90']:.4f} min={ds['min']:.4f} max={ds['max']:.4f}")
    print(f"[eval] IoU  mean={js['mean']:.4f} std={js['std']:.4f}")
    print(f"[eval] wrote {out_csv}")


if __name__ == "__main__":
    main()
