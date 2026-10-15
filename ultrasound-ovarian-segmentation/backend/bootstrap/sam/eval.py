# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Dice / IoU of the built SAM-256 pipeline on the MMOTU-2D val split.

Verifies the three IRs produced by ``export_yolo.py`` / ``export_encoder.py`` /
``export_decoder.py`` before they are used in a demo. This runs the EXACT runtime
pipeline the app uses — it drives ``src/segmenters/sam.py:SamSegmenter`` — so the
Dice reported here is what the live overlay reflects:

    frame -> YOLOv8n@320 bbox -> SAM encoder@256 -> multimask decoder ->
    argmax-by-IoU -> mask (native) -> Dice / IoU vs native GT ({stem}_binary.PNG).

Reference result at res 256: end-to-end Dice mean 0.8207 / IoU 0.7394 over the
469-image MMOTU-2D val split. The KPI gate passes when mean Dice >= 0.80.

Usage (Windows, inside the app venv):

    python -m backend.bootstrap.sam.eval --device GPU --data data/OTU_2d
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow `import src...` when run as `python -m backend.bootstrap.sam.eval` from
# the app root (and when run directly).
_APP_ROOT = Path(__file__).resolve().parents[3]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))


def dice_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> tuple[float, float]:
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
    ap = argparse.ArgumentParser(description="End-to-end SAM-256 Dice/IoU eval on MMOTU-2D val.")
    ap.add_argument("--sam-encoder", default="models/sam_encoder_256_stockft/encoder.xml")
    ap.add_argument("--sam-decoder", default="models/sam_decoder_256_multimask/decoder.xml")
    ap.add_argument("--sam-yolo", default="models/yolo_mmotu_320/yolov8n_mmotu.xml")
    ap.add_argument("--device", default="GPU", choices=["CPU", "GPU", "NPU"])
    ap.add_argument("--data", default="data/OTU_2d")
    ap.add_argument("--val-list", default=None, help="Defaults to <data>/val.txt.")
    ap.add_argument("--res", type=int, default=256, help="Encoder input resolution.")
    ap.add_argument("--limit", type=int, default=0, help="If >0, eval at most this many images (smoke).")
    ap.add_argument("--kpi", type=float, default=0.80, help="Mean-Dice pass gate.")
    ap.add_argument("--out-csv", default="reports/dice_sam_e2e_256.csv")
    args = ap.parse_args()

    data = Path(args.data)
    images_dir = data / "images"
    masks_dir = data / "annotations"
    val_list = Path(args.val_list) if args.val_list else (data / "val.txt")
    if not val_list.exists():
        raise FileNotFoundError(f"val list missing: {val_list}")
    stems = [ln.strip() for ln in val_list.read_text().splitlines() if ln.strip()]
    if args.limit > 0:
        stems = stems[: args.limit]

    try:
        from src.segmenters.sam import SamSegmenter
    except Exception as exc:  # pragma: no cover - env guard
        raise SystemExit(f"[eval] could not import SamSegmenter from src.segmenters.sam: {exc}")

    seg = SamSegmenter(args.sam_encoder, args.sam_decoder, args.sam_yolo,
                       device=args.device, enc_res=args.res)
    print(f"[eval] device={args.device} val stems={len(stems)} data={data}")

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
        img = cv2.imread(str(ip))
        gt = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            missing += 1
            continue
        gt_bin = (gt > 0).astype(np.uint8)

        result = seg.infer(img)
        pred = result.mask if result.mask is not None else np.zeros_like(gt_bin, dtype=bool)

        d, j = dice_iou(pred, gt_bin)
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
    print(f"[eval] Dice mean={ds['mean']:.4f} median={ds['median']:.4f} std={ds['std']:.4f} "
          f"p10={ds['p10']:.4f} p90={ds['p90']:.4f} min={ds['min']:.4f} max={ds['max']:.4f}")
    print(f"[eval] IoU  mean={js['mean']:.4f} std={js['std']:.4f}")
    print(f"[eval] wrote {out_csv}")
    verdict = "PASS" if ds["mean"] >= args.kpi else f"FAIL (need >= {args.kpi:.2f})"
    print(f"[eval] KPI: mean Dice >= {args.kpi:.2f} -> {verdict}")


if __name__ == "__main__":
    main()
