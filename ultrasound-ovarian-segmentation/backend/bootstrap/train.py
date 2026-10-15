# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fine-tune a promptless SegFormer MiT-B5 on MMOTU-2D (the DS2Net model).

DS2Net (github.com/cv516Buaa/MMOTU_DS2Net, arXiv 2207.06799, Apache-2.0) is a
SegFormer MiT-B5 semantic-segmentation model for ovarian ultrasound. Its
dual-branch head only matters for 2D<->CEUS domain adaptation, so for a single
2D ultrasound stream we train the faithful single-modality stand-in: a
SegFormer-B5 that takes the whole frame and outputs a 2-class (background /
tumor) mask — no bbox, no detector, no prompt.

Init : nvidia/mit-b5 ImageNet backbone (Apache-2.0); decode head random.
Data : OTU_2d with images/{stem}.JPG, annotations/{stem}_binary.PNG, and
       train.txt / val.txt split lists.
Loss : cross-entropy + soft-Dice on the foreground class.
Device: Intel Arc iGPU (xpu) with graceful CPU fallback.

The authoritative accuracy number comes from exporting this checkpoint to fp16
IR (export.py) and scoring Dice with eval.py on the iGPU — same runtime, val
list and metric the app uses. The val Dice printed here (torch, res-space) is a
convergence sanity check only; the reference MMOTU-2D val result is
Dice 0.8657 / IoU 0.7913 at res 256.

Usage (Windows, inside the app venv with the backend requirements installed):

    python -m backend.bootstrap.train ^
        --data data/OTU_2d --backbone-dir models/mit_b5_hf ^
        --res 256 --epochs 40 --batch 4 --lr 6e-5 --device xpu ^
        --out models/segformer_b5_mmotu

Smoke test the pipeline end to end in a few steps:

    python -m backend.bootstrap.train --data data/OTU_2d ^
        --backbone-dir models/mit_b5_hf --device xpu --smoke-steps 8 ^
        --out models/_seg_smoke
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Force offline loading so training never blocks on an HF hub network call.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ImageNet normalisation (SegFormer image-processor default). The HF model does
# NOT normalise internally, so we do it here before the forward pass. This must
# match export.py / eval.py / the app segmenter exactly.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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


class MMOTUSeg(Dataset):
    """(pixel_values[3,R,R] float [0,1], labels[R,R] long {0,1}) pairs."""

    def __init__(self, root: Path, split: str, res: int) -> None:
        self.res = res
        images_dir = root / "images"
        masks_dir = root / "annotations"
        list_file = root / f"{split}.txt"
        for d in (images_dir, masks_dir):
            if not d.exists():
                raise FileNotFoundError(f"Missing {d}")
        if not list_file.exists():
            raise FileNotFoundError(f"Missing split list {list_file}")
        stems = [ln.strip() for ln in list_file.read_text().splitlines() if ln.strip()]
        self.items: list[tuple[Path, Path]] = []
        miss = 0
        for s in stems:
            ip = _find_image(images_dir, s)
            mp = _find_mask(masks_dir, s)
            if ip is not None and mp is not None:
                self.items.append((ip, mp))
            else:
                miss += 1
        if not self.items:
            raise RuntimeError(f"No (image,mask) pairs for split={split} under {root}")
        print(f"[train] {split}: {len(self.items)} pairs ({miss} stems skipped)")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        ip, mp = self.items[idx]
        img = Image.open(ip).convert("RGB").resize((self.res, self.res), Image.BILINEAR)
        msk = Image.open(mp).resize((self.res, self.res), Image.NEAREST)
        arr = np.asarray(img, dtype=np.float32) / 255.0            # HWC [0,1]
        lab = (np.asarray(msk) > 0).astype(np.int64)               # {0,1}
        pix = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,R,R]
        labels = torch.from_numpy(lab)                             # [R,R]
        return pix, labels


def build_model(backbone_dir: str, num_labels: int) -> torch.nn.Module:
    from transformers import SegformerForSemanticSegmentation

    print(f"[train] init SegformerForSemanticSegmentation from '{backbone_dir}' "
          f"(num_labels={num_labels}, decode head random)")
    return SegformerForSemanticSegmentation.from_pretrained(
        backbone_dir, num_labels=num_labels, ignore_mismatched_sizes=True,
    )


def seg_losses(logits: torch.Tensor, labels: torch.Tensor, res: int):
    """logits [B,C,h,w] (h=res/4) -> upsample to res; return (ce+dice, ce, dice)."""
    logits = F.interpolate(logits, size=(res, res), mode="bilinear", align_corners=False)
    ce = F.cross_entropy(logits, labels)
    probs = torch.softmax(logits, dim=1)[:, 1]                     # foreground prob [B,R,R]
    tgt = (labels == 1).float()
    inter = (probs * tgt).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
    dice = (1.0 - (2.0 * inter + 1.0) / (union + 1.0)).mean()
    return ce + dice, ce, dice


@torch.no_grad()
def validate(model, loader, device, mean, std, res: int) -> float:
    """Mean per-image Dice at res-space (torch, fp32). Convergence check only."""
    model.eval()
    tot, n = 0.0, 0
    for pix, labels in loader:
        pix = pix.to(device)
        labels = labels.to(device)
        logits = model(pixel_values=(pix - mean) / std).logits
        logits = F.interpolate(logits, size=(res, res), mode="bilinear", align_corners=False)
        pred = (logits.argmax(dim=1) == 1).float()
        g = (labels == 1).float()
        inter = (pred * g).sum(dim=(1, 2))
        denom = pred.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
        dice = torch.where(denom > 0, (2 * inter) / denom, torch.ones_like(denom))
        tot += float(dice.sum().item())
        n += pix.size(0)
    model.train()
    return tot / max(1, n)


def _save(model, out, quiet: bool = False) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    if quiet:
        print(f"[train] checkpoint -> {out}")
    else:
        print(f"[train] saved HF model to {out}\n"
              f"[train] export to an OpenVINO IR with:\n"
              f"        python -m backend.bootstrap.export --from-pretrained {out} --res 256")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Promptless SegFormer-B5 fine-tune on MMOTU-2D (the DS2Net model)."
    )
    ap.add_argument("--data", default="data/OTU_2d")
    ap.add_argument("--backbone-dir", default="models/mit_b5_hf",
                    help="Local nvidia/mit-b5 snapshot (ImageNet backbone, Apache-2.0).")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--num-labels", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--device", default="xpu", choices=["cpu", "cuda", "xpu"])
    ap.add_argument("--workers", type=int, default=0, help="0 = safest on Windows/xpu.")
    ap.add_argument("--out", default="models/segformer_b5_mmotu")
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--smoke-steps", type=int, default=0,
                    help="If >0, stop after this many optimizer steps and save (pipeline smoke).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    want = args.device
    if want == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("[train] WARN: xpu unavailable -> cpu")
        want = "cpu"
    if want == "cuda" and not torch.cuda.is_available():
        print("[train] WARN: cuda unavailable -> cpu")
        want = "cpu"
    device = torch.device(want)
    print(f"[train] device={device}")

    data_root = Path(args.data)
    train_ds = MMOTUSeg(data_root, "train", args.res)
    val_ds = MMOTUSeg(data_root, "val", args.res)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers)

    model = build_model(args.backbone_dir, args.num_labels).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] trainable params: {n_train / 1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)

    step = 0
    best_val = -1.0
    t_prev = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for pix, labels in train_loader:
            pix = pix.to(device)
            labels = labels.to(device)
            logits = model(pixel_values=(pix - mean) / std).logits
            loss, ce, dice = seg_losses(logits, labels, args.res)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            nb += 1
            ep_loss += float(loss.item())
            now = time.time()
            step_ms = (now - t_prev) * 1000.0   # loss.item() above forces a device sync
            t_prev = now
            if step <= 5 or step % 20 == 0:
                print(f"[train] ep{epoch} step{step} loss={loss.item():.4f} "
                      f"ce={ce.item():.4f} dice={dice.item():.4f} {step_ms:.0f}ms/step")
            if args.smoke_steps and step >= args.smoke_steps:
                print(f"[train] smoke stop at {step} steps")
                _save(model, args.out)
                return
        print(f"[train] epoch {epoch} mean_loss={ep_loss / max(1, nb):.4f}")
        if args.val_every and (epoch % args.val_every == 0 or epoch == args.epochs):
            vd = validate(model, val_loader, device, mean, std, args.res)
            print(f"[train] epoch {epoch} val_dice(res-space,torch)={vd:.4f}")
            _save(model, args.out, quiet=True)  # recoverable latest checkpoint
            if vd > best_val:
                best_val = vd
                _save(model, f"{args.out}_best", quiet=True)
                print(f"[train] new best val_dice={best_val:.4f} -> {args.out}_best")

    _save(model, args.out)
    print(f"[train] done. best val_dice(res-space,torch)={best_val:.4f}")


if __name__ == "__main__":
    main()
