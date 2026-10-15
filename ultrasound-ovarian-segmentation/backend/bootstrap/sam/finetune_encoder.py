# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native-256 SAM image-encoder fine-tune on MMOTU-2D (the SAM-256 encoder).

Trains ONLY the SAM ViT-B image encoder. The prompt encoder and mask decoder are
loaded from the stock Meta checkpoint and FROZEN — the encoder learns to produce
embeddings the stock decoder turns into ovarian tumor masks when prompted with
the GT-derived bounding box. Loss = Dice + BCE on the low-resolution decoder
logits. Init source is Meta stock ``sam_vit_b_01ec64.pth`` (Apache-2.0); the
output checkpoint is ``models/sam256_stockft.pth``.

The output checkpoint uses the standard SAM state-dict layout
(``image_encoder.*`` / ``prompt_encoder.*`` / ``mask_decoder.*``) so
``export_encoder.py`` consumes it verbatim.

PREPROCESSING — must match the runtime exactly. The app's encoder input is
``BGR->RGB, /255.0`` with NO mean/std normalisation (see
``src/segmenters/sam.py:preprocess_encoder`` and ``export_encoder.py``). Training
uses the identical ``/255.0`` scaling here so the fine-tuned encoder sees the same
input distribution at train and inference time; this is what reproduces the
reference end-to-end Dice 0.8207.

Usage (Windows, inside the app venv with backend/requirements-sam.txt installed):

    python -m backend.bootstrap.sam.finetune_encoder ^
        --weights models/sam_vit_b_01ec64.pth --data data/OTU_2d ^
        --epochs 10 --batch 4 --lr 1e-4 --device xpu ^
        --out models/sam256_stockft.pth

Smoke test the pipeline end to end in a few steps:

    python -m backend.bootstrap.sam.finetune_encoder ^
        --weights models/sam_vit_b_01ec64.pth --data data/OTU_2d ^
        --epochs 1 --batch 2 --device cpu --smoke-steps 20 --out models/_sam_smoke.pth
"""
from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
# Model assembly (image encoder @ 256², rebuilt prompt encoder @ 16×16 grid,
# stock mask decoder — all three loaded from a SAM state dict, then encoder
# left trainable and everything else frozen).
# -----------------------------------------------------------------------------

def build_sam_components(weights: Path, enc_res: int = 256) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Return (image_encoder, prompt_encoder, mask_decoder) at ``enc_res``.

    Encoder pos_embed is bicubic-interpolated and rel_pos tables are
    linear-interpolated from the checkpoint grid to ``enc_res//16``. Tensors that
    still shape-mismatch (the 8 global-attention rel_pos tables at a new grid)
    are dropped and trained from PyTorch random init — that re-learning is the
    point of the native-256 fine-tune. The prompt encoder is rebuilt at the
    ``(enc_res//16, enc_res//16)`` grid; the mask decoder is grid-agnostic and
    loaded verbatim.
    """
    from segment_anything.modeling.image_encoder import ImageEncoderViT
    from segment_anything.modeling.mask_decoder import MaskDecoder
    from segment_anything.modeling.prompt_encoder import PromptEncoder
    from segment_anything.modeling.transformer import TwoWayTransformer

    image_encoder = ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=enc_res,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=256,
    )
    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(enc_res // 16, enc_res // 16),
        input_image_size=(enc_res, enc_res),
        mask_in_chans=16,
    )
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )

    if not weights.exists():
        raise FileNotFoundError(
            f"Missing SAM checkpoint: {weights}. Download Meta stock "
            "sam_vit_b_01ec64.pth (Apache-2.0) from "
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        )

    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)

    # ---- Image encoder: interpolate pos_embed + rel_pos, drop shape-mismatch --
    enc_state = {
        k.replace("image_encoder.", ""): v
        for k, v in ckpt.items() if k.startswith("image_encoder.")
    }
    if "pos_embed" in enc_state:
        pe = enc_state["pos_embed"]
        dst_n = enc_res // 16
        if pe.shape[1] != dst_n:
            pe4d = pe.permute(0, 3, 1, 2)
            pe4d = F.interpolate(pe4d, size=(dst_n, dst_n), mode="bicubic", align_corners=False)
            enc_state["pos_embed"] = pe4d.permute(0, 2, 3, 1).contiguous()
            print(f"[finetune] pos_embed interpolated {pe.shape[1]}x{pe.shape[1]} -> {dst_n}x{dst_n}")

    target_rel_len = 2 * (enc_res // 16) - 1
    for _k, _v in list(enc_state.items()):
        if not (_k.endswith(".attn.rel_pos_h") or _k.endswith(".attn.rel_pos_w")):
            continue
        if _v.shape[0] == target_rel_len:
            continue
        _v3d = _v.unsqueeze(0).permute(0, 2, 1)
        _v3d = F.interpolate(_v3d, size=target_rel_len, mode="linear", align_corners=False)
        enc_state[_k] = _v3d.permute(0, 2, 1).squeeze(0).contiguous()

    target_enc = image_encoder.state_dict()
    keep_enc = {k: v for k, v in enc_state.items()
                if k in target_enc and v.shape == target_enc[k].shape}
    dropped_enc = sorted(set(enc_state) - set(keep_enc))
    missing, unexpected = image_encoder.load_state_dict(keep_enc, strict=False)
    print(f"[finetune] image_encoder loaded (kept={len(keep_enc)}, dropped={len(dropped_enc)}, "
          f"missing={len(missing)}, unexpected={len(unexpected)})")
    if dropped_enc:
        print(f"[finetune]   dropped (shape mismatch, trained from random init): {dropped_enc}")

    # ---- Prompt encoder: rebuild-at-target-grid + drop shape-mismatch --------
    pe_state = {
        k.replace("prompt_encoder.", ""): v
        for k, v in ckpt.items() if k.startswith("prompt_encoder.")
    }
    target_pe = prompt_encoder.state_dict()
    keep_pe = {k: v for k, v in pe_state.items() if k in target_pe and v.shape == target_pe[k].shape}
    prompt_encoder.load_state_dict(keep_pe, strict=False)
    print(f"[finetune] prompt_encoder loaded (kept={len(keep_pe)}, dropped={len(set(pe_state) - set(keep_pe))})")

    # ---- Mask decoder: verbatim ---------------------------------------------
    md_state = {
        k.replace("mask_decoder.", ""): v
        for k, v in ckpt.items() if k.startswith("mask_decoder.")
    }
    md_missing, md_unexpected = mask_decoder.load_state_dict(md_state, strict=False)
    print(f"[finetune] mask_decoder loaded ({len(md_state)} tensors, "
          f"missing={len(md_missing)}, unexpected={len(md_unexpected)})")

    return image_encoder, prompt_encoder, mask_decoder


# -----------------------------------------------------------------------------
# Dataset (OTU_2d layout, shared with the DS2Net stages)
# -----------------------------------------------------------------------------

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


class MMOTU256(Dataset):
    """Return (image[0,1] CHW, mask[H,W] in {0,1}, bbox_xyxy at enc_res).

    Image scaling is ``/255.0`` with NO mean/std normalisation — identical to the
    runtime ``preprocess_encoder`` so the encoder trains on the inference input
    distribution.
    """

    def __init__(self, root: Path, split: str, enc_res: int = 256):
        self.enc_res = enc_res
        images_dir = root / "images"
        masks_dir = root / "annotations"
        list_file = root / f"{split}.txt"
        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(f"Expected {images_dir} and {masks_dir}")
        if not list_file.exists():
            raise FileNotFoundError(f"Expected split list at {list_file}")

        stems = [Path(ln.strip()).stem for ln in list_file.read_text().splitlines() if ln.strip()]
        self.items: list[tuple[Path, Path]] = []
        for stem in stems:
            ip = _find_image(images_dir, stem)
            mp = _find_mask(masks_dir, stem)
            if ip is not None and mp is not None:
                self.items.append((ip, mp))
        if not self.items:
            raise RuntimeError(f"No usable (image, mask) pairs found for split={split} under {root}")
        print(f"[finetune] MMOTU {split}: {len(self.items)} pairs")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.items[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read {img_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read {mask_path}")

        img = cv2.resize(img, (self.enc_res, self.enc_res), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.enc_res, self.enc_res), interpolation=cv2.INTER_NEAREST)
        # BGR->RGB, /255.0 — matches the runtime preprocess_encoder exactly.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mask_bin = (mask > 0).astype(np.uint8)

        _, binary = cv2.threshold(mask_bin, 0, 1, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            bbox = [0.0, 0.0, float(self.enc_res - 1), float(self.enc_res - 1)]
        else:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            bbox = [float(x), float(y), float(x + w), float(y + h)]

        img_t = torch.from_numpy(img).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_bin.astype(np.float32))
        bbox_t = torch.tensor(bbox, dtype=torch.float32)
        return img_t, mask_t, bbox_t


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits.squeeze(1))
    inter = (probs * target).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2.0 * inter + eps) / (union + eps)
    return (1.0 - dice).mean()


def bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.squeeze(1), target)


# -----------------------------------------------------------------------------
# Forward pass (frozen prompt encoder + frozen decoder, trainable encoder)
# -----------------------------------------------------------------------------

def forward_batch(image_encoder: nn.Module,
                  prompt_encoder: nn.Module,
                  mask_decoder: nn.Module,
                  imgs: torch.Tensor,     # [B, 3, H, W] in [0,1]
                  bboxes: torch.Tensor,   # [B, 4] xyxy in enc_res space
                  ) -> torch.Tensor:
    """Return low-res mask logits [B, 1, h, w]. The decoder runs per-image because
    SAM's decoder expects ``1 image × N prompts``, not ``N images × 1 prompt``."""
    image_embeddings = image_encoder(imgs)   # [B, 256, H/16, W/16]
    boxes = bboxes.unsqueeze(1)              # [B, 1, 4]
    with torch.no_grad():
        image_pe = prompt_encoder.get_dense_pe()

    outs: list[torch.Tensor] = []
    for i in range(imgs.size(0)):
        with torch.no_grad():
            sparse_i, dense_i = prompt_encoder(points=None, boxes=boxes[i:i + 1], masks=None)
        low_res_i, _ = mask_decoder(
            image_embeddings=image_embeddings[i:i + 1],
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_i,
            dense_prompt_embeddings=dense_i,
            multimask_output=False,
        )
        outs.append(low_res_i)
    return torch.cat(outs, dim=0)


# -----------------------------------------------------------------------------
# Train / validate
# -----------------------------------------------------------------------------

def compute_dice_metric(logits: torch.Tensor, target: torch.Tensor) -> float:
    probs = torch.sigmoid(logits.squeeze(1))
    pred = (probs > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2.0 * inter + 1e-6) / (union + 1e-6)
    return float(dice.mean().item())


@torch.no_grad()
def validate(image_encoder, prompt_encoder, mask_decoder, val_loader, device, enc_res: int) -> float:
    image_encoder.eval()
    total, n = 0.0, 0
    for imgs, masks, bboxes in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        bboxes = bboxes.to(device, non_blocking=True)
        logits = forward_batch(image_encoder, prompt_encoder, mask_decoder, imgs, bboxes)
        logits_full = F.interpolate(logits, size=(enc_res, enc_res), mode="bilinear", align_corners=False)
        total += compute_dice_metric(logits_full, masks) * imgs.size(0)
        n += imgs.size(0)
    image_encoder.train()
    return total / max(1, n)


def save_checkpoint(out_path: Path, image_encoder, prompt_encoder, mask_decoder) -> None:
    """Save in SAM's flat 'image_encoder.*' / 'prompt_encoder.*' / 'mask_decoder.*' layout."""
    state: dict[str, torch.Tensor] = {}
    for k, v in image_encoder.state_dict().items():
        state[f"image_encoder.{k}"] = v.detach().cpu()
    for k, v in prompt_encoder.state_dict().items():
        state[f"prompt_encoder.{k}"] = v.detach().cpu()
    for k, v in mask_decoder.state_dict().items():
        state[f"mask_decoder.{k}"] = v.detach().cpu()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(out_path))


def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description="Native-256 SAM encoder fine-tune on MMOTU-2D.")
    ap.add_argument("--weights", default="models/sam_vit_b_01ec64.pth",
                    help="Init SAM checkpoint (Meta stock sam_vit_b_01ec64.pth, Apache-2.0).")
    ap.add_argument("--data", default="data/OTU_2d",
                    help="MMOTU-2D root (images/, annotations/, train.txt, val.txt).")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--device", default="xpu", choices=["cpu", "cuda", "xpu"])
    ap.add_argument("--workers", type=int, default=0, help="0 = safest on Windows/xpu.")
    ap.add_argument("--out", default="models/sam256_stockft.pth")
    ap.add_argument("--smoke-steps", type=int, default=0,
                    help="If >0, run at most this many optimizer steps total (pipeline smoke).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    want = args.device
    if want == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("[finetune] WARN: xpu unavailable -> cpu")
        want = "cpu"
    if want == "cuda" and not torch.cuda.is_available():
        print("[finetune] WARN: cuda unavailable -> cpu")
        want = "cpu"
    device = torch.device(want)

    weights = Path(args.weights)
    data_root = Path(args.data)
    out_path = Path(args.out)
    print(f"[finetune] device={device}, weights={weights}, data={data_root}")

    image_encoder, prompt_encoder, mask_decoder = build_sam_components(weights, args.imgsz)
    image_encoder.to(device)
    prompt_encoder.to(device)
    mask_decoder.to(device)

    freeze(prompt_encoder)
    freeze(mask_decoder)
    image_encoder.train()

    n_trainable = sum(p.numel() for p in image_encoder.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in prompt_encoder.parameters() if p.requires_grad) \
        + sum(p.numel() for p in mask_decoder.parameters() if p.requires_grad)
    print(f"[finetune] trainable params (image_encoder): {n_trainable / 1e6:.1f}M")
    print(f"[finetune] frozen params (prompt+decoder):   {n_frozen / 1e6:.1f}M (must be 0)")

    # Hash a decoder tensor to verify the frozen contract post-training.
    decoder_probe_key = "transformer.layers.0.self_attn.q_proj.weight"
    decoder_probe = mask_decoder.state_dict()[decoder_probe_key].detach().cpu().clone()

    train_ds = MMOTU256(data_root, "train", args.imgsz)
    val_ds = MMOTU256(data_root, "val", args.imgsz)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))

    optim = torch.optim.AdamW(image_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)

    best_val_dice = -1.0
    global_step = 0
    stop = False
    t_start = time.perf_counter()

    for epoch in range(args.epochs):
        if stop:
            break
        epoch_loss_sum, epoch_loss_n = 0.0, 0
        t_epoch = time.perf_counter()
        for imgs, masks, bboxes in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)

            logits = forward_batch(image_encoder, prompt_encoder, mask_decoder, imgs, bboxes)
            logits_full = F.interpolate(logits, size=(args.imgsz, args.imgsz),
                                        mode="bilinear", align_corners=False)
            loss = dice_loss(logits_full, masks) + bce_loss(logits_full, masks)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(image_encoder.parameters(), max_norm=1.0)
            optim.step()
            sched.step()

            epoch_loss_sum += float(loss.item())
            epoch_loss_n += 1
            global_step += 1
            if global_step % 20 == 0 or global_step == 1:
                print(f"[finetune] ep {epoch + 1}/{args.epochs} step {global_step}/{total_steps} "
                      f"loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e}")
            if args.smoke_steps and global_step >= args.smoke_steps:
                stop = True
                break

        val_dice = validate(image_encoder, prompt_encoder, mask_decoder, val_loader, device, args.imgsz)
        dt = time.perf_counter() - t_epoch
        print(f"[finetune] EPOCH {epoch + 1}/{args.epochs} done in {dt:.1f}s "
              f"train_loss_mean={epoch_loss_sum / max(1, epoch_loss_n):.4f} val_dice={val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_checkpoint(out_path, image_encoder, prompt_encoder, mask_decoder)
            print(f"[finetune] saved best checkpoint (val_dice={val_dice:.4f}) -> {out_path}")

    decoder_probe_now = mask_decoder.state_dict()[decoder_probe_key].detach().cpu()
    if not torch.equal(decoder_probe, decoder_probe_now):
        raise RuntimeError(
            f"FROZEN DECODER MUTATED at {decoder_probe_key}. "
            "Encoder-only fine-tune contract violated."
        )
    print(f"[finetune] freeze check OK: {decoder_probe_key} unchanged")

    total_dt = time.perf_counter() - t_start
    print(f"[finetune] done in {total_dt / 60:.1f} min. best_val_dice={best_val_dice:.4f} out={out_path}")
    print("[finetune] next: python -m backend.bootstrap.sam.export_encoder "
          f"--weights {out_path}")


if __name__ == "__main__":
    main()
