# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export a trained SegFormer-B5 checkpoint to a fp16 OpenVINO IR (what the app loads).

Loads the HF checkpoint produced by ``train.py`` and converts it straight to
OpenVINO IR with the PyTorch frontend (``ov.convert_model`` with an
``example_input``), locked to a static ``[1,3,res,res]`` shape. The IR is written
to ``models/ds2net_segformer_b5_<res>/model.xml`` — exactly the default the app
loads (see ``src/config.py``).

IMPORTANT — always pass ``--from-pretrained <checkpoint>``. SegFormer-B5's op
graph (and therefore its latency) is architecture-bound, so you *can* build the
IR from config with random weights, but that IR will segment nothing
(Dice ~0). Only an IR exported from a trained checkpoint reproduces the
reference Dice 0.8657 / IoU 0.7913.

Usage (Windows, inside the app venv with the backend requirements installed):

    python -m backend.bootstrap.export ^
        --from-pretrained models/segformer_b5_mmotu --res 256

Then verify it before shipping it into a run:

    python -m backend.bootstrap.eval ^
        --ir models/ds2net_segformer_b5_256/model.xml --res 256 ^
        --data data/OTU_2d --device GPU
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Force offline loading so export never blocks on an HF hub network call.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import openvino as ov
import torch

# MiT-B5 backbone preset (SegFormer paper / nvidia/mit-b5). Only used when no
# checkpoint is given (latency-only export); num_labels drives the final 1x1.
B5_CONFIG = dict(
    num_channels=3,
    num_encoder_blocks=4,
    depths=[3, 6, 40, 3],
    sr_ratios=[8, 4, 2, 1],
    hidden_sizes=[64, 128, 320, 512],
    patch_sizes=[7, 3, 3, 3],
    strides=[4, 2, 2, 2],
    num_attention_heads=[1, 2, 5, 8],
    mlp_ratios=[4, 4, 4, 4],
    hidden_act="gelu",
    decoder_hidden_size=768,
)


class SegformerLogits(torch.nn.Module):
    """Wrap the HF model so forward(x) returns a single logits tensor.

    The HF module returns a dataclass; the tracer wants a plain Tensor. Logits
    come out at input/4 resolution (SegFormer does not upsample internally); the
    app / eval upsample + argmax to recover the full-resolution mask.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def build_segformer_b5(num_labels: int, from_pretrained: str | None) -> torch.nn.Module:
    """Return a SegFormer-B5 model (trained weights if --from-pretrained, else random)."""
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except Exception as exc:  # pragma: no cover - env guard
        raise SystemExit(
            "[export] transformers not installed. Inside the app venv run:\n"
            "    pip install -r backend/requirements.txt\n"
            f"(import error: {type(exc).__name__}: {exc})"
        )

    if from_pretrained:
        print(f"[export] loading trained checkpoint: {from_pretrained}")
        model = SegformerForSemanticSegmentation.from_pretrained(
            from_pretrained, num_labels=num_labels, ignore_mismatched_sizes=True
        )
    else:
        print("[export] WARNING: no --from-pretrained given -> building B5 from config with "
              "RANDOM weights. The IR will compile and run but segment nothing (Dice ~0). "
              "Pass --from-pretrained <checkpoint> for a usable model.")
        config = SegformerConfig(num_labels=num_labels, **B5_CONFIG)
        model = SegformerForSemanticSegmentation(config)
    return SegformerLogits(model.eval())


def export_one(model: torch.nn.Module, res: int, out_dir: Path) -> Path:
    """Convert the model at RESxRES to a static-shape fp16 OpenVINO IR."""
    dummy = torch.randn(1, 3, res, res)
    print(f"[export] OpenVINO convert (static [1,3,{res},{res}]) ...")
    with torch.no_grad():
        ov_model = ov.convert_model(model, example_input=dummy)
    # Lock to a static [1,3,res,res] shape: the PyTorch frontend leaves H/W
    # dynamic otherwise, which gives the GPU a dynamic graph and makes every
    # resolution's IR identical. Static shapes = clean per-resolution latency.
    ov_model.reshape([1, 3, res, res])
    out_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(out_dir / "model.xml"), compress_to_fp16=True)
    (out_dir / ".converted_ok").write_text("fp16\n")
    print(f"[export] wrote {out_dir / 'model.xml'}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export a trained SegFormer-B5 checkpoint to a fp16 OpenVINO IR."
    )
    ap.add_argument("--from-pretrained", default=None,
                    help="Trained HF checkpoint dir (from train.py). Required for a usable IR.")
    ap.add_argument("--res", type=int, default=256,
                    help="Network input resolution (square). 256 = reference operating point.")
    ap.add_argument("--num-labels", type=int, default=2,
                    help="Segmentation classes (MMOTU binary seg = 2).")
    ap.add_argument("--out", default=None,
                    help="Output IR dir (default: models/ds2net_segformer_b5_<res>).")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("models") / f"ds2net_segformer_b5_{args.res}"
    model = build_segformer_b5(args.num_labels, args.from_pretrained)
    ir_dir = export_one(model, args.res, out_dir)

    print("\n[export] done. Point the app at this IR (it is the default for res 256):")
    print(f"    {ir_dir / 'model.xml'}")
    print("[export] verify accuracy before a demo:")
    print(f"    python -m backend.bootstrap.eval --ir {ir_dir / 'model.xml'} "
          f"--res {args.res} --data data/OTU_2d --device GPU")


if __name__ == "__main__":
    main()
