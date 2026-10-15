# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export the fine-tuned 256×256 SAM image encoder to a fp16 OpenVINO IR.

Loads the ``image_encoder.*`` weights from the fine-tuned checkpoint produced by
``finetune_encoder.py`` (``models/sam256_stockft.pth``) into a SAM ViT-B image
encoder forced to ``img_size=256``, traces it to ONNX, and converts to a fp16
OpenVINO IR at ``models/sam_encoder_256_stockft/encoder.xml`` — exactly the
default the app loads (see ``src/config.py`` ``--sam-encoder``). The IR input is
``image [1,3,256,256]`` (``BGR->RGB, /255.0``, no mean/std — matching the runtime
and the fine-tune).

pos_embed (bicubic) and rel_pos tables (linear) are interpolated from the
checkpoint grid to ``256//16`` if needed, so a 1024-native stock checkpoint can
also be converted directly for throughput measurement. By default the converter
HARD-FAILS when the weights file is missing (``--allow-random`` builds a
random-init encoder for throughput-only benchmarks — accuracy will be invalid).

Licence: SAM is Apache-2.0.

Usage (Windows, inside the app venv with backend/requirements-sam.txt installed):

    python -m backend.bootstrap.sam.export_encoder --weights models/sam256_stockft.pth
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import openvino as ov
import torch
import torch.nn as nn


def build_encoder(weights: Path, enc_res: int, allow_random: bool) -> nn.Module:
    """Construct a SAM ViT-B image encoder at ``img_size=enc_res`` with weights loaded."""
    try:
        from segment_anything.modeling.image_encoder import ImageEncoderViT
    except Exception as exc:  # pragma: no cover - env guard
        raise SystemExit(
            "[export-enc] segment-anything not installed. Inside the app venv run:\n"
            "    pip install -r backend/requirements-sam.txt\n"
            f"(import error: {type(exc).__name__}: {exc})"
        )

    encoder = ImageEncoderViT(
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
    encoder.eval()

    if not weights.exists():
        if not allow_random:
            raise FileNotFoundError(
                f"[export-enc] FATAL {weights} not found. Refusing to build a random-init "
                "encoder for an accuracy run. Produce it with "
                "`python -m backend.bootstrap.sam.finetune_encoder`, or pass --allow-random "
                "for a throughput-only benchmark."
            )
        print(f"[export-enc] WARN {weights} not found; using random init (--allow-random).")
        return encoder

    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    enc_state = {
        k.replace("image_encoder.", ""): v
        for k, v in ckpt.items() if k.startswith("image_encoder.")
    }
    if "pos_embed" in enc_state:
        pe = enc_state["pos_embed"]
        dst_n = enc_res // 16
        if pe.shape[1] != dst_n:
            pe4d = pe.permute(0, 3, 1, 2)
            pe4d = torch.nn.functional.interpolate(pe4d, size=(dst_n, dst_n), mode="bicubic", align_corners=False)
            enc_state["pos_embed"] = pe4d.permute(0, 2, 3, 1).contiguous()
            print(f"[export-enc] pos_embed interpolated {pe.shape[1]}x{pe.shape[1]} -> {dst_n}x{dst_n}")

    target_rel_len = 2 * (enc_res // 16) - 1
    for _k, _v in list(enc_state.items()):
        if not (_k.endswith(".attn.rel_pos_h") or _k.endswith(".attn.rel_pos_w")):
            continue
        if _v.shape[0] == target_rel_len:
            continue
        _v3d = _v.unsqueeze(0).permute(0, 2, 1)
        _v3d = torch.nn.functional.interpolate(_v3d, size=target_rel_len, mode="linear", align_corners=False)
        enc_state[_k] = _v3d.permute(0, 2, 1).squeeze(0).contiguous()

    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    print(f"[export-enc] loaded {len(enc_state)} tensors from {weights.name} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return encoder


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export the fine-tuned 256×256 SAM image encoder to a fp16 OpenVINO IR."
    )
    ap.add_argument("--weights", default="models/sam256_stockft.pth",
                    help="Fine-tuned SAM checkpoint from finetune_encoder.py.")
    ap.add_argument("--enc-res", type=int, default=256,
                    help="Encoder input resolution (256 = reference operating point).")
    ap.add_argument("--out", default="models/sam_encoder_256_stockft",
                    help="Output IR dir (default: models/sam_encoder_256_stockft).")
    ap.add_argument("--allow-random", action="store_true",
                    help="Build a random-init encoder if --weights is missing (throughput-only).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = build_encoder(Path(args.weights), args.enc_res, args.allow_random)

    dummy = torch.randn(1, 3, args.enc_res, args.enc_res, dtype=torch.float32)
    onnx_path = out_dir / f"encoder_{args.enc_res}.onnx"
    print(f"[export-enc] ONNX export (static [1,3,{args.enc_res},{args.enc_res}], opset 17) -> {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            encoder, dummy, str(onnx_path),
            input_names=["image"], output_names=["image_embed"],
            opset_version=17, do_constant_folding=True, dynamic_axes=None,
        )

    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(out_dir / "encoder.xml"), compress_to_fp16=True)
    (out_dir / ".converted_ok").write_text(f"fp16\nenc_res={args.enc_res}\nweights={args.weights}\n")
    print(f"[export-enc] wrote {out_dir / 'encoder.xml'}")
    print("[export-enc] next: python -m backend.bootstrap.sam.export_decoder")


if __name__ == "__main__":
    main()
