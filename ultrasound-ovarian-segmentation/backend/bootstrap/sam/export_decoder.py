# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export Meta's stock SAM prompt encoder + mask decoder to a fp16 OpenVINO IR.

The SAM-256 arch fine-tunes only the image encoder; the prompt encoder and mask
decoder are stock Meta SAM ViT-B. This stage exports them (from
``sam_vit_b_01ec64.pth``) to ``models/sam_decoder_256_multimask/decoder.xml`` —
exactly the default the app loads (see ``src/config.py`` ``--sam-decoder``). The
decoder is exported in MULTIMASK mode: it emits 3 candidate masks + 3
predicted-IoU scores per prompt, and the runtime picks the mask with the highest
predicted IoU (``src/segmenters/sam.py:pick_best_mask``).

The prompt encoder is rebuilt at the ``enc_res//16`` grid so the decoder accepts
the 256-encoder's ``image_embeddings [1,256,16,16]``. Only the box-prompt path
(two corner points) is exported; the other inputs are kept in the IR signature
for stability but ignored (the runtime resizes the low-res mask to native itself,
so ``orig_im_size`` is a no-op here).

Static input shapes (all fixed, NPU-safe):
  image_embeddings  [1, 256, enc_res//16, enc_res//16]  float32
  point_coords      [1, 2, 2]          float32  (box corners, enc_res space)
  point_labels      [1, 2]             float32  (2=top-left, 3=bottom-right)
  mask_input        [1, 1, 256, 256]   float32  (zeroed)
  has_mask_input    [1]                float32  (0)
  orig_im_size      [2]                int32    (ignored)
Outputs:
  masks             [1, 3, 256, 256]   float32  (low-res logits)
  iou_predictions   [1, 3]             float32

Licence: SAM is Apache-2.0.

Usage (Windows, inside the app venv with backend/requirements-sam.txt installed):

    python -m backend.bootstrap.sam.export_decoder --weights models/sam_vit_b_01ec64.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import openvino as ov
import torch
import torch.nn as nn

SAM_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
EMBED_DIM = 256


class SamDecoderWrapper(nn.Module):
    """Stock SAM prompt encoder + mask decoder with fixed-shape box-prompt inputs."""

    def __init__(self, sam) -> None:
        super().__init__()
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder

    def forward(
        self,
        image_embeddings: torch.Tensor,   # [1, 256, g, g]
        point_coords: torch.Tensor,       # [1, 2, 2]   (TL, BR corners)
        point_labels: torch.Tensor,       # [1, 2]      (kept for IR signature; ignored)
        mask_input: torch.Tensor,         # [1, 1, 256, 256]   (kept for signature; ignored)
        has_mask_input: torch.Tensor,     # [1]         (kept for signature; ignored)
        orig_im_size: torch.Tensor,       # [2]         int32 (kept for signature; ignored)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Use SAM's boxes= code path (plain tensor indexing exports cleanly under
        # the OV 2026 ONNX frontend; the points= path uses label-scatter the
        # frontend rejects). Behaviour is identical to the 2-corner prompt.
        boxes = torch.cat([point_coords[:, 0, :], point_coords[:, 1, :]], dim=-1).unsqueeze(1)  # [1,1,4]
        sparse, dense = self.prompt_encoder(points=None, boxes=boxes, masks=None)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
        )
        # Keep the unused inputs alive in the IR signature (0*x is a no-op).
        zero_keep = (point_labels.sum() * 0.0 + mask_input.sum() * 0.0
                     + has_mask_input.sum() * 0.0 + orig_im_size.float().sum() * 0.0)
        return low_res_masks + zero_keep, iou_predictions + zero_keep


def build_decoder_wrapper(weights: Path, enc_res: int) -> SamDecoderWrapper:
    from segment_anything import sam_model_registry
    from segment_anything.modeling.prompt_encoder import PromptEncoder

    print(f"[export-dec] loading SAM ViT-B weights from {weights} ...")
    sam = sam_model_registry["vit_b"](checkpoint=str(weights))
    sam.eval()

    # Rebuild the prompt encoder at the smaller grid so its positional encoding
    # matches the 256-encoder's image_embeddings. The mask decoder is grid-agnostic.
    if enc_res != 1024:
        new_pe = PromptEncoder(
            embed_dim=EMBED_DIM,
            image_embedding_size=(enc_res // 16, enc_res // 16),
            input_image_size=(enc_res, enc_res),
            mask_in_chans=16,
        )
        stock = sam.prompt_encoder.state_dict()
        target = new_pe.state_dict()
        keep = {k: v for k, v in stock.items() if k in target and v.shape == target[k].shape}
        new_pe.load_state_dict(keep, strict=False)
        new_pe.eval()
        sam.prompt_encoder = new_pe
        print(f"[export-dec] prompt_encoder rebuilt for enc_res={enc_res} "
              f"(kept={len(keep)}, dropped={len(set(stock) - set(keep))})")

    wrapper = SamDecoderWrapper(sam)
    wrapper.eval()
    return wrapper


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export stock SAM prompt encoder + multimask mask decoder to a fp16 OpenVINO IR."
    )
    ap.add_argument("--weights", default="models/sam_vit_b_01ec64.pth",
                    help=f"SAM ViT-B weights (Apache-2.0). Download: {SAM_WEIGHTS_URL}")
    ap.add_argument("--enc-res", type=int, default=256,
                    help="Encoder input resolution; drives image_embeddings grid (enc_res // 16).")
    ap.add_argument("--out", default="models/sam_decoder_256_multimask",
                    help="Output IR dir (default: models/sam_decoder_256_multimask).")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[export-dec] ERROR: SAM weights not found at {weights}")
        print(f"[export-dec] Download with:  curl -L -o {weights} {SAM_WEIGHTS_URL}")
        sys.exit(1)

    enc_res = args.enc_res
    grid = enc_res // 16
    print(f"[export-dec] enc_res={enc_res}, image_embeddings grid={grid}x{grid}, multimask=True")

    try:
        wrapper = build_decoder_wrapper(weights, enc_res)
    except ImportError as exc:
        raise SystemExit(
            "[export-dec] segment-anything not installed. Inside the app venv run:\n"
            "    pip install -r backend/requirements-sam.txt\n"
            f"(import error: {exc})"
        )

    dummy = (
        torch.zeros(1, EMBED_DIM, grid, grid, dtype=torch.float32),  # image_embeddings
        torch.zeros(1, 2, 2, dtype=torch.float32),                   # point_coords
        torch.zeros(1, 2, dtype=torch.float32),                      # point_labels
        torch.zeros(1, 1, 256, 256, dtype=torch.float32),            # mask_input
        torch.zeros(1, dtype=torch.float32),                         # has_mask_input
        torch.tensor([enc_res, enc_res], dtype=torch.int32),         # orig_im_size
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "decoder.onnx"
    print(f"[export-dec] ONNX export (static shapes, opset 17) -> {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, str(onnx_path),
            input_names=["image_embeddings", "point_coords", "point_labels",
                         "mask_input", "has_mask_input", "orig_im_size"],
            output_names=["masks", "iou_predictions"],
            opset_version=17, do_constant_folding=True, dynamic_axes=None,
        )

    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(out_dir / "decoder.xml"), compress_to_fp16=True)
    (out_dir / ".converted_ok").write_text(f"fp16\nenc_res={enc_res}\nmultimask=True\n")
    print(f"[export-dec] wrote {out_dir / 'decoder.xml'}")

    core = ov.Core()
    m = core.read_model(str(out_dir / "decoder.xml"))
    print("[export-dec] IR inputs:")
    for inp in m.inputs:
        print(f"    {inp.get_any_name():18s} {inp.partial_shape}")
    print("[export-dec] IR outputs:")
    for out in m.outputs:
        print(f"    {out.get_any_name():18s} {out.partial_shape}")
    print("[export-dec] next: python -m backend.bootstrap.sam.eval")


if __name__ == "__main__":
    main()
