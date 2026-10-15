# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-preparation bootstrap for the SAM-256 arch of the ovarian app.

SAM-256 is a three-model pipeline:

    frame -> YOLOv8n@320 bbox -> SAM ViT-B encoder@256 (fine-tuned) ->
    stock SAM prompt-encoder + mask-decoder (frozen, multimask) ->
    argmax-by-IoU -> tumor mask

Six build stages produce the three OpenVINO IRs the app loads (see
``src/config.py``), and one eval stage verifies the assembled pipeline:

    prepare_yolo_data.py  MMOTU-2D masks -> YOLOv8 detection labels
    train_yolo.py         YOLOv8n fine-tune on MMOTU-2D (the bbox prompt)
    export_yolo.py        trained YOLO -> models/yolo_mmotu_320/yolov8n_mmotu.xml
    finetune_encoder.py   stock SAM ViT-B encoder fine-tune @256 -> models/sam256_stockft.pth
    export_encoder.py     sam256_stockft.pth -> models/sam_encoder_256_stockft/encoder.xml
    export_decoder.py     stock SAM prompt-enc + mask-decoder -> models/sam_decoder_256_multimask/decoder.xml
    eval.py               end-to-end Dice/IoU of the built pipeline on the MMOTU-2D val split

The app does NOT ship trained IRs. Run these (or ``prepare_model.ps1 -Arch sam``)
to produce the three IRs before launching ``run.ps1 -Arch sam``. The reference
MMOTU-2D val result is end-to-end Dice 0.8207 / IoU 0.7394 at res 256.

Licence: the SAM path (image encoder + prompt encoder + mask decoder, via
``segment-anything``) is Apache-2.0. Only the YOLO detector (``ultralytics``) is
AGPL-3.0, and it is BUILD-TIME ONLY — the runtime app loads the exported
OpenVINO IR with OpenCV/OpenVINO and never imports ultralytics.
"""
