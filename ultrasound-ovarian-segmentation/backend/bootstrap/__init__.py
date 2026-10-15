# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-preparation bootstrap for the ovarian segmentation app.

Three stages, each runnable on its own:

    train.py    MMOTU-2D fine-tune of a promptless SegFormer-B5 (the DS2Net model)
    export.py   HF checkpoint -> fp16 OpenVINO IR (what the app loads)
    eval.py     Dice / IoU of an exported IR on the MMOTU-2D val split

The app does NOT ship a trained IR. Run these (or ``prepare_model.ps1``) to
produce ``models/ds2net_segformer_b5_256/model.xml`` before launching ``run.ps1``.
"""
