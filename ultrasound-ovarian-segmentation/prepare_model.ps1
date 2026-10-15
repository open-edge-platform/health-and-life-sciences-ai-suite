# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Model preparation: export (and optionally train) a segmentation model and place
# its OpenVINO IR(s) where the app expects them.
#
# -Arch ds2net (default): the promptless SegFormer-B5 model ->
#   models\ds2net_segformer_b5_<Res>\model.xml
# -Arch sam: the SAM-256 pipeline (three IRs) ->
#   models\yolo_mmotu_320\yolov8n_mmotu.xml
#   models\sam_encoder_256_stockft\encoder.xml
#   models\sam_decoder_256_multimask\decoder.xml
#
# The app does NOT ship trained IRs. Run this once per arch before .\run.ps1.
#
# Usage (from this folder, PowerShell, after .\setup.ps1):
#   # DS2Net: export an existing trained checkpoint -> IR, then verify Dice:
#   .\prepare_model.ps1 -Verify
#   # DS2Net: train on the Intel iGPU first, then export + verify:
#   .\prepare_model.ps1 -Train -Verify
#
#   # SAM: export the three IRs from downloaded/fine-tuned weights, then verify:
#   .\prepare_model.ps1 -Arch sam -Verify
#   # SAM: train YOLO + fine-tune the encoder first, then export + verify:
#   .\prepare_model.ps1 -Arch sam -Train -Verify
#
[CmdletBinding()]
param(
    [ValidateSet("ds2net", "sam")]
    [string]$Arch = "ds2net",
    [string]$Checkpoint = "models\segformer_b5_mmotu",
    [int]$Res = 256,
    [string]$Data = "data\OTU_2d",
    [ValidateSet("GPU", "CPU", "NPU")]
    [string]$Device = "GPU",
    [switch]$Train,
    [switch]$Verify,
    [string]$Venv = "venv",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "virtual environment not found ($py). Run .\setup.ps1 first."
}

if ($Arch -eq "sam") {
    # SAM-256 pipeline: three IRs (YOLO@320 bbox, encoder@256, multimask decoder).
    # Probe importability without aborting under $ErrorActionPreference = "Stop":
    # a fresh venv prints a traceback on stderr that Windows PowerShell 5.1
    # promotes to a terminating NativeCommandError. Relax it just here.
    $eap = $ErrorActionPreference; $ErrorActionPreference = "SilentlyContinue"
    & $py -c "import torch, ultralytics, segment_anything" 2>$null
    $haveBackend = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $eap
    if ($SkipInstall -or $haveBackend) {
        Write-Host "[prepare] SAM model-prep dependencies already present - skipping pip install"
    } else {
        Write-Host "[prepare] installing SAM model-prep dependencies (backend\requirements-sam.txt)"
        & $py -m pip install -r backend\requirements-sam.txt
        if ($LASTEXITCODE -ne 0) {
            Write-Error ("pip install failed. Reuse a venv that already has " +
                "torch+ultralytics+segment-anything via -Venv <path>, or install from a local wheel cache " +
                "with --no-index --find-links <wheels_dir>.")
        }
    }

    $stockWeights = "models\sam_vit_b_01ec64.pth"
    if (-not (Test-Path $stockWeights)) {
        Write-Error ("stock SAM ViT-B weights not found: $stockWeights`n" +
            "Download (Apache-2.0):`n" +
            "  curl -L -o $stockWeights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
    }

    if ($Train) {
        Write-Host "[prepare] SAM -Train: YOLO labels + YOLOv8n train + SAM encoder fine-tune (the long step)"
        & $py -m backend.bootstrap.sam.prepare_yolo_data --data $Data
        if ($LASTEXITCODE -ne 0) { Write-Error "prepare_yolo_data failed" }
        & $py -m backend.bootstrap.sam.train_yolo --device cpu
        if ($LASTEXITCODE -ne 0) { Write-Error "train_yolo failed" }
        & $py -m backend.bootstrap.sam.finetune_encoder --weights $stockWeights --data $Data --device xpu
        if ($LASTEXITCODE -ne 0) { Write-Error "finetune_encoder failed" }
    }

    $yoloWeights = "models\yolov8n_mmotu.pt"
    $encWeights = "models\sam256_stockft.pth"
    if (-not (Test-Path $yoloWeights)) {
        Write-Error ("trained YOLO weights not found: $yoloWeights`n" +
            "Run with -Train, or place a trained YOLOv8n MMOTU-2D checkpoint there.")
    }
    if (-not (Test-Path $encWeights)) {
        Write-Error ("fine-tuned SAM encoder not found: $encWeights`n" +
            "Run with -Train, or place your downloaded sam256_stockft.pth there.")
    }

    Write-Host "[prepare] exporting the three SAM IRs (YOLO@320, encoder@256, multimask decoder)"
    & $py -m backend.bootstrap.sam.export_yolo
    if ($LASTEXITCODE -ne 0) { Write-Error "export_yolo failed" }
    & $py -m backend.bootstrap.sam.export_encoder --weights $encWeights
    if ($LASTEXITCODE -ne 0) { Write-Error "export_encoder failed" }
    & $py -m backend.bootstrap.sam.export_decoder --weights $stockWeights
    if ($LASTEXITCODE -ne 0) { Write-Error "export_decoder failed" }

    $encXml = "models\sam_encoder_256_stockft\encoder.xml"
    if (-not (Test-Path $encXml)) { Write-Error "export did not produce $encXml" }

    if ($Verify) {
        Write-Host "[prepare] verifying end-to-end Dice/IoU on the MMOTU-2D val split (device=$Device)"
        & $py -m backend.bootstrap.sam.eval --device $Device --data $Data
        if ($LASTEXITCODE -ne 0) { Write-Error "eval failed" }
        Write-Host "[prepare] expected reference at res 256: end-to-end Dice ~0.8207 / IoU ~0.7394"
    }

    Write-Host "[prepare] done. SAM IRs ready under models\yolo_mmotu_320, models\sam_encoder_256_stockft, models\sam_decoder_256_multimask"
    Write-Host "[prepare] launch the demo:"
    Write-Host "  .\run.ps1 -Arch sam -Source file -Input <clip.mp4> -Device $Device"
    return
}

# Skip the online install when torch+transformers are already importable (e.g. an
# existing venv). Pass -Venv <path> to reuse an environment, or -SkipInstall to
# force-skip.
#
# Probe importability without aborting under $ErrorActionPreference = "Stop": a
# fresh venv prints a traceback on stderr that Windows PowerShell 5.1 promotes to
# a terminating NativeCommandError. Relax it just here.
$eap = $ErrorActionPreference; $ErrorActionPreference = "SilentlyContinue"
& $py -c "import torch, transformers" 2>$null
$haveBackend = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $eap
if ($SkipInstall -or $haveBackend) {
    Write-Host "[prepare] model-prep dependencies already present - skipping pip install"
} else {
    Write-Host "[prepare] installing model-prep dependencies (backend\requirements.txt)"
    & $py -m pip install -r backend\requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error ("pip install failed. Reuse a venv that already has " +
            "torch+transformers via -Venv <path>, or install from a local wheel cache with " +
            "--no-index --find-links <wheels_dir>.")
    }
}

if ($Train) {
    Write-Host "[prepare] training SegFormer-B5 on MMOTU-2D (this is the long step)"
    & $py -m backend.bootstrap.train --data $Data --res $Res --device xpu --out $Checkpoint
    if ($LASTEXITCODE -ne 0) { Write-Error "training failed" }
}

if (-not (Test-Path $Checkpoint)) {
    Write-Error ("trained checkpoint not found: $Checkpoint`n" +
        "Train it first with:  .\prepare_model.ps1 -Train`n" +
        "or point -Checkpoint at an existing HF SegFormer checkpoint.")
}

$irDir = "models\ds2net_segformer_b5_$Res"
Write-Host "[prepare] exporting $Checkpoint -> $irDir (fp16 OpenVINO IR)"
& $py -m backend.bootstrap.export --from-pretrained $Checkpoint --res $Res --out $irDir
if ($LASTEXITCODE -ne 0) { Write-Error "export failed" }

$irXml = Join-Path $irDir "model.xml"
if (-not (Test-Path $irXml)) { Write-Error "export did not produce $irXml" }

if ($Verify) {
    Write-Host "[prepare] verifying Dice/IoU on the MMOTU-2D val split (device=$Device)"
    & $py -m backend.bootstrap.eval --ir $irXml --res $Res --data $Data --device $Device
    if ($LASTEXITCODE -ne 0) { Write-Error "eval failed" }
    Write-Host "[prepare] expected reference at res 256: Dice ~0.8657 / IoU ~0.7913"
}

Write-Host "[prepare] done. Model ready at $irXml"
Write-Host "[prepare] launch the demo:"
Write-Host "  .\run.ps1 -Source file -Input <clip.mp4> -Device $Device"
