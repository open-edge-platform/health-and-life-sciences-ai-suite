# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-time setup: create a local virtual environment and install runtime deps.
# Usage (from this folder, PowerShell):
#   .\setup.ps1
#
[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Venv = "venv",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path $Venv)) {
    Write-Host "[setup] creating virtual environment: $Venv"
    & $Python -m venv $Venv
}

$py = Join-Path $Venv "Scripts\python.exe"

# If the runtime deps are already importable (e.g. an existing venv), skip the
# online pip install so setup works offline. Pass -Venv <path> to reuse an
# environment, or -SkipInstall to force-skip.
#
# Probe importability WITHOUT aborting: a fresh venv prints a ModuleNotFoundError
# traceback on stderr, which Windows PowerShell 5.1 promotes to a terminating
# NativeCommandError under $ErrorActionPreference = "Stop". Relax it just here.
$eap = $ErrorActionPreference; $ErrorActionPreference = "SilentlyContinue"
& $py -c "import openvino, cv2, numpy" 2>$null
$haveRuntime = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $eap

if ($SkipInstall -or $haveRuntime) {
    Write-Host "[setup] runtime dependencies already present - skipping pip install"
} else {
    Write-Host "[setup] upgrading pip"
    & $py -m pip install --upgrade pip wheel
    Write-Host "[setup] installing requirements.txt"
    & $py -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error ("pip install failed. Reuse a venv that already has " +
            "openvino/opencv/numpy/pywin32 via -Venv <path>, or install from a local wheel " +
            "cache: python -m pip install --no-index --find-links <wheels_dir> -r requirements.txt")
    }
}

Write-Host "[setup] done. Next:"
Write-Host "  .\prepare_model.ps1 -Verify                 # produce models\ds2net_segformer_b5_256\model.xml"
Write-Host "  .\run.ps1 -Source file -Input <clip.mp4>    # segment a video"
Write-Host "  (model prep walkthrough: docs/user-guide/get-started/model-preparation.md)"
