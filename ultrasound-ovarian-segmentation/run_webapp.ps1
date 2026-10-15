# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch the ovarian ultrasound web dashboard (Flask control plane + React UI).
#
# The model/use-case is chosen on the CLI (-UseCase); the browser shows device
# selection plus live hardware-utilization and pipeline KPIs, and Start runs that
# model. The segmentation video itself opens in a separate OpenCV pop-up
# window (not streamed to the browser). Inference reuses the same src/ pipeline
# as run.ps1, so there is a single code path for the CLI and the UI.
#
# Usage (from this folder, PowerShell):
#   .\run_webapp.ps1                     # default model (DS2Net)
#   .\run_webapp.ps1 -UseCase sam        # run the SAM model instead
#   .\run_webapp.ps1 -Venv C:\path\to\venv
#   .\run_webapp.ps1 -Port 8080 -NoBrowser
#
[CmdletBinding()]
param(
    [string]$Venv = "venv",
    [int]$Port = 8080,
    [string]$UseCase = "ds2net",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# The model/use-case is chosen here on the CLI; the UI only displays it and the
# Start button runs it. Accept a friendly alias (ds2net / sam) or a full key.
switch -Regex ($UseCase) {
    '^(ds2net|ovarian_ds2net)$'         { $UseCase = "ovarian_ds2net" }
    '^(sam|ultrasam|ovarian_ultrasam)$' { $UseCase = "ovarian_ultrasam" }
}
$env:WEBAPP_USECASE = $UseCase

$py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "virtual environment not found ($py). Run .\setup.ps1 first (or pass -Venv)."
}

$env:WEBAPP_PORT = "$Port"
$url = "http://127.0.0.1:$Port"

if (-not $NoBrowser) {
    # The React UI polls readiness and retries, so opening before the server is
    # fully up is harmless.
    Start-Process $url | Out-Null
}

Write-Host "[run_webapp] model=$UseCase  ->  $url"
Write-Host "[run_webapp] $py -m webapp.server"
& $py -m webapp.server
