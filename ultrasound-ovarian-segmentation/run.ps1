# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch the ovarian tumor segmentation pop-up.
# Usage (from this folder, PowerShell):
#   .\run.ps1 -Source file   -Input C:\path\to\clip.mp4
#   .\run.ps1 -Source webcam -Input 0
#   .\run.ps1 -Source folder -Input C:\path\to\frames   -Device GPU
#
#   # Pick the model arch (default ds2net); SAM-256 needs its three IRs prepared:
#   .\run.ps1 -Arch sam -Source file -Input C:\path\to\clip.mp4
#
# Any extra app flags go after a literal `--` and are forwarded as-is, e.g.:
#   .\run.ps1 -Source file -Input clip.mp4 -- --record out.mp4 --log-file reports\run.log
#
[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("file", "webcam", "folder")]
    [string]$Source = "file",
    [Alias("Input")]
    [string]$InputPath = "",
    [ValidateSet("ds2net", "sam")]
    [string]$Arch = "ds2net",
    [ValidateSet("GPU", "CPU", "NPU", "AUTO")]
    [string]$Device = "GPU",
    [string]$Model = "",
    [string]$Venv = "venv",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "virtual environment not found ($py). Run .\setup.ps1 first."
}

$cliArgs = @("-m", "src.app", "--source", $Source, "--device", $Device, "--model-arch", $Arch)
if ($InputPath) { $cliArgs += @("--input", $InputPath) }
if ($Model) { $cliArgs += @("--model", $Model) }
if ($Extra) { $cliArgs += $Extra }

Write-Host "[run] $py $($cliArgs -join ' ')"
& $py @cliArgs
