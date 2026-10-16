# Troubleshooting

## `virtual environment not found`

Run `.\setup.ps1` from the `ultrasound-ovarian-segmentation` folder first. It
creates `venv` and installs `requirements.txt`.

## PowerShell blocks the scripts

Allow scripts for the current session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## `model IR not found: models\ds2net_segformer_b5_256\model.xml`

The app does not ship a model. Either drop in a prebuilt IR or produce one:

- **Prebuilt IR** — if you have the exported `model.xml` + `model.bin` (for
  example, from another box), place both under `models\ds2net_segformer_b5_256\`.
  That is all the app needs to run.
- **Produce it:**

  ```powershell
  .\prepare_model.ps1 -Verify          # export an existing checkpoint
  .\prepare_model.ps1 -Train -Verify   # or train first
  ```

See [Model Preparation](./get-started/model-preparation.md).

## Device not available / falls back to CPU

If `--device GPU` is not in OpenVINO's available devices, the app logs a warning
and falls back to CPU. Check that:

- Intel graphics drivers are installed and current.
- The device appears in Task Manager and `venv\Scripts\python.exe -c "import openvino as ov; print(ov.Core().available_devices)"` lists `GPU`.

> **NPU:** neither model (DS2Net or SAM) is supported on the NPU. Passing
> `--device NPU` logs a notice and uses the GPU (or CPU) instead.

## No pop-up window appears

- Make sure OpenCV has GUI support: `requirements.txt` installs `opencv-python`
  (not `opencv-python-headless`). A headless build renders no window. Verify with
  `venv\Scripts\python.exe -c "import cv2; print(cv2.getBuildInformation())"` and
  look for a GUI backend (e.g. WIN32UI).
- Run from an interactive desktop session (not a service / SSH-only session).
- Check the console for a source or device error that prevented the run from
  starting.

## Mean Dice is ~0 after export

Almost always one of:

- The IR was exported **without** `--from-pretrained` (random weights). Re-run
  `export.py --from-pretrained models\segformer_b5_mmotu --res 256`.
- `--res` does not match the resolution the IR was exported at.
- Dataset paths are wrong — `eval.py` could not find `images/` or
  `annotations/` under `--data`, or `val.txt` lists the wrong stems.

## SAM arch: IR not found or low Dice

With `-Arch sam` the app loads three IRs (YOLO bbox, SAM encoder, SAM decoder).
If any is missing, prepare them — and confirm the stock SAM ViT-B weights were
downloaded to `models\sam_vit_b_01ec64.pth` first:

```powershell
.\prepare_model.ps1 -Arch sam -Verify
```

A low end-to-end Dice (well under 0.82) usually means the encoder IR was exported
from random weights (missing `--weights models\sam256_stockft.pth`) or the
encoder was fine-tuned with the wrong preprocessing — the fine-tune and the
runtime must both scale images by `/255` (no mean/std). See
[Model Preparation](./get-started/model-preparation.md#sam-256-arch).

## SAM training runs on CPU — how to use the iGPU

`.\prepare_model.ps1 -Arch sam -Train` trains on the Intel iGPU by default
(`-TrainDevice GPU` → PyTorch `xpu`) for both the YOLO detector and the encoder
fine-tune. To force CPU, pass `-TrainDevice CPU`. If training still falls back to
CPU with `-TrainDevice GPU`, the venv's PyTorch has no Intel XPU support — confirm
with:

```powershell
venv\Scripts\python.exe -c "import torch; print(hasattr(torch,'xpu') and torch.xpu.is_available())"
```

If this prints `False`, install a PyTorch build with Intel XPU support (or run on
CPU with `-TrainDevice CPU`, which is slower). `-TrainDevice` selects the training
device (default `GPU`) and is separate from `-Device`, which selects the OpenVINO
eval/runtime device. The runtime/export device name is the OpenVINO `GPU`; the
training device name is the PyTorch `xpu` — they are not interchangeable.

## GPU KPI prints `OVER`

Unexpected on typical ultrasound workloads (the pipeline is host-bound, ~12–24%
iGPU). If you see `OVER`, the device was saturated by other work, or the governor
was disabled (`--no-governor`). Re-run without `--no-governor`, or lower
`--gpu-cap` to pace inference down further.

## GPU HUD shows "unavailable" / governor inactive

The governor reads Windows GPU-Engine performance counters via **pywin32**. If
`pywin32` is missing, or you run on a non-Windows machine, the governor degrades
to a no-op and the HUD marks GPU utilization unavailable — the app still runs. On
Windows, reinstall deps: `.\setup.ps1`.

## Webcam opens the wrong camera

`--input` is the device index. Try `0`, `1`, `2`. Close other apps that may hold
the camera.

## Web dashboard: page does not load

- Confirm the server is running: the console prints
  `[webapp] serving on http://127.0.0.1:8080`. Open that exact URL.
- If the port is busy, start on another port: `.\run_webapp.ps1 -Port 8090`.
- A blank page usually means the prebuilt UI is missing. The clone must contain
  `webapp\ui\dist\index.html` (it is committed). If it is absent you are on a
  branch without the build — re-clone.

## Web dashboard: Start fails with `input not found`

Enter a valid path in **Video / image source** (a `.mp4` file or a folder of
frames), or leave the field blank to use the `data\OTU_2d\images` dataset you
prepared during model preparation. The path is resolved on the machine running
the server.

## Web dashboard: `HW monitoring: DISABLED`

The browser still shows CPU + memory (via `psutil`), but the iGPU / NPU / power
graphs need the Intel vendor tools and their Python deps:

- Install `pandas` + `pywin32` into the `venv`.
- Copy a `monitoring\tools` folder (SoCWatch / Level Zero) from your benchmark
  install into this clone's `monitoring\tools`.

These are intentionally not committed (≈139 MB of vendor binaries). Without them
the pipeline and its latency / FPS / iGPU KPIs still work.
