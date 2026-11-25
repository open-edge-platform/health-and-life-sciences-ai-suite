<!--hide_directive
```{eval-rst}
.. meta::
   :description: Solutions to common issues with the Ultrasound Ovarian Segmentation sample application, including model, device, display, and web dashboard problems.
   :keywords: troubleshooting, model IR, GPU fallback, NPU, pop-up window, Dice, web dashboard
```
hide_directive-->

# Troubleshooting

## Error: "virtual environment not found"

Run `.\setup.ps1` from the `ultrasound-ovarian-segmentation` folder first. It creates `venv`
and installs `requirements.txt`.

## PowerShell Blocks the Scripts

Allow scripts for the current session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Error: "model IR not found: models\ds2net_segformer_b5_256\model.xml"

The application does not include a model. Either drop in a prebuilt Intermediate
Representation (IR) or produce one:

- **Prebuilt IR** — if you have the exported `model.xml` and `model.bin` (for example, from
  another machine), place both under `models\ds2net_segformer_b5_256\`. That is all the
  application needs to run.
- **Produce it:**

  ```powershell
  .\prepare_model.ps1 -Verify          # export an existing checkpoint
  .\prepare_model.ps1 -Train -Verify   # or train first
  ```

See [Model Preparation](./get-started/model-preparation.md) for details.

## Device Not Available or Falls Back to CPU

If `--device GPU` is not among the available devices of OpenVINO™,
the application logs a warning and falls back to CPU. Check that:

- Intel graphics drivers are installed and current.
- The device appears in Task Manager and
  `venv\Scripts\python.exe -c "import openvino as ov; print(ov.Core().available_devices)"`
  lists `GPU`.

> [!NOTE]
> **NPU:** neither model (DS2Net or SAM) is supported on the NPU. Passing `--device NPU` logs
> a notice and uses the GPU (or CPU) instead.

## No Pop-Up Window Appears

- Make sure OpenCV has GUI support: `requirements.txt` installs `opencv-python`
  (not `opencv-python-headless`). A headless build renders no window. Verify with
  `venv\Scripts\python.exe -c "import cv2; print(cv2.getBuildInformation())"` and look
  for a GUI backend (e.g., WIN32UI).
- Run from an interactive desktop session (not a service or SSH-only session).
- Check the console for a source or device error that prevented the run from starting.

## Mean Dice Is ~0 After Export

The most likely causes are:

- The IR was exported **without** `--from-pretrained` (random weights). Re-run:

  ```powershell
  venv\Scripts\python.exe -m backend.bootstrap.export `
      --from-pretrained models\segformer_b5_mmotu --res 256
  ```

- `--res` does not match the resolution the IR was exported at.
- Dataset paths are wrong — `eval.py` could not find `images/` or `annotations/` under
  `--data`, or `val.txt` lists the wrong stems.

## SAM Architecture: IR Not Found or Low Dice

With `-Arch sam`, the application loads three IRs (YOLO bounding box, SAM encoder,
SAM decoder). If any is missing, prepare them and confirm the stock SAM ViT-B weights were
downloaded to `models\sam_vit_b_01ec64.pth` first:

```powershell
.\prepare_model.ps1 -Arch sam -Verify
```

A low end-to-end Dice (well under 0.82) usually means the encoder IR was exported from
random weights (missing `--weights models\sam256_stockft.pth`) or the encoder was
fine-tuned with the wrong preprocessing. The fine-tune and the runtime must both scale
images by `/255` (no mean or standard deviation normalization). See
[Model Preparation](./get-started/model-preparation.md#sam-256-architecture).

## SAM Training Runs on CPU — How to Use the iGPU

`.\prepare_model.ps1 -Arch sam -Train` trains on the Intel iGPU by default
(`-TrainDevice GPU` → PyTorch `xpu`) for both the YOLO detector and the encoder fine-tune.
To force CPU, pass `-TrainDevice CPU`. If training still falls back to CPU with
`-TrainDevice GPU`, the PyTorch build in `venv` has no Intel XPU support. Verify with:

```powershell
venv\Scripts\python.exe -c "import torch; print(hasattr(torch,'xpu') and torch.xpu.is_available())"
```

If this prints `False`, install a PyTorch build with Intel XPU support (or run on CPU with
`-TrainDevice CPU`, which is slower). `-TrainDevice` selects the training device (default
`GPU`) and is separate from `-Device`, which selects the OpenVINO™ evaluation/runtime
device. The runtime/export device name is the OpenVINO™ `GPU`; the training device name is
the PyTorch `xpu` — they are not interchangeable.

## GPU KPI Prints "OVER"

This is unexpected on typical ultrasound workloads (the pipeline is host-bound, utilizing
~12–24% iGPU). If you see `OVER`, the device was saturated by other work, or the governor
was disabled (`--no-governor`). Re-run without `--no-governor`, or lower `--gpu-cap` to
further pace inference down.

## GPU HUD Shows "unavailable" or "governor inactive"

The governor reads Windows GPU-Engine performance counters via **pywin32**. If `pywin32` is
missing, or you run on a non-Windows machine, the governor becomes inactive and the HUD
marks GPU utilization unavailable — the application still runs. On Windows, reinstall
dependencies: `.\setup.ps1`.

## Webcam Opens the Wrong Camera

`--input` is the device index. Try `0`, `1`, or `2`. Close other applications that may
hold the camera.

## Web Dashboard: Page Does Not Load

- Confirm the server is running: the console prints
  `[webapp] serving on http://127.0.0.1:8080`. Open that exact URL.
- If the port is busy, start on another port: `.\run_webapp.ps1 -Port 8090`.
- A blank page usually means the prebuilt UI (`webapp\ui\dist`) is missing.
  `.\run_webapp.ps1` builds it automatically on the first launch — this needs
  Node.js 18+ on `PATH` — then reuses it on later runs. If the page stays blank,
  confirm Node.js is installed and re-run `.\run_webapp.ps1` so it can rebuild,
  or build the UI manually: `cd webapp\ui; npm install; npm run build`.

## Web Dashboard: Start Fails with "input not found"

Enter a valid path in **Video / image source** (a `.mp4` file or a folder of frames), or
leave the field blank to use the `data\OTU_2d\images` dataset you prepared during model
preparation. The path is resolved on the machine running the server.

## Web Dashboard: Start Fails with "input source must be inside the app folder or your home folder"

For safety, the web dashboard only reads sources inside the application folder or your
user profile folder (for example `C:\Users\<you>`). To use clips stored elsewhere, such
as another drive, list the extra folders in `WEBAPP_INPUT_ROOTS` (separate multiple
folders with `;`) before launching:

```powershell
$env:WEBAPP_INPUT_ROOTS = "D:\ultrasound;E:\clips"
.\run_webapp.ps1
```

The command-line application (`run.ps1`) is not affected.

## Web Dashboard: "HW monitoring: DISABLED"

The browser still shows CPU and memory (via `psutil`), but the iGPU, NPU, and power graphs
need the Intel vendor tools and their Python dependencies:

- Install `pandas` and `pywin32` into the `venv`.
- Copy the vendor tools from your benchmark install into this clone's
  `monitoring\tools`: **SoCWatch** (drives the power chart) and **Level Zero**
  (drives the NPU chart). They are independent — copy either or both; whichever
  you omit simply shows as unavailable while every other metric keeps working.

These are intentionally not committed (≈139 MB of vendor binaries). Without them, the
pipeline and its latency, FPS, and iGPU KPIs still work.
