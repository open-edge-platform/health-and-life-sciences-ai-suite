# Ultrasound Ovarian Segmentation Sample App

> [!NOTE]
> This application is for **reference and evaluation purposes only**. It is
> **not intended for direct use in clinical or diagnostic environments** and is
> not validated for such a purpose.

Real-time ovarian tumor segmentation in 2D ultrasound on the Intel iGPU (with a
CPU fallback) using OpenVINO. The app ships two interchangeable models, selected
with `-Arch`: the default **promptless SegFormer-B5 (DS2Net)** — it takes the
whole frame and outputs a tumor mask, with no bounding-box prompt and no separate
detector — and an optional **prompted SAM-256** pipeline (YOLOv8n bbox → SAM
ViT-B encoder → mask decoder). Either way the mask is overlaid on a live pop-up
window. A built-in **GPU utilization cap** keeps the Intel iGPU under an 80%
busiest-engine ceiling and records the peak/mean so the KPI can be proven after a
run.

|   |   |
|---|---|
| **Models** | `ds2net` — promptless SegFormer MiT-B5 (default); `sam` — YOLOv8n bbox → SAM-256. Both fp16 OpenVINO IR |
| **Accuracy** | DS2Net Dice 0.8657 / IoU 0.7913; SAM Dice 0.8207 / IoU 0.7394 — MMOTU-2D val (res 256) |
| **Inference** | Pure OpenVINO runtime (no PyTorch at run time) |
| **Runtime** | Native Windows Python app (PowerShell + venv) |
| **Inputs** | Video file, USB webcam, or a folder of ultrasound stills |
| **Display** | OpenCV pop-up: original + segmented side-by-side (or `--view overlay`) + FPS / GPU HUD |
| **GPU cap** | Adaptive governor enforces the iGPU < 80% utilization KPI |

## Topology

Decoupled capture / inference / display threads, so the displayed video stays
smooth regardless of inference speed.

```text
source (file / webcam / folder)
        |
        v
  capture thread ──► display queue ──► display (main thread): overlay + HUD pop-up
        │
        └────────► inference queue ──► inference thread: OpenVINO SegFormer ──► mask
                                              │
                                              └── GPU governor paces this loop
                                                  if iGPU utilization nears the cap
```

> With `-Arch sam` the inference thread runs the YOLO → SAM encoder → mask-decoder
> pipeline instead of SegFormer; the capture / inference / display topology is
> identical.

## Model Preparation

This app does **not** ship a trained model. It loads its default IR from
`models/ds2net_segformer_b5_256/model.xml`; get one there in any of these ways:

- **Already have a prebuilt IR?** If someone handed you the exported `model.xml`
  + `model.bin` (for example, copied from another evaluation box), just place the
  two files under `models\ds2net_segformer_b5_256\` and skip to
  [Quickstart](#quickstart) — no model-prep dependencies, dataset, or training
  needed.
- **Export an existing checkpoint** — if you already have a trained SegFormer
  checkpoint at `models/segformer_b5_mmotu`:

  ```powershell
  .\setup.ps1                     # one-time: create venv + install runtime deps
  .\prepare_model.ps1 -Verify     # export checkpoint -> fp16 IR, then verify Dice
  ```

- **Train from scratch** — download the MMOTU-2D dataset and the MiT-B5 backbone,
  then train on the Intel iGPU before exporting:

  ```powershell
  .\setup.ps1
  .\prepare_model.ps1 -Train -Verify   # train -> export -> verify
  ```

The commands above build the default `ds2net` arch. For the `sam` arch, add
`-Arch sam` — it produces three IRs and needs the stock SAM ViT-B weights
downloaded to `models/` first (see the guide):

```powershell
.\prepare_model.ps1 -Arch sam -Verify        # export the three SAM IRs, then verify
.\prepare_model.ps1 -Arch sam -Train -Verify # train YOLO + fine-tune encoder first
```

See [Model Preparation](docs/user-guide/get-started/model-preparation.md) for the
full end-to-end walkthrough (dataset layout, backbone download, training, export,
and the expected Dice).

## Quickstart

From this directory, after model preparation:

```powershell
.\run.ps1 -Source file -Input C:\path\to\clip.mp4 -Device GPU
```

A pop-up window opens showing the **original frame alongside the segmented frame**
(tumor-mask overlay) and a HUD (display/inference FPS, latency, and live iGPU
utilization); pass `--view overlay` for a single overlaid pane. Press `q` or `Esc`
to quit, or close the window. On exit the app prints the GPU KPI verdict, for
example:

```text
GPU KPI (< 80%): PASS — peak busiest-engine 17.8% mean 11.5% (samples=900, throttle_engaged=False)
```

## Web Dashboard (UI)

Prefer a browser UI over the CLI? `run_webapp.ps1` serves a React dashboard —
device / use-case selection, live inference latency / FPS, the iGPU KPI, and
CPU / iGPU / NPU / power / memory graphs — on top of the **same** `src/` pipeline
(the segmentation itself still opens in the OpenCV pop-up). It needs a few extra
Python packages (Flask + psutil, and pandas for the power graph):

```powershell
.\venv\Scripts\pip install -r webapp\requirements.txt
.\run_webapp.ps1                                         # serves http://127.0.0.1:8080
```

Full setup, including the hardware-utilization graphs, is in
[Get Started](docs/user-guide/get-started.md#4-run-the-web-dashboard-ui).

## Common Commands

Run against a folder of ultrasound stills (paced like a slideshow):

```powershell
.\run.ps1 -Source folder -Input C:\path\to\frames -Device GPU
```

Run against a USB webcam (device index 0):

```powershell
.\run.ps1 -Source webcam -Input 0 -Device GPU
```

Run on the CPU instead of the iGPU (development / sanity fallback):

```powershell
.\run.ps1 -Source file -Input clip.mp4 -Device CPU
```

Run the prompted SAM-256 arch instead of the default DS2Net (after preparing its
IRs):

```powershell
.\run.ps1 -Arch sam -Source file -Input C:\path\to\clip.mp4 -Device GPU
```

> Both models target the Intel iGPU (`-Device GPU`); `CPU` is a development /
> sanity fallback. The **NPU is not supported** — passing `-Device NPU` logs a
> notice and uses the GPU (or CPU) instead.

Record an annotated video, or tighten the GPU cap (extra flags after `--` are
forwarded to the app):

```powershell
.\run.ps1 -Source file -Input clip.mp4 -- --record out.mp4 --gpu-cap 70
```

See [Runtime Configuration](docs/user-guide/runtime-configuration.md) for every
flag and environment variable.

## GPU Utilization Cap

The app builds a `GpuGovernor` per run (`src/gpu_governor.py`). A background
thread samples the Intel iGPU busiest-engine utilization from Windows GPU-Engine
performance counters; a proportional controller grows a small per-frame delay
when utilization approaches the target (70%) and hard-kicks it over the cap
(80%), so sustained iGPU utilization stays under the KPI ceiling. On real
ultrasound workloads the pipeline is host-bound (~12–24% iGPU), so the throttle
never engages — the cap is a proven safety net, not a bottleneck. The governor
degrades to a no-op on non-Windows machines (the counter source is unavailable),
so the code still imports and runs for development.

## Project Layout

```text
ultrasound-ovarian-segmentation/
  src/                      # runtime app (pure OpenVINO)
    app.py                  #   capture / inference / display pipeline + CLI
    config.py               #   CLI + env configuration
    segmenters/             #   model arches: base, ds2net, sam, yolo, ov_runner
    sources.py              #   file / webcam / folder capture
    display.py              #   overlay + HUD pop-up
    gpu_governor.py         #   adaptive iGPU < 80% utilization cap
  backend/                  # model preparation (train / export / eval)
    bootstrap/train.py      #   DS2Net: MMOTU-2D fine-tune of SegFormer-B5
    bootstrap/export.py     #   DS2Net: checkpoint -> fp16 OpenVINO IR
    bootstrap/eval.py       #   DS2Net: Dice / IoU verification on the val split
    bootstrap/sam/          #   SAM: YOLO + encoder/decoder prep + eval
    requirements.txt        #   DS2Net model-prep deps (torch, transformers, ...)
    requirements-sam.txt    #   SAM model-prep deps (+ segment-anything, ultralytics)
  webapp/                   # web dashboard: Flask control plane + prebuilt React UI
  monitoring/               # hardware-utilization collectors + governor counter source
  models/                   # produced locally (git-ignored); see models/README.md
  docs/user-guide/          # setup, runtime configuration, troubleshooting
  setup.ps1                 # create venv + install runtime deps
  prepare_model.ps1         # -Arch {ds2net,sam}: train/export model(s) into models/
  run.ps1                   # -Arch {ds2net,sam}: launch the segmentation pop-up
  run_webapp.ps1            # launch the web dashboard (UI) on the same pipeline
  requirements.txt          # runtime deps (openvino, opencv, numpy, pywin32)
```

## License

Apache-2.0. See [third_party_programs_ultrasound-ovarian-segmentation.txt](third_party_programs_ultrasound-ovarian-segmentation.txt)
for third-party notices, and note the backbone/dataset licenses called out in
the Model Preparation guide.
