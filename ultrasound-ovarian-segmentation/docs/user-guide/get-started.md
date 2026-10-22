# Get Started

This guide takes you from a fresh clone to a running segmentation pop-up on a
Windows machine with an Intel® Core™ Ultra (Intel® Arc™ iGPU).

## Prerequisites

- Windows 11 with an Intel® Core™ Ultra processor (Intel® Arc™ iGPU).
- Up-to-date Intel graphics drivers. The iGPU must appear in Task Manager and be
  visible to the OpenVINO™ Toolkit (OpenVINO™).
- Python 3.11 (64-bit) on `PATH`.
- [Node.js](https://nodejs.org) 18+ on `PATH` — used once to build the web
  dashboard UI; the first `run_webapp.ps1` builds it automatically.
- PowerShell. If scripts are blocked, allow them for the current session only:

  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

## 1. Install runtime dependencies

From the `ultrasound-ovarian-segmentation` folder:

```powershell
.\setup.ps1
```

This creates a local `venv` and installs the runtime dependencies
(`requirements.txt`): OpenVINO™, OpenCV, NumPy, and pywin32 (for the GPU-Engine
counters the governor reads).

## 2. Prepare the model

The application does not include a trained model. It loads its default Intermediate
Representation (IR) from `models\ds2net_segformer_b5_256\model.xml`. Pick the case that
matches your situation:

- **You were handed a prebuilt IR** (for example, copied from another evaluation
  machine). Place the two files — `model.xml` and `model.bin` — under
  `models\ds2net_segformer_b5_256\` and jump to
  [step 3](#3-run-the-segmentation-pop-up). No dataset, training, or model-preparation
  dependencies are required.

- **You have a trained checkpoint** at `models\segformer_b5_mmotu`. Export it to
  the IR and verify Dice:

  ```powershell
  .\prepare_model.ps1 -Verify
  ```

- **You are starting from nothing.** Training needs the MMOTU-2D dataset and the
  MiT-B5 backbone downloaded first (see the [Model Preparation](./get-started/model-preparation.md)
  walkthrough), then:

  ```powershell
  .\prepare_model.ps1 -Train -Verify   # train on the Intel iGPU, then export + verify
  ```

For the optional prompted SAM-256 architecture, add `-Arch sam` (it needs the stock SAM
ViT-B weights downloaded first). The full walkthrough — dataset layout, backbone,
training, export, expected Dice, and the SAM recipe — is in
[Model Preparation](./get-started/model-preparation.md).

## 3. Run the segmentation pop-up

```powershell
.\run.ps1 -Source file -Input C:\path\to\clip.mp4 -Device GPU
```

A window opens showing the **original frame next to the segmented frame**
(tumor-mask overlay), plus a HUD showing inference FPS/latency and live iGPU
utilization. Press `q` or `Esc`
to quit, or close the window.

For a single overlaid pane, pass `--view overlay` to the command:

```powershell
.\run.ps1 -Source file -Input C:\path\to\clip.mp4 -Device GPU -- --view overlay
```

Other sources:

```powershell
.\run.ps1 -Source folder -Input C:\path\to\frames -Device GPU
.\run.ps1 -Source webcam -Input 0 -Device GPU
```

Run the prompted SAM-256 architecture (after preparing its IRs):

```powershell
.\run.ps1 -Arch sam -Source file -Input C:\path\to\clip.mp4 -Device GPU
```

## 4. Run the web dashboard (UI)

The web dashboard is the primary way to use the application: on top of the same
`src/` pipeline the CLI uses, it adds device/use-case selection in the browser
plus live hardware utilization (CPU / iGPU / NPU / power / memory) and pipeline
KPIs.

### Prepare the dashboard (one-time)

Have all of this ready **before** you launch, so the first **Start** is seamless
and every graph lights up:

1. **Model** — the dashboard loads the same IR you built in
   [step 2](#2-prepare-the-model) (`models\ds2net_segformer_b5_256\model.xml`).
2. **An input** — a clip (`.mp4`) or a folder of frames ready on disk, or the
   `data\OTU_2d\images` dataset you prepared in [step 2](#2-prepare-the-model).
3. **Python packages** — the UI needs a few beyond the CLI (Flask and psutil, plus
   pandas for the power graph). Add them to the `venv`:

   ```powershell
   .\venv\Scripts\pip install -r webapp\requirements.txt
   ```

4. **Node.js on `PATH`** — the browser UI is built from source **automatically**
   on your first `run_webapp.ps1`, so there is nothing to build by hand; just
   install [Node.js](https://nodejs.org) 18+. (To pre-build instead, run
   `cd webapp\ui; npm install; npm run build`.)
5. **Hardware-utilization tools** — CPU and memory need nothing extra, but the
   iGPU / NPU / power graphs use Intel vendor tools (SoCWatch, Level Zero) that
   are **not** committed (as they total ~139 MB). Stage them once by copying a
   `monitoring\tools` folder from your existing benchmark install into this
   clone. Without them the application still runs and shows CPU and memory.

### Launch and run

Launch:

```powershell
.\run_webapp.ps1
```

On the **first** launch this builds the browser UI from source (needs Node.js;
one-time, which takes about a minute to complete), then serves it at <http://127.0.0.1:8080> and opens
your browser; later launches skip the build. Next:

1. Pick a **Use case** (default *DS2Net Tumor Segmentation*), a **Device**
   (Intel iGPU or CPU), and **Precision** (FP16).
2. In **Video / image source**, enter the path to a clip (`.mp4`) or a folder of
   frames — or leave it blank to use the `data\OTU_2d\images` dataset you
   prepared during model preparation.
3. Click **Start** in the top bar.

The segmentation video opens in a separate OpenCV pop-up (*Original |
Segmentation*); press `Esc` in that window, or **Stop** in the UI, to end the
run. The browser shows live inference latency / FPS, the iGPU KPI, and the
hardware-utilization graphs. Use `-Port 8090` to change the port or `-NoBrowser`
to skip launching the browser.

## 5. Confirm the GPU KPI

On exit the application prints the governor verdict, for example:

```text
GPU KPI (< 80%): PASS — peak busiest-engine 17.8% mean 11.5% (samples=900, throttle_engaged=False)
```

`PASS` means sustained Intel iGPU busiest-engine utilization stayed under the 80%
cap for the whole run. On typical ultrasound workloads the pipeline is host-bound
(utilizing ~12–24% iGPU), so `throttle_engaged=False` is expected — the governor is a proven
safety net rather than a bottleneck. See
[Runtime Configuration](./runtime-configuration.md) to change the cap or disable
the governor to measure uncapped utilization.

## Next steps

- [Runtime Configuration](./runtime-configuration.md) — all flags and environment variables.
- [Troubleshooting](./troubleshooting.md) — device not found, no pop-up, low Dice.

<!--hide_directive
:::{toctree}
:hidden:

./get-started/model-preparation.md

:::
hide_directive-->
