# Get Started

This guide takes you from a fresh clone to a running segmentation pop-up on a
Windows machine with an Intel® Core™ Ultra (Intel® Arc™ iGPU).

> **Important:** Run every command in this guide in **Windows PowerShell 5.1**
> or **PowerShell 7** — not in Command Prompt (CMD). The setup and run scripts
> are PowerShell (`.ps1`) files; CMD does not execute them and opens them as
> text files instead.
>
> A regular (non-Administrator) PowerShell window is enough for cloning, setup,
> model preparation, the segmentation pop-up, and the web dashboard. The only
> exception is the **power** graph in the web dashboard: SoCWatch needs
> Administrator rights, so to see it, launch `.\run_webapp.ps1` from a
> PowerShell window opened with **Run as administrator**. Without it, all other
> graphs still work.

## Prerequisites

- Windows 11 with an Intel® Core™ Ultra processor (Intel® Arc™ iGPU).
- Up-to-date Intel graphics drivers. The iGPU must appear in Task Manager and be
  visible to the OpenVINO™ Toolkit (OpenVINO™).
- Python 3.11 (64-bit) on `PATH`.
- [Git for Windows](https://git-scm.com/download/win) on `PATH` — used to clone
  the repository.
- [Node.js](https://nodejs.org) 18+ on `PATH` — used once to build the web
  dashboard UI; the first `run_webapp.ps1` builds it automatically.
- Windows PowerShell 5.1 (included with Windows 11) or PowerShell 7. To open
  it, press **Start**, type `PowerShell`, and select **Windows PowerShell**. If
  running a script fails with a `PSSecurityException` / "running scripts is
  disabled on this system" error, allow scripts for the current session only
  (no Administrator rights needed):

  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

## Clone the repository

Clone the v2026.3.0 release of the suite. The `--sparse` option downloads only
the Ultrasound Ovarian Segmentation application, not the other applications in
the suite. To learn more about partial cloning, see the
[Repository Cloning guide](https://docs.openedgeplatform.intel.com/2026.3/OEP-articles/contribution-guide.html#repository-cloning-partial-cloning).

In PowerShell, go to the folder where you want the code and run:

```powershell
git clone --filter=blob:none --sparse --branch release-2026.3.0 https://github.com/open-edge-platform/health-and-life-sciences-ai-suite.git
cd health-and-life-sciences-ai-suite
git sparse-checkout set ultrasound-ovarian-segmentation
cd ultrasound-ovarian-segmentation
```

`release-2026.3.0` is the v2026.3.0 release branch. To pin an exact release
tag instead, replace it with the tag name (for example, `v2026.3.0`); to get
the latest development version, use `main`.

All commands in the following steps are run from this
`ultrasound-ovarian-segmentation` folder.

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
5. **Hardware-utilization tools** — CPU and memory monitoring need no extra installation. iGPU, NPU, and power graphs use Intel vendor tools (SoCWatch and Level Zero). Obtain these tools in one of the following ways:

   - **Existing benchmark installation:** copy the complete `monitoring\tools` folder from another machine that already has the HLS benchmark tools staged.
   - **Fresh setup:** obtain SoCWatch and Level Zero through your organization's Intel software/software-license channel or the approved Intel client-GPU installation package; do not download binaries from an untrusted source.
   - **Source build:** if your environment provides the Level Zero and performance-tools source trees, copy or build their outputs into `monitoring\tools` using the layout described in [Troubleshooting](./troubleshooting.md#web-dashboard-hw-monitoring-disabled).

   The application still runs and reports CPU and memory when these vendor tools are absent. iGPU, NPU, and power graphs remain disabled until the tools are staged and the app is restarted.

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
