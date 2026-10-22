# Runtime Configuration

Every option is available as a CLI flag and as an environment variable (with the CLI
flag taking precedence). Launch via `run.ps1` in a typical situation, or call the application
module directly for full control:

```powershell
venv\Scripts\python.exe -m src.app --source file --input clip.mp4 --device GPU --gpu-cap 80
```

## run.ps1 parameters

| Parameter | Values | Default | Description |
| --- | --- | --- | --- |
| `-Source` | `file` / `webcam` / `folder` | `file` | Input type |
| `-Input` | path / index / folder | — | Video path, webcam index, or stills folder |
| `-Arch` | `ds2net` / `sam` | `ds2net` | Model architecture to run |
| `-Device` | `GPU` / `CPU` / `AUTO` | `GPU` | OpenVINO™ inference device; NPU is not supported |
| `-Model` | path | application default | Override the DS2Net Intermediate Representation (IR) `model.xml` |
| `-Venv` | path | `venv` | Virtual environment folder |
| `-Extra` | — | — | Flags after `--` are forwarded to the application |

Example of forwarding extra flags:

```powershell
.\run.ps1 -Source file -Input clip.mp4 -- --record out.mp4 --gpu-cap 70 --alpha 0.5
```

## Application flags

### Source

| Flag | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `--source` | `SOURCE` | `file` | `file` / `webcam` / `folder` |
| `--input` | `INPUT` | — | Video path, webcam index, or stills folder |
| `--no-loop` | `LOOP=0` | loop on | Stop at end instead of looping file/folder sources |
| `--width` | `WIDTH` | `1280` | Requested capture width |
| `--height` | `HEIGHT` | `720` | Requested capture height |
| `--target-fps` | `TARGET_FPS` | `30` | Pacing for folder/webcam sources |

### Model / inference

| Flag | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `--model-arch` | `MODEL_ARCH` | `ds2net` | `ds2net` (promptless SegFormer) / `sam` (YOLO bounding box → SAM-256) |
| `--device` | `DEVICE` | `GPU` | `GPU` (recommended) / `CPU` / `AUTO` |
| `--model` | `MODEL` | `models/ds2net_segformer_b5_256/model.xml` | DS2Net OpenVINO™ IR path |
| `--res` | `RES` | `256` | DS2Net network input resolution (must match the exported IR) |
| `--sam-encoder` | `SAM_ENCODER` | `models/sam_encoder_256_stockft/encoder.xml` | SAM encoder IR (`sam` architecture) |
| `--sam-decoder` | `SAM_DECODER` | `models/sam_decoder_256_multimask/decoder.xml` | SAM multimask-decoder IR (`sam` architecture) |
| `--sam-yolo` | `SAM_YOLO` | `models/yolo_mmotu_320/yolov8n_mmotu.xml` | YOLO bounding box IR (`sam` architecture) |
| `--frame-skip` | `FRAME_SKIP` | `1` | Run inference every Nth captured frame |

> [!NOTE]
>
> `--res` must equal the resolution the IR was exported at. The default DS2Net IR
> is exported at 256; if you export at another resolution, pass the matching
> `--res`. The `sam` architecture is fixed at 256 and uses the three `--sam-*` IR paths
> instead of `--model`.
>
> ---
>
> **Device:** both models target the Intel integrated GPU (iGPU) through the `GPU`
> device. `CPU` is a development or sanity fallback. The **NPU is not supported**; passing `NPU` logs a notice and
> falls back to GPU or CPU.

### GPU utilization cap

| Flag | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `--gpu-cap` | `GPU_CAP` | `80` | Hard iGPU busiest-engine utilization ceiling (KPI) |
| `--no-governor` | `NO_GOVERNOR=1` | off | Disable the governor (measure uncapped utilization) |

The governor targets a value of around 70% and hard-kicks over the cap, clamping a per-frame
delay in `[0, 1.0] s`. It reads Windows GPU-Engine counters via pywin32; on non-Windows
machines it becomes inactive so the application still runs for development.

### Display / output

| Flag | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `--view` | `VIEW` | `side_by_side` | Layout: `side_by_side` (original + segmented) or `overlay` (mask drawn on the frame) |
| `--alpha` | `ALPHA` | `0.45` | Mask overlay opacity (0–1) |
| `--mask-color` | `MASK_COLOR` | `0,0,255` | Overlay color as `B,G,R` (default red) |
| `--display-scale` | `DISPLAY_SCALE` | `1.0` | Scale the output window |
| `--window` | `WINDOW` | `Ovarian Tumor Segmentation` | Window title |
| `--headless` | `HEADLESS=1` | off | No window (useful for recording / soak tests) |
| `--record` | `RECORD` | — | Write an annotated `.mp4` of the output (captures the current view) |
| `--log-file` | `LOG_FILE` | — | Path to a log file for recording runtime messages (including KPI verdicts) |

## Examples

Measure uncapped iGPU utilization (governor off), headless, recording:

```powershell
.\run.ps1 -Source folder -Input C:\frames -Device GPU -- --no-governor --headless --record soak.mp4
```

Tighten the GPU cap and use a green mask:

```powershell
.\run.ps1 -Source file -Input clip.mp4 -Device GPU -- --gpu-cap 70 --mask-color 0,255,0
```

Use the single-pane overlay instead of the default side-by-side view:

```powershell
.\run.ps1 -Source file -Input clip.mp4 -Device GPU -- --view overlay
```
