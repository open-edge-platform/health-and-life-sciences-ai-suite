# Model Preparation

This app does **not** ship a trained model. You produce the OpenVINO IR locally
and the app loads it from `models/ds2net_segformer_b5_256/model.xml` (its
default). This page covers the full path: dataset, backbone, training, export,
and verification.

> **Why not ship the IR?** Reproducibility (you can retrain on your own data),
> artifact size (the IR exceeds common git limits), and licensing — the training
> backbone and dataset carry their own licenses (see the end of this page). The
> app ships the recipe, not the weights.

> **Two model architectures.** The app ships two interchangeable segmentation
> models, selected with `-Arch`:
>
> - **`ds2net`** (default) — a promptless SegFormer-B5. This is the recipe
>   immediately below.
> - **`sam`** — a prompted SAM-256 pipeline (YOLO bbox → SAM encoder → mask
>   decoder). See [SAM-256 arch](#sam-256-arch) for its separate recipe.

## What you produce

```
models/
  segformer_b5_mmotu/            # trained Hugging Face checkpoint (train.py)
  ds2net_segformer_b5_256/
    model.xml                    # the IR the app loads by default
    model.bin
    .converted_ok
```

Reference accuracy at res 256: **Dice mean 0.8657 / median 0.922, IoU mean
0.7913** on the 469-image MMOTU-2D validation split.

> **Already have the exported IR?** If someone handed you `model.xml` +
> `model.bin` (for example, copied from another box), you can skip this entire
> page: place the two files under `models\ds2net_segformer_b5_256\` and run the
> app. The dataset, training, and model-prep dependencies are only needed to
> *produce* the IR, not to run it.

## Model background

"DS2Net" (github.com/cv516Buaa/MMOTU_DS2Net, arXiv 2207.06799, Apache-2.0) is a
SegFormer MiT-B5 semantic-segmentation model for ovarian ultrasound. Its
dual-branch head only matters for 2D↔CEUS domain adaptation; for a single 2D
ultrasound stream we train the faithful single-modality stand-in — a SegFormer-B5
that takes the whole frame and outputs a 2-class (background / tumor) mask. It is
**promptless**: no bounding box, no separate detector.

## Option A — export an existing checkpoint

If you already have a trained checkpoint at `models/segformer_b5_mmotu`, you only
need the export + verify step:

```powershell
.\prepare_model.ps1 -Verify
```

This installs the model-prep dependencies, exports the checkpoint to a fp16
OpenVINO IR at `models/ds2net_segformer_b5_256`, and scores Dice/IoU on the val
split. Skip to [Verify the result](#verify-the-result).

## Option B — train from scratch

### 1. Dataset

Download the **MMOTU-2D (OTU_2d)** dataset from
[MMOTU_DS2Net](https://github.com/cv516Buaa/MMOTU_DS2Net) (Apache-2.0) and arrange
it like this under the app folder:

```
data/OTU_2d/
  images/        # {stem}.JPG ultrasound frames
  annotations/   # {stem}_binary.PNG binary tumor masks ({0, 255})
  train.txt      # one image stem per line (training split)
  val.txt        # one image stem per line (validation split, 469 stems)
```

`train.py` / `eval.py` resolve images and masks by stem, tolerating common case
and extension variants.

### 2. Backbone

Training initializes from the **nvidia/mit-b5** ImageNet backbone. On a machine
with internet access, download the Hugging Face snapshot and copy it to the
target box at `models/mit_b5_hf`:

```python
# on an internet-connected machine
from huggingface_hub import snapshot_download
snapshot_download("nvidia/mit-b5", local_dir="mit_b5_hf")
```

> **License note:** `nvidia/mit-b5` carries NVIDIA's SegFormer license
> (research / non-commercial). It is used only as the training initialization and
> is not redistributed by this app. For commercial or non-research use, review
> that license and consider a permissively licensed backbone or training from
> scratch.

### 3. Install training dependencies

Training on the Intel Arc iGPU needs the Intel XPU build of PyTorch. Inside the
app `venv`:

```powershell
venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/xpu
venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

(`prepare_model.ps1` installs `backend\requirements.txt` for you; install the XPU
torch wheel first if you want iGPU training. CPU training also works, slower.)

### 4. Train

```powershell
.\prepare_model.ps1 -Train -Verify
```

or run the trainer directly for more control:

```powershell
venv\Scripts\python.exe -m backend.bootstrap.train ^
    --data data\OTU_2d --backbone-dir models\mit_b5_hf ^
    --res 256 --epochs 40 --batch 4 --lr 6e-5 --device xpu ^
    --out models\segformer_b5_mmotu
```

The val Dice printed during training is a torch, resolution-space convergence
check only — the authoritative number comes from the OpenVINO eval below.

Smoke-test the pipeline end to end in a few steps first:

```powershell
venv\Scripts\python.exe -m backend.bootstrap.train --data data\OTU_2d ^
    --backbone-dir models\mit_b5_hf --device xpu --smoke-steps 8 --out models\_seg_smoke
```

## Export to OpenVINO IR

```powershell
venv\Scripts\python.exe -m backend.bootstrap.export ^
    --from-pretrained models\segformer_b5_mmotu --res 256
```

This writes `models\ds2net_segformer_b5_256\model.xml` (static `[1,3,256,256]`,
fp16) — the app's default.

> **Always pass `--from-pretrained`.** SegFormer-B5's latency is
> architecture-bound, so the exporter *can* build an IR from random weights, but
> that IR segments nothing (Dice ~0). Only an IR exported from a trained
> checkpoint reproduces the reference Dice.

## Verify the result

```powershell
venv\Scripts\python.exe -m backend.bootstrap.eval ^
    --ir models\ds2net_segformer_b5_256\model.xml --res 256 ^
    --data data\OTU_2d --device GPU
```

Expected on the full val split:

```text
[eval] Dice mean=0.8657 ... 
[eval] IoU  mean=0.7913 ...
```

A mean Dice near 0 means the IR was exported from random weights (missing
`--from-pretrained`) or the dataset paths are wrong. See
[Troubleshooting](../troubleshooting.md).

## Run the app

```powershell
.\run.ps1 -Source file -Input C:\path\to\clip.mp4 -Device GPU
```

## SAM-256 arch

The SAM-256 arch (`-Arch sam`) is a three-model, *prompted* pipeline:

```text
frame → YOLOv8n@320 bbox → SAM ViT-B encoder@256 (fine-tuned) →
  stock SAM prompt-encoder + mask-decoder (frozen, multimask) →
  argmax-by-IoU → tumor mask
```

It fine-tunes only the SAM image encoder on MMOTU-2D; the prompt encoder and mask
decoder are stock Meta SAM ViT-B. Reference accuracy at res 256: **end-to-end
Dice mean 0.8207 / IoU 0.7394** on the 469-image val split (0.9215 Dice with a
ground-truth box — the gap is YOLO localization error).

### What you produce

```
models/
  yolov8n_mmotu.pt                       # trained YOLO checkpoint (train_yolo.py)
  sam256_stockft.pth                     # fine-tuned SAM encoder weights (finetune_encoder.py)
  yolo_mmotu_320/yolov8n_mmotu.xml       # bbox IR (export_yolo.py)
  sam_encoder_256_stockft/encoder.xml    # encoder IR (export_encoder.py)
  sam_decoder_256_multimask/decoder.xml  # decoder IR (export_decoder.py)
```

The app loads the three `.xml` IRs (see `src/config.py` defaults).

### 1. Dataset

Same **MMOTU-2D (OTU_2d)** dataset and layout as the DS2Net arch above — download
it once. The YOLO stage derives one `lesion` box per image from the mask contour.

### 2. Download the stock SAM weights

Download Meta's stock SAM ViT-B checkpoint (**Apache-2.0**) to `models/`:

```powershell
curl -L -o models\sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

This one checkpoint initializes the encoder fine-tune and provides the frozen
prompt-encoder + mask-decoder for the decoder export.

### 3. Install dependencies

```powershell
venv\Scripts\python.exe -m pip install -r backend\requirements-sam.txt
```

> **License note:** `ultralytics` (YOLOv8) is **AGPL-3.0** and is used
> **build-time only** to train and export the detector. The runtime app loads the
> exported OpenVINO IR with OpenVINO/OpenCV and never imports ultralytics, so the
> shipped app is not a derivative work of it. Substitute any box detector that
> emits an `[x, y, w, h]` lesion box if AGPL is unacceptable for your deployment.
> `segment-anything` is Apache-2.0.

### Option A — export from existing weights

If you already have `models/sam256_stockft.pth` and `models/yolov8n_mmotu.pt`
(e.g. downloaded), export the three IRs and verify:

```powershell
.\prepare_model.ps1 -Arch sam -Verify
```

### Option B — train, then export

```powershell
.\prepare_model.ps1 -Arch sam -Train -Verify
```

or run each stage directly for more control:

```powershell
venv\Scripts\python.exe -m backend.bootstrap.sam.prepare_yolo_data --data data\OTU_2d
venv\Scripts\python.exe -m backend.bootstrap.sam.train_yolo --device cpu
venv\Scripts\python.exe -m backend.bootstrap.sam.finetune_encoder ^
    --weights models\sam_vit_b_01ec64.pth --data data\OTU_2d --device xpu
venv\Scripts\python.exe -m backend.bootstrap.sam.export_yolo
venv\Scripts\python.exe -m backend.bootstrap.sam.export_encoder --weights models\sam256_stockft.pth
venv\Scripts\python.exe -m backend.bootstrap.sam.export_decoder --weights models\sam_vit_b_01ec64.pth
```

The encoder fine-tune uses `/255.0` image scaling (no mean/std) — identical to the
runtime encoder preprocessing — so the trained encoder matches the inference input
distribution. This is what reproduces the reference Dice.

### Verify

```powershell
venv\Scripts\python.exe -m backend.bootstrap.sam.eval --device GPU --data data\OTU_2d
```

Expected on the full val split:

```text
[eval] Dice mean=0.8207 ...
[eval] IoU  mean=0.7394 ...
[eval] KPI: mean Dice >= 0.80 -> PASS
```

### Run the SAM app

```powershell
.\run.ps1 -Arch sam -Source file -Input C:\path\to\clip.mp4 -Device GPU
```

## Licenses summary

| Component | License | Notes |
|---|---|---|
| This app | Apache-2.0 | code only |
| DS2Net reference | Apache-2.0 | model / method (`ds2net` arch) |
| SAM (segment-anything) | Apache-2.0 | encoder + prompt-encoder + mask-decoder (`sam` arch) |
| YOLOv8 (ultralytics) | AGPL-3.0 | **build-time only** detector (`sam` arch); not imported at runtime |
| MMOTU-2D dataset | Apache-2.0 | downloaded locally, not redistributed |
| nvidia/mit-b5 backbone | NVIDIA SegFormer (research/non-commercial) | `ds2net` training init only, not redistributed |
| PyTorch | BSD-3-Clause | model-prep only |
| Transformers | Apache-2.0 | `ds2net` model-prep only |
| OpenVINO / OpenCV | Apache-2.0 | runtime |

See [third_party_programs_ultrasound-ovarian-segmentation.txt](../../../third_party_programs_ultrasound-ovarian-segmentation.txt)
for the full notices.
