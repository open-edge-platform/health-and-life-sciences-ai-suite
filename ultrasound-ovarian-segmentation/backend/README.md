# Model Preparation (backend)

The app ships the model-preparation recipe, not the weights. These modules
produce the OpenVINO IR(s) the runtime app loads.

## `ds2net` arch (default) — promptless SegFormer-B5

| Module | Purpose |
|---|---|
| `backend.bootstrap.train` | MMOTU-2D fine-tune of a promptless SegFormer-B5 |
| `backend.bootstrap.export` | trained checkpoint → fp16 OpenVINO IR |
| `backend.bootstrap.eval` | Dice / IoU verification on the val split |

## `sam` arch — YOLO bbox → SAM-256 pipeline (three IRs)

| Module | Purpose |
|---|---|
| `backend.bootstrap.sam.prepare_yolo_data` | MMOTU-2D masks → YOLO `lesion` boxes |
| `backend.bootstrap.sam.train_yolo` | YOLOv8n detector train (AGPL-3.0, build-time only) |
| `backend.bootstrap.sam.finetune_encoder` | SAM ViT-B encoder fine-tune @256 (`/255` preprocessing) |
| `backend.bootstrap.sam.export_yolo` | YOLO `.pt` → fp16 IR @320 |
| `backend.bootstrap.sam.export_encoder` | fine-tuned encoder → fp16 IR @256 |
| `backend.bootstrap.sam.export_decoder` | stock SAM prompt-enc + mask-dec → multimask IR |
| `backend.bootstrap.sam.eval` | end-to-end Dice / IoU verification on the val split |

Drive them with `..\prepare_model.ps1` (add `-Arch sam` for the SAM pipeline), or
run each with `-m` (e.g. `python -m backend.bootstrap.export --help`). Full
walkthrough:
[../docs/user-guide/get-started/model-preparation.md](../docs/user-guide/get-started/model-preparation.md).
