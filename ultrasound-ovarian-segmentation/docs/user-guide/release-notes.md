# Release Notes

## 1.0.0

Initial release of the Ultrasound Ovarian Segmentation sample app.

- Real-time promptless ovarian tumor segmentation on 2D ultrasound using a
  SegFormer-B5 (DS2Net) fp16 OpenVINO IR.
- Decoupled capture / inference / display pipeline with an OpenCV overlay pop-up
  and FPS / latency / GPU HUD.
- Inputs: video file, USB webcam, or a folder of stills. Device: Intel iGPU
  (with a CPU development / sanity fallback).
- Built-in GPU utilization governor enforcing the Intel iGPU < 80% busiest-engine
  KPI, with a per-run PASS/OVER verdict.
- Model-preparation backend (`backend/bootstrap`): MMOTU-2D fine-tune, fp16
  OpenVINO export, and Dice/IoU verification. The app ships the recipe, not the
  weights; reference accuracy is Dice 0.8657 / IoU 0.7913 at res 256 on the
  MMOTU-2D val split.
- Optional **`sam` model architecture** (`-Arch sam`): a prompted YOLOv8n bbox →
  SAM ViT-B encoder (fine-tuned @256) → stock SAM mask-decoder pipeline, selected
  at runtime alongside the default `ds2net`. Reference accuracy Dice 0.8207 /
  IoU 0.7394 on the MMOTU-2D val split. Its model-prep lives in
  `backend/bootstrap/sam`. The SAM encoder/decoder path is Apache-2.0; the YOLOv8
  detector (ultralytics) is AGPL-3.0 and used **build-time only** to produce the
  bbox IR — the runtime loads only the exported OpenVINO IR and does not link
  ultralytics.
