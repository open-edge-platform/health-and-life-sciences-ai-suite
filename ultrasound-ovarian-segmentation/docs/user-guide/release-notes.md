# Release Notes: Ultrasound Ovarian Segmentation Sample Application

## Version 2026.3.0

**Release Date**: *TBD*

This is the initial release of the Ultrasound Ovarian Segmentation sample application. It
includes the following features:

- Real-time, promptless ovarian tumor segmentation on 2D ultrasound using a
  SegFormer-B5 (DS2Net) model in the FP16 OpenVINO™ Toolkit (OpenVINO™) Intermediate
  Representation (IR) format.
- Decoupled capture, inference, and display pipeline with an OpenCV overlay pop-up window
  and a heads-up display (HUD) showing FPS, latency, and GPU metrics.
- Supported inputs: a video file, a USB webcam, or a folder of still images. Supported device:
  Intel® integrated GPU (iGPU), with a CPU fallback for development and sanity checks.
- Built-in GPU utilization governor that enforces the KPI of less than 80% utilization on the
  busiest iGPU engine, with a per-run PASS or OVER verdict.
- Model preparation backend (`backend/bootstrap`) that fine-tunes the model on the MMOTU-2D
  dataset, exports it to OpenVINO™ IR in FP16 precision, and verifies Dice and IoU scores. The
  application ships the recipe, not the model weights. Reference accuracy is Dice 0.8657 and
  IoU 0.7913 at resolution 256 on the MMOTU-2D validation split.
- Optional **`sam` model architecture** (`-Arch sam`), selectable at runtime as an alternative
  to the default `ds2net` architecture. It uses a prompted pipeline that chains a YOLOv8n
  bounding-box detector, a Segment Anything Model (SAM) ViT-B encoder fine-tuned at resolution
  256, and the stock SAM mask decoder. Reference accuracy is Dice 0.8207 and IoU 0.7394 on the
  MMOTU-2D validation split. Its model preparation code is in `backend/bootstrap/sam`. The SAM
  encoder and decoder path is licensed under Apache-2.0. The YOLOv8 detector (Ultralytics) is
  licensed under AGPL-3.0 and is used **at build time only** to produce the bounding-box IR.
  At runtime, the application loads only the exported OpenVINO™ IR and does not import `ultralytics`.
