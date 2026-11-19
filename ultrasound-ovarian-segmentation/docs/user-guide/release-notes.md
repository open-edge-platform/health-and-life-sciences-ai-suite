# Release Notes: Ultrasound Ovarian Segmentation

## Version 2026.3.0

**Release Date**: TBD

### Overview

This is the initial release of the Ultrasound Ovarian Segmentation sample application.

### New

- Real-time, promptless ovarian-tumor segmentation on 2D ultrasound using a SegFormer-B5 (DS2Net) model in FP16 OpenVINO™ Intermediate Representation (IR) format.
- Decoupled capture, inference, and display pipeline with an OpenCV overlay window and a heads-up display (HUD) showing FPS, latency, and GPU metrics.
- Supported inputs: video file, USB webcam, or folder of still images. Supported device: Intel® integrated GPU (iGPU), with CPU fallback for development and sanity checks.
- GPU utilization governor enforcing less than 80% utilization on the busiest iGPU engine, with PASS or OVER results for each run.
- Model-preparation backend (`backend/bootstrap`) for fine-tuning on MMOTU-2D, FP16 OpenVINO IR export, and Dice/IoU verification. The application ships the recipe, not model weights. Reference accuracy is Dice 0.8657 and IoU 0.7913 at resolution 256 on the MMOTU-2D validation split.
- Optional `sam` architecture (`-Arch sam`) using a prompted pipeline with a YOLOv8n bounding-box detector, fine-tuned SAM ViT-B encoder, and stock SAM mask decoder. Reference accuracy is Dice 0.8207 and IoU 0.7394 at resolution 256 on the MMOTU-2D validation split.
