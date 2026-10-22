# Ultrasound Ovarian Segmentation Sample Application

<!--hide_directive
::::{container} component_header_row
<div class="component_card_widget">
  <a class="icon_github" href="https://github.com/open-edge-platform/health-and-life-sciences-ai-suite/tree/main/ultrasound-ovarian-segmentation">
     GitHub
  </a>
  <a class="icon_document" href="https://github.com/open-edge-platform/health-and-life-sciences-ai-suite/blob/main/ultrasound-ovarian-segmentation/README.md">
     Readme
  </a>
</div>
hide_directive-->

> [!NOTE]
> This application is for **reference and evaluation purposes only**. It is
  **not intended for direct use in clinical or diagnostic environments** and is not
  validated for such a purpose.

<!--hide_directive :::: hide_directive-->

The application demonstrates how the OpenVINO™ Toolkit (OpenVINO™) uses Intel hardware
acceleration (Intel iGPU, with CPU fallback) for AI-based real-time ovarian tumor segmentation
in 2D ultrasound. It offers two interchangeable models, selected with `-Arch`: a default
promptless SegFormer-B5 (DS2Net) and an optional prompted SAM-256 pipeline (YOLO bounding box
→ SAM encoder → mask decoder). An architecture that decouples capture, inference, and display
keeps the displayed video smooth regardless of inference speed, and a built-in GPU governor
enforces the KPI of keeping Intel iGPU utilization below 80%.

Supported inputs:

- Video file (for demonstrations and benchmarking)
- USB camera or webcam
- Folder with ultrasound still images (paced like a slideshow)

## Next Steps

- See [Model Preparation](./get-started/model-preparation.md) to produce the OpenVINO™
  Intermediate Representation (IR) that the application loads. The application does not
  include a trained model.
- See [Get Started](./get-started.md) for step-by-step setup and run instructions, including
  the **web dashboard (UI)**, which is the primary way to control the application.
- See [Runtime Configuration](./runtime-configuration.md) for all flags and environment
  variables that the application exposes.
- See [Troubleshooting](./troubleshooting.md) for common issues.
- See [Release Notes](./release-notes.md) for version history.

<!--hide_directive
:::{toctree}
:hidden:

Get Started <get-started.md>
Runtime Configuration <./runtime-configuration.md>
Troubleshooting <./troubleshooting.md>
Release Notes <./release-notes.md>

:::
hide_directive-->
