# Release Notes: Surgical Instrument

## Version 2026.3.0-rc1

**Release Date**: TBD

### New

- Added Intel Panther Lake (PTL) to the supported hardware.

### Improved

- Image tags updated to the `2026.3.0-rc1` release, including the DL Streamer Pipeline Server base image.
- `make up` in registry mode now fails clearly when an image pull fails instead of silently falling back to a local build.
- Documentation updates for the new repository location, dataset sizes, and repository layout.

### Fixed

- Fixed container startup failure (`ModuleNotFoundError: openvino`) on the `2026.3.0-rc1` base image.
- Fixed model-training errors on Intel GPU (XPU).
- Fixed host-setup failures when installing base tools.

## Version 2026.2.0

**Release Date**: September 10, 2026

This is the initial release of the application. It is intended for reference
and evaluation purposes only and not for direct use in clinical or diagnostic
environments.

**New**:

- Docker Compose driven workflow with a single `make up` entrypoint that
  supports both registry pull (default) and local source build via
  `REGISTRY=false`.
- Decoupled capture / inference / display architecture: display shows every
  captured frame while OpenVINO inference runs on its own thread.
- Pure OpenVINO detector (no `ultralytics` runtime coupling) with YOLO
  letterbox preprocessing and numpy NMS.
- Multiple sources through a single interface: Basler camera (via `pypylon`),
  USB / V4L2 webcam, and video file.
- OpenGL vsync presenter with `cv2` and headless fallback.
- Optional Basler capture trigger modes (`off` / `software` / `vsync`) with
  present-completion phase-lock for low photon-to-pixel latency.
- Fullscreen OpenGL direct-scanout to bypass the compositor.
- Per-stage latency CSV (`trigger_to_grab`, `grab_to_display`,
  `trigger_to_display`, `infer_ms`, `disp_fps`, `cap_fps`) for A/B comparison.
- Optional core pinning and `SCHED_FIFO` real-time priority for capture,
  inference, and display threads.
- Restructured layout with `src/`, `docker/`, and `docs/` folders and a
  compose-only Makefile.
- Optional local model-preparation flow: `make setup-prerequisites` installs the host
  prerequisites (Docker + Intel client GPU stack), then
  `make backend-venv && make backend-bootstrap` fetches the dataset, trains
  YOLO11n on the Intel iGPU, and exports a FP16 OpenVINO IR. See
  [Model Preparation](./get-started/model-preparation.md).
- REAL-Colon dataset support: bootstrap auto-detects the REAL-Colon
  `*_frames/` + `*_annotations/` layout and converts Pascal VOC XML bounding
  boxes to YOLO labels. Legacy mask-based drops (ColonDB-style images+masks)
  remain supported as a fallback.
- `make check-l0` and `make doctor` preflight targets for host prerequisite
  verification.
