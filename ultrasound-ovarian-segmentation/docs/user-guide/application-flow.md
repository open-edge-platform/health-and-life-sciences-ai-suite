<!--hide_directive
```{eval-rst}
.. meta::
   :description: Describes the end-to-end flow of the Ultrasound Ovarian Segmentation sample application, from environment preparation to shutdown.
   :keywords: ultrasound, ovarian segmentation, application flow, DS2Net, SAM, PowerShell, web dashboard
```
hide_directive-->

# Ultrasound Ovarian Segmentation Application Flow

This guide describes the end-to-end flow for the Ultrasound Ovarian Segmentation sample application.

## 1. Prepare the Environment

1. Clone the repository and enter `ultrasound-ovarian-segmentation`.
2. Open Windows PowerShell in that directory.
3. Run `.\setup.ps1` to create the virtual environment and install runtime dependencies.
4. Obtain or produce the trained model IR as described in [Model Preparation](./get-started/model-preparation.md).

## 2. Select the Model and Input

- The default `ds2net` flow is promptless SegFormer-B5 segmentation.
- The optional `sam` flow uses a YOLOv8n prompt, Segment Anything Model (SAM) encoder, and SAM decoder.
- Supported inputs are a video file, webcam, or folder of still ultrasound frames.

## 3. Start the Application

Run the command-line sample:

```powershell
.\run.ps1 -Source file -Input C:\path\to\clip.mp4 -Device GPU
```

Alternatively, launch the web dashboard:

```powershell
.\run_webapp.ps1
```

At the browser dashboard, select the device, precision, and input source, then click **Start**.
For the browser, the use case is fixed at launch (`.\run_webapp.ps1 -UseCase ds2net`, default,
or `-UseCase sam`) and shown as a read-only badge.

## 4. Processing Flow

1. Capture reads frames from the selected video, webcam, or image folder.
2. Preprocessing converts frames to the model's required size and tensor format.
3. The selected segmenter runs on CPU or integrated GPU (iGPU):
   - DS2Net produces a whole-frame tumor mask.
   - SAM-256 produces a prompted bounding box and instance mask.
4. The display overlays the mask and/or prompt box on the original frame.
5. The UI reports FPS, inference latency, GPU utilization, and the GPU-utilization KPI.
6. File and folder sources loop by default. The run ends when you stop it (press q or Esc, or
   click **Stop**), or at end-of-file if `--no-loop` (or `LOOP=0`) is set. Capture, inference,
   display, and monitoring then shut down cleanly. See [Runtime Configuration](./runtime-configuration.md#application-flags).

## 5. Stop the Application

- In the OpenCV window, press `q` or `Esc`, or close the window.
- In the web dashboard, click **Stop**.
- Close the browser and terminal process if no application window remains open.

For cleanup of the virtual environment and downloaded or generated assets, see [Uninstall](./uninstall.md).
