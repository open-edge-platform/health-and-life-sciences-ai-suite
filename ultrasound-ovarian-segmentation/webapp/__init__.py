"""Unified ultrasound AI web app (native Windows/Linux, no Docker).

A small Flask control plane + static dashboard that mirrors the Surgical_Instrument
suite UX: left config panel (use-case / device / precision / Start-Stop), right
Pipeline Performance + Model & Input + Platform panels, live metrics over SSE.

The inference video itself is rendered in a separate OpenCV pop-up window
(Original | Segmentation, side by side) rather than streamed into the browser,
which keeps end-to-end latency representative of the pipeline.
"""
