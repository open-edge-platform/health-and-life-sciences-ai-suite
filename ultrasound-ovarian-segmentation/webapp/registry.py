"""Use-case registry for the unified ultrasound AI app.

Each entry describes how to locate a model's OpenVINO IR, which devices and
precisions it supports, and the KPI targets used to colour the Pipeline
Performance table. The web UI enumerates these and lets the user pick a
device + precision before starting the pop-up pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # app/


@dataclass(frozen=True)
class UseCase:
    key: str
    label: str
    modality: str          # "cardiac" | "ovarian"
    task: str              # "segmentation" | "ejection_fraction" | "ovarian_sam"
    model_name: str
    ir_template: str       # e.g. "cache/ir/deeplabv3_{precision}/deeplabv3.xml"
    input_shape: tuple
    precisions: tuple
    devices: tuple
    default_video: str
    kpi_latency_ms: float
    kpi_fps: float
    enabled: bool = True
    note: str = ""
    decoder_ir_template: str = ""
    bbox_ir_template: str = ""

    def ir_path(self, precision: str) -> Path:
        return ROOT / self.ir_template.format(precision=precision)

    def decoder_ir_path(self, precision: str) -> Path:
        if not self.decoder_ir_template:
            return ROOT / "cache" / "ir" / "__missing_decoder__" / "decoder.xml"
        return ROOT / self.decoder_ir_template.format(precision=precision)

    def bbox_ir_path(self, precision: str) -> Path:
        if not self.bbox_ir_template:
            return ROOT / "cache" / "ir" / "__missing_bbox__" / "bbox.xml"
        return ROOT / self.bbox_ir_template.format(precision=precision)


USE_CASES: dict[str, UseCase] = {
    "ovarian_ds2net": UseCase(
        key="ovarian_ds2net",
        label="Ovarian - DS2Net Tumor Segmentation",
        modality="ovarian",
        task="segmentation",
        model_name="DS2Net promptless SegFormer-B5 (256, fine-tuned on MMOTU-2D)",
        ir_template="models/ds2net_segformer_b5_256/model.xml",
        input_shape=(1, 3, 256, 256),
        precisions=("fp16",),
        devices=("CPU", "GPU"),
        default_video="data/OTU_2d/images",
        kpi_latency_ms=30.0,
        kpi_fps=30.0,
        enabled=True,
        note="Promptless SegFormer-B5 segmentation. Default, validated path. "
             "iGPU and CPU supported; NPU is not supported for this model.",
    ),
    "ovarian_ultrasam": UseCase(
        key="ovarian_ultrasam",
        label="Ovarian \u2014 UltraSAM Tumor Segmentation",
        modality="ovarian",
        task="ovarian_sam",
        model_name="YOLO@320 bbox + SAM-256 (stock ViT-B fine-tuned on MMOTU) + multimask decoder",
        ir_template="models/sam_encoder_256_stockft/encoder.xml",
        input_shape=(1, 3, 256, 256),
        precisions=("fp16",),
        devices=("CPU", "GPU", "NPU"),
        default_video="data/mmotu2d/images",
        kpi_latency_ms=35.0,
        kpi_fps=30.0,
        enabled=True,
        decoder_ir_template="models/sam_decoder_256_multimask/decoder.xml",
        bbox_ir_template="models/yolo_mmotu_320/yolov8n_mmotu.xml",
        note="Freeze frame \u2192 YOLO@320 ROI \u2192 license-clean SAM-256 encoder "
             "(stock Meta SAM ViT-B, Apache-2.0, fine-tuned on MMOTU-2D) + multimask "
             "decoder tumour mask. CPU/iGPU produce the mask; NPU decoder is a known "
             "OV 2026.2 regression (empty mask) tracked as a separate defect.",
    ),
}
