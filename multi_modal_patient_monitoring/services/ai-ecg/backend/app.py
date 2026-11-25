from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import shutil
import time
from inference.embedding_engine import HubertECGInferenceEngine, resolve_data_path

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="ECG OpenVINO API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

hubert_engine = HubertECGInferenceEngine()

# ---- STREAM STATE (VERY IMPORTANT) ----
stream_files = []
stream_index = 0


@app.get("/health")
def health():
    return {"status": "ok"}


def _data_path(filename: str) -> str:
    try:
        return resolve_data_path(filename)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")


@app.post("/upload")
async def upload_ecg(file: UploadFile = File(...)):
    if not file.filename.endswith(".mat"):
        raise HTTPException(status_code=400, detail="Only .mat files are allowed")

    file_path = _data_path(file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()

    return {"status": "uploaded", "filename": os.path.basename(file_path)}


@app.get("/predict/{filename}")
def predict_ecg(filename: str):
    file_path = _data_path(filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # HuBERT-ECG is now the primary inference path.
        h_pred = hubert_engine.predict(filename)
        return {
            "file": filename,
            "signal": h_pred["signal"],
            "result": h_pred["classification"],
            "inference_ms": h_pred["inference_ms"],
            "length": h_pred["length"],
            "hubert_embedding": h_pred["embedding"],
            "hubert_inference_ms": h_pred["inference_ms"],
        }
    except Exception:
        logger.exception("Inference failed for %s", filename)
        raise HTTPException(status_code=500, detail="Inference failed")


# -------- NEW TRUE STREAMING ENDPOINT --------
@app.get("/predict_stream_next")
def predict_stream_next():
    """
    Returns ONE file inference per call.
    Cycles through files endlessly.
    """
    global stream_files, stream_index

    if not stream_files:
        stream_files = sorted(
            [f for f in os.listdir(DATA_DIR) if f.endswith(".mat")]
        )
        stream_index = 0

    if not stream_files:
        raise HTTPException(status_code=404, detail="No .mat files found")

    filename = stream_files[stream_index]
    stream_index = (stream_index + 1) % len(stream_files)

    try:
        h_pred = hubert_engine.predict(filename)
        return {
            "file": filename,
            "signal": h_pred["signal"],
            "result": h_pred["classification"],
            "inference_ms": h_pred["inference_ms"],
            "length": h_pred["length"],
            "hubert_embedding": h_pred["embedding"],
            "hubert_inference_ms": h_pred["inference_ms"],
        }
    except Exception:
        logger.exception("Stream inference failed for %s", filename)
        raise HTTPException(status_code=500, detail="Inference failed")


@app.get("/predict_hubert/{filename}")
def predict_hubert(filename: str):
    """Run inference with the HuBERT-ECG backbone and return an embedding.

    This does not perform AF/arrhythmia classification; it exposes the
    pooled feature vector, which can be used by downstream components
    or for experimentation.
    """

    file_path = _data_path(filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        pred = hubert_engine.predict(filename)
        return {
            "file": filename,
            "signal": pred["signal"],
            "embedding": pred["embedding"],
            "inference_ms": pred["inference_ms"],
            "length": pred["length"],
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception:  # pragma: no cover - runtime path
        logger.exception("HuBERT inference failed for %s", filename)
        raise HTTPException(status_code=500, detail="Inference failed")
