"""Sortformer diarization sidecar (runs ONLY in the `sortformer` container).

This process owns NeMo and its own torch build, isolated from the base ASR
image. It loads nvidia/diar_sortformer_4spk-v1 once and serves diarization over
HTTP. Audio is referenced by path on the shared `case_data` volume — the base
worker and this sidecar see the same filesystem paths.

Endpoints:
  GET  /health           -> {"status": "ok", "model_loaded": bool}
  POST /diarize          -> {"segments": [{start,end,speaker}]}
       body: {"audio_path": "<path on shared volume>", "num_speakers": int|null}
"""
import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.sortformer_service import build_segments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sortformer_server")

SORTFORMER_MODEL = os.getenv("SORTFORMER_NEMO_MODEL", "nvidia/diar_sortformer_4spk-v1")

_model = None


def get_model():
    global _model
    if _model is None:
        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel
        logger.info("Loading Sortformer (%s)…", SORTFORMER_MODEL)
        _model = SortformerEncLabelModel.from_pretrained(model_name=SORTFORMER_MODEL)
        if torch.cuda.is_available():
            _model = _model.cuda()
        _model.eval()
        logger.info("Sortformer loaded.")
    return _model


app = FastAPI(title="Sortformer Diarization Sidecar", version="1.0.0")


class DiarizeRequest(BaseModel):
    audio_path: str
    num_speakers: int | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/diarize")
def diarize(req: DiarizeRequest):
    if not os.path.exists(req.audio_path):
        raise HTTPException(status_code=404, detail=f"audio not found: {req.audio_path}")
    model = get_model()
    logger.info("Sortformer inference on %s", req.audio_path)
    preds = model.diarize(audio=[req.audio_path], batch_size=1)
    file_preds = preds[0] if (preds and isinstance(preds[0], (list, tuple))) else (preds or [])
    segments = build_segments(file_preds)
    if not segments:
        logger.warning("Sortformer produced no segments for %s", req.audio_path)
    return {"segments": segments}
