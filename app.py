import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
import config
from api.routes.stt import router as stt_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    from services.whisper_service import load_model as load_whisper
    from services.diarization_service import load_pipeline as load_diarization
    load_whisper()
    load_diarization()
    yield


app = FastAPI(title="Speech Intelligence API", version="1.0.0", lifespan=lifespan)
app.include_router(stt_router, prefix="/stt")


@app.get("/health")
def health():
    return {"status": "ok"}
