import os
import warnings

# API key for request auth (X-API-Key header). Empty = auth disabled (local dev).
API_KEY = os.getenv("API_KEY", "")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
# Whisper model download cache. Overridable so Docker can point it at a mounted
# volume separate from the `models/` Python package (avoids clobbering schemas).
MODEL_DIR = os.getenv("MODEL_DIR", "models")

MAX_FILE_SIZE_MB = 500
# Audio + common video containers (ffmpeg extracts the audio track from video).
ALLOWED_EXTENSIONS = {
    # audio
    ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac",
    # video (audio extracted via ffmpeg)
    ".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".3gp",
}

# Device auto-detects so the same image runs on any instance: CUDA (GPU,
# float16) when available, else CPU (int8). Override with env vars if needed.
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False
_DEVICE = os.getenv("DEVICE", "cuda" if _HAS_CUDA else "cpu")

WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = _DEVICE
WHISPER_COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE", "float16" if _DEVICE == "cuda" else "int8"
)

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
_hf_token = os.getenv("HF_TOKEN", "")
if not _hf_token:
    warnings.warn(
        "HF_TOKEN env var not set. Pyannote diarization will fail at runtime.",
        RuntimeWarning,
        stacklevel=1,
    )
PYANNOTE_AUTH_TOKEN = _hf_token

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_DEVICE = _DEVICE
NLLB_MAX_LENGTH = 1024

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

# Dynamic VAD. VAD clips quiet/narrowband phone audio (whole file read as
# silence -> Whisper echoes the initial prompt). So VAD is disabled when the
# converted audio's mean loudness falls below this dBFS floor; otherwise VAD
# runs at minimum aggressiveness, trimming only long real silence.
VAD_MIN_MEAN_DB = -38.0          # mean dBFS below this = low volume, VAD off
VAD_MIN_SILENCE_MS = 2000        # only cut 2s+ silence (forensic: keep more audio)
VAD_SPEECH_PAD_MS = 600          # wider padding so speech edges not clipped

# NOTE: forensic mode keeps every segment (no hallucination drop filter).
# This is evidence audio, so possible hallucinations are flagged via each
# segment's confidence / compression_ratio for human review, never deleted.

# NOTE: no Whisper initial_prompt. An English domain prompt biased the model
# into emitting English even for Telugu speech, breaking faithful code-switch
# transcription (and collapsed long calls to a prompt-echo segment). Output
# now follows the spoken language/script as detected.

# --- Phase 1: async foundation (Celery / Redis / Postgres / case store) ---

# SQLAlchemy URL. Local/test default is in-memory SQLite; deploy sets Postgres
# via env (postgresql+psycopg2://...).
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+pysqlite:///./forensic_local.db")

# Celery broker + result backend.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Root of the immutable case/evidence tree (originals, derivatives, audit).
CASE_STORE_PATH = os.getenv("CASE_STORE_PATH", "case_data")

CPU_QUEUE = "cpu_queue"
GPU_QUEUE = "gpu_queue"

# Run Celery tasks inline (no broker) — set true in tests.
CELERY_TASK_ALWAYS_EAGER = os.getenv("CELERY_TASK_ALWAYS_EAGER", "false").lower() == "true"
