import os
import warnings

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

MAX_FILE_SIZE_MB = 500
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}

WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

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
NLLB_DEVICE = "cuda"
NLLB_MAX_LENGTH = 1024

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

CONFIDENCE_THRESHOLD = 0.3  # Skip segments below this confidence
