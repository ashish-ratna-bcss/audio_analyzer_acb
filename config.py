import os

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

MAX_FILE_SIZE_MB = 500
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}

WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
PYANNOTE_AUTH_TOKEN = os.getenv("HF_TOKEN", "")

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_DEVICE = "cuda"
NLLB_MAX_LENGTH = 1024

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
