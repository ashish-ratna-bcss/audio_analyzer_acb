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

# Dynamic VAD. VAD clips quiet/narrowband phone audio (whole file read as
# silence -> Whisper echoes the initial prompt). So VAD is disabled when the
# converted audio's mean loudness falls below this dBFS floor; otherwise VAD
# runs at minimum aggressiveness, trimming only long real silence.
VAD_MIN_MEAN_DB = -30.0          # mean dBFS below this = low volume, VAD off
VAD_MIN_SILENCE_MS = 700         # min silence cut when VAD on (least aggressive)
VAD_SPEECH_PAD_MS = 400          # padding around speech so edges not clipped

# Hallucination filter: a segment is dropped only when BOTH signals agree it is
# garbage (high repetition AND very low model confidence). Quiet or legitimately
# repeated speech is never dropped, so the full conversation is preserved.
HALLUCINATION_COMPRESSION_RATIO = 2.4  # above this = repetitive/looping output
HALLUCINATION_CONFIDENCE = 0.2         # below this = model very unsure

# Primes Whisper's first 30s with domain vocabulary and style. Biases recognition
# of code-switched Telugu/English, names, amounts, and backchannels in call audio.
INITIAL_PROMPT = (
    "This is a recorded customer support phone call between an agent and a customer. "
    "Speakers mix Telugu and English (code-switching). "
    "The conversation includes greetings, account numbers, names, amounts in rupees, "
    "dates, order IDs, complaints, and confirmations like haan, sare, okay, thank you. "
    "Transcribe with correct punctuation."
)
