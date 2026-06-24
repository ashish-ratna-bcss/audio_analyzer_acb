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
VAD_MIN_MEAN_DB = -55.0          # mean dBFS below this = low volume, VAD off (lowered to catch quiet audio)
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

# --- Phase 3: recall branches (enhancement / VAD union / separation) ---
# Silero VAD sensitivity for L3 segment detection. Env-overridable for tuning
# without a rebuild.
# Threshold 0.10 was far too low: on noisy forensic recordings Silero flagged the
# WHOLE file as speech (every branch -> one 0..end block), so gap-recovery forced
# IndicConformer onto pure silence and it hallucinated cross-script garbage
# (fabricated text — the worst forensic failure). Empirically on BVR_23_02_2021:
# 0.10 -> whole file; 0.30 -> 94s/69 segs; 0.50 -> 60s/50 segs (ground-truth
# speech ~65s). 0.35 keeps recall while cutting noise-only regions.
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "250"))  # 250ms: cuts pure noise bursts while keeping short words (~సార్ ≥300ms)
VAD_SPEECH_PAD_MS_L3 = int(os.getenv("VAD_SPEECH_PAD_MS_L3", "250"))  # less bridging of separate utterances into one block
VAD_MIN_SILENCE_MS_L3 = int(os.getenv("VAD_MIN_SILENCE_MS_L3", "200"))  # merge segments within 200ms (avoids word-level fragmentation)
# Recall branches (enhanced / separated) in the L3 VAD union. The additive union
# adds ANY region a branch flags as speech. DeepFilterNet enhancement creates
# artifacts Silero reads as speech, so enhanced VAD injects phantom regions ->
# ASR runs on noise -> fabricated text. For forensic evidence, anchor speech
# detection on the ORIGINAL track only (precision over recall). Set to 1 to
# restore the additive recall behaviour.
VAD_INCLUDE_RECALL_BRANCHES = os.getenv("VAD_INCLUDE_RECALL_BRANCHES", "0") == "1"
TABLE_MIN_CHARS = int(os.getenv("TABLE_MIN_CHARS", "3"))  # drop 1-2 char pure noise; Telugu short words (సార్=4) pass
DFN_MODEL = "DeepFilterNet3"          # DeepFilterNet3 enhancement
DEMUCS_MODEL = "htdemucs_ft"          # HTDemucs separation checkpoint

# --- Phase 4: attribution + multi-pass ASR ---

# Pre-ASR: MMS-LID audio-grounded language identification (independent of Whisper decoder).
MMS_LID_MODEL = os.getenv("MMS_LID_MODEL", "facebook/mms-lid-256")

# Pass 2: AI4Bharat IndicConformer-600M — single checkpoint, all 22 scheduled Indian languages.
# Loads via AutoModel with trust_remote_code=True; no NeMo dependency.
INDIC_CONFORMER_MODEL = os.getenv("INDIC_CONFORMER_MODEL", "ai4bharat/indic-conformer-600m-multilingual")

# Pass 3: SeamlessM4T v2 — Meta multilingual end-to-end model (run on original audio).
SEAMLESS_MODEL = os.getenv("SEAMLESS_MODEL", "facebook/seamless-m4t-v2-large")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/LaBSE")

# --- Phase: intelligent segmentation + overlap separation (accuracy-first) ---

# Gap recovery: VAD-confirmed speech with NO diarization turn is windowed and
# transcribed as Speaker_unknown. Recovers regions pyannote drops as non-speech
# (loud cross-talk, far-field, music-masked speech) and low-volume conversation.
# Nothing inside the VAD union ever goes untranscribed.
GAP_RECOVERY_ENABLED = os.getenv("GAP_RECOVERY_ENABLED", "true").lower() == "true"
GAP_WINDOW_S = float(os.getenv("GAP_WINDOW_S", "10.0"))   # split long gaps into windows (shorter = fewer silence hallucinations)
GAP_MIN_DUR_S = float(os.getenv("GAP_MIN_DUR_S", "1.0"))  # ignore micro-gaps

# pyannote 3.1 sensitivity. min_duration_off=0 stops the pipeline bridging over
# short pauses into one block, yielding finer/quieter speaker turns.
DIARIZATION_MIN_DURATION_OFF = float(os.getenv("DIARIZATION_MIN_DURATION_OFF", "0.0"))

# Overlap separation: SpeechBrain SepFormer splits cross-talk windows into
# per-speaker streams; each stream is transcribed independently by all 3 ASR
# passes so both/all overlapped voices are recovered (not just the loudest).
OVERLAP_SEPARATION_ENABLED = os.getenv("OVERLAP_SEPARATION_ENABLED", "true").lower() == "true"
SEPFORMER_MODEL = os.getenv("SEPFORMER_MODEL", "speechbrain/sepformer-wsj02mix")
OVERLAP_MIN_DUR_S = float(os.getenv("OVERLAP_MIN_DUR_S", "0.5"))

# Per-clip dynamic loudness normalization before ASR — boosts quiet speech so
# Whisper/IndicConformer/Seamless decode it instead of treating it as silence.
CLIP_NORMALIZE = os.getenv("CLIP_NORMALIZE", "true").lower() == "true"

# Below this MMS-LID top-1 confidence, ignore its routing and fall back to
# Whisper auto-detect (low-conf LID misfires poison all 3 ASR passes).
MMS_LID_MIN_CONFIDENCE = float(os.getenv("MMS_LID_MIN_CONFIDENCE", "0.5"))

# --- Phase: independent multi-model ASR + cross-model validation ---

# Allowed language ISO-639-1 set. Empty = open auto-detect (default). Set e.g.
# ALLOWED_LANGS=te,en,hi to constrain a known-language case and kill LID misroutes.
ALLOWED_LANGS = {c.strip() for c in os.getenv("ALLOWED_LANGS", "").split(",") if c.strip()}

# Min MMS-LID top-1 confidence for a clip's LID to count toward the file vote
# and to be trusted over the file prior.
LID_VOTE_MIN_CONF = float(os.getenv("LID_VOTE_MIN_CONF", "0.5"))

# Min MMS-LID confidence required for a clip to be routed to a language that
# DIFFERS from the file prior. Recordings are usually dominated by one language;
# per-clip LID on noise/short clips returns random languages at mid confidence,
# poisoning those segments. A clip matching the file prior is trusted at
# LID_VOTE_MIN_CONF; deviating from it demands this higher bar (genuine
# code-switch still passes). Keeps auto-detect fully general — the prior is
# whatever language actually dominates the file, no hardcoded language.
LID_DEVIATE_MIN_CONF = float(os.getenv("LID_DEVIATE_MIN_CONF", "0.85"))

# Whisper no_speech_prob above this blanks the pass (true non-speech / silence).
NO_SPEECH_MAX = float(os.getenv("NO_SPEECH_MAX", "0.6"))

# Mean pairwise embedding cosine below this flags a segment for cross-model disagreement.
AGREEMENT_MIN = float(os.getenv("AGREEMENT_MIN", "0.6"))

# EBU R128 integrated loudness target for per-clip normalization (LUFS).
LOUDNORM_LUFS = float(os.getenv("LOUDNORM_LUFS", "-16.0"))

# Known ASR hallucination phrases emitted on non-speech (training-data ghosts).
# Matched case- and punctuation-insensitive. Env-extendable via GHOST_PHRASES_EXTRA (comma-sep).
GHOST_PHRASES = [
    "thank you", "thank you.", "thanks for watching", "thanks for watching!",
    "please subscribe", "subscribe", "like and subscribe", ". .", "...",
    "[music]", "[music playing]", "[applause]", "(music)",
    "ご視聴ありがとうございました", "Продолжение следует...",
] + [p.strip() for p in os.getenv("GHOST_PHRASES_EXTRA", "").split(",") if p.strip()]

# --- Phase: IndicConformer-only ASR (single model + self-cross-check) ---

# Enhanced-vs-original self-cross-check: below this agreement the two IndicConformer
# runs disagree (enhancement changed the words) -> flag enh_orig_divergence.
INDIC_SELFCHECK_MIN = float(os.getenv("INDIC_SELFCHECK_MIN", "0.6"))

# Final confidence below this flags the segment for human review.
INDIC_CONF_MIN = float(os.getenv("INDIC_CONF_MIN", "0.5"))

# Hard suppression floor. At/below this enh-vs-original agreement the two passes
# share essentially no signal (one empty or totally divergent) -> the "text" is
# noise the model emitted onto non-speech. For forensic integrity we BLANK it
# (mark unintelligible, keep the clip for review) rather than enter fabricated
# words into the record. Distinct from INDIC_SELFCHECK_MIN, which only flags.
# Conservative default keeps borderline-real low-agreement speech.
INDIC_SUPPRESS_BELOW = float(os.getenv("INDIC_SUPPRESS_BELOW", "0.1"))

# --- Phase: LLM transcript enhancement layer (L6b) ---
# Additive ASR character/spelling correction. Writes ONLY to
# candidates["llm_enhancement"]; never mutates seg.text, timestamps, speakers,
# or any existing response. Default DISABLED so existing behavior is the default
# on laptop/tests and any deploy that does not opt in.
LLM_ENHANCEMENT_ENABLED = os.getenv("LLM_ENHANCEMENT_ENABLED", "false").lower() == "true"
# Provider: "ollama" (local, air-gappable; stdlib urllib) or "anthropic" (lazy import).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
# Primary Ollama server + model (local box, qwen2.5 14B instruct, q4_K_M quant).
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:14b-instruct-q4_K_M")
LLM_OLLAMA_URL = os.getenv("LLM_OLLAMA_URL", "http://localhost:11434")
# Fallback Ollama server + model, used when the primary URL/model is unreachable
# or errors (keeps the layer working if the local 14B is down). Empty URL = no
# fallback. Defaults to the shared remote box with the 7B instruct model.
LLM_OLLAMA_FALLBACK_URL = os.getenv("LLM_OLLAMA_FALLBACK_URL", "http://32.192.131.130:11434")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "qwen2.5:7b-instruct-q4_K_M")
LLM_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Min ASR confidence for a segment to be sent to the LLM at all (0.0 = attempt all).
LLM_ENHANCEMENT_MIN_CONF = float(os.getenv("LLM_ENHANCEMENT_MIN_CONF", "0.0"))
# Min LLM-reported correction_confidence for a correction to actually be applied;
# below this the correction is dropped and the raw text is kept (unchanged).
LLM_CORRECTION_MIN_CONF = float(os.getenv("LLM_CORRECTION_MIN_CONF", "0.92"))
# Deterministic guard: reject a correction whose word count deviates from the
# original by more than this ratio (catches add/remove/rewrite the prompt missed).
LLM_MAX_WORD_DELTA_RATIO = float(os.getenv("LLM_MAX_WORD_DELTA_RATIO", "0.15"))
# Parallel Ollama workers for L6b. ASR is done by the time L6b runs, so the
# GPU is free. Match this to OLLAMA_NUM_PARALLEL on the Ollama server so it
# actually batches requests instead of queuing them. 4 fits comfortably in
# 24GB VRAM with qwen2.5:14b-instruct-q4_K_M (~8-9GB per model instance).
LLM_MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "4"))

# Per-segment LLM call timeout (seconds). Empty / "none" / "0" = no timeout
# (block until the model responds), matching the deployed OLLAMA_TIMEOUT=None.
_llm_timeout_raw = os.getenv("LLM_ENHANCEMENT_TIMEOUT_S", "").strip().lower()
LLM_ENHANCEMENT_TIMEOUT_S = (
    None if _llm_timeout_raw in ("", "none", "0") else int(_llm_timeout_raw)
)
