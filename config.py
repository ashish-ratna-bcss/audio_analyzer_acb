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

# Dual-engine ASR: run Whisper-large-v3 alongside IndicConformer per clip and let
# asr_selector pick the better output (code-mix/numbers/entities -> Whisper; pure
# native Telugu -> IndicConformer). False = IndicConformer-only (legacy behaviour).
ASR_DUAL_ENGINE = os.getenv("ASR_DUAL_ENGINE", "true").lower() == "true"

# Third engine: a language-fine-tuned Whisper (default Telugu) for pure-native
# turns where generic large-v3 mishears fast/noisy native speech. Selector routes
# per turn: code-mix/numbers/Latin -> generic large-v3; pure-native -> this
# fine-tune; IndicConformer remains the rescue/cross-check. Loaded via transformers
# (downloads to MODEL_DIR like the other models). Enabled only when the file's
# language prior is in ASR_FT_LANGS. Set ASR_TELUGU_ENGINE=false to disable.
ASR_TELUGU_ENGINE = os.getenv("ASR_TELUGU_ENGINE", "true").lower() == "true"
WHISPER_TELUGU_MODEL = os.getenv("WHISPER_TELUGU_MODEL", "vasista22/whisper-telugu-large-v2")
# File-prior languages that activate the fine-tuned engine (the fine-tune is
# Telugu-only; other priors skip it and use generic large-v3 + IndicConformer).
ASR_FT_LANGS = {c.strip() for c in os.getenv("ASR_FT_LANGS", "te").split(",") if c.strip()}

# --- L7: deterministic domain-glossary correction (additive, presentation-only) ---
# Maps known ASR mishears (and case/script variants) of loan/debt-recovery domain
# entities to their canonical form. Applied when building the result; raw seg.text
# is never mutated. Curated + explicit only (no fuzzy match) so it cannot
# hallucinate. Extend at deploy via GLOSSARY_EXTRA (JSON: {canonical: [aliases]}).
GLOSSARY_CORRECTION_ENABLED = os.getenv("GLOSSARY_CORRECTION_ENABLED", "true").lower() == "true"
GLOSSARY = {
    "mPokket": ["ఇంటి పక్కనుంచి", "ఇంటి పక్క", "నిపోకెటి", "ఎం-పాకెట్", "ఎం పాకెట్", "mpokket", "m pokket"],
    "CIBIL": ["Sibulhamper", "Sibul hamper", "సిబుల్", "సిబిల్ హ్యాంపర్", "cibil"],
    "EMI": ["emi"],
    "NACH": ["nach"],
    "penalty": ["పెనాల్టీ", "ఫనాల్టి", "Penalty"],
    "recovery notice": ["Recovery notice", "రికవరీ నోటీస్", "recovery notis"],
    "reference": ["రిఫరెన్స్", "Reference"],
    "overdue": ["ఓవర్ డ్యూ", "ఓవర్ ది", "over due", "overdue"],
    "waiver": ["వేవర్", "లేవర్", "waiver"],
}
_glossary_extra = os.getenv("GLOSSARY_EXTRA", "").strip()
if _glossary_extra:
    try:
        import json as _json_g
        for _canon, _aliases in _json_g.loads(_glossary_extra).items():
            GLOSSARY.setdefault(_canon, [])
            GLOSSARY[_canon].extend(a for a in _aliases if a not in GLOSSARY[_canon])
    except Exception:
        warnings.warn("GLOSSARY_EXTRA is not valid JSON; ignored.", RuntimeWarning, stacklevel=1)

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

# Coalesce consecutive same-speaker turns separated by <= this gap into one unit
# BEFORE clip/ASR. pyannote over-segments conversational speech into many short
# turns; with the whole-file Whisper slice, tiny turns cut sentence context at
# their boundaries and garble (esp. short utterances). Merging same-speaker
# turns across natural pauses restores sentence-length spans -> Whisper's fluent
# decode survives. Safe: an interjection by the other speaker breaks the merge
# chain, so turns never merge across a speaker change. 0.5s was too tight (normal
# pauses exceed it); 1.5s spans conversational pauses without crossing speakers.
DIARIZATION_SAME_SPEAKER_GAP_S = float(os.getenv("DIARIZATION_SAME_SPEAKER_GAP_S", "1.5"))

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
