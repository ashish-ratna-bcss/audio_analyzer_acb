"""Whisper-large-v3 ASR engine (faster-whisper / CTranslate2).

Second ASR engine alongside IndicConformer. Strong on code-mixed Telugu+English,
numbers, entities, Hindi/other languages, and punctuation — exactly where the
CTC IndicConformer checkpoint is weak. The dual-engine selector (asr_selector)
inspects both outputs and picks the better per segment.

Decode params are the forensic-tuned settings proven on this evidence audio:
beam_size=10 (recovers cost/amount exchanges a narrow beam dropped),
condition_on_previous_text=False (breaks repetition loops), temperature=0
(deterministic, defensible record), repetition_penalty=1.3 (kills loops without
script garbage). No initial_prompt — it biased the model toward English and
broke faithful code-switch transcription.
"""
import math
import logging

import config

logger = logging.getLogger(__name__)

_model = None


def _load():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        logger.info("Loading Whisper (%s, %s/%s)…", config.WHISPER_MODEL,
                    config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE)
        _model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
            download_root=config.MODEL_DIR,
        )
        logger.info("Whisper loaded.")
    return _model


def transcribe_clip(wav_path: str, lang_code: str | None) -> dict:
    """Transcribe one pre-segmented clip with a forced language.

    Mirrors indic_asr_service.transcribe_clip's contract so run_asr can treat the
    two engines uniformly. Clips are already VAD-segmented, so VAD is OFF here.
    Concatenates the decoded windows into one text and aggregates the per-window
    forensic signals: confidence = mean exp(avg_logprob); no_speech_prob = max
    (most non-speech window); compression_ratio = max (most repetitive window).

    Never raises — on failure returns empty text so the selector falls back to
    IndicConformer.
    """
    empty = {"text": "", "confidence": None, "no_speech_prob": None,
             "compression_ratio": None, "language": lang_code,
             "model": config.WHISPER_MODEL}
    try:
        model = _load()
        kwargs = {}
        if lang_code and lang_code != "auto":
            kwargs["language"] = lang_code
        segments_iter, info = model.transcribe(
            wav_path,
            task="transcribe",
            word_timestamps=False,
            beam_size=10,
            condition_on_previous_text=False,
            temperature=0.0,
            repetition_penalty=1.3,
            no_speech_threshold=config.NO_SPEECH_MAX,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            vad_filter=False,
            **kwargs,
        )
        texts, confs, no_speech, compression = [], [], [], []
        for seg in segments_iter:
            t = (seg.text or "").strip()
            if t:
                texts.append(t)
            confs.append(math.exp(seg.avg_logprob))
            no_speech.append(seg.no_speech_prob)
            compression.append(seg.compression_ratio)
        text = " ".join(texts).strip()
        return {
            "text": text,
            "confidence": round(sum(confs) / len(confs), 3) if confs else None,
            "no_speech_prob": round(max(no_speech), 3) if no_speech else None,
            "compression_ratio": round(max(compression), 3) if compression else None,
            "language": (info.language if info else lang_code),
            "model": config.WHISPER_MODEL,
        }
    except Exception as exc:
        logger.warning("Whisper failed for %s (lang=%s): %s", wav_path, lang_code, exc)
        return empty
