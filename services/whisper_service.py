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
            temperature=list(config.ASR_TEMPERATURES),
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


def transcribe_words(wav_path: str, lang_code: str | None) -> dict:
    """Whole-file decode with word-level timestamps (WhisperX-style).

    Whisper is a long-context model: decoding the ENTIRE file at once (rather
    than tiny VAD clips) is what produces fluent, punctuated, correctly-numbered
    output. Word timestamps then let the caller slice each diarization turn's
    text by time. Feed RAW audio — speech enhancement creates artifacts that
    degrade Whisper (see denoising-hurts-ASR literature); Whisper was trained on
    real-world noisy audio.

    Returns {"words": [{"start","end","word","prob"}], "text": <full>,
             "language": <detected/forced>}. Never raises — empty on failure so
    the pipeline falls back to per-clip IndicConformer.
    """
    empty = {"words": [], "text": "", "language": lang_code}
    try:
        model = _load()
        kwargs = {}
        if lang_code and lang_code != "auto":
            kwargs["language"] = lang_code
        segments_iter, info = model.transcribe(
            wav_path,
            task="transcribe",
            word_timestamps=True,
            beam_size=10,
            condition_on_previous_text=False,
            temperature=list(config.ASR_TEMPERATURES),
            repetition_penalty=1.3,
            no_speech_threshold=config.NO_SPEECH_MAX,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
                                speech_pad_ms=config.VAD_SPEECH_PAD_MS),
            **kwargs,
        )
        words, texts = [], []
        for seg in segments_iter:
            t = (seg.text or "").strip()
            if t:
                texts.append(t)
            for w in (seg.words or []):
                token = (w.word or "").strip()
                if not token:
                    continue
                words.append({"start": round(w.start, 3), "end": round(w.end, 3),
                              "word": token, "prob": round(getattr(w, "probability", 0.0) or 0.0, 3)})
        return {"words": words, "text": " ".join(texts).strip(),
                "language": (info.language if info else lang_code)}
    except Exception as exc:
        logger.warning("Whisper whole-file failed for %s (lang=%s): %s",
                       wav_path, lang_code, exc)
        return empty


def slice_words(words: list, start: float, end: float) -> dict:
    """Collect Whisper words whose midpoint falls within [start, end] (the
    diarization turn) into that turn's text + mean word probability (confidence).
    Midpoint membership avoids double-counting words that straddle a boundary."""
    picked = [w for w in (words or [])
              if start <= (w["start"] + w["end"]) / 2.0 <= end]
    text = " ".join(w["word"] for w in picked).strip()
    if not picked:
        return {"text": "", "confidence": None}
    # Word probabilities may be absent (e.g. the HF-pipeline fine-tuned engine
    # exposes none) -> confidence is None rather than a fabricated score.
    probs = [w["prob"] for w in picked if isinstance(w.get("prob"), (int, float))]
    conf = round(sum(probs) / len(probs), 3) if probs else None
    return {"text": text, "confidence": conf}
