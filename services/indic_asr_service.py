import logging
import config

logger = logging.getLogger(__name__)

# Cache per language: lang_code -> transformers pipeline | None (unavailable)
_models: dict = {}


def _model_id(lang_code: str) -> str | None:
    if lang_code in config.INDIC_WHISPER_LANGS:
        return f"{config.INDIC_WHISPER_BASE}-{lang_code}"
    return None


def _load_for_lang(lang_code: str):
    """Lazy-load IndicWhisper model for lang_code. Returns pipeline or None."""
    if lang_code in _models:
        return _models[lang_code]

    model_id = _model_id(lang_code)
    if model_id is None:
        _models[lang_code] = None
        return None

    try:
        import torch
        from transformers import pipeline as hf_pipeline
        device = 0 if config.WHISPER_DEVICE == "cuda" else -1
        dtype = torch.float16 if config.WHISPER_DEVICE == "cuda" else torch.float32
        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=dtype,
            token=config.PYANNOTE_AUTH_TOKEN or None,
        )
        _models[lang_code] = pipe
        logger.info("Loaded IndicWhisper %s", model_id)
        return pipe
    except Exception as exc:
        logger.warning("IndicWhisper unavailable for %s (%s): %s", lang_code, model_id, exc)
        _models[lang_code] = None
        return None


def transcribe_clip(wav_path: str, detected_lang: str) -> dict:
    """Pass 2: AI4Bharat IndicWhisper, language-routed.

    Falls back to Whisper large-v3 forced to detected_lang if the
    per-language IndicWhisper model is unavailable on HuggingFace.
    """
    pipe = _load_for_lang(detected_lang)

    if pipe is not None:
        try:
            result = pipe(wav_path)
            text = (result.get("text") or "").strip()
            # transformers ASR pipeline returns no per-segment logprob;
            # use 0.65 as a reasonable mid-range when the model produces output.
            confidence = 0.65 if text else 0.0
            return {"text": text, "confidence": confidence, "language": detected_lang}
        except Exception as exc:
            logger.warning("IndicWhisper inference failed for %s: %s", wav_path, exc)

    # Fallback: Whisper large-v3 forced to the detected language
    from services import whisper_service
    lang = detected_lang if detected_lang and detected_lang != "und" else "auto"
    res = whisper_service.transcribe(wav_path, language=lang, use_vad=False)
    segs = res["segments"]
    if not segs:
        return {"text": "", "confidence": 0.0, "language": detected_lang}
    text = " ".join(s["text"] for s in segs).strip()
    conf = round(sum(s["confidence"] for s in segs) / len(segs), 3)
    return {"text": text, "confidence": conf, "language": detected_lang}
