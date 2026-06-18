"""
Pass 2: AI4Bharat IndicConformer-600M multilingual.
Single checkpoint covering all 22 scheduled Indian languages + English.
Loads via AutoModel with trust_remote_code=True — no NeMo dependency.
Lang code routed from MMS-LID (not from Whisper pass-1 detected_language).
Falls back to Whisper-large-v3 forced-language for non-Indic content.
"""
import logging
import config

logger = logging.getLogger(__name__)

_model = None

# Languages natively supported by IndicConformer-600M (2-letter + extended codes).
_INDIC_SUPPORTED = {
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "ks", "kok",
    "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat",
    "sd", "ta", "te", "ur", "en",
}


def _load():
    global _model
    if _model is None:
        from transformers import AutoModel
        logger.info("Loading IndicConformer-600M (%s)…", config.INDIC_CONFORMER_MODEL)
        _model = AutoModel.from_pretrained(
            config.INDIC_CONFORMER_MODEL,
            trust_remote_code=True,
        )
        if config.WHISPER_DEVICE == "cuda":
            import torch
            _model = _model.to("cuda")
        _model.eval()
        logger.info("IndicConformer-600M loaded.")
    return _model


def transcribe_clip(wav_path: str, lang_code: str) -> dict:
    """
    Pass 2 ASR. lang_code: ISO 639-1 from MMS-LID routing (not Whisper).
    For Indic languages: runs IndicConformer-600M (real model, not Whisper).
    For non-Indic: runs Whisper large-v3 forced to lang_code.
    """
    model_used = config.INDIC_CONFORMER_MODEL

    if lang_code in _INDIC_SUPPORTED:
        try:
            import torch
            import torchaudio

            model = _load()
            wav, sr = torchaudio.load(wav_path)
            wav = torch.mean(wav, dim=0, keepdim=True)  # mono [1, T]
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            if config.WHISPER_DEVICE == "cuda":
                wav = wav.to("cuda")

            result = model(wav, lang_code, "rnnt")
            if isinstance(result, (list, tuple)):
                text = result[0] if result else ""
            else:
                text = result or ""
            text = str(text).strip()

            confidence = 0.75 if text else 0.0
            return {
                "text": text,
                "confidence": confidence,
                "language": lang_code,
                "model": model_used,
            }

        except Exception as exc:
            logger.warning(
                "IndicConformer failed for %s (lang=%s): %s. Falling back to Whisper forced.",
                wav_path, lang_code, exc,
            )
            model_used = "whisper_forced_lang_fallback"

    else:
        # Non-Indic language — IndicConformer doesn't cover it.
        model_used = "whisper_forced_lang"

    # Fallback: Whisper large-v3 forced to routing lang
    from services import whisper_service
    lang = lang_code if lang_code and lang_code != "und" else "auto"
    res = whisper_service.transcribe(wav_path, language=lang, use_vad=False)
    segs = res["segments"]
    if not segs:
        return {"text": "", "confidence": 0.0, "language": lang_code, "model": model_used}
    text = " ".join(s["text"] for s in segs).strip()
    conf = round(sum(s["confidence"] for s in segs) / len(segs), 3)
    return {"text": text, "confidence": conf, "language": lang_code, "model": model_used}
