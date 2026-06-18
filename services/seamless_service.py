import logging
import config

logger = logging.getLogger(__name__)

_processor = None
_model = None

# Whisper ISO 639-1 codes -> SeamlessM4T BCP-47 language tags
_LANG_MAP: dict[str, str] = {
    "te": "tel", "ta": "tam", "kn": "kan", "ml": "mal",
    "hi": "hin", "mr": "mar", "bn": "ben", "gu": "guj",
    "pa": "pan", "ur": "urd", "en": "eng", "ar": "arb",
    "fr": "fra", "de": "deu", "es": "spa", "ja": "jpn",
    "ko": "kor", "zh": "cmn", "ru": "rus", "it": "ita",
    "pt": "por", "nl": "nld", "pl": "pol", "tr": "tur",
    "vi": "vie", "th": "tha", "id": "ind", "si": "sin",
    "da": "dan", "fi": "fin", "sv": "swe", "no": "nob",
}


def load_seamless():
    global _processor, _model
    if _model is None:
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
        logger.info("Loading SeamlessM4T v2 (%s)…", config.SEAMLESS_MODEL)
        _processor = AutoProcessor.from_pretrained(
            config.SEAMLESS_MODEL,
            token=config.PYANNOTE_AUTH_TOKEN or None,
        )
        _model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            config.SEAMLESS_MODEL,
            token=config.PYANNOTE_AUTH_TOKEN or None,
        )
        if config.WHISPER_DEVICE == "cuda":
            _model = _model.to("cuda").half()
        logger.info("SeamlessM4T v2 loaded.")
    return _processor, _model


def transcribe_clip(wav_path: str, detected_lang: str) -> dict:
    """Pass 3: SeamlessM4T v2 ASR — independent multilingual model."""
    import torch
    import librosa

    tgt_lang = _LANG_MAP.get(detected_lang, "eng")

    try:
        processor, model = load_seamless()
        waveform, _ = librosa.load(wav_path, sr=16000, mono=True)

        inputs = processor(
            audios=waveform,
            return_tensors="pt",
            sampling_rate=16000,
        )

        device = "cuda" if config.WHISPER_DEVICE == "cuda" else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                tgt_lang=tgt_lang,
                generate_speech=False,
            )

        text = processor.decode(
            output_tokens[0].tolist()[0], skip_special_tokens=True
        ).strip()
        confidence = 0.6 if text else 0.0
        return {"text": text, "confidence": confidence, "language": detected_lang}

    except Exception as exc:
        logger.warning("SeamlessM4T inference failed for %s: %s", wav_path, exc)
        return {"text": "", "confidence": 0.0, "language": detected_lang}
