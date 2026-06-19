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
        import torch
        from transformers import AutoModel
        logger.info("Loading IndicConformer-600M (%s)…", config.INDIC_CONFORMER_MODEL)
        _model = AutoModel.from_pretrained(
            config.INDIC_CONFORMER_MODEL,
            trust_remote_code=True,
        )
        if config.WHISPER_DEVICE == "cuda":
            _model = _model.to("cuda")
        _model.eval()
        logger.info("IndicConformer-600M loaded.")
    return _model


def transcribe_clip(wav_path: str, lang_code: str) -> dict:
    """Pass 2 ASR — IndicConformer-600M only. Independent of Whisper.

    On a language IndicConformer does not cover, ABSTAIN (empty + flag). Never
    silently fall back to Whisper — that would break model independence and let
    pass 2 masquerade as pass 1. Confidence is None ("unscored"): the CTC
    checkpoint exposes no score, and a fabricated constant must not enter the
    forensic record.
    """
    if lang_code not in _INDIC_SUPPORTED:
        return {"text": "", "confidence": None, "language": lang_code,
                "model": "indic_unsupported", "abstained": True}
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
        # Disable nvFuser for this call — it generates invalid CUDA C++ (nvrtc template
        # errors) for conformer layers under torch 2.3 + CUDA 12.1. "none" forces eager
        # mode without any kernel fusion; functionally identical, runs on GPU.
        with torch.jit.fuser("none"):
            result = model(wav, lang_code, "ctc")
        if isinstance(result, (list, tuple)):
            text = (result[0] if result else "") or ""
        else:
            text = result or ""
        text = str(text).strip()
        return {"text": text, "confidence": None, "language": lang_code,
                "model": config.INDIC_CONFORMER_MODEL, "abstained": False}
    except Exception as exc:
        logger.warning("IndicConformer failed for %s (lang=%s): %s", wav_path, lang_code, exc)
        return {"text": "", "confidence": None, "language": lang_code,
                "model": "indic_error", "abstained": True}
