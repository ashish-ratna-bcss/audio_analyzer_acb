import config

_model = None


def load_indic():
    global _model
    if _model is None:
        from transformers import pipeline
        _model = pipeline("automatic-speech-recognition", model=config.INDIC_ASR_MODEL,
                          token=config.PYANNOTE_AUTH_TOKEN or None,
                          trust_remote_code=True)
    return _model


def transcribe_clip(wav_path: str) -> dict:
    model = load_indic()
    out = model(wav_path)
    text = (out.get("text") if isinstance(out, dict) else str(out)) or ""
    # HF ASR pipeline gives no logprob; use a neutral mid confidence for the
    # third opinion (cross-model agreement, not this score, drives flagging).
    return {"text": text.strip(), "confidence": 0.5}
