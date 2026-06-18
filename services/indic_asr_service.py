import config

_model = None

# indicwav2vec_v1_telugu is a standard wav2vec2 CTC model — no custom code,
# works with AutoModelForCTC, compatible with transformers pipeline().
_INDIC_MODEL = "ai4bharat/indicwav2vec_v1_telugu"


def load_indic():
    global _model
    if _model is None:
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        _model = pipeline("automatic-speech-recognition", model=_INDIC_MODEL,
                          token=config.PYANNOTE_AUTH_TOKEN or None,
                          device=device)
    return _model


def transcribe_clip(wav_path: str) -> dict:
    model = load_indic()
    out = model(wav_path)
    text = (out.get("text") if isinstance(out, dict) else str(out)) or ""
    return {"text": text.strip(), "confidence": 0.5}
