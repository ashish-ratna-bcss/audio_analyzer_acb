import config

_model = None


def load_indic():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _model


def transcribe_clip(wav_path: str) -> dict:
    """Pass 3: Whisper forced to Telugu — independent opinion vs auto-detect passes."""
    model = load_indic()
    segs, _ = model.transcribe(wav_path, language="te", beam_size=10,
                               condition_on_previous_text=False, vad_filter=False)
    segs = list(segs)
    if not segs:
        return {"text": "", "confidence": 0.0}
    text = " ".join(s.text for s in segs).strip()
    conf = round(sum(s.avg_logprob for s in segs) / len(segs), 3)
    return {"text": text, "confidence": max(0.0, min(1.0, (conf + 1.0) / 1.0))}
