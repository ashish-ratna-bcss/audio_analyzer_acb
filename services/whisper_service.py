from faster_whisper import WhisperModel
import config

_model: WhisperModel = None


def _estimate_confidence(text: str) -> float:
    """Estimate confidence based on text characteristics."""
    if not text or len(text.strip()) < 2:
        return 0.1  # Very short = low confidence

    words = text.split()
    if len(words) > 0:
        # Check for word repetition (gibberish indicator)
        if len(set(words)) < len(words) * 0.3:  # <30% unique words
            return 0.2  # Repetitive = low confidence

    return 0.8  # Normal text = high confidence


def load_model():
    global _model
    if _model is None:
        _model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
            download_root=config.MODEL_DIR,
        )
    return _model


def transcribe(audio_path: str, language: str = "auto") -> dict:
    model = load_model()
    kwargs = {}
    if language != "auto":
        kwargs["language"] = language

    segments_iter, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=400,
        ),
        **kwargs
    )
    segments = []
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "confidence": _estimate_confidence(seg.text),
        })

    return {
        "language": info.language,
        "duration": round(info.duration, 3),
        "segments": segments,
    }
