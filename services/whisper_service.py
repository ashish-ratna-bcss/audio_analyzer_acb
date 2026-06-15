from faster_whisper import WhisperModel
import config

_model: WhisperModel = None


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

    # Enable word-level timestamps for better precision in segment alignment and timing accuracy
    segments_iter, info = model.transcribe(audio_path, word_timestamps=True, **kwargs)
    segments = []
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "confidence": round(seg.confidence, 3) if hasattr(seg, "confidence") else 0.5,
        })

    return {
        "language": info.language,
        "duration": round(info.duration, 3),
        "segments": segments,
    }
