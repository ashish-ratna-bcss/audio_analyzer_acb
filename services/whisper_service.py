from faster_whisper import WhisperModel
import config

_model: WhisperModel = None


def _estimate_confidence(text: str, duration: float = 0.0) -> float:
    if not text or len(text.strip()) < 2:
        return 0.1

    words = text.split()
    word_count = len(words)

    # Only flag extreme repetition: >6 words AND <20% unique (obvious loops)
    if word_count > 6 and len(set(words)) < word_count * 0.2:
        return 0.2

    # Only flag extreme duration: >15 seconds per word (clear hallucination)
    if duration > 0 and word_count > 0:
        if duration / word_count > 15.0:
            return 0.15

    return 0.8


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
            threshold=0.5,              # Speech probability threshold
            min_silence_duration_ms=300, # Shorter silence = more splits
            speech_pad_ms=200,          # Less padding = tighter segments
            min_speech_duration_ms=250, # Skip very short noise bursts
        ),
        **kwargs
    )
    segments = []
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "confidence": _estimate_confidence(seg.text, round(seg.end - seg.start, 3)),
        })

    return {
        "language": info.language,
        "duration": round(info.duration, 3),
        "segments": segments,
    }
