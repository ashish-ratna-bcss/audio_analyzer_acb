import config

_model = None
_get_ts = None


def load_vad():
    """Lazy-load Silero VAD. Heavy import deferred so unit tests need no model."""
    global _model, _get_ts
    if _model is None:
        from silero_vad import load_silero_vad, get_speech_timestamps
        _model = load_silero_vad()
        _get_ts = get_speech_timestamps
    return _model, _get_ts


def detect_speech(wav_path: str) -> list[dict]:
    import soundfile as sf
    import torch
    model, get_ts = load_vad()
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    ts = get_ts(
        torch.from_numpy(audio), model,
        sampling_rate=sr,
        threshold=config.VAD_THRESHOLD,
        min_speech_duration_ms=config.VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=config.VAD_MIN_SILENCE_MS_L3,
        speech_pad_ms=config.VAD_SPEECH_PAD_MS_L3,
        return_seconds=True,
    )
    return [{"start": round(t["start"], 3), "end": round(t["end"], 3)} for t in ts]
