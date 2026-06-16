import math
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


def _decode(model, audio_path, language, use_vad, initial_prompt):
    """Single Whisper pass. Returns (segments, info)."""
    kwargs = {}
    if language != "auto":
        kwargs["language"] = language

    # Dynamic VAD: caller passes use_vad=False for low-volume audio that VAD
    # would otherwise strip entirely. When on, run least-aggressive settings.
    if use_vad:
        kwargs["vad_filter"] = True
        kwargs["vad_parameters"] = dict(
            min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
            speech_pad_ms=config.VAD_SPEECH_PAD_MS,
        )
    else:
        kwargs["vad_filter"] = False

    segments_iter, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        beam_size=5,
        # Keep conversation context coherent across 30s windows.
        condition_on_previous_text=True,
        # NOTE: hallucination_silence_threshold removed. Combined with
        # initial_prompt it caused long calls to collapse to a single
        # prompt-echo segment (the model emitted the prompt's imperative,
        # then silence-suppression wiped the rest). Repetition is still
        # bounded by compression_ratio_threshold + repetition_penalty.
        no_speech_threshold=0.6,               # drop true silence
        log_prob_threshold=-1.0,               # temperature fallback on low-confidence
        compression_ratio_threshold=2.4,       # catch repetition loops
        repetition_penalty=1.1,                # mild anti-loop bias
        initial_prompt=initial_prompt,
        **kwargs
    )

    segments = []
    for seg in segments_iter:
        # Real confidence from the model's own log-probability.
        confidence = round(math.exp(seg.avg_logprob), 3)
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "confidence": confidence,
            "no_speech_prob": round(seg.no_speech_prob, 3),
            "compression_ratio": round(seg.compression_ratio, 3),
        })
    return segments, info


def _coverage(segments, duration):
    if not duration:
        return 1.0
    speech = sum(s["end"] - s["start"] for s in segments)
    return speech / duration


def transcribe(audio_path: str, language: str = "auto", use_vad: bool = True) -> dict:
    model = load_model()

    # First pass with the domain prompt (biases code-switched Telugu/English,
    # names, amounts). On some audio the prompt makes Whisper collapse into
    # echoing it; detect that via low speech coverage and re-run without it.
    segments, info = _decode(model, audio_path, language, use_vad,
                             config.INITIAL_PROMPT)

    if _coverage(segments, info.duration) < config.MIN_SPEECH_COVERAGE:
        fallback, info = _decode(model, audio_path, language, use_vad,
                                 initial_prompt=None)
        # Keep whichever pass recovered more speech.
        if _coverage(fallback, info.duration) > _coverage(segments, info.duration):
            segments = fallback

    return {
        "language": info.language,
        "duration": round(info.duration, 3),
        "segments": segments,
    }
