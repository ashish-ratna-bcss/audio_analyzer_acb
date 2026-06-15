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


def transcribe(audio_path: str, language: str = "auto") -> dict:
    model = load_model()
    kwargs = {}
    if language != "auto":
        kwargs["language"] = language

    segments_iter, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        beam_size=5,
        # Keep conversation context coherent across 30s windows. Loops are
        # suppressed by hallucination_silence_threshold + compression_ratio,
        # not by disabling conditioning (which would fragment the transcript).
        condition_on_previous_text=True,
        hallucination_silence_threshold=2.0,   # skip hallucinated text in long silences
        no_speech_threshold=0.6,               # drop true silence
        log_prob_threshold=-1.0,               # temperature fallback on low-confidence
        compression_ratio_threshold=2.4,       # catch repetition loops
        repetition_penalty=1.1,                # mild anti-loop bias
        # Soft VAD: remove only real silence (>=500ms). Aggressive VAD was
        # clipping quiet/distant speakers on narrowband phone audio.
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        initial_prompt=config.INITIAL_PROMPT,
        **kwargs
    )

    segments = []
    for seg in segments_iter:
        # Real confidence from the model's own log-probability, not a
        # word-uniqueness heuristic (which deleted legitimate repeated speech).
        confidence = round(math.exp(seg.avg_logprob), 3)
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "confidence": confidence,
            "no_speech_prob": round(seg.no_speech_prob, 3),
            "compression_ratio": round(seg.compression_ratio, 3),
        })

    return {
        "language": info.language,
        "duration": round(info.duration, 3),
        "segments": segments,
    }
