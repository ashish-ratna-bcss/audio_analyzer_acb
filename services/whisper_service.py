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


def transcribe(audio_path: str, language: str = "auto", use_vad: bool = True,
               task: str = "transcribe") -> dict:
    """Run one Whisper pass.

    task="transcribe" -> faithful native transcript (code-switched as spoken).
    task="translate"  -> Whisper's built-in speech->English translation.
    """
    model = load_model()
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

    # No initial_prompt: it must stay a FAITHFUL transcription. An English
    # prompt biased Whisper into emitting English even for Telugu speech
    # (and, with hallucination_silence_threshold, collapsed long calls to a
    # single prompt-echo segment). Without it, output stays in the spoken
    # language/script -- pure English -> English, code-switched Telugu+English
    # -> the same mix. Language is auto-detected unless the caller forces one.
    segments_iter, info = model.transcribe(
        audio_path,
        task=task,
        word_timestamps=True,
        # Wider beam = more accurate decoding. On evidence audio beam_size=10
        # recovered the critical cost/amount exchange that beam_size=5 dropped.
        beam_size=10,
        # OFF: with it on, a hallucinated phrase fed forward into each window
        # and snowballed into a repetition loop (charan call looped one phrase
        # and lost most speech). Off = each window decoded independently, which
        # breaks the loop and recovers the full transcript.
        condition_on_previous_text=False,
        # Deterministic decoding (single temperature, no sampling fallback) so
        # the same evidence audio always yields the same transcript -- required
        # for a defensible forensic record. The default temperature fallback
        # was non-deterministic and occasionally truncated a long call.
        temperature=0.0,
        # repetition_penalty raised 1.1 -> 1.3: at temperature=0 there is no
        # sampling to escape a repetition loop, so a stronger deterministic
        # penalty is needed. 1.3 collapsed the "fifteen" loop (compression
        # ratio 3.68 -> 1.94) without the script garbage that no_repeat_ngram
        # caused or the English drift that 1.5 caused.
        repetition_penalty=1.3,
        no_speech_threshold=0.6,              # drop true silence
        log_prob_threshold=-1.0,              # marks low-confidence segments
        compression_ratio_threshold=2.4,      # marks repetitive segments
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

    return {
        "language": info.language,
        "duration": round(info.duration, 3),
        "segments": segments,
    }
