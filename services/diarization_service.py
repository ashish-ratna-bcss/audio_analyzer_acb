from pyannote.audio import Pipeline
import torch
import config

_pipeline = None


def load_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline.from_pretrained(
            config.DIARIZATION_MODEL,
            use_auth_token=config.PYANNOTE_AUTH_TOKEN,
        )
        # Run pyannote on CPU to avoid CUDA state conflicts with DeepFilterNet/Whisper
        # cuDNN Conv layers in DF trigger CUDNN_STATUS_NOT_SUPPORTED warnings that
        # corrupt CUDA context for subsequent pyannote .to("cuda") calls on A10G.
        _pipeline.to(torch.device("cpu"))
    return _pipeline


def diarize(audio_path: str, num_speakers: int | None = 2) -> list[dict]:
    pipeline = load_pipeline()
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    diarization = pipeline(audio_path, **kwargs)

    speaker_map: dict[str, str] = {}
    speaker_counter = 1
    segments = []

    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        if speaker_label not in speaker_map:
            speaker_map[speaker_label] = f"Speaker_{speaker_counter}"
            speaker_counter += 1
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker_map[speaker_label],
        })

    return segments


def diarize_with_overlap(audio_path: str, num_speakers: int | None = None) -> list[dict]:
    """pyannote 3.1 turns WITH overlapped speech retained — overlapping instants
    yield multiple turns rather than being collapsed to one speaker."""
    pipeline = load_pipeline()
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    diarization = pipeline(audio_path, **kwargs)
    speaker_map, counter, segments = {}, 1, []
    for turn, _, label in diarization.itertracks(yield_label=True):
        if label not in speaker_map:
            speaker_map[label] = f"Speaker_{counter}"
            counter += 1
        segments.append({"start": round(turn.start, 3), "end": round(turn.end, 3),
                         "speaker": speaker_map[label]})
    return segments
