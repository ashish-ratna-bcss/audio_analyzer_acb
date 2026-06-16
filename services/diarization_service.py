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
        _pipeline.to(torch.device(config.WHISPER_DEVICE))
    return _pipeline


def diarize(audio_path: str) -> list[dict]:
    pipeline = load_pipeline()
    diarization = pipeline(audio_path)

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
