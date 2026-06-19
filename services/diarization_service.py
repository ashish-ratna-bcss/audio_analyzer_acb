import logging

from pyannote.audio import Pipeline
import torch
import config

logger = logging.getLogger(__name__)

_pipeline = None


def _apply_sensitivity(pipeline):
    """Re-instantiate pyannote with finer hyperparameters so quieter/shorter
    speaker turns register instead of being merged into surrounding silence.
    Guarded: if the param schema differs across versions, keep defaults."""
    try:
        current = pipeline.parameters(instantiated=True)
        if isinstance(current, dict) and "segmentation" in current:
            seg = dict(current.get("segmentation") or {})
            seg["min_duration_off"] = config.DIARIZATION_MIN_DURATION_OFF
            current = {**current, "segmentation": seg}
            pipeline.instantiate(current)
            logger.info("pyannote sensitivity applied: min_duration_off=%s",
                        config.DIARIZATION_MIN_DURATION_OFF)
    except Exception as exc:
        logger.warning("pyannote sensitivity tuning skipped: %s", exc)


def load_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline.from_pretrained(
            config.DIARIZATION_MODEL,
            use_auth_token=config.PYANNOTE_AUTH_TOKEN,
        )
        if config.WHISPER_DEVICE == "cuda":
            # Flush any pending CUDA ops from DeepFilterNet (L3) before pyannote
            # initializes on GPU. cuDNN Conv plan failures in DF leave dirty state
            # that causes illegal memory access when pyannote calls .to("cuda").
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        _pipeline.to(torch.device(config.WHISPER_DEVICE))
        _apply_sensitivity(_pipeline)
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
    yield multiple turns rather than being collapsed to one speaker. Dispatches
    to Sortformer if configured."""
    if getattr(config, "DIARIZER", "pyannote") == "sortformer":
        # Sortformer runs in a separate sidecar (own NeMo/torch). On any failure
        # fall back to pyannote so L4 never breaks the job.
        try:
            from services import sortformer_client
            segs = sortformer_client.diarize_with_overlap(audio_path, num_speakers)
            if segs:
                return segs
            logger.warning("Sortformer returned no segments; falling back to pyannote.")
        except Exception as e:
            logger.warning("Sortformer sidecar failed (%s); falling back to pyannote.", e)

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
