"""NVIDIA Sortformer end-to-end diarization (pilot, behind config.DIARIZER).

Sortformer is a single end-to-end neural diarizer (no separate VAD + speaker
embedding + clustering cascade). It emits per-speaker segments that may overlap
in time, which is exactly what `diarize_with_overlap` must return so the L5
active-speaker partition can mark cross-talk units.

Loaded only when config.DIARIZER == "sortformer"; NeMo is an optional heavy
dependency and is not imported at module load.
"""
import logging
import re
from typing import List, Dict, Any

import config

logger = logging.getLogger(__name__)

_model = None


def get_model():
    global _model
    if _model is None:
        from nemo.collections.asr.models import SortformerEncLabelModel
        logger.info("Loading Sortformer (nvidia/diar_sortformer_4spk-v1)…")
        _model = SortformerEncLabelModel.from_pretrained(
            model_name="nvidia/diar_sortformer_4spk-v1")
        if getattr(config, "_HAS_CUDA", False):
            _model = _model.cuda()
        _model.eval()
        logger.info("Sortformer loaded.")
    return _model


def _parse_segment(entry) -> tuple | None:
    """Normalize one Sortformer prediction into (start, end, speaker_label).

    NeMo's Sortformer.diarize() returns, per audio file, a list of predictions.
    Each prediction is either a string "start end speaker" (e.g. "0.23 5.67
    speaker_0") or a sequence/tuple of those three fields. Be liberal about both.
    """
    if isinstance(entry, str):
        parts = [p for p in re.split(r"[,\s]+", entry.strip()) if p]
    elif isinstance(entry, (list, tuple)):
        parts = [str(p) for p in entry]
    else:
        return None
    if len(parts) < 3:
        return None
    try:
        start = float(parts[0])
        end = float(parts[1])
    except ValueError:
        return None
    speaker = "_".join(parts[2:]).strip()
    return start, end, speaker


def diarize_with_overlap(audio_path: str, num_speakers: int | None = None) -> List[Dict[str, Any]]:
    """Run Sortformer on one file. Returns overlap-preserving turns:
    [{start, end, speaker: "Speaker_N"}], sorted by start.
    """
    model = get_model()
    logger.info("Sortformer inference on %s", audio_path)

    # NeMo 2.x: diarize() returns a list (one item per input audio) of predicted
    # speaker-active segments. Overlapping speakers yield separate, time-overlapping
    # entries — retained as distinct turns (do NOT collapse).
    preds = model.diarize(audio=[audio_path], batch_size=1)

    # diarize() returns either a flat list of "begin end speaker" segments (single
    # audio) or a list-of-lists (one per input audio). Normalize to this file's list.
    if preds and isinstance(preds[0], (list, tuple)):
        file_preds = preds[0]
    else:
        file_preds = preds or []

    speaker_map: dict[str, str] = {}
    counter = 1
    segments: list[dict] = []
    for entry in file_preds:
        parsed = _parse_segment(entry)
        if parsed is None:
            continue
        start, end, raw_spk = parsed
        if end <= start:
            continue
        if raw_spk not in speaker_map:
            speaker_map[raw_spk] = f"Speaker_{counter}"
            counter += 1
        segments.append({"start": round(start, 3), "end": round(end, 3),
                         "speaker": speaker_map[raw_spk]})

    if not segments:
        logger.warning("Sortformer produced no segments for %s", audio_path)
    return sorted(segments, key=lambda x: x["start"])
