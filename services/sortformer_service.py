"""Pure parsing for NVIDIA Sortformer diarization output.

No NeMo / torch imports here so the BASE image can import this module safely.
The actual Sortformer model runs in the separate `sortformer` sidecar container
(see sortformer_server.py); the base pipeline reaches it via sortformer_client.
Both share these parsing helpers so the wire format stays consistent.
"""
import re
from typing import List, Dict, Any


def parse_segment(entry) -> tuple | None:
    """Normalize one Sortformer prediction into (start, end, speaker_label).

    NeMo's diarize() returns, per file, predictions that are either a string
    "begin end speaker" (comma- or space-separated, e.g. "0.23, 5.67, 0") or a
    sequence of those three fields. Be liberal about both.
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


def build_segments(file_preds) -> List[Dict[str, Any]]:
    """Map raw Sortformer predictions for one file to overlap-preserving turns:
    [{start, end, speaker: "Speaker_N"}], sorted by start. Overlapping speakers
    stay as separate, time-overlapping turns (not collapsed)."""
    speaker_map: dict[str, str] = {}
    counter = 1
    segments: list[dict] = []
    for entry in file_preds or []:
        parsed = parse_segment(entry)
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
    return sorted(segments, key=lambda x: x["start"])
