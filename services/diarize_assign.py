def _overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(segment, turns, min_overlap: float = 0.1) -> dict:
    """Overlap-aware: every diarization turn covering the segment by >= min_overlap
    seconds contributes its speaker. Never drops a segment — with no overlap it
    falls back to the nearest turn's speaker."""
    s, e = segment["start"], segment["end"]
    hits = []
    for t in turns:
        ov = _overlap(s, e, t["start"], t["end"])
        if ov >= min_overlap:
            hits.append((ov, t["speaker"]))
    if hits:
        speakers, seen = [], set()
        for _, spk in sorted(hits, key=lambda x: -x[0]):
            if spk not in seen:
                seen.add(spk); speakers.append(spk)
        speakers.sort()
        return {"speakers": speakers, "overlap": len(speakers) > 1}

    if not turns:
        return {"speakers": ["Speaker_1"], "overlap": False}
    nearest = min(turns, key=lambda t: min(abs(s - t["start"]), abs(e - t["end"])))
    return {"speakers": [nearest["speaker"]], "overlap": False}
