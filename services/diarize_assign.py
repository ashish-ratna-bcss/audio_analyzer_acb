def _overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(segment, turns, min_overlap: float = 0.1) -> dict:
    """
    Assign dominant speaker to segment by total overlap duration.
    Returns primary speaker (most speech time in segment) + secondary list.
    Falls back to nearest turn if no overlap found.
    """
    s, e = segment["start"], segment["end"]

    # Accumulate total overlap per speaker
    overlap_by_spk: dict[str, float] = {}
    for t in turns:
        ov = _overlap(s, e, t["start"], t["end"])
        if ov >= min_overlap:
            overlap_by_spk[t["speaker"]] = overlap_by_spk.get(t["speaker"], 0.0) + ov

    if overlap_by_spk:
        # Primary = speaker with most total overlap time
        ranked = sorted(overlap_by_spk.items(), key=lambda x: -x[1])
        primary = ranked[0][0]
        secondary = [spk for spk, _ in ranked[1:]]
        return {
            "speakers": [primary],
            "primary": primary,
            "secondary": secondary,
            "overlap": len(ranked) > 1,
            "overlap_seconds": {spk: round(sec, 3) for spk, sec in ranked},
        }

    # No overlap — fall back to nearest turn
    if not turns:
        return {"speakers": ["Speaker_1"], "primary": "Speaker_1",
                "secondary": [], "overlap": False, "overlap_seconds": {}}
    nearest = min(turns, key=lambda t: min(abs(s - t["start"]), abs(e - t["end"])))
    return {"speakers": [nearest["speaker"]], "primary": nearest["speaker"],
            "secondary": [], "overlap": False, "overlap_seconds": {}}
