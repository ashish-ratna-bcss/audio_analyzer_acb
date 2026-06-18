def union_segments(branches):
    """Merge multiple branches of {start,end} segments into one additive union:
    overlapping or touching intervals are coalesced; nothing is dropped."""
    intervals = []
    for branch in branches:
        for seg in branch:
            intervals.append((float(seg["start"]), float(seg["end"])))
    if not intervals:
        return []
    intervals.sort()
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:  # overlap or touch
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [{"start": s, "end": e} for s, e in merged]


def total_duration(segs) -> float:
    return round(sum(s["end"] - s["start"] for s in segs), 6)


def should_include_separation(pre_union_count: int, stem_count: int) -> bool:
    """Per design: include the separated stem only if it does not REDUCE the
    detected region count vs. the pre-separation union."""
    return stem_count >= pre_union_count
