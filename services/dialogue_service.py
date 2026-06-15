def group_turns(segments: list[dict]) -> list[dict]:
    """Merge consecutive same-speaker segments into dialogue turns.

    Input segments must be in chronological order (Whisper yields them that way
    and alignment/translation preserve it). Each turn keeps the speaker, the span
    start/end, the joined text, an average confidence, and joined translated_text
    when present. Raw per-segment timing is left untouched for the caller to keep.
    """
    turns: list[dict] = []
    for seg in segments:
        if turns and turns[-1]["speaker"] == seg["speaker"]:
            turn = turns[-1]
            turn["end"] = seg["end"]
            turn["text"] = (turn["text"] + " " + seg["text"]).strip()
            turn["_confidences"].append(seg.get("confidence", 1.0))
            if seg.get("translated_text"):
                prev = turn.get("translated_text") or ""
                turn["translated_text"] = (prev + " " + seg["translated_text"]).strip()
        else:
            turns.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "translated_text": seg.get("translated_text"),
                "_confidences": [seg.get("confidence", 1.0)],
            })

    for turn in turns:
        confidences = turn.pop("_confidences")
        turn["confidence"] = round(sum(confidences) / len(confidences), 3)

    return turns
