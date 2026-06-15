def align_segments(whisper_segments: list[dict], speaker_segments: list[dict]) -> list[dict]:
    aligned = []
    for wseg in whisper_segments:
        w_start, w_end = wseg["start"], wseg["end"]
        best_speaker = "Unknown"
        best_overlap = 0.0
        best_distance = float('inf')  # Distance fallback for zero-overlap cases

        for sseg in speaker_segments:
            overlap_start = max(w_start, sseg["start"])
            overlap_end = min(w_end, sseg["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sseg["speaker"]
                best_distance = float('inf')  # Reset distance when overlap found
            elif overlap == best_overlap == 0:
                # No overlap: use closest speaker by time distance
                distance = min(abs(w_start - sseg["start"]), abs(w_end - sseg["end"]))
                if distance < best_distance:
                    best_distance = distance
                    best_speaker = sseg["speaker"]

        aligned.append({
            "speaker": best_speaker,
            "start": w_start,
            "end": w_end,
            "text": wseg["text"],
        })

    return aligned
