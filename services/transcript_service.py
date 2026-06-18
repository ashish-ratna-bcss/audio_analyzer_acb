import json

from services import storage


def final_path(case_id: str, file_id: str) -> str:
    return storage.derivative_path(case_id, file_id, "final",
                                   f"{file_id}_certified_transcript.json")


def build(case_id, file_id, source_hash, segments, *, status) -> dict:
    out = []
    for s in segments:
        out.append({
            "segment_id": s.id,
            "start": s.start, "end": s.end, "speaker": s.speaker,
            "overlap": "+" in (s.speaker or ""),
            "text": s.text, "language": s.detected_language,
            "confidence": s.confidence, "source_pass": s.source_pass,
            "flagged_for_review": bool(s.flagged),
            "review_status": s.review_status,
            "reviewer_id": None,
        })
    return {"file_id": file_id, "case_id": case_id,
            "source_hash_sha256": source_hash, "segments": out, "status": status}


def write(case_id, file_id, data) -> str:
    path = final_path(case_id, file_id)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path
