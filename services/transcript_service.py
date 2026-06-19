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


PASS_FILE_NAMES = {
    "pass1_whisper": "whisper_transcript",
    "pass2_indic_conformer": "indic_transcript",
    "pass3_seamless": "seamless_transcript",
}


def build_per_model(file_id, segments, pass_key) -> dict:
    """Independent transcript for one model from persisted segment candidates."""
    out = []
    for s in segments:
        cand = (s.candidates or {}).get(pass_key) or {}
        out.append({
            "segment_id": s.id, "start": s.start, "end": s.end,
            "speaker": s.speaker, "language": s.detected_language,
            "text": cand.get("text", ""), "confidence": cand.get("confidence"),
            "hallucination": cand.get("hallucination"),
            "flagged_for_review": bool(s.flagged),
        })
    return {"file_id": file_id, "model": pass_key, "segments": out}


def build_validation_report(file_id, segments) -> dict:
    out, flags = [], {}
    for s in segments:
        c = s.candidates or {}
        w = c.get("pass1_whisper") or {}
        i = c.get("pass2_indic_conformer") or {}
        m = c.get("pass3_seamless") or {}
        seg_flags = [v.get("hallucination") for v in (w, i, m) if v.get("hallucination")]
        for fr in seg_flags:
            flags[fr] = flags.get(fr, 0) + 1
        out.append({
            "segment_id": s.id, "start": s.start, "end": s.end, "speaker": s.speaker,
            "whisper": {"text": w.get("text", ""), "confidence": w.get("confidence")},
            "indic": {"text": i.get("text", ""), "confidence": i.get("confidence")},
            "seamless": {"text": m.get("text", ""), "confidence": m.get("confidence")},
            "agreement": c.get("agreement"), "consensus_pass": c.get("consensus_pass"),
            "flags": seg_flags,
        })
    agreements = [o["agreement"] for o in out if isinstance(o["agreement"], (int, float))]
    summary = {
        "segments_total": len(out),
        "mean_agreement": round(sum(agreements) / len(agreements), 3) if agreements else 0.0,
        "hallucination_flags": flags,
    }
    return {"file_id": file_id, "segments": out, "summary": summary}


def write_named(case_id, file_id, name, data) -> str:
    path = storage.derivative_path(case_id, file_id, "final", f"{file_id}_{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path
