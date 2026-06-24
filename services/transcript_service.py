import json

import config
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


def build_indic_validation(file_id, segments) -> dict:
    """Single-model (IndicConformer) validation report: enhanced-vs-original
    self-cross-check per segment + flag summary."""
    out, flags = [], {}
    for s in segments:
        c = (s.candidates or {}).get("indic_conformer") or {}
        h = c.get("hallucination")
        if h:
            flags[h] = flags.get(h, 0) + 1
        out.append({
            "segment_id": s.id, "start": s.start, "end": s.end, "speaker": s.speaker,
            "language": s.detected_language,
            "enh_text": c.get("enh_text", ""), "org_text": c.get("org_text", ""),
            "text": c.get("text", ""), "agreement": c.get("agreement"),
            "confidence": c.get("confidence"), "abstained": c.get("abstained"),
            "hallucination": h, "flagged": bool(s.flagged),
        })
    agg = [o["agreement"] for o in out if isinstance(o["agreement"], (int, float))]
    summary = {
        "segments_total": len(out),
        "segments_flagged": sum(1 for s in segments if s.flagged),
        "mean_agreement": round(sum(agg) / len(agg), 3) if agg else 0.0,
        "hallucination_flags": flags,
    }
    return {"file_id": file_id, "model": "indic_conformer",
            "segments": out, "summary": summary}


def _mmss(t: float) -> str:
    """Seconds -> MM.SS (floored), zero-padded, matching the court table style."""
    t = max(0.0, float(t))
    m = int(t // 60)
    s = int(t % 60)
    return f"{m:02d}.{s:02d}"


def build_conversation_table(file_id, segments) -> dict:
    """Court-ready Time/Person/Conversation table from the final transcript.
    Skips empty segments; numbers sequentially; MM.SS timestamps. Person is the
    diarization speaker label (renamed to A.O / Complt by a human in review)."""
    rows = []
    for s in sorted(segments, key=lambda x: x.start):
        txt = (getattr(s, "text", "") or "").strip()
        if not txt or len(txt) < config.TABLE_MIN_CHARS:
            continue
        rows.append({
            "sl": len(rows) + 1,
            "time": _mmss(s.start),
            "person": s.speaker,
            "conversation": txt,
            "language": getattr(s, "detected_language", None),
        })
    return {"file_id": file_id, "rows": rows}


def render_conversation_markdown(table: dict) -> str:
    lines = ["| Sl | Time | Person | Conversation |", "|--|--|--|--|"]
    for r in table["rows"]:
        conv = r["conversation"].replace("|", "\\|")
        lines.append(f"| {r['sl']} | {r['time']} | {r['person']} | {conv} |")
    return "\n".join(lines)


def _raw_text(segment) -> str:
    """Raw IndicConformer ASR for a segment, independent of any later mutation."""
    return ((segment.candidates or {}).get("indic_conformer") or {}).get("text", "") or ""


def _enhanced_text(segment) -> str:
    """LLM-corrected text if a correction was applied, else the raw ASR text.
    Always defined — falls back to raw when L6b did not run / failed / rejected."""
    llm = (segment.candidates or {}).get("llm_enhancement") or {}
    if llm.get("correction_status") == "corrected":
        return llm.get("corrected_text") or _raw_text(segment)
    return _raw_text(segment)


def build_raw(file_id, segments) -> list:
    """Full raw IndicConformer transcription, per segment (reads candidates,
    never seg.text). Independent of the LLM enhancement layer."""
    out = []
    for s in sorted(segments, key=lambda x: x.start):
        out.append({
            "segment_id": s.id, "speaker": s.speaker,
            "start": s.start, "end": s.end,
            "text": _raw_text(s),
            "language": s.detected_language, "confidence": s.confidence,
        })
    return out


def build_enhanced(file_id, segments) -> list:
    """LLM-corrected transcription, per segment. `text` falls back to the raw
    ASR text whenever enhancement did not apply; `correction_status` explains."""
    out = []
    for s in sorted(segments, key=lambda x: x.start):
        llm = (s.candidates or {}).get("llm_enhancement") or {}
        out.append({
            "segment_id": s.id, "speaker": s.speaker,
            "start": s.start, "end": s.end,
            "text": _enhanced_text(s),
            "language": s.detected_language, "confidence": s.confidence,
            "correction_status": llm.get("correction_status", "not_run"),
            "correction_confidence": llm.get("correction_confidence"),
            "original_text": llm.get("original_text"),
        })
    return out


def enrich_diarization(diar: dict, segments, *, use_enhanced: bool) -> dict:
    """Join the pyannote speaker timeline with per-turn text (raw or enhanced),
    matching each turn to the segment with the greatest time overlap. Speakers,
    timestamps and model_version are passed through unchanged."""
    seg_list = sorted(segments, key=lambda x: x.start)

    def _text_for(start, end):
        best, best_overlap = None, 0.0
        for s in seg_list:
            overlap = min(s.end, end) - max(s.start, start)
            if overlap > best_overlap:
                best_overlap, best = overlap, s
        if best is None or best_overlap <= 0.0:
            return ""
        return _enhanced_text(best) if use_enhanced else _raw_text(best)

    timeline = []
    for turn in (diar.get("timeline") or []):
        timeline.append({**turn, "text": _text_for(turn["start"], turn["end"])})

    return {
        "speakers": diar.get("speakers", []),
        "model_version": diar.get("model_version"),
        "timeline": timeline,
    }


def write_named(case_id, file_id, name, data) -> str:
    path = storage.derivative_path(case_id, file_id, "final", f"{file_id}_{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def write_named_text(case_id, file_id, name, ext, text) -> str:
    path = storage.derivative_path(case_id, file_id, "final", f"{file_id}_{name}.{ext}")
    with open(path, "w") as f:
        f.write(text)
    return path
