"""Score a dataset and aggregate — overall, per-file, and per ASR engine."""
from eval import metrics
from eval.normalize import normalize_text


def _mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


def _engine_text(segments):
    """Group hypothesis text by source_pass (whisper / telugu_whisper /
    indic_conformer) so we can score each engine's contribution separately."""
    by = {}
    for s in segments or []:
        eng = s.get("source_pass") or "unknown"
        txt = s.get("text_corrected") or s.get("text") or ""
        if txt:
            by.setdefault(eng, []).append(txt)
    return {k: " ".join(v).strip() for k, v in by.items()}


def evaluate(pairs, *, glossary=None) -> dict:
    """Score every pair; return aggregate + per_file + by_engine report dict."""
    per_file, errors = [], []
    engine_ref, engine_hyp = {}, {}      # accumulate per-engine ref/hyp text

    for p in pairs:
        if p.get("error"):
            errors.append({"case_id": p["case_id"], "error": p["error"]})
            continue
        try:
            sc = metrics.score_pair(p["reference"], p["hypothesis"], glossary=glossary)
        except Exception as exc:  # never let one bad pair kill the run
            errors.append({"case_id": p["case_id"], "error": f"score failed: {exc}"})
            continue
        per_file.append({
            "case_id": p["case_id"], **sc,
            "ref_excerpt": normalize_text(p["reference"])[:120],
            "hyp_excerpt": normalize_text(p["hypothesis"])[:120],
        })
        # Per-engine: compare each engine's concatenated hyp to the full reference.
        for eng, htext in _engine_text(p.get("segments")).items():
            engine_ref.setdefault(eng, []).append(p["reference"])
            engine_hyp.setdefault(eng, []).append(htext)

    aggregate = {
        m: _mean([f[m] for f in per_file])
        for m in ("wer", "cer", "sfr", "number_acc", "entity_acc")
    }
    by_engine = {}
    for eng in engine_hyp:
        ref = " ".join(engine_ref[eng])
        hyp = " ".join(engine_hyp[eng])
        by_engine[eng] = {**metrics.score_pair(ref, hyp, glossary=glossary),
                          "segments": sum(1 for x in engine_hyp[eng] if x)}

    per_file.sort(key=lambda f: f["wer"], reverse=True)   # worst first for triage
    return {
        "files_scored": len(per_file),
        "files_errored": len(errors),
        "aggregate": aggregate,
        "by_engine": by_engine,
        "per_file": per_file,
        "errors": errors,
        "worst_files": [f["case_id"] for f in per_file[:10]],
    }
