"""Load an evaluation dataset of (reference, hypothesis) pairs from disk.

Layout (one dir per case):
    eval_data/<case>/reference.json   # [{"text":..., "speaker":..., ...}] OR
    eval_data/<case>/reference.txt    # plain verified transcript
    eval_data/<case>/result.json      # our API /jobs/{id}/result output (hypothesis)
                                      #   OR hypothesis.txt (plain)

reference.json may be a list of segment dicts (with "text") or the
conversation_table shape ({"rows":[{"conversation":...}]}). The hypothesis is
taken from result.json (transcript.segments, carrying source_pass for the
per-engine breakdown) or a plain hypothesis.txt.
"""
import json
import os


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _reference_text(case_dir: str) -> str | None:
    jp, tp = os.path.join(case_dir, "reference.json"), os.path.join(case_dir, "reference.txt")
    if os.path.exists(jp):
        data = json.load(open(jp, encoding="utf-8"))
        rows = data.get("rows") if isinstance(data, dict) else data
        if isinstance(data, dict) and "segments" in data:
            rows = data["segments"]
        parts = [(r.get("text") or r.get("conversation") or "") for r in (rows or [])]
        return " ".join(p for p in parts if p).strip()
    if os.path.exists(tp):
        return _read_text(tp).strip()
    return None


def _hypothesis(case_dir: str):
    """Return (text, segments) where segments carry source_pass when available."""
    rp = os.path.join(case_dir, "result.json")
    if os.path.exists(rp):
        d = json.load(open(rp, encoding="utf-8"))
        segs = (d.get("transcript") or {}).get("segments") or []
        # prefer glossary-corrected text when present, else raw
        text = " ".join((s.get("text_corrected") or s.get("text") or "") for s in segs).strip()
        return text, segs
    hp = os.path.join(case_dir, "hypothesis.txt")
    if os.path.exists(hp):
        return _read_text(hp).strip(), []
    return None, []


def load_dataset(root: str) -> list:
    """Yield pairs: {case_id, reference, hypothesis, segments, error}. A pair with
    a missing reference or hypothesis is included with an `error` so the run can
    report it rather than silently skipping."""
    pairs = []
    if not os.path.isdir(root):
        return pairs
    for case_id in sorted(os.listdir(root)):
        case_dir = os.path.join(root, case_id)
        if not os.path.isdir(case_dir):
            continue
        ref = _reference_text(case_dir)
        hyp, segs = _hypothesis(case_dir)
        err = None
        if ref is None:
            err = "missing reference"
        elif hyp is None:
            err = "missing hypothesis"
        pairs.append({"case_id": case_id, "reference": ref or "",
                      "hypothesis": hyp or "", "segments": segs, "error": err})
    return pairs
