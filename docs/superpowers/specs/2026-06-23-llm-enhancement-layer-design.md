# LLM Transcript Enhancement Layer — Feasibility & Final Design

Date: 2026-06-23
Status: DRAFT — awaiting approval before implementation
Branch (proposed): `feat/llm-enhancement-layer` (off current `IndicConformer`)

---

## 1. Objective

Add a multilingual LLM enhancement layer that corrects ASR character/spelling
errors on existing transcript segments, without touching ASR, diarization,
timestamps, speaker labels, segmentation, or any existing API response.

The job result must expose **four new fields** alongside (not replacing) the
existing payload:

| Field | Content |
|-------|---------|
| `raw` | full raw IndicConformer transcription (per segment) |
| `raw_diarization` | pyannote speaker timeline + raw text |
| `enhanced` | LLM-corrected transcription (per segment) |
| `enhanced_diarization` | pyannote speaker timeline + enhanced text |

`enhanced` is **always present**. When the LLM is disabled, down, times out, or
a correction fails a guard, `enhanced` falls back to the raw text and a
`correction_status` explains why. The pipeline behaves exactly as it does today
when the LLM is unavailable.

---

## 2. Feasibility — VERDICT: FEASIBLE, LOW RISK

Verified against current codebase (`IndicConformer` branch):

- **Clean insertion point.** `pipeline/tasks.py::run_pipeline` runs
  L0→L1→L2→L2b→L3→L4→L5→L6→L8. A new stage slots between L6 (confidence report,
  `tasks.py:582`) and L8 (output, `tasks.py:592`).
- **No schema change.** `Segment.candidates` is already a JSON column
  (`db/models.py:71`). Enhancement output is stored under a new key
  `candidates["llm_enhancement"]`. No Alembic migration required.
- **Existing outputs untouched.** `transcript`, `diarization`,
  `conversation_table` (`api/routes/cases.py:69-75`), `/review/*`
  (`api/routes/review.py`), `certified_transcript.json`, `indic_transcript.json`,
  `validation_report.json`, `conversation_table.{json,md}` all read
  `seg.text` / `candidates["indic_conformer"]`. The enhancement layer **never
  writes `seg.text`**, so every existing response is byte-identical.
- **Raw ASR already preserved forever.** `_emit_segment` (`tasks.py:364-371`)
  persists `candidates["indic_conformer"]["text"]` (plus `enh_text`/`org_text`).
  `raw` reads from there; it is independent of any later mutation.
- **No new laptop dependency.** Ollama is called via stdlib `urllib.request`.
  `anthropic` (optional provider) is lazy-imported inside the call function,
  matching the existing "heavy libs lazy-imported, mocked on laptop" convention
  (`tests/conftest.py::_stub_models`).

### Risk register

| Risk | Mitigation |
|------|------------|
| LLM hallucination / rewriting (forensic-fatal) | Strict prompt + deterministic post-guards (word-count, script, number, empty) + raw always retained in `candidates["indic_conformer"]` |
| LLM provider down (Ollama/Anthropic) | Service never raises → `correction_status="error"` → `enhanced`=raw. Pipeline completes normally. |
| Latency (N segments × LLM call) | Per-segment timeout `LLM_ENHANCEMENT_TIMEOUT_S`; optional Anthropic Batches in a later phase; stage is skippable via `LLM_ENHANCEMENT_ENABLED=false` |
| Existing tests break | Additive only; `seg.text` untouched; new stage no-ops when disabled (test default) |

---

## 3. Architecture

```
L0 → L1 → L2 → L2b → L3 → L4 → L5 → L6 → [L6b: LLM enhancement] → L8
                                          (writes candidates only,
                                           NEVER seg.text)
```

`L6b` naming follows the existing `L2b` convention (`L7` is already used by the
review audit stage label `L7_review`, so it is avoided).

Data flow for the four response fields (all assembled in `_build_result`):

```
seg.candidates["indic_conformer"]["text"]      ─┬─► raw[].text
                                                └─► raw_diarization.timeline[].text
seg.candidates["llm_enhancement"]["corrected_text"]
   (|| fallback to indic_conformer text)        ─┬─► enhanced[].text
                                                 └─► enhanced_diarization.timeline[].text
pyannote timeline (diarization artifact JSON)   ────► both *_diarization speakers/turns
```

`seg.text` (= raw ASR, suppression applied) continues to feed the **existing**
`transcript` / `conversation_table` exactly as today.

---

## 4. Response Shape (`GET /jobs/{id}/result` and the by-file variant)

```jsonc
{
  "job_id": "...", "case_id": "...", "file_id": "...",
  "status": "...", "source_hash_sha256": "...",

  // ── EXISTING — unchanged, byte-identical ──
  "transcript": { ... },
  "diarization": { "speakers": [...], "timeline": [...], "model_version": "..." },
  "conversation_table": { "rows": [...] },

  // ── NEW ──
  "raw": [
    { "segment_id": "uuid", "speaker": "Speaker_1", "start": 25.36, "end": 26.10,
      "text": "నాకు ఈ రోజు హైదరాబాదు వెలాలి", "language": "te", "confidence": 0.61 }
  ],
  "raw_diarization": {
    "speakers": ["Speaker_1","Speaker_2"],
    "model_version": "pyannote/speaker-diarization-3.1",
    "timeline": [ { "start": 25.36, "end": 26.10, "speaker": "Speaker_1",
                    "text": "నాకు ఈ రోజు హైదరాబాదు వెలాలి" } ]
  },
  "enhanced": [
    { "segment_id": "uuid", "speaker": "Speaker_1", "start": 25.36, "end": 26.10,
      "text": "నాకు ఈ రోజు హైదరాబాద్ వెళ్లాలి", "language": "te", "confidence": 0.61,
      "correction_status": "corrected", "correction_confidence": 0.94 }
  ],
  "enhanced_diarization": {
    "speakers": ["Speaker_1","Speaker_2"],
    "model_version": "pyannote/speaker-diarization-3.1",
    "timeline": [ { "start": 25.36, "end": 26.10, "speaker": "Speaker_1",
                    "text": "నాకు ఈ రోజు హైదరాబాద్ వెళ్లాలి" } ]
  }
}
```

### `correction_status` values

| Status | Meaning | `enhanced.text` |
|--------|---------|-----------------|
| `corrected` | LLM ran, applied a correction | LLM output |
| `unchanged` | LLM ran, no change needed | raw text |
| `skipped` | segment confidence below attempt threshold | raw text |
| `guard_rejected` | LLM output failed a deterministic guard | raw text |
| `error` | provider down / timeout / exception | raw text |
| `not_run` | `LLM_ENHANCEMENT_ENABLED=false` | raw text |

---

## 5. Enhancement Service (`services/llm_enhancement_service.py`)

Pure-orchestration module; heavy/network calls isolated and guarded.

```python
def enhance_segment(segment: dict) -> dict:
    """Always returns a correction record. Never raises."""
    original = segment.get("text", "")
    unchanged = {"correction_status": "unchanged", "correction_confidence": 1.0,
                 "original_text": original, "corrected_text": original, "changes": []}

    if not config.LLM_ENHANCEMENT_ENABLED:
        return {**unchanged, "correction_status": "not_run"}
    if not original.strip():
        return {**unchanged, "correction_status": "skipped"}
    if (segment.get("confidence") or 0.0) < config.LLM_ENHANCEMENT_MIN_CONF:
        return {**unchanged, "correction_status": "skipped"}

    try:
        result = _call_llm(segment)                 # provider-dispatched, timeout-bounded
        if result.get("correction_status") != "corrected":
            return {**unchanged, **result}
        if result.get("correction_confidence", 0.0) < config.LLM_CORRECTION_MIN_CONF:
            return {**unchanged, "correction_status": "unchanged"}
        if not _passes_guards(original, result["corrected_text"], segment):
            return {**unchanged, "correction_status": "guard_rejected"}
        return result
    except Exception:
        return {**unchanged, "correction_status": "error"}
```

### Deterministic post-guards (`_passes_guards`)

Second line of defense independent of the prompt:

1. **Word-count guard** — reject if
   `abs(len(corr.split()) - len(orig.split())) / max(len(orig.split()),1) > LLM_MAX_WORD_DELTA_RATIO`.
2. **Script guard** — the dominant Unicode block of `corr` must match `orig`
   (no Telugu→Latin translation, etc.).
3. **Number guard** — every maximal digit run in `orig` must appear in `corr`.
4. **Empty guard** — reject if `corr` empty while `orig` non-empty.
5. **Overlap guard** — if `segment["overlap"]` is true, tighten
   `LLM_MAX_WORD_DELTA_RATIO` to ~0 (only single-token spelling fixes survive).

### Providers

- `ollama` (default, fully local/air-gappable): POST to
  `{LLM_OLLAMA_URL}/api/generate` via stdlib `urllib.request`, `format=json`.
- `anthropic` (optional, higher quality): lazy `import anthropic`, model
  `LLM_MODEL` (default `claude-haiku-4-5-20251001`).

Both return the structured JSON correction record. Prompt enforces the 14 rules
+ hard safety constraints from the requirement (correct-only, no add/remove/
translate/paraphrase/summarize, preserve language/numbers/meaning, JSON-only,
uncertainty→unchanged, overlap→extreme conservatism).

---

## 6. Pipeline Integration (`pipeline/tasks.py`)

```python
def _l6b_enhance(job, session):
    segs = repo.list_segments(session, job.file_id)
    corrected = 0
    for seg in segs:
        rec = llm_enhancement_service.enhance_segment({
            "segment_id": seg.id, "speaker": seg.speaker,
            "start": seg.start, "end": seg.end,
            "text": seg.text, "language": seg.detected_language,
            "confidence": seg.confidence, "overlap": "+" in (seg.speaker or ""),
        })
        cands = dict(seg.candidates or {})
        cands["llm_enhancement"] = rec          # NOTE: seg.text is NOT modified
        seg.candidates = cands
        corrected += 1 if rec["correction_status"] == "corrected" else 0
    session.commit()
    au.append_entry(job.case_id, file_id=job.file_id, stage="L6b",
                    parameters={"corrected": corrected, "total": len(segs)},
                    session=session)
    session.commit()
```

Inserted in the existing L4/L5/L6 try-block, right after
`_write_confidence_report(...)` (`tasks.py:583`):

```python
repo.update_job(s, job_id, stage="L6b"); s.commit(); stage("L6b")
_l6b_enhance(job, s)
```

`STAGE_PROGRESS`: add `"L6b": 90` (L6=85, L8=95 unchanged).

Failure isolation: `_l6b_enhance` cannot raise (service swallows all). Even so
it lives inside the existing try-block that already fails the job safely.

---

## 7. Transcript Builders (`services/transcript_service.py`)

Three additive pure functions (no change to existing functions):

- `build_raw(file_id, segments) -> list` — reads
  `candidates["indic_conformer"]["text"]`.
- `build_enhanced(file_id, segments) -> list` — reads
  `candidates["llm_enhancement"]["corrected_text"]`, falls back to raw text;
  carries `correction_status` + `correction_confidence`.
- `enrich_diarization(diar, segments, *, use_enhanced) -> dict` — joins the
  pyannote timeline turns to per-turn text by max-overlap segment match.

---

## 8. API Change (`api/routes/cases.py`)

Single edit: `_build_result` return dict gains four keys at the end. The
diarization artifact is already loaded there (`cases.py:63-67`), so no extra IO.

```python
    return {
        ...                                   # existing keys unchanged
        "transcript": transcript,
        "diarization": diar,
        "conversation_table": table,
        # new
        "raw":                  ts.build_raw(file_id, segs),
        "raw_diarization":      ts.enrich_diarization(diar, segs, use_enhanced=False),
        "enhanced":             ts.build_enhanced(file_id, segs),
        "enhanced_diarization": ts.enrich_diarization(diar, segs, use_enhanced=True),
    }
```

No new endpoint. No change to `/jobs/{id}`, `/review/*`, `/cases`, `certify`,
webhooks, or the L8 file outputs.

---

## 9. Config Additions (`config.py`)

```python
LLM_ENHANCEMENT_ENABLED  = os.getenv("LLM_ENHANCEMENT_ENABLED", "false").lower() == "true"
LLM_PROVIDER             = os.getenv("LLM_PROVIDER", "ollama")          # ollama | anthropic
LLM_MODEL                = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_OLLAMA_URL           = os.getenv("LLM_OLLAMA_URL", "http://localhost:11434")
LLM_ENHANCEMENT_MIN_CONF = float(os.getenv("LLM_ENHANCEMENT_MIN_CONF", "0.0"))  # attempt all by default
LLM_CORRECTION_MIN_CONF  = float(os.getenv("LLM_CORRECTION_MIN_CONF", "0.85"))  # apply only confident corrections
LLM_MAX_WORD_DELTA_RATIO = float(os.getenv("LLM_MAX_WORD_DELTA_RATIO", "0.15"))
LLM_ENHANCEMENT_TIMEOUT_S= int(os.getenv("LLM_ENHANCEMENT_TIMEOUT_S", "30"))
```

Default `ENABLED=false` ⇒ existing behavior is the default everywhere (laptop,
tests, and any deploy that does not opt in). `anthropic` package added to
requirements as deploy-only (commented/optional), like other model deps.

---

## 10. Tests

- `tests/test_llm_enhancement_service.py` (new): guards (word-count, script,
  number, empty, overlap), each `correction_status` path, provider-down →
  `error`, disabled → `not_run`. LLM call mocked; no network.
- `tests/conftest.py`: add `llm_enhancement_service.enhance_segment` to the
  `_stub_models` autouse stub returning `{"correction_status":"not_run",...}` so
  pipeline-driving tests stay model-free and offline.
- `tests/test_transcript_service.py` (extend): `build_raw`, `build_enhanced`
  fallback when no `llm_enhancement` key, `enrich_diarization` overlap join.
- `tests/test_api_jobs.py` (extend): result payload contains the four new keys
  and `enhanced` mirrors `raw` when enhancement disabled.
- Regression: full existing suite must stay green (pre-existing
  `test_alignment_service` failure excluded, per project memory).

---

## 11. Implementation Phases

1. **Service + config + tests** — `llm_enhancement_service.py`, `config.py`,
   `test_llm_enhancement_service.py`, conftest stub. (No pipeline wiring yet.)
2. **Transcript builders** — `build_raw`, `build_enhanced`,
   `enrich_diarization` + tests.
3. **Pipeline + API** — `_l6b_enhance` stage, `STAGE_PROGRESS`, `_build_result`
   four keys + api test.
4. **(Optional, later)** Anthropic Batches for throughput; queryable
   `llm_enhanced` column via a migration if reporting needs it.

---

## 12. Non-Goals / Guarantees

- Does NOT modify `seg.text`, timestamps, speakers, segment boundaries.
- Does NOT change ASR, diarization, VAD, separation, or any L0–L6 / L8 logic.
- Does NOT alter existing response fields or add/remove endpoints.
- Does NOT translate, summarize, paraphrase, or invent content (prompt + guards).
- Raw ASR is always retrievable; enhancement is purely additive and reversible.
