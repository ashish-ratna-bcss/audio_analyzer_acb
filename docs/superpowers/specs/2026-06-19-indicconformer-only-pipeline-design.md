# IndicConformer-Only Pipeline + Sortformer Diarization Pilot

Date: 2026-06-19
Branch: `feat/forensic-v2-pipeline`
Status: approved design

## Goal

Simplify the forensic ASR pipeline to a **single ASR model — AI4Bharat IndicConformer-600M** —
removing Whisper and SeamlessM4T, while preserving forensic defensibility (confidence +
flagging) that the dropped cross-model validation previously provided. Separately, **pilot
NVIDIA Sortformer diarization against tuned pyannote** and keep whichever scores better on the
ground-truth table. Emit a court-ready MM.SS Time/Person/Conversation table.

Decisions (user-approved):
- **ASR**: IndicConformer only. 22 Indic languages + English transcribed; non-Indic stretches
  **abstain + flag** (no Whisper fallback).
- **Diarization**: pilot Sortformer vs tuned-pyannote, measure on ground truth, keep winner.

This is decomposed into two phases, each independently shippable:
- **Phase 1** — IndicConformer-only ASR refactor (this is the load-bearing change).
- **Phase 2** — Sortformer diarization pilot + evaluation harness.

---

## Phase 1 — IndicConformer-only ASR

### What is removed
- `services/whisper_service.py`, `services/seamless_service.py` — no longer called by the pipeline
  (kept on disk but unwired; `run_models` stops importing them).
- 3-way `cross_model.compare_passes` consensus — there is only one model now, so "consensus"
  is replaced by single-model confidence + a self-cross-check.
- Per-model output files for whisper/seamless. Only the IndicConformer transcript + validation
  remain.

### What replaces the dropped validation (forensic confidence)

IndicConformer's `forward(wav, lang, 'ctc')` returns text only (logprobs computed internally,
not exposed). So confidence comes from two signals:

1. **Dual-run self-cross-check (primary).** Transcribe the SAME unit twice with IndicConformer:
   once on the **enhanced** clip, once on the **original** clip. Agreement between the two
   (LaBSE cosine via `embedding_service.similarity`, plus normalized length ratio) is the
   confidence proxy:
   - high agreement → high confidence, `auto_accept`;
   - low agreement → enhancement changed the words → flag `enh_orig_divergence` for review.
   This catches both over-suppression artifacts and unstable decodes — the same protective role
   the old enhanced-vs-original Whisper cross-check played, now single-model.
2. **CTC mean-logprob (best-effort).** In `indic_asr_service`, attempt to obtain the greedy-path
   mean log-probability if the loaded model object exposes the encoder/logprobs; if not
   reachable, `ctc_confidence = None` and the dual-run agreement is authoritative. Never fabricate.

Final `confidence` = CTC mean-logprob when available, else the dual-run agreement score.

### Language routing (kept + hardened)
- `lang_id_service` MMS-LID + `vote_file_language` file prior stay.
- `run_indic` routes language: clip MMS-LID top1 if `conf ≥ LID_VOTE_MIN_CONF` and
  (`ALLOWED_LANGS` empty or in set), else file prior, else — since there is no Whisper
  self-detect anymore — the file prior, else `None`.
- If routed language ∉ `_INDIC_SUPPORTED` → **abstain** (empty + flag `non_indic_abstain`).
- `ALLOWED_LANGS` becomes the primary misroute guard. Default stays open; deployments set it
  (e.g. `te,hi,en`).

### Pipeline changes (`pipeline/tasks.py`)
- Replace `run_models` with `run_indic(clean_clip, raw_clip, *, file_prior)`:
  - MMS-LID route as above;
  - `i_enh = indic_asr_service.transcribe_clip(clean_clip, lang)`;
  - `i_org = indic_asr_service.transcribe_clip(raw_clip, lang)`;
  - both through `hallucination_filter.filter_pass` (no_speech n/a — Indic has none — so
    repetition + ghost only);
  - compute agreement + confidence; choose `i_enh` text as primary (enhanced is cleaner),
    fall back to `i_org` if `i_enh` empty.
  - returns `{lang_id, text, confidence, ctc_confidence, agreement, abstained, source}`.
- `_emit_segment` simplified: one candidate (`indic`) + `enh`/`org` texts + agreement.
  DB `source_pass="indic_conformer"`, `text` = chosen text, `confidence` as above.
  Flag when: abstained, OR `enh_orig_divergence` (agreement < `INDIC_SELFCHECK_MIN`), OR
  confidence < `INDIC_CONF_MIN`, OR repetition/ghost.
- Overlap path still uses SepFormer stems → each stem `run_indic(stem, stem, ...)`.

### `cross_model.py`
Keep `normalized_edit_distance`; add `selfcheck_confidence(enh_text, org_text, *, embed_fn) ->
{agreement, confidence}` (length-ratio gated cosine). The 3-way `compare_passes` is retired from
the pipeline (left in file/tests but unused by tasks).

### Output (L8) — `transcript_service.py`
- `final/{file}_indic_transcript.json` — the transcript (segments: text, confidence, language,
  agreement, flags).
- `final/{file}_validation_report.json` — per segment `{enh_text, org_text, agreement,
  ctc_confidence, flags}` + summary.
- `final/{file}_certified_transcript.json` — same content, drives review/certify.
- **NEW** `final/{file}_conversation_table.{json,md}` — court format:
  `Sl | Time(MM.SS) | Person | Conversation`. `build_conversation_table(file_id, segments)`.
  Person = diarization speaker label (human renames A.O/Complt in review).
- Drop whisper/seamless file emission.

### config additions
```
INDIC_SELFCHECK_MIN = float(os.getenv("INDIC_SELFCHECK_MIN", "0.6"))  # enh-vs-org agreement floor
INDIC_CONF_MIN      = float(os.getenv("INDIC_CONF_MIN", "0.5"))       # confidence flag floor
# ALLOWED_LANGS already exists; document setting te,hi,en for ACB cases.
```

### Phase 1 testing (laptop, mocked model)
- `tests/test_cross_model.py` — add `selfcheck_confidence` cases (identical→high, divergent→low,
  empty→flag).
- `tests/test_indic_abstain.py` — already covers abstain; add CTC-confidence-None path.
- `tests/test_pipeline_attribution.py` — replace `run_models` tests with `run_indic`: dual-run
  called twice, divergence flags, abstain on non-Indic, single candidate persisted,
  `source_pass=="indic_conformer"`.
- `tests/test_transcript_outputs.py` — add `build_conversation_table` shape (MM.SS formatting,
  numbering, speaker column).
- Whole suite green.

---

## Phase 2 — Sortformer diarization pilot (separate plan)

### Intent
Evaluate NVIDIA Sortformer (`nvidia/diar_sortformer_4spk-v1`, end-to-end, overlap-native, up to 4
speakers) against tuned pyannote 3.1, choose the better diarizer on the actual ACB ground truth.

### Approach (behind a flag — no blind replacement)
- `config.DIARIZER = os.getenv("DIARIZER", "pyannote")` — `pyannote | sortformer`.
- New `services/sortformer_service.py` exposing the same contract pyannote does:
  `diarize_with_overlap(path) -> [{start, end, speaker}]`. Loads via NeMo
  (`from nemo.collections.asr.models import SortformerEncLabelModel`). NeMo added to a SEPARATE
  optional requirements layer to keep the base image lean and avoid version clashes with the
  existing torch/transformers pins.
- `diarization_service.diarize_with_overlap` dispatches on `config.DIARIZER`.

### Tuning pyannote (regardless of pilot outcome)
- Pass `num_speakers` hint when known; expose `DIARIZATION_MIN_DURATION_OFF` (already present).
- These alone may close the gap and avoid the NeMo dependency.

### Evaluation harness
- `tools/diar_eval.py`: given a reference table (your ground-truth MM.SS table → CSV) and a
  produced speaker timeline, compute DER + turn-boundary F1 + speaker-count accuracy.
- Run both diarizers on the BVR test clip (05:28–06:33 window), score against the ground truth,
  record results in `docs/superpowers/diar_pilot_results.md`. Keep the winner as default.

### Risk / gating
- NeMo is heavy and clashed historically (the repo avoided it). Disk on the server is tight —
  build Sortformer image change only after `docker system prune` headroom is confirmed.
- If Sortformer integration cost or DER gain is not worth it, ship tuned-pyannote and shelve
  Sortformer. Decision recorded in the pilot results doc.

Phase 2 gets its own implementation plan after Phase 1 ships and is validated on the server.

---

## Data flow after Phase 1

```
L0 ingest -> L1 normalize(48k,16k) -> L2 DeepFilterNet3 -> L3 VAD union
-> L4 diarize (pyannote now; Sortformer-or-pyannote after Phase 2)
-> L5:
     build units (+gap) -> preprocess_service.prepare_clip -> {clean, raw}
     MMS-LID pre-sweep -> file_prior
     per unit: run_indic(clean, raw, file_prior)
        = MMS-LID route -> IndicConformer(clean) + IndicConformer(raw)
          -> hallucination_filter -> selfcheck_confidence -> (text, confidence, flags)
     _emit_segment (single candidate)
-> L6 confidence report
-> L8 outputs: indic_transcript.json + validation_report.json
   + certified_transcript.json + conversation_table.{json,md}
```

## Robustness / 0-fail
- One model call wrapped per run; failure → empty flagged segment, never aborts job.
- Abstain explicit on non-Indic; no hidden fallback.
- loudnorm/trim non-fatal (Phase-0 already).
- embedding failure in self-check → agreement 0.0 + flag (degrade, not crash).
- Reconcile units==segments preserved. WORM/audit/manifest untouched.

## Out of scope
- No new ASR models beyond IndicConformer (and the shelved Whisper/Seamless code stays but unwired).
- No translation changes. No UI. No custody-chain changes.
- Phase 2 NeMo/Sortformer only behind its flag; not a Phase-1 dependency.
