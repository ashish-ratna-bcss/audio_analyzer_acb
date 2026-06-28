# Verbatim Telugu ASR — Program Design + Sub-Project 1 (Evaluation Harness)

**Date:** 2026-06-28
**Branch:** IndicConformer
**Goal:** Drive the self-hosted pipeline toward verbatim-grade Telugu+English
transcription on noisy ACB forensic audio.

## Constraints (fixed)
- **Air-gapped**, single **24GB** GPU. No cloud APIs ever.
- Languages: Telugu + English code-mix (+ Hindi). Dravidian sandhi, code-switching.
- Audio: noisy, far-field, overlapping, variable volume (sting/court/phone).
- **Labeled data available**: evidence audio + human-verified transcripts.
- Current stack (~11.5GB VRAM): Whisper-large-v3 + vasista22 Telugu fine-tune +
  IndicConformer-600M, selected per-turn; glossary correction; pyannote diarization.

## Why fine-tuning, not a MoE swap
No zero-shot open model is verbatim on this acoustic domain (cloud SOTA ≈ 19% WER;
open models worse). MoE/LLM-ASR options don't fit the constraints:
- **Shrutam-2** (best MoE): ~24GB alone (can't co-reside) + non-commercial license.
- **Gemma 3n / Voxtral** (~12-16GB): fit, but zero-shot still not verbatim on ACB
  acoustics; script-collapse risk.
Every domain winner in the research (Vividh-ASR R-MFT, Vaani, IndicWhisper) wins by
**fine-tuning**. We have the data. That is the unlock.

## Program decomposition (dependency order)

1. **Evaluation harness** ← this spec. Measure WER / CER / Script-Fidelity-Rate /
   number+entity accuracy of pipeline output vs verified transcripts. Foundation:
   nothing else is provable without it.
2. **Training-data pipeline.** Audio + verified transcripts → cleaned, normalized,
   aligned train/val/test splits in fine-tuner format.
3. **Coverage fix + audio front-end** (independent). Emit uncovered whole-file
   Whisper speech as `Speaker_unknown`; gated denoise/loudnorm so quiet/noisy
   regions are detected. Fine-tuning can't help speech the pipeline drops.
4. **Domain fine-tune (LoRA/QLoRA + R-MFT curriculum).** Fine-tune the Telugu
   engine on ACB data (hard→clean reverse curriculum). Slots into the existing
   selector. The verbatim lever. Re-train as transcripts accumulate.
5. **(Optional) LLM-ASR engine eval.** Only if #4 plateaus: evaluate Gemma 3n E4B
   as an added selector engine, scored by harness #1.

Build order: **1 → (2 ∥ 3) → 4 → 5**. Each later sub-project gets its own spec.

---

# Sub-Project 1 — Evaluation Harness

## Purpose
Turn "I want assured/verbatim output" into measurable numbers, broken down per
file, per segment, and **per ASR engine** (so we see which engine wins where and
whether a change helps or regresses). Runs fully offline against verified
transcripts. This is the control loop for every later change.

## Success criteria
- Given a dataset of (audio, verified transcript) pairs, produce a report with:
  WER, CER, Script-Fidelity-Rate (SFR), number-accuracy, entity-accuracy —
  aggregate + per-file.
- Per-engine breakdown (whisper / telugu_whisper / indic_conformer) using the
  stored `source_pass`.
- Deterministic, repeatable; no network. Reuses existing deps (jiwer is present).
- A single command runs the whole dataset and writes JSON + human-readable report.

## Components (each isolated, unit-testable)

- `eval/normalize.py` — Indic-aware text normalization (pure functions):
  Unicode NFC; lowercase Latin; strip/normalize punctuation; collapse whitespace;
  optional sandhi-tolerant normalization. Used identically on reference + hypothesis
  before scoring so WER isn't inflated by cosmetic differences.
- `eval/metrics.py` — pure scoring functions:
  - `wer(ref, hyp)`, `cer(ref, hyp)` via jiwer on normalized text.
  - `script_fidelity_rate(hyp, expected_blocks)` — % of chars in allowed Unicode
    blocks (Telugu 0C00–0C7F + Latin + digits + punct). Detects script collapse.
  - `number_accuracy(ref, hyp)` — digit-run recall/precision (forensic: "302"≠"307").
  - `entity_accuracy(ref, hyp, glossary)` — domain entity preservation.
- `eval/reference_store.py` — load (audio_path, reference_text) pairs from a
  dataset dir; reference format = plain text or the conversation-table JSON we
  already emit (verified + corrected by a human).
- `eval/runner.py` — for each pair: obtain hypothesis (run the pipeline OR read a
  stored job result), normalize, score, collect per-engine subsets, aggregate.
- `eval/run.py` (CLI) — `python -m eval.run --dataset eval_data/ --out report.json`.

## Data flow
```
eval_data/<case>/audio.* + reference.txt|json
        │
        ├─► (pipeline run OR stored /result)  ─► hypothesis transcript (+ source_pass)
        │
   normalize(ref) , normalize(hyp)
        │
   metrics: WER, CER, SFR, number, entity   (overall + per-file + per-engine)
        │
   report.json  +  report.md  (sorted worst-first for triage)
```

## Dataset layout
```
eval_data/
  <case_id>/
    audio.wav|mp4|...
    reference.json     # [{start,end,speaker,text}] verified, OR
    reference.txt      # plain verified transcript (file-level)
```
Start with **file-level** scoring (concatenated reference vs hypothesis) — robust
when segment timestamps don't align. Add segment-level alignment later if needed.

## Report shape
```json
{
  "dataset": "eval_data", "files": N, "generated_at": "...",
  "aggregate": {"wer":0.x,"cer":0.x,"sfr":0.x,"number_acc":0.x,"entity_acc":0.x},
  "by_engine": {"telugu_whisper":{...}, "whisper":{...}, "indic_conformer":{...}},
  "per_file": [{"case_id":"...","wer":...,"cer":...,"sfr":...,
                "ref_excerpt":"...","hyp_excerpt":"..."}],
  "worst_files": ["case_x", ...]
}
```

## Error handling
- Missing reference / unreadable audio → skip with a recorded reason, never crash
  the run.
- Empty hypothesis → WER 1.0 (full miss), recorded.
- jiwer/normalization failure on a pair → recorded per-file error, run continues.

## Testing
- `metrics.py` — unit tests with known ref/hyp pairs: identical→0 WER; script
  collapse (Telugu→Devanagari)→low SFR; number swap (302→307)→number_acc penalty;
  code-mix entity preservation.
- `normalize.py` — NFC idempotence, punctuation/whitespace, Latin lowercasing,
  Telugu untouched semantically.
- `runner.py` — small fixture dataset (1-2 fake pairs + stubbed hypothesis) →
  asserts aggregate + per-engine breakdown shape.

## Out of scope (this sub-project)
- The fine-tuning itself (sub-project 4).
- Segment-level forced alignment (start file-level; revisit if needed).
- Any model/pipeline change — the harness only *measures*.

## Deliverable
An offline `eval/` module + CLI that, run against the verified ACB transcripts,
prints the current pipeline's WER/CER/SFR/number/entity scores with a per-engine
breakdown — the baseline we drive down through coverage fix → data pipeline →
fine-tuning.
