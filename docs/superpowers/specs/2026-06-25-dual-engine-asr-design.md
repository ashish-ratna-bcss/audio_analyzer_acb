# Dual-Engine ASR with Output-Content Selection

**Date:** 2026-06-25
**Branch:** IndicConformer
**Status:** Approved — implementing

## Problem

IndicConformer-600M alone produces low-accuracy output on code-mixed Telugu/English
forensic call audio: foreign-script garbage from LID misroutes (mitigated by
`ALLOWED_LANGS=te,en,hi`), transliterated English instead of recognised English
words, garbled numbers, and no punctuation. The reference (frontier-model)
transcript shows fluent Telugu+English code-mix with spelled-out English entities
("mPokket", "CIBIL"), correct numbers ("1100", "68%"), and punctuation.

IndicConformer is strong only on pure native Telugu. Whisper-large-v3 is strong on
code-mix, English, Hindi/other languages, numbers, and punctuation. Combine them.

## Approach

Dual-run both engines per clip, then select the winner by inspecting the actual
output content (not a noisy LID guess). No new pipeline stages — all changes land
in the existing L5 ASR step.

### Data flow (per clip)

```
clip (enhanced + raw) ──┬─► IndicConformer (routing_lang)  → text_indic   [existing]
                        └─► Whisper-large-v3 (forced lang) → text_whisper  [restored]
                                          │
                                  selector(indic, whisper, lid, signals)
                                          │
                              winning text + source + confidence + flag
```

MMS-LID runs first as today: per-clip language + file prior. It feeds both the
routing_lang for IndicConformer and the forced language for Whisper.

## Components

| Unit | Responsibility | Depends on |
|------|---------------|-----------|
| `services/whisper_service.py` (restore) | Transcribe one clip, forced language, forensic decode params (beam_size=10, condition_on_previous_text=False, temperature=0, repetition_penalty=1.3). Returns text + confidence (`exp(avg_logprob)`) + no_speech_prob + compression_ratio | faster-whisper |
| `services/asr_selector.py` (new) | Pure function: (indic_text, whisper_text, lid, signals) → (winner_text, source, confidence, agreement, reason, flag). No I/O — fully unit-testable | — |
| `run_asr()` in `pipeline/tasks.py` (rename `run_indic`) | Orchestrate: run both engines on clip, call selector, return unified record | both services + selector |
| `services/cross_model.py` (reuse) | LaBSE similarity for pure-Telugu agreement | embedding_service |

## Selection logic

```
1. has_latin_or_digit(whisper_text)            → Whisper wins (code-mix / numbers / entities)
2. both pure-Telugu script:
     LaBSE similarity >= AGREEMENT_MIN          → IndicConformer wins (sharper native script)
     similarity <  AGREEMENT_MIN                → keep Whisper, FLAG (engines disagree)
3. IndicConformer abstained (unsupported lang)  → Whisper wins
4. Whisper empty / high no_speech_prob          → IndicConformer fallback
```

Stored confidence: Whisper `exp(avg_logprob)` when Whisper wins; LaBSE agreement
when IndicConformer wins.

## Storage / response (additive)

Each segment's `candidates` stores both `pass1_whisper` and `pass2_indic_conformer`
(text + confidence + signals), plus chosen `source`
("whisper" | "indic_enhanced" | "indic_original"), `agreement`, `selection_reason`.
`seg.text` = winner. Existing `build_validation_report` / `enrich_diarization`
already read these. API response shape unchanged.

## Config

```python
WHISPER_MODEL = "large-v3"      # already present
ASR_DUAL_ENGINE = true          # new toggle; false = indic-only (current behaviour)
AGREEMENT_MIN = 0.6             # already present, reused for selector
```

## Testing

- `asr_selector` — pure unit tests: code-mix→Whisper, pure-Telugu-agree→Indic,
  pure-Telugu-disagree→flag, abstain→Whisper, empty→fallback.
- `whisper_service` — restore old tests, mock faster-whisper.
- Integration — one dual-run on a BVR clip: assert no foreign scripts, English +
  numbers present.

## Out of scope

- LLM enhancement (L6b) — separate layer, stays removed.
- VRAM: Whisper ~3GB + Indic ~2.5GB + LID ~1GB + pyannote ~1GB fit 24GB; all resident.
- Latency: dual-run ~doubles L5; both fast on GPU, acceptable.
