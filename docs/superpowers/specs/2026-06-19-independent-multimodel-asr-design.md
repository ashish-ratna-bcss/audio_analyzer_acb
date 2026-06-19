# Independent Multi-Model ASR + Robust Preprocessing + Cross-Model Validation

Date: 2026-06-19
Branch: `feat/forensic-v2-pipeline`
Status: approved design

## Goal

Rework the L2/L3/L5/L8 stages of the forensic pipeline so that:

1. All three ASR models (Whisper large-v3, IndicConformer-600M, SeamlessM4T-v2) run
   **truly independently** on the **same robustly-preprocessed clip**.
2. Each model emits its **own independent transcript file**.
3. The three outputs are **validated together** with a multilingual-embedding consensus
   that is script-agnostic.
4. The known defects that degrade output quality are fixed.

This is an accuracy-first forensic pipeline: recall is preserved, nothing inside the
VAD union is dropped, and everything uncertain is flagged for human certification.

## Defects being fixed

| ID | Defect | Location | Fix |
|----|--------|----------|-----|
| D1 | `source_pass="pass1_whisper"` hardcoded; `winning = p1 or p2 or p3` (whisper always wins) | `pipeline/tasks.py:376,380` | Consensus-driven winner; `source_pass` reflects the actual chosen pass |
| D2 | `no_speech_prob` captured but discarded; ghost phrases leak ("Thank you", "[Music]", "ご視聴…") | `pipeline/tasks.py:152` (`_whisper_clip`) | New `hallucination_filter`: no-speech gate + ghost-phrase blocklist + existing repetition detector |
| D3 | Cross-model disagreement uses char edit-distance → always fires across scripts (Latin vs Telugu) | `services/cross_model.py` | Replace with LaBSE multilingual embedding cosine across all 3 passes |
| D4 | Unconstrained per-clip LID → wild misroute (Korean/Shona on a Telugu call) | `pipeline/tasks.py:303` (`_three_pass_asr`) | File-level majority-vote prior + confidence gate; clip LID only trusted above gate |
| D5 | IndicConformer confidence hardcoded `0.75`; silent fallback to Whisper breaks independence | `services/indic_asr_service.py:71,90` | Real/derived confidence or `null`; on unsupported language **abstain** (empty + flag), never masquerade as Whisper |

## Design decisions (user-approved)

- **Output**: three separate per-model transcript files + one validation report. (No
  single merged "winning" transcript is the deliverable; consensus is still computed and
  persisted to drive the existing review/certify flow.)
- **Preprocessing**: one uniform robust clip fed identically to all three models.
- **Language**: open auto-detect retained, tamed by per-file majority vote + confidence gate.
- **Validation**: multilingual sentence-embedding (LaBSE, already configured) consensus.

## Components

### 1. `services/preprocess_service.py` (new)

`prepare_clip(enhanced_src, original_src, start, end, workdir, idx, speaker) -> dict`

Per transcription unit, produce one clean clip used by all three models:

1. Cut `[start, end]` from the **enhanced** (DeepFilterNet3) full-file audio.
2. EBU R128 loudness normalization (ffmpeg `loudnorm`, target -16 LUFS) — replaces the
   current peak-only `CLIP_NORMALIZE`. Falls back to peak normalize if `loudnorm` fails.
3. VAD-trim leading/trailing silence only (never cut interior speech) using existing
   `vad_service`. If trimming would empty the clip, keep the untrimmed clip.

Returns `{"clean": clean_wav_path, "raw": raw_org_wav_path}`. The raw original cut is
always retained for audit (no preprocessing), written via existing `clip_service.cut`.

All three models consume `clean`. The previous behavior (Seamless on original, others on
enhanced) is removed — uniform input.

### 2. Language voting — `services/lang_id_service.py` additions + L5 pre-sweep

- New `vote_file_language(per_clip_lids, allowed_langs, min_conf) -> str|None`: majority
  vote over per-clip MMS-LID top1 codes whose confidence ≥ `min_conf` and (if
  `allowed_langs` non-empty) within the allowed set. Returns ISO 639-1 prior or `None`.
- In `_l5_l6_segments`: a cheap **pre-sweep** runs `lang_id_service.identify` on every unit
  clip first (MMS-LID is fast relative to ASR), collecting per-clip top1. Compute
  `file_prior` once.
- Per-clip routing in `run_models`:
  - `routing_lang = clip_top1` if `clip_top1_conf ≥ MMS_LID_MIN_CONFIDENCE` **and**
    (`ALLOWED_LANGS` empty or `clip_top1 ∈ ALLOWED_LANGS`);
  - else `file_prior` if set;
  - else Whisper self-detected language.
- `ALLOWED_LANGS` defaults to empty (open auto) but is env-configurable
  (e.g. `te,en,hi`) so a known-language case can constrain it.

### 3. `run_models()` — replaces `_three_pass_asr` in `pipeline/tasks.py`

Runs the three models independently on the same `clean` clip:

- **Whisper**: `language=routing_lang` when prior is confident, else `auto`. Output passed
  through `hallucination_filter`. `no_speech_prob` and `compression_ratio` retained at clip
  level (mean across sub-segments) for gating.
- **IndicConformer**: routed by `routing_lang`. If `routing_lang ∉ _INDIC_SUPPORTED` →
  **abstain**: return `{"text": "", "confidence": None, "abstained": True,
  "flag": "indic_unsupported_lang"}`. No Whisper fallback. When it does run, confidence is
  derived from the model's output (CTC score if exposed by the checkpoint; otherwise
  `None` = "unscored", never a fabricated constant). Output filtered.
- **SeamlessM4T**: `tgt_lang` from `routing_lang`. Real generation-score confidence (already
  implemented). Output filtered.

Returns a dict with `lang_id` metadata + `whisper`/`indic`/`seamless` pass results, each
carrying `text`, `confidence` (may be `None`), `language`, and any `flag`.

### 4. `services/hallucination_filter.py` (new)

`filter_pass(result, *, no_speech_prob=None) -> dict` blanks a pass and zeroes confidence
when any holds:

- `no_speech_prob` is not None and `> NO_SPEECH_MAX` (default 0.6);
- normalized text matches a ghost phrase in `GHOST_PHRASES` (case/punctuation-insensitive;
  seed list: "thank you", "thanks for watching", "please subscribe", "[music]",
  "[applause]", "ご視聴ありがとうございました", "продолжение следует", "subscribe", ". .");
- existing repetition heuristic (`_has_repetition`, moved here) detects a loop.

Returns `{**result, "text": "", "confidence": 0.0, "hallucination": "<reason>"}` on a hit,
else the input unchanged. Pure function (no model/IO) → fully unit-testable.

### 5. `services/cross_model.py` (rewritten)

`compare_passes(texts: dict, confidences: dict, *, embed_fn, agreement_min=0.6) -> dict`

- Consider only non-empty passes.
- `embed_fn` returns pairwise cosine similarity (injected so tests pass a stub; production
  wires `embedding_service`). LaBSE is multilingual → Telugu vs Latin compares by meaning.
- `agreement` = mean pairwise cosine over non-empty passes (0.0 if <2 non-empty).
- `consensus_pass` = medoid (highest mean similarity to the others); tie-break by
  confidence; if only one non-empty pass, it is the consensus.
- `flagged` when: fewer than 2 non-empty passes, OR `agreement < agreement_min`, OR all
  confidences below `conf_thresh`. `flag_reason` one of
  `insufficient_passes | cross_model_disagreement | low_confidence`.
- Returns `{consensus_pass, consensus_text, agreement, confidence, flagged, flag_reason}`.

`_emit_segment` uses `consensus_pass`/`consensus_text` for the DB `text` and `source_pass`
(fixes D1), and stores `agreement` in `candidates`.

### 6. Output (L8) — `services/transcript_service.py` + `pipeline/tasks.py`

From the persisted segment candidates, write:

- `final/{file}_whisper_transcript.json`
- `final/{file}_indic_transcript.json`
- `final/{file}_seamless_transcript.json`

Each is an independent transcript: `[{segment_id, start, end, speaker, text, confidence,
language, flagged_for_review}]` for that model only (a pass that abstained/was filtered
shows empty text + its flag).

- `final/{file}_validation_report.json`: per segment `{segment_id, whisper, indic,
  seamless, agreement, consensus_pass, flags}` + file-level summary (counts, mean agreement,
  flag-reason histogram).
- `final/{file}_certified_transcript.json` **kept** = consensus view, so the existing
  `/cases/.../certify` and `/review` endpoints keep working unchanged.

New `transcript_service` functions: `build_per_model(...)`, `build_validation_report(...)`,
`write_named(case_id, file_id, name, data)`.

### 7. `config.py` additions

```
ALLOWED_LANGS      = set(os.getenv("ALLOWED_LANGS", "").split(",")) - {""}   # empty = open
LID_VOTE_MIN_CONF  = float(os.getenv("LID_VOTE_MIN_CONF", "0.5"))
NO_SPEECH_MAX      = float(os.getenv("NO_SPEECH_MAX", "0.6"))
AGREEMENT_MIN      = float(os.getenv("AGREEMENT_MIN", "0.6"))
LOUDNORM_LUFS      = float(os.getenv("LOUDNORM_LUFS", "-16.0"))
GHOST_PHRASES      = [...]   # seed list above, env-extendable
GAP_WINDOW_S       default 20.0 -> 10.0   # shorter windows = fewer silence-hallucinations
```

## Data flow

```
L0 ingest (WORM + hash)            [unchanged]
L1 normalize dual-rate             [unchanged]
L2 enhance (DeepFilterNet3)        [unchanged]  -> enhanced full file
L3 VAD union                       [unchanged]  -> union segments
L4 diarize (+overlap)              [unchanged]  -> turns
L5 segments:
   build units (hybrid + gap)      [unchanged]
   PRE-SWEEP MMS-LID over clips  --[new]--> file_prior
   for each unit:
     preprocess_service.prepare_clip  --[new]--> clean.wav (+ raw audit)
     run_models(clean)               --[new]--> whisper / indic / seamless (independent)
       each -> hallucination_filter  --[new]
     cross_model.compare_passes      --[rewritten]--> agreement + consensus
     _emit_segment                   --[fixed D1/D5]--> persist all candidates + consensus
L6 confidence report               [unchanged shape]
L8 output:
   3 per-model transcript files     --[new]
   validation_report.json           --[new]
   certified_transcript.json        [kept = consensus]
```

## Error handling / robustness ("0-fail")

- Every model call is independently try/wrapped; a model raising returns an empty
  flagged pass, never aborts the segment or the job. A segment always produces a row.
- IndicConformer abstains explicitly on unsupported language (no hidden Whisper).
- `loudnorm` failure → peak-normalize fallback → raw clip; preprocessing never fatal.
- Embedding failure in `compare_passes` → agreement `0.0` + `flagged` (degrade, not crash).
- All existing reconcile count/duration checks preserved (units==segments).
- Job-level try/except per stage block retained; partial failures quarantine/flag,
  not silent.

## Testing strategy

Laptop is build/mock-only (no GPU models). All new logic is pure or dependency-injected so
it is fully unit-testable with mocked model outputs (matches existing `unittest.mock`
convention). New test files:

- `tests/test_hallucination_filter.py` — ghost phrases, no_speech gate, repetition, clean
  text passes through untouched.
- `tests/test_lang_vote.py` — majority vote, confidence gate, allowed-set restriction,
  all-low-confidence → None.
- `tests/test_cross_model.py` (extended) — agreement math, medoid consensus, tie-break,
  insufficient-pass/disagreement/low-conf flags, embedding-failure degrade. Uses a stub
  `embed_fn`.
- `tests/test_preprocess_service.py` — trim/normalize call wiring + fallbacks (ffmpeg mocked).
- `tests/test_transcript_outputs.py` — 3 per-model files + validation report shape from
  mock segment candidates; abstained pass renders empty + flag.
- `tests/test_pipeline_attribution.py` (extended) — `run_models` independence: indic
  abstains on non-Indic, no model masquerades, consensus drives `source_pass`.

Whole suite must pass (`pytest`) on the laptop before deploy.

## Build / deploy

1. Implement + unit tests green on laptop.
2. Commit on `feat/forensic-v2-pipeline`, push to GitHub.
3. On server (`98.86.63.69`): pull, `docker compose build`, `docker compose up -d`.
4. Real end-to-end validation: re-run the test video, confirm three per-model transcripts
   differ meaningfully, validation report agreement is sane, and silence regions no longer
   emit ghost text.

## Out of scope (YAGNI)

- No new ASR models or model upgrades.
- No translation-layer changes.
- No UI/frontend.
- No diarization (L4) algorithm changes.
- No re-architecture of the WORM/audit/manifest custody chain.
