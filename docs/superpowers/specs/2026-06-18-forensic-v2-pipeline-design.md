# Forensic Audio Pipeline v2 — API Design Spec

**Date:** 2026-06-18
**Status:** Approved, pre-implementation
**Source architecture:** `ACB_Forensic_Audio_Pipeline_Architecture_v2.md`
**Scope:** Build the full v2 high-recall, court-defensible pipeline as a headless API. No frontend. Single-server, self-contained, deployable anywhere with `docker compose up`.

---

## 1. Goals & Constraints

**Goal:** Implement every layer (L0–L9) of the v2 architecture as an asynchronous, persisted, forensic-grade transcription API for ACB covert evidence audio (Telugu/English/Hindi/Urdu code-switch).

**Hard constraints:**
- **Accuracy first, no VRAM compromise.** All heavy models included (DeepFilterNet3, HTDemucs, secondary Indic ASR, multilingual embeddings, NLLB). Models run sequentially per file; queues prevent GPU contention.
- **GPU primary, CPU fallback.** Auto-detect device. GPU → float16; no GPU → int8 and slow (still functional). Image runs on any single box.
- **Single-server.** Celery, Redis, Postgres, workers, nginx all in one `docker-compose.yml`. No external managed services. One `.env`.
- **Never miss potential speech.** Every branch that could suppress quiet speech runs *alongside* the original; reconvergence only at the additive VAD union.
- **Court-defensible.** SHA-256 on first bytes, immutable originals, lineage hashing, hash-chained audit ledger, model checkpoint pinning, mandatory human certification.
- **No model execution on the build laptop.** Code only. Pure-logic unit tests run locally; GPU/model integration tests marked and skipped except on the deploy box.

**Out of scope:** Frontend/UI. DOPAMS integration (only JSON contract compatibility). Multi-tenant auth beyond the existing X-API-Key.

---

## 2. Topology (single server)

`docker-compose.yml`, 7 services:

| Service | Role |
|---|---|
| `nginx` | TLS termination + reverse proxy → `api` |
| `api` | FastAPI orchestrator: job submit/status, manifest writes, result + QA REST. No model inference. |
| `redis` | Celery broker + result backend |
| `postgres` | case/job/segment/review metadata + audit-ledger indexed mirror |
| `worker-cpu` | Celery `cpu_queue`: ingest, hash, ffmpeg normalize, standalone VAD + union, output gen, ledger writes |
| `worker-gpu` | Celery `gpu_queue`: enhancement, separation, diarization, multi-pass ASR, confidence |
| `flower` | Celery monitoring dashboard |

**Volumes:** `case_data` (cases tree + immutable originals + audit), `model_cache` (whisper/indic/dfn3/demucs weights), `hf_cache` (pyannote/HF), `pg_data`, `redis_data`.

**Workers share one image**, differ only by queue flag and the `deploy.resources` GPU reservation (present on `worker-gpu` only). On a CPU-only box, the GPU reservation is ignored / the block is removed; device auto-detect routes everything to CPU.

**Deploy:** populate `.env` (`API_KEY`, `HF_TOKEN`, optional `DEVICE`), `docker compose up -d`. Models download to `model_cache`/`hf_cache` on first run (or pre-baked).

---

## 3. Async Job Model

The current synchronous `POST /stt/transcribe` cannot survive 3-pass multi-branch inference on long (90–120 min) files. Replaced with a job-based API.

```
POST /cases                                       -> { case_id }
POST /cases/{case_id}/files                        -> upload; { file_id, job_id }; pipeline enqueued
GET  /jobs/{job_id}                                 -> { status, stage, per_layer_segment_counts, degraded_flags }
GET  /cases/{case_id}/files/{file_id}/transcript    -> current/certified transcript JSON
GET  /cases/{case_id}/files/{file_id}/confidence    -> confidence report JSON
GET  /cases/{case_id}/files/{file_id}/diarization   -> speaker timeline JSON
GET  /cases/{case_id}/audit                         -> audit ledger (paginated, from postgres mirror)
```

- Pipeline is a Celery task chain with **per-segment checkpointing** — a worker crash resumes from the last completed segment, not file start.
- Every layer records segment-count in/out. A mismatch between adjacent layers raises a **pipeline-level alert** (v2 §8 "silent under-reporting" critical rule), not just a per-segment flag.
- `job.status` ∈ `queued | running | needs_review | certified | failed | quarantined`.
- The legacy sync `/stt/transcribe` endpoint is **removed**; small-file convenience is served by submitting a job and polling (fast on GPU).

---

## 4. Pipeline Layers → Celery Tasks

Queue assignment and order (branches run alongside original; union is additive-only):

| Layer | Queue | Task |
|---|---|---|
| L0 | cpu | Ingest: SHA-256 of byte-exact original → immutable WORM write → manifest + case/file registration. Runs before any decode. |
| L1 | cpu | FFmpeg normalize → `{file_id}_48k.wav` (enhance/diar) + `{file_id}_16k_mono.wav` (ASR). Transform logged. Video audio extracted. Probe failure → quarantine. |
| L2 | gpu | DeepFilterNet3 → `{file_id}_dfn3.wav`. Parallel; never replaces original. Worker exception → `degraded_enhancement`, continue original-only. |
| L2b | gpu | HTDemucs `htdemucs_ft` vocal stem. **Opt-in per job, off by default.** Excluded from union if it reduces detected segment count vs. pre-separation. |
| L3 | cpu | Standalone Silero VAD on **each** branch (original, enhanced, stem), `threshold≈0.25`, `min_speech_duration_ms=100`, `speech_pad_ms=300`, `min_silence_duration_ms≈100`. Output sets **unioned**: timestamp is speech if *any* branch flags it. |
| L4 | gpu | pyannote 3.1 overlap-aware diarization on union regions. Produces speaker turns + explicit overlap regions. Low confidence → single "multi-speaker" segment flagged for manual tagging, never dropped. |
| L5 | gpu | Multi-pass ASR on VAD-bounded clips: pass1 Whisper-lv3 on enhanced, pass2 Whisper-lv3 on original, pass3 Indic model. High-recall decode params (carried from current `whisper_service`). Hallucination (compression/repetition) → re-run segment at smaller chunk with temperature ladder. |
| L6 | cpu | Cross-model compare: normalized edit distance + multilingual embedding similarity + per-pass avg logprob → single per-segment confidence + disagreement flag. VAD-positive but any pass empty → forced lowest-confidence category. |
| L7 | cpu | Route: disagreement OR low confidence OR VAD-positive/ASR-empty → review queue (priority-ordered); else auto-accept (full provenance retained). |
| L8 | cpu | Emit 4 JSON artifacts (transcript, diarization timeline, confidence report, audit log) + human-readable export. |
| L9 | all | Append hash-chained entry to `audit/{case_id}/ledger.jsonl` on every task; mirror into postgres. Cross-cutting. |

**Decode params (L5 pass1/pass2, from current tuned `whisper_service.py`):** `beam_size=10`, `condition_on_previous_text=False`, `temperature=0.0` with fallback ladder `[0,0.2,0.4,0.6,0.8,1.0]` on hallucination re-run, `repetition_penalty=1.3`, `no_speech_threshold=0.3` (lowered for recall), `log_prob_threshold=-1.5` (loosened), `compression_ratio_threshold` raised slightly, `word_timestamps=True`, **no initial_prompt**. Keep-all-segments (no hallucination drop filter); flag via confidence/compression_ratio.

---

## 5. Models

| Layer | Model | Device | Notes |
|---|---|---|---|
| L2 | DeepFilterNet3 (latest stable) | gpu (cpu workable) | Native 48k where supported |
| L2b | HTDemucs `htdemucs_ft` | gpu | Advisory input to VAD only |
| L3 | Silero VAD (standalone `silero-vad` pkg) | cpu | Replaces faster-whisper builtin VAD so per-branch union is possible |
| L4 | `pyannote/speaker-diarization-3.1` | gpu | Overlapped-speech detection enabled |
| L5 p1/p2 | faster-whisper `large-v3` | gpu fp16 / cpu int8 | |
| L5 p3 | Indic ASR — `ai4bharat/indic-conformer` family or IndicWhisper (pin checkpoint at deploy) | gpu | Telugu/Hindi/Urdu code-switch |
| L6 | Multilingual sentence embedding (LaBSE / `paraphrase-multilingual-mpnet`) | cpu/gpu | Similarity that tolerates transliteration |
| opt | NLLB-200 distilled-600M | gpu | Retained text-MT alternative to Whisper translate |

All model versions pinned by **checkpoint SHA** in config and recorded in the ledger.

---

## 6. Persistence & Chain-of-Custody

**Directory layout** (per v2 §6):

```
/forensic-audio/cases/{case_id}/
  manifest.json
  originals/{file_id}__original.{ext}          # READ-ONLY, immutable (chattr +i / RO mount)
  derivatives/{file_id}/
    normalized/{file_id}_48k.wav, {file_id}_16k_mono.wav
    enhanced/{file_id}_dfn3.wav
    separated/{file_id}_vocal_stem.wav          # only if analyst-enabled
    vad/{file_id}_segments_union.json
    diarization/{file_id}_speaker_timeline.json
    asr/pass1_enhanced/…, pass2_original/…, pass3_secondary/…
    confidence/{file_id}_confidence_report.json
    review_queue/{file_id}_flagged_segments.json
    final/{file_id}_certified_transcript.json
/forensic-audio/audit/{case_id}/ledger.jsonl    # append-only, hash-chained
```

**Integrity rules:**
- SHA-256 of byte-exact original computed **before** any other process touches the file; stored in per-case manifest.
- Every derivative hashed on creation; manifest entry records the **parent** artifact hash → traceable lineage graph.
- Ledger append-only and hash-chained: each entry embeds prior entry's hash; retroactive edits detectable.
- Postgres is an **indexed mirror** for fast lookup; the JSONL ledger on the immutable volume is the **source of truth**.
- Final transcript marked `machine_assisted_pending_certification`; not "final" until a named reviewer signs off every flagged segment. Auto-accepted segments carry full model/version/parameter provenance.

**Postgres tables:** `cases`, `files`, `jobs`, `segments`, `reviews`, `audit_entries` (mirror). SQLAlchemy ORM + Alembic migrations.

---

## 7. Output Schemas

Exactly as v2 §7:
- **Transcript** (`final/{file_id}_certified_transcript.json`): `file_id, case_id, source_hash_sha256, segments[{segment_id, start, end, speaker, overlap, text, language, confidence, source_pass, flagged_for_review, review_status, reviewer_id}], status`.
- **Diarization timeline**: `file_id, speakers[], timeline[{start, end, speakers[], overlap}], model_version`.
- **Confidence report**: `file_id, segments_total, segments_auto_accepted, segments_flagged, flag_reasons{}, per_segment[{segment_id, pass*_text_hash, edit_distance_norm, embedding_similarity, avg_logprob, flag_reason}]`.
- **Audit ledger entry** (JSONL): `entry_id, timestamp, case_id, file_id, stage, model, model_checkpoint_sha256, parameters{}, input_hash, output_hash, operator, prev_entry_hash, entry_hash`.

Pydantic models in `models/schemas.py` mirror these one-to-one.

---

## 8. Human QA REST API (L7, headless)

```
GET  /review/queue?case_id=&status=pending          -> flagged segments, priority-ordered
GET  /review/segments/{segment_id}                   -> { clip_orig_url, clip_enhanced_url, candidates[pass1,pass2,pass3], metrics }
POST /review/segments/{segment_id}                   -> { decision: accept|edit|reject, text?, reviewer_id }
POST /cases/{case_id}/files/{file_id}/certify        -> finalize; 409 if any flagged segment unresolved
```

- Review state machine per segment: `pending → (accepted|edited|rejected)`; certification gated on zero `pending` flagged segments.
- Clip URLs serve the stored per-segment original + enhanced audio (synchronized playback is a client concern; API just provides both clips).
- All review actions written to the audit ledger.

---

## 9. Decision Points & Fallbacks (v2 §3)

| Condition | Fallback |
|---|---|
| Enhancement crashes | continue original-only, flag `degraded_enhancement` |
| Separation reduces VAD segment count | exclude stem from union, log |
| Diarization fails on overlap | keep region as single "multi-speaker", flag for manual tagging |
| Whisper hallucination | re-run segment at smaller chunk + temperature ladder |
| All ASR passes empty on VAD-positive segment | route to QA as highest priority; never auto "no speech" |
| Corrupt/unsupported input | quarantine, alert case manager, never silently drop |
| GPU OOM long file | segment-chunked inference (default behavior) |
| Worker crash mid-job | per-segment checkpoint resume |

---

## 10. Testing Strategy (no model execution on build box)

- **Pure-logic unit tests run locally** (mocking model outputs): SHA hashing, ledger hash-chaining + tamper detection, VAD union math, segment-count reconciliation/alert, cross-model compare + confidence scoring, alignment, dialogue grouping, schema validation, QA review state machine, certification gating.
- **GPU/model integration tests** marked `@pytest.mark.gpu` (or `model`), skipped by default, run only on the deploy box.
- Existing service tests retained and adapted to new module boundaries.

---

## 11. Phased Delivery

One spec (this doc); implementation plan in 5 independently-shippable phases:

1. **Foundation** — `docker-compose` (redis/postgres/celery/flower/nginx/api/workers), config refactor, DB schema + Alembic migrations, X-API-Key auth, health, async job skeleton + status endpoint.
2. **Chain-of-custody** — L0/L1/L9: SHA hashing, immutable WORM store, manifest + lineage, hash-chained ledger + postgres mirror, count-reconciliation alerts, quarantine path.
3. **Recall branches** — L2 DeepFilterNet3 enhance, L3 standalone Silero VAD per-branch + union, L2b HTDemucs separation (gated, exclude-if-reduces rule).
4. **Attribution + ASR** — L4 overlap-aware diarization, L5 3-pass ASR (whisper enhanced + whisper original + Indic), hallucination re-run ladder, L6 cross-model compare + confidence.
5. **Outputs + QA** — L8 four JSON artifacts + human-readable export, L7 review REST API + certification gating.

Each phase: own task list, own review checkpoint, deployable state at phase end.

---

## 12. Carry-Over From Current Code

Reused, not rebuilt:
- Tuned high-recall Whisper decode params (`services/whisper_service.py`).
- Dynamic VAD-off for low-volume audio (`config.py` `VAD_MIN_MEAN_DB`) — folded into L3 per-branch logic.
- ffmpeg normalization recipe (`-map 0:a:0?`, `aresample=async=1:first_pts=0`) — extended to dual-rate output.
- pyannote 3.1 diarization (`services/diarization_service.py`) — extended with overlap detection.
- Alignment + dialogue grouping (`services/alignment_service.py`, `services/dialogue_service.py`).
- NLLB translation service (`services/translation_service.py`) — retained as optional text-MT.
- X-API-Key auth (`api/auth.py`).
