# Forensic v2 — Phase 3 (Recall Branches) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Add the high-recall branches — DeepFilterNet3 enhancement (L2), standalone Silero VAD run per branch then **unioned** (L3), and gated HTDemucs source separation (L2b) — so no quiet voice is decided out by any single branch.

**Architecture:** Model wrappers (`enhancement_service`, `separation_service`, `vad_service`) lazy-load their models and are NOT exercised locally (gpu-marked). All decision logic — the VAD **union** algorithm and the separation **include/exclude gate** — lives in pure functions unit-tested with synthetic segment lists. The pipeline orchestrates branches, monkeypatched in tests.

**Tech Stack:** DeepFilterNet, demucs, silero-vad, torch.

## Global Constraints

- No ML model execution locally. Model wrappers are import-guarded + gpu-marked.
- VAD config: `threshold=0.25`, `min_speech_duration_ms=100`, `speech_pad_ms=300`, `min_silence_duration_ms=100`.
- Branches are **additive**: union never has fewer regions than any input branch. Separation is excluded from the union when it yields fewer regions than the pre-separation union.
- Enhancement failure is non-fatal: flag `degraded_enhancement`, continue original-only.
- All Phase 1+2 tests stay green.

---

### Task 1: VAD union algorithm (pure)

**Files:**
- Create: `services/vad_union.py`
- Test: `tests/test_vad_union.py`

**Interfaces:**
- `vad_union.union_segments(branches: list[list[dict]]) -> list[dict]` — input is a list of branch segment-lists (`{"start","end"}`); output is the merged, sorted, coalesced union (overlapping/touching intervals joined).
- `vad_union.total_duration(segs) -> float`.
- `vad_union.should_include_separation(pre_union_count: int, stem_count: int) -> bool` — `stem_count >= pre_union_count`.

- [ ] **Step 1: Failing test**

```python
# tests/test_vad_union.py
from services.vad_union import union_segments, total_duration, should_include_separation


def test_union_merges_overlaps_across_branches():
    a = [{"start": 0.0, "end": 2.0}, {"start": 5.0, "end": 6.0}]
    b = [{"start": 1.5, "end": 3.0}, {"start": 10.0, "end": 11.0}]
    u = union_segments([a, b])
    assert u == [
        {"start": 0.0, "end": 3.0},
        {"start": 5.0, "end": 6.0},
        {"start": 10.0, "end": 11.0},
    ]


def test_union_preserves_lone_branch_segments():
    a = []
    b = [{"start": 4.4, "end": 9.0}]
    assert union_segments([a, b]) == [{"start": 4.4, "end": 9.0}]


def test_touching_intervals_coalesce():
    a = [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}]
    assert union_segments([a]) == [{"start": 0.0, "end": 2.0}]


def test_total_duration():
    assert total_duration([{"start": 0.0, "end": 2.0}, {"start": 5.0, "end": 6.0}]) == 3.0


def test_separation_gate():
    assert should_include_separation(10, 12) is True
    assert should_include_separation(10, 10) is True
    assert should_include_separation(10, 7) is False
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement `services/vad_union.py`**

```python
def union_segments(branches):
    intervals = []
    for branch in branches:
        for seg in branch:
            intervals.append((float(seg["start"]), float(seg["end"])))
    if not intervals:
        return []
    intervals.sort()
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:  # overlap or touch
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [{"start": s, "end": e} for s, e in merged]


def total_duration(segs) -> float:
    return round(sum(s["end"] - s["start"] for s in segs), 6)


def should_include_separation(pre_union_count: int, stem_count: int) -> bool:
    return stem_count >= pre_union_count
```

- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** `feat: phase3 VAD union algorithm + separation gate (pure)`.

---

### Task 2: Config + requirements for recall models

**Files:**
- Modify: `config.py` (VAD params, model ids/checkpoints)
- Modify: `requirements.txt`
- Test: `tests/test_config_phase3.py`

**Interfaces:**
- Produces: `config.VAD_THRESHOLD=0.25`, `config.VAD_MIN_SPEECH_MS=100`, `config.VAD_SPEECH_PAD_MS_L3=300`, `config.VAD_MIN_SILENCE_MS_L3=100`, `config.DFN_MODEL="DeepFilterNet3"`, `config.DEMUCS_MODEL="htdemucs_ft"`.

- [ ] **Step 1: Failing test**

```python
# tests/test_config_phase3.py
import config


def test_recall_config_present():
    assert config.VAD_THRESHOLD == 0.25
    assert config.VAD_MIN_SPEECH_MS == 100
    assert config.VAD_SPEECH_PAD_MS_L3 == 300
    assert config.VAD_MIN_SILENCE_MS_L3 == 100
    assert config.DFN_MODEL == "DeepFilterNet3"
    assert config.DEMUCS_MODEL == "htdemucs_ft"
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Append to `config.py`**

```python
# --- Phase 3: recall branches (enhancement / VAD union / separation) ---
VAD_THRESHOLD = 0.25
VAD_MIN_SPEECH_MS = 100
VAD_SPEECH_PAD_MS_L3 = 300
VAD_MIN_SILENCE_MS_L3 = 100
DFN_MODEL = "DeepFilterNet3"          # DeepFilterNet3 enhancement
DEMUCS_MODEL = "htdemucs_ft"          # HTDemucs separation checkpoint
```

- [ ] **Step 4: Append to `requirements.txt`** (installed only on the deploy/GPU box)

```
deepfilternet==0.5.6
demucs==4.0.1
silero-vad==5.1
```

- [ ] **Step 5: Run, expect pass. Commit** `feat: phase3 recall config + model deps`.

---

### Task 3: Model wrappers (lazy, gpu-marked — not run locally)

**Files:**
- Create: `services/vad_service.py`
- Create: `services/enhancement_service.py`
- Create: `services/separation_service.py`
- Test: `tests/test_recall_wrappers_import.py`

**Interfaces:**
- `vad_service.detect_speech(wav_path: str) -> list[dict]` — Silero VAD at `config.VAD_THRESHOLD`, returns `[{"start","end"}]` in seconds. Lazy `load_vad()`.
- `enhancement_service.enhance(in_wav: str, out_wav: str) -> str` — DeepFilterNet3 denoise; lazy `load_dfn()`.
- `separation_service.separate_vocals(in_wav: str, out_wav: str) -> str` — HTDemucs vocal stem; lazy `load_demucs()`.

- [ ] **Step 1: Failing test** (import-only; model load is gpu-marked elsewhere)

```python
# tests/test_recall_wrappers_import.py
def test_wrappers_import_and_expose_callables():
    from services import vad_service, enhancement_service, separation_service
    assert callable(vad_service.detect_speech)
    assert callable(enhancement_service.enhance)
    assert callable(separation_service.separate_vocals)
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement `services/vad_service.py`**

```python
import config

_model = None
_get_ts = None


def load_vad():
    """Lazy-load Silero VAD. Heavy import deferred so unit tests need no model."""
    global _model, _get_ts
    if _model is None:
        from silero_vad import load_silero_vad, get_speech_timestamps
        _model = load_silero_vad()
        _get_ts = get_speech_timestamps
    return _model, _get_ts


def detect_speech(wav_path: str) -> list[dict]:
    import soundfile as sf
    model, get_ts = load_vad()
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    import torch
    ts = get_ts(
        torch.from_numpy(audio), model,
        sampling_rate=sr,
        threshold=config.VAD_THRESHOLD,
        min_speech_duration_ms=config.VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=config.VAD_MIN_SILENCE_MS_L3,
        speech_pad_ms=config.VAD_SPEECH_PAD_MS_L3,
        return_seconds=True,
    )
    return [{"start": round(t["start"], 3), "end": round(t["end"], 3)} for t in ts]
```

- [ ] **Step 4: Implement `services/enhancement_service.py`**

```python
import config

_state = None


def load_dfn():
    global _state
    if _state is None:
        from df.enhance import init_df
        model, df_state, _ = init_df()
        _state = (model, df_state)
    return _state


def enhance(in_wav: str, out_wav: str) -> str:
    import soundfile as sf
    from df.enhance import enhance as df_enhance, load_audio, save_audio
    model, df_state = load_dfn()
    audio, _ = load_audio(in_wav, sr=df_state.sr())
    enhanced = df_enhance(model, df_state, audio)
    save_audio(out_wav, enhanced, df_state.sr())
    return out_wav
```

- [ ] **Step 5: Implement `services/separation_service.py`**

```python
import config

_model = None


def load_demucs():
    global _model
    if _model is None:
        from demucs.pretrained import get_model
        _model = get_model(config.DEMUCS_MODEL)
    return _model


def separate_vocals(in_wav: str, out_wav: str) -> str:
    import torch, torchaudio
    from demucs.apply import apply_model
    model = load_demucs()
    wav, sr = torchaudio.load(in_wav)
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    sources = apply_model(model, wav[None], device="cpu")[0]
    vocals = sources[model.sources.index("vocals")]
    torchaudio.save(out_wav, vocals, model.samplerate)
    return out_wav
```

- [ ] **Step 6: Run, expect pass** (import works without loading models, since heavy imports are inside functions).
- [ ] **Step 7: Commit** `feat: phase3 recall model wrappers (vad/enhance/separate, lazy)`.

---

### Task 4: Job separation opt-in + migration 0002

**Files:**
- Modify: `db/models.py` (add `Job.options` JSON)
- Create: `alembic/versions/0002_job_options.py`
- Modify: `db/repository.py` (`create_job` accepts `options`)
- Modify: `api/routes/cases.py` (accept `separate` form field)
- Test: `tests/test_job_options.py`

**Interfaces:**
- `Job.options: dict` (default `{}`), holds `{"separate": bool}`.
- `repository.create_job(session, case_id, file_id, options=None) -> str`.

- [ ] **Step 1: Failing test**

```python
# tests/test_job_options.py
from db import base as dbbase, repository as repo


def setup_module():
    dbbase.init_db()


def test_job_stores_options():
    with dbbase.get_session() as s:
        c = repo.create_case(s)
        f = repo.create_file(s, c, "a.wav", ".wav")
        j = repo.create_job(s, c, f, options={"separate": True})
        s.commit()
    with dbbase.get_session() as s:
        assert repo.get_job(s, j).options == {"separate": True}
```

- [ ] **Step 2: Run, expect fail** (`create_job() got unexpected keyword 'options'`).

- [ ] **Step 3: Add column** — in `db/models.py` `Job`, after `degraded_flags`:

```python
    options: Mapped[dict | None] = mapped_column(JSON, default=dict)
```

- [ ] **Step 4: Migration `alembic/versions/0002_job_options.py`**

```python
"""job options

Revision ID: 0002
Revises: 0001
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("options", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "options")
```

- [ ] **Step 5: Update `repository.create_job`**

```python
def create_job(session, case_id: str, file_id: str, options=None) -> str:
    job = Job(case_id=case_id, file_id=file_id, status=JobStatus.QUEUED,
              stage=None, degraded_flags=[], options=options or {})
    session.add(job)
    session.flush()
    return job.id
```

- [ ] **Step 6: Update `api/routes/cases.py`** — add `Form` import and `separate` param to `upload_file`:

Signature becomes:

```python
async def upload_file(case_id: str, audio: UploadFile = File(...),
                      separate: bool = Form(default=False)):
```

And create the job with options:

```python
        job_id = repo.create_job(s, case_id, file_id, options={"separate": separate})
```

(Add `Form` to the `fastapi` import line.)

- [ ] **Step 7: Run test + regression**

Run: `pytest tests/test_job_options.py tests/test_api_jobs.py tests/test_migration.py -v`
Note: `test_migration` asserts table existence — still valid. Expected PASS.

- [ ] **Step 8: Commit** `feat: phase3 job separation opt-in + migration 0002`.

---

### Task 5: Pipeline L2/L3/L2b integration (mocked models in tests)

**Files:**
- Modify: `pipeline/tasks.py` (add `_l2_enhance`, `_l2b_separate`, `_l3_vad_union`; wire after L1)
- Test: `tests/test_pipeline_recall.py`

**Interfaces:**
- After L1, `run_pipeline`:
  1. L2: `enhanced = _l2_enhance(...)` — on exception, `add_degraded("degraded_enhancement")`, `enhanced=None`.
  2. L2b (only if `job.options["separate"]`): `stem = _l2b_separate(...)`.
  3. L3: run `vad_service.detect_speech` on original-16k + enhanced(+stem); `union_segments`; if stem present and `should_include_separation` is False, drop stem from union and log. Write `vad/{file_id}_segments_union.json`. Ledger entry. `reconcile.check` each branch count ≤ union count.
- Writes `derivatives/{file_id}/vad/{file_id}_segments_union.json` = `{"segments":[...], "branch_counts":{...}}`.

- [ ] **Step 1: Failing test** (models monkeypatched — no GPU)

```python
# tests/test_pipeline_recall.py
import os, json, shutil, subprocess
import pytest
from db import base as dbbase, repository as repo
from db.models import JobStatus
from services import audit_service as au, manifest_service as man, storage
from services import vad_service, enhancement_service, separation_service
from pipeline import tasks as ptasks

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def setup_module():
    dbbase.init_db()


def _patch_store(monkeypatch, tmp_path):
    for mod in (ptasks.config, au.config, man.config, storage.config):
        monkeypatch.setattr(mod, "CASE_STORE_PATH", str(tmp_path))


def _make_tone(path):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=1", "-ar", "22050", str(path)],
                   capture_output=True, check=True)


def _stage(tmp_path, monkeypatch, options=None):
    _patch_store(monkeypatch, tmp_path)
    with dbbase.get_session() as s:
        case_id = repo.create_case(s)
        file_id = repo.create_file(s, case_id, "tone.wav", ".wav")
        job_id = repo.create_job(s, case_id, file_id, options=options or {})
        s.commit()
    inbox = os.path.join(str(tmp_path), "cases", case_id, "inbox")
    os.makedirs(inbox, exist_ok=True)
    _make_tone(os.path.join(inbox, f"{file_id}.wav"))
    return case_id, file_id, job_id


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_vad_union_written_two_branches(tmp_path, monkeypatch):
    # enhancement returns a file; VAD returns one segment per branch (offset)
    monkeypatch.setattr(enhancement_service, "enhance",
                        lambda i, o: (shutil_copy(i, o)))
    monkeypatch.setattr(vad_service, "detect_speech",
                        lambda p: [{"start": 0.0, "end": 1.0}])
    case_id, file_id, job_id = _stage(tmp_path, monkeypatch)
    result = ptasks.run_pipeline.apply(args=[job_id]).get()
    assert result == JobStatus.NEEDS_REVIEW
    vad_json = os.path.join(str(tmp_path), "cases", case_id, "derivatives",
                            file_id, "vad", f"{file_id}_segments_union.json")
    data = json.load(open(vad_json))
    assert data["segments"] == [{"start": 0.0, "end": 1.0}]
    assert "original" in data["branch_counts"] and "enhanced" in data["branch_counts"]


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_enhancement_failure_flags_degraded(tmp_path, monkeypatch):
    def _boom(i, o):
        raise RuntimeError("dfn exploded")
    monkeypatch.setattr(enhancement_service, "enhance", _boom)
    monkeypatch.setattr(vad_service, "detect_speech",
                        lambda p: [{"start": 0.0, "end": 1.0}])
    case_id, file_id, job_id = _stage(tmp_path, monkeypatch)
    result = ptasks.run_pipeline.apply(args=[job_id]).get()
    assert result == JobStatus.NEEDS_REVIEW
    with dbbase.get_session() as s:
        assert "degraded_enhancement" in repo.get_job(s, job_id).degraded_flags


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_separation_excluded_when_fewer_segments(tmp_path, monkeypatch):
    monkeypatch.setattr(enhancement_service, "enhance",
                        lambda i, o: shutil_copy(i, o))
    monkeypatch.setattr(separation_service, "separate_vocals",
                        lambda i, o: shutil_copy(i, o))

    def _vad(path):
        # stem branch (filename has 'stem') returns NO speech -> fewer -> excluded
        if "stem" in path:
            return []
        return [{"start": 0.0, "end": 1.0}]
    monkeypatch.setattr(vad_service, "detect_speech", _vad)

    case_id, file_id, job_id = _stage(tmp_path, monkeypatch, options={"separate": True})
    result = ptasks.run_pipeline.apply(args=[job_id]).get()
    assert result == JobStatus.NEEDS_REVIEW
    vad_json = os.path.join(str(tmp_path), "cases", case_id, "derivatives",
                            file_id, "vad", f"{file_id}_segments_union.json")
    data = json.load(open(vad_json))
    assert data["separation_included"] is False
    assert data["segments"] == [{"start": 0.0, "end": 1.0}]


# helper: copy that returns dest (monkeypatch enhancement/separation as pass-through)
def shutil_copy(src, dst):
    import shutil
    shutil.copyfile(src, dst)
    return dst
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Edit `pipeline/tasks.py`** — add imports at top:

```python
from services import vad_service, enhancement_service, separation_service
from services import vad_union
import json
```

- [ ] **Step 4: Add branch functions to `pipeline/tasks.py`** (before `run_pipeline`):

```python
def _l2_enhance(job, in16, source_hash, session):
    """DeepFilterNet3 enhancement (parallel branch). On failure, flag degraded
    and return None so downstream runs original-only."""
    out = storage.derivative_path(job.case_id, job.file_id, "enhanced",
                                  f"{job.file_id}_dfn3.wav")
    try:
        enhancement_service.enhance(in16, out)
    except Exception as e:  # never fatal — original branch still carries recall
        repo.update_job(session, job.id, add_degraded="degraded_enhancement")
        session.commit()
        au.append_entry(job.case_id, file_id=job.file_id, stage="L2",
                        parameters={"error": str(e)}, session=session)
        session.commit()
        return None
    h = sha256_file(out)
    man.register_derivative(job.case_id, job.file_id, "enhanced_dfn3", out, h,
                            parent_sha256=source_hash)
    au.append_entry(job.case_id, file_id=job.file_id, stage="L2",
                    model=config.DFN_MODEL, input_hash=source_hash,
                    output_hash=h, session=session)
    session.commit()
    return out


def _l2b_separate(job, in16, source_hash, session):
    out = storage.derivative_path(job.case_id, job.file_id, "separated",
                                  f"{job.file_id}_vocal_stem.wav")
    try:
        separation_service.separate_vocals(in16, out)
    except Exception as e:
        au.append_entry(job.case_id, file_id=job.file_id, stage="L2b",
                        parameters={"error": str(e)}, session=session)
        session.commit()
        return None
    h = sha256_file(out)
    man.register_derivative(job.case_id, job.file_id, "separated_stem", out, h,
                            parent_sha256=source_hash)
    au.append_entry(job.case_id, file_id=job.file_id, stage="L2b",
                    model=config.DEMUCS_MODEL, input_hash=source_hash,
                    output_hash=h, session=session)
    session.commit()
    return out


def _l3_vad_union(job, in16, enhanced, stem, session):
    branches = {"original": vad_service.detect_speech(in16)}
    if enhanced:
        branches["enhanced"] = vad_service.detect_speech(enhanced)

    pre_union = vad_union.union_segments(list(branches.values()))
    separation_included = None
    if stem:
        stem_segs = vad_service.detect_speech(stem)
        separation_included = vad_union.should_include_separation(
            len(pre_union), len(stem_segs))
        if separation_included:
            branches["separated"] = stem_segs

    union = vad_union.union_segments(list(branches.values()))
    branch_counts = {k: len(v) for k, v in branches.items()}
    # Additive guarantee: union has at least as many regions as any branch.
    for name, segs in branches.items():
        reconcile.check(f"L3:{name}", len(segs), "L3:union", len(union))

    out = storage.derivative_path(job.case_id, job.file_id, "vad",
                                  f"{job.file_id}_segments_union.json")
    payload = {"segments": union, "branch_counts": branch_counts,
               "separation_included": separation_included}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    au.append_entry(job.case_id, file_id=job.file_id, stage="L3",
                    parameters={"branch_counts": branch_counts,
                                "union_count": len(union)}, session=session)
    session.commit()
    return union
```

- [ ] **Step 5: Wire into `run_pipeline`** — replace the placeholder-remainder block. Change `PLACEHOLDER_STAGES` to `["L4", "L5", "L6", "L7", "L8"]` and insert the L2/L2b/L3 calls before it:

```python
    # L2/L2b/L3 recall branches.
    try:
        with get_session() as s:
            job = repo.get_job(s, job_id)
            in16 = storage.derivative_path(job.case_id, job.file_id, "normalized",
                                           f"{job.file_id}_16k_mono.wav")
            repo.update_job(s, job_id, stage="L2"); s.commit()
            enhanced = _l2_enhance(job, in16, source_hash, s)
            stem = None
            if (job.options or {}).get("separate"):
                repo.update_job(s, job_id, stage="L2b"); s.commit()
                stem = _l2b_separate(job, in16, source_hash, s)
            repo.update_job(s, job_id, stage="L3"); s.commit()
            _l3_vad_union(job, in16, enhanced, stem, s)
    except Exception as e:
        with get_session() as s:
            repo.update_job(s, job_id, status=JobStatus.FAILED, error=str(e))
            s.commit()
        raise
```

(Keep `PLACEHOLDER_STAGES = ["L4", "L5", "L6", "L7", "L8"]` walking afterward to `needs_review`.)

- [ ] **Step 6: Run, expect pass** — `pytest tests/test_pipeline_recall.py -v`.
- [ ] **Step 7: Full suite** — `pytest -q`; green except pre-existing alignment test.
- [ ] **Step 8: Commit** `feat: phase3 L2 enhance + L2b separate + L3 VAD union pipeline`.

---

## Phase 3 Done — Definition of Done

- VAD union algorithm merges multi-branch segments additively (pure, fully tested).
- Pipeline runs original + enhanced (+ optional stem) VAD branches, writes `vad/..._segments_union.json` with per-branch counts.
- Enhancement failure flags `degraded_enhancement`, never fatal.
- Separation gated: excluded from union when it reduces region count; recorded in the VAD JSON.
- Reconciliation guarantees the union never has fewer regions than any branch.
- All prior tests green (pre-existing alignment failure excepted).

**Next:** Phase 4 (attribution + multi-pass ASR L4/L5/L6).
