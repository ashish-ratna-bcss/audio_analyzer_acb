# Forensic v2 — Phase 5 (Outputs + Human QA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Emit the certified-transcript JSON (L8) and provide the headless human-QA REST API (L7) — review queue, per-segment candidates + clip audio, accept/edit/reject, and a gated certification endpoint that flips the transcript to `certified` only when every flagged segment is signed off.

**Architecture:** No models — pure API + DB + JSON. `Segment` gains candidate-pass texts + clip paths so the reviewer sees all three opinions. A review router exposes queue/detail/decision/clip endpoints; certification enforces zero-pending-flags. All testable locally.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic, Pydantic.

## Global Constraints

- Transcript status `machine_assisted_pending_certification` until certified.
- Certification 409s if any flagged segment is still `pending`.
- Every review action + certification appends to the audit ledger.
- All prior tests stay green.

---

### Task 1: Segment candidates + clip paths (migration 0003) and L5 persistence

**Files:**
- Modify: `db/models.py` (Segment: `candidates` JSON, `clip_original`, `clip_enhanced`)
- Create: `alembic/versions/0003_segment_candidates.py`
- Modify: `db/repository.py` (`add_segment` accepts the new fields; add `get_segment`)
- Modify: `pipeline/tasks.py` (`_l5_l6_segments` persists candidates + clip paths)
- Test: `tests/test_segment_candidates.py`

**Interfaces:**
- `Segment.candidates: dict` = `{"pass1_enhanced","pass2_original","pass3_indic"}`.
- `Segment.clip_original: str`, `Segment.clip_enhanced: str`.
- `repository.get_segment(session, segment_id) -> Segment | None`.
- `repository.add_segment(..., candidates=None, clip_original=None, clip_enhanced=None)`.

- [ ] **Step 1: Failing test**

```python
# tests/test_segment_candidates.py
from db import base as dbbase, repository as repo


def setup_module():
    dbbase.init_db()


def test_segment_stores_candidates_and_clips():
    with dbbase.get_session() as s:
        c = repo.create_case(s); f = repo.create_file(s, c, "a.wav", ".wav")
        seg_id = repo.add_segment(
            s, file_id=f, start=0.0, end=1.0, speaker="Speaker_1",
            text="hi", confidence=0.6, source_pass="pass1_enhanced",
            flagged=True, review_status="pending",
            candidates={"pass1_enhanced": "hi", "pass2_original": "hi there",
                        "pass3_indic": "hi"},
            clip_original="/p/org.wav", clip_enhanced="/p/enh.wav")
        s.commit()
    with dbbase.get_session() as s:
        seg = repo.get_segment(s, seg_id)
        assert seg.candidates["pass2_original"] == "hi there"
        assert seg.clip_original == "/p/org.wav"
```

- [ ] **Step 2: Run, expect fail** (`add_segment() got unexpected keyword 'candidates'`).

- [ ] **Step 3: Add columns** — in `db/models.py` `Segment`, after `review_status`:

```python
    candidates: Mapped[dict | None] = mapped_column(JSON, default=dict)
    clip_original: Mapped[str | None] = mapped_column(String, nullable=True)
    clip_enhanced: Mapped[str | None] = mapped_column(String, nullable=True)
```

- [ ] **Step 4: Migration `alembic/versions/0003_segment_candidates.py`**

```python
"""segment candidates + clips

Revision ID: 0003
Revises: 0002
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("candidates", sa.JSON(), nullable=True))
    op.add_column("segments", sa.Column("clip_original", sa.String(), nullable=True))
    op.add_column("segments", sa.Column("clip_enhanced", sa.String(), nullable=True))


def downgrade() -> None:
    for c in ["candidates", "clip_original", "clip_enhanced"]:
        op.drop_column("segments", c)
```

- [ ] **Step 5: Update `repository.add_segment` + add `get_segment`**

```python
def add_segment(session, *, file_id, start, end, speaker, text, confidence,
                source_pass, flagged, review_status=None, candidates=None,
                clip_original=None, clip_enhanced=None):
    seg = Segment(file_id=file_id, start=start, end=end, speaker=speaker,
                  text=text, confidence=confidence, source_pass=source_pass,
                  flagged=flagged, review_status=review_status,
                  candidates=candidates or {}, clip_original=clip_original,
                  clip_enhanced=clip_enhanced)
    session.add(seg)
    session.flush()
    return seg.id


def get_segment(session, segment_id: str):
    return session.get(Segment, segment_id)
```

- [ ] **Step 6: Update `pipeline/tasks.py` `_l5_l6_segments`** — keep the clip paths and pass `candidates` + clip paths into `add_segment`. Change the `repo.add_segment(...)` call to include:

```python
        seg_id = repo.add_segment(
            session, file_id=job.file_id, start=region["start"], end=region["end"],
            speaker="+".join(spk["speakers"]), text=winning,
            confidence=verdict["confidence"], source_pass=source_pass,
            flagged=verdict["flagged"],
            review_status="pending" if verdict["flagged"] else "auto_accepted",
            candidates=texts, clip_original=clip_org, clip_enhanced=clip_enh)
```

- [ ] **Step 7: Run test + full suite. Commit** `feat: phase5 segment candidates + clip paths (migration 0003)`.

---

### Task 2: L8 certified-transcript writer + pipeline integration

**Files:**
- Create: `services/transcript_service.py`
- Modify: `pipeline/tasks.py` (L8 stage writes transcript; drop placeholder walk)
- Test: `tests/test_transcript_service.py`

**Interfaces:**
- `transcript_service.build(case_id, file_id, source_hash, segments, *, status) -> dict` — assembles the v2 transcript schema.
- `transcript_service.write(case_id, file_id, data) -> str` — writes `final/{file_id}_certified_transcript.json`.
- `transcript_service.final_path(case_id, file_id) -> str`.

- [ ] **Step 1: Failing test**

```python
# tests/test_transcript_service.py
import json
from services import transcript_service as ts


def test_build_and_write(monkeypatch, tmp_path):
    monkeypatch.setattr(ts.storage.config, "CASE_STORE_PATH", str(tmp_path))

    class Seg:
        def __init__(self, **k): self.__dict__.update(k)

    segs = [Seg(id="seg1", start=0.0, end=1.0, speaker="Speaker_1", text="hi",
                confidence=0.6, source_pass="pass1_enhanced", flagged=True,
                review_status="pending")]
    data = ts.build("c1", "f1", "hashA", segs,
                    status="machine_assisted_pending_certification")
    assert data["source_hash_sha256"] == "hashA"
    assert data["segments"][0]["flagged_for_review"] is True
    assert data["status"] == "machine_assisted_pending_certification"

    path = ts.write("c1", "f1", data)
    assert json.load(open(path))["file_id"] == "f1"
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement `services/transcript_service.py`**

```python
import json

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
            "text": s.text, "language": None,
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
```

- [ ] **Step 4: Wire L8 into `pipeline/tasks.py`** — remove the `PLACEHOLDER_STAGES` walk block entirely and replace with an L8 stage, then final status:

```python
    # L8 output generation.
    try:
        with get_session() as s:
            job = repo.get_job(s, job_id)
            repo.update_job(s, job_id, stage="L8"); s.commit()
            segs = repo.list_segments(s, job.file_id)
            src_hash = repo.get_file(s, job.file_id).source_sha256
            data = ts.build(job.case_id, job.file_id, src_hash, segs,
                            status="machine_assisted_pending_certification")
            ts.write(job.case_id, job.file_id, data)
            au.append_entry(job.case_id, file_id=job.file_id, stage="L8",
                            parameters={"segments": len(segs)}, session=s)
            s.commit()
            repo.update_job(s, job_id, status=JobStatus.NEEDS_REVIEW); s.commit()
        return JobStatus.NEEDS_REVIEW
    except Exception as e:
        with get_session() as s:
            repo.update_job(s, job_id, status=JobStatus.FAILED, error=str(e)); s.commit()
        raise
```

Add import at top of `pipeline/tasks.py`: `from services import transcript_service as ts`. Delete `PLACEHOLDER_STAGES` and its `for stage in PLACEHOLDER_STAGES` loop.

- [ ] **Step 5: Run `tests/test_transcript_service.py` + `tests/test_pipeline_attribution.py`. Commit** `feat: phase5 L8 certified-transcript output`.

---

### Task 3: Review REST API — queue, segment detail, decision, clip audio

**Files:**
- Create: `api/routes/review.py`
- Modify: `app.py` (include review router)
- Modify: `db/repository.py` (`list_flagged_segments`, `add_review`, `set_segment_review`)
- Test: `tests/test_review_api.py`

**Interfaces:**
- `GET /review/queue?case_id=&status=pending` → `[{segment_id, file_id, start, end, speaker, confidence, review_status, flag_reason?}]` (flagged only).
- `GET /review/segments/{segment_id}` → `{segment_id, candidates, clip_original_url, clip_enhanced_url, confidence, review_status}`.
- `GET /review/segments/{segment_id}/clip/{which}` → audio FileResponse (`which ∈ original|enhanced`).
- `POST /review/segments/{segment_id}` body `{decision: accept|edit|reject, text?, reviewer_id}` → updates segment review_status (`accepted|edited|rejected`), writes a `Review`, ledger entry.
- repo: `list_flagged_segments(session, case_id, review_status=None)`, `add_review(session, segment_id, decision, text, reviewer_id)`, `set_segment_review(session, segment_id, review_status, text=None)`.

- [ ] **Step 1: Failing test**

```python
# tests/test_review_api.py
import pytest
from fastapi.testclient import TestClient
from db import base as dbbase, repository as repo
import app as appmod

client = TestClient(appmod.app)


def setup_module():
    dbbase.init_db()


def _seed_flagged():
    with dbbase.get_session() as s:
        c = repo.create_case(s); f = repo.create_file(s, c, "a.wav", ".wav")
        seg = repo.add_segment(s, file_id=f, start=0.0, end=1.0, speaker="Speaker_1",
            text="hi", confidence=0.3, source_pass="pass1_enhanced", flagged=True,
            review_status="pending",
            candidates={"pass1_enhanced": "hi", "pass2_original": "hi there",
                        "pass3_indic": "hi"})
        s.commit()
    return c, f, seg


def test_queue_lists_flagged():
    c, f, seg = _seed_flagged()
    r = client.get(f"/review/queue?case_id={c}&status=pending")
    assert r.status_code == 200
    ids = [x["segment_id"] for x in r.json()]
    assert seg in ids


def test_segment_detail_returns_candidates():
    c, f, seg = _seed_flagged()
    r = client.get(f"/review/segments/{seg}")
    assert r.status_code == 200
    assert r.json()["candidates"]["pass2_original"] == "hi there"


def test_submit_edit_decision():
    c, f, seg = _seed_flagged()
    r = client.post(f"/review/segments/{seg}",
                    json={"decision": "edit", "text": "hi there friend",
                          "reviewer_id": "officer_7"})
    assert r.status_code == 200
    with dbbase.get_session() as s:
        updated = repo.get_segment(s, seg)
        assert updated.review_status == "edited"
        assert updated.text == "hi there friend"


def test_reject_decision():
    c, f, seg = _seed_flagged()
    r = client.post(f"/review/segments/{seg}",
                    json={"decision": "reject", "reviewer_id": "officer_7"})
    assert r.status_code == 200
    with dbbase.get_session() as s:
        assert repo.get_segment(s, seg).review_status == "rejected"
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Add repo helpers** — append to `db/repository.py` (uses `Review` import):

```python
from db.models import Review


def list_flagged_segments(session, case_id, review_status=None):
    q = (session.query(Segment).join(File, Segment.file_id == File.id)
         .filter(File.case_id == case_id, Segment.flagged == True))  # noqa: E712
    if review_status:
        q = q.filter(Segment.review_status == review_status)
    return q.order_by(Segment.confidence).all()


def add_review(session, segment_id, decision, text, reviewer_id):
    r = Review(segment_id=segment_id, decision=decision, text=text,
               reviewer_id=reviewer_id)
    session.add(r)
    session.flush()
    return r.id


def set_segment_review(session, segment_id, review_status, text=None):
    seg = session.get(Segment, segment_id)
    seg.review_status = review_status
    if text is not None:
        seg.text = text
    session.flush()
    return seg


def count_pending_flagged(session, file_id):
    return (session.query(Segment)
            .filter(Segment.file_id == file_id, Segment.flagged == True,  # noqa: E712
                    Segment.review_status == "pending").count())
```

- [ ] **Step 4: Implement `api/routes/review.py`**

```python
import os

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from api.auth import require_api_key
from db.base import get_session
from db import repository as repo
from services import audit_service as au

router = APIRouter(prefix="/review", dependencies=[Depends(require_api_key)])

_DECISION_TO_STATUS = {"accept": "accepted", "edit": "edited", "reject": "rejected"}


class ReviewDecision(BaseModel):
    decision: str
    text: Optional[str] = None
    reviewer_id: str


@router.get("/queue")
def queue(case_id: str, status: str = "pending"):
    with get_session() as s:
        segs = repo.list_flagged_segments(s, case_id, review_status=status)
        return [{"segment_id": x.id, "file_id": x.file_id, "start": x.start,
                 "end": x.end, "speaker": x.speaker, "confidence": x.confidence,
                 "review_status": x.review_status} for x in segs]


@router.get("/segments/{segment_id}")
def segment_detail(segment_id: str):
    with get_session() as s:
        seg = repo.get_segment(s, segment_id)
        if seg is None:
            raise HTTPException(status_code=404, detail="Unknown segment_id")
        return {"segment_id": seg.id, "candidates": seg.candidates or {},
                "clip_original_url": f"/review/segments/{seg.id}/clip/original",
                "clip_enhanced_url": f"/review/segments/{seg.id}/clip/enhanced",
                "confidence": seg.confidence, "review_status": seg.review_status,
                "text": seg.text}


@router.get("/segments/{segment_id}/clip/{which}")
def segment_clip(segment_id: str, which: str):
    with get_session() as s:
        seg = repo.get_segment(s, segment_id)
        if seg is None:
            raise HTTPException(status_code=404, detail="Unknown segment_id")
        path = seg.clip_enhanced if which == "enhanced" else seg.clip_original
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Clip not available")
    return FileResponse(path, media_type="audio/wav")


@router.post("/segments/{segment_id}")
def submit_decision(segment_id: str, body: ReviewDecision):
    if body.decision not in _DECISION_TO_STATUS:
        raise HTTPException(status_code=400, detail="decision must be accept|edit|reject")
    with get_session() as s:
        seg = repo.get_segment(s, segment_id)
        if seg is None:
            raise HTTPException(status_code=404, detail="Unknown segment_id")
        new_text = body.text if body.decision == "edit" else None
        repo.add_review(s, segment_id, body.decision, body.text, body.reviewer_id)
        repo.set_segment_review(s, segment_id, _DECISION_TO_STATUS[body.decision],
                                text=new_text)
        s.commit()
        au.append_entry(seg.file_id and _case_of(s, seg.file_id), file_id=seg.file_id,
                        stage="L7_review",
                        parameters={"segment_id": segment_id,
                                    "decision": body.decision,
                                    "reviewer_id": body.reviewer_id}, session=s)
        s.commit()
    return {"segment_id": segment_id, "review_status": _DECISION_TO_STATUS[body.decision]}


def _case_of(session, file_id):
    f = repo.get_file(session, file_id)
    return f.case_id if f else "unknown"
```

- [ ] **Step 5: Include router** — in `app.py`:

```python
from api.routes.review import router as review_router
...
app.include_router(review_router)
```

- [ ] **Step 6: Run test + full suite. Commit** `feat: phase5 human QA review REST API`.

---

### Task 4: Certification endpoint + gating

**Files:**
- Modify: `api/routes/review.py` (add certify endpoint) — or `api/routes/cases.py`. Put it in `cases.py`.
- Test: `tests/test_certify_api.py`

**Interfaces:**
- `POST /cases/{case_id}/files/{file_id}/certify` → `200 {"status":"certified"}` when zero pending flagged; `409` otherwise. Flips job → `CERTIFIED`, rewrites transcript `status="certified"`, ledger entry.

- [ ] **Step 1: Failing test**

```python
# tests/test_certify_api.py
import json
from fastapi.testclient import TestClient
from db import base as dbbase, repository as repo
from db.models import JobStatus
from services import transcript_service as ts, storage
import app as appmod

client = TestClient(appmod.app)


def setup_module():
    dbbase.init_db()


def _seed(tmp_path, monkeypatch, flagged_pending: bool):
    for mod in (ts.storage.config, storage.config):
        monkeypatch.setattr(mod, "CASE_STORE_PATH", str(tmp_path))
    with dbbase.get_session() as s:
        c = repo.create_case(s); f = repo.create_file(s, c, "a.wav", ".wav")
        repo.set_file_hash(s, f, "hashA")
        j = repo.create_job(s, c, f)
        repo.add_segment(s, file_id=f, start=0.0, end=1.0, speaker="Speaker_1",
            text="hi", confidence=0.3, source_pass="pass1_enhanced",
            flagged=flagged_pending,
            review_status="pending" if flagged_pending else "auto_accepted")
        s.commit()
    # write an initial transcript artifact
    with dbbase.get_session() as s:
        segs = repo.list_segments(s, f)
        ts.write(c, f, ts.build(c, f, "hashA", segs,
                 status="machine_assisted_pending_certification"))
    return c, f, j


def test_certify_blocked_when_pending(tmp_path, monkeypatch):
    c, f, j = _seed(tmp_path, monkeypatch, flagged_pending=True)
    r = client.post(f"/cases/{c}/files/{f}/certify")
    assert r.status_code == 409


def test_certify_succeeds_when_clear(tmp_path, monkeypatch):
    c, f, j = _seed(tmp_path, monkeypatch, flagged_pending=False)
    r = client.post(f"/cases/{c}/files/{f}/certify")
    assert r.status_code == 200
    with dbbase.get_session() as s:
        assert repo.get_job(s, j).status == JobStatus.CERTIFIED
    data = json.load(open(ts.final_path(c, f)))
    assert data["status"] == "certified"
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Add `repository.latest_job_for_file`** — append to `db/repository.py`:

```python
def latest_job_for_file(session, file_id):
    return (session.query(Job).filter(Job.file_id == file_id)
            .order_by(Job.created_at.desc()).first())
```

- [ ] **Step 4: Add certify endpoint to `api/routes/cases.py`**

```python
from db.models import JobStatus
from services import audit_service as au
from services import transcript_service as ts
import json as _json


@router.post("/cases/{case_id}/files/{file_id}/certify",
             dependencies=[Depends(require_api_key)])
def certify(case_id: str, file_id: str):
    with get_session() as s:
        pending = repo.count_pending_flagged(s, file_id)
        if pending > 0:
            raise HTTPException(status_code=409,
                                detail=f"{pending} flagged segment(s) still pending review")
        # Flip the persisted transcript to certified.
        path = ts.final_path(case_id, file_id)
        if os.path.exists(path):
            data = _json.load(open(path))
            data["status"] = "certified"
            ts.write(case_id, file_id, data)
        job = repo.latest_job_for_file(s, file_id)
        if job is not None:
            repo.update_job(s, job.id, status=JobStatus.CERTIFIED)
        s.commit()
        au.append_entry(case_id, file_id=file_id, stage="certify",
                        parameters={"result": "certified"}, session=s)
        s.commit()
    return {"status": "certified"}
```

(Add the `get_session`/`repo` imports if not already present in `cases.py` — they are.)

- [ ] **Step 5: Run test + full suite. Commit** `feat: phase5 certification endpoint + gating`.

---

## Phase 5 Done — Definition of Done

- Pipeline writes `final/{file_id}_certified_transcript.json` (status `machine_assisted_pending_certification`).
- `/review/queue`, `/review/segments/{id}` (+ candidates + clip URLs), `/review/segments/{id}/clip/{which}`, and `POST /review/segments/{id}` (accept/edit/reject) work; each logged to the ledger.
- `POST /cases/{id}/files/{fid}/certify` 409s while any flagged segment is pending, else flips job → `CERTIFIED` and transcript → `certified`.
- All prior tests green (pre-existing alignment failure excepted).

**This completes the v2 build (L0–L9).**
