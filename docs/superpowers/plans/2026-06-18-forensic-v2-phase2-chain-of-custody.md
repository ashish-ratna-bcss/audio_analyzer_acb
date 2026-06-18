# Forensic v2 — Phase 2 (Chain-of-Custody) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Make the pipeline forensic — SHA-256 the original on first bytes, WORM-store it, build a hash-chained tamper-evident audit ledger (JSONL source of truth + Postgres mirror), normalize audio to dual-rate, and quarantine bad inputs. Replaces the skeleton `L0`/`L1` stages with real work.

**Architecture:** Pure helper modules (`hashing`, `audit_service`, `storage`, `manifest_service`) carry all logic and are unit-tested without models. The Celery `run_pipeline` calls L0 (ingest+hash) then L1 (ffmpeg dual-rate), each appending a ledger entry. ffmpeg is invoked on real tiny WAVs generated in-test (ffmpeg is a system tool, not a model — allowed locally).

**Tech Stack:** hashlib, ffmpeg (cli), SQLAlchemy, JSON Lines.

## Global Constraints

- No ML model execution locally. ffmpeg CLI use is allowed (system tool).
- Ledger JSONL at `{CASE_STORE_PATH}/audit/{case_id}/ledger.jsonl` is source of truth; Postgres `audit_entries` is a mirror.
- `entry_hash = sha256(prev_entry_hash || canonical_json(payload))`, hex. First entry uses `prev_entry_hash = ""`.
- Originals are written once then made read-only (`chattr +i` best-effort; on failure fall back to `chmod 0444` and log — never hard-fail the job on immutability).
- All existing Phase 1 tests stay green.

---

### Task 1: SHA-256 hashing utility

**Files:**
- Create: `services/hashing.py`
- Test: `tests/test_hashing.py`

**Interfaces:**
- Produces: `sha256_bytes(data: bytes) -> str`, `sha256_file(path: str) -> str` (streamed, 1MB chunks).

- [ ] **Step 1: Failing test**

```python
# tests/test_hashing.py
import hashlib
from services.hashing import sha256_bytes, sha256_file


def test_sha256_bytes_matches_hashlib():
    data = b"forensic evidence bytes"
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_sha256_file_streams(tmp_path):
    p = tmp_path / "a.bin"
    payload = b"x" * (1024 * 1024 + 7)  # > one chunk
    p.write_bytes(payload)
    assert sha256_file(str(p)) == hashlib.sha256(payload).hexdigest()
```

- [ ] **Step 2: Run, expect fail** — `pytest tests/test_hashing.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement `services/hashing.py`**

```python
import hashlib

_CHUNK = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()
```

- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** `feat: phase2 sha256 hashing utility`.

---

### Task 2: Hash-chained audit ledger service

**Files:**
- Create: `services/audit_service.py`
- Modify: `db/repository.py` (add `add_audit_entry`, `list_audit_entries`)
- Test: `tests/test_audit_service.py`

**Interfaces:**
- Consumes: `services.hashing.sha256_bytes`, `config.CASE_STORE_PATH`, `db` repository.
- Produces:
  - `audit_service.ledger_path(case_id) -> str`
  - `audit_service.append_entry(case_id, *, file_id, stage, model=None, model_checkpoint_sha256=None, parameters=None, input_hash=None, output_hash=None, operator="pipeline", session=None) -> dict` — computes chain hash, appends JSONL line, mirrors to Postgres when `session` given, returns the entry dict.
  - `audit_service.verify_chain(case_id) -> bool` — recompute chain, detect tamper.
  - `db.repository.add_audit_entry(session, **entry)` , `list_audit_entries(session, case_id) -> list`.

- [ ] **Step 1: Failing test**

```python
# tests/test_audit_service.py
import json
from services import audit_service as au


def _read_lines(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def test_chain_links_and_verifies(monkeypatch, tmp_path):
    monkeypatch.setattr(au.config, "CASE_STORE_PATH", str(tmp_path))
    e1 = au.append_entry("case-x", file_id="f1", stage="L0", output_hash="aaa")
    e2 = au.append_entry("case-x", file_id="f1", stage="L1", input_hash="aaa",
                         output_hash="bbb")
    assert e1["prev_entry_hash"] == ""
    assert e2["prev_entry_hash"] == e1["entry_hash"]
    assert au.verify_chain("case-x") is True

    lines = _read_lines(au.ledger_path("case-x"))
    assert len(lines) == 2 and lines[1]["stage"] == "L1"


def test_tamper_breaks_chain(monkeypatch, tmp_path):
    monkeypatch.setattr(au.config, "CASE_STORE_PATH", str(tmp_path))
    au.append_entry("case-y", file_id="f1", stage="L0", output_hash="aaa")
    au.append_entry("case-y", file_id="f1", stage="L1", output_hash="bbb")
    p = au.ledger_path("case-y")
    lines = open(p).read().splitlines()
    rec = json.loads(lines[0]); rec["output_hash"] = "TAMPERED"
    lines[0] = json.dumps(rec)
    open(p, "w").write("\n".join(lines) + "\n")
    assert au.verify_chain("case-y") is False
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement `services/audit_service.py`**

```python
import os
import json
from datetime import datetime, timezone

import config
from services.hashing import sha256_bytes

# Fields hashed into the chain (order fixed for determinism). entry_hash and
# prev_entry_hash are intentionally excluded from the hashed payload.
_PAYLOAD_FIELDS = ["timestamp", "case_id", "file_id", "stage", "model",
                   "model_checkpoint_sha256", "parameters", "input_hash",
                   "output_hash", "operator"]


def ledger_path(case_id: str) -> str:
    return os.path.join(config.CASE_STORE_PATH, "audit", case_id, "ledger.jsonl")


def _canonical(entry: dict) -> bytes:
    payload = {k: entry.get(k) for k in _PAYLOAD_FIELDS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _last_entry_hash(path: str) -> str:
    if not os.path.exists(path):
        return ""
    last = ""
    with open(path) as f:
        for line in f:
            if line.strip():
                last = json.loads(line)["entry_hash"]
    return last


def append_entry(case_id, *, file_id, stage, model=None,
                 model_checkpoint_sha256=None, parameters=None,
                 input_hash=None, output_hash=None, operator="pipeline",
                 session=None) -> dict:
    path = ledger_path(case_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prev = _last_entry_hash(path)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "file_id": file_id,
        "stage": stage,
        "model": model,
        "model_checkpoint_sha256": model_checkpoint_sha256,
        "parameters": parameters,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "operator": operator,
        "prev_entry_hash": prev,
    }
    entry["entry_hash"] = sha256_bytes(prev.encode() + _canonical(entry))
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    if session is not None:
        from db import repository as repo
        repo.add_audit_entry(
            session, case_id=case_id, file_id=file_id, stage=stage,
            payload={k: entry[k] for k in _PAYLOAD_FIELDS},
            prev_entry_hash=prev, entry_hash=entry["entry_hash"],
        )
    return entry


def verify_chain(case_id: str) -> bool:
    path = ledger_path(case_id)
    if not os.path.exists(path):
        return True
    prev = ""
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("prev_entry_hash") != prev:
                return False
            expected = sha256_bytes(prev.encode() + _canonical(entry))
            if entry.get("entry_hash") != expected:
                return False
            prev = entry["entry_hash"]
    return True
```

- [ ] **Step 4: Add repository functions** — append to `db/repository.py`:

```python
from db.models import AuditEntry


def add_audit_entry(session, *, case_id, file_id, stage, payload,
                    prev_entry_hash, entry_hash):
    e = AuditEntry(case_id=case_id, file_id=file_id, stage=stage,
                   payload=payload, prev_entry_hash=prev_entry_hash,
                   entry_hash=entry_hash)
    session.add(e)
    session.flush()
    return e


def list_audit_entries(session, case_id):
    return (session.query(AuditEntry)
            .filter(AuditEntry.case_id == case_id)
            .order_by(AuditEntry.id).all())
```

- [ ] **Step 5: Run, expect pass.**
- [ ] **Step 6: Commit** `feat: phase2 hash-chained audit ledger (jsonl + pg mirror)`.

---

### Task 3: Storage + manifest service

**Files:**
- Create: `services/storage.py`
- Create: `services/manifest_service.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- `storage.case_dir(case_id)`, `storage.originals_dir(case_id)`, `storage.derivatives_dir(case_id, file_id)`, `storage.derivative_path(case_id, file_id, subdir, filename)` — all return paths and `os.makedirs` the parent.
- `storage.write_original(case_id, file_id, ext, src_path) -> (dest_path, sha256)` — moves the staged upload into `originals/{file_id}__original{ext}`, hashes it, makes it read-only (best-effort immutability).
- `manifest_service.manifest_path(case_id)`, `manifest_service.load(case_id) -> dict`, `manifest_service.register_file(case_id, file_id, original_filename, sha256)`, `manifest_service.register_derivative(case_id, file_id, kind, path, sha256, parent_sha256)`.

- [ ] **Step 1: Failing test**

```python
# tests/test_storage.py
from services import storage, manifest_service as man
from services.hashing import sha256_file


def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "CASE_STORE_PATH", str(tmp_path))
    monkeypatch.setattr(man.config, "CASE_STORE_PATH", str(tmp_path))


def test_write_original_moves_hashes_and_locks(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    src = tmp_path / "staged.wav"
    src.write_bytes(b"RIFFfake")
    dest, digest = storage.write_original("c1", "f1", ".wav", str(src))
    assert dest.endswith("f1__original.wav")
    assert digest == sha256_file(dest)
    assert not src.exists()  # moved, not copied


def test_manifest_lineage(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    man.register_file("c1", "f1", "REC001.wav", "hashA")
    man.register_derivative("c1", "f1", "normalized_16k", "/p/16k.wav",
                            "hashB", parent_sha256="hashA")
    m = man.load("c1")
    assert m["files"]["f1"]["source_sha256"] == "hashA"
    deriv = m["files"]["f1"]["derivatives"][0]
    assert deriv["kind"] == "normalized_16k"
    assert deriv["parent_sha256"] == "hashA" and deriv["sha256"] == "hashB"
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement `services/storage.py`**

```python
import os
import shutil
import stat
import subprocess

import config
from services.hashing import sha256_file


def case_dir(case_id: str) -> str:
    return os.path.join(config.CASE_STORE_PATH, "cases", case_id)


def originals_dir(case_id: str) -> str:
    p = os.path.join(case_dir(case_id), "originals")
    os.makedirs(p, exist_ok=True)
    return p


def derivatives_dir(case_id: str, file_id: str) -> str:
    p = os.path.join(case_dir(case_id), "derivatives", file_id)
    os.makedirs(p, exist_ok=True)
    return p


def derivative_path(case_id: str, file_id: str, subdir: str, filename: str) -> str:
    p = os.path.join(derivatives_dir(case_id, file_id), subdir)
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, filename)


def _make_immutable(path: str) -> None:
    # Best-effort WORM: try chattr +i, else fall back to read-only perms.
    try:
        subprocess.run(["chattr", "+i", path], capture_output=True, check=True)
        return
    except Exception:
        pass
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass


def write_original(case_id: str, file_id: str, ext: str, src_path: str):
    dest = os.path.join(originals_dir(case_id), f"{file_id}__original{ext}")
    shutil.move(src_path, dest)
    digest = sha256_file(dest)
    _make_immutable(dest)
    return dest, digest
```

- [ ] **Step 4: Implement `services/manifest_service.py`**

```python
import os
import json

import config


def manifest_path(case_id: str) -> str:
    d = os.path.join(config.CASE_STORE_PATH, "cases", case_id)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "manifest.json")


def load(case_id: str) -> dict:
    p = manifest_path(case_id)
    if not os.path.exists(p):
        return {"case_id": case_id, "files": {}}
    with open(p) as f:
        return json.load(f)


def _save(case_id: str, data: dict) -> None:
    with open(manifest_path(case_id), "w") as f:
        json.dump(data, f, indent=2)


def register_file(case_id, file_id, original_filename, sha256):
    m = load(case_id)
    m["files"].setdefault(file_id, {})
    m["files"][file_id].update({
        "original_filename": original_filename,
        "source_sha256": sha256,
        "derivatives": m["files"][file_id].get("derivatives", []),
    })
    _save(case_id, m)


def register_derivative(case_id, file_id, kind, path, sha256, parent_sha256):
    m = load(case_id)
    entry = m["files"].setdefault(file_id, {"derivatives": []})
    entry.setdefault("derivatives", []).append({
        "kind": kind, "path": path, "sha256": sha256,
        "parent_sha256": parent_sha256,
    })
    _save(case_id, m)
```

- [ ] **Step 5: Run, expect pass.**
- [ ] **Step 6: Commit** `feat: phase2 storage (WORM originals) + manifest lineage`.

---

### Task 4: FFmpeg dual-rate normalization

**Files:**
- Modify: `services/ffmpeg_service.py` (add `convert_dual_rate`, keep `convert_to_wav`, `measure_mean_volume`)
- Test: `tests/test_ffmpeg_dual_rate.py`

**Interfaces:**
- `ffmpeg_service.convert_dual_rate(input_path, out_48k, out_16k) -> (out_48k, out_16k)` — 48 kHz mono + 16 kHz mono, same decode recipe (`-map 0:a:0?`, `aresample=async=1:first_pts=0`). Raises `UnsupportedFormatError` / `RuntimeError` as today.

- [ ] **Step 1: Failing test** (ffmpeg CLI is allowed — generates a real tone)

```python
# tests/test_ffmpeg_dual_rate.py
import subprocess, shutil, wave
import pytest
from services.ffmpeg_service import convert_dual_rate

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _make_tone(path):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=1", "-ar", "22050", str(path)],
                   capture_output=True, check=True)


def test_dual_rate_outputs(tmp_path):
    src = tmp_path / "tone.wav"; _make_tone(src)
    o48 = tmp_path / "48k.wav"; o16 = tmp_path / "16k.wav"
    convert_dual_rate(str(src), str(o48), str(o16))
    with wave.open(str(o48)) as w:
        assert w.getframerate() == 48000 and w.getnchannels() == 1
    with wave.open(str(o16)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
```

- [ ] **Step 2: Run, expect fail** (`ImportError: cannot import name 'convert_dual_rate'`).

- [ ] **Step 3: Implement** — append to `services/ffmpeg_service.py`:

```python
def _ffmpeg_pcm(input_path: str, output_path: str, sample_rate: int) -> str:
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-map", "0:a:0?",
        "-acodec", "pcm_s16le",
        "-af", "aresample=async=1:first_pts=0",
        "-ac", "1", "-ar", str(sample_rate),
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    return output_path


def convert_dual_rate(input_path: str, out_48k: str, out_16k: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(f"Unsupported format: {ext}")
    _ffmpeg_pcm(input_path, out_48k, 48000)
    _ffmpeg_pcm(input_path, out_16k, 16000)
    return out_48k, out_16k
```

- [ ] **Step 4: Run, expect pass** (or skip if no ffmpeg — still committed).
- [ ] **Step 5: Commit** `feat: phase2 ffmpeg dual-rate (48k+16k) normalization`.

---

### Task 5: L0/L1 pipeline integration + reconciliation + quarantine

**Files:**
- Create: `pipeline/reconcile.py`
- Modify: `api/routes/cases.py` (stage upload as `inbox/{file_id}{ext}`, store nothing else)
- Modify: `pipeline/tasks.py` (real L0 + L1; ledger entries; quarantine on ffmpeg failure)
- Modify: `db/repository.py` (add `set_file_hash`, `get_file`)
- Test: `tests/test_reconcile.py`, extend `tests/test_pipeline_skeleton.py` → add `tests/test_pipeline_l0_l1.py`

**Interfaces:**
- `pipeline.reconcile.check(stage_in, count_in, stage_out, count_out)` — raises `ReconciliationError` if `count_out < count_in` (silent-drop guard).
- `db.repository.set_file_hash(session, file_id, sha256)`, `get_file(session, file_id)`.
- `pipeline.tasks.run_pipeline` real stages: L0 ingest→hash→WORM→manifest→`set_file_hash`→ledger; L1 dual-rate→register derivatives→ledger; on ffmpeg failure set job + file status `quarantined`, ledger a `quarantine` entry, return `JobStatus.QUARANTINED`. L2–L8 remain placeholder.

- [ ] **Step 1: Failing test — reconciliation**

```python
# tests/test_reconcile.py
import pytest
from pipeline.reconcile import check, ReconciliationError


def test_equal_counts_ok():
    check("L0", 1, "L1", 1)  # no raise


def test_drop_raises():
    with pytest.raises(ReconciliationError):
        check("L3", 10, "L4", 7)
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement `pipeline/reconcile.py`**

```python
class ReconciliationError(Exception):
    """A layer emitted fewer units than it received — possible silent drop."""


def check(stage_in: str, count_in: int, stage_out: str, count_out: int) -> None:
    if count_out < count_in:
        raise ReconciliationError(
            f"{stage_in}->{stage_out} dropped units: {count_in} -> {count_out}")
```

- [ ] **Step 4: Add repository helpers** — append to `db/repository.py`:

```python
def get_file(session, file_id: str):
    return session.get(File, file_id)


def set_file_hash(session, file_id: str, sha256: str):
    f = session.get(File, file_id)
    f.source_sha256 = sha256
    session.flush()
    return f


def set_file_status(session, file_id: str, status: str):
    f = session.get(File, file_id)
    f.status = status
    session.flush()
    return f
```

- [ ] **Step 5: Failing test — L0/L1 integration**

```python
# tests/test_pipeline_l0_l1.py
import os, shutil, subprocess
import pytest
from db import base as dbbase
from db import repository as repo
from db.models import JobStatus
from services import audit_service as au, manifest_service as man
from pipeline import tasks as ptasks

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def setup_module():
    dbbase.init_db()


def _make_tone(path):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=1", "-ar", "22050", str(path)],
                   capture_output=True, check=True)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_l0_l1_hashes_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(ptasks.config, "CASE_STORE_PATH", str(tmp_path))
    monkeypatch.setattr(au.config, "CASE_STORE_PATH", str(tmp_path))
    monkeypatch.setattr(man.config, "CASE_STORE_PATH", str(tmp_path))
    from services import storage
    monkeypatch.setattr(storage.config, "CASE_STORE_PATH", str(tmp_path))

    with dbbase.get_session() as s:
        case_id = repo.create_case(s)
        file_id = repo.create_file(s, case_id, "tone.wav", ".wav")
        job_id = repo.create_job(s, case_id, file_id)
        s.commit()

    inbox = os.path.join(str(tmp_path), "cases", case_id, "inbox")
    os.makedirs(inbox, exist_ok=True)
    _make_tone(os.path.join(inbox, f"{file_id}.wav"))

    result = ptasks.run_pipeline.apply(args=[job_id]).get()
    assert result == JobStatus.NEEDS_REVIEW

    with dbbase.get_session() as s:
        f = repo.get_file(s, file_id)
        assert f.source_sha256 and len(f.source_sha256) == 64
    assert au.verify_chain(case_id) is True
    m = man.load(case_id)
    assert m["files"][file_id]["source_sha256"] == f.source_sha256


def test_quarantine_on_missing_input(tmp_path, monkeypatch):
    monkeypatch.setattr(ptasks.config, "CASE_STORE_PATH", str(tmp_path))
    monkeypatch.setattr(au.config, "CASE_STORE_PATH", str(tmp_path))
    monkeypatch.setattr(man.config, "CASE_STORE_PATH", str(tmp_path))
    from services import storage
    monkeypatch.setattr(storage.config, "CASE_STORE_PATH", str(tmp_path))

    with dbbase.get_session() as s:
        case_id = repo.create_case(s)
        file_id = repo.create_file(s, case_id, "missing.wav", ".wav")
        job_id = repo.create_job(s, case_id, file_id)
        s.commit()
    # no inbox file staged -> L0 cannot find original -> quarantine
    result = ptasks.run_pipeline.apply(args=[job_id]).get()
    assert result == JobStatus.QUARANTINED
    with dbbase.get_session() as s:
        assert repo.get_job(s, job_id).status == JobStatus.QUARANTINED
```

- [ ] **Step 6: Run, expect fail.**

- [ ] **Step 7: Rewrite `pipeline/tasks.py`**

```python
import os

import config
from pipeline.celery_app import celery
from pipeline import reconcile
from db.base import get_session
from db import repository as repo
from db.models import JobStatus
from services import audit_service as au
from services import storage
from services import manifest_service as man
from services.hashing import sha256_file
from services.ffmpeg_service import convert_dual_rate, UnsupportedFormatError

# L2-L8 are still placeholder; Phases 3-5 replace them with real layer tasks.
PLACEHOLDER_STAGES = ["L2", "L3", "L4", "L5", "L6", "L7", "L8"]


def _inbox_original(case_id: str, file_id: str, ext: str) -> str:
    return os.path.join(config.CASE_STORE_PATH, "cases", case_id, "inbox",
                        f"{file_id}{ext}")


def _l0_ingest(job, session) -> str:
    """Hash the byte-exact original, WORM-store it, register manifest + file row.
    Returns the source sha256. Raises FileNotFoundError if no staged input."""
    ext = repo.get_file(session, job.file_id).ext
    staged = _inbox_original(job.case_id, job.file_id, ext)
    if not os.path.exists(staged):
        raise FileNotFoundError(f"no staged original for file {job.file_id}")
    dest, digest = storage.write_original(job.case_id, job.file_id, ext, staged)
    fname = repo.get_file(session, job.file_id).original_filename
    man.register_file(job.case_id, job.file_id, fname, digest)
    repo.set_file_hash(session, job.file_id, digest)
    session.commit()
    au.append_entry(job.case_id, file_id=job.file_id, stage="L0",
                    output_hash=digest, session=session)
    session.commit()
    return dest, digest


def _l1_normalize(job, original_path: str, source_hash: str, session):
    out48 = storage.derivative_path(job.case_id, job.file_id, "normalized",
                                    f"{job.file_id}_48k.wav")
    out16 = storage.derivative_path(job.case_id, job.file_id, "normalized",
                                    f"{job.file_id}_16k_mono.wav")
    convert_dual_rate(original_path, out48, out16)
    for kind, path in [("normalized_48k", out48), ("normalized_16k", out16)]:
        h = sha256_file(path)
        man.register_derivative(job.case_id, job.file_id, kind, path, h,
                                parent_sha256=source_hash)
        au.append_entry(job.case_id, file_id=job.file_id, stage="L1",
                        input_hash=source_hash, output_hash=h, session=session)
    session.commit()
    reconcile.check("L0", 1, "L1", 1)
    return out48, out16


def _quarantine(job_id, case_id, file_id, reason: str):
    with get_session() as s:
        repo.update_job(s, job_id, status=JobStatus.QUARANTINED, error=reason)
        repo.set_file_status(s, file_id, "quarantined")
        s.commit()
    au.append_entry(case_id, file_id=file_id, stage="quarantine",
                    parameters={"reason": reason})


@celery.task(name="pipeline.run_pipeline")
def run_pipeline(job_id: str) -> str:
    with get_session() as s:
        job = repo.get_job(s, job_id)
        if job is None:
            raise ValueError(f"job not found: {job_id}")
        case_id, file_id = job.case_id, job.file_id
        repo.update_job(s, job_id, status=JobStatus.RUNNING, stage="L0")
        s.commit()

    # L0 + L1 with quarantine on bad/missing input.
    try:
        with get_session() as s:
            job = repo.get_job(s, job_id)
            original_path, source_hash = _l0_ingest(job, s)
        with get_session() as s:
            job = repo.get_job(s, job_id)
            repo.update_job(s, job_id, stage="L1")
            s.commit()
            _l1_normalize(job, original_path, source_hash, s)
    except (FileNotFoundError, UnsupportedFormatError, RuntimeError) as e:
        _quarantine(job_id, case_id, file_id, str(e))
        return JobStatus.QUARANTINED

    # Placeholder remainder (Phases 3-5).
    try:
        for stage in PLACEHOLDER_STAGES:
            with get_session() as s:
                repo.update_job(s, job_id, stage=stage)
                s.commit()
        with get_session() as s:
            repo.update_job(s, job_id, status=JobStatus.NEEDS_REVIEW)
            s.commit()
        return JobStatus.NEEDS_REVIEW
    except Exception as e:
        with get_session() as s:
            repo.update_job(s, job_id, status=JobStatus.FAILED, error=str(e))
            s.commit()
        raise
```

- [ ] **Step 8: Update `api/routes/cases.py`** — stage upload by file_id. Replace the body of `upload_file` after the `Case` existence check with:

```python
    # Create the file row first so we can name the staged upload by file_id;
    # L0 (pipeline) finds it deterministically at cases/{case}/inbox/{file_id}{ext}.
    with get_session() as s:
        file_id = repo.create_file(s, case_id, audio.filename or f"upload{ext}", ext)
        job_id = repo.create_job(s, case_id, file_id)
        s.commit()

    inbox = os.path.join(config.CASE_STORE_PATH, "cases", case_id, "inbox")
    os.makedirs(inbox, exist_ok=True)
    staged = os.path.join(inbox, f"{file_id}{ext}")
    content = await audio.read()
    if len(content) > config.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    async with aiofiles.open(staged, "wb") as f:
        await f.write(content)

    run_pipeline.delay(job_id)
    return {"file_id": file_id, "job_id": job_id}
```

Remove the now-unused `import uuid` if present.

- [ ] **Step 9: Run new + regression tests**

Run: `pytest tests/test_reconcile.py tests/test_pipeline_l0_l1.py tests/test_api_jobs.py tests/test_pipeline_skeleton.py -v`
Expected: PASS (skips ffmpeg-gated tests if ffmpeg absent). Note: `tests/test_pipeline_skeleton.py::test_run_pipeline_walks_to_needs_review` now needs a staged input OR should be updated to expect QUARANTINED — update that test to assert `JobStatus.QUARANTINED` (no input staged), since the skeleton walk is replaced by real L0.

Update `tests/test_pipeline_skeleton.py` `test_run_pipeline_walks_to_needs_review`:

```python
def test_run_pipeline_quarantines_without_input():
    with dbbase.get_session() as s:
        case_id = repo.create_case(s)
        file_id = repo.create_file(s, case_id, "a.wav", ".wav")
        job_id = repo.create_job(s, case_id, file_id)
        s.commit()
    result = ptasks.run_pipeline.apply(args=[job_id]).get()
    assert result == dbmodels.JobStatus.QUARANTINED
```

(Delete the old `test_run_pipeline_walks_to_needs_review`; keep `test_run_pipeline_marks_failed_on_bad_job_id`.)

- [ ] **Step 10: Run full suite** — `pytest -q -m "not gpu and not model"`; all green except the known pre-existing `test_alignment_service` failure.
- [ ] **Step 11: Commit** `feat: phase2 L0 ingest/hash/WORM + L1 normalize + reconciliation + quarantine`.

---

## Phase 2 Done — Definition of Done

- Upload → L0 hashes byte-exact original, WORM-locks it, writes manifest, sets `file.source_sha256`.
- L1 produces 48k + 16k derivatives, each hashed with parent lineage in the manifest.
- Audit ledger is hash-chained and `verify_chain` detects tampering; Postgres mirrors entries.
- Missing/corrupt input → job + file `quarantined`, never silently dropped.
- All Phase 1 + Phase 2 tests green (pre-existing alignment test failure excepted).

**Next:** Phase 3 (recall branches L2/L3/L2b).
