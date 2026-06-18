# Forensic v2 — Phase 1 (Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the single-server async foundation — Celery/Redis/Postgres infra, database schema, job-based API skeleton, and deploy artifacts — onto which Phases 2–5 attach the pipeline layers.

**Architecture:** FastAPI `api` accepts case/file uploads and enqueues a Celery job; a placeholder pipeline task walks a job through stage names and updates Postgres. SQLAlchemy ORM + Alembic migrations. Everything (api, redis, postgres, workers, flower, nginx) ships in one `docker-compose.yml`. No model inference in this phase.

**Tech Stack:** FastAPI, Celery 5, Redis 7, Postgres 16, SQLAlchemy 2.0, Alembic, Pydantic 2, Docker Compose.

## Global Constraints

- Python 3.12. Pin new deps: `celery==5.4.0`, `redis==5.0.7`, `sqlalchemy==2.0.31`, `alembic==1.13.2`, `psycopg2-binary==2.9.9`.
- **No model execution on the build laptop.** Phase 1 has no models; all tests run locally.
- Tests use **SQLite** (`DATABASE_URL=sqlite+pysqlite:///:memory:`) and **Celery eager mode** (`CELERY_TASK_ALWAYS_EAGER=true`). Deploy uses Postgres + real Redis.
- Device/auth config behavior from existing `config.py` is preserved; only additive changes.
- All IDs are UUID4 strings. Timestamps are timezone-aware UTC.
- Existing `services/*` modules are NOT touched in Phase 1 (carried over in later phases).
- Frequent commits: one per task minimum. TDD: failing test first.

---

### Task 1: Dependencies + config foundation

**Files:**
- Modify: `requirements.txt`
- Modify: `config.py`
- Test: `tests/test_config_phase1.py`

**Interfaces:**
- Produces: `config.DATABASE_URL: str`, `config.REDIS_URL: str`, `config.CASE_STORE_PATH: str`, `config.CPU_QUEUE: str = "cpu_queue"`, `config.GPU_QUEUE: str = "gpu_queue"`, `config.CELERY_TASK_ALWAYS_EAGER: bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_phase1.py
import importlib


def _reload_config(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import config
    return importlib.reload(config)


def test_defaults_present(monkeypatch):
    cfg = _reload_config(monkeypatch)
    assert cfg.CPU_QUEUE == "cpu_queue"
    assert cfg.GPU_QUEUE == "gpu_queue"
    # sane local defaults
    assert cfg.DATABASE_URL.startswith("sqlite") or cfg.DATABASE_URL.startswith("postgresql")
    assert cfg.REDIS_URL.startswith("redis://")
    assert isinstance(cfg.CASE_STORE_PATH, str) and cfg.CASE_STORE_PATH


def test_env_overrides(monkeypatch):
    cfg = _reload_config(
        monkeypatch,
        DATABASE_URL="postgresql+psycopg2://u:p@db:5432/forensic",
        REDIS_URL="redis://redis:6379/1",
        CASE_STORE_PATH="/data/forensic-audio",
        CELERY_TASK_ALWAYS_EAGER="true",
    )
    assert cfg.DATABASE_URL == "postgresql+psycopg2://u:p@db:5432/forensic"
    assert cfg.REDIS_URL == "redis://redis:6379/1"
    assert cfg.CASE_STORE_PATH == "/data/forensic-audio"
    assert cfg.CELERY_TASK_ALWAYS_EAGER is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_phase1.py -v`
Expected: FAIL — `AttributeError: module 'config' has no attribute 'CPU_QUEUE'`.

- [ ] **Step 3: Append to `config.py`** (after existing content, do not remove anything)

```python
# --- Phase 1: async foundation (Celery / Redis / Postgres / case store) ---

# SQLAlchemy URL. Local/test default is in-memory SQLite; deploy sets Postgres
# via env (postgresql+psycopg2://...).
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+pysqlite:///./forensic_local.db")

# Celery broker + result backend.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Root of the immutable case/evidence tree (originals, derivatives, audit).
CASE_STORE_PATH = os.getenv("CASE_STORE_PATH", "case_data")

CPU_QUEUE = "cpu_queue"
GPU_QUEUE = "gpu_queue"

# Run Celery tasks inline (no broker) — set true in tests.
CELERY_TASK_ALWAYS_EAGER = os.getenv("CELERY_TASK_ALWAYS_EAGER", "false").lower() == "true"
```

- [ ] **Step 4: Update `requirements.txt`** — append:

```
celery==5.4.0
redis==5.0.7
sqlalchemy==2.0.31
alembic==1.13.2
psycopg2-binary==2.9.9
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config_phase1.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt config.py tests/test_config_phase1.py
git commit -m "feat: phase1 config (db/redis/case-store/queue settings)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Database layer — Base, session, models

**Files:**
- Create: `db/__init__.py`
- Create: `db/base.py`
- Create: `db/models.py`
- Test: `tests/test_db_models.py`

**Interfaces:**
- Consumes: `config.DATABASE_URL`.
- Produces:
  - `db.base.Base` (DeclarativeBase), `db.base.engine`, `db.base.SessionLocal`, `db.base.get_session()` (contextmanager yielding a Session), `db.base.init_db()` (create_all — test/bootstrap only).
  - `db.models.Case(id, created_at)`, `File(id, case_id, original_filename, ext, source_sha256, status, created_at)`, `Job(id, file_id, case_id, status, stage, degraded_flags, error, created_at, updated_at)`, `Segment(id, file_id, start, end, speaker, text, confidence, source_pass, flagged, review_status, created_at)`, `Review(id, segment_id, decision, text, reviewer_id, created_at)`, `AuditEntry(id, case_id, file_id, stage, payload, prev_entry_hash, entry_hash, created_at)`.
  - Status string constants `db.models.JobStatus` with `QUEUED, RUNNING, NEEDS_REVIEW, CERTIFIED, FAILED, QUARANTINED`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db_models.py
import os
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"

import importlib
import config; importlib.reload(config)
from db import base as dbbase
importlib.reload(dbbase)
from db import models as dbmodels
importlib.reload(dbmodels)


def test_create_case_file_job_roundtrip():
    dbbase.init_db()
    with dbbase.get_session() as s:
        case = dbmodels.Case(id="case-1")
        s.add(case)
        f = dbmodels.File(id="file-1", case_id="case-1",
                          original_filename="REC001.wav", ext=".wav",
                          status="ingested")
        s.add(f)
        job = dbmodels.Job(id="job-1", file_id="file-1", case_id="case-1",
                           status=dbmodels.JobStatus.QUEUED, stage="L0")
        s.add(job)
        s.commit()

    with dbbase.get_session() as s:
        job = s.get(dbmodels.Job, "job-1")
        assert job.status == "queued"
        assert job.stage == "L0"
        assert job.case_id == "case-1"
        assert job.created_at is not None


def test_job_degraded_flags_json():
    dbbase.init_db()
    with dbbase.get_session() as s:
        s.add(dbmodels.Case(id="case-2"))
        s.add(dbmodels.File(id="file-2", case_id="case-2",
                            original_filename="a.mp4", ext=".mp4", status="ingested"))
        s.add(dbmodels.Job(id="job-2", file_id="file-2", case_id="case-2",
                           status=dbmodels.JobStatus.RUNNING, stage="L2",
                           degraded_flags=["degraded_enhancement"]))
        s.commit()
    with dbbase.get_session() as s:
        assert s.get(dbmodels.Job, "job-2").degraded_flags == ["degraded_enhancement"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'db'`.

- [ ] **Step 3: Create `db/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Create `db/base.py`**

```python
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

import config


class Base(DeclarativeBase):
    pass


# SQLite needs check_same_thread=False for the threaded test/dev server.
_connect_args = {"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(config.DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db():
    """Create all tables. Used by tests and first-run bootstrap; production
    schema is managed by Alembic migrations."""
    # Import models so they register on Base.metadata before create_all.
    from db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
```

- [ ] **Step 5: Create `db/models.py`**

```python
import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Float, Boolean, Integer, ForeignKey, DateTime, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    NEEDS_REVIEW = "needs_review"
    CERTIFIED = "certified"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class Case(Base):
    __tablename__ = "cases"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class File(Base):
    __tablename__ = "files"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    original_filename: Mapped[str] = mapped_column(String)
    ext: Mapped[str] = mapped_column(String)
    source_sha256: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="registered")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    file_id: Mapped[str] = mapped_column(ForeignKey("files.id"), index=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    status: Mapped[str] = mapped_column(String, default=JobStatus.QUEUED, index=True)
    stage: Mapped[str | None] = mapped_column(String, nullable=True)
    degraded_flags: Mapped[list | None] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Segment(Base):
    __tablename__ = "segments"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    file_id: Mapped[str] = mapped_column(ForeignKey("files.id"), index=True)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    speaker: Mapped[str | None] = mapped_column(String, nullable=True)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_pass: Mapped[str | None] = mapped_column(String, nullable=True)
    flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    review_status: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    segment_id: Mapped[str] = mapped_column(ForeignKey("segments.id"), index=True)
    decision: Mapped[str] = mapped_column(String)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer_id: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditEntry(Base):
    __tablename__ = "audit_entries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    file_id: Mapped[str | None] = mapped_column(ForeignKey("files.id"), nullable=True)
    stage: Mapped[str] = mapped_column(String)
    payload: Mapped[dict] = mapped_column(JSON)
    prev_entry_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    entry_hash: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_db_models.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add db/ tests/test_db_models.py
git commit -m "feat: phase1 database layer (sqlalchemy models + session)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Alembic initial migration

**Files:**
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/script.py.mako`
- Create: `alembic/versions/0001_initial_schema.py`
- Test: `tests/test_migration.py`

**Interfaces:**
- Consumes: `db.base.Base`, `config.DATABASE_URL`.
- Produces: a runnable migration creating all six tables; `alembic upgrade head` works against the configured DB.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_migration.py
import os, subprocess, sys, tempfile, pathlib


def test_alembic_upgrade_head_creates_tables():
    tmp = tempfile.mkdtemp()
    db_path = pathlib.Path(tmp) / "mig.db"
    env = {**os.environ, "DATABASE_URL": f"sqlite+pysqlite:///{db_path}"}
    root = pathlib.Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"],
                       cwd=root, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    import sqlite3
    con = sqlite3.connect(db_path)
    names = {row[0] for row in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert {"cases", "files", "jobs", "segments", "reviews", "audit_entries"} <= names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_migration.py -v`
Expected: FAIL — alembic not configured (`No config file 'alembic.ini' found`).

- [ ] **Step 3: Create `alembic.ini`**

```ini
[alembic]
script_location = alembic
prepend_sys_path = .

[loggers]
keys = root

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
```

- [ ] **Step 4: Create `alembic/script.py.mako`**

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

- [ ] **Step 5: Create `alembic/env.py`**

```python
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

import config as app_config
from db.base import Base
import db.models  # noqa: F401  (register tables on Base.metadata)

config = context.config
config.set_main_option("sqlalchemy.url", app_config.DATABASE_URL)
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(url=app_config.DATABASE_URL, target_metadata=target_metadata,
                      literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.", poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 6: Create `alembic/versions/0001_initial_schema.py`**

```python
"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-06-18
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cases",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "files",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("case_id", sa.String(), sa.ForeignKey("cases.id"), index=True),
        sa.Column("original_filename", sa.String()),
        sa.Column("ext", sa.String()),
        sa.Column("source_sha256", sa.String(), nullable=True),
        sa.Column("status", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("file_id", sa.String(), sa.ForeignKey("files.id"), index=True),
        sa.Column("case_id", sa.String(), sa.ForeignKey("cases.id"), index=True),
        sa.Column("status", sa.String(), index=True),
        sa.Column("stage", sa.String(), nullable=True),
        sa.Column("degraded_flags", sa.JSON()),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "segments",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("file_id", sa.String(), sa.ForeignKey("files.id"), index=True),
        sa.Column("start", sa.Float()),
        sa.Column("end", sa.Float()),
        sa.Column("speaker", sa.String(), nullable=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("source_pass", sa.String(), nullable=True),
        sa.Column("flagged", sa.Boolean()),
        sa.Column("review_status", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "reviews",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("segment_id", sa.String(), sa.ForeignKey("segments.id"), index=True),
        sa.Column("decision", sa.String()),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("reviewer_id", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "audit_entries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("case_id", sa.String(), sa.ForeignKey("cases.id"), index=True),
        sa.Column("file_id", sa.String(), sa.ForeignKey("files.id"), nullable=True),
        sa.Column("stage", sa.String()),
        sa.Column("payload", sa.JSON()),
        sa.Column("prev_entry_hash", sa.String(), nullable=True),
        sa.Column("entry_hash", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    for t in ["audit_entries", "reviews", "segments", "jobs", "files", "cases"]:
        op.drop_table(t)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_migration.py -v`
Expected: PASS (1 passed).

- [ ] **Step 8: Commit**

```bash
git add alembic.ini alembic/ tests/test_migration.py
git commit -m "feat: phase1 alembic initial schema migration

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Repository helpers

**Files:**
- Create: `db/repository.py`
- Test: `tests/test_repository.py`

**Interfaces:**
- Consumes: `db.base.get_session`, `db.models`.
- Produces (all take an explicit `session`):
  - `create_case(session) -> str` (returns case_id)
  - `create_file(session, case_id, original_filename, ext) -> str` (returns file_id)
  - `create_job(session, case_id, file_id) -> str` (returns job_id, status=QUEUED)
  - `get_job(session, job_id) -> Job | None`
  - `update_job(session, job_id, *, status=None, stage=None, error=None, add_degraded=None) -> Job`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repository.py
import os
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
import importlib
import config; importlib.reload(config)
from db import base as dbbase; importlib.reload(dbbase)
from db import models as dbmodels; importlib.reload(dbmodels)
from db import repository as repo; importlib.reload(repo)


def setup_module():
    dbbase.init_db()


def test_create_chain_and_update():
    with dbbase.get_session() as s:
        case_id = repo.create_case(s)
        file_id = repo.create_file(s, case_id, "REC001.wav", ".wav")
        job_id = repo.create_job(s, case_id, file_id)
        s.commit()

    with dbbase.get_session() as s:
        job = repo.get_job(s, job_id)
        assert job.status == dbmodels.JobStatus.QUEUED
        assert job.case_id == case_id and job.file_id == file_id

    with dbbase.get_session() as s:
        repo.update_job(s, job_id, status=dbmodels.JobStatus.RUNNING, stage="L2",
                        add_degraded="degraded_enhancement")
        s.commit()

    with dbbase.get_session() as s:
        job = repo.get_job(s, job_id)
        assert job.status == "running"
        assert job.stage == "L2"
        assert "degraded_enhancement" in job.degraded_flags
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_repository.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'db.repository'`.

- [ ] **Step 3: Create `db/repository.py`**

```python
from db import models
from db.models import Case, File, Job, JobStatus


def create_case(session) -> str:
    case = Case()
    session.add(case)
    session.flush()
    return case.id


def create_file(session, case_id: str, original_filename: str, ext: str) -> str:
    f = File(case_id=case_id, original_filename=original_filename, ext=ext,
             status="registered")
    session.add(f)
    session.flush()
    return f.id


def create_job(session, case_id: str, file_id: str) -> str:
    job = Job(case_id=case_id, file_id=file_id, status=JobStatus.QUEUED,
              stage=None, degraded_flags=[])
    session.add(job)
    session.flush()
    return job.id


def get_job(session, job_id: str):
    return session.get(Job, job_id)


def update_job(session, job_id: str, *, status=None, stage=None, error=None,
               add_degraded=None):
    job = session.get(Job, job_id)
    if job is None:
        raise ValueError(f"job not found: {job_id}")
    if status is not None:
        job.status = status
    if stage is not None:
        job.stage = stage
    if error is not None:
        job.error = error
    if add_degraded is not None:
        flags = list(job.degraded_flags or [])
        if add_degraded not in flags:
            flags.append(add_degraded)
        job.degraded_flags = flags
    session.flush()
    return job
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_repository.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add db/repository.py tests/test_repository.py
git commit -m "feat: phase1 repository helpers (case/file/job)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Celery app + skeleton pipeline task

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/celery_app.py`
- Create: `pipeline/tasks.py`
- Test: `tests/test_pipeline_skeleton.py`

**Interfaces:**
- Consumes: `config.REDIS_URL`, `config.CELERY_TASK_ALWAYS_EAGER`, `db.repository`, `db.base.get_session`.
- Produces:
  - `pipeline.celery_app.celery` (Celery instance, eager when configured).
  - `pipeline.tasks.PIPELINE_STAGES: list[str] = ["L0","L1","L2","L3","L4","L5","L6","L7","L8"]`.
  - `pipeline.tasks.run_pipeline(job_id: str) -> str` (Celery task; walks stages, sets each `stage`, ends `status=NEEDS_REVIEW`; on exception sets `status=FAILED` + error and re-raises). Returns final status.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_skeleton.py
import os
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "true"
import importlib
import config; importlib.reload(config)
from db import base as dbbase; importlib.reload(dbbase)
from db import models as dbmodels; importlib.reload(dbmodels)
from db import repository as repo; importlib.reload(repo)
from pipeline import tasks as ptasks; importlib.reload(ptasks)


def setup_module():
    dbbase.init_db()


def test_run_pipeline_walks_to_needs_review():
    with dbbase.get_session() as s:
        case_id = repo.create_case(s)
        file_id = repo.create_file(s, case_id, "a.wav", ".wav")
        job_id = repo.create_job(s, case_id, file_id)
        s.commit()

    result = ptasks.run_pipeline.apply(args=[job_id]).get()
    assert result == dbmodels.JobStatus.NEEDS_REVIEW

    with dbbase.get_session() as s:
        job = repo.get_job(s, job_id)
        assert job.status == dbmodels.JobStatus.NEEDS_REVIEW
        assert job.stage == "L8"


def test_run_pipeline_marks_failed_on_bad_job_id():
    import pytest
    with pytest.raises(ValueError):
        ptasks.run_pipeline.apply(args=["nonexistent"]).get()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_skeleton.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline'`.

- [ ] **Step 3: Create `pipeline/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Create `pipeline/celery_app.py`**

```python
from celery import Celery

import config

celery = Celery("forensic", broker=config.REDIS_URL, backend=config.REDIS_URL)
celery.conf.update(
    task_always_eager=config.CELERY_TASK_ALWAYS_EAGER,
    task_eager_propagates=True,
    task_default_queue=config.CPU_QUEUE,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)

# Ensure task modules are imported so they register on the app.
celery.autodiscover_tasks(["pipeline"])
import pipeline.tasks  # noqa: E402,F401
```

- [ ] **Step 5: Create `pipeline/tasks.py`**

```python
from pipeline.celery_app import celery
from db.base import get_session
from db import repository as repo
from db.models import JobStatus

# Placeholder stage walk. Phases 2-5 replace each stage with a real layer task
# (ingest/hash, ffmpeg, enhance, VAD union, diarize, ASR, confidence, route, output).
PIPELINE_STAGES = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]


@celery.task(name="pipeline.run_pipeline")
def run_pipeline(job_id: str) -> str:
    with get_session() as s:
        repo.update_job(s, job_id, status=JobStatus.RUNNING)
        s.commit()
    try:
        for stage in PIPELINE_STAGES:
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

Note: the `run_pipeline.apply()` re-raise path in `test_run_pipeline_marks_failed_on_bad_job_id` raises inside the first `update_job` (bad id) before status flips — `ValueError` propagates, which the test asserts.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_pipeline_skeleton.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add pipeline/ tests/test_pipeline_skeleton.py
git commit -m "feat: phase1 celery app + skeleton pipeline task

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Job-based API routes + app wiring (remove legacy sync endpoint)

**Files:**
- Create: `api/routes/cases.py`
- Modify: `app.py`
- Delete: `api/routes/stt.py` (legacy sync transcribe removed per spec §3)
- Delete: `tests/test_api.py` (tested the removed endpoint) — replaced below
- Create: `tests/test_api_jobs.py`

**Interfaces:**
- Consumes: `db.repository`, `db.base`, `pipeline.tasks.run_pipeline`, `api.auth.require_api_key`, `config`.
- Produces endpoints:
  - `POST /cases` → `201 {"case_id": str}`
  - `POST /cases/{case_id}/files` (multipart `audio`) → `202 {"file_id": str, "job_id": str}`; rejects bad ext `400`, oversize `413`, unknown case `404`. Enqueues `run_pipeline`.
  - `GET /jobs/{job_id}` → `200 {"job_id","case_id","file_id","status","stage","degraded_flags"}`; `404` if missing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_jobs.py
import os
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "true"
os.environ["CASE_STORE_PATH"] = "/tmp/forensic_test_store"
os.environ["API_KEY"] = ""  # auth disabled for test

import importlib
import config; importlib.reload(config)
from db import base as dbbase; importlib.reload(dbbase)
import db.models  # noqa
from fastapi.testclient import TestClient
import app as appmod; importlib.reload(appmod)

client = TestClient(appmod.app)


def setup_module():
    dbbase.init_db()


def test_full_job_flow():
    r = client.post("/cases")
    assert r.status_code == 201
    case_id = r.json()["case_id"]

    files = {"audio": ("REC001.wav", b"RIFFfakewavdata", "audio/wav")}
    r = client.post(f"/cases/{case_id}/files", files=files)
    assert r.status_code == 202, r.text
    body = r.json()
    job_id = body["job_id"]
    assert body["file_id"]

    # eager celery already ran the skeleton pipeline to needs_review
    r = client.get(f"/jobs/{job_id}")
    assert r.status_code == 200
    jb = r.json()
    assert jb["status"] == "needs_review"
    assert jb["stage"] == "L8"
    assert jb["case_id"] == case_id


def test_bad_extension_rejected():
    r = client.post("/cases")
    case_id = r.json()["case_id"]
    files = {"audio": ("note.txt", b"hello", "text/plain")}
    r = client.post(f"/cases/{case_id}/files", files=files)
    assert r.status_code == 400


def test_unknown_case_404():
    files = {"audio": ("a.wav", b"x", "audio/wav")}
    r = client.post("/cases/does-not-exist/files", files=files)
    assert r.status_code == 404


def test_unknown_job_404():
    assert client.get("/jobs/nope").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_jobs.py -v`
Expected: FAIL — `app` still includes old `stt` router / no `/cases` route (`404` on POST /cases).

- [ ] **Step 3: Create `api/routes/cases.py`**

```python
import os
import uuid

import aiofiles
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, status

import config
from api.auth import require_api_key
from db.base import get_session
from db import repository as repo
from db.models import Case
from pipeline.tasks import run_pipeline

router = APIRouter()


@router.post("/cases", status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(require_api_key)])
def create_case():
    with get_session() as s:
        case_id = repo.create_case(s)
        s.commit()
    return {"case_id": case_id}


@router.post("/cases/{case_id}/files", status_code=status.HTTP_202_ACCEPTED,
             dependencies=[Depends(require_api_key)])
async def upload_file(case_id: str, audio: UploadFile = File(...)):
    ext = os.path.splitext(audio.filename or "")[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    with get_session() as s:
        if s.get(Case, case_id) is None:
            raise HTTPException(status_code=404, detail="Unknown case_id")

    # Land the upload under the case store; L0 (Phase 2) hashes + WORM-locks it.
    inbox = os.path.join(config.CASE_STORE_PATH, "cases", case_id, "inbox")
    os.makedirs(inbox, exist_ok=True)
    staged = os.path.join(inbox, f"{uuid.uuid4()}{ext}")
    content = await audio.read()
    if len(content) > config.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    async with aiofiles.open(staged, "wb") as f:
        await f.write(content)

    with get_session() as s:
        file_id = repo.create_file(s, case_id, audio.filename or staged, ext)
        job_id = repo.create_job(s, case_id, file_id)
        s.commit()

    run_pipeline.delay(job_id)
    return {"file_id": file_id, "job_id": job_id}


@router.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str):
    with get_session() as s:
        job = repo.get_job(s, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return {
            "job_id": job.id,
            "case_id": job.case_id,
            "file_id": job.file_id,
            "status": job.status,
            "stage": job.stage,
            "degraded_flags": job.degraded_flags or [],
        }
```

- [ ] **Step 4: Replace `app.py`** with:

```python
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

import config
from api.routes.cases import router as cases_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(config.CASE_STORE_PATH, exist_ok=True)
    # Bootstrap tables when running against SQLite (local/dev). Postgres deploys
    # run `alembic upgrade head` instead.
    if config.DATABASE_URL.startswith("sqlite"):
        from db.base import init_db
        init_db()
    yield


app = FastAPI(title="Forensic Audio Pipeline API", version="2.0.0", lifespan=lifespan)
app.include_router(cases_router)


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 5: Delete legacy files**

```bash
git rm api/routes/stt.py tests/test_api.py
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_api_jobs.py -v`
Expected: PASS (4 passed).

- [ ] **Step 7: Run the full suite (no models exercised)**

Run: `pytest -q -m "not gpu and not model"`
Expected: all Phase 1 tests pass; pre-existing service tests that import heavy libs may be collected — if any fail purely on missing heavy deps locally, leave them (they run on deploy box). New Phase 1 tests must be green.

- [ ] **Step 8: Commit**

```bash
git add api/routes/cases.py app.py tests/test_api_jobs.py
git commit -m "feat: phase1 job-based API (cases/files/jobs), remove legacy sync endpoint

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Deploy artifacts — docker-compose (7 services), Dockerfile, .env.example

**Files:**
- Modify: `Dockerfile`
- Create: `docker-compose.yml` (overwrite existing single-service file)
- Modify: `.env.example`
- Create: `nginx/forensic.conf`
- Create: `docker-entrypoint-api.sh`
- Test: `tests/test_compose_valid.py`

**Interfaces:**
- Produces: a `docker compose config`-valid stack with services `nginx, api, redis, postgres, worker-cpu, worker-gpu, flower`. (No containers are run in Phase 1 tests — only static validation.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compose_valid.py
import pathlib, yaml


def test_compose_has_seven_services():
    root = pathlib.Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "docker-compose.yml").read_text())
    services = set(data["services"])
    assert services == {
        "nginx", "api", "redis", "postgres",
        "worker-cpu", "worker-gpu", "flower",
    }


def test_worker_gpu_has_gpu_reservation():
    root = pathlib.Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "docker-compose.yml").read_text())
    devs = data["services"]["worker-gpu"]["deploy"]["resources"]["reservations"]["devices"]
    assert any(d.get("driver") == "nvidia" for d in devs)


def test_api_has_no_gpu_reservation():
    root = pathlib.Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "docker-compose.yml").read_text())
    assert "deploy" not in data["services"]["api"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compose_valid.py -v`
Expected: FAIL — current `docker-compose.yml` has only the `stt` service.

- [ ] **Step 3: Overwrite `docker-compose.yml`**

```yaml
services:
  nginx:
    image: nginx:latest
    ports: ["443:443", "80:80"]
    volumes:
      - ./nginx/forensic.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on: [api]
    restart: unless-stopped

  api:
    build: .
    image: forensic-audio
    env_file: .env
    environment:
      - DATABASE_URL=postgresql+psycopg2://pipeline:pipeline@postgres:5432/forensic
      - REDIS_URL=redis://redis:6379/0
      - CASE_STORE_PATH=/data/forensic-audio
    command: ["/bin/bash", "/app/docker-entrypoint-api.sh"]
    volumes:
      - case_data:/data/forensic-audio
    depends_on: [redis, postgres]
    restart: unless-stopped

  redis:
    image: redis:7
    volumes: [redis_data:/data]
    restart: unless-stopped

  postgres:
    image: postgres:16
    environment:
      - POSTGRES_DB=forensic
      - POSTGRES_USER=pipeline
      - POSTGRES_PASSWORD=pipeline
    volumes: [pg_data:/var/lib/postgresql/data]
    restart: unless-stopped

  worker-cpu:
    image: forensic-audio
    build: .
    env_file: .env
    environment:
      - DATABASE_URL=postgresql+psycopg2://pipeline:pipeline@postgres:5432/forensic
      - REDIS_URL=redis://redis:6379/0
      - CASE_STORE_PATH=/data/forensic-audio
    command: celery -A pipeline.celery_app worker -Q cpu_queue --concurrency=4 -n cpu@%h
    volumes:
      - case_data:/data/forensic-audio
      - model_cache:/app/model_cache
    depends_on: [redis, postgres]
    restart: unless-stopped

  worker-gpu:
    image: forensic-audio
    build: .
    env_file: .env
    environment:
      - DATABASE_URL=postgresql+psycopg2://pipeline:pipeline@postgres:5432/forensic
      - REDIS_URL=redis://redis:6379/0
      - CASE_STORE_PATH=/data/forensic-audio
      - CUDA_VISIBLE_DEVICES=0
    command: celery -A pipeline.celery_app worker -Q gpu_queue --concurrency=1 -n gpu@%h
    volumes:
      - case_data:/data/forensic-audio
      - model_cache:/app/model_cache
      - hf_cache:/root/.cache/huggingface
    depends_on: [redis, postgres]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

  flower:
    image: forensic-audio
    build: .
    env_file: .env
    environment:
      - REDIS_URL=redis://redis:6379/0
    command: celery -A pipeline.celery_app flower --port=5555
    ports: ["5555:5555"]
    depends_on: [redis]
    restart: unless-stopped

volumes:
  case_data:
  redis_data:
  pg_data:
  model_cache:
  hf_cache:
```

- [ ] **Step 4: Create `docker-entrypoint-api.sh`**

```bash
#!/usr/bin/env bash
set -e
# Apply DB migrations, then serve. Idempotent: re-running upgrade is a no-op.
alembic upgrade head
exec uvicorn app:app --host 0.0.0.0 --port 8009 --workers 1
```

- [ ] **Step 5: Modify `Dockerfile`** — add bash + entrypoint perms. Replace the apt line and the final CMD:

Change the apt install line to include `bash`:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg bash \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*
```

Replace the final `CMD` line with:

```dockerfile
RUN chmod +x /app/docker-entrypoint-api.sh
EXPOSE 8009
CMD ["/bin/bash", "/app/docker-entrypoint-api.sh"]
```

- [ ] **Step 6: Create `nginx/forensic.conf`**

```nginx
server {
    listen 80;
    client_max_body_size 600M;

    location / {
        proxy_pass http://api:8009;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 600s;
    }
}
```

- [ ] **Step 7: Update `.env.example`** — append:

```
# --- Phase 1 async stack (compose sets these; override for external services) ---
# DATABASE_URL=postgresql+psycopg2://pipeline:pipeline@postgres:5432/forensic
# REDIS_URL=redis://redis:6379/0
# CASE_STORE_PATH=/data/forensic-audio
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_compose_valid.py -v`
Expected: PASS (3 passed).

- [ ] **Step 9: Validate compose syntax (no containers started)**

Run: `docker compose config >/dev/null && echo OK`
Expected: `OK` (skip if docker not installed on build box — the yaml test already guards structure).

- [ ] **Step 10: Commit**

```bash
git add docker-compose.yml Dockerfile .env.example nginx/forensic.conf docker-entrypoint-api.sh tests/test_compose_valid.py
git commit -m "feat: phase1 single-server compose stack (7 services) + migrate-on-boot

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 1 Done — Definition of Done

- `pytest tests/test_config_phase1.py tests/test_db_models.py tests/test_migration.py tests/test_repository.py tests/test_pipeline_skeleton.py tests/test_api_jobs.py tests/test_compose_valid.py -v` → all green.
- `POST /cases` → `POST /cases/{id}/files` → `GET /jobs/{id}` flow works end-to-end with eager Celery + SQLite locally.
- `docker compose config` valid; stack defines all 7 services; only `worker-gpu` reserves GPU.
- Legacy synchronous `/stt/transcribe` removed.

**Next:** Phase 2 (Chain-of-custody — L0/L1/L9) gets its own plan, replacing the `L0`/`L1` skeleton stages with real ingest/hash/ffmpeg + ledger.
