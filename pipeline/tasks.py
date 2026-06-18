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


def _l0_ingest(job, session):
    """Hash the byte-exact original, WORM-store it, register manifest + file row.
    Returns (original_path, source_sha256). Raises FileNotFoundError if no
    staged input."""
    file_row = repo.get_file(session, job.file_id)
    ext = file_row.ext
    staged = _inbox_original(job.case_id, job.file_id, ext)
    if not os.path.exists(staged):
        raise FileNotFoundError(f"no staged original for file {job.file_id}")
    dest, digest = storage.write_original(job.case_id, job.file_id, ext, staged)
    man.register_file(job.case_id, job.file_id, file_row.original_filename, digest)
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
