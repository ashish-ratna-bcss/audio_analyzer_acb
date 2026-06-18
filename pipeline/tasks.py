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
