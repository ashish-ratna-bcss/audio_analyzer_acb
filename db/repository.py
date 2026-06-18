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
