from db import base as dbbase
from db import models as dbmodels
from db import repository as repo


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
