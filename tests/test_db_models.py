from db import base as dbbase
from db import models as dbmodels


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
