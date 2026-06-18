import pytest

from db import base as dbbase
from db import models as dbmodels
from db import repository as repo
from pipeline import tasks as ptasks


def setup_module():
    dbbase.init_db()


def test_run_pipeline_quarantines_without_input():
    # No staged original -> L0 cannot find it -> quarantine (never silent-drop).
    with dbbase.get_session() as s:
        case_id = repo.create_case(s)
        file_id = repo.create_file(s, case_id, "a.wav", ".wav")
        job_id = repo.create_job(s, case_id, file_id)
        s.commit()

    result = ptasks.run_pipeline.apply(args=[job_id]).get()
    assert result == dbmodels.JobStatus.QUARANTINED

    with dbbase.get_session() as s:
        assert repo.get_job(s, job_id).status == dbmodels.JobStatus.QUARANTINED


def test_run_pipeline_marks_failed_on_bad_job_id():
    with pytest.raises(ValueError):
        ptasks.run_pipeline.apply(args=["nonexistent"]).get()
