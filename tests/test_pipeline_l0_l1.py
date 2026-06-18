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


def _patch_store(monkeypatch, tmp_path):
    from services import storage
    for mod in (ptasks.config, au.config, man.config, storage.config):
        monkeypatch.setattr(mod, "CASE_STORE_PATH", str(tmp_path))


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_l0_l1_hashes_and_logs(tmp_path, monkeypatch):
    _patch_store(monkeypatch, tmp_path)
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
        digest = f.source_sha256
        assert digest and len(digest) == 64
    assert au.verify_chain(case_id) is True
    m = man.load(case_id)
    assert m["files"][file_id]["source_sha256"] == digest
    kinds = {d["kind"] for d in m["files"][file_id]["derivatives"]}
    assert {"normalized_48k", "normalized_16k"} <= kinds


def test_quarantine_on_missing_input(tmp_path, monkeypatch):
    _patch_store(monkeypatch, tmp_path)
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
        assert repo.get_file(s, file_id).status == "quarantined"
