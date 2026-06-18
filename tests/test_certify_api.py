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
