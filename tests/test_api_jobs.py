from fastapi.testclient import TestClient

from db import base as dbbase
import db.models  # noqa: F401
import app as appmod

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
