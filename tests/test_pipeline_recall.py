import os, json, shutil, subprocess
import pytest
from db import base as dbbase, repository as repo
from db.models import JobStatus
from services import audit_service as au, manifest_service as man, storage
from services import vad_service, enhancement_service, separation_service
from pipeline import tasks as ptasks

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def setup_module():
    dbbase.init_db()


def _shutil_copy(src, dst):
    shutil.copyfile(src, dst)
    return dst


def _patch_store(monkeypatch, tmp_path):
    for mod in (ptasks.config, au.config, man.config, storage.config):
        monkeypatch.setattr(mod, "CASE_STORE_PATH", str(tmp_path))


def _make_tone(path):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=1", "-ar", "22050", str(path)],
                   capture_output=True, check=True)


def _stage(tmp_path, monkeypatch, options=None):
    _patch_store(monkeypatch, tmp_path)
    with dbbase.get_session() as s:
        case_id = repo.create_case(s)
        file_id = repo.create_file(s, case_id, "tone.wav", ".wav")
        job_id = repo.create_job(s, case_id, file_id, options=options or {})
        s.commit()
    inbox = os.path.join(str(tmp_path), "cases", case_id, "inbox")
    os.makedirs(inbox, exist_ok=True)
    _make_tone(os.path.join(inbox, f"{file_id}.wav"))
    return case_id, file_id, job_id


def _vad_json_path(tmp_path, case_id, file_id):
    return os.path.join(str(tmp_path), "cases", case_id, "derivatives",
                        file_id, "vad", f"{file_id}_segments_union.json")


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_vad_union_written_two_branches(tmp_path, monkeypatch):
    monkeypatch.setattr(enhancement_service, "enhance", _shutil_copy)
    monkeypatch.setattr(vad_service, "detect_speech",
                        lambda p: [{"start": 0.0, "end": 1.0}])
    case_id, file_id, job_id = _stage(tmp_path, monkeypatch)
    assert ptasks.run_pipeline.apply(args=[job_id]).get() == JobStatus.NEEDS_REVIEW
    data = json.load(open(_vad_json_path(tmp_path, case_id, file_id)))
    assert data["segments"] == [{"start": 0.0, "end": 1.0}]
    assert "original" in data["branch_counts"] and "enhanced" in data["branch_counts"]


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_enhancement_failure_flags_degraded(tmp_path, monkeypatch):
    def _boom(i, o):
        raise RuntimeError("dfn exploded")
    monkeypatch.setattr(enhancement_service, "enhance", _boom)
    monkeypatch.setattr(vad_service, "detect_speech",
                        lambda p: [{"start": 0.0, "end": 1.0}])
    case_id, file_id, job_id = _stage(tmp_path, monkeypatch)
    assert ptasks.run_pipeline.apply(args=[job_id]).get() == JobStatus.NEEDS_REVIEW
    with dbbase.get_session() as s:
        assert "degraded_enhancement" in repo.get_job(s, job_id).degraded_flags


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_separation_excluded_when_fewer_segments(tmp_path, monkeypatch):
    monkeypatch.setattr(enhancement_service, "enhance", _shutil_copy)
    monkeypatch.setattr(separation_service, "separate_vocals", _shutil_copy)

    def _vad(path):
        if "stem" in path:   # stem branch finds no speech -> fewer -> excluded
            return []
        return [{"start": 0.0, "end": 1.0}]
    monkeypatch.setattr(vad_service, "detect_speech", _vad)

    case_id, file_id, job_id = _stage(tmp_path, monkeypatch, options={"separate": True})
    assert ptasks.run_pipeline.apply(args=[job_id]).get() == JobStatus.NEEDS_REVIEW
    data = json.load(open(_vad_json_path(tmp_path, case_id, file_id)))
    assert data["separation_included"] is False
    assert data["segments"] == [{"start": 0.0, "end": 1.0}]
