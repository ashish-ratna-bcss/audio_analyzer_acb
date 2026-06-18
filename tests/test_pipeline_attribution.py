import os, json, shutil, subprocess
import pytest
from db import base as dbbase, repository as repo
from db.models import JobStatus
from services import (audit_service as au, manifest_service as man, storage,
                      diarization_service, whisper_service, indic_asr_service,
                      embedding_service)
from pipeline import tasks as ptasks

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def setup_module():
    dbbase.init_db()


def _patch_store(monkeypatch, tmp_path):
    for mod in (ptasks.config, au.config, man.config, storage.config):
        monkeypatch.setattr(mod, "CASE_STORE_PATH", str(tmp_path))


def _tone(path):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-ar", "22050", str(path)], capture_output=True, check=True)


def _mock_models(monkeypatch):
    monkeypatch.setattr(diarization_service, "diarize_with_overlap",
                        lambda p, num_speakers=None: [
                            {"start": 0.0, "end": 1.0, "speaker": "Speaker_1"}])
    monkeypatch.setattr(whisper_service, "transcribe",
                        lambda path, **k: {"language": "te", "duration": 1.0,
                            "segments": [{"start": 0.0, "end": 1.0,
                            "text": "hello cost fifteen", "confidence": 0.9,
                            "no_speech_prob": 0.1, "compression_ratio": 1.2}]})
    monkeypatch.setattr(indic_asr_service, "transcribe_clip",
                        lambda p: {"text": "hello cost fifteen", "confidence": 0.5})
    monkeypatch.setattr(embedding_service, "similarity", lambda a, b: 0.95)


@pytest.mark.skipif(not HAS_FFMPEG, reason="no ffmpeg")
def test_segments_persisted_and_artifacts_written(tmp_path, monkeypatch):
    _patch_store(monkeypatch, tmp_path)
    _mock_models(monkeypatch)
    with dbbase.get_session() as s:
        case_id = repo.create_case(s)
        file_id = repo.create_file(s, case_id, "tone.wav", ".wav")
        job_id = repo.create_job(s, case_id, file_id)
        s.commit()
    inbox = os.path.join(str(tmp_path), "cases", case_id, "inbox")
    os.makedirs(inbox, exist_ok=True)
    _tone(os.path.join(inbox, f"{file_id}.wav"))

    assert ptasks.run_pipeline.apply(args=[job_id]).get() == JobStatus.NEEDS_REVIEW

    with dbbase.get_session() as s:
        segs = repo.list_segments(s, file_id)
        assert len(segs) == 1
        assert segs[0].speaker == "Speaker_1"
        assert segs[0].text == "hello cost fifteen"
        assert segs[0].flagged is False  # all passes agree, high conf
    conf = json.load(open(os.path.join(str(tmp_path), "cases", case_id,
        "derivatives", file_id, "confidence", f"{file_id}_confidence_report.json")))
    assert conf["segments_total"] == 1
    assert au.verify_chain(case_id) is True
