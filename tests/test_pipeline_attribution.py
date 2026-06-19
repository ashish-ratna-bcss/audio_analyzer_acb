import os, json, shutil, subprocess
import pytest
from db import base as dbbase, repository as repo
from db.models import JobStatus
from unittest.mock import patch
from services import (audit_service as au, manifest_service as man, storage,
                      diarization_service, whisper_service, indic_asr_service,
                      seamless_service, embedding_service, lang_id_service)
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
    monkeypatch.setattr(lang_id_service, "identify",
                        lambda p: {"top1": "tel", "top1_confidence": 0.9,
                                   "top2": None, "top2_confidence": 0.0})
    monkeypatch.setattr(indic_asr_service, "transcribe_clip",
                        lambda p, lang=None: {"text": "hello cost fifteen",
                            "confidence": None, "language": lang,
                            "model": "indic", "abstained": False})
    monkeypatch.setattr(seamless_service, "transcribe_clip",
                        lambda p, lang=None: {"text": "hello cost fifteen",
                            "confidence": 0.8, "language": lang})
    monkeypatch.setattr(embedding_service, "similarity", lambda a, b: 0.95)


def test_run_models_independent_indic_abstains_non_indic():
    with patch.object(ptasks.lang_id_service, "identify",
                      return_value={"top1": "kor", "top1_confidence": 0.9,
                                    "top2": None, "top2_confidence": 0.0}), \
         patch.object(ptasks.lang_id_service, "to_iso639_1", return_value="ko"), \
         patch.object(ptasks, "_whisper_clip",
                      return_value={"text": "annyeong", "confidence": 0.7,
                                    "language": "ko", "no_speech_prob": 0.1}), \
         patch.object(ptasks.indic_asr_service, "transcribe_clip",
                      return_value={"text": "", "confidence": None, "language": "ko",
                                    "model": "indic_unsupported", "abstained": True}), \
         patch.object(ptasks.seamless_service, "transcribe_clip",
                      return_value={"text": "annyeong", "confidence": 0.6, "language": "ko"}):
        asr = ptasks.run_models("/tmp/clean.wav", file_prior=None)

    assert asr["indic"]["abstained"] is True
    assert asr["indic"]["text"] == ""          # never masquerades as whisper
    assert asr["whisper"]["text"] == "annyeong"


def test_run_models_blanks_ghost_phrase():
    with patch.object(ptasks.lang_id_service, "identify",
                      return_value={"top1": "eng", "top1_confidence": 0.2,
                                    "top2": None, "top2_confidence": 0.0}), \
         patch.object(ptasks.lang_id_service, "to_iso639_1", return_value="en"), \
         patch.object(ptasks, "_whisper_clip",
                      return_value={"text": "Thank you.", "confidence": 0.6,
                                    "language": "en", "no_speech_prob": 0.2}), \
         patch.object(ptasks.indic_asr_service, "transcribe_clip",
                      return_value={"text": "Thank you.", "confidence": None, "language": "en",
                                    "model": "x", "abstained": False}), \
         patch.object(ptasks.seamless_service, "transcribe_clip",
                      return_value={"text": "[Music playing]", "confidence": 0.3, "language": "en"}):
        asr = ptasks.run_models("/tmp/clean.wav", file_prior=None)

    assert asr["whisper"]["text"] == ""        # ghost blanked
    assert asr["seamless"]["text"] == ""        # [Music playing] blanked


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
