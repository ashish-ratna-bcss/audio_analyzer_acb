import os, json, shutil, subprocess
import pytest
from db import base as dbbase, repository as repo
from db.models import JobStatus
from unittest.mock import patch
from services import (audit_service as au, manifest_service as man, storage,
                      diarization_service, indic_asr_service,
                      embedding_service, lang_id_service)
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
                        lambda p, num_speakers=None: (
                            [{"start": 0.0, "end": 1.0, "speaker": "Speaker_1"}], "mock-diarizer"))
    monkeypatch.setattr(lang_id_service, "identify",
                        lambda p: {"top1": "tel", "top1_confidence": 0.9,
                                   "top2": None, "top2_confidence": 0.0})
    # IndicConformer is the only ASR model now; run twice (enh + org) per segment.
    monkeypatch.setattr(indic_asr_service, "transcribe_clip",
                        lambda p, lang=None: {"text": "hello cost fifteen",
                            "confidence": None, "language": lang,
                            "model": "indic", "abstained": False})
    monkeypatch.setattr(embedding_service, "similarity", lambda a, b: 0.97)


def test_run_indic_abstains_non_indic():
    # routed to a non-Indic language -> IndicConformer abstains, no fallback.
    with patch.object(ptasks.lang_id_service, "identify",
                      return_value={"top1": "kor", "top1_confidence": 0.9,
                                    "top2": None, "top2_confidence": 0.0}), \
         patch.object(ptasks.lang_id_service, "to_iso639_1", return_value="ko"), \
         patch.object(ptasks.indic_asr_service, "transcribe_clip",
                      return_value={"text": "", "confidence": None, "language": "ko",
                                    "model": "indic_unsupported", "abstained": True}), \
         patch.object(ptasks.embedding_service, "similarity", return_value=0.0):
        asr = ptasks.run_indic("/tmp/clean.wav", "/tmp/org.wav", file_prior=None)

    assert asr["abstained"] is True
    assert asr["text"] == ""
    assert asr["source"] == "none"


def test_run_indic_dualrun_agreement_high():
    # enhanced and original decodes agree -> high confidence, not divergent.
    with patch.object(ptasks.lang_id_service, "identify",
                      return_value={"top1": "tel", "top1_confidence": 0.9,
                                    "top2": None, "top2_confidence": 0.0}), \
         patch.object(ptasks.lang_id_service, "to_iso639_1", return_value="te"), \
         patch.object(ptasks.indic_asr_service, "transcribe_clip",
                      return_value={"text": "ధర పదిహేను రూపాయలు", "confidence": None,
                                    "language": "te", "model": "indic", "abstained": False}), \
         patch.object(ptasks.embedding_service, "similarity", return_value=0.99):
        asr = ptasks.run_indic("/tmp/clean.wav", "/tmp/org.wav", file_prior="te")

    assert asr["text"] == "ధర పదిహేను రూపాయలు"
    assert asr["agreement"] >= 0.9
    assert asr["source"] == "indic_enhanced"


def test_run_indic_blanks_ghost_phrase():
    with patch.object(ptasks.lang_id_service, "identify",
                      return_value={"top1": "eng", "top1_confidence": 0.9,
                                    "top2": None, "top2_confidence": 0.0}), \
         patch.object(ptasks.lang_id_service, "to_iso639_1", return_value="en"), \
         patch.object(ptasks.indic_asr_service, "transcribe_clip",
                      return_value={"text": "Thank you.", "confidence": None, "language": "en",
                                    "model": "indic", "abstained": False}), \
         patch.object(ptasks.embedding_service, "similarity", return_value=0.0):
        asr = ptasks.run_indic("/tmp/clean.wav", "/tmp/org.wav", file_prior="en")

    assert asr["text"] == ""                   # ghost blanked both runs
    assert asr["hallucination"] == "ghost_phrase"


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

    assert ptasks.run_pipeline.apply(args=[job_id]).get() == JobStatus.COMPLETED

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
