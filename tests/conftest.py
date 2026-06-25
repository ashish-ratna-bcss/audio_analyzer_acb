import os

# Set the test environment BEFORE any application module imports config/db.
# In-memory SQLite + StaticPool (see db/base.py) gives a shared schema across
# sessions; Celery runs inline so the skeleton pipeline executes synchronously.
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")
os.environ.setdefault("API_KEY", "")
os.environ.setdefault("CASE_STORE_PATH", "/tmp/forensic_test_store")

import importlib
import pytest


@pytest.fixture(autouse=True)
def _reset_config_baseline():
    """Some tests reload `config` with overridden env to assert parsing. Reset
    the singleton to the conftest test baseline before every test so a leaked
    postgres URL / production path can't bleed into other tests. The DB engine
    is bound at db.base import time and is unaffected by this reload."""
    import config
    importlib.reload(config)
    yield


# Service-unit-test modules exercise the real wrapper functions directly, so
# the global model stub must NOT apply to them.
_NO_STUB_MODULES = {
    "test_whisper_service", "test_translation_service", "test_diarization_service",
    "test_cross_model", "test_diarize_assign", "test_vad_union",
    "test_recall_wrappers_import", "test_phase4_wrappers_import",
    "test_indic_abstain", "test_hallucination_filter", "test_lang_vote",
    "test_preprocess_service", "test_transcript_outputs", "test_sortformer_service",
    "test_sortformer_client",
}


@pytest.fixture(autouse=True)
def _stub_models(monkeypatch, request):
    """No ML model runs on the build box. Stub the heavy model wrappers with
    cheap defaults so any pipeline-driving test is model-free; tests that care
    about specific model behavior override these with their own monkeypatch."""
    if request.module.__name__.split(".")[-1] in _NO_STUB_MODULES:
        yield
        return
    import shutil

    def _copy(src, dst):
        shutil.copyfile(src, dst)
        return dst

    try:
        from services import vad_service, enhancement_service, separation_service
    except Exception:
        return
    monkeypatch.setattr(vad_service, "detect_speech",
                        lambda p: [{"start": 0.0, "end": 1.0}], raising=False)
    monkeypatch.setattr(enhancement_service, "enhance", _copy, raising=False)
    monkeypatch.setattr(separation_service, "separate_vocals", _copy, raising=False)

    try:
        from services import (diarization_service,
                              indic_asr_service, embedding_service)
        monkeypatch.setattr(diarization_service, "diarize_with_overlap",
            lambda p, num_speakers=None: ([{"start": 0.0, "end": 1.0, "speaker": "Speaker_1"}], "mock-diarizer"),
            raising=False)
        monkeypatch.setattr(indic_asr_service, "transcribe_clip",
            lambda p, lang=None: {"text": "stub", "confidence": None, "language": lang,
                                  "model": "stub", "abstained": False}, raising=False)
        monkeypatch.setattr(embedding_service, "similarity", lambda a, b: 0.95, raising=False)
    except Exception:
        pass

    # Dual-engine ASR: stub Whisper so pipeline-driving tests never load
    # faster-whisper. Pure-native stub text -> selector keeps IndicConformer,
    # preserving the legacy single-engine behaviour these tests assert.
    try:
        from services import whisper_service
        monkeypatch.setattr(whisper_service, "transcribe_clip",
            lambda p, lang=None: {"text": "స్టబ్", "confidence": 0.9,
                                  "no_speech_prob": 0.0, "compression_ratio": 1.0,
                                  "language": lang, "model": "stub"}, raising=False)
    except Exception:
        pass
    yield

