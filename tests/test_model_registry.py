"""Unit tests for VRAM-release model unloading."""
import importlib

import config
from services import model_registry, whisper_service, embedding_service


def test_unload_nulls_cached_globals(monkeypatch):
    monkeypatch.setattr(config, "ASR_UNLOAD_AFTER_JOB", True)
    # simulate loaded models
    whisper_service._model = object()
    embedding_service._model = object()
    freed = model_registry.unload_all()
    assert whisper_service._model is None
    assert embedding_service._model is None
    assert "whisper_service._model" in freed
    assert "embedding_service._model" in freed


def test_unload_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(config, "ASR_UNLOAD_AFTER_JOB", False)
    whisper_service._model = object()
    freed = model_registry.unload_all()
    assert freed == []
    assert whisper_service._model is not None
    whisper_service._model = None   # cleanup


def test_unload_handles_already_none(monkeypatch):
    monkeypatch.setattr(config, "ASR_UNLOAD_AFTER_JOB", True)
    whisper_service._model = None
    embedding_service._model = None
    # nothing loaded -> empty freed list, no error
    assert model_registry.unload_all() == []


def teardown_module():
    importlib.reload(config)
