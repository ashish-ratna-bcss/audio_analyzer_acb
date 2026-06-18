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


@pytest.fixture(autouse=True)
def _stub_models(monkeypatch):
    """No ML model runs on the build box. Stub the heavy model wrappers with
    cheap defaults so any pipeline-driving test is model-free; tests that care
    about specific model behavior override these with their own monkeypatch."""
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
    yield

