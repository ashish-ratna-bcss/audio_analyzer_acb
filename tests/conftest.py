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

