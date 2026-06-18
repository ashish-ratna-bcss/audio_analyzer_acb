import importlib


def _reload_config(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import config
    return importlib.reload(config)


def test_defaults_present(monkeypatch):
    cfg = _reload_config(monkeypatch)
    assert cfg.CPU_QUEUE == "cpu_queue"
    assert cfg.GPU_QUEUE == "gpu_queue"
    # sane local defaults
    assert cfg.DATABASE_URL.startswith("sqlite") or cfg.DATABASE_URL.startswith("postgresql")
    assert cfg.REDIS_URL.startswith("redis://")
    assert isinstance(cfg.CASE_STORE_PATH, str) and cfg.CASE_STORE_PATH


def test_env_overrides(monkeypatch):
    cfg = _reload_config(
        monkeypatch,
        DATABASE_URL="postgresql+psycopg2://u:p@db:5432/forensic",
        REDIS_URL="redis://redis:6379/1",
        CASE_STORE_PATH="/data/forensic-audio",
        CELERY_TASK_ALWAYS_EAGER="true",
    )
    assert cfg.DATABASE_URL == "postgresql+psycopg2://u:p@db:5432/forensic"
    assert cfg.REDIS_URL == "redis://redis:6379/1"
    assert cfg.CASE_STORE_PATH == "/data/forensic-audio"
    assert cfg.CELERY_TASK_ALWAYS_EAGER is True
