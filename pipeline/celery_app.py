from celery import Celery

import config

celery = Celery("forensic", broker=config.REDIS_URL, backend=config.REDIS_URL)
celery.conf.update(
    task_always_eager=config.CELERY_TASK_ALWAYS_EAGER,
    task_eager_propagates=True,
    task_default_queue=config.CPU_QUEUE,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)

# Ensure task modules are imported so they register on the app.
celery.autodiscover_tasks(["pipeline"])
import pipeline.tasks  # noqa: E402,F401
