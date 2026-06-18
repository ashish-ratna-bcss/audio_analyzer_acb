#!/usr/bin/env bash
set -e
# Apply DB migrations, then serve. Idempotent: re-running upgrade is a no-op.
alembic upgrade head
exec uvicorn app:app --host 0.0.0.0 --port 8009 --workers 1
