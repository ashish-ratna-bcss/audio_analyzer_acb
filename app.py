import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

import config
from api.routes.cases import router as cases_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(config.CASE_STORE_PATH, exist_ok=True)
    # Bootstrap tables when running against SQLite (local/dev). Postgres deploys
    # run `alembic upgrade head` instead.
    if config.DATABASE_URL.startswith("sqlite"):
        from db.base import init_db
        init_db()
    yield


app = FastAPI(title="Forensic Audio Pipeline API", version="2.0.0", lifespan=lifespan)
app.include_router(cases_router)


@app.get("/health")
def health():
    return {"status": "ok"}
