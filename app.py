import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from api.routes.cases import router as cases_router
from api.routes.review import router as review_router


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

# CORS: the browser UI calls this API cross-origin (different host/port), so the
# browser blocks responses unless these headers are present (curl is unaffected,
# which is why the API tested fine from the shell but the web app saw it as
# "unreachable"). Auth is via the X-API-Key header (not cookies), so a wildcard
# origin is safe with credentials disabled. Restrict via CORS_ORIGINS (comma-sep).
_cors = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cases_router)
app.include_router(review_router)


@app.get("/")
@app.get("/health")
def health():
    """Root + /health both return 200 so UI reachability probes pass."""
    return {"status": "ok", "service": "forensic-audio", "version": "2.0.0"}
