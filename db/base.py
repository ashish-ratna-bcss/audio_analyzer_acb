from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool

import config


class Base(DeclarativeBase):
    pass


# SQLite needs check_same_thread=False for the threaded test/dev server.
_is_sqlite = config.DATABASE_URL.startswith("sqlite")
_connect_args = {"check_same_thread": False} if _is_sqlite else {}
# In-memory SQLite lives in the connection; StaticPool keeps a single shared
# connection so every session sees the same tables/data (tests).
_engine_kwargs = {}
if _is_sqlite and ":memory:" in config.DATABASE_URL:
    _engine_kwargs["poolclass"] = StaticPool
engine = create_engine(config.DATABASE_URL, connect_args=_connect_args, future=True,
                       **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db():
    """Create all tables. Used by tests and first-run bootstrap; production
    schema is managed by Alembic migrations."""
    # Import models so they register on Base.metadata before create_all.
    from db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
