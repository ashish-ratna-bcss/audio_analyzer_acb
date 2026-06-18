import os, subprocess, sys, tempfile, pathlib


def test_alembic_upgrade_head_creates_tables():
    tmp = tempfile.mkdtemp()
    db_path = pathlib.Path(tmp) / "mig.db"
    env = {**os.environ, "DATABASE_URL": f"sqlite+pysqlite:///{db_path}"}
    root = pathlib.Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"],
                       cwd=root, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    import sqlite3
    con = sqlite3.connect(db_path)
    names = {row[0] for row in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert {"cases", "files", "jobs", "segments", "reviews", "audit_entries"} <= names
