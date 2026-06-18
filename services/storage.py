import os
import shutil
import stat
import subprocess

import config
from services.hashing import sha256_file


def case_dir(case_id: str) -> str:
    return os.path.join(config.CASE_STORE_PATH, "cases", case_id)


def originals_dir(case_id: str) -> str:
    p = os.path.join(case_dir(case_id), "originals")
    os.makedirs(p, exist_ok=True)
    return p


def derivatives_dir(case_id: str, file_id: str) -> str:
    p = os.path.join(case_dir(case_id), "derivatives", file_id)
    os.makedirs(p, exist_ok=True)
    return p


def derivative_path(case_id: str, file_id: str, subdir: str, filename: str) -> str:
    p = os.path.join(derivatives_dir(case_id, file_id), subdir)
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, filename)


def _make_immutable(path: str) -> None:
    # Best-effort WORM: try chattr +i, else fall back to read-only perms.
    try:
        subprocess.run(["chattr", "+i", path], capture_output=True, check=True)
        return
    except Exception:
        pass
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass


def write_original(case_id: str, file_id: str, ext: str, src_path: str):
    dest = os.path.join(originals_dir(case_id), f"{file_id}__original{ext}")
    shutil.move(src_path, dest)
    digest = sha256_file(dest)
    _make_immutable(dest)
    return dest, digest
