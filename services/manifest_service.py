import os
import json

import config


def manifest_path(case_id: str) -> str:
    d = os.path.join(config.CASE_STORE_PATH, "cases", case_id)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "manifest.json")


def load(case_id: str) -> dict:
    p = manifest_path(case_id)
    if not os.path.exists(p):
        return {"case_id": case_id, "files": {}}
    with open(p) as f:
        return json.load(f)


def _save(case_id: str, data: dict) -> None:
    with open(manifest_path(case_id), "w") as f:
        json.dump(data, f, indent=2)


def register_file(case_id, file_id, original_filename, sha256):
    m = load(case_id)
    m["files"].setdefault(file_id, {})
    m["files"][file_id].update({
        "original_filename": original_filename,
        "source_sha256": sha256,
        "derivatives": m["files"][file_id].get("derivatives", []),
    })
    _save(case_id, m)


def register_derivative(case_id, file_id, kind, path, sha256, parent_sha256):
    m = load(case_id)
    entry = m["files"].setdefault(file_id, {"derivatives": []})
    entry.setdefault("derivatives", []).append({
        "kind": kind, "path": path, "sha256": sha256,
        "parent_sha256": parent_sha256,
    })
    _save(case_id, m)
