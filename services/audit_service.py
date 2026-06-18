import os
import json
from datetime import datetime, timezone

import config
from services.hashing import sha256_bytes

# Fields hashed into the chain (order fixed for determinism). entry_hash and
# prev_entry_hash are intentionally excluded from the hashed payload.
_PAYLOAD_FIELDS = ["timestamp", "case_id", "file_id", "stage", "model",
                   "model_checkpoint_sha256", "parameters", "input_hash",
                   "output_hash", "operator"]


def ledger_path(case_id: str) -> str:
    return os.path.join(config.CASE_STORE_PATH, "audit", case_id, "ledger.jsonl")


def _canonical(entry: dict) -> bytes:
    payload = {k: entry.get(k) for k in _PAYLOAD_FIELDS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _last_entry_hash(path: str) -> str:
    if not os.path.exists(path):
        return ""
    last = ""
    with open(path) as f:
        for line in f:
            if line.strip():
                last = json.loads(line)["entry_hash"]
    return last


def append_entry(case_id, *, file_id, stage, model=None,
                 model_checkpoint_sha256=None, parameters=None,
                 input_hash=None, output_hash=None, operator="pipeline",
                 session=None) -> dict:
    path = ledger_path(case_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prev = _last_entry_hash(path)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "file_id": file_id,
        "stage": stage,
        "model": model,
        "model_checkpoint_sha256": model_checkpoint_sha256,
        "parameters": parameters,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "operator": operator,
        "prev_entry_hash": prev,
    }
    entry["entry_hash"] = sha256_bytes(prev.encode() + _canonical(entry))
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    if session is not None:
        from db import repository as repo
        repo.add_audit_entry(
            session, case_id=case_id, file_id=file_id, stage=stage,
            payload={k: entry[k] for k in _PAYLOAD_FIELDS},
            prev_entry_hash=prev, entry_hash=entry["entry_hash"],
        )
    return entry


def verify_chain(case_id: str) -> bool:
    path = ledger_path(case_id)
    if not os.path.exists(path):
        return True
    prev = ""
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("prev_entry_hash") != prev:
                return False
            expected = sha256_bytes(prev.encode() + _canonical(entry))
            if entry.get("entry_hash") != expected:
                return False
            prev = entry["entry_hash"]
    return True
