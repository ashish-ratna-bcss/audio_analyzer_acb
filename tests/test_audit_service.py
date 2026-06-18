import json
from services import audit_service as au


def _read_lines(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def test_chain_links_and_verifies(monkeypatch, tmp_path):
    monkeypatch.setattr(au.config, "CASE_STORE_PATH", str(tmp_path))
    e1 = au.append_entry("case-x", file_id="f1", stage="L0", output_hash="aaa")
    e2 = au.append_entry("case-x", file_id="f1", stage="L1", input_hash="aaa",
                         output_hash="bbb")
    assert e1["prev_entry_hash"] == ""
    assert e2["prev_entry_hash"] == e1["entry_hash"]
    assert au.verify_chain("case-x") is True

    lines = _read_lines(au.ledger_path("case-x"))
    assert len(lines) == 2 and lines[1]["stage"] == "L1"


def test_tamper_breaks_chain(monkeypatch, tmp_path):
    monkeypatch.setattr(au.config, "CASE_STORE_PATH", str(tmp_path))
    au.append_entry("case-y", file_id="f1", stage="L0", output_hash="aaa")
    au.append_entry("case-y", file_id="f1", stage="L1", output_hash="bbb")
    p = au.ledger_path("case-y")
    lines = open(p).read().splitlines()
    rec = json.loads(lines[0]); rec["output_hash"] = "TAMPERED"
    lines[0] = json.dumps(rec)
    open(p, "w").write("\n".join(lines) + "\n")
    assert au.verify_chain("case-y") is False
