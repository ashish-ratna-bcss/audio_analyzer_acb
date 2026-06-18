import json
from services import transcript_service as ts


def test_build_and_write(monkeypatch, tmp_path):
    monkeypatch.setattr(ts.storage.config, "CASE_STORE_PATH", str(tmp_path))

    class Seg:
        def __init__(self, **k): self.__dict__.update(k)

    segs = [Seg(id="seg1", start=0.0, end=1.0, speaker="Speaker_1", text="hi",
                confidence=0.6, source_pass="pass1_enhanced", flagged=True,
                review_status="pending")]
    data = ts.build("c1", "f1", "hashA", segs,
                    status="machine_assisted_pending_certification")
    assert data["source_hash_sha256"] == "hashA"
    assert data["segments"][0]["flagged_for_review"] is True
    assert data["status"] == "machine_assisted_pending_certification"

    path = ts.write("c1", "f1", data)
    assert json.load(open(path))["file_id"] == "f1"
