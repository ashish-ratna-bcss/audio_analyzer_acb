from fastapi.testclient import TestClient
from db import base as dbbase, repository as repo
import app as appmod

client = TestClient(appmod.app)


def setup_module():
    dbbase.init_db()


def _seed_flagged():
    with dbbase.get_session() as s:
        c = repo.create_case(s); f = repo.create_file(s, c, "a.wav", ".wav")
        seg = repo.add_segment(s, file_id=f, start=0.0, end=1.0, speaker="Speaker_1",
            text="hi", confidence=0.3, source_pass="pass1_enhanced", flagged=True,
            review_status="pending",
            candidates={"pass1_enhanced": "hi", "pass2_original": "hi there",
                        "pass3_indic": "hi"})
        s.commit()
    return c, f, seg


def test_queue_lists_flagged():
    c, f, seg = _seed_flagged()
    r = client.get(f"/review/queue?case_id={c}&status=pending")
    assert r.status_code == 200
    ids = [x["segment_id"] for x in r.json()]
    assert seg in ids


def test_segment_detail_returns_candidates():
    c, f, seg = _seed_flagged()
    r = client.get(f"/review/segments/{seg}")
    assert r.status_code == 200
    assert r.json()["candidates"]["pass2_original"] == "hi there"


def test_submit_edit_decision():
    c, f, seg = _seed_flagged()
    r = client.post(f"/review/segments/{seg}",
                    json={"decision": "edit", "text": "hi there friend",
                          "reviewer_id": "officer_7"})
    assert r.status_code == 200
    with dbbase.get_session() as s:
        updated = repo.get_segment(s, seg)
        assert updated.review_status == "edited"
        assert updated.text == "hi there friend"


def test_reject_decision():
    c, f, seg = _seed_flagged()
    r = client.post(f"/review/segments/{seg}",
                    json={"decision": "reject", "reviewer_id": "officer_7"})
    assert r.status_code == 200
    with dbbase.get_session() as s:
        assert repo.get_segment(s, seg).review_status == "rejected"
