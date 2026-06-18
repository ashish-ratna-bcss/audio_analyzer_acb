from db import base as dbbase, repository as repo


def setup_module():
    dbbase.init_db()


def test_segment_stores_candidates_and_clips():
    with dbbase.get_session() as s:
        c = repo.create_case(s); f = repo.create_file(s, c, "a.wav", ".wav")
        seg_id = repo.add_segment(
            s, file_id=f, start=0.0, end=1.0, speaker="Speaker_1",
            text="hi", confidence=0.6, source_pass="pass1_enhanced",
            flagged=True, review_status="pending",
            candidates={"pass1_enhanced": "hi", "pass2_original": "hi there",
                        "pass3_indic": "hi"},
            clip_original="/p/org.wav", clip_enhanced="/p/enh.wav")
        s.commit()
    with dbbase.get_session() as s:
        seg = repo.get_segment(s, seg_id)
        assert seg.candidates["pass2_original"] == "hi there"
        assert seg.clip_original == "/p/org.wav"
