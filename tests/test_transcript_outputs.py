from types import SimpleNamespace
from services import transcript_service as ts


def _seg():
    return SimpleNamespace(
        id="s1", start=1.0, end=2.0, speaker="Speaker_1",
        detected_language="te", flagged=True,
        candidates={
            "pass1_whisper": {"text": "hello", "confidence": 0.8},
            "pass2_indic_conformer": {"text": "హలో", "confidence": None},
            "pass3_seamless": {"text": "", "confidence": 0.0, "hallucination": "no_speech"},
            "agreement": 0.71, "consensus_pass": "pass1_whisper",
        })


def test_build_per_model_whisper():
    d = ts.build_per_model("f1", [_seg()], "pass1_whisper")
    assert d["file_id"] == "f1"
    assert d["model"] == "pass1_whisper"
    seg = d["segments"][0]
    assert seg["text"] == "hello"
    assert seg["confidence"] == 0.8
    assert seg["speaker"] == "Speaker_1"


def test_build_per_model_seamless_empty_keeps_flag():
    d = ts.build_per_model("f1", [_seg()], "pass3_seamless")
    seg = d["segments"][0]
    assert seg["text"] == ""
    assert seg["flagged_for_review"] is True


def test_validation_report_shape():
    r = ts.build_validation_report("f1", [_seg()])
    assert r["file_id"] == "f1"
    seg = r["segments"][0]
    assert seg["whisper"]["text"] == "hello"
    assert seg["indic"]["text"] == "హలో"
    assert seg["agreement"] == 0.71
    assert seg["consensus_pass"] == "pass1_whisper"
    assert "summary" in r
    assert r["summary"]["segments_total"] == 1


def _table_segs():
    return [
        SimpleNamespace(id="a", start=328.45, end=329.0, speaker="Speaker_1",
                        detected_language="te", text="వచ్చేసినవా?"),
        SimpleNamespace(id="b", start=329.3, end=330.0, speaker="Speaker_2",
                        detected_language="te", text=""),       # empty -> skipped
        SimpleNamespace(id="c", start=393.0, end=393.9, speaker="Speaker_1",
                        detected_language="te", text="సరే"),
    ]


def test_conversation_table_json():
    t = ts.build_conversation_table("f1", _table_segs())
    rows = t["rows"]
    assert len(rows) == 2                       # empty segment dropped
    assert rows[0]["sl"] == 1
    assert rows[0]["time"] == "05.28"           # 328.45s -> MM.SS (floored seconds)
    assert rows[0]["person"] == "Speaker_1"
    assert rows[0]["conversation"] == "వచ్చేసినవా?"
    assert rows[1]["time"] == "06.33"


def test_conversation_table_markdown():
    md = ts.render_conversation_markdown(ts.build_conversation_table("f1", _table_segs()))
    assert "| Sl | Time | Person | Conversation |" in md
    assert "వచ్చేసినవా?" in md
    assert "05.28" in md
