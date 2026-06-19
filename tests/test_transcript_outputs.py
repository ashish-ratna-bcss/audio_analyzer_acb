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
