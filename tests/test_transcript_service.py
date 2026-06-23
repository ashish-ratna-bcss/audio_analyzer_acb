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


class _Seg:
    def __init__(self, **k):
        self.__dict__.update(k)


def _seg(sid, start, end, raw_text, *, llm=None, speaker="Speaker_1", lang="te", conf=0.6):
    cands = {"indic_conformer": {"text": raw_text}}
    if llm is not None:
        cands["llm_enhancement"] = llm
    return _Seg(id=sid, start=start, end=end, speaker=speaker,
                detected_language=lang, confidence=conf, candidates=cands)


def test_build_raw_reads_candidates_not_seg_text():
    segs = [_seg("s1", 0.0, 1.0, "raw telugu")]
    out = ts.build_raw("f1", segs)
    assert out[0]["text"] == "raw telugu"
    assert out[0]["speaker"] == "Speaker_1"
    assert out[0]["language"] == "te"


def test_build_enhanced_falls_back_to_raw_when_no_llm():
    segs = [_seg("s1", 0.0, 1.0, "raw text")]      # no llm_enhancement key
    out = ts.build_enhanced("f1", segs)
    assert out[0]["text"] == "raw text"
    assert out[0]["correction_status"] == "not_run"


def test_build_enhanced_uses_corrected_when_corrected():
    llm = {"correction_status": "corrected", "correction_confidence": 0.9,
           "corrected_text": "fixed text", "changes": []}
    segs = [_seg("s1", 0.0, 1.0, "raw text", llm=llm)]
    out = ts.build_enhanced("f1", segs)
    assert out[0]["text"] == "fixed text"
    assert out[0]["correction_status"] == "corrected"


def test_build_enhanced_falls_back_on_error_status():
    llm = {"correction_status": "error", "corrected_text": "raw text", "changes": []}
    segs = [_seg("s1", 0.0, 1.0, "raw text", llm=llm)]
    out = ts.build_enhanced("f1", segs)
    assert out[0]["text"] == "raw text"
    assert out[0]["correction_status"] == "error"


def test_enrich_diarization_joins_text_by_overlap():
    llm = {"correction_status": "corrected", "correction_confidence": 0.9,
           "corrected_text": "fixed", "changes": []}
    segs = [_seg("s1", 0.0, 2.0, "raw", llm=llm)]
    diar = {"speakers": ["Speaker_1"], "model_version": "pyannote/x",
            "timeline": [{"start": 0.1, "end": 1.9, "speaker": "Speaker_1"}]}

    raw_d = ts.enrich_diarization(diar, segs, use_enhanced=False)
    enh_d = ts.enrich_diarization(diar, segs, use_enhanced=True)
    assert raw_d["timeline"][0]["text"] == "raw"
    assert enh_d["timeline"][0]["text"] == "fixed"
    assert raw_d["model_version"] == "pyannote/x"
    assert raw_d["speakers"] == ["Speaker_1"]
