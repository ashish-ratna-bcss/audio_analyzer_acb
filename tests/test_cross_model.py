from services.cross_model import normalized_edit_distance, compare_passes


def _stub_embed(score):
    return lambda a, b: score


def test_edit_distance_bounds():
    assert normalized_edit_distance("abc", "abc") == 0.0
    assert normalized_edit_distance("abc", "xyz") == 1.0


def test_high_agreement_not_flagged():
    r = compare_passes(
        {"whisper": "the cost is fifteen", "indic": "ధర పదిహేను", "seamless": "the cost is fifteen"},
        {"whisper": 0.9, "indic": None, "seamless": 0.8},
        embed_fn=_stub_embed(0.92))
    assert r["flagged"] is False
    assert r["agreement"] >= 0.9
    assert r["consensus_pass"] in {"whisper", "indic", "seamless"}
    assert r["consensus_text"]


def test_insufficient_passes_flagged():
    r = compare_passes(
        {"whisper": "", "indic": "", "seamless": "something"},
        {"whisper": 0.0, "indic": None, "seamless": 0.7},
        embed_fn=_stub_embed(0.9))
    assert r["flagged"] is True
    assert r["flag_reason"] == "insufficient_passes"
    assert r["consensus_pass"] == "seamless"   # the only non-empty pass


def test_disagreement_flagged():
    r = compare_passes(
        {"whisper": "the cost is fifty", "indic": "totally different", "seamless": "unrelated words"},
        {"whisper": 0.8, "indic": 0.8, "seamless": 0.8},
        embed_fn=_stub_embed(0.2))
    assert r["flagged"] is True
    assert r["flag_reason"] == "cross_model_disagreement"


def test_low_confidence_flagged():
    r = compare_passes(
        {"whisper": "same text", "indic": "same text", "seamless": "same text"},
        {"whisper": 0.2, "indic": 0.2, "seamless": 0.2},
        embed_fn=_stub_embed(0.95))
    assert r["flagged"] is True
    assert r["flag_reason"] == "low_confidence"


def test_all_empty():
    r = compare_passes(
        {"whisper": "", "indic": "", "seamless": ""},
        {"whisper": 0.0, "indic": None, "seamless": 0.0},
        embed_fn=_stub_embed(0.0))
    assert r["flagged"] is True
    assert r["flag_reason"] == "insufficient_passes"
    assert r["consensus_text"] == ""


def test_embed_failure_degrades_not_crashes():
    def boom(a, b):
        raise RuntimeError("embed down")
    r = compare_passes(
        {"whisper": "a", "indic": "b", "seamless": "c"},
        {"whisper": 0.9, "indic": 0.9, "seamless": 0.9},
        embed_fn=boom)
    assert r["flagged"] is True
    assert r["agreement"] == 0.0
