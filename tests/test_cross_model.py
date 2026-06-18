from services.cross_model import normalized_edit_distance, compare_passes


def test_edit_distance_bounds():
    assert normalized_edit_distance("abc", "abc") == 0.0
    assert normalized_edit_distance("", "") == 0.0
    assert normalized_edit_distance("abc", "xyz") == 1.0
    assert 0 < normalized_edit_distance("kitten", "sitting") < 1


def test_agreement_high_confidence():
    r = compare_passes(
        {"p1": "the cost is fifteen", "p2": "the cost is fifteen", "p3": "the cost is fifteen"},
        {"p1": 0.9, "p2": 0.85, "p3": 0.8}, vad_positive=True, embedding_sim=0.95)
    assert r["disagreement"] is False
    assert r["flagged"] is False
    assert r["flag_reason"] is None
    assert r["confidence"] > 0.8


def test_empty_pass_on_vad_positive_is_lowest():
    r = compare_passes(
        {"p1": "", "p2": "something", "p3": "something"},
        {"p1": 0.0, "p2": 0.7, "p3": 0.7}, vad_positive=True, embedding_sim=0.9)
    assert r["flagged"] is True
    assert r["flag_reason"] == "vad_positive_asr_empty"
    assert r["confidence"] == 0.0


def test_disagreement_flagged():
    r = compare_passes(
        {"p1": "the cost is fifty", "p2": "the cost is fifteen", "p3": "totally different words here"},
        {"p1": 0.8, "p2": 0.8, "p3": 0.8}, vad_positive=True, embedding_sim=0.2)
    assert r["disagreement"] is True
    assert r["flagged"] is True
    assert r["flag_reason"] == "cross_model_disagreement"


def test_low_confidence_flagged():
    r = compare_passes(
        {"p1": "same text", "p2": "same text", "p3": "same text"},
        {"p1": 0.2, "p2": 0.2, "p3": 0.2}, vad_positive=True, embedding_sim=0.95)
    assert r["flagged"] is True
    assert r["flag_reason"] == "low_logprob_confidence"
