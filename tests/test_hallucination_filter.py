from services.hallucination_filter import has_repetition, filter_pass


def test_repetition_consecutive():
    assert has_repetition("go go go now") is True


def test_repetition_low_unique_ratio():
    assert has_repetition("a a a a b a a a") is True


def test_no_repetition_normal():
    assert has_repetition("the cost is fifteen rupees") is False


def test_ghost_phrase_blanked():
    out = filter_pass({"text": "Thank you.", "confidence": 0.6})
    assert out["text"] == ""
    assert out["confidence"] == 0.0
    assert out["hallucination"] == "ghost_phrase"


def test_no_speech_blanked():
    out = filter_pass({"text": "real words here", "confidence": 0.7}, no_speech_prob=0.9)
    assert out["text"] == ""
    assert out["hallucination"] == "no_speech"


def test_repetition_blanked():
    out = filter_pass({"text": "fifteen fifteen fifteen fifteen", "confidence": 0.5})
    assert out["text"] == ""
    assert out["hallucination"] == "repetition"


def test_clean_passes_through():
    src = {"text": "the cost is fifteen rupees", "confidence": 0.8}
    out = filter_pass(src, no_speech_prob=0.1)
    assert out == src
    assert "hallucination" not in out
