"""Unit tests for whisper_service — faster-whisper mocked."""
from unittest.mock import MagicMock, patch

import services.whisper_service as ws


def _word(start, end, word, prob=0.9):
    w = MagicMock()
    w.start, w.end, w.word, w.probability = start, end, word, prob
    return w


def _seg(text, words=None):
    s = MagicMock()
    s.text = text
    s.words = words or []
    return s


def _mock_model(segments, language="te"):
    m = MagicMock()
    m.transcribe.return_value = (iter(segments), MagicMock(language=language))
    return m


def test_transcribe_words_collects_word_timestamps():
    segs = [
        _seg(" రేవంత్ గారు", [_word(0.5, 1.0, "రేవంత్", 0.8), _word(1.0, 1.4, "గారు", 0.9)]),
        _seg(" 1100 రూపాయలు", [_word(2.0, 2.5, "1100", 0.95), _word(2.5, 3.0, "రూపాయలు", 0.85)]),
    ]
    with patch.object(ws, "_load", return_value=_mock_model(segs)):
        r = ws.transcribe_words("/tmp/file.wav", "te")
    assert len(r["words"]) == 4
    assert r["words"][0] == {"start": 0.5, "end": 1.0, "word": "రేవంత్", "prob": 0.8}
    assert r["text"] == "రేవంత్ గారు 1100 రూపాయలు"
    assert r["language"] == "te"


def test_transcribe_words_forces_language_and_uses_vad():
    with patch.object(ws, "_load", return_value=_mock_model([_seg(" హలో", [_word(0.0, 0.4, "హలో")])])):
        ws.transcribe_words("/tmp/file.wav", "te")
        kwargs = ws._load().transcribe.call_args[1]
    assert kwargs["language"] == "te"
    assert kwargs["word_timestamps"] is True
    assert kwargs["vad_filter"] is True
    assert kwargs["condition_on_previous_text"] is False


def test_transcribe_words_never_raises():
    with patch.object(ws, "_load", side_effect=RuntimeError("boom")):
        r = ws.transcribe_words("/tmp/file.wav", "te")
    assert r["words"] == []
    assert r["text"] == ""


def test_slice_words_midpoint_membership():
    words = [
        {"start": 0.5, "end": 1.0, "word": "రేవంత్", "prob": 0.8},   # mid 0.75 -> in [0,2]
        {"start": 1.0, "end": 1.4, "word": "గారు", "prob": 0.9},     # mid 1.2  -> in [0,2]
        {"start": 2.0, "end": 2.5, "word": "1100", "prob": 1.0},     # mid 2.25 -> NOT in [0,2]
    ]
    r = ws.slice_words(words, 0.0, 2.0)
    assert r["text"] == "రేవంత్ గారు"
    assert r["confidence"] == round((0.8 + 0.9) / 2, 3)


def test_slice_words_empty_when_no_overlap():
    words = [{"start": 5.0, "end": 5.5, "word": "x", "prob": 0.9}]
    r = ws.slice_words(words, 0.0, 2.0)
    assert r["text"] == ""
    assert r["confidence"] is None


def test_slice_words_handles_empty_list():
    r = ws.slice_words([], 0.0, 2.0)
    assert r["text"] == ""
    assert r["confidence"] is None
