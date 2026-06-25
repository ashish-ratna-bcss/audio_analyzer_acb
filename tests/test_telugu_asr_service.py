"""Unit tests for telugu_asr_service.transcribe_words — transformers pipe mocked."""
from unittest.mock import patch

import services.telugu_asr_service as ts


def _fake_pipe(result):
    def _call(path, **kwargs):
        _call.kwargs = kwargs
        return result
    return _call


def test_transcribe_words_maps_chunks_to_words():
    result = {
        "text": "రేవంత్ గారు",
        "chunks": [
            {"text": "రేవంత్", "timestamp": (0.5, 1.0)},
            {"text": " గారు", "timestamp": (1.0, 1.4)},
        ],
    }
    with patch.object(ts, "_load", return_value=_fake_pipe(result)):
        r = ts.transcribe_words("/tmp/file.wav", "te")
    assert len(r["words"]) == 2
    assert r["words"][0] == {"start": 0.5, "end": 1.0, "word": "రేవంత్", "prob": None}
    assert r["words"][1]["word"] == "గారు"
    assert r["text"] == "రేవంత్ గారు"
    assert r["language"] == "te"


def test_transcribe_words_skips_missing_timestamps():
    result = {"text": "x", "chunks": [
        {"text": "x", "timestamp": (None, None)},
        {"text": "", "timestamp": (1.0, 2.0)},
    ]}
    with patch.object(ts, "_load", return_value=_fake_pipe(result)):
        r = ts.transcribe_words("/tmp/file.wav", "te")
    assert r["words"] == []


def test_transcribe_words_requests_word_timestamps_and_language():
    pipe = _fake_pipe({"text": "హలో", "chunks": [{"text": "హలో", "timestamp": (0.0, 0.4)}]})
    with patch.object(ts, "_load", return_value=pipe):
        ts.transcribe_words("/tmp/file.wav", "te")
    assert pipe.kwargs["return_timestamps"] == "word"
    assert pipe.kwargs["generate_kwargs"]["language"] == "te"
    assert pipe.kwargs["generate_kwargs"]["task"] == "transcribe"


def test_transcribe_words_never_raises():
    with patch.object(ts, "_load", side_effect=RuntimeError("boom")):
        r = ts.transcribe_words("/tmp/file.wav", "te")
    assert r["words"] == []
    assert r["text"] == ""
