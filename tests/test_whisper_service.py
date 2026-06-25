"""Unit tests for whisper_service.transcribe_clip — faster-whisper mocked."""
import math
from unittest.mock import MagicMock, patch

import services.whisper_service as ws


def _seg(text, avg_logprob=-0.2, no_speech_prob=0.05, compression_ratio=1.5):
    s = MagicMock()
    s.text = text
    s.avg_logprob = avg_logprob
    s.no_speech_prob = no_speech_prob
    s.compression_ratio = compression_ratio
    return s


def _mock_model(segments, language="te"):
    m = MagicMock()
    m.transcribe.return_value = (iter(segments), MagicMock(language=language))
    return m


def test_transcribe_clip_concatenates_and_aggregates():
    model = _mock_model([
        _seg(" statement ఇచ్చినా", avg_logprob=-0.2, no_speech_prob=0.05, compression_ratio=1.4),
        _seg(" bank లో", avg_logprob=-0.4, no_speech_prob=0.30, compression_ratio=1.9),
    ])
    with patch.object(ws, "_load", return_value=model):
        r = ws.transcribe_clip("/tmp/x.wav", "te")
    assert r["text"] == "statement ఇచ్చినా bank లో"
    # mean of the two exp(avg_logprob)
    expected = round((math.exp(-0.2) + math.exp(-0.4)) / 2, 3)
    assert r["confidence"] == expected
    assert r["no_speech_prob"] == 0.30           # max (most non-speech window)
    assert r["compression_ratio"] == 1.9         # max (most repetitive window)
    assert r["model"] == ws.config.WHISPER_MODEL


def test_transcribe_clip_forces_language():
    model = _mock_model([_seg(" హలో")])
    with patch.object(ws, "_load", return_value=model):
        ws.transcribe_clip("/tmp/x.wav", "te")
    kwargs = model.transcribe.call_args[1]
    assert kwargs["language"] == "te"
    assert kwargs["vad_filter"] is False
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["temperature"] == 0.0


def test_transcribe_clip_auto_when_no_lang():
    model = _mock_model([_seg(" హలో")])
    with patch.object(ws, "_load", return_value=model):
        ws.transcribe_clip("/tmp/x.wav", None)
    kwargs = model.transcribe.call_args[1]
    assert "language" not in kwargs


def test_transcribe_clip_never_raises():
    def _boom():
        raise RuntimeError("model load failed")
    with patch.object(ws, "_load", side_effect=_boom):
        r = ws.transcribe_clip("/tmp/x.wav", "te")
    assert r["text"] == ""
    assert r["confidence"] is None
    assert r["model"] == ws.config.WHISPER_MODEL


def test_transcribe_clip_empty_segments():
    model = _mock_model([])
    with patch.object(ws, "_load", return_value=model):
        r = ws.transcribe_clip("/tmp/x.wav", "te")
    assert r["text"] == ""
    assert r["confidence"] is None
    assert r["no_speech_prob"] is None
