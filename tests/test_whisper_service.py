import math
import pytest
from unittest.mock import MagicMock, patch


import services.whisper_service  # ensure submodule is importable before patching


def _make_segment(start, end, text, avg_logprob=-0.2, no_speech_prob=0.05,
                  compression_ratio=1.5):
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    seg.avg_logprob = avg_logprob
    seg.no_speech_prob = no_speech_prob
    seg.compression_ratio = compression_ratio
    return seg


def test_transcribe_returns_expected_shape():
    mock_segment = _make_segment(0.0, 3.5, " hello world", avg_logprob=-0.2)

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        iter([mock_segment]),
        MagicMock(language="te", duration=3.5)
    )

    with patch("services.whisper_service._model", mock_model):
        from services.whisper_service import transcribe
        result = transcribe("/tmp/fake.wav", language="te")

    assert result["language"] == "te"
    assert result["duration"] == 3.5
    assert len(result["segments"]) == 1
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][0]["end"] == 3.5
    assert result["segments"][0]["text"] == "hello world"
    # confidence derived from the model's own avg_logprob, not a heuristic
    assert result["segments"][0]["confidence"] == round(math.exp(-0.2), 3)
    assert result["segments"][0]["no_speech_prob"] == 0.05
    assert result["segments"][0]["compression_ratio"] == 1.5


def test_transcribe_auto_language():
    mock_segment = _make_segment(0.0, 2.0, " నమస్కారం")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        iter([mock_segment]),
        MagicMock(language="te", duration=2.0)
    )

    with patch("services.whisper_service._model", mock_model):
        from services.whisper_service import transcribe
        result = transcribe("/tmp/fake.wav", language="auto")

    mock_model.transcribe.assert_called_once()
    call_kwargs = mock_model.transcribe.call_args[1]
    assert "language" not in call_kwargs or call_kwargs.get("language") is None
