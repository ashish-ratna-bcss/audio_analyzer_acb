import pytest
from unittest.mock import MagicMock, patch


def test_transcribe_returns_expected_shape():
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 3.5
    mock_segment.text = " hello world"

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


def test_transcribe_auto_language():
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 2.0
    mock_segment.text = " నమస్కారం"

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
