import pytest
from unittest.mock import MagicMock, patch


def _make_mock_pipeline(turn_list):
    """turn_list: [(start, end, speaker_label), ...]"""
    mock_diarization = MagicMock()
    mock_turns = []
    for start, end, label in turn_list:
        segment = MagicMock()
        segment.start = start
        segment.end = end
        mock_turns.append((segment, None, label))
    mock_diarization.itertracks.return_value = iter(mock_turns)

    mock_pipeline = MagicMock()
    mock_pipeline.return_value = mock_diarization
    return mock_pipeline


def test_diarize_returns_speaker_segments():
    mock_pipeline = _make_mock_pipeline([
        (0.0, 3.5, "SPEAKER_00"),
        (4.0, 7.0, "SPEAKER_01"),
        (7.5, 10.0, "SPEAKER_00"),
    ])

    with patch("services.diarization_service._pipeline", mock_pipeline):
        from services.diarization_service import diarize
        result = diarize("/tmp/fake.wav")

    assert len(result) == 3
    assert result[0] == {"start": 0.0, "end": 3.5, "speaker": "Speaker_1"}
    assert result[1] == {"start": 4.0, "end": 7.0, "speaker": "Speaker_2"}
    assert result[2] == {"start": 7.5, "end": 10.0, "speaker": "Speaker_1"}


def test_diarize_speaker_label_normalization():
    mock_pipeline = _make_mock_pipeline([
        (0.0, 2.0, "SPEAKER_00"),
    ])

    with patch("services.diarization_service._pipeline", mock_pipeline):
        from services.diarization_service import diarize
        result = diarize("/tmp/fake.wav")

    assert result[0]["speaker"] == "Speaker_1"
