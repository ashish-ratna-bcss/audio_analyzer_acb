import pytest
from models.schemas import TranscribeRequest, TranscribeResponse, Segment


def test_transcribe_request_defaults():
    req = TranscribeRequest()
    assert req.language == "auto"
    assert req.diarize is True
    assert req.translate is False
    assert req.translate_to == "en"


def test_transcribe_request_custom():
    req = TranscribeRequest(language="te", diarize=False, translate=True)
    assert req.language == "te"
    assert req.diarize is False
    assert req.translate is True


def test_transcribe_request_invalid_language():
    with pytest.raises(Exception):
        TranscribeRequest(language="fr")


def test_segment_optional_translation():
    seg = Segment(speaker="Speaker_1", start=0.5, end=4.3, text="hello")
    assert seg.translated_text is None


def test_transcribe_response_structure():
    resp = TranscribeResponse(
        language="te",
        duration=125.5,
        text="full transcript",
        segments=[
            Segment(speaker="Speaker_1", start=0.5, end=4.3, text="hello")
        ]
    )
    assert resp.language == "te"
    assert len(resp.segments) == 1
