from models.schemas import (
    TranscribeRequest, TranscribeResponse, Block, DialogueTurn, SegmentDetail,
)


def test_transcribe_request_defaults():
    req = TranscribeRequest()
    assert req.language == "auto"
    assert req.debug is False


def test_transcribe_request_custom():
    req = TranscribeRequest(language="te", debug=True)
    assert req.language == "te"
    assert req.debug is True


def test_dialogue_turn_confidence_optional():
    turn = DialogueTurn(start=0.5, end=4.3, speaker="Speaker_1", text="hello")
    assert turn.confidence is None


def test_transcribe_response_structure():
    resp = TranscribeResponse(
        language="te",
        duration=125.5,
        raw=Block(dialogue=[
            DialogueTurn(start=0.5, end=4.3, speaker="Speaker_1", text="హలో"),
        ]),
        english=Block(dialogue=[
            DialogueTurn(start=0.5, end=4.3, speaker="Speaker_1", text="hello"),
        ]),
    )
    assert resp.language == "te"
    assert len(resp.raw.dialogue) == 1
    assert len(resp.english.dialogue) == 1
    assert resp.english.dialogue[0].text == "hello"
    assert resp.raw.segments is None  # debug-only


def test_block_debug_segments():
    block = Block(
        dialogue=[DialogueTurn(start=0.0, end=1.0, speaker="Speaker_1", text="hi", confidence=0.9)],
        segments=[SegmentDetail(
            start=0.0, end=1.0, speaker="Speaker_1", text="hi",
            confidence=0.9, no_speech_prob=0.1, compression_ratio=1.5,
        )],
    )
    assert block.segments[0].confidence == 0.9
    assert block.dialogue[0].confidence == 0.9
