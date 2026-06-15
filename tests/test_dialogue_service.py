from services.dialogue_service import group_turns


def test_merges_consecutive_same_speaker():
    segments = [
        {"speaker": "Speaker_1", "start": 0.0, "end": 1.0, "text": "hello", "confidence": 0.8},
        {"speaker": "Speaker_1", "start": 1.0, "end": 2.0, "text": "how are you", "confidence": 0.6},
        {"speaker": "Speaker_2", "start": 2.0, "end": 3.0, "text": "good", "confidence": 0.9},
    ]
    turns = group_turns(segments)

    assert len(turns) == 2
    assert turns[0]["speaker"] == "Speaker_1"
    assert turns[0]["start"] == 0.0
    assert turns[0]["end"] == 2.0
    assert turns[0]["text"] == "hello how are you"
    assert turns[0]["confidence"] == round((0.8 + 0.6) / 2, 3)
    assert turns[1]["speaker"] == "Speaker_2"
    assert turns[1]["text"] == "good"


def test_preserves_alternating_sequence():
    segments = [
        {"speaker": "Speaker_1", "start": 0.0, "end": 1.0, "text": "a", "confidence": 1.0},
        {"speaker": "Speaker_2", "start": 1.0, "end": 2.0, "text": "b", "confidence": 1.0},
        {"speaker": "Speaker_1", "start": 2.0, "end": 3.0, "text": "c", "confidence": 1.0},
    ]
    turns = group_turns(segments)

    assert [t["speaker"] for t in turns] == ["Speaker_1", "Speaker_2", "Speaker_1"]
    assert [t["text"] for t in turns] == ["a", "b", "c"]


def test_joins_translated_text():
    segments = [
        {"speaker": "Speaker_1", "start": 0.0, "end": 1.0, "text": "namaste",
         "confidence": 0.8, "translated_text": "hello"},
        {"speaker": "Speaker_1", "start": 1.0, "end": 2.0, "text": "ela unnaru",
         "confidence": 0.8, "translated_text": "how are you"},
    ]
    turns = group_turns(segments)

    assert len(turns) == 1
    assert turns[0]["translated_text"] == "hello how are you"


def test_empty_input():
    assert group_turns([]) == []
