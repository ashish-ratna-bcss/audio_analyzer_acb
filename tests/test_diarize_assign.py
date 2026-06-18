from services.diarize_assign import assign_speakers

TURNS = [
    {"start": 0.0, "end": 5.0, "speaker": "Speaker_1"},
    {"start": 4.0, "end": 9.0, "speaker": "Speaker_2"},
]


def test_single_speaker():
    r = assign_speakers({"start": 0.0, "end": 3.0}, TURNS)
    assert r == {"speakers": ["Speaker_1"], "overlap": False}


def test_overlap_two_speakers():
    r = assign_speakers({"start": 4.2, "end": 4.9}, TURNS)
    assert set(r["speakers"]) == {"Speaker_1", "Speaker_2"}
    assert r["overlap"] is True


def test_no_overlap_falls_back_to_nearest():
    r = assign_speakers({"start": 20.0, "end": 21.0}, TURNS)
    assert r["speakers"] == ["Speaker_2"]  # nearest by time
    assert r["overlap"] is False
