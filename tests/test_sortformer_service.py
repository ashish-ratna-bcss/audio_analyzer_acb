from services import sortformer_service as ss


def test_parse_segment_space():
    assert ss.parse_segment("0.23 5.67 speaker_0") == (0.23, 5.67, "speaker_0")


def test_parse_segment_comma():
    assert ss.parse_segment("0.23, 5.67, 0") == (0.23, 5.67, "0")


def test_parse_segment_tuple():
    assert ss.parse_segment([1.0, 2.5, "speaker_1"]) == (1.0, 2.5, "speaker_1")


def test_parse_segment_bad():
    assert ss.parse_segment("garbage") is None
    assert ss.parse_segment("a b c") is None
    assert ss.parse_segment(42) is None


def test_build_segments_maps_and_overlaps():
    preds = ["0.0 3.0 speaker_0", "2.5 5.0 speaker_1", "5.0 6.0 speaker_0"]
    out = ss.build_segments(preds)
    assert out[0] == {"start": 0.0, "end": 3.0, "speaker": "Speaker_1"}
    assert out[1] == {"start": 2.5, "end": 5.0, "speaker": "Speaker_2"}  # overlap retained
    assert out[2] == {"start": 5.0, "end": 6.0, "speaker": "Speaker_1"}  # same raw -> same label


def test_build_segments_drops_bad_and_zero_len():
    out = ss.build_segments(["bad", "2.0 2.0 spk", "1.0 2.0 spk0"])
    assert out == [{"start": 1.0, "end": 2.0, "speaker": "Speaker_1"}]


def test_build_segments_empty():
    assert ss.build_segments([]) == []
    assert ss.build_segments(None) == []
