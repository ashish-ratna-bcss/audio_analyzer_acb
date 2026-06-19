from unittest.mock import patch
from services import sortformer_service as ss


def test_parse_segment_string():
    assert ss._parse_segment("0.23 5.67 speaker_0") == (0.23, 5.67, "speaker_0")


def test_parse_segment_tuple():
    assert ss._parse_segment([1.0, 2.5, "speaker_1"]) == (1.0, 2.5, "speaker_1")


def test_parse_segment_bad():
    assert ss._parse_segment("garbage") is None
    assert ss._parse_segment("a b c") is None
    assert ss._parse_segment(42) is None


def test_diarize_maps_speakers_and_overlap():
    class _FakeModel:
        def diarize(self, audio, batch_size=1, include_tensor_outputs=False):
            # two speakers, with an overlapping interval retained as separate turns
            return [[
                "0.0 3.0 speaker_0",
                "2.5 5.0 speaker_1",     # overlaps speaker_0
                "5.0 6.0 speaker_0",
            ]]

    with patch.object(ss, "get_model", return_value=_FakeModel()):
        out = ss.diarize_with_overlap("/tmp/x.wav")

    assert len(out) == 3
    assert out[0] == {"start": 0.0, "end": 3.0, "speaker": "Speaker_1"}
    assert out[1] == {"start": 2.5, "end": 5.0, "speaker": "Speaker_2"}
    assert out[2] == {"start": 5.0, "end": 6.0, "speaker": "Speaker_1"}  # same raw spk -> same label


def test_diarize_empty():
    class _FakeModel:
        def diarize(self, audio, batch_size=1, include_tensor_outputs=False):
            return [[]]

    with patch.object(ss, "get_model", return_value=_FakeModel()):
        assert ss.diarize_with_overlap("/tmp/x.wav") == []
