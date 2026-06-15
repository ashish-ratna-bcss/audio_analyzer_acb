import pytest
from services.alignment_service import align_segments


def test_basic_alignment():
    whisper_segs = [
        {"start": 0.5, "end": 4.3, "text": "Hello sir"},
        {"start": 4.5, "end": 8.7, "text": "What is your issue"},
        {"start": 9.0, "end": 12.0, "text": "My file is pending"},
    ]
    speaker_segs = [
        {"start": 0.0, "end": 4.4, "speaker": "Speaker_1"},
        {"start": 4.4, "end": 9.0, "speaker": "Speaker_2"},
        {"start": 9.0, "end": 13.0, "speaker": "Speaker_1"},
    ]
    result = align_segments(whisper_segs, speaker_segs)
    assert len(result) == 3
    assert result[0]["speaker"] == "Speaker_1"
    assert result[0]["text"] == "Hello sir"
    assert result[1]["speaker"] == "Speaker_2"
    assert result[2]["speaker"] == "Speaker_1"


def test_no_speaker_segments_defaults_unknown():
    whisper_segs = [{"start": 0.0, "end": 2.0, "text": "hello"}]
    result = align_segments(whisper_segs, [])
    assert result[0]["speaker"] == "Unknown"


def test_segment_assigned_to_max_overlap_speaker():
    whisper_segs = [{"start": 1.0, "end": 5.0, "text": "test"}]
    speaker_segs = [
        {"start": 0.0, "end": 2.5, "speaker": "Speaker_1"},  # overlap 1.5s
        {"start": 2.5, "end": 6.0, "speaker": "Speaker_2"},  # overlap 2.5s
    ]
    result = align_segments(whisper_segs, speaker_segs)
    assert result[0]["speaker"] == "Speaker_2"


def test_start_end_times_preserved():
    whisper_segs = [{"start": 1.1, "end": 3.3, "text": "hi"}]
    speaker_segs = [{"start": 0.0, "end": 5.0, "speaker": "Speaker_1"}]
    result = align_segments(whisper_segs, speaker_segs)
    assert result[0]["start"] == 1.1
    assert result[0]["end"] == 3.3


def test_zero_overlap_uses_time_distance_fallback():
    """Segments at boundaries with zero overlap should use closest speaker by time."""
    whisper_segs = [
        {"start": 4.4, "end": 9.0, "text": "boundary segment"},
    ]
    speaker_segs = [
        {"start": 0.0, "end": 4.4, "speaker": "Speaker_1"},  # ends exactly where wseg starts
        {"start": 9.0, "end": 13.0, "speaker": "Speaker_2"},  # starts exactly where wseg ends
    ]
    result = align_segments(whisper_segs, speaker_segs)
    # Both have 0 overlap. Speaker_1 ends at 4.4 (distance 0 from wseg start)
    # Speaker_2 starts at 9.0 (distance 0 from wseg end). Should pick first one found with min distance.
    assert result[0]["speaker"] == "Speaker_1"


def test_multiple_zero_overlap_picks_closest():
    """With multiple zero-overlap options, should pick the one closest in time."""
    whisper_segs = [{"start": 5.0, "end": 6.0, "text": "gap segment"}]
    speaker_segs = [
        {"start": 0.0, "end": 4.0, "speaker": "Speaker_1"},  # distance: min(|5-0|, |6-4|) = 2
        {"start": 7.0, "end": 10.0, "speaker": "Speaker_2"},  # distance: min(|5-7|, |6-10|) = 2
        {"start": 5.5, "end": 6.5, "speaker": "Speaker_3"},  # distance: min(|5-5.5|, |6-6.5|) = 0.5 (closest)
    ]
    result = align_segments(whisper_segs, speaker_segs)
    assert result[0]["speaker"] == "Speaker_3"
