from services.vad_union import union_segments, total_duration, should_include_separation


def test_union_merges_overlaps_across_branches():
    a = [{"start": 0.0, "end": 2.0}, {"start": 5.0, "end": 6.0}]
    b = [{"start": 1.5, "end": 3.0}, {"start": 10.0, "end": 11.0}]
    u = union_segments([a, b])
    assert u == [
        {"start": 0.0, "end": 3.0},
        {"start": 5.0, "end": 6.0},
        {"start": 10.0, "end": 11.0},
    ]


def test_union_preserves_lone_branch_segments():
    a = []
    b = [{"start": 4.4, "end": 9.0}]
    assert union_segments([a, b]) == [{"start": 4.4, "end": 9.0}]


def test_touching_intervals_coalesce():
    a = [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}]
    assert union_segments([a]) == [{"start": 0.0, "end": 2.0}]


def test_total_duration():
    assert total_duration([{"start": 0.0, "end": 2.0}, {"start": 5.0, "end": 6.0}]) == 3.0


def test_separation_gate():
    assert should_include_separation(10, 12) is True
    assert should_include_separation(10, 10) is True
    assert should_include_separation(10, 7) is False
