"""Unit tests for coverage-recovery window grouping (pipeline.tasks)."""
from pipeline.tasks import _uncovered_windows


def _w(start, end, word="x"):
    return {"start": start, "end": end, "word": word, "prob": 0.9}


def test_all_covered_returns_empty():
    words = [_w(1.0, 1.5), _w(2.0, 2.5)]
    covered = [(0.0, 3.0)]
    assert _uncovered_windows(words, covered, gap_s=2.0, max_win_s=15.0) == []


def test_uncovered_region_grouped():
    # words at 5-7s fall outside the only covered interval (0-3)
    words = [_w(5.0, 5.5), _w(5.6, 6.0), _w(6.2, 7.0)]
    covered = [(0.0, 3.0)]
    wins = _uncovered_windows(words, covered, gap_s=2.0, max_win_s=15.0)
    assert wins == [(5.0, 7.0)]


def test_gap_splits_windows():
    # 1s gap stays together; a 3s gap (> gap_s=2) splits
    words = [_w(5.0, 5.5), _w(6.0, 6.5), _w(10.0, 10.5)]
    covered = []
    wins = _uncovered_windows(words, covered, gap_s=2.0, max_win_s=15.0)
    assert wins == [(5.0, 6.5), (10.0, 10.5)]


def test_max_window_splits():
    # continuous words but window cap 3s forces a split
    words = [_w(0.0, 0.5), _w(0.6, 1.0), _w(1.1, 1.5), _w(2.0, 2.5),
             _w(3.1, 3.6), _w(3.7, 4.2)]
    covered = []
    wins = _uncovered_windows(words, covered, gap_s=2.0, max_win_s=3.0)
    assert len(wins) >= 2
    # first window must not exceed the cap
    assert wins[0][1] - wins[0][0] <= 3.0 + 1e-6


def test_midpoint_membership():
    # a word straddling the covered boundary is judged by its midpoint
    words = [_w(2.5, 3.5)]      # midpoint 3.0 -> inside (0,3)? 3.0<=3.0 covered
    assert _uncovered_windows(words, [(0.0, 3.0)], 2.0, 15.0) == []
    words = [_w(3.0, 4.0)]      # midpoint 3.5 -> uncovered
    assert _uncovered_windows(words, [(0.0, 3.0)], 2.0, 15.0) == [(3.0, 4.0)]
