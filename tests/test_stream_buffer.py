"""Contracts for the live stream ring.

The cases that matter are eviction and the ring seam: every other bug in a
structure like this shows up as a plot that is subtly wrong for a few frames
after the buffer wraps, which is exactly the failure nobody notices.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.stream_buffer import (MIN_CAPACITY, StreamBuffer, capacity_for_budget)


def _frame(val, ny=2, nx=3, channels=("change",)):
    return {c: np.full((ny, nx), float(val), np.float32) for c in channels}


def _fill(buf, lo, hi, channels=("change",)):
    for i in range(lo, hi):
        buf.append(i, _frame(i, buf.ny, buf.nx, channels))


def test_span_tracks_appends():
    buf = StreamBuffer(("change",), 2, 3, capacity=10)
    assert (buf.start, buf.count, buf.frontier) == (0, 0, 0)
    _fill(buf, 0, 4)
    assert (buf.start, buf.count, buf.frontier) == (0, 4, 4)
    assert not buf.evicted


def test_non_contiguous_append_is_rejected():
    """A worker that skipped a frame must not be able to produce a buffer whose
    span claims uniform sampling. Every time axis downstream believes it."""
    buf = StreamBuffer(("change",), 2, 3, capacity=10)
    _fill(buf, 0, 3)
    with pytest.raises(ValueError, match="non-contiguous"):
        buf.append(7, _frame(7))
    with pytest.raises(ValueError, match="non-contiguous"):
        buf.append(1, _frame(1))            # backwards, too


def test_reset_starts_a_new_island_anywhere():
    buf = StreamBuffer(("change",), 2, 3, capacity=10)
    _fill(buf, 0, 5)
    buf.reset(900)
    assert (buf.start, buf.count, buf.frontier) == (900, 0, 900)
    _fill(buf, 900, 903)
    assert buf.covers(900, 903)
    assert not buf.covers(0, 5)             # the old island is gone
    assert buf.window(900, 903, "change")[0, 0, 0] == pytest.approx(900.0)


def test_eviction_slides_the_span_and_keeps_capacity():
    buf = StreamBuffer(("change",), 2, 3, capacity=8)
    _fill(buf, 0, 20)
    assert buf.evicted
    assert buf.count == 8
    assert (buf.start, buf.frontier) == (12, 20)
    assert not buf.covers(11, 20)
    assert buf.covers(12, 20)


def test_window_is_correct_across_the_ring_seam():
    """The seam case: a span that wraps must come back in time order, not in ring
    order. Getting this wrong reorders the wavelet's input silently."""
    buf = StreamBuffer(("change",), 1, 1, capacity=5)
    _fill(buf, 0, 8)                        # ring now holds 3..7, wrapped
    got = buf.window(3, 8, "change")[:, 0, 0]
    assert list(got) == [3.0, 4.0, 5.0, 6.0, 7.0]


def test_window_returns_a_copy_not_a_view():
    """Aliasing must not depend on whether the span happened to wrap."""
    buf = StreamBuffer(("change",), 1, 1, capacity=5)
    _fill(buf, 0, 3)
    got = buf.window(0, 3, "change")
    got[0, 0, 0] = -999.0
    assert buf.window(0, 3, "change")[0, 0, 0] == pytest.approx(0.0)


def test_window_refuses_an_unretained_span():
    buf = StreamBuffer(("change",), 1, 1, capacity=5)
    _fill(buf, 0, 8)
    with pytest.raises(ValueError, match="not retained"):
        buf.window(0, 4, "change")          # evicted
    with pytest.raises(ValueError, match="not retained"):
        buf.window(6, 12, "change")         # past the frontier


def test_latest_bounds_the_wavelet_window_and_reports_absolute_position():
    buf = StreamBuffer(("change",), 1, 1, capacity=100, start=50)
    _fill(buf, 50, 90)
    arr, first = buf.latest(10, "change")
    assert first == 80 and arr.shape[0] == 10
    assert arr[0, 0, 0] == pytest.approx(80.0)
    # Asking for more than exists yields what exists, not padding.
    arr, first = buf.latest(999, "change")
    assert first == 50 and arr.shape[0] == 40


def test_latest_on_an_empty_buffer_is_empty_not_a_crash():
    buf = StreamBuffer(("change",), 2, 3, capacity=10)
    arr, first = buf.latest(20, "change")
    assert arr.shape == (0, 2, 3) and first == 0


def test_covers_rejects_empty_and_inverted_ranges():
    buf = StreamBuffer(("change",), 1, 1, capacity=10)
    _fill(buf, 0, 5)
    assert not buf.covers(2, 2)
    assert not buf.covers(4, 1)
    assert buf.covers(0, 5)


def test_multi_channel_frames_stay_aligned_through_a_wrap():
    chans = ("change", "intensity")
    buf = StreamBuffer(chans, 1, 1, capacity=4)
    for i in range(10):
        buf.append(i, {"change": np.full((1, 1), i, np.float32),
                       "intensity": np.full((1, 1), i * 100, np.float32)})
    c = buf.window(6, 10, "change")[:, 0, 0]
    v = buf.window(6, 10, "intensity")[:, 0, 0]
    assert list(c) == [6.0, 7.0, 8.0, 9.0]
    assert list(v) == [600.0, 700.0, 800.0, 900.0]


def test_failed_append_leaves_the_retained_span_untouched():
    """The corruption case. On a FULL ring, index % capacity is the oldest
    retained frame's slot, so an append that validates as it writes would clobber
    one channel of a live frame before failing on the next -- leaving that frame
    holding data from two different points in the clip while the span still
    claims to be intact."""
    chans = ("change", "intensity")
    buf = StreamBuffer(chans, 1, 1, capacity=4)
    for i in range(4):
        buf.append(i, {"change": np.full((1, 1), i, np.float32),
                       "intensity": np.full((1, 1), i * 100, np.float32)})
    before_c = buf.window(0, 4, "change")[:, 0, 0].copy()
    before_i = buf.window(0, 4, "intensity")[:, 0, 0].copy()
    with pytest.raises(KeyError):
        buf.append(4, {"change": np.full((1, 1), 999.0, np.float32)})
    assert (buf.start, buf.count) == (0, 4)
    assert list(buf.window(0, 4, "change")[:, 0, 0]) == list(before_c)
    assert list(buf.window(0, 4, "intensity")[:, 0, 0]) == list(before_i)


def test_missing_or_misshaped_channel_is_rejected_at_append():
    buf = StreamBuffer(("change", "intensity"), 2, 3, capacity=4)
    with pytest.raises(KeyError):
        buf.append(0, {"change": np.zeros((2, 3), np.float32)})
    with pytest.raises(ValueError, match="shape"):
        buf.append(0, {"change": np.zeros((5, 5), np.float32),
                       "intensity": np.zeros((2, 3), np.float32)})


def test_as_channel_dict_matches_channeldata_shape():
    buf = StreamBuffer(("change", "intensity"), 2, 3, capacity=10)
    _fill(buf, 0, 6, channels=("change", "intensity"))
    d = buf.as_channel_dict(2, 5)
    assert set(d) == {"change", "intensity"}
    assert all(v.shape == (3, 2, 3) for v in d.values())


def test_capacity_for_budget_scales_and_has_a_floor():
    # 2 channels, 50x50 blocks, float32 -> 20000 B/frame
    assert capacity_for_budget(20_000 * 500, 2, 50, 50) == 500
    assert capacity_for_budget(1, 2, 50, 50) == MIN_CAPACITY


def test_rejects_degenerate_construction():
    with pytest.raises(ValueError):
        StreamBuffer(("change",), 2, 3, capacity=0)
    with pytest.raises(ValueError):
        StreamBuffer((), 2, 3, capacity=10)
