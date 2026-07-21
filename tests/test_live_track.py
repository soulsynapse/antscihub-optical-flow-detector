"""Coverage, staleness and COI-honest writes on the whole-video detection track.

The properties under test are the ones the strip's honesty rests on: an
unexamined frame must never be readable as a quiet one, a window's
cone-of-influence edges must not be written at all, and a settings change must
leave earlier work visible-but-stale rather than blanked or silently reused.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.live_track import (TrackStamp, WholeVideoTrack,
                             band_power_bytes, coi_trim)


def _stamp(channel="change", band=(0.5, 4.0), blocks=6, block_size=64):
    return TrackStamp(channel=channel, freq_band_hz=band, grid=(4, 4),
                      region_index=0, region_blocks=blocks, downsample=1.0,
                      block_size=block_size)


def _track(T=400, blocks=6, fps=24.0):
    tr = WholeVideoTrack(n_frames=T, fps=fps, n_blocks=blocks)
    tr.set_stamp(_stamp(blocks=blocks))
    tr.set_detector(value_band=(1.0, np.inf), count_band=(3.0, np.inf),
                    detect_window=1, centered=True)
    return tr


# -- coverage is not zero ----------------------------------------------------

def test_new_track_is_entirely_uncovered_not_quiet():
    tr = _track()
    assert not tr.covered.any()
    assert tr.coverage_fraction() == 0.0
    # The series read as zeros, which is exactly why nothing may consult them
    # without the mask -- the point of the mask existing.
    assert tr.gate.sum() == 0.0


def test_written_quiet_frames_are_covered_and_distinguishable_from_unwritten():
    tr = _track(T=200)
    bp = np.zeros((100, 6), np.float32)      # genuinely quiet: no block in band
    a, b = tr.write(0, bp, trim=0)
    assert (a, b) == (0, 100)
    assert tr.covered[:100].all()
    assert not tr.covered[100:].any()
    assert tr.gate[:100].sum() == 0.0        # quiet, and known to be quiet


# -- the cone of influence is not written ------------------------------------

def test_coi_trim_matches_torrence_compo_efolding():
    # tau = 1.369/f seconds at w0=6; the 1.46/f in the plan was ~7% too large.
    assert coi_trim((0.5, 4.0), 24.0) == pytest.approx(np.ceil(1.369 / 0.5 * 24),
                                                       abs=1)
    # Widest cone in the band sets the trim: the low edge, not the high one.
    assert coi_trim((0.5, 20.0), 24.0) == coi_trim((0.5, 4.0), 24.0)
    assert coi_trim((5.0, 20.0), 24.0) < coi_trim((0.5, 20.0), 24.0)


def test_write_discards_both_edges_of_a_window():
    tr = _track(T=1000)
    bp = np.ones((300, 6), np.float32)
    a, b = tr.write(400, bp, trim=33)
    assert (a, b) == (433, 667)
    assert not tr.covered[:433].any()
    assert tr.covered[433:667].all()
    assert not tr.covered[667:].any()


def test_a_window_shorter_than_its_own_cone_writes_nothing():
    tr = _track(T=1000)
    a, b = tr.write(400, np.ones((40, 6), np.float32), trim=33)
    assert b <= a
    assert not tr.covered.any()


def test_clip_edges_keep_their_frames_via_trim_overrides():
    # The record really does start at frame 0, so those frames are at the edge
    # of the data rather than at an arbitrary cut; discarding them would leave
    # the clip's opening permanently unexamined.
    tr = _track(T=1000)
    a, b = tr.write(0, np.ones((300, 6), np.float32), trim=33, trim_head=0)
    assert a == 0
    assert tr.covered[0]


def test_a_later_overlapping_window_supplies_a_previous_windows_cone():
    tr = _track(T=1000)
    tr.write(0, np.ones((300, 6), np.float32), trim=33, trim_head=0)
    covered_after_first = int(tr.covered.sum())
    # The next window overlaps, so frames that were edge-contaminated before are
    # interior now. This is the T35 fix: the contaminated values are never
    # written, and coverage is filled by a window that could see them.
    tr.write(200, np.ones((300, 6), np.float32), trim=33)
    assert int(tr.covered.sum()) > covered_after_first
    assert tr.covered[233:467].all()


# -- staleness ---------------------------------------------------------------

def test_changing_the_channel_leaves_earlier_frames_covered_but_stale():
    tr = _track(T=400)
    tr.write(0, np.full((200, 6), 5.0, np.float32), trim=0)
    assert tr.current[:200].all()
    changed = tr.set_stamp(_stamp(channel="intensity"))
    assert changed
    assert tr.covered[:200].all()            # still examined
    assert tr.stale[:200].all()              # but not under these settings
    assert not tr.current[:200].any()
    assert tr.coverage_fraction() == 0.0     # nothing current yet


def test_a_no_op_stamp_change_does_not_invalidate():
    tr = _track(T=400)
    tr.write(0, np.full((200, 6), 5.0, np.float32), trim=0)
    assert not tr.set_stamp(_stamp())        # same settings, same object value
    assert tr.current[:200].all()


def test_returning_to_a_previous_stamp_makes_its_frames_current_again():
    tr = _track(T=400)
    tr.write(0, np.full((100, 6), 5.0, np.float32), trim=0)
    tr.set_stamp(_stamp(channel="intensity"))
    tr.write(200, np.full((100, 6), 5.0, np.float32), trim=0)
    tr.set_stamp(_stamp())                   # back to the original
    assert tr.current[:100].all()
    assert tr.stale[200:300].all()


def test_threshold_changes_do_not_gray_anything_out():
    # The whole reason value/count band and detect window are absent from the
    # stamp: a count-band nudge must re-derive over what is covered, not force a
    # re-stream of it.
    tr = _track(T=400)
    tr.write(0, np.full((200, 6), 5.0, np.float32), trim=0)
    tr.set_detector(value_band=(1.0, np.inf), count_band=(1.0, np.inf),
                    detect_window=5, centered=True)
    assert tr.current[:200].all()
    assert tr.gate[:200].sum() > 0


def test_count_band_retunes_the_gate_over_retained_coverage():
    tr = _track(T=400)
    tr.write(0, np.full((200, 6), 5.0, np.float32), trim=0)   # 6 blocks in band
    tr.set_detector(value_band=(1.0, np.inf), count_band=(7.0, np.inf),
                    detect_window=1, centered=True)
    assert tr.gate[:200].sum() == 0          # 6 < 7, nothing fires
    tr.set_detector(value_band=(1.0, np.inf), count_band=(5.0, np.inf),
                    detect_window=1, centered=True)
    assert tr.gate[:200].sum() == 200        # re-derived, no refill needed


def test_value_band_rederives_counts_from_retained_band_power():
    tr = _track(T=400)
    tr.write(0, np.full((200, 6), 5.0, np.float32), trim=0)
    tr.set_detector(value_band=(10.0, np.inf), count_band=(1.0, np.inf),
                    detect_window=1, centered=True)
    assert tr.count[:200].sum() == 0         # 5.0 is no longer in band
    tr.set_detector(value_band=(1.0, np.inf), count_band=(1.0, np.inf),
                    detect_window=1, centered=True)
    assert tr.count[:200].sum() == 200 * 6


def test_declining_band_power_leaves_thresholds_needing_a_refill():
    tr = WholeVideoTrack(n_frames=400, fps=24.0, n_blocks=6,
                         retain_band_power=False)
    tr.set_stamp(_stamp())
    tr.set_detector(value_band=(1.0, np.inf), count_band=(1.0, np.inf),
                    detect_window=1, centered=True)
    tr.write(0, np.full((200, 6), 5.0, np.float32), trim=0)
    assert tr.count[:200].sum() == 200 * 6
    # No rows retained, so a value-band change cannot re-derive; the frames keep
    # what they were computed with rather than being silently recomputed wrong.
    tr.set_detector(value_band=(10.0, np.inf), count_band=(1.0, np.inf),
                    detect_window=1, centered=True)
    assert tr.count[:200].sum() == 200 * 6


# -- gaps must not be averaged across ----------------------------------------

def test_windowed_gate_does_not_smear_across_an_uncovered_gap():
    tr = _track(T=600)
    tr.set_detector(value_band=(1.0, np.inf), count_band=(5.0, np.inf),
                    detect_window=41, centered=True)
    # Two islands of strong signal with an unexamined gap between them. If the
    # windowed mean ran across the whole array, the zeros standing in for the
    # gap would drag the gate down at both island edges -- a dip caused by
    # nobody having looked, presented as data.
    tr.write(0, np.full((200, 6), 5.0, np.float32), trim=0)
    tr.write(400, np.full((200, 6), 5.0, np.float32), trim=0)
    assert tr.gate[:200].all()
    assert tr.gate[400:600].all()
    assert tr.gate[200:400].sum() == 0.0     # uncovered, so not gated


def test_a_stale_regions_gate_is_frozen_not_regated():
    # A stale frame's `count` was computed under the value band in force when it
    # was written. Re-gating it against a count band chosen afterwards would
    # combine two settings that never coexisted and present the result as that
    # stretch's detection -- and would make the gray bars visibly move while you
    # tune knobs that provably do not apply to them.
    tr = _track(T=400)
    tr.write(0, np.full((200, 6), 5.0, np.float32), trim=0)
    assert tr.gate[:200].sum() == 200
    tr.set_stamp(_stamp(channel="intensity"))          # everything goes stale
    frozen = tr.gate[:200].copy()
    tr.set_detector(value_band=(1.0, np.inf), count_band=(99.0, np.inf),
                    detect_window=1, centered=True)
    np.testing.assert_array_equal(tr.gate[:200], frozen)


def test_a_current_regions_gate_still_retunes():
    tr = _track(T=400)
    tr.write(0, np.full((200, 6), 5.0, np.float32), trim=0)
    tr.set_detector(value_band=(1.0, np.inf), count_band=(99.0, np.inf),
                    detect_window=1, centered=True)
    assert tr.gate[:200].sum() == 0


def test_intervals_and_gaps_are_complementary():
    tr = _track(T=500)
    tr.write(100, np.full((200, 6), 5.0, np.float32), trim=0)
    assert tr.gaps() == [(0, 100), (300, 500)]
    assert tr.detected_intervals() == [(100, 300)]


def test_stale_detections_are_excluded_from_navigation():
    tr = _track(T=400)
    tr.write(0, np.full((200, 6), 5.0, np.float32), trim=0)
    assert tr.detected_intervals() == [(0, 200)]
    tr.set_stamp(_stamp(channel="intensity"))
    # Still on the strip (it happened), but not something to step into as an
    # answer to the question now being asked.
    assert tr.detected_intervals(current_only=True) == []
    assert tr.detected_intervals(current_only=False) == [(0, 200)]


# -- guards ------------------------------------------------------------------

def test_write_without_a_stamp_is_refused():
    tr = WholeVideoTrack(n_frames=100, fps=24.0, n_blocks=6)
    with pytest.raises(RuntimeError, match="set_stamp"):
        tr.write(0, np.zeros((10, 6), np.float32))


def test_a_grid_change_without_a_new_stamp_is_refused():
    tr = _track()
    with pytest.raises(ValueError, match="grid moved"):
        tr.write(0, np.zeros((10, 99), np.float32))


def test_band_power_bytes_prices_the_retention_choice():
    assert band_power_bytes(30_000, 377) == 30_000 * 377 * 4      # ~45 MB
    assert band_power_bytes(0, 377) == 0


def test_writes_past_the_end_of_the_clip_are_clipped_not_rejected():
    tr = _track(T=100)
    a, b = tr.write(50, np.ones((100, 6), np.float32), trim=0)
    assert (a, b) == (50, 100)
    assert tr.covered[50:100].all()
