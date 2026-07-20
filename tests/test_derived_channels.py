"""Contracts for the derived per-block channels.

The tests that matter here are the breakdown cases, not the happy path: the
background percentile is only useful while the animal is in the minority, and
the intensive normalisation is only safe while it refuses to answer for blocks
holding no animal.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.derived_channels import (background_level, intensive, occupancy,
                                   ratio)


def _scene(T=100, ny=4, nx=5, bg=100.0, animal=40.0, occupied=None):
    """A still substrate at ``bg`` with a darker animal parked in block (1, 2)
    for the frames in ``occupied``."""
    arr = np.full((T, ny, nx), bg, np.float32)
    for t in (range(20) if occupied is None else occupied):
        arr[t, 1, 2] = animal
    return arr


def test_background_recovers_substrate_when_animal_is_a_minority():
    arr = _scene(T=100, occupied=range(20))
    bg = background_level(arr, "darker")
    assert bg.shape == (4, 5)
    assert np.allclose(bg, 100.0)


def test_background_breaks_down_when_animal_dominates_the_window():
    """Past the polarity's breakdown point the 'background' becomes the animal.
    This is the failure a too-short window produces, so it must be a known
    quantity -- and the two polarities fail DIFFERENTLY, which is the part worth
    pinning: signed annihilates, unsigned inverts."""
    arr = _scene(T=100, occupied=range(80))
    bg = background_level(arr, "darker")
    assert bg[1, 2] == pytest.approx(40.0)          # the animal, not substrate

    # Signed: the animal now sits AT the background (deviation 0) and the
    # substrate deviates on the side rectification discards. The block reports
    # no animal in any frame -- a silent, total loss of the channel.
    occ = occupancy(arr, "darker", background=bg)
    assert occ[:, 1, 2].max() == pytest.approx(0.0)

    # Unsigned: the substrate frames survive as a deviation, so the channel
    # inverts rather than vanishing -- wrong, but visibly wrong.
    flipped = occupancy(arr, "abs", background=bg)
    assert flipped[0, 1, 2] == pytest.approx(0.0)
    assert flipped[90, 1, 2] == pytest.approx(60.0)


def test_occupancy_is_extensive_in_coverage():
    """Half-covered block reads half the occupancy of a fully covered one --
    the property that makes it the right denominator for `intensive`."""
    arr = np.full((50, 2, 2), 100.0, np.float32)
    arr[:10, 0, 0] = 40.0                            # full coverage: delta 60
    arr[:10, 0, 1] = 70.0                            # half coverage: delta 30
    occ = occupancy(arr, "darker")
    assert occ[0, 0, 0] == pytest.approx(2.0 * occ[0, 0, 1])


def test_signed_polarity_rejects_deviation_on_the_wrong_side():
    """A brightening on 'darker' footage is a shadow or a lighting drift, not an
    animal, and must not register as occupancy."""
    arr = np.full((50, 1, 2), 100.0, np.float32)
    arr[:10, 0, 0] = 40.0                            # darker: the animal
    arr[:10, 0, 1] = 160.0                           # brighter: not the animal
    occ = occupancy(arr, "darker")
    assert occ[0, 0, 0] > 0
    assert occ[0, 0, 1] == pytest.approx(0.0)
    # "abs" cannot tell them apart, which is the cost of not knowing polarity.
    both = occupancy(arr, "abs")
    assert both[0, 0, 0] > 0 and both[0, 0, 1] > 0


def test_supplied_background_lets_a_short_window_borrow_a_long_one():
    long_arr = _scene(T=200, occupied=range(40))
    bg = background_level(long_arr, "darker")
    # A short window in which the animal never leaves: its own percentile would
    # call the animal background, the borrowed level does not.
    short = long_arr[:30]
    assert background_level(short, "darker")[1, 2] == pytest.approx(40.0)
    occ = occupancy(short, "darker", background=bg)
    assert occ[0, 1, 2] == pytest.approx(60.0)


def test_background_and_percentile_together_are_rejected():
    """Silently ignoring the percentile is worse than refusing: borrowing a
    background is exactly when a caller is likely to also be tuning one."""
    arr = _scene(T=50)
    bg = background_level(arr, "darker")
    with pytest.raises(ValueError, match="not both"):
        occupancy(arr, "darker", percentile=90.0, background=bg)


def test_fully_masked_block_is_nan_and_warns_nothing():
    """A block masked for the whole window has no evidence to form a background
    from, so NaN is the honest answer -- but it must not spray RuntimeWarnings
    through a pass that runs once a second."""
    arr = _scene(T=60)
    arr[:, 0, 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # any warning fails the test
        bg = background_level(arr, "darker")
        occ = occupancy(arr, "darker", background=bg)
    assert np.isnan(bg[0, 0])
    assert np.isnan(occ[:, 0, 0]).all()


def test_nan_blocks_do_not_void_the_background():
    """A masked block reduces to NaN for the frames it was masked; that must not
    propagate into a background that then voids the block for all frames."""
    arr = _scene(T=100, occupied=range(20))
    arr[5:15, 0, 0] = np.nan
    bg = background_level(arr, "darker")
    assert np.isfinite(bg).all()
    assert bg[0, 0] == pytest.approx(100.0)


def test_intensive_returns_nan_where_there_is_no_animal_to_normalise_by():
    """Zero would read as a confident 'no activity per unit animal'; NaN forces
    the consumer to decide. This is the silent-false-negative contract."""
    occ = np.zeros((3, 1, 2), np.float32)
    occ[:, 0, 0] = 50.0                              # animal present
    energy = np.full((3, 1, 2), 10.0, np.float32)
    out = intensive(energy, occ)
    assert out[0, 0, 0] == pytest.approx(0.2)
    assert np.isnan(out[0, 0, 1])


def test_intensive_separates_vigour_from_coverage():
    """The whole point: twice the animal with twice the energy is the SAME
    intensity, and that is what the raw extensive channel cannot say."""
    occ = np.array([[[10.0, 20.0]]], np.float32)
    energy = np.array([[[5.0, 10.0]]], np.float32)
    out = intensive(energy, occ)
    assert out[0, 0, 0] == pytest.approx(out[0, 0, 1])


def test_intensive_floor_does_not_move_with_the_window():
    """The regression that matters for live streaming: an earlier version took
    the floor from a quantile of the array passed in, so the same block changed
    value -- and flipped between finite and NaN -- as the ring filled. The floor
    is absolute; a short window and a long one must agree on the overlap."""
    rng = np.random.default_rng(0)
    occ = rng.uniform(0.5, 40.0, (400, 3, 3)).astype(np.float32)
    energy = rng.uniform(1.0, 10.0, (400, 3, 3)).astype(np.float32)
    short = intensive(energy[:40], occ[:40])
    long = intensive(energy, occ)
    assert np.allclose(short, long[:40], equal_nan=True)


def test_ratio_refuses_a_zero_denominator():
    change = np.array([[[10.0, 0.0]]], np.float32)
    appearance = np.array([[[4.0, 0.0]]], np.float32)
    out = ratio(appearance, change)
    assert out[0, 0, 0] == pytest.approx(0.4)
    assert np.isnan(out[0, 0, 1])


def test_ratio_is_unbounded_unless_the_caller_declares_a_bound():
    """`tensor_speed / sqrt(change)` and `strain / speed` exceed 1 routinely.
    Clipping by default would collapse every block above unity onto the same
    value and destroy the dynamic range the channel exists for."""
    out = ratio(np.array([[[7.0]]], np.float32), np.array([[[2.0]]], np.float32))
    assert out[0, 0, 0] == pytest.approx(3.5)
    # `appearance / change` IS bounded, and says so explicitly to absorb float
    # error at tiny magnitudes.
    bounded = ratio(np.array([[[1.0 + 1e-7]]], np.float32),
                    np.array([[[1.0]]], np.float32), clip=(0.0, 1.0))
    assert bounded[0, 0, 0] <= 1.0


def test_empty_window_gives_an_empty_background_not_a_crash():
    assert background_level(np.zeros((0, 3, 3), np.float32)).shape == (3, 3)


def test_shape_mismatches_are_rejected():
    with pytest.raises(ValueError):
        occupancy(np.zeros((5, 2, 2), np.float32),
                  background=np.zeros((3, 3), np.float32))
    with pytest.raises(ValueError):
        intensive(np.zeros((5, 2, 2), np.float32), np.zeros((5, 3, 3), np.float32))
    with pytest.raises(ValueError):
        background_level(np.zeros((5, 5), np.float32))
    with pytest.raises(ValueError):
        occupancy(np.zeros((5, 2, 2), np.float32), polarity="sideways")
