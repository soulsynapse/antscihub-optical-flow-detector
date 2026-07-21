"""Tests for the downsample/block-size cost model.

The load-bearing claims are (a) the knee sits where per-pixel work equals the
decode floor, (b) a two-scale fit recovers the coefficients without needing span
names, and (c) storage is invariant under scale when the block tracks it -- that
last one is the measured basis for presenting downsample and block_size as two
separate levers rather than one "quality" slider.
"""
from __future__ import annotations


import pytest

from core.config import FlowConfig
from core.replicates import build_layout
from core.cost_model import (CostModel, PassSample, atlas_cells, boxes_from_tiles,
                             format_bytes, format_duration, grid_cells,
                             mixing_error, storage_bytes_per_hour,
                             working_px_per_body_length)


def _boxes(n=7, w=297, h=1200):
    return [(w, h)] * n


# -- the model ---------------------------------------------------------------
def test_seconds_per_frame_is_floor_plus_quadratic():
    m = CostModel(fixed_s=0.008, per_pixel_s=0.032, provisional=False)
    assert m.seconds_per_frame(1.0) == pytest.approx(0.040)
    assert m.seconds_per_frame(0.5) == pytest.approx(0.008 + 0.032 * 0.25)
    assert m.seconds_per_frame(0.0) == pytest.approx(0.008)


def test_knee_is_where_per_pixel_work_equals_the_decode_floor():
    m = CostModel(fixed_s=0.01, per_pixel_s=0.04, provisional=False)
    knee = m.knee_scale()
    assert knee == pytest.approx(0.5)
    # ...which is exactly the point of unit elasticity: 1% less scale buys 1%
    # less time. That equivalence is the whole justification for the marker.
    assert m.elasticity(knee) == pytest.approx(1.0)
    assert m.elasticity(0.9) > 1.0        # above: resolution is cheap
    assert m.elasticity(0.2) < 1.0        # below: paying pixels for nothing


def test_knee_matches_the_measured_5k_clip():
    """Sanity-check against todo.md's measured numbers, not a made-up curve.

    479 frames of GX010047c2: ~3.8 s decode floor, ~15.6 s of math at scale 1.0.
    The approved mockup put the knee near 0.50 by eye; the closed form agrees.
    """
    m = CostModel(fixed_s=3.8 / 479, per_pixel_s=15.6 / 479, provisional=False)
    assert m.knee_scale() == pytest.approx(0.494, abs=0.01)


def test_provisional_model_refuses_to_report_a_knee():
    """Measured on GX010047c2: prefetch hides decode so completely that a
    one-pass split reads a 0.04 ms/frame floor where the fit finds 11.79 --
    a knee of 0.05 against a true 0.70. Wrong in the direction that makes
    aggressive downsampling look free, so the model must decline to answer."""
    one = CostModel.from_spans({"decode": 0.01}, frames=120, scale=1.0, wall=4.30)
    assert one.provisional
    assert one.knee_scale() is None
    # The coefficients are still exposed -- it is the *actionable* readout that
    # is withheld, so a caller cannot accidentally present it as a frontier.
    assert one.per_pixel_s > 0


def test_knee_is_none_when_decode_already_dominates_at_full_resolution():
    # Floor above the per-pixel cost puts s* >= 1: every scale is on the flat
    # part, so there is no interior knee to mark and the honest readout is that
    # the lever buys nothing here.
    assert CostModel(fixed_s=0.05, per_pixel_s=0.01, provisional=False).knee_scale() is None
    assert CostModel(fixed_s=0.0, per_pixel_s=0.04, provisional=False).knee_scale() is None


def test_knee_clamps_to_min_scale():
    m = CostModel(fixed_s=1e-9, per_pixel_s=1.0, provisional=False)
    assert m.knee_scale(min_scale=0.05) == pytest.approx(0.05)


def test_fit_reproduces_measured_wall_times_on_real_footage():
    """Regression against real measurements, which is what justifies the
    quadratic form at all. GX010047c2, 7 replicates, 120 frames, block tracking
    the scale; wall times from an actual driven run."""
    measured = {1.0: 4.30, 0.5: 2.17, 0.25: 1.56}
    m = CostModel.fit([PassSample(scale=s, frames=120, wall=w)
                       for s, w in measured.items()])
    for s, w in measured.items():
        assert m.seconds_per_frame(s) * 120 == pytest.approx(w, rel=0.03)
    assert m.knee_scale() == pytest.approx(0.70, abs=0.02)


def test_fit_recovers_coefficients_from_two_scales():
    truth = CostModel(fixed_s=0.008, per_pixel_s=0.032, provisional=False)
    samples = [PassSample(scale=s, frames=479, wall=truth.seconds_per_frame(s) * 479)
               for s in (1.0, 0.5, 0.25)]
    fitted = CostModel.fit(samples)
    assert not fitted.provisional
    assert fitted.n_samples == 3
    assert fitted.fixed_s == pytest.approx(0.008, abs=1e-6)
    assert fitted.per_pixel_s == pytest.approx(0.032, abs=1e-6)
    assert fitted.knee_scale() == pytest.approx(truth.knee_scale())


def test_fit_falls_back_to_span_split_without_two_distinct_scales():
    s = PassSample(scale=1.0, frames=100, wall=4.0,
                   spans={"decode": 1.0, "flow_solve": 2.0})
    one = CostModel.fit([s])
    assert one.provisional
    assert one.fixed_s == pytest.approx(0.01)
    # Duplicated scale is still underdetermined, not a two-point fit.
    assert CostModel.fit([s, s]).provisional


def test_fit_never_returns_negative_coefficients():
    # Noise-dominated samples over a narrow scale range can regress to a
    # negative floor, which would make the knee a NaN.
    noisy = [PassSample(scale=0.9, frames=10, wall=1.0),
             PassSample(scale=1.0, frames=10, wall=0.2)]
    m = CostModel.fit(noisy)
    assert m.fixed_s >= 0 and m.per_pixel_s >= 0
    assert m.knee_scale() is None or 0 < m.knee_scale() <= 1.0


# -- what may not share a fit (Batch S slice 5) ------------------------------
def test_a_fit_refuses_to_mix_source_and_clip_passes():
    """fixed_s IS the decode floor, and cutting ROI clips moves it ~25x. A fit
    over both reads that step as scale, which moves the knee toward
    'downsampling is free' -- the one direction this model must never err in."""
    before = PassSample(scale=1.0, frames=100, wall=4.0)
    after = PassSample(scale=0.5, frames=100, wall=1.0, source_kind="clips:abc")
    with pytest.raises(ValueError, match="different sources"):
        CostModel.fit([before, after])


def test_a_fit_refuses_to_mix_channel_sets():
    """A change-only pass measured 1.59x faster than the full four."""
    one = PassSample(scale=1.0, frames=100, wall=4.0, channels=("change",))
    four = PassSample(scale=0.5, frames=100, wall=3.0,
                      channels=("appearance", "change"))
    with pytest.raises(ValueError, match="channel sets"):
        CostModel.fit([one, four])


def test_two_clip_cuts_at_different_quality_do_not_share_a_fit():
    """The token is the manifest's provenance_key, not a bare 'clips': below
    lossless they are different pixels and a different floor."""
    assert mixing_error([PassSample(scale=1.0, frames=1, wall=1.0,
                                    source_kind="clips:aaa"),
                         PassSample(scale=0.5, frames=1, wall=1.0,
                                    source_kind="clips:bbb")])


def test_an_unrecorded_channel_set_mixes_with_anything():
    """Empty means UNRECORDED, not 'no channels'. Refusing on absence would
    reject every sample built before the field existed, turning a provenance
    improvement into a regression for archived and hand-built samples."""
    assert mixing_error([PassSample(scale=1.0, frames=100, wall=4.0),
                         PassSample(scale=0.5, frames=100, wall=2.0,
                                    channels=("change",))]) is None


def test_samples_agreeing_on_both_axes_still_fit():
    m = CostModel.fit([
        PassSample(scale=1.0, frames=100, wall=4.0, source_kind="clips:aaa",
                   channels=("change",)),
        PassSample(scale=0.5, frames=100, wall=1.75, source_kind="clips:aaa",
                   channels=("change",))])
    assert not m.provisional
    assert m.n_samples == 2


def test_the_default_source_kind_is_a_fact_not_a_guess():
    """Every pass timed before routing existed decoded the source, so the
    default lets archived samples keep fitting together."""
    assert PassSample(scale=1.0, frames=1, wall=1.0).source_kind == "source"


def test_from_spans_prices_untimed_remainder_as_per_pixel():
    # wall 4.0 with only 3.0 of spans: the missing second must be priced, and
    # attributing it to the per-pixel half raises the knee, i.e. errs toward
    # keeping resolution.
    m = CostModel.from_spans({"decode": 1.0, "flow_solve": 2.0},
                             frames=100, scale=1.0, wall=4.0)
    assert m.fixed_s == pytest.approx(0.01)
    assert m.per_pixel_s == pytest.approx(0.03)
    assert m.provisional


def test_from_spans_is_scale_corrected():
    """A pass measured at 0.5 must extrapolate to 4x the per-pixel cost at 1.0."""
    m = CostModel.from_spans({"decode": 1.0}, frames=100, scale=0.5, wall=3.0)
    assert m.per_pixel_s == pytest.approx((2.0 / 100) / 0.25)
    assert m.seconds_per_frame(0.5) == pytest.approx(0.03)


def test_degenerate_inputs_do_not_raise():
    assert CostModel.from_spans({}, frames=0, scale=1.0, wall=0.0).seconds_per_frame(1.0) == 0.0
    assert CostModel.fit([]).provisional


# -- projection --------------------------------------------------------------
def test_corpus_projection_and_realtime_factor():
    m = CostModel(fixed_s=0.0, per_pixel_s=1.0 / 24, provisional=False)
    # One second of work per 24 frames == exactly realtime at 24 fps.
    assert m.realtime_factor(1.0, fps=24.0) == pytest.approx(1.0)
    assert m.hours_for_corpus(1.0, corpus_hours=3000, fps=24.0) == pytest.approx(3000)
    assert m.hours_for_corpus(1.0, 3000, 24.0, workers=10) == pytest.approx(300)


# -- storage -----------------------------------------------------------------
_REPS = [{"id": 0, "frac": (0.02, 0.05, 0.15, 0.95)},
         {"id": 1, "frac": (0.20, 0.05, 0.33, 0.95)},
         {"id": 2, "frac": (0.38, 0.05, 0.51, 0.95)}]


def test_storage_is_scale_invariant_when_block_tracks_scale():
    """The measured basis for the two-lever split: moving scale must not move
    storage, or the compute knob would be silently charging the user disk."""
    flow = FlowConfig()          # block_size None == tracks the scale
    sizes = {storage_bytes_per_hour(
        atlas_cells(_REPS, 5312, 2988, s, flow.resolve_block_size(s)),
        fps=24.0, n_channels=5) for s in (1.0, 0.5, 0.25)}
    lo, hi = min(sizes), max(sizes)
    assert hi / lo < 1.15        # rounding of the block grid only


def test_block_size_is_the_storage_lever():
    fine = storage_bytes_per_hour(atlas_cells(_REPS, 5312, 2988, 1.0, 16),
                                  fps=24.0, n_channels=5)
    coarse = storage_bytes_per_hour(atlas_cells(_REPS, 5312, 2988, 1.0, 64),
                                    fps=24.0, n_channels=5)
    assert fine / coarse > 10        # todo.md Batch K measured ~13x


def test_atlas_cells_counts_the_padding_that_is_actually_allocated():
    """Channels are stored over the packed atlas, so the per-tile box sum
    under-counts. Using the boxes for a storage figure was a ~17% under-report
    on the 7-replicate clip -- this pins the two apart deliberately."""
    layout = build_layout(_REPS, 5312, 2988, 1.0, 64)
    cells = atlas_cells(_REPS, 5312, 2988, 1.0, 64)
    assert cells == layout.atlas_grid[0] * layout.atlas_grid[1]
    assert cells > grid_cells(boxes_from_tiles(layout.tiles), 1.0, 64)


def test_atlas_cells_falls_back_to_the_whole_frame_without_replicates():
    assert atlas_cells([], 128, 256, 1.0, 64) == 2 * 4


def test_grid_cells_counts_partial_blocks():
    # 100px at block 64 is two cells, not one: reduction includes the partial.
    assert grid_cells([(100, 100)], 1.0, 64) == 4


def test_grid_cells_never_collapses_a_tile_to_zero():
    assert grid_cells([(10, 10)], 0.01, 64) == 1


def test_boxes_from_tiles_accepts_dicts_and_dataclasses():
    dicts = [{"id": 0, "source_box": (10, 20, 110, 220)}]
    assert boxes_from_tiles(dicts) == [(100, 200)]
    layout = build_layout([{"id": 0, "frac": (0.0, 0.0, 0.5, 1.0)}],
                          src_width=800, src_height=600, scale=1.0, block_size=64)
    assert boxes_from_tiles(layout.tiles) == [(400, 600)]


def test_grid_cells_matches_build_layout_geometry():
    """The projection must count the same cells extraction actually writes."""
    reps = [{"id": 0, "frac": (0.0, 0.0, 0.37, 0.61)},
            {"id": 1, "frac": (0.4, 0.1, 0.9, 0.8)}]
    for scale in (1.0, 0.5, 0.25):
        block = FlowConfig().resolve_block_size(scale)
        layout = build_layout(reps, 5312, 2988, scale, block)
        expected = sum(t.grid[0] * t.grid[1] for t in layout.tiles)
        assert grid_cells(boxes_from_tiles(layout.tiles), scale, block) == expected


# -- calibration + formatting ------------------------------------------------
def test_working_px_per_body_length_follows_the_roi_convention():
    # core/roi.py:418 -- calibration is source px/mm, scaled through to working.
    assert working_px_per_body_length(2.0, 8.0, 0.5) == pytest.approx(8.0)
    assert working_px_per_body_length(None, 8.0, 1.0) is None
    assert working_px_per_body_length(2.0, None, 1.0) is None
    assert working_px_per_body_length(0.0, 8.0, 1.0) is None


def test_formatters():
    assert format_bytes(512) == "512 B"
    assert format_bytes(1536) == "1.5 KB"
    assert format_bytes(2 * 1024 ** 4) == "2.0 TB"
    assert format_duration(0.5) == "30 min"
    assert format_duration(12) == "12.0 h"
    assert format_duration(24 * 125) == "125.0 d"
