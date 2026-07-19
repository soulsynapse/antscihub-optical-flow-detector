"""Calibration arithmetic, and the invariant the whole sub-tool rests on."""
from __future__ import annotations

import math

import pytest

from core.calibration import (Calibration, line_error_px, line_length_px,
                              mm_from_line, pixels_per_mm_from_line,
                              relative_error,
                              working_px_per_body_length_from_line)
from core.cost_model import working_px_per_body_length


def test_line_length_is_euclidean():
    assert line_length_px((0, 0), (3, 4)) == pytest.approx(5.0)
    assert line_length_px((10, 10), (10, 10)) == 0.0


def test_pixels_per_mm_from_line():
    assert pixels_per_mm_from_line(200.0, 50.0) == pytest.approx(4.0)


@pytest.mark.parametrize("length, mm", [(0.0, 10.0), (-1.0, 10.0),
                                        (100.0, 0.0), (100.0, -5.0)])
def test_degenerate_calibration_inputs_raise(length, mm):
    with pytest.raises(ValueError):
        pixels_per_mm_from_line(length, mm)


def test_mm_from_line_round_trips():
    ppm = pixels_per_mm_from_line(200.0, 50.0)
    assert mm_from_line(80.0, ppm) == pytest.approx(20.0)


def test_line_error_shrinks_with_zoom():
    # One display pixel per endpoint: magnifying buys precision proportionally,
    # and fitting a big frame into a small widget costs it.
    assert line_error_px(1.0) == pytest.approx(math.sqrt(2))
    assert line_error_px(4.0) == pytest.approx(math.sqrt(2) / 4)
    assert line_error_px(0.25) == pytest.approx(math.sqrt(2) * 4)
    with pytest.raises(ValueError):
        line_error_px(0.0)


def test_relative_error_saturates_on_a_degenerate_line():
    assert relative_error(0.0, 1.4) == 1.0
    assert relative_error(140.0, 1.4) == pytest.approx(0.01)
    # Never reports worse than 100%: an error larger than the line means the
    # line is meaningless, and 4000% would read as a precise statement.
    assert relative_error(0.5, 1.4) == 1.0


def test_the_fiducial_cancels_out_of_the_resolution_answer():
    """The invariant the sub-tool is designed around.

    Working px per body length is ``body_line_px * scale`` regardless of which
    ruler was used, so the animal line alone answers it exactly. If this ever
    fails, the dialog's claim that a ruler is optional is false.
    """
    body_px = 137.0
    for known_mm in (5.0, 12.5, 300.0):
        ppm = pixels_per_mm_from_line(400.0, known_mm)
        cal = Calibration(pixels_per_mm=ppm, body_length_px=body_px)
        for scale in (1.0, 0.5, 0.25):
            via_mm = working_px_per_body_length(
                cal.pixels_per_mm, cal.body_length_mm, scale)
            direct = working_px_per_body_length_from_line(body_px, scale)
            assert via_mm == pytest.approx(direct)
            assert direct == pytest.approx(body_px * scale)


def test_body_length_mm_needs_a_fiducial():
    cal = Calibration(pixels_per_mm=None, body_length_px=100.0)
    assert cal.body_length_mm is None
    # ...but the resolution answer is available anyway.
    assert cal.working_px_per_body_length(0.5) == pytest.approx(50.0)


def test_errors_combine_in_quadrature():
    cal = Calibration(pixels_per_mm=4.0, body_length_px=100.0,
                      fiducial_rel_err=0.03, body_rel_err=0.04)
    assert cal.body_length_mm_rel_err == pytest.approx(0.05)
    assert Calibration(pixels_per_mm=4.0, body_length_px=100.0,
                       body_rel_err=0.04).body_length_mm_rel_err is None


def test_partial_calibration_writes_only_what_it_measured():
    """A merge must not clear a field the user set by hand elsewhere."""
    animal_only = Calibration(pixels_per_mm=None, body_length_px=90.0)
    fields = animal_only.as_replicate_fields()
    assert "pixels_per_mm" not in fields
    assert "body_length_mm" not in fields
    assert fields["body_length_px"] == pytest.approx(90.0)

    full = Calibration(pixels_per_mm=4.0, body_length_px=90.0)
    fields = full.as_replicate_fields()
    assert fields["pixels_per_mm"] == pytest.approx(4.0)
    assert fields["body_length_mm"] == pytest.approx(22.5)
    assert fields["body_length_px"] == pytest.approx(90.0)

    assert Calibration(pixels_per_mm=None,
                       body_length_px=None).as_replicate_fields() == {}


def test_cost_model_prefers_the_mm_pair_and_falls_back_to_pixels():
    # Both present: the mm pair wins, because a user may have corrected it.
    assert working_px_per_body_length(4.0, 10.0, 0.5, 999.0) == pytest.approx(20.0)
    # Fiducial-free replicate still resolves.
    assert working_px_per_body_length(None, None, 0.5, 80.0) == pytest.approx(40.0)
    # Nothing measured at all.
    assert working_px_per_body_length(None, None, 1.0) is None
    assert working_px_per_body_length(4.0, None, 1.0) is None
