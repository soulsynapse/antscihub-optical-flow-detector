"""The in-band highlight overlay's fast path (see FINDINGS.md, playback lag).

The highlight branch in ``ScalogramExplorer._redraw_video`` is gated on the
selected channel's density matrix, so it switches on the instant the Morlet cube
lands and then runs on every frame. Measured at 1280x720 it cost 6.4 ms/frame
against a 1.6 ms pre-cube baseline -- a 5x on the overlay, and the visible cause
of playback dropping to ~1/3 realtime once "change energy" loaded.

The fix is a pure speed change, so the test that matters is that it is a pure
speed change: the fast path must paint EXACTLY the pixels the old one did. A
performance fix that quietly alters the picture is worse than the lag, because
the picture is what the detection threshold is tuned against by eye.
"""
from __future__ import annotations

import os
import unittest

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtWidgets import QApplication

from gui.explorers.scalogram_explorer import ScalogramExplorer

_APP = QApplication.instance() or QApplication([])


def _reference_blend(roi: np.ndarray, mm: np.ndarray) -> np.ndarray:
    """The ORIGINAL implementation, kept verbatim as the oracle.

    Deliberately a copy rather than a call into the explorer: an oracle that
    shared code with the thing under test could not detect a change to it.
    """
    roi = roi.copy()
    tint = np.zeros_like(roi)
    tint[..., 1] = 255
    blended = cv2.addWeighted(roi, 0.5, tint, 0.5, 0)
    np.copyto(roi, blended, where=(mm > 0)[:, :, None])
    return roi


def _fast_blend(explorer, roi: np.ndarray, mm: np.ndarray) -> np.ndarray:
    """What _redraw_video now does, through the explorer's own tint cache."""
    roi = roi.copy()
    blended = cv2.addWeighted(roi, 0.5, explorer._tint(roi.shape), 0.5, 0)
    cv2.copyTo(blended, mm, roi)
    return roi


class _TintOwner:
    """Just the tint-cache slice of the explorer, so these tests need no video,
    no cache and no cube -- the helper under test touches nothing else."""

    def __init__(self):
        self._tint_cache = {}

    _tint = ScalogramExplorer._tint


class HighlightBlendTest(unittest.TestCase):
    def setUp(self):
        self.ex = _TintOwner()

    def _check(self, roi, mm, msg):
        want = _reference_blend(roi, mm)
        got = _fast_blend(self.ex, roi, mm)
        self.assertTrue(
            np.array_equal(want, got),
            f"{msg}: fast path changed the picture "
            f"({int((want != got).sum())} differing subpixels)")

    def test_identical_across_mask_densities(self):
        """Every regime, including the two boundaries: cv2.copyTo and
        np.copyto(where=) must agree on which pixels get written."""
        rng = np.random.default_rng(0)
        roi = rng.integers(0, 256, (180, 240, 3), dtype=np.uint8)
        blocks = rng.random((18, 24))
        for frac in (0.0, 0.05, 0.5, 0.95, 1.0):
            grid = (blocks < frac).astype(np.uint8)
            mm = cv2.resize(grid, (240, 180), interpolation=cv2.INTER_NEAREST)
            self._check(roi, mm, f"{frac:.0%} of blocks passing")

    def test_identical_on_a_non_contiguous_view(self):
        """_redraw_video passes ``out[dy0:dy1, dx0:dx1]`` -- a sliced, non-
        contiguous view. cv2.copyTo has to write through it in place, which is
        the one behaviour that could differ from numpy's and would show up as a
        highlight that lands on the wrong replicate tile."""
        rng = np.random.default_rng(1)
        full = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
        roi = full[40:220, 60:300]
        self.assertFalse(roi.flags["C_CONTIGUOUS"], "precondition: a real view")
        grid = (rng.random((18, 24)) < 0.4).astype(np.uint8)
        mm = cv2.resize(grid, (roi.shape[1], roi.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        self._check(roi, mm, "sliced ROI view")

    def test_rounding_is_preserved_at_the_extremes(self):
        """The blend is a 0.5/0.5 addWeighted, so odd channel values land on a
        .5 and the two implementations must round it the same way. Random uint8
        data covers this statistically; this pins it deterministically."""
        for val in (0, 1, 127, 128, 129, 254, 255):
            roi = np.full((8, 8, 3), val, np.uint8)
            mm = np.ones((8, 8), np.uint8)
            self._check(roi, mm, f"uniform channel value {val}")

    def test_tint_is_cached_per_shape_and_is_the_green_layer(self):
        """The allocation this removes. A wrong constant here would tint the
        highlight the wrong colour, so assert the value, not just the identity.
        """
        a = self.ex._tint((10, 12, 3))
        self.assertEqual(a.shape, (10, 12, 3))
        self.assertEqual(a.dtype, np.uint8)
        self.assertTrue((a[..., 0] == 0).all() and (a[..., 2] == 0).all(),
                        "tint is not pure green in BGR")
        self.assertTrue((a[..., 1] == 255).all(), "green channel is not full")
        self.assertIs(self.ex._tint((10, 12, 3)), a, "tint was re-allocated")
        self.assertIsNot(self.ex._tint((10, 13, 3)), a,
                         "a different ROI shape reused the wrong tint")


if __name__ == "__main__":
    unittest.main()
