"""MiniPlot's decimated envelope: the vectorised form must match the loop.

`MiniPlot.paintEvent` rebuilt its per-pixel-column envelope on every paint --
a Python loop calling np.nanmax once per column, so once per frame during
playback for a series that had not changed. At T=1200 that measured 1.70
ms/paint against 0.11 ms for the DensityPlot beside it. It is now memoised and
vectorised (1.70 -> 0.20 ms).

The memo is safe by construction (keyed on a version counter with a single
writer). The VECTORISATION is not: `np.fmax.reduceat` has its own rules for
empty and degenerate index runs, and the envelope takes a MAX per column
precisely so a brief burst survives decimation -- an off-by-one in the bin
edges would silently shift or swallow one. Hence an oracle test.
"""
from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtWidgets import QApplication

from gui.explorers.speed_explorer import MiniPlot, PixelBarPlot

_APP = QApplication.instance() or QApplication([])


def _reference_envelope(y: np.ndarray, cols: int, lo: float) -> np.ndarray:
    """The ORIGINAL loop, kept verbatim as the oracle."""
    n = y.size
    edges = np.linspace(0, n, cols + 1).astype(int)
    env = np.empty(cols, np.float32)
    for i in range(cols):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        seg = y[a:b]
        env[i] = np.nanmax(seg) if seg.size else lo
    return env


class EnvelopeEquivalenceTest(unittest.TestCase):
    def _check(self, y, cols, msg):
        pl = MiniPlot("t")
        pl.set_series(y)
        lo, _ = pl._data_range()
        want = _reference_envelope(pl.y, cols, lo)
        got = pl._envelope(cols, lo)
        self.assertEqual(got.shape, want.shape, f"{msg}: wrong column count")
        np.testing.assert_allclose(
            got, want, rtol=0, atol=0,
            err_msg=f"{msg}: vectorised envelope diverged from the loop")

    def test_matches_the_loop_across_decimation_ratios(self):
        rng = np.random.default_rng(0)
        for n in (2, 7, 100, 1200, 5000):
            y = rng.random(n).astype(np.float32) * 100
            for cols in (1, 2, 13, min(n, 460), n):
                self._check(y, cols, f"n={n} cols={cols}")

    def test_no_decimation_is_the_identity(self):
        """cols == n: every column is one sample, so the envelope IS the
        series. The tightest constraint on the bin edges."""
        rng = np.random.default_rng(1)
        y = rng.random(64).astype(np.float32)
        pl = MiniPlot("t")
        pl.set_series(y)
        lo, _ = pl._data_range()
        np.testing.assert_allclose(pl._envelope(64, lo), y, rtol=0, atol=0)

    def test_a_single_frame_burst_survives_decimation(self):
        """The reason it is a max and not a mean (class docstring). A 1-frame
        spike in 1200 frames decimated to 460 columns must still be the peak."""
        y = np.full(1200, 1.0, np.float32)
        y[733] = 99.0
        pl = MiniPlot("t")
        pl.set_series(y)
        lo, _ = pl._data_range()
        env = pl._envelope(460, lo)
        self.assertEqual(float(env.max()), 99.0, "the burst was decimated away")
        self.assertEqual(int((env == 99.0).sum()), 1,
                         "the burst was smeared across columns")

    def test_nan_columns(self):
        """NaN handling is the one deliberate divergence from the oracle: an
        all-NaN column made the loop emit NaN, which becomes a NaN QPointF and
        an undefined polyline vertex. It now falls back to the axis minimum --
        which is what the loop's own unreachable `else lo` branch intended.
        Partially-NaN columns must still match exactly.
        """
        y = np.arange(100, dtype=np.float32)
        y[10:20] = np.nan                     # some columns fully NaN
        y[55] = np.nan                        # a partial column
        pl = MiniPlot("t")
        pl.set_series(y)
        lo, _ = pl._data_range()
        env = pl._envelope(50, lo)
        self.assertFalse(np.isnan(env).any(), "a NaN reached the polyline")
        want = _reference_envelope(pl.y, 50, lo)
        both = ~np.isnan(want)
        np.testing.assert_allclose(env[both], want[both], rtol=0, atol=0,
                                   err_msg="non-NaN columns diverged")


class MemoInvalidationTest(unittest.TestCase):
    """The memos are only safe if every writer of the series bumps the version.
    These assert the invalidation, not the speed."""

    def test_a_new_series_invalidates_the_envelope(self):
        pl = MiniPlot("t")
        pl.set_series(np.zeros(100, np.float32))
        first = pl._envelope(50, 0.0).copy()
        pl.set_series(np.full(100, 7.0, np.float32))
        second = pl._envelope(50, 0.0)
        self.assertFalse(np.array_equal(first, second),
                         "a new series reused the previous envelope")
        self.assertEqual(float(second.max()), 7.0)

    def test_a_new_series_invalidates_the_data_range(self):
        pl = MiniPlot("t")
        pl.set_series(np.linspace(0, 1, 50, dtype=np.float32))
        self.assertAlmostEqual(pl._data_range()[1], 1.0)
        pl.set_series(np.linspace(0, 500, 50, dtype=np.float32))
        self.assertAlmostEqual(pl._data_range()[1], 500.0,
                               msg="stale range memo survived a new series")

    def test_the_bar_image_is_rebuilt_for_a_new_series(self):
        pl = PixelBarPlot("t")
        pl.resize(200, 80)
        pl.set_series(np.zeros(100, np.float32))
        a = pl._bar_image(120, 60, 1.0)
        pl.set_series(np.full(100, 5.0, np.float32))
        b = pl._bar_image(120, 60, 5.0)
        self.assertIsNot(a, b, "stale bar image served after a new series")

    def test_the_density_matrix_invalidates_the_base_memos(self):
        """DensityPlot.set_matrix writes self.y directly instead of going
        through set_series, so it is the writer most likely to forget."""
        from gui.explorers.speed_explorer import DensityPlot
        dp = DensityPlot("t")
        dp.set_matrix(np.ones((40, 5), np.float32))
        v1 = dp._ver
        dp.set_matrix(np.full((40, 5), 9.0, np.float32))
        self.assertNotEqual(dp._ver, v1, "set_matrix did not bump the version")
        self.assertIsNone(dp._env_memo, "set_matrix left a stale envelope memo")


if __name__ == "__main__":
    unittest.main()
