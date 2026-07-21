"""Widget library that used to live in speed_explorer.py, now gui/explorers/plots.py.

The cache-backed SpeedExplorer (and its _cluster_inband clump helper) were removed
with the flow-cache subsystem; what remains are the reusable plot widgets the live
tensor surface depends on -- MiniPlot, DensityPlot, PixelBarPlot -- plus the shared
_regions_from_meta geometry check. These tests exercise those without a cache.
"""
from __future__ import annotations

import os
import time
import unittest

import numpy as np

# The widgets are QWidgets, but their geometry/range contracts are testable without
# a display server.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QSignalSpy, QTest
from PyQt6.QtWidgets import QApplication

from gui.explorers.plots import DensityPlot, MiniPlot, _regions_from_meta
from gui.video_panel import FrameView


class WidgetGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_rejects_tile_geometry_outside_the_atlas(self):
        meta = {
            "replicate_tiles": [{
                "id": 1,
                "atlas_bbox": [0, 0, 4, 2],
                "grid": [4, 2],
            }]
        }
        with self.assertRaisesRegex(ValueError, "outside cache grid"):
            _regions_from_meta(meta, (3, 2))

    def test_density_plot_spans_the_matrix_and_reads_out_the_max(self):
        pl = DensityPlot("dist", "px/s")
        pl.resize(120, MiniPlot.EXPANDED_H)
        # (T=3, K=4): a low-speed bulk with one fast block per frame, exactly the
        # sparse-signal case a spatial mean would bury.
        m = np.array([[0.0, 0.1, 0.2, 8.0],
                      [0.0, 0.1, 0.2, 9.0],
                      [0.0, 0.1, 0.2, 5.0]], np.float32)
        pl.set_matrix(m)
        # The value axis spans the whole per-block range, not the mean's range.
        self.assertEqual(pl._data_range(), (0.0, 9.0))
        # The cursor readout tracks the fastest block per frame, not the mean.
        np.testing.assert_array_equal(pl.y, [8.0, 9.0, 5.0])
        # The heatmap renders as an image at the plot's pixel size (binned, so it
        # never scales with T*K), and is cached until the data changes.
        r = pl._plot_rect()
        img = pl._density_image(int(r.width()), int(r.height()), 0.0, 9.0)
        self.assertIsNotNone(img)
        self.assertEqual((img.width(), img.height()),
                         (int(r.width()), int(r.height())))
        self.assertIs(pl._density_image(int(r.width()), int(r.height()),
                                        0.0, 9.0), img)
        # It is a real detection channel: seeding the band opens it wide.
        pl.set_band_active(True)
        self.assertEqual(pl.band(), (float("-inf"), float("inf")))

    def test_frame_view_focus_preserves_full_frame_click_coordinates(self):
        view = FrameView()
        view.resize(600, 400)
        focus = (0.25, 0.2, 0.75, 0.8)
        # Pixels are a native-detail crop, while emitted points stay in the
        # 200x100 full-frame coordinate system.
        view.set_frame(np.zeros((60, 100, 3), np.uint8), image_frac=focus,
                       coordinate_size=(200, 100))
        view.set_focus_frac(focus)
        view.show()
        self.app.processEvents()
        try:
            source_aspect = 100 / 60
            drawn_aspect = view._draw_rect.width() / view._draw_rect.height()
            # Integer target pixels permit sub-pixel rounding error, but the
            # image must never be independently stretched to the widget bounds.
            self.assertAlmostEqual(drawn_aspect, source_aspect, delta=0.01)
            self.assertTrue(view._draw_rect.width() < view.width() or
                            view._draw_rect.height() < view.height())

            click_spy = QSignalSpy(view.clicked)
            back_spy = QSignalSpy(view.back_requested)
            center = view._draw_rect.center()
            QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=center)
            self.assertEqual(len(click_spy), 1)
            point = click_spy[0][0]
            self.assertAlmostEqual(point.x(), 100, delta=1)
            self.assertAlmostEqual(point.y(), 50, delta=1)

            QTest.mouseClick(view, Qt.MouseButton.RightButton, pos=center)
            self.assertEqual(len(back_spy), 1)
        finally:
            view.close()


class DensityPlotDataRangeTests(unittest.TestCase):
    """_data_range is memoised because it was ~100% of this widget's repaint
    cost: paintEvent, _line_ys and _apply_drag each call it, so it ran several
    times per mouse-move and once per plot per frame during playback, every
    time re-deriving a constant from a full T x B scan (22.5 ms of a 22.8 ms
    repaint at 1800 frames x 8000 blocks). The memo must not change what it
    ANSWERS, only how often it is asked."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _reference(m):
        """The pre-memo implementation, verbatim: the range of the FINITE
        cells, falling back to 0..1 when there are none."""
        finite = m[np.isfinite(m)]
        if finite.size == 0:
            return 0.0, 1.0
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def test_matches_the_unmemoised_range_on_every_edge_case(self):
        cases = {
            "plain": np.array([[1.0, 5.0], [2.0, 3.0]], np.float32),
            "with NaN": np.array([[1.0, np.nan], [2.0, 3.0]], np.float32),
            # +/-inf must NOT become the axis bound: the contract is the finite
            # range, and the fast nanmin/nanmax path cannot express that.
            "with +inf": np.array([[1.0, np.inf], [2.0, 3.0]], np.float32),
            "with -inf": np.array([[-np.inf, 5.0], [2.0, 3.0]], np.float32),
            "inf both": np.array([[-np.inf, np.inf], [2.0, 3.0]], np.float32),
            "all NaN": np.full((2, 2), np.nan, np.float32),
            "all inf": np.full((2, 2), np.inf, np.float32),
            "constant": np.full((2, 2), 4.0, np.float32),
            "empty": np.zeros((0, 0), np.float32),
        }
        for name, m in cases.items():
            with self.subTest(name):
                pl = DensityPlot("x")
                pl.set_matrix(m)
                self.assertEqual(pl._data_range(), self._reference(m))

    def test_repeated_calls_are_stable(self):
        pl = DensityPlot("x")
        pl.set_matrix(np.array([[1.0, 5.0], [2.0, 3.0]], np.float32))
        first = pl._data_range()
        for _ in range(5):
            self.assertEqual(pl._data_range(), first)

    def test_a_new_matrix_invalidates_the_memo(self):
        """The whole hazard of caching a derived value: a stale range would
        rescale every band handle against data that is no longer there."""
        pl = DensityPlot("x")
        pl.set_matrix(np.array([[1.0, 5.0]], np.float32))
        self.assertEqual(pl._data_range(), (1.0, 5.0))
        pl.set_matrix(np.array([[10.0, 90.0]], np.float32))
        self.assertEqual(pl._data_range(), (10.0, 90.0))
        pl.set_matrix(np.zeros((0, 0), np.float32))
        self.assertEqual(pl._data_range(), (0.0, 1.0))

    def test_repaint_does_not_scale_with_block_count(self):
        """The point of the memo. The heatmap image is already cached, so once
        the range is too, a repaint is O(pixels) -- a 20x block increase must
        not show up as a 20x repaint."""
        def per_frame_ms(B, T=400):
            m = (np.random.default_rng(0).random((T, B), dtype=np.float32) + 0.1)
            pl = DensityPlot("x")
            pl.resize(460, MiniPlot.EXPANDED_H)
            pl.set_expanded(True)
            pl.set_collapsed(False)
            pl.set_matrix(m)
            pl.grab()                       # warm the image cache
            t = time.perf_counter()
            for i in range(20):
                pl.cursor = i % T
                pl.grab()                   # forces a real synchronous paint
            return (time.perf_counter() - t) / 20 * 1000

        small = per_frame_ms(200)
        large = per_frame_ms(4000)          # 20x the blocks
        # Generous bound: this asserts the O(T*B) scan is gone, not a timing
        # target. Before the memo this ratio tracked the block count directly.
        self.assertLess(large, max(small * 5, 2.0),
                        f"repaint scaled with block count: "
                        f"{small:.2f} ms -> {large:.2f} ms")


class StickyRangeTests(unittest.TestCase):
    """A live pass replaces the series ~10x a second. Autoscaling per series then
    rescales the axis continuously, which moves a placed band's pixel position
    while the threshold behind it has not changed, and lets a burst two windows
    ago silently redefine what "high" means."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_off_by_default_so_existing_callers_keep_autoscaling(self):
        pl = MiniPlot("x")
        pl.set_series(np.array([0.0, 10.0], np.float32))
        self.assertEqual(pl._data_range(), (0.0, 10.0))
        pl.set_series(np.array([0.0, 2.0], np.float32))
        self.assertEqual(pl._data_range(), (0.0, 2.0))

    def test_the_axis_only_ever_widens(self):
        pl = MiniPlot("x")
        pl.set_sticky_range(True)
        pl.set_series(np.array([1.0, 5.0], np.float32))
        self.assertEqual(pl._data_range(), (1.0, 5.0))
        pl.set_series(np.array([2.0, 3.0], np.float32))      # narrower
        self.assertEqual(pl._data_range(), (1.0, 5.0))       # held
        pl.set_series(np.array([-4.0, 9.0], np.float32))     # wider both ways
        self.assertEqual(pl._data_range(), (-4.0, 9.0))
        pl.set_series(np.array([0.0, 1.0], np.float32))
        self.assertEqual(pl._data_range(), (-4.0, 9.0))

    def test_an_empty_series_does_not_pollute_the_accumulated_bounds(self):
        """An empty plot's raw range is the 0..1 placeholder, not a measurement.
        Folding it in would drag the floor to 0 the first time a window arrives
        short -- and it never recovers, because the range only widens."""
        pl = MiniPlot("x")
        pl.set_sticky_range(True)
        pl.set_series(np.array([5.0, 9.0], np.float32))
        self.assertEqual(pl._data_range(), (5.0, 9.0))
        pl.set_series(np.zeros(0, np.float32))
        self.assertEqual(pl._data_range(), (5.0, 9.0))

    def test_reset_re_seeds_from_the_current_series(self):
        """Called when the plot changes what it MEASURES -- another replicate,
        another channel -- so one scope's outliers cannot land on another's
        axis."""
        pl = MiniPlot("x")
        pl.set_sticky_range(True)
        pl.set_series(np.array([0.0, 100.0], np.float32))
        self.assertEqual(pl._data_range(), (0.0, 100.0))
        pl.set_series(np.array([1.0, 3.0], np.float32))
        pl.reset_sticky_range()
        self.assertEqual(pl._data_range(), (1.0, 3.0))

    def test_it_applies_to_the_heatmap_and_the_bar_plot_too(self):
        """Both override the raw range; the sticky layer wraps them rather than
        being reimplemented per subclass."""
        dp = DensityPlot("x")
        dp.set_sticky_range(True)
        dp.set_matrix(np.array([[1.0, 8.0]], np.float32))
        self.assertEqual(dp._data_range(), (1.0, 8.0))
        dp.set_matrix(np.array([[2.0, 3.0]], np.float32))
        self.assertEqual(dp._data_range(), (1.0, 8.0))

    def test_a_held_axis_holds_a_band_handle_still(self):
        """The point of the whole thing: the band is stored in data units, so a
        rescaling axis silently moves where it is DRAWN. Same threshold, same
        pixel."""
        pl = MiniPlot("x")
        pl.resize(300, MiniPlot.BASE_H)
        pl.set_sticky_range(True)
        pl.set_band_active(True)
        pl.set_series(np.array([0.0, 100.0], np.float32))
        pl.band_lo, pl.band_hi = 20.0, 80.0
        before = pl._line_ys()[:2]
        pl.set_series(np.array([0.0, 10.0], np.float32))     # would rescale 10x
        self.assertEqual(pl._line_ys()[:2], before)


class BandInRangeTest(unittest.TestCase):
    """The windowed-count threshold plot autoscales to the current window (it is
    NOT sticky), so a quiet window whose own max sits below the threshold would
    leave the count band off the top of the axis -- the one control the plot
    exists to read, gone exactly when the window goes silent. Including the band
    in the drawn range keeps it anchored at its true value on every window."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_off_by_default_so_the_band_never_widens_the_axis(self):
        pl = MiniPlot("x")
        pl.set_band_active(True)
        pl.band_lo, pl.band_hi = 0.0, 500.0
        pl.set_series(np.array([0.0, 10.0], np.float32))
        self.assertEqual(pl._data_range(), (0.0, 10.0))       # band ignored

    def test_a_threshold_above_the_window_max_stays_on_the_axis(self):
        pl = MiniPlot("x")
        pl.set_include_band_in_range(True)
        pl.set_band_active(True)
        pl.band_lo, pl.band_hi = 50.0, float("inf")           # a lower threshold
        pl.set_series(np.array([0.0, 10.0], np.float32))      # quiet window
        lo, hi = pl._data_range()
        self.assertEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 50.0)                     # reaches the band

    def test_infinite_and_unset_endpoints_do_not_touch_the_range(self):
        pl = MiniPlot("x")
        pl.set_include_band_in_range(True)
        pl.set_band_active(True)
        pl.band_lo, pl.band_hi = float("-inf"), float("inf")  # both unbounded
        pl.set_series(np.array([2.0, 8.0], np.float32))
        self.assertEqual(pl._data_range(), (2.0, 8.0))
        pl.band_lo, pl.band_hi = None, None                   # unset
        self.assertEqual(pl._data_range(), (2.0, 8.0))

    def test_the_handle_reads_its_true_value_when_the_window_drops_below_it(self):
        """The re-anchor the design asks for. _line_ys clamps the band into the
        drawn range, so a quiet window pins the handle to the top either way --
        the difference is what the top MEANS. Without widening the axis top is the
        window max (5), so the handle at the top misreports the threshold as 5;
        widening makes the top the threshold itself, so the handle at the top
        reads its true 40."""
        pl = MiniPlot("x")
        pl.resize(300, MiniPlot.BASE_H)
        pl.set_include_band_in_range(True)
        pl.set_band_active(True)
        pl.band_lo, pl.band_hi = 40.0, float("inf")
        pl.set_series(np.array([0.0, 5.0], np.float32))       # quiet window
        ylo, _, r = pl._line_ys()
        lo, hi = pl._data_range()
        self.assertEqual(hi, 40.0)                            # axis reaches it
        self.assertAlmostEqual(pl._val_of(ylo, r, lo, hi), 40.0, places=4)


if __name__ == "__main__":
    unittest.main()
