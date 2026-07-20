from __future__ import annotations

import os
import time
import unittest

import numpy as np

# The explorer is a QWidget, but its cache/geometry contract is testable without
# a display server.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QSignalSpy, QTest
from PyQt6.QtWidgets import QApplication

from gui.explorers.speed_explorer import (DensityPlot, MiniPlot, SpeedExplorer,
                                          _cluster_inband, _regions_from_meta)
from gui.video_panel import FrameView


class _CacheStub:
    def __init__(self):
        self.meta = {
            "video_path": "missing.mp4",
            "fps": 10.0,
            "block_size": 4,
            "grid": [3, 2],
            "src_width": 100,
            "src_height": 50,
            "work_width": 8,
            "work_height": 12,
            "downsample": 1.0,
            "features": ["speed"],
            "replicate_tiles": [
                {
                    "id": 10,
                    "label": "left",
                    "frac": [0.0, 0.0, 0.5, 1.0],
                    "source_box": [0, 0, 50, 50],
                    "grid": [1, 2],
                    "atlas_bbox": [0, 0, 1, 2],
                },
                {
                    "id": 11,
                    "label": "right",
                    "frac": [0.5, 0.0, 1.0, 1.0],
                    "source_box": [50, 0, 100, 50],
                    "grid": [1, 2],
                    "atlas_bbox": [2, 0, 3, 2],
                },
            ],
        }
        # The middle atlas row is an unowned separator. Its deliberately huge
        # values make accidental inclusion immediately visible.
        self._speed = np.array([
            [[1, 2], [999, 999], [3, 4]],
            [[2, 3], [999, 999], [4, 5]],
        ], dtype=np.float32)

    def read(self, name: str) -> np.ndarray:
        if name != "speed":
            raise KeyError(name)
        return self._speed.copy()


class SpeedExplorerRegionTests(unittest.TestCase):
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

    def test_plots_and_clumps_exclude_sparse_atlas_separators(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            # Overview pools the four owned cells, not the six cells in the
            # sparse storage atlas.
            self.assertEqual(explorer.n_blocks, 4)
            # The raw per-frame channel reads out the fastest owned block.
            np.testing.assert_array_equal(explorer.plots["dist"].y, [4, 5])
            # The detection channel is the windowed mean speed (W=2, centered).
            # Frame 0's centered window truncates to itself (raw values); frame
            # 1 averages both frames.
            np.testing.assert_array_equal(
                explorer.plots["dist_w"].matrix,
                [[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]])

            # A band that spans the whole value range accepts every owned block.
            explorer.plots["dist_w"].band_lo = 0.0
            explorer.plots["dist_w"].band_hi = 1e9
            explorer._recompute_sweep()
            np.testing.assert_array_equal(explorer.plots["count"].y, [4, 4])
            # The two tiles are independent 1x2 components, never one 1x4 clump.
            np.testing.assert_array_equal(explorer.plots["clump"].y, [2, 2])
        finally:
            explorer.close()

    def test_selected_plot_owns_the_band_and_expands(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            dist_w = explorer.plots["dist_w"]
            # The windowed speed plot is the sole channel band: always expanded
            # and always carrying the band, seeded wide open.
            self.assertTrue(dist_w.band_active)
            self.assertEqual(dist_w.maximumHeight(), MiniPlot.EXPANDED_H)
            self.assertEqual(explorer._band(), (float("-inf"), float("inf")))

            # The raw per-frame density is context only -- no band on it.
            self.assertFalse(explorer.plots["dist"].band_active)
            # The windowed-count gate carries its own always-active band.
            self.assertTrue(explorer.plots["count_w"].band_active)
        finally:
            explorer.close()

    def test_dragging_a_band_line_updates_detection(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            dist_w = explorer.plots["dist_w"]
            dist_w.resize(200, MiniPlot.EXPANDED_H)
            committed = QSignalSpy(dist_w.band_committed)

            # Grab the maximum handle at its current pixel row and drag it down
            # to just above the bulk of the windowed-speed distribution.
            r = dist_w._plot_rect()
            data_lo, data_hi = dist_w._data_range()
            _, yhi, _ = dist_w._line_ys()
            QTest.mousePress(dist_w, Qt.MouseButton.LeftButton,
                             pos=QPoint(int(r.right()) - 2, int(yhi)))
            target = dist_w._y_of(3.0, r, data_lo, data_hi)
            QTest.mouseMove(dist_w, pos=QPoint(int(r.right()) - 2, int(target)))
            QTest.mouseRelease(dist_w, Qt.MouseButton.LeftButton,
                               pos=QPoint(int(r.right()) - 2, int(target)))

            self.assertEqual(len(committed), 1)
            lo, hi = explorer._band()
            self.assertAlmostEqual(hi, 3.0, delta=0.2)
            self.assertLessEqual(lo, hi)
            # Detection reflects the dragged band, applied to windowed speed.
            windowed = explorer._w_cache
            np.testing.assert_array_equal(
                explorer.plots["count"].y,
                ((windowed >= lo) & (windowed <= hi)).sum(1))
        finally:
            explorer.close()

    def test_handle_at_plot_edge_means_unbounded_side(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            dist_w = explorer.plots["dist_w"]
            dist_w.resize(200, MiniPlot.EXPANDED_H)
            r = dist_w._plot_rect()
            data_lo, data_hi = dist_w._data_range()

            # Fresh band accepts every owned block's windowed speed.
            explorer._recompute_sweep()
            np.testing.assert_array_equal(explorer.plots["count"].y, [4, 4])

            # Drag the lo handle up into the plot: a finite lower threshold.
            x = int(r.right()) - 2
            ylo, _, _ = dist_w._line_ys()
            QTest.mousePress(dist_w, Qt.MouseButton.LeftButton,
                             pos=QPoint(x, int(ylo)))
            target = dist_w._y_of(3.0, r, data_lo, data_hi)
            QTest.mouseMove(dist_w, pos=QPoint(x, int(target)))
            QTest.mouseRelease(dist_w, Qt.MouseButton.LeftButton,
                               pos=QPoint(x, int(target)))
            lo, hi = explorer._band()
            self.assertAlmostEqual(lo, 3.0, delta=0.2)
            # The untouched hi handle stays unbounded: the top region includes
            # the absolute highest values.
            self.assertEqual(hi, float("inf"))
            windowed = explorer._w_cache
            np.testing.assert_array_equal(
                explorer.plots["count"].y, (windowed >= lo).sum(1))

            # Dragging lo off the bottom edge re-opens that side.
            ylo, _, _ = dist_w._line_ys()
            QTest.mousePress(dist_w, Qt.MouseButton.LeftButton,
                             pos=QPoint(x, int(ylo)))
            QTest.mouseMove(dist_w, pos=QPoint(x, int(r.bottom()) + 1))
            QTest.mouseRelease(dist_w, Qt.MouseButton.LeftButton,
                               pos=QPoint(x, int(r.bottom()) + 1))
            self.assertEqual(explorer._band(),
                             (float("-inf"), float("inf")))
        finally:
            explorer.close()

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

    def test_cluster_inband_bridges_gaps_without_inflating_size(self):
        # Three in-band blocks with a one-block gap between each: . lets us tell
        # single-link Chebyshev clustering apart from morphological dilation.
        mask = np.array([[1, 0, 1, 0, 1]], dtype=bool)
        w = np.ones_like(mask, dtype=float)

        # gap=1 is plain 8-connectivity: the blocks are Chebyshev distance 2
        # apart, so they stay three separate size-1 clumps. (Dilation-by-1 would
        # wrongly merge them at the 2*gap overlap -- this guards that.)
        _, sizes = _cluster_inband(mask, w, 1)
        self.assertEqual(sorted(sizes.values()), [1.0, 1.0, 1.0])

        # gap=2 bridges all three into one clump, but the size counts only the
        # three real in-band blocks, never the two bridged gaps (=5 if dilated).
        _, sizes = _cluster_inband(mask, w, 2)
        self.assertEqual(list(sizes.values()), [3.0])

        # Valid-area weights are honoured: a half-area block counts as 0.5.
        w2 = np.array([[1.0, 0, 0.5, 0, 1.0]])
        _, sizes = _cluster_inband(mask, w2, 2)
        self.assertAlmostEqual(sum(sizes.values()), 2.5)

    def test_windowed_count_in_band_drives_positive_detection_channel(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            # Wide-open channel band: all 4 owned blocks are in band every
            # frame, so the in-band count is a flat 4. The detection window D
            # is 1 frame here, so the windowed count equals the raw count, and
            # with the count band seeded wide open detection fires every frame.
            explorer._recompute_sweep()
            np.testing.assert_array_equal(explorer.plots["count"].y, [4, 4])
            np.testing.assert_array_equal(explorer.plots["count_w"].y, [4, 4])
            np.testing.assert_array_equal(explorer.plots["detect"].y, [1, 1])
            # The clump diagnostic still runs on the windowed field: each 1x2
            # tile is one gap=1 clump of weighted size 2.
            np.testing.assert_array_equal(explorer.plots["clump"].y, [2, 2])

            # Raising the count band's min above 4 rejects every frame: the
            # sustained-evidence threshold is now unmet.
            explorer.plots["count_w"].band_lo = 5.0
            explorer._recompute_detect()
            np.testing.assert_array_equal(explorer.plots["detect"].y, [0, 0])
        finally:
            explorer.close()

    def test_replicate_click_focuses_data_and_caches_band_per_scope(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            speed_scale = explorer.vmax
            unbounded = (float("-inf"), float("inf"))
            # The overview band seeds wide open; place a finite threshold so we
            # can tell below whether it survives a scope round-trip.
            self.assertEqual(explorer._band(), unbounded)
            explorer.plots["dist_w"].band_lo = 3.0

            explorer._redraw_video()
            explorer._on_video_clicked(QPoint(25, 25))
            self.assertEqual(explorer.active_region_index, 0)
            self.assertEqual(explorer.n_blocks, 2)
            self.assertEqual(explorer.video_view.focus_frac,
                             (0.0, 0.0, 0.5, 1.0))
            # This replicate has never been visited before, so its own band
            # starts wide open -- a *different* scope's threshold would be
            # meaningless on this replicate's value scale.
            self.assertEqual(explorer._band(), unbounded)
            self.assertEqual(explorer.vmax, speed_scale)
            np.testing.assert_array_equal(explorer.plots["dist"].y, [2, 3])
            # The detection channel is this replicate's two owned blocks,
            # windowed (W=2, centered): frame 0 truncates to itself, frame 1
            # averages both frames.
            np.testing.assert_array_equal(
                explorer._w_cache, [[1.0, 2.0], [1.5, 2.5]])
            windowed = explorer._w_cache
            lo, hi = explorer._band()
            np.testing.assert_array_equal(
                explorer.plots["count"].y,
                ((windowed >= lo) & (windowed <= hi)).sum(1))

            explorer.focus_mode.setCurrentIndex(1)
            flow_preview = explorer._base_frame()
            self.assertEqual(flow_preview.shape, (4, 8, 3))
            np.testing.assert_array_equal(flow_preview[..., 0],
                                          flow_preview[..., 1])

            explorer._clear_region_focus()
            self.assertEqual(explorer.active_region_index, -1)
            self.assertEqual(explorer.n_blocks, 4)
            self.assertIsNone(explorer.video_view.focus_frac)
            # Returning to the overview restores the threshold it held before
            # the focus switch -- the same scope, so its scale hasn't changed.
            self.assertEqual(explorer._band(), (3.0, float("inf")))
            self.assertEqual(explorer.vmax, speed_scale)

            # Re-focusing the replicate restores wide-open again: its own band
            # was never touched during the earlier visit, only the pooled
            # scope's was.
            explorer._on_video_clicked(QPoint(25, 25))
            self.assertEqual(explorer._band(), unbounded)
            explorer.plots["dist_w"].band_lo = 2.5
            explorer._clear_region_focus()
            explorer._on_video_clicked(QPoint(25, 25))
            # Now this replicate's own cached band comes back too.
            self.assertEqual(explorer._band(), (2.5, float("inf")))
        finally:
            explorer.close()

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

    def test_space_toggles_standalone_playback_after_video_click(self):
        explorer = SpeedExplorer(_CacheStub())
        explorer.show()
        self.app.processEvents()
        try:
            QTest.mouseClick(explorer.video_view, Qt.MouseButton.LeftButton,
                             pos=explorer.video_view._draw_rect.center())
            self.assertIs(QApplication.focusWidget(), explorer.video_view)

            QTest.keyClick(explorer.video_view, Qt.Key.Key_Space)
            self.app.processEvents()
            self.assertTrue(explorer.playing)
            self.assertTrue(explorer.timer.isActive())

            QTest.keyClick(explorer.video_view, Qt.Key.Key_Space)
            self.app.processEvents()
            self.assertFalse(explorer.playing)
            self.assertFalse(explorer.timer.isActive())
        finally:
            explorer.close()

    def test_shift_temporarily_hides_and_restores_every_video_overlay(self):
        explorer = SpeedExplorer(_CacheStub())
        explorer.show()
        self.app.processEvents()
        try:
            explorer.video_view.setFocus()
            selected_mode = explorer.overlay_mode.currentText()
            self.assertTrue(explorer.video_view.boxes)
            overlay_pixel = explorer.video_view._pix.toImage().pixelColor(1, 1)

            QTest.keyPress(explorer.video_view, Qt.Key.Key_Shift)
            self.app.processEvents()
            self.assertTrue(explorer._overlay_peek_hidden)
            self.assertTrue(explorer.video_view._overlays_hidden)
            self.assertEqual(explorer.overlay_mode.currentText(), selected_mode)
            raw_pixel = explorer.video_view._pix.toImage().pixelColor(1, 1)
            self.assertEqual(raw_pixel.getRgb()[:3], (0, 0, 0))
            self.assertNotEqual(raw_pixel, overlay_pixel)

            QTest.keyRelease(explorer.video_view, Qt.Key.Key_Shift)
            self.app.processEvents()
            self.assertFalse(explorer._overlay_peek_hidden)
            self.assertFalse(explorer.video_view._overlays_hidden)
            self.assertEqual(explorer.overlay_mode.currentText(), selected_mode)
            self.assertTrue(explorer.video_view.boxes)
            restored_pixel = explorer.video_view._pix.toImage().pixelColor(1, 1)
            self.assertEqual(restored_pixel, overlay_pixel)
        finally:
            explorer.close()


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
        # Every plot here is a QWidget; without this the class passes only when
        # some earlier class in the file happened to create the application.
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
