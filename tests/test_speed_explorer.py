from __future__ import annotations

import os
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
            np.testing.assert_array_equal(explorer.plots["dist"].y, [4, 5])

            # A band that spans the whole value range accepts every owned block.
            explorer.plots["dist"].band_lo = 0.0
            explorer.plots["dist"].band_hi = 1e9
            explorer._recompute_threshold_series()
            explorer._recompute_clump()
            np.testing.assert_array_equal(explorer.plots["count"].y, [4, 4])
            # The two tiles are independent 1x2 components, never one 1x4 clump.
            np.testing.assert_array_equal(explorer.plots["clump"].y, [2, 2])
        finally:
            explorer.close()

    def test_selected_plot_owns_the_band_and_expands(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            dist = explorer.plots["dist"]
            # The per-block speed plot is the sole detection channel: always
            # expanded and always carrying the band, seeded wide open.
            self.assertTrue(dist.band_active)
            self.assertEqual(dist.maximumHeight(), MiniPlot.EXPANDED_H)
            self.assertEqual(explorer._band(), (float("-inf"), float("inf")))

            # A non-sweep sparkline never grows a band.
            self.assertFalse(explorer.plots["count"].band_active)
        finally:
            explorer.close()

    def test_dragging_a_band_line_updates_detection(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            dist = explorer.plots["dist"]
            dist.resize(200, MiniPlot.EXPANDED_H)
            committed = QSignalSpy(dist.band_committed)

            # Grab the maximum handle at its current pixel row and drag it down
            # to just above the fastest per-block speed's neighbourhood.
            r = dist._plot_rect()
            data_lo, data_hi = dist._data_range()
            _, yhi, _ = dist._line_ys()
            QTest.mousePress(dist, Qt.MouseButton.LeftButton,
                             pos=QPoint(int(r.right()) - 2, int(yhi)))
            target = dist._y_of(3.0, r, data_lo, data_hi)
            QTest.mouseMove(dist, pos=QPoint(int(r.right()) - 2, int(target)))
            QTest.mouseRelease(dist, Qt.MouseButton.LeftButton,
                               pos=QPoint(int(r.right()) - 2, int(target)))

            self.assertEqual(len(committed), 1)
            lo, hi = explorer._band()
            self.assertAlmostEqual(hi, 3.0, delta=0.2)
            self.assertLessEqual(lo, hi)
            # Detection now reflects the dragged band, applied to per-block speed.
            selected = explorer._active_block_values(explorer.speed)
            np.testing.assert_array_equal(
                explorer.plots["count"].y,
                ((selected >= lo) & (selected <= hi)).sum(1))
        finally:
            explorer.close()

    def test_handle_at_plot_edge_means_unbounded_side(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            dist = explorer.plots["dist"]
            dist.resize(200, MiniPlot.EXPANDED_H)
            r = dist._plot_rect()
            data_lo, data_hi = dist._data_range()

            # Fresh band accepts the fastest per-block speeds (4 and 5) even
            # though the plotted density series tops out at 3.5 -- the original
            # bug clamped hi to the plot max and silently rejected them.
            explorer._recompute_threshold_series()
            np.testing.assert_array_equal(explorer.plots["count"].y, [4, 4])

            # Drag the lo handle up into the plot: a finite lower threshold.
            x = int(r.right()) - 2
            ylo, _, _ = dist._line_ys()
            QTest.mousePress(dist, Qt.MouseButton.LeftButton,
                             pos=QPoint(x, int(ylo)))
            target = dist._y_of(3.0, r, data_lo, data_hi)
            QTest.mouseMove(dist, pos=QPoint(x, int(target)))
            QTest.mouseRelease(dist, Qt.MouseButton.LeftButton,
                               pos=QPoint(x, int(target)))
            lo, hi = explorer._band()
            self.assertAlmostEqual(lo, 3.0, delta=0.2)
            # The untouched hi handle stays unbounded: the top region includes
            # the absolute highest values.
            self.assertEqual(hi, float("inf"))
            selected = explorer._active_block_values(explorer.speed)
            np.testing.assert_array_equal(
                explorer.plots["count"].y, (selected >= lo).sum(1))

            # Dragging lo off the bottom edge re-opens that side.
            ylo, _, _ = dist._line_ys()
            QTest.mousePress(dist, Qt.MouseButton.LeftButton,
                             pos=QPoint(x, int(ylo)))
            QTest.mouseMove(dist, pos=QPoint(x, int(r.bottom()) + 1))
            QTest.mouseRelease(dist, Qt.MouseButton.LeftButton,
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

    def test_nonempty_clump_drives_positive_detection_channel(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            # Wide-open band: each 1x2 tile is one gap=1 clump of weighted size 2,
            # in both frames, so the largest-clump series is a flat 2 -- and
            # detection fires every frame. There is no min-size gate anymore:
            # any non-empty clump is a positive detection.
            explorer._recompute_clump()
            np.testing.assert_array_equal(explorer.plots["clump"].y, [2, 2])
            np.testing.assert_array_equal(explorer.plots["detect"].y, [1, 1])
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
            explorer.plots["dist"].band_lo = 3.0

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
            selected = explorer._active_block_values(explorer.speed)
            np.testing.assert_array_equal(
                selected, explorer.speed[:, 0:1, 0:2].reshape(2, -1))
            lo, hi = explorer._band()
            np.testing.assert_array_equal(
                explorer.plots["count"].y,
                ((selected >= lo) & (selected <= hi)).sum(1))

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
            explorer.plots["dist"].band_lo = 2.5
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


if __name__ == "__main__":
    unittest.main()
