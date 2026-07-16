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
            np.testing.assert_array_equal(explorer.plots["max"].y, [4, 5])

            # A band that spans the whole value range accepts every owned block.
            explorer.plots["mean"].band_lo = 0.0
            explorer.plots["mean"].band_hi = 1e9
            explorer._recompute_threshold_series()
            explorer._recompute_clump()
            np.testing.assert_array_equal(explorer.plots["count"].y, [4, 4])
            # The two tiles are independent 1x2 components, never one 1x4 clump.
            np.testing.assert_array_equal(explorer.plots["clump"].y, [2, 2])
        finally:
            explorer.close()

    def test_temporal_signal_matches_structure_tensor_window_contract(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            explorer.win_frames = 2
            explorer._recompute_temporal_series()

            # Spatial means are 2.5 then 3.5; the trailing W=2 window averages
            # [frame 0] then [frames 0, 1].
            np.testing.assert_allclose(
                explorer.plots["roll_mean"].y, [2.5, 3.0])
            self.assertEqual(explorer.plots["roll_mean"].unit, "px/s")
            owned = explorer._active_block_values(explorer.speed)
            fields = np.stack(list(explorer._iter_windowed_fields(owned)))
            np.testing.assert_allclose(
                explorer.plots["roll_mean"].y, fields.mean(1))
            explorer.detect_on = "windowed"
            np.testing.assert_allclose(
                explorer._detection_field_at(1, owned), fields[1])
            self.assertEqual(
                explorer.temporal_mode_lbl.text(),
                "Trailing-window mean (same contract as structure tensor)")
            self.assertFalse(hasattr(explorer, "temporal_mode_combo"))
        finally:
            explorer.close()

    def test_window_is_trailing_and_never_reads_future_frames(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            explorer.win_frames = 3
            explorer._recompute_temporal_series()

            # With only two frames, W=3 uses one sample at frame 0 and two at
            # frame 1. It neither centers the window nor repeats edge frames.
            np.testing.assert_allclose(
                explorer.plots["roll_mean"].y, [2.5, 3.0])
            hi, lo, neff = explorer._window_bounds(3)
            np.testing.assert_array_equal(hi, [1, 2])
            np.testing.assert_array_equal(lo, [0, 0])
            np.testing.assert_array_equal(neff, [1, 2])

            left_tile = explorer.speed[:, 0:1, 0:2]
            np.testing.assert_array_equal(
                explorer._detection_field_at(0, left_tile), [[1.0, 2.0]])
        finally:
            explorer.close()

    def test_detect_on_switches_thresholding_to_the_windowed_block_field(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            self.assertFalse(hasattr(explorer, "detect_combo"))
            self.assertFalse(hasattr(explorer, "detect_header"))
            check_index = explorer.plot_col.indexOf(
                explorer.detect_checks["per_frame"])
            plot_index = explorer.plot_col.indexOf(explorer.plots["mean"])
            self.assertEqual(explorer.plot_col.getItemPosition(check_index)[1], 0)
            self.assertEqual(explorer.plot_col.getItemPosition(plot_index)[1], 1)
            self.assertTrue(explorer.detect_checks["per_frame"].isChecked())
            self.assertFalse(explorer.detect_checks["windowed"].isChecked())
            explorer.win_frames = 2

            # Per-frame channel: the band gates the raw per-block speeds.
            explorer.plots["mean"].band_lo = 3.6
            explorer.plots["mean"].band_hi = 1e9
            explorer._recompute_threshold_series()
            explorer._recompute_clump()
            np.testing.assert_array_equal(explorer.plots["count"].y, [1, 2])
            np.testing.assert_array_equal(explorer.plots["clump"].y, [1, 2])

            # Windowed channel: the band gates the trailing-window block field.
            explorer.detect_checks["windowed"].setChecked(True)
            explorer.plots["roll_mean"].band_lo = 3.6
            explorer.plots["roll_mean"].band_hi = 1e9
            explorer._recompute_threshold_series()
            explorer._recompute_clump()

            self.assertEqual(explorer.detect_on, "windowed")
            self.assertFalse(explorer.detect_checks["per_frame"].isChecked())
            self.assertTrue(explorer.detect_checks["windowed"].isChecked())
            np.testing.assert_array_equal(explorer.plots["count"].y, [1, 1])
            np.testing.assert_array_equal(explorer.plots["clump"].y, [1, 1])
            self.assertEqual(explorer.plots["cond_mean"].unit, "px/s")
        finally:
            explorer.close()

    def test_selected_plot_owns_the_band_and_expands(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            mean = explorer.plots["mean"]
            roll = explorer.plots["roll_mean"]
            # The initially selected channel is expanded and carries the band,
            # seeded to its own data range; unselected plots do not.
            self.assertTrue(mean.band_active)
            self.assertEqual(mean.maximumHeight(), MiniPlot.EXPANDED_H)
            self.assertEqual(roll.maximumHeight(), MiniPlot.BASE_H)
            self.assertFalse(roll.band_active)
            # A fresh band is wide open: unbounded on both sides, so per-block
            # values above the plotted mean series' max are still accepted.
            self.assertEqual(explorer._band(), (float("-inf"), float("inf")))

            # A non-sweep sparkline never grows a band.
            self.assertFalse(explorer.plots["count"].band_active)

            # Switching channel moves the band + expansion to the new plot.
            explorer.detect_checks["windowed"].setChecked(True)
            self.assertFalse(mean.band_active)
            self.assertEqual(mean.maximumHeight(), MiniPlot.BASE_H)
            self.assertTrue(roll.band_active)
            self.assertEqual(roll.maximumHeight(), MiniPlot.EXPANDED_H)
            self.assertIs(explorer._selected_plot(), roll)
        finally:
            explorer.close()

    def test_dragging_a_band_line_updates_detection(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            mean = explorer.plots["mean"]
            mean.resize(200, MiniPlot.EXPANDED_H)
            committed = QSignalSpy(mean.band_committed)

            # Grab the maximum handle at its current pixel row and drag it down
            # to just above the fastest per-block speed's neighbourhood.
            r = mean._plot_rect()
            data_lo, data_hi = mean._data_range()
            _, yhi, _ = mean._line_ys()
            QTest.mousePress(mean, Qt.MouseButton.LeftButton,
                             pos=QPoint(int(r.right()) - 2, int(yhi)))
            target = mean._y_of(3.0, r, data_lo, data_hi)
            QTest.mouseMove(mean, pos=QPoint(int(r.right()) - 2, int(target)))
            QTest.mouseRelease(mean, Qt.MouseButton.LeftButton,
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
            mean = explorer.plots["mean"]
            mean.resize(200, MiniPlot.EXPANDED_H)
            r = mean._plot_rect()
            data_lo, data_hi = mean._data_range()

            # Fresh band accepts the fastest per-block speeds (4 and 5) even
            # though the plotted mean series tops out at 3.5 -- the original
            # bug clamped hi to the plot max and silently rejected them.
            explorer._recompute_threshold_series()
            np.testing.assert_array_equal(explorer.plots["count"].y, [4, 4])

            # Drag the lo handle up into the plot: a finite lower threshold.
            x = int(r.right()) - 2
            ylo, _, _ = mean._line_ys()
            QTest.mousePress(mean, Qt.MouseButton.LeftButton,
                             pos=QPoint(x, int(ylo)))
            target = mean._y_of(3.0, r, data_lo, data_hi)
            QTest.mouseMove(mean, pos=QPoint(x, int(target)))
            QTest.mouseRelease(mean, Qt.MouseButton.LeftButton,
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
            ylo, _, _ = mean._line_ys()
            QTest.mousePress(mean, Qt.MouseButton.LeftButton,
                             pos=QPoint(x, int(ylo)))
            QTest.mouseMove(mean, pos=QPoint(x, int(r.bottom()) + 1))
            QTest.mouseRelease(mean, Qt.MouseButton.LeftButton,
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

    def test_size_gate_drives_positive_detection_channel(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            # Wide-open band: each 1x2 tile is one gap=1 clump of weighted size 2,
            # in both frames, so the largest-clump series is a flat 2.
            explorer._recompute_clump()
            np.testing.assert_array_equal(explorer.plots["clump"].y, [2, 2])

            # Size gate at 2 fires every frame; at 3 nothing reaches it. Moving
            # the gate only re-gates -- it never re-clusters.
            explorer.clump_min = 2
            explorer._recompute_detect()
            np.testing.assert_array_equal(explorer.plots["detect"].y, [1, 1])
            explorer.clump_min = 3
            explorer._recompute_detect()
            np.testing.assert_array_equal(explorer.plots["detect"].y, [0, 0])

            # The gate is a spatial channel: it vanishes on a scalar detection.
            explorer.detect_checks["scalar:p90"].setChecked(True)
            self.assertTrue(explorer.plots["detect"].isHidden())
            self.assertFalse(explorer.gap_slider.isEnabled())
            self.assertFalse(explorer.size_slider.isEnabled())
        finally:
            explorer.close()

    def test_scalar_plot_detection_is_binary_and_highlights_whole_replicate(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            expected_targets = {
                "per_frame", "scalar:median", "scalar:p90", "scalar:p99",
                "scalar:max", "scalar:sstd", "scalar:peak", "windowed",
                "scalar:roll_std",
            }
            self.assertEqual(set(explorer.detect_checks), expected_targets)

            explorer.detect_checks["scalar:p90"].setChecked(True)
            explorer.plots["p90"].band_lo = 4.0
            explorer.plots["p90"].band_hi = 1e9
            explorer._recompute_threshold_series()

            # P90 is 3.7 then 4.7 across the four owned blocks; only 4.7 is in band.
            np.testing.assert_array_equal(explorer.plots["count"].y, [0, 1])
            self.assertEqual(explorer.plots["count"].unit, "0/1")
            self.assertIn("90th pct speed", explorer.plots["count"].title)
            for key in ("frac", "clump", "cond_mean", "energy"):
                self.assertTrue(explorer.plots[key].isHidden())

            explorer.overlay_mode.setCurrentText("Raw frame")
            explorer.frame = 0
            explorer._redraw_video()
            quiet = explorer.video_view._pix.toImage()
            self.assertEqual(quiet.pixelColor(25, 25).getRgb()[:3], (0, 0, 0))

            explorer.frame = 1
            explorer._redraw_video()
            detected = explorer.video_view._pix.toImage()
            for x in (25, 75):
                r, g, b = detected.pixelColor(x, 25).getRgb()[:3]
                self.assertGreater(g, r)
                self.assertGreater(g, b)

            explorer.detect_checks["per_frame"].setChecked(True)
            self.assertEqual(explorer.plots["count"].unit, "blk")
            for key in ("frac", "clump", "cond_mean", "energy"):
                self.assertFalse(explorer.plots[key].isHidden())
        finally:
            explorer.close()

    def test_replicate_click_focuses_data_and_reseeds_band_to_that_scope(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            speed_scale = explorer.vmax
            unbounded = (float("-inf"), float("inf"))
            # The overview band seeds wide open; place a finite threshold so the
            # focus switch below demonstrably wipes it.
            self.assertEqual(explorer._band(), unbounded)
            explorer.plots["mean"].band_lo = 3.0

            explorer._redraw_video()
            explorer._on_video_clicked(QPoint(25, 25))
            self.assertEqual(explorer.active_region_index, 0)
            self.assertEqual(explorer.n_blocks, 2)
            self.assertEqual(explorer.video_view.focus_frac,
                             (0.0, 0.0, 0.5, 1.0))
            # Focusing a replicate discards the pooled-scope threshold: the band
            # reseeds wide open rather than staying frozen on the old scale.
            self.assertEqual(explorer._band(), unbounded)
            self.assertEqual(explorer.vmax, speed_scale)
            np.testing.assert_array_equal(explorer.plots["max"].y, [2, 3])
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
            # Returning to the overview re-seeds wide open again.
            self.assertEqual(explorer._band(), unbounded)
            self.assertEqual(explorer.vmax, speed_scale)
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
