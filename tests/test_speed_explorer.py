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

from gui.speed_explorer import SpeedExplorer, _regions_from_meta
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

            explorer.threshold = 0.5
            explorer._recompute_threshold_series()
            explorer._recompute_clump()
            np.testing.assert_array_equal(explorer.plots["count"].y, [4, 4])
            # The two tiles are independent 1x2 components, never one 1x4 clump.
            np.testing.assert_array_equal(explorer.plots["clump"].y, [2, 2])
        finally:
            explorer.close()

    def test_replicate_click_focuses_data_and_right_click_returns_to_overview(self):
        explorer = SpeedExplorer(_CacheStub())
        try:
            explorer.thr_slider.setValue(400)
            threshold = explorer.threshold
            speed_scale = explorer.vmax

            explorer._redraw_video()
            explorer._on_video_clicked(QPoint(25, 25))
            self.assertEqual(explorer.active_region_index, 0)
            self.assertEqual(explorer.n_blocks, 2)
            self.assertEqual(explorer.video_view.focus_frac,
                             (0.0, 0.0, 0.5, 1.0))
            self.assertEqual(explorer.thr_slider.value(), 400)
            self.assertEqual(explorer.threshold, threshold)
            self.assertEqual(explorer.vmax, speed_scale)
            np.testing.assert_array_equal(explorer.plots["max"].y, [2, 3])
            selected = explorer._active_block_values(explorer.speed)
            np.testing.assert_array_equal(
                selected, explorer.speed[:, 0:1, 0:2].reshape(2, -1))
            np.testing.assert_array_equal(
                explorer.plots["count"].y, (selected > threshold).sum(1))

            explorer.focus_mode.setCurrentIndex(1)
            flow_preview = explorer._base_frame()
            self.assertEqual(flow_preview.shape, (4, 8, 3))
            np.testing.assert_array_equal(flow_preview[..., 0],
                                          flow_preview[..., 1])

            explorer._clear_region_focus()
            self.assertEqual(explorer.active_region_index, -1)
            self.assertEqual(explorer.n_blocks, 4)
            self.assertIsNone(explorer.video_view.focus_frac)
            self.assertEqual(explorer.thr_slider.value(), 400)
            self.assertEqual(explorer.threshold, threshold)
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
