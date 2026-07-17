from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtWidgets import QApplication

from gui.explorers.structure_tensor_explorer import StructureTensorExplorer
from gui.explorers.speed_explorer import MiniPlot
from gui.explorers.variance_explorer import DETECT_TARGETS
from core.tensor_channels import extract_channels


class _CacheStub:
    def __init__(self):
        self.meta = {
            "video_path": "missing.mp4",
            "backend": "stub",
            "fps": 2.0,
            "n_frames": 4,
            "block_size": 4,
            "grid": [3, 2],
            "src_width": 100,
            "src_height": 50,
            "work_width": 8,
            "work_height": 12,
            "downsample": 1.0,
            "features": ["u", "v", "speed"],
            "replicate_tiles": [
                {
                    "id": 10,
                    "label": "left",
                    "frac": [0.0, 0.0, 0.5, 1.0],
                    "source_box": [0, 0, 50, 50],
                    "work_width": 8,
                    "work_height": 4,
                    "grid": [1, 2],
                    "atlas_bbox": [0, 0, 1, 2],
                },
                {
                    "id": 11,
                    "label": "right",
                    "frac": [0.5, 0.0, 1.0, 1.0],
                    "source_box": [50, 0, 100, 50],
                    "work_width": 8,
                    "work_height": 4,
                    "grid": [1, 2],
                    "atlas_bbox": [2, 0, 3, 2],
                },
            ],
        }
        self._speed = np.array([
            [[0, 0], [999, 999], [0, 0]],
            [[1, 1], [999, 999], [1, 1]],
            [[2, 2], [999, 999], [2, 2]],
            [[3, 3], [999, 999], [3, 3]],
        ], dtype=np.float32)

    def read(self, name: str) -> np.ndarray:
        if name == "speed":
            return self._speed.copy()
        raise KeyError(name)


def _channels() -> dict:
    # The middle row is an unowned sparse-atlas separator.  Large values make
    # accidental pooling immediately visible in the variance trace.
    intensity = np.array([
        [[0, 0], [999, 999], [0, 0]],
        [[2, 2], [999, 999], [2, 2]],
        [[4, 4], [999, 999], [4, 4]],
        [[6, 6], [999, 999], [6, 6]],
    ], dtype=np.float32)
    change = np.array([
        [[0, 0], [999, 999], [0, 0]],
        [[2, 2], [999, 999], [2, 2]],
        [[4, 4], [999, 999], [4, 4]],
        [[8, 8], [999, 999], [8, 8]],
    ], dtype=np.float32)
    appearance = change * 0.5
    texture = np.ones_like(intensity)
    tensor_speed = np.array([
        [[0, 0], [999, 999], [0, 0]],
        [[1, 1], [999, 999], [1, 1]],
        [[2.5, 2.5], [999, 999], [2.5, 2.5]],
        [[3, 3], [999, 999], [3, 3]],
    ], dtype=np.float32)
    return {
        "intensity": intensity,
        "change": change,
        "appearance": appearance,
        "texture": texture,
        "tensor_speed": tensor_speed,
        "meta": {"approximated": False},
    }


class StructureTensorExplorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_constructs_and_uses_only_owned_blocks(self):
        with patch("gui.explorers.variance_explorer.load_or_extract_channels",
                   return_value=_channels()):
            explorer = StructureTensorExplorer(_CacheStub())
        try:
            self.assertIn("Structure tensor explorer", explorer.windowTitle())
            self.assertEqual(explorer.n_blocks, 4)
            # The selected detection channel (appearance ENERGY -- the fraction
            # is an overlay-only diagnostic) expands and carries the band; all
            # other plots stay at their class's base height without a band.
            appear_w = explorer.plots["appear_w"]
            self.assertTrue(appear_w.band_active)
            self.assertEqual(appear_w.maximumHeight(), MiniPlot.EXPANDED_H)
            # The windowed-count gate carries its own always-active band,
            # independent of which channel is selected.
            count_w = explorer.plots["count_w"]
            self.assertTrue(count_w.band_active)
            self.assertEqual(count_w.maximumHeight(), MiniPlot.EXPANDED_H)
            self.assertTrue(all(
                plot.maximumHeight() == type(plot).BASE_H
                and not plot.band_active
                for key, plot in explorer.plots.items()
                if key not in ("appear_w", "count_w")))
            # Wide-open count band: every frame is a positive detection.
            np.testing.assert_array_equal(explorer.plots["detect"].y, 1.0)
            self.assertGreater(explorer.change_floor, 0.0)
            # A fresh band is wide open on both sides: an unset threshold accepts
            # every per-block value, including those above the plotted series max.
            self.assertEqual(explorer._band(), (float("-inf"), float("inf")))
            # Density-heatmap channels expose the per-frame MAX over owned
            # blocks as their readout series; the stub's owned blocks are
            # uniform per frame, so these are also the exact channel values.
            np.testing.assert_allclose(
                explorer.plots["variance"].y, [0.0, 1.0, 1.0, 1.0])
            # appearance = change * 0.5 -> owned per-frame [0, 1, 2, 4];
            # centered windows lean backward for even W, so W=2 covers
            # [t-1, t] and the means are [0, 0.5, 1.5, 3].
            np.testing.assert_allclose(
                explorer.plots["appear_w"].y, [0.0, 0.5, 1.5, 3.0])
            # The change floor comes from OWNED blocks only (median of the
            # positive owned change values [2,4,8] = 4), never the 999 atlas
            # separators, which would inflate it to 8 and gate everything.
            self.assertEqual(explorer.change_floor, 4.0)
            # The fraction plot is gone; the classifier read survives as the
            # "Appearance fraction" overlay, gated to 0 below the change floor.
            self.assertNotIn("frac", explorer.plots)
            self.assertGreaterEqual(
                explorer.overlay_mode.findText("Appearance fraction"), 0)
            # Raw speed traces are gone; the flow audit keeps the difference
            # reads. |tensor - cached| owned means: |[0,1,2.5,3]-[0,1,2,3]|.
            self.assertNotIn("tensor_speed", explorer.plots)
            self.assertNotIn("cached_speed", explorer.plots)
            np.testing.assert_allclose(
                explorer.plots["speed_absdiff"].y, [0.0, 0.0, 0.5, 0.0])
            self.assertGreaterEqual(
                explorer.overlay_mode.findText("Tensor speed"), 0)
            self.assertGreaterEqual(
                explorer.overlay_mode.findText("Relative speed disagreement"), 0)
            explorer.overlay_mode.setCurrentText("Tensor speed")
            tensor_scale = explorer._overlay_scale()
            explorer.overlay_mode.setCurrentText("Cached flow speed")
            self.assertEqual(explorer._overlay_scale(), tensor_scale)
            self.assertFalse(hasattr(explorer, "detect_combo"))
            self.assertFalse(hasattr(explorer, "thr_slider"))
            self.assertEqual(set(explorer.detect_checks), set(DETECT_TARGETS))
            self.assertTrue(
                explorer.detect_checks["appearance energy"].isChecked())
            # Windows are CENTERED (an event's windowed mass peaks on the
            # event, not W/2 frames later): odd W=3 covers [t-1, t+1],
            # truncated at both clip edges. Owned per-frame appearance
            # [0, 1, 2, 4] -> [0.5, 1, 7/3, 3].
            explorer.win_slider.setValue(3)
            explorer._apply_window_change()
            np.testing.assert_allclose(
                explorer.plots["appear_w"].y, [0.5, 1.0, 7.0 / 3.0, 3.0],
                rtol=1e-5)
            # Unchecking the box switches every windowed read to the causal
            # trailing convention [t-W+1, t]: same data, mass W/2 later.
            explorer.centered_chk.setChecked(False)
            np.testing.assert_allclose(
                explorer.plots["appear_w"].y, [0.0, 0.5, 1.0, 7.0 / 3.0],
                rtol=1e-5)
        finally:
            explorer.close()

    def test_detect_checkbox_switches_channel_and_band_gates_detection(self):
        with patch("gui.explorers.variance_explorer.load_or_extract_channels",
                   return_value=_channels()):
            explorer = StructureTensorExplorer(_CacheStub())
        try:
            # Default selection ships with a matching overlay and vs-W density.
            self.assertEqual(explorer.overlay_mode.currentText(),
                             "Appearance energy")
            explorer.detect_checks["change energy"].setChecked(True)
            self.assertEqual(explorer.detect, "change energy")
            self.assertFalse(
                explorer.detect_checks["appearance energy"].isChecked())
            change_w = explorer.plots["change_w"]
            self.assertTrue(change_w.band_active)
            self.assertEqual(change_w.maximumHeight(), MiniPlot.EXPANDED_H)
            self.assertFalse(explorer.plots["appear_w"].band_active)
            # Overlay and the value-vs-W density both follow the selection.
            self.assertEqual(explorer.overlay_mode.currentText(),
                             "Change energy Jtt")
            self.assertIn("Change energy", explorer.plots["vw_sel"].title)

            # Owned change energy is uniform per frame ([0, 2, 4, 8]); with the
            # default W=2 trailing window the block field is [0, 1, 3, 6].
            self.assertEqual(explorer.win_frames, 2)
            change_w.band_lo = 2.0
            change_w.band_hi = 1e9
            explorer._recompute_sweep()
            np.testing.assert_array_equal(
                explorer.plots["count"].y, [0, 0, 4, 4])
            # The two replicate tiles are separate 1x2 components, never one
            # pooled clump.
            np.testing.assert_array_equal(
                explorer.plots["clump"].y, [0, 0, 2, 2])
            # Detection window D=2 (fps): centered mean (backward-leaning for
            # even D, so D=2 covers [t-1, t]) of count [0, 0, 4, 4] is
            # [0, 0, 2, 4] -- the first in-band frame is diluted by the quiet
            # frame before it.
            self.assertEqual(explorer.sweep_win, 2)
            np.testing.assert_allclose(
                explorer.plots["count_w"].y, [0.0, 0.0, 2.0, 4.0])
            # A min handle above the diluted value demands SUSTAINED evidence:
            # frame 2's momentary 4-block burst reads 2 after integration and
            # is rejected; only frame 3, where the count has persisted a full
            # window, detects.
            count_w = explorer.plots["count_w"]
            count_w.band_lo, count_w.band_hi = 3.0, float("inf")
            explorer._recompute_detect()
            np.testing.assert_array_equal(
                explorer.plots["detect"].y, [0, 0, 0, 1])
        finally:
            explorer.close()

    def test_region_change_reseeds_band_to_new_scope(self):
        with patch("gui.explorers.variance_explorer.load_or_extract_channels",
                   return_value=_channels()):
            explorer = StructureTensorExplorer(_CacheStub())
        try:
            explorer.detect_checks["change energy"].setChecked(True)
            explorer.plots["change_w"].band_lo = 123.0
            explorer.plots["change_w"].band_hi = 456.0
            idx = explorer.region_combo.findData(0)
            explorer.region_combo.setCurrentIndex(idx)
            # The frozen band must not survive into the new scope; it re-seeds
            # wide open rather than staying on the old scale.
            self.assertEqual(explorer._band(),
                             (float("-inf"), float("inf")))
        finally:
            explorer.close()


class _ExtractionCacheStub:
    def __init__(self):
        self.meta = {
            "video_path": "synthetic.mp4",
            "fps": 10.0,
            "n_frames": 2,
            "block_size": 4,
            "grid": [4, 4],
            "src_width": 16,
            "src_height": 16,
            "work_width": 16,
            "work_height": 16,
            "downsample": 1.0,
            "features": ["u", "v", "speed"],
            "config": {"preprocess": {"downsample": 1.0, "normalize": "off"}},
        }
        self._zero = np.zeros((2, 4, 4), np.float32)

    def read(self, name):
        if name in ("u", "v"):
            return self._zero.copy()
        raise KeyError(name)


class _VideoSourceStub:
    frames = None

    def __init__(self, _path):
        pass

    def iter_frames(self, _start, _end):
        for i, frame in enumerate(self.frames):
            yield i, frame

    def release(self):
        pass


class TensorChannelExtractionTests(unittest.TestCase):
    def test_extracts_nonzero_tensor_speed_from_translated_texture(self):
        rng = np.random.default_rng(4)
        first = rng.integers(20, 236, size=(16, 16), dtype=np.uint8)
        matrix = np.float32([[1, 0, 0.35], [0, 1, -0.25]])
        second = cv2.warpAffine(
            first, matrix, (16, 16), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT)
        _VideoSourceStub.frames = [
            cv2.cvtColor(first, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(second, cv2.COLOR_GRAY2BGR),
        ]

        with patch("core.tensor_channels.VideoSource", _VideoSourceStub):
            channels = extract_channels(_ExtractionCacheStub(), sigma=1.0)

        self.assertEqual(channels["tensor_speed"].shape, (2, 4, 4))
        np.testing.assert_array_equal(channels["tensor_speed"][0], 0.0)
        self.assertGreater(float(channels["tensor_speed"][1].mean()), 0.1)
        self.assertTrue(np.isfinite(channels["tensor_speed"]).all())


if __name__ == "__main__":
    unittest.main()
