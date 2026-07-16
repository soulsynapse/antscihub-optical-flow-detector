from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtWidgets import QApplication

from gui.structure_tensor_explorer import StructureTensorExplorer
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
        with patch("gui.variance_explorer.load_or_extract_channels",
                   return_value=_channels()):
            explorer = StructureTensorExplorer(_CacheStub())
        try:
            self.assertIn("Structure tensor explorer", explorer.windowTitle())
            self.assertEqual(explorer.n_blocks, 4)
            self.assertTrue(all(
                plot.minimumHeight() == 132 and plot.maximumHeight() == 132
                for plot in explorer.plots.values()))
            self.assertGreater(explorer.change_floor, 0.0)
            self.assertTrue(np.isfinite(explorer.threshold))
            np.testing.assert_allclose(
                explorer.plots["variance"].y, [0.0, 1.0, 1.0, 1.0])
            np.testing.assert_allclose(
                explorer.plots["tensor_speed"].y, [0.0, 1.0, 2.5, 3.0])
            np.testing.assert_allclose(
                explorer.plots["cached_speed"].y, [0.0, 1.0, 2.0, 3.0])
            self.assertGreaterEqual(
                explorer.overlay_mode.findText("Tensor speed"), 0)
            self.assertGreaterEqual(
                explorer.overlay_mode.findText("Relative speed disagreement"), 0)
            explorer.overlay_mode.setCurrentText("Tensor speed")
            tensor_scale = explorer._overlay_scale()
            explorer.overlay_mode.setCurrentText("Cached flow speed")
            self.assertEqual(explorer._overlay_scale(), tensor_scale)
            self.assertEqual(
                explorer.detect_combo.currentText(), "appearance fraction")
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
