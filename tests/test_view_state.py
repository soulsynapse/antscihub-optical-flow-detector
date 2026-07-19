"""View-state round trip across a live re-extract rebuild (T17).

The live surface captures state off the old explorer and applies it to a freshly
built one. Anything missing from that dict silently reverts to a default on every
whole-video pass -- which is exactly how the two detection threshold bands were
being lost.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

import numpy as np
from PyQt6.QtWidgets import QApplication

from core.channel_source import ChannelData
from gui.explorers.scalogram_explorer import ScalogramExplorer


def _channel_data(T=64, ny=4, nx=4) -> ChannelData:
    rng = np.random.default_rng(0)
    meta = {
        "fps": 30.0,
        "grid": [ny, nx],
        "block_size": 64,
        "n_frames": T,
        "replicate_tiles": [{"id": 0, "label": "r0", "atlas_bbox": [0, 0, ny, nx],
                             "frac": [0.0, 0.0, 1.0, 1.0]}],
    }
    channels = {name: rng.random((T, ny, nx), dtype=np.float32) + 0.1
                for name in ("change", "appearance", "tensor_speed", "intensity")}
    return ChannelData(meta=meta, channels=channels)


class ViewStateRoundTripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _explorer(self):
        ex = ScalogramExplorer.from_channel_data(
            _channel_data(), video_path=None, own_shortcuts=False,
            own_status=False)
        self.addCleanup(self._destroy, ex)
        return ex

    def _destroy(self, ex):
        ex.close()
        ex.deleteLater()
        self.app.processEvents()

    def test_detection_bands_survive_a_rebuild(self):
        old = self._explorer()
        old.channel = "change energy Jtt"
        old.density_plots["change energy Jtt"].band_lo = 0.25
        old.density_plots["change energy Jtt"].band_hi = 0.75
        old.count_w_plot.band_lo = 2.0
        old.count_w_plot.band_hi = 9.0

        new = self._explorer()
        new.apply_view_state(old.capture_view_state())

        self.assertEqual(new.detection_params()["value_band"], (0.25, 0.75))
        self.assertEqual(new.detection_params()["count_band"], (2.0, 9.0))

    def test_unselected_channels_keep_their_bands(self):
        old = self._explorer()
        old.density_plots["intensity"].band_lo = 1.5
        old.density_plots["intensity"].band_hi = 3.5

        new = self._explorer()
        new.apply_view_state(old.capture_view_state())

        dp = new.density_plots["intensity"]
        self.assertEqual((dp.band_lo, dp.band_hi), (1.5, 3.5))

    def test_an_unplaced_band_stays_unplaced(self):
        # Carrying None through as +/-inf would turn "never set" into an explicit
        # unbounded threshold and defeat set_band_active's lazy seeding.
        old = self._explorer()
        old.density_plots["appearance energy"].band_lo = None
        old.density_plots["appearance energy"].band_hi = None

        new = self._explorer()
        new.apply_view_state(old.capture_view_state())

        dp = new.density_plots["appearance energy"]
        self.assertIsNone(dp.band_lo)
        self.assertIsNone(dp.band_hi)

    def test_older_state_without_the_new_keys_still_applies(self):
        # A dict captured before this fix must not raise.
        new = self._explorer()
        new.apply_view_state({"channel": "change energy Jtt", "frame": 3, "sweep_win": 5,
                              "centered": True, "freq_band": (1.0, 4.0)})
        self.assertEqual(new.channel, "change energy Jtt")


if __name__ == "__main__":
    unittest.main()
