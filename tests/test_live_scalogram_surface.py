from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtWidgets import QApplication

from gui.explorers.live_scalogram_surface import LiveScalogramSurface
from tests.test_channel_source import _write_moving_square


class LiveScalogramSurfaceBlockTests(unittest.TestCase):
    """A Block change is block-independent downstream of the per-pixel tensor
    solve, so it must re-reduce the cached block=1 channels rather than decode
    and solve the window again."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls._dir = tempfile.mkdtemp(prefix="live_surface_")
        cls.video = os.path.join(cls._dir, "moving.mp4")
        _write_moving_square(cls.video)

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove(cls.video)
            os.rmdir(cls._dir)
        except OSError:
            pass

    def _surface(self):
        reps = [{"id": 0, "label": "all", "frac": (0.0, 0.0, 1.0, 1.0)}]
        # singleShot is patched out so constructing the surface does not kick off
        # the opening extract pass.
        with patch("gui.explorers.live_scalogram_surface.QTimer.singleShot"):
            surface = LiveScalogramSurface(self.video, reps)
        self.addCleanup(surface.close)
        surface._show_channel_data = MagicMock()
        surface.extract = MagicMock()
        return surface

    def test_block_change_re_reduces_cached_pixel_channels(self):
        surface = self._surface()
        start, n = surface._window()
        cached = object()
        surface._pp = cached
        surface._pp_key = surface._pp_signature(surface._build_cfg(), start, n)

        reduced = object()
        with patch("gui.explorers.live_scalogram_surface.reduce_channel_data",
                   return_value=reduced) as reduce:
            surface.block_spin.setValue(5)
            surface._on_block_changed()

        surface.extract.assert_not_called()
        self.assertIs(reduce.call_args.args[0], cached)
        self.assertEqual(reduce.call_args.args[1].flow.block_size, 5)
        surface._show_channel_data.assert_called_once_with(reduced)

    def test_block_change_re_extracts_when_cache_misses_the_window(self):
        surface = self._surface()
        surface._pp = object()
        surface._pp_key = ("stale",)      # signature from a different window

        surface.block_spin.setValue(5)
        surface._on_block_changed()

        surface.extract.assert_called_once()
        surface._show_channel_data.assert_not_called()

    def test_block_change_is_ignored_while_a_pass_is_running(self):
        surface = self._surface()
        start, n = surface._window()
        surface._pp = object()
        surface._pp_key = surface._pp_signature(surface._build_cfg(), start, n)
        surface._worker = MagicMock()

        surface.block_spin.setValue(5)
        surface._on_block_changed()

        surface.extract.assert_not_called()
        surface._show_channel_data.assert_not_called()
        surface._worker = None


if __name__ == "__main__":
    unittest.main()
