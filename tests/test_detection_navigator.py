"""The whole-clip strip at the bottom of the live surface.

Its job is to keep three claims apart -- unexamined, examined-and-quiet, and
examined-under-other-settings -- and to be a usable seeker from frame zero. What
is tested here is the part of that which the bar HEIGHT carries.
"""
from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtWidgets import QApplication

from gui.explorers.detection_timeline import _BASE_ROWS, _GATE_ROWS, _Strip


class _Track:
    """The minimum WholeVideoTrack surface _Strip.set_track reads."""

    def __init__(self, clump, gate=None, current=None, covered=None):
        self.clump = np.asarray(clump, np.float32)
        n = self.clump.size
        self.n_frames = n
        self.fps = 30.0
        self.gate = np.zeros(n, np.float32) if gate is None else np.asarray(gate, np.float32)
        self.current = np.ones(n, bool) if current is None else np.asarray(current, bool)
        self.covered = np.ones(n, bool) if covered is None else np.asarray(covered, bool)


class LogBarHeightTests(unittest.TestCase):
    """The clump series is a block count with a heavy tail. Against a linear
    axis one large event sets the maximum and every ordinary event below it is
    drawn one or two pixels tall -- indistinguishable from examined-and-quiet,
    which is exactly the collapse this strip exists to prevent."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    W, H = 256, 84

    def _bar_heights(self, clump):
        strip = _Strip()
        strip.set_track(_Track(clump))
        rgb = self._rgb(strip)
        top = self.H - _GATE_ROWS - _BASE_ROWS
        # A column's bar height is how many rows above the gate band differ from
        # the examined-but-empty background.
        bg = rgb[0, 0]
        col = rgb[:top, :, :]
        return (np.abs(col.astype(int) - bg.astype(int)).sum(2) > 0).sum(0)

    def _rgb(self, strip):
        img = strip._build_image(self.W, self.H)
        self.assertIsNotNone(img)
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        # .copy() is load-bearing: frombuffer does not own the memory, and `img`
        # is released when this returns, leaving the array pointing at freed
        # bytes that read as plausible-but-wrong pixels.
        return np.frombuffer(ptr, np.uint8).reshape(
            self.H, img.bytesPerLine())[:, : self.W * 3].reshape(
                self.H, self.W, 3).copy()

    def test_a_small_event_under_a_large_one_is_still_visibly_tall(self):
        # 1 against a peak of 10000: linearly that is 0.01% of the plot height,
        # i.e. zero pixels. On a log axis it is a bar you can see.
        clump = np.zeros(self.W, np.float32)
        clump[10] = 1.0
        clump[200] = 10000.0
        h = self._bar_heights(clump)
        top = self.H - _GATE_ROWS - _BASE_ROWS
        self.assertGreater(h[10], 0, "a real event was drawn as nothing")
        # log1p(1)/log1p(10000) is ~7.5%, so it clears a couple of pixels; the
        # linear 0.01% could not.
        self.assertGreater(h[10] / max(1, top), 0.02)
        self.assertGreater(h[200], h[10])

    def test_the_largest_event_still_reaches_the_top(self):
        clump = np.zeros(self.W, np.float32)
        clump[5] = 3.0
        clump[100] = 900.0
        h = self._bar_heights(clump)
        top = self.H - _GATE_ROWS - _BASE_ROWS
        self.assertGreaterEqual(h[100], top - 3)

    def test_a_genuine_zero_stays_a_zero_height_bar(self):
        """log1p, not log: an examined-and-quiet frame must not acquire a bar,
        or the strip claims a detection where the detector said nothing."""
        clump = np.zeros(self.W, np.float32)
        clump[50] = 20.0
        h = self._bar_heights(clump)
        self.assertEqual(int(h[0]), 0)
        self.assertEqual(int(h[120]), 0)

    def test_bar_height_is_monotonic_in_the_clump_value(self):
        clump = np.zeros(self.W, np.float32)
        for i, v in enumerate((1.0, 4.0, 16.0, 64.0, 256.0)):
            clump[i * 20 + 3] = v
        h = self._bar_heights(clump)
        got = [int(h[i * 20 + 3]) for i in range(5)]
        self.assertEqual(got, sorted(got))
        self.assertEqual(len(set(got)), len(got), f"heights collapsed: {got}")


if __name__ == "__main__":
    unittest.main()
