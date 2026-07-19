"""Batch E: the detection readout (T15, T16), plus T27/T28.

These assert PIXELS, not state flags. Both features are pictures of the
detection gate, and both fail in the same silent direction -- a badge that does
not draw, or a detection span rounded below one pixel, reads to the user as
"nothing was detected" while the detector is firing. A test asserting only that
``_detected`` is True, or that the mask was stored, passes in exactly that case.
"""
from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QApplication

from gui.explorers.speed_explorer import MiniPlot
from gui.video_panel import FrameView

_APP = QApplication.instance() or QApplication([])

DETECT_G = (60, 220, 110)


def _render(w) -> np.ndarray:
    """Paint a widget into an (h, w, 3) uint8 RGB array."""
    img = QImage(w.width(), w.height(), QImage.Format.Format_RGB888)
    img.fill(0)
    p = QPainter(img)
    w.render(p)
    p.end()
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    arr = np.frombuffer(ptr, np.uint8).reshape(img.height(),
                                               img.bytesPerLine() // 3, 3)
    return arr[:, :img.width()].copy()


def _has_badge(px: np.ndarray) -> bool:
    """A run of the badge's exact fill colour. Exact, because the badge is a
    flat fill with no alpha -- anything approximate would also match the
    translucent detection spans."""
    return bool((np.abs(px.astype(int) - np.array(DETECT_G)).sum(2) == 0).any())


class DetectedBadgeTest(unittest.TestCase):
    """T16."""

    def _view(self) -> FrameView:
        v = FrameView()
        v.resize(320, 240)
        v.set_frame(np.full((120, 160, 3), 40, np.uint8))
        return v

    def test_badge_absent_when_not_detected(self):
        v = self._view()
        self.assertFalse(_has_badge(_render(v)), "badge drawn with no detection")

    def test_badge_drawn_on_detection(self):
        v = self._view()
        v.set_detected(True)
        self.assertTrue(_has_badge(_render(v)), "badge missing on a detection")

    def test_badge_survives_the_shift_held_peek(self):
        """The whole point of T16's "must survive" clause.

        Shift-to-peek hides annotations so the raw pixels can be judged -- which
        is precisely when the user is deciding whether a detection is real. The
        boxes must go; the verdict must not.
        """
        v = self._view()
        v.set_boxes([(0.1, 0.1, 0.5, 0.5, "r1", "#ff0000", False)])
        v.set_detected(True)
        v.set_overlays_hidden(True)
        px = _render(v)
        self.assertTrue(_has_badge(px), "badge vanished while Shift was held")
        red = (px[:, :, 0] > 150) & (px[:, :, 1] < 80) & (px[:, :, 2] < 80)
        self.assertFalse(red.any(),
                         "peek failed to hide the boxes; test proves nothing")

    def test_badge_clears(self):
        v = self._view()
        v.set_detected(True)
        v.set_detected(False)
        self.assertFalse(_has_badge(_render(v)), "badge stuck on after clearing")


class BadgeMatchesSpanColourTest(unittest.TestCase):
    """The badge and the span shading report the same gate on two widgets that
    cannot import each other (speed_explorer imports video_panel). Two greens
    drifting apart would read as two different states."""

    def test_the_two_greens_are_one_colour(self):
        from gui.explorers.speed_explorer import DETECT
        from gui.video_panel import DETECT_BADGE
        self.assertEqual(DETECT.getRgb(), DETECT_BADGE.getRgb())


class DetectionSpanTest(unittest.TestCase):
    """T15."""

    def _plot(self, n: int, mask: np.ndarray) -> MiniPlot:
        pl = MiniPlot("t")
        pl.resize(400, MiniPlot.BASE_H)
        pl.set_series(np.linspace(0, 1, n, dtype=np.float32))
        pl.set_detect_mask(mask)
        return pl

    @staticmethod
    def _green_cols(px: np.ndarray) -> np.ndarray:
        """Columns tinted by the span shading.

        Two discriminators, both needed. Green must lead blue, which excludes
        the cyan series line (120,215,255). And the tint must cover most of the
        column, because a span is a full-height fill -- a lone pixel is the
        antialiased edge of the axis text or the polyline, which a per-pixel
        ``.any()`` reports as a detection.
        """
        r, g, b = px[:, :, 0].astype(int), px[:, :, 1].astype(int), \
            px[:, :, 2].astype(int)
        tinted = (g > r + 8) & (g > b + 5)
        return np.flatnonzero(tinted.sum(0) >= 0.5 * px.shape[0])

    def test_no_spans_without_a_mask(self):
        pl = self._plot(500, np.zeros(500, bool))
        self.assertEqual(self._green_cols(_render(pl)).size, 0)

    def test_span_is_drawn(self):
        n = 500
        mask = np.zeros(n, bool)
        mask[200:300] = True
        cols = self._green_cols(_render(self._plot(n, mask)))
        self.assertGreater(cols.size, 20, "a 100-frame detection barely drew")

    def test_single_frame_detection_survives_a_long_clip(self):
        """The load-bearing one. One positive frame in 14000 is 0.03 px wide at
        this widget size; without the 1 px floor it rounds to nothing and the
        panel silently shows a clean clip."""
        n = 14000
        mask = np.zeros(n, bool)
        mask[7000] = True
        cols = self._green_cols(_render(self._plot(n, mask)))
        self.assertGreater(cols.size, 0,
                           "a single-frame detection rounded away to nothing")

    def test_run_touching_the_end_is_drawn(self):
        """A detection running to the final frame has no falling edge; a naive
        diff drops it, and the end of a clip is where an event often sits."""
        n = 500
        mask = np.zeros(n, bool)
        mask[450:] = True
        self.assertGreater(self._green_cols(_render(self._plot(n, mask))).size,
                           10, "a detection at the clip end was not drawn")

    def test_a_new_series_clears_the_previous_mask(self):
        """Batch D's empty-rather-than-stale rule. Switching replicate sets a
        new series; a mask held across it would shade the new clip with the old
        one's detections -- wrong in the direction that INVENTS events, and
        indistinguishable from a real result."""
        n = 500
        mask = np.zeros(n, bool)
        mask[100:400] = True
        pl = self._plot(n, mask)
        self.assertGreater(self._green_cols(_render(pl)).size, 10,
                           "precondition: spans drew at all")
        pl.set_series(np.linspace(0, 1, n, dtype=np.float32))
        self.assertEqual(self._green_cols(_render(pl)).size, 0,
                         "stale detections shaded onto a new series")

    def test_mask_shorter_than_series_does_not_raise(self):
        pl = self._plot(500, np.ones(200, bool))
        self.assertGreater(self._green_cols(_render(pl)).size, 10)


class AutoCollapseEmptyTest(unittest.TestCase):
    """T27 -- lifted from DensityPlot to MiniPlot, since the sweep plots carry a
    series rather than a matrix."""

    def test_line_plot_collapses_when_its_series_is_empty(self):
        pl = MiniPlot("t")
        pl.set_collapsible(True)
        pl.set_auto_collapse_empty(True)
        self.assertTrue(pl.is_collapsed(), "empty sweep plot drew its black slab")
        self.assertEqual(pl.maximumHeight(), MiniPlot.COLLAPSED_H)

    def test_it_opens_when_a_series_arrives(self):
        pl = MiniPlot("t")
        pl.set_auto_collapse_empty(True)
        pl.set_series(np.ones(10, np.float32))
        self.assertFalse(pl.is_collapsed())

    def test_it_recollapses_when_the_series_goes_away(self):
        pl = MiniPlot("t")
        pl.set_auto_collapse_empty(True)
        pl.set_series(np.ones(10, np.float32))
        pl.set_series(np.zeros(0, np.float32))
        self.assertTrue(pl.is_collapsed())

    def test_auto_collapse_does_not_clear_user_intent(self):
        """Batch D's two-state rule, re-checked on the new code path: data
        arriving must not re-open a plot the user deliberately shut."""
        pl = MiniPlot("t")
        pl.set_auto_collapse_empty(True)
        pl.set_collapsed(True)
        pl.set_series(np.ones(10, np.float32))
        self.assertTrue(pl.is_collapsed(), "arriving data overrode the user's [+]")


if __name__ == "__main__":
    unittest.main()
