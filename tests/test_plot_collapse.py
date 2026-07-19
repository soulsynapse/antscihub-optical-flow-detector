"""Batch D: the [+] collapse mechanism on the MiniPlot family (T13, T14, T10).

The point of collapsing is not cosmetic -- it is that a collapsed plot performs
NO work. These tests therefore assert the absence of work (paint calls, cached
images, band-power sums) rather than only the resulting geometry, because a
collapse that still paints would look identical in a screenshot and would fix
nothing about T10.
"""
from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QSignalSpy, QTest
from PyQt6.QtWidgets import QApplication

from gui.explorers.scalogram_explorer import ScalogramPlot
from gui.explorers.speed_explorer import DensityPlot, MiniPlot, PixelBarPlot


# Module-level, and deliberately so: a QApplication that only a local holds is
# garbage collected the moment the function returns, and the next QWidget dies
# with "Must construct a QApplication before a QWidget" -- as a hard abort, not
# an exception, so pytest reports a bare exit 9 with no failing test named.
_APP = QApplication.instance() or QApplication([])


def _app():
    return _APP


class CollapseGeometryTest(unittest.TestCase):
    def setUp(self):
        _app()

    def test_off_by_default_so_existing_explorers_are_unchanged(self):
        """Opt-in. speed_explorer and variance_explorer share MiniPlot and never
        call set_collapsible, so their geometry must be bit-identical to before."""
        pl = MiniPlot("t")
        self.assertFalse(pl.is_collapsed())
        self.assertEqual(pl.maximumHeight(), MiniPlot.BASE_H)
        pl.set_expanded(True)
        self.assertEqual(pl.maximumHeight(), MiniPlot.EXPANDED_H)

    def test_collapsed_height_is_header_only(self):
        pl = MiniPlot("t")
        pl.set_collapsible(True)
        pl.set_collapsed(True)
        self.assertEqual(pl.maximumHeight(), MiniPlot.COLLAPSED_H)

    def test_collapse_wins_over_expanded(self):
        """set_expanded and set_collapsed are independent axes. Collapsing an
        expanded plot must actually shrink it, and un-collapsing must restore
        the EXPANDED height -- not silently drop back to BASE_H, which would
        make the selected detection channel quietly lose its double height."""
        pl = MiniPlot("t")
        pl.set_expanded(True)
        pl.set_collapsed(True)
        self.assertEqual(pl.maximumHeight(), MiniPlot.COLLAPSED_H)
        pl.set_collapsed(False)
        self.assertEqual(pl.maximumHeight(), MiniPlot.EXPANDED_H)

    def test_expanded_state_set_while_collapsed_is_remembered(self):
        pl = MiniPlot("t")
        pl.set_collapsed(True)
        pl.set_expanded(True)
        self.assertEqual(pl.maximumHeight(), MiniPlot.COLLAPSED_H)
        pl.set_collapsed(False)
        self.assertEqual(pl.maximumHeight(), MiniPlot.EXPANDED_H)


class CollapsedDoesNoWorkTest(unittest.TestCase):
    """The load-bearing half of T14/T10."""

    def setUp(self):
        _app()

    def test_collapsed_plot_does_not_paint(self):
        pl = MiniPlot("t")
        pl.set_series(np.arange(100, dtype=np.float32))
        calls = []
        pl._data_range = lambda *a, **k: (calls.append(1) or (0.0, 1.0))

        pl.set_collapsible(True)
        pl.set_collapsed(True)
        pl.resize(200, pl.maximumHeight())
        pl.grab()
        self.assertEqual(calls, [], "collapsed plot rendered its series")

        pl.set_collapsed(False)
        pl.resize(200, pl.maximumHeight())
        pl.grab()
        self.assertTrue(calls, "expanded plot did not render")

    def test_collapsing_releases_the_rendered_image(self):
        """A cached QImage is the memory half. It is reconstructible from the
        matrix, so dropping it is free; the matrix itself is kept, since
        rebuilding that needs the cube the explorer may have evicted."""
        dp = DensityPlot("d")
        dp.set_matrix(np.random.rand(40, 12).astype(np.float32))
        dp.resize(200, dp.maximumHeight())
        dp.grab()
        self.assertIsNotNone(dp._img)

        dp.set_collapsible(True)
        dp.set_collapsed(True)
        self.assertIsNone(dp._img, "collapsed plot kept its rendered image")
        self.assertTrue(dp.matrix.size, "collapse must not discard the matrix")

    def test_scalogram_releases_its_image_too(self):
        sp = ScalogramPlot("s", np.linspace(0.5, 8.0, 12))
        sp.set_scalogram(np.random.rand(12, 40).astype(np.float32))
        sp.resize(200, sp.maximumHeight())
        sp.grab()
        self.assertIsNotNone(sp._img)
        sp.set_collapsible(True)
        sp.set_collapsed(True)
        self.assertIsNone(sp._img)

    def test_set_cursor_on_collapsed_plot_schedules_no_repaint(self):
        """Scrubbing moves the cursor on every plot every frame. A collapsed
        one must not even queue the paint event."""
        pl = MiniPlot("t")
        pl.set_series(np.arange(100, dtype=np.float32))
        pl.set_collapsible(True)
        pl.set_collapsed(True)
        seen = []
        pl.update = lambda *a, **k: seen.append(1)
        pl.set_cursor(50)
        self.assertEqual(seen, [])


class ToggleInteractionTest(unittest.TestCase):
    def setUp(self):
        _app()

    def _click(self, pl, x, y):
        QTest.mouseClick(pl, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, QPoint(x, y))

    def test_marker_click_toggles(self):
        pl = MiniPlot("t")
        pl.set_collapsible(True)
        pl.set_collapsed(True)
        pl.resize(200, pl.maximumHeight())
        spy = QSignalSpy(pl.collapse_toggled)
        self._click(pl, 8, 8)
        self.assertFalse(pl.is_collapsed())
        self.assertEqual(len(spy), 1)
        self.assertTrue(spy[0][0], "signal reports expanded=True on opening")

    def test_marker_click_does_not_seek(self):
        """The marker sits inside the plot rect, so without an explicit guard
        opening a plot would also jump the video to frame 0."""
        pl = MiniPlot("t")
        pl.set_series(np.arange(100, dtype=np.float32))
        pl.set_collapsible(True)
        pl.resize(200, pl.maximumHeight())
        spy = QSignalSpy(pl.seek_requested)
        self._click(pl, 8, 8)
        self.assertEqual(len(spy), 0, "toggling the header also seeked")

    def test_collapsed_plot_swallows_seeks_and_band_drags(self):
        pl = MiniPlot("t")
        pl.set_series(np.arange(100, dtype=np.float32))
        pl.set_band_active(True)
        pl.set_collapsible(True)
        pl.set_collapsed(True)
        pl.resize(200, pl.maximumHeight())
        seek = QSignalSpy(pl.seek_requested)
        band = QSignalSpy(pl.band_changed)
        self._click(pl, 120, 8)
        self.assertEqual(len(seek), 0)
        self.assertEqual(len(band), 0)

    def test_pixelbar_inherits_the_toggle(self):
        pl = PixelBarPlot("bars", unit="blocks")
        pl.set_series(np.arange(50, dtype=np.float32))
        pl.set_collapsible(True)
        pl.set_collapsed(True)
        pl.resize(200, pl.maximumHeight())
        pl.grab()                      # must not raise on the collapsed path
        self._click(pl, 8, 8)
        self.assertFalse(pl.is_collapsed())


class AutoCollapseEmptyTest(unittest.TestCase):
    """T13: an unpopulated per-channel heatmap collapses to checkbox height."""

    def setUp(self):
        _app()

    def test_empty_matrix_collapses_populated_expands(self):
        dp = DensityPlot("d")
        dp.set_collapsible(True)
        dp.set_auto_collapse_empty(True)
        self.assertTrue(dp.is_collapsed())
        self.assertEqual(dp.maximumHeight(), MiniPlot.COLLAPSED_H)

        dp.set_matrix(np.random.rand(20, 6).astype(np.float32))
        self.assertFalse(dp.is_collapsed())
        self.assertEqual(dp.maximumHeight(), MiniPlot.BASE_H)

        dp.set_matrix(np.zeros((0, 0), np.float32))
        self.assertTrue(dp.is_collapsed())

    def test_user_collapse_survives_data_arriving(self):
        """The two states are separate for this reason: data showing up must
        not re-open a plot the user deliberately shut."""
        dp = DensityPlot("d")
        dp.set_collapsible(True)
        dp.set_auto_collapse_empty(True)
        dp.set_collapsed(True)
        dp.set_matrix(np.random.rand(20, 6).astype(np.float32))
        self.assertTrue(dp.is_collapsed(), "arriving data overrode user intent")

        dp.set_collapsed(False)
        self.assertFalse(dp.is_collapsed())

    def test_auto_collapsed_plot_cannot_be_opened_by_the_marker(self):
        """There is nothing to show, so the toggle is inert -- and the marker
        renders as [.] rather than [+] so it does not advertise otherwise."""
        dp = DensityPlot("d")
        dp.set_collapsible(True)
        dp.set_auto_collapse_empty(True)
        dp.resize(200, dp.maximumHeight())
        QTest.mouseClick(dp, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, QPoint(8, 8))
        self.assertTrue(dp.is_collapsed())
        self.assertTrue(dp._auto_collapsed)


class CollapseDoesNotDisarmTheDetectorTest(unittest.TestCase):
    """Visibility must never decide what is COMPUTED (regression).

    The selected channel's band-power matrix is the detector's input, not a
    picture: _recompute_counts and _recompute_clump read it directly. Skipping
    its sum because the plot was collapsed emptied the entire detection sweep
    while the cube sat built in the cache -- a silent false negative, and a
    deadlock too, since the empty matrix re-armed the auto-collapse that caused
    it. Collapsed-by-default (T14) made that the state on every open.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def _explorer(self, T=200, ny=5, nx=5):
        from core.channel_source import ChannelData
        from gui.explorers.scalogram_explorer import ScalogramExplorer
        rng = np.random.default_rng(0)
        meta = {"fps": 30.0, "grid": [ny, nx], "block_size": 64, "n_frames": T,
                "replicate_tiles": [{"id": 0, "label": "r0",
                                     "atlas_bbox": [0, 0, ny, nx],
                                     "frac": [0.0, 0.0, 1.0, 1.0]}]}

        ch = {n: rng.random((T, ny, nx), dtype=np.float32) + 0.1
              for n in ("change", "appearance", "tensor_speed", "intensity")}
        ex = ScalogramExplorer.from_channel_data(
            ChannelData(meta=meta, channels=ch), video_path=None,
            own_shortcuts=False, own_status=False)
        self.addCleanup(self._destroy, ex)
        return ex, T

    def _destroy(self, ex):
        ex.close()
        if ex._worker is not None:
            ex._worker.wait()
        ex.deleteLater()
        self.app.processEvents()

    def _settle(self, ex):
        import time
        for _ in range(200):
            self.app.processEvents()
            if ex._worker is None and ex._sg_cache:
                break
            time.sleep(0.02)
        for _ in range(5):
            self.app.processEvents()
            time.sleep(0.02)
        self.app.processEvents()

    def test_sweep_is_populated_with_every_plot_collapsed(self):
        ex, T = self._explorer()
        self._settle(ex)
        self.assertTrue(ex._sg_cache, "cube never built; test proves nothing")
        sel = ex._selected_density()
        self.assertTrue(sel.is_collapsed(), "precondition: collapsed by default")
        self.assertEqual(sel.matrix.shape[0], T,
                         "selected channel's matrix was skipped while collapsed")
        for name, pl in (("count", ex.count_plot),
                         ("windowed count", ex.count_w_plot),
                         ("detect", ex.detect_plot),
                         ("clump", ex.clump_plot)):
            self.assertEqual(pl.y.size, T,
                             f"{name} series empty with plots collapsed")

    def test_unselected_channels_are_still_skipped(self):
        """The optimization must survive the fix -- otherwise T10 buys nothing."""
        ex, T = self._explorer()
        self._settle(ex)
        for name, dp in ex.density_plots.items():
            if name == ex.channel:
                continue
            self.assertEqual(dp.matrix.size, 0,
                             f"{name} was summed despite being collapsed")


if __name__ == "__main__":
    unittest.main()
