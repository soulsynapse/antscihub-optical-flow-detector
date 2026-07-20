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
        # The selected channel's heatmap is now permanently open (it carries the
        # value band), so the collapse it must survive can no longer be reached
        # by default -- it is forced here. The exemption in _refresh_densities is
        # still the thing under test: visibility must never decide what is
        # computed, whatever route sets the flag.
        sel.set_collapsed(True)
        ex._refresh_densities()
        self.assertTrue(sel.is_collapsed(), "precondition: collapsed")
        self.assertEqual(sel.matrix.shape[0], T,
                         "selected channel's matrix was skipped while collapsed")
        for name, pl in (("count", ex.count_plot),
                         ("windowed count", ex.count_w_plot),
                         ("clump", ex.clump_plot)):
            self.assertEqual(pl.y.size, T,
                             f"{name} series empty with plots collapsed")
        # T15 moved the gate off its own plot and onto count_w_plot as shading.
        # It is asserted on the explorer's own array, not on the shading: the
        # point of this test is that the detection is COMPUTED, and reading it
        # back off a widget would let a drawing change mask a detection change.
        self.assertEqual(ex.detect.size, T,
                         "detection gate empty with plots collapsed")

    def test_threshold_plot_is_exempt_from_collapsed_by_default(self):
        """T28. count_w_plot carries the detection threshold band and, since
        T15, the detection shading. Collapsing it by default put the primary
        tuning control and its result one click away on open -- following T14
        literally, and defeating the reason the panel is opened at all."""
        ex, _ = self._explorer()
        self._settle(ex)
        self.assertFalse(ex.count_w_plot.is_collapsed(),
                         "the threshold plot opened collapsed")
        self.assertTrue(ex.count_plot.is_collapsed(),
                        "T14 stopped applying to the other sweep plots")

    def test_the_three_control_readouts_open_uncollapsed(self):
        """Scalogram, selected-channel heatmap and windowed count each carry a
        drag control (frequency band, value band, detection threshold) and show
        what the drag did, so all three OPEN uncollapsed rather than following
        T14's collapsed-by-default."""
        ex, _ = self._explorer()
        self._settle(ex)
        for name, pl in (("scalogram", ex.scalo_plot),
                         ("selected density", ex._selected_density()),
                         ("windowed count", ex.count_w_plot)):
            self.assertFalse(pl.is_collapsed(), f"{name} opened collapsed")

    def test_only_the_two_band_plots_lose_their_toggle(self):
        """Batch N stripped the [+] from all three. The selected heatmap got
        its toggle back: defaulting it open already keeps the control visible,
        and being able to collapse it is the only way to A/B whether the
        heatmap's repaint is what costs 3x during playback. The scalogram and
        the windowed count stay permanent."""
        ex, _ = self._explorer()
        self._settle(ex)
        self.assertTrue(ex._selected_density()._collapsible,
                        "the selected heatmap cannot be collapsed for an A/B")
        for name, pl in (("scalogram", ex.scalo_plot),
                         ("windowed count", ex.count_w_plot)):
            self.assertFalse(pl._collapsible, f"{name} grew a [+]")

    def test_collapsing_the_selected_heatmap_does_not_disarm_the_detector(self):
        """The whole reason the toggle is safe to offer. _refresh_densities
        exempts the selected channel BY NAME, so collapsing it stops the paint
        and leaves the detector's input intact -- the T14 near-miss in reverse.
        """
        ex, T = self._explorer()
        self._settle(ex)
        sel = ex._selected_density()
        ex._on_density_toggled(ex.channel, False)
        sel.set_collapsed(True)
        ex._refresh_densities()
        ex._recompute_sweep()
        self.assertTrue(sel.is_collapsed(), "precondition: collapsed")
        self.assertEqual(sel.matrix.shape[0], T,
                         "collapsing the selected heatmap emptied its matrix")
        self.assertEqual(ex.detect.size, T,
                         "collapsing the selected heatmap disarmed the gate")

    def test_a_selected_heatmap_collapsed_by_the_user_stays_collapsed(self):
        """Intent survives a round trip through deselection and back."""
        ex, _ = self._explorer()
        self._settle(ex)
        first = ex.channel
        other = next(n for n in ex.density_plots if n != first)
        ex._on_density_toggled(first, False)          # collapse while selected
        ex.channel = other
        ex._apply_selected_plot_ui()
        ex.channel = first                            # come back to it
        ex._apply_selected_plot_ui()
        self.assertTrue(ex.density_plots[first].is_collapsed(),
                        "reselecting sprang a user-collapsed heatmap open")

    def test_selection_moves_the_permanent_heatmap(self):
        """Which density plot opens follows the channel selection: the newly
        selected one opens; the old one closes again, since it was only ever
        open because it was selected."""
        ex, _ = self._explorer()
        self._settle(ex)
        first = ex.channel
        other = next(n for n in ex.density_plots if n != first)
        ex.channel = other
        ex._apply_selected_plot_ui()
        self.assertFalse(ex.density_plots[other].is_collapsed())
        self.assertTrue(ex.density_plots[first].is_collapsed(),
                        "a heatmap open only by selection stayed open")

    def test_visiting_every_channel_does_not_leave_them_all_summing(self):
        """_refresh_densities keys off is_collapsed(), so a plot left open by a
        selection that has moved on stays in `wanted` forever. Cycling the
        channels would then make every later band drag re-sum every cube --
        the exact per-refresh cost T14 removed."""
        ex, _ = self._explorer()
        self._settle(ex)
        names = list(ex.density_plots)
        for n in names:
            ex.channel = n
            ex._apply_selected_plot_ui()
        open_now = [n for n in names if not ex.density_plots[n].is_collapsed()]
        self.assertEqual(open_now, [ex.channel],
                         "every visited channel stayed open and billable")

    def test_a_user_opened_heatmap_survives_deselection(self):
        """The flip side: an explicit [+] is real intent and must outlive the
        selection that happens to move away from it."""
        ex, _ = self._explorer()
        self._settle(ex)
        first = ex.channel
        other = next(n for n in ex.density_plots if n != first)
        # Deselect `first` so it is collapsible, then open it as the user would.
        ex.channel = other
        ex._apply_selected_plot_ui()
        ex._on_density_toggled(first, True)
        ex.density_plots[first].set_collapsed(False)
        # Move the selection again; the explicit open must not be undone.
        ex.channel = names_next = next(n for n in ex.density_plots
                                       if n not in (first, other))
        ex._apply_selected_plot_ui()
        self.assertFalse(ex.density_plots[first].is_collapsed(),
                         "an explicitly opened heatmap was force-closed")
        self.assertEqual(ex.channel, names_next)

    def test_a_non_collapsible_plot_still_marks_an_auto_collapse(self):
        """count_w_plot has no [+], but T27 can still auto-collapse it. Without
        a marker it just silently loses its body, which is the unexplained
        control the [.] form exists to prevent."""
        ex, _ = self._explorer()
        self._settle(ex)
        pl = ex.count_w_plot
        self.assertFalse(pl._collapsible)
        pl.set_series(np.zeros(0, np.float32))       # empty -> auto-collapse
        self.assertTrue(pl.is_collapsed(), "precondition: auto-collapsed")
        self.assertTrue(pl._has_marker(), "no marker on a collapsed pane")

    def test_gate_reaches_the_plot_it_is_shaded_on(self):
        """T15's wiring: the gate is computed (asserted above) AND published."""
        ex, T = self._explorer()
        self._settle(ex)
        ex.count_w_plot.band_lo, ex.count_w_plot.band_hi = -1.0, float("inf")
        ex._recompute_detect()
        mask = ex.count_w_plot._detect_mask
        self.assertIsNotNone(mask, "gate never reached count_w_plot")
        self.assertEqual(mask.size, T)
        self.assertTrue(mask.all(),
                        "a wide-open band detected nothing; gate is not wired")

    def test_badge_tracks_the_gate_across_a_seek(self):
        """T16 at the explorer level: the badge follows the cursor, not just
        the recompute. A badge that only updates on a band drag would sit stale
        through an entire scrub."""
        ex, T = self._explorer()
        self._settle(ex)
        ex.detect = np.zeros(T, np.float32)
        ex.detect[T // 2] = 1.0
        ex._apply_frame(T // 2)
        self.assertTrue(ex.video_view._detected, "badge missed a positive frame")
        ex._apply_frame(T // 2 + 1)
        self.assertFalse(ex.video_view._detected, "badge stuck on after seeking")

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
