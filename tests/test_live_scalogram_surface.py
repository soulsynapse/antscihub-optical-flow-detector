from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from gui.explorers import live_scalogram_surface
from gui.explorers.live_scalogram_surface import (_Busy, _Cancelled,
                                                  _StreamWorker,
                                                  LiveScalogramSurface)
from tests.test_channel_source import _write_moving_square


class _QtTestCase(unittest.TestCase):
    """Anything constructing a QObject needs the application to exist first, and
    these tests must not depend on some other module having created it."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])


class _SurfaceTestCase(_QtTestCase):
    """Shared fixture: one throwaway clip and a surface whose opening extract and
    terminal display step are stubbed, so tests drive the control flow only."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
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

    def _destroy(self, surface):
        """close() only runs closeEvent; without an actual delete the widget and
        its timers outlive the test and crash Qt during a later module's
        teardown. deleteLater needs an event loop turn to take effect."""
        surface.close()
        surface.deleteLater()
        self.app.processEvents()

    def _surface(self):
        reps = [{"id": 0, "label": "all", "frac": (0.0, 0.0, 1.0, 1.0)}]
        # singleShot is patched out so constructing the surface does not kick off
        # the opening extract pass.
        with patch("gui.explorers.live_scalogram_surface.QTimer.singleShot"):
            surface = LiveScalogramSurface(self.video, reps)
        self.addCleanup(self._destroy, surface)
        surface._show_channel_data = MagicMock()
        surface.extract = MagicMock()
        return surface


class TypedKnobCommitTests(_SurfaceTestCase):
    """Typing into a knob must not fire a pass per keystroke, and pressing Enter
    must hand focus back: while a spin box's line edit holds focus it eats the
    Space that toggles playback."""

    # Typed text per knob, each a value the box will actually accept: a no-op
    # edit commits nothing and would prove nothing. The downsample "0.25" is the
    # case that motivated this -- it passes through 0 and 0.2 on the way, and
    # with keyboard tracking on each of those committed a value and armed a pass.
    KNOBS = (("ds_spin", "0.25"), ("block_spin", "5"),
             ("len_spin", "1.0"), ("start_slider", "3"))

    def _shown_surface(self):
        # Focus does not move on a widget that was never shown, so every
        # assertion here would pass vacuously against an unshown surface.
        surface = self._surface()
        surface.show()
        self.app.processEvents()
        return surface

    def test_typing_does_not_arm_a_pass_until_enter(self):
        surface = self._shown_surface()
        for name, typed in self.KNOBS:
            spin = getattr(surface, name)
            with self.subTest(knob=name):
                spin.setFocus()
                spin.selectAll()
                surface._debounce.stop()
                surface._block_debounce.stop()
                QTest.keyClicks(spin, typed)
                self.assertFalse(surface._debounce.isActive())
                self.assertFalse(surface._block_debounce.isActive())

                QTest.keyClick(spin, Qt.Key.Key_Return)
                self.assertTrue(surface._debounce.isActive()
                                or surface._block_debounce.isActive())

    def test_enter_returns_focus_to_the_surface(self):
        surface = self._shown_surface()
        for name, _ in self.KNOBS:
            spin = getattr(surface, name)
            with self.subTest(knob=name):
                spin.setFocus()
                QTest.keyClick(spin, Qt.Key.Key_Return)
                self.assertFalse(spin.hasFocus())
                # The main window's Space handler walks up from the focus widget
                # looking for toggle_playback(); the surface must carry focus for
                # that walk to find one.
                self.assertTrue(surface.hasFocus())
                self.assertTrue(callable(getattr(surface, "toggle_playback")))

    def test_clicking_away_leaves_focus_where_the_user_put_it(self):
        """editingFinished also fires on focus-out. Grabbing focus back there
        would yank it out of whatever the user just clicked into."""
        surface = self._shown_surface()
        surface.ds_spin.setFocus()
        surface.norm_combo.setFocus()

        self.assertTrue(surface.norm_combo.hasFocus())
        self.assertFalse(surface.hasFocus())


class LiveScalogramSurfaceBlockTests(_SurfaceTestCase):
    """A Block change is block-independent downstream of the per-pixel tensor
    solve, so it must re-reduce the cached block=1 channels rather than decode
    and solve the window again."""

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

    def test_block_change_is_ignored_during_the_whole_video_pass(self):
        surface = self._surface()
        start, n = surface._window()
        surface._pp = object()
        surface._pp_key = surface._pp_signature(surface._build_cfg(), start, n)
        surface._proc_worker = MagicMock()

        surface.block_spin.setValue(5)
        surface._on_block_changed()

        surface.extract.assert_not_called()
        surface._show_channel_data.assert_not_called()
        surface._proc_worker = None

    def test_block_change_mid_extract_supersedes_the_stale_pass(self):
        """A cache hit would be overwritten by the extract in flight, so even the
        cheap re-reduce path has to defer to a superseding extract."""
        surface = self._surface()
        start, n = surface._window()
        surface._pp = object()
        surface._pp_key = surface._pp_signature(surface._build_cfg(), start, n)
        surface._worker = MagicMock()

        surface.block_spin.setValue(5)
        surface._on_block_changed()

        surface.extract.assert_called_once()
        surface._show_channel_data.assert_not_called()
        surface._worker = None


class StreamWorkerCancelTests(_QtTestCase):
    """The worker's only cancel point is the progress callback it hands to the
    extractor, so the flag has to turn a tick into an unwind."""

    def test_tick_emits_progress_until_cancelled_then_raises(self):
        w = _StreamWorker()
        self.assertFalse(w.is_cancelled())
        with patch.object(_StreamWorker, "progress") as sig:
            w._tick(20, 100)
            sig.emit.assert_called_once_with(20, 100)
            w.cancel()
            self.assertTrue(w.is_cancelled())
            with self.assertRaises(_Cancelled):
                w._tick(40, 100)
            sig.emit.assert_called_once()      # no further progress after cancel

    def test_run_routes_each_outcome_to_exactly_one_signal(self):
        # `done` is emitted by the concrete _run, so the success case stubs a _run
        # that emits it -- otherwise this asserts nothing about the done path.
        cases = [
            (lambda w: w.done.emit("payload"), "done"),
            (lambda w: (_ for _ in ()).throw(_Cancelled()), "cancelled"),
            (lambda w: (_ for _ in ()).throw(ValueError("boom")), "failed"),
        ]
        for body, expected in cases:
            with self.subTest(expected=expected):
                w = _StreamWorker()
                w._run = lambda body=body, w=w: body(w)
                fired = []
                for name in ("done", "cancelled", "failed"):
                    getattr(w, name).connect(lambda *a, n=name: fired.append(n))
                w.run()
                self.assertEqual(fired, [expected])

    def test_failed_carries_the_exception_type_and_message(self):
        w = _StreamWorker()
        w._run = MagicMock(side_effect=ValueError("boom"))
        msgs = []
        w.failed.connect(msgs.append)
        w.run()
        self.assertEqual(msgs, ["ValueError: boom"])


class LiveScalogramSurfaceStopTests(_SurfaceTestCase):
    """Extract / Process become Stop while their own pass runs (todo 7)."""

    def test_buttons_toggle_to_stop_for_the_running_pass_only(self):
        surface = self._surface()
        surface._set_busy(_Busy.EXTRACT)
        self.assertEqual(surface.extract_btn.text(), "Stop")
        self.assertTrue(surface.extract_btn.isEnabled())
        self.assertFalse(surface.process_btn.isEnabled())

        surface._set_busy(_Busy.PROCESS)
        self.assertEqual(surface.process_btn.text(), "Stop")
        self.assertTrue(surface.process_btn.isEnabled())
        self.assertFalse(surface.extract_btn.isEnabled())

        surface._set_busy(None)
        self.assertEqual(surface.extract_btn.text(), "Extract")
        self.assertEqual(surface.process_btn.text(), "Process whole video ▶")
        self.assertTrue(surface.extract_btn.isEnabled())
        self.assertTrue(surface.process_btn.isEnabled())

    def test_clicking_stop_cancels_instead_of_starting_another_pass(self):
        surface = self._surface()
        surface._worker = MagicMock()
        surface._on_extract_clicked()
        surface._worker.cancel.assert_called_once()
        surface.extract.assert_not_called()
        # Both stay disabled until the worker's `cancelled` actually lands.
        self.assertFalse(surface.extract_btn.isEnabled())
        self.assertFalse(surface.process_btn.isEnabled())
        surface._worker = None

    def test_knob_change_mid_extract_supersedes_rather_than_dropping_the_edit(self):
        surface = self._surface()
        del surface.extract              # exercise the real supersede path
        worker = MagicMock()
        surface._worker = worker

        surface.extract()
        worker.cancel.assert_called_once()
        self.assertTrue(surface._restart_extract)

        # The replacement only starts once the stopped pass has unwound.
        surface.extract = MagicMock()
        surface._on_extract_cancelled()
        surface.extract.assert_called_once()
        self.assertFalse(surface._restart_extract)
        self.assertIsNone(surface._worker)

    def test_pass_finishing_before_the_cancel_lands_still_honours_the_supersede(self):
        """A cancel set after the worker's last tick never reaches `cancelled` --
        `done` arrives instead. If that path did not consume _restart_extract the
        knob edit would be silently dropped AND the stale flag would make the
        next Stop restart the pass instead of stopping it."""
        surface = self._surface()
        del surface.extract
        surface._worker = MagicMock()
        surface.extract()                       # knob change arms the supersede
        self.assertTrue(surface._restart_extract)

        surface.extract = MagicMock()
        surface._pending_pp = False
        surface._pending_cfg = surface._build_cfg()
        surface._on_extracted(object())         # the race: done, not cancelled

        surface.extract.assert_called_once()    # the edit is honoured
        self.assertFalse(surface._restart_extract)
        # The superseded result must not be shown -- a newer pass is on its way.
        surface._show_channel_data.assert_not_called()

    def test_cancelled_extract_leaves_the_pixel_cache_bookkeeping_clean(self):
        surface = self._surface()
        surface._worker = MagicMock()
        surface._pending_pp = True
        surface._pending_key = ("k",)
        surface._pending_cfg = object()

        surface._on_extract_cancelled()

        self.assertFalse(surface._pending_pp)
        self.assertIsNone(surface._pending_key)
        self.assertIsNone(surface._pending_cfg)
        self.assertEqual(surface.extract_btn.text(), "Extract")
        self.assertIn("stopped", surface.status_lbl.text())

    def test_extract_is_refused_while_the_whole_video_pass_owns_the_decoder(self):
        surface = self._surface()
        del surface.extract
        surface._proc_worker = MagicMock()

        surface.extract()

        surface._proc_worker.cancel.assert_not_called()
        self.assertFalse(surface._restart_extract)
        surface._proc_worker = None


class CostSampleTests(_SurfaceTestCase):
    """Timing carried off a completed pass is what lets the downsample dialog
    price the lever from measurement instead of assertion."""

    @staticmethod
    def _cd(scale, block, wall, frames=100, spans=None):
        cd = MagicMock()
        cd.meta = {"timing": {"scale": scale, "block": block, "wall": wall,
                              "frames": frames, "spans": spans or {}}}
        return cd

    def test_sample_is_recorded_per_scale_and_block(self):
        surface = self._surface()
        surface._record_cost_sample(self._cd(1.0, 64, 4.0))
        surface._record_cost_sample(self._cd(0.5, 32, 2.0))
        self.assertEqual(set(surface._cost_samples), {(1.0, 64), (0.5, 32)})

    def test_a_pass_without_timing_is_ignored(self):
        surface = self._surface()
        cd = MagicMock()
        cd.meta = {}
        surface._record_cost_sample(cd)
        surface._record_cost_sample(self._cd(1.0, 64, 4.0, frames=0))
        self.assertEqual(surface._cost_samples, {})
        self.assertEqual(surface._cost_model(), (None, None))

    def test_re_recording_a_key_still_identifies_the_newest_regime(self):
        # dict order keeps a re-recorded key at its ORIGINAL position, so the
        # newest sample cannot be read off insertion order.
        surface = self._surface()
        surface._record_cost_sample(self._cd(1.0, 32, 4.0))
        surface._record_cost_sample(self._cd(1.0, 32, 4.1))
        surface._record_cost_sample(self._cd(0.5, 16, 2.0))
        self.assertEqual(surface._last_cost_key, (0.5, 16))
        # Both are pinned (tracked would be 64 and 32), but to DIFFERENT blocks,
        # so they are two regimes of one sample each and neither can be fitted.
        model, _block = surface._cost_model()
        self.assertTrue(model.provisional)

    def test_model_stays_provisional_while_only_one_scale_is_measured(self):
        surface = self._surface()
        surface._record_cost_sample(self._cd(1.0, 64, 4.0, spans={"decode": 0.01}))
        model, block = surface._cost_model()
        self.assertTrue(model.provisional)
        # ...and therefore declines to place a knee, which is the whole point:
        # one prefetched pass cannot see the decode floor.
        self.assertIsNone(model.knee_scale())

    def test_samples_at_different_blocks_are_not_fitted_together(self):
        surface = self._surface()
        surface._record_cost_sample(self._cd(0.5, 1, 9.0))    # block=1 pixel cache
        surface._record_cost_sample(self._cd(1.0, 64, 4.0))
        model, block = surface._cost_model()
        self.assertTrue(model.provisional)      # neither regime has two scales

    def test_fit_uses_the_regime_with_the_most_scales_not_the_newest(self):
        """Whether a pass runs at block=1 depends on whether its per-pixel
        footprint fits the budget, which scales with the square of the scale --
        so dragging Downsample naturally splits samples across regimes. Taking
        only the newest regime would leave the model provisional forever."""
        surface = self._surface()
        surface._record_cost_sample(self._cd(1.0, 1, 8.0))
        surface._record_cost_sample(self._cd(0.5, 1, 3.0))
        surface._record_cost_sample(self._cd(0.25, 64, 1.5))   # newest, alone
        self.assertEqual(surface._last_cost_key, (0.25, 64))
        model, block = surface._cost_model()
        self.assertFalse(model.provisional)
        self.assertEqual(model.n_samples, 2)
        self.assertIsNotNone(model.knee_scale())

    def test_the_fitted_regime_is_reported_alongside_the_model(self):
        """A block=1 pass does no reduction and is far slower than a production
        pass, so the dialog has to be told which regime the numbers came from
        rather than presenting them as a batch-run projection."""
        surface = self._surface()
        surface._record_cost_sample(self._cd(1.0, 1, 8.0))
        surface._record_cost_sample(self._cd(0.5, 1, 3.0))
        model, block = surface._cost_model()
        self.assertEqual(block, 1)
        self.assertFalse(model.provisional)

    def test_ties_between_regimes_go_to_the_newest(self):
        surface = self._surface()
        surface._record_cost_sample(self._cd(1.0, 1, 8.0))     # pinned block 1
        surface._record_cost_sample(self._cd(0.5, 1, 3.0))
        surface._record_cost_sample(self._cd(1.0, 64, 6.0))    # tracked (64·1.0)
        surface._record_cost_sample(self._cd(0.5, 32, 2.5))    # tracked (64·0.5)
        model, block = surface._cost_model()
        self.assertEqual(model.n_samples, 2)
        # The tracked pair is newest, so its (cheaper) wall times drive the fit.
        self.assertIsNone(block)
        self.assertAlmostEqual(model.seconds_per_frame(1.0) * 100, 6.0, places=6)

    def test_a_tracked_sweep_is_one_regime_not_one_group_per_scale(self):
        """The dialog's empirical sweep runs at the production block, which on
        the `auto` default means the block MOVES with the scale by design. Group
        on the raw block and every row lands alone, leaving the model provisional
        forever -- the exact opposite of what running the sweep is for."""
        surface = self._surface()
        for scale, block, wall in ((1.0, 64, 6.0), (0.5, 32, 2.5),
                                   (0.25, 16, 1.6)):
            surface._record_cost_sample(self._cd(scale, block, wall))
        model, block = surface._cost_model()
        self.assertFalse(model.provisional)
        self.assertEqual(model.n_samples, 3)
        # None, not a number: no single block describes these passes, and there
        # is no upper-bound caveat to make because this IS what production does.
        self.assertIsNone(block)
        self.assertIsNotNone(model.knee_scale())

    def test_a_pinned_block_sweep_reports_that_block(self):
        surface = self._surface()
        surface._record_cost_sample(self._cd(1.0, 16, 5.0), 16)
        surface._record_cost_sample(self._cd(0.5, 16, 2.0), 16)
        _model, block = surface._cost_model()
        self.assertEqual(block, 16)

    def test_a_pass_pinned_to_the_block_tracking_would_pick_stays_pinned(self):
        """The regime cannot be read off the resolved block. Pin Block to 64 and
        sweep: at scale 1.0 tracking would ALSO have chosen 64, so inferring
        would file the reference row as tracked, split the sweep into two groups
        and drop the widest, highest-leverage point from the fit."""
        surface = self._surface()
        for scale, wall in ((1.0, 6.0), (0.5, 3.0), (0.25, 2.0)):
            surface._record_cost_sample(self._cd(scale, 64, wall), 64)
        model, block = surface._cost_model()
        self.assertEqual(block, 64)
        self.assertEqual(model.n_samples, 3)        # not 2

    def test_a_pass_with_no_measured_wall_is_not_a_sample(self):
        # A zero-cost point drags the fitted decode floor toward zero, which is
        # what puts the knee where aggressive downsampling looks free.
        surface = self._surface()
        surface._record_cost_sample(self._cd(1.0, 64, 0.0))
        self.assertEqual(surface._cost_samples, {})


class SweepPanelTests(_QtTestCase):
    """Panel states that only showed up by driving the real window."""

    @staticmethod
    def _sp(scale, wall=2.0, frames=100, grid=(4, 4), block=8):
        from core.scale_sweep import ScalePass
        return ScalePass(scale=scale, block=block, grid=grid, frames=frames,
                         wall=wall)

    def _panel(self):
        from gui.cost_panels import SweepPanel
        p = SweepPanel("note")
        self.addCleanup(p.deleteLater)
        return p

    def test_pending_rows_are_retired_when_a_sweep_ends_early(self):
        """A stopped sweep left its remaining rows reading "waiting" forever,
        claiming passes were still coming with nothing running."""
        p = self._panel()
        p.begin(["1.00", "0.50", "0.25"])
        p.add_row("1.00", self._sp(1.0), "4.7 d", "26.4 GB")
        p.finish("stopped")
        self.assertEqual(p._rows["0.50"][-1].text(), "not run")
        self.assertEqual(p._rows["0.25"][-1].text(), "not run")
        self.assertNotEqual(p._rows["1.00"][-1].text(), "not run")

    def test_a_row_reports_its_speedup_against_the_reference(self):
        p = self._panel()
        p.begin(["1.00", "0.50"])
        ref = self._sp(1.0, wall=4.0)
        p.add_row("1.00", ref, "4.7 d", "26.4 GB")
        p.add_row("0.50", self._sp(0.5, wall=2.0), "2.4 d", "26.4 GB", ref)
        self.assertIn("2.0", p._rows["0.50"][1].text())
        self.assertIn("faster", p._rows["0.50"][1].text())
        self.assertNotIn("faster", p._rows["1.00"][1].text())

    def test_a_pass_with_no_measured_wall_is_flagged(self):
        p = self._panel()
        p.begin(["1.00"])
        p.add_row("1.00", self._sp(1.0, wall=0.0), "-", "26.4 GB")
        self.assertIn("e6a23c", p._rows["1.00"][0].styleSheet())

    def test_a_stale_refusal_reason_clears_once_the_run_is_available(self):
        p = self._panel()
        p.set_available(False, "Another pass is running")
        self.assertIn("Another pass", p.status.text())
        p.set_available(True)
        self.assertEqual(p.status.text(), "")


class FrontierStorageTests(_QtTestCase):
    """Storage rides a second axis on the frontier. Under the tracked block it
    is dead flat while the time curve falls, which is the two-lever argument in
    one picture -- and normalizing both onto one axis would destroy it."""

    def _plot(self):
        from gui.cost_panels import FrontierPlot
        p = FrontierPlot()
        p.resize(400, 240)
        self.addCleanup(p.deleteLater)
        return p

    def test_a_second_series_widens_the_right_margin_for_its_axis(self):
        p = self._plot()
        p.set_curve([1.0, 0.5], [10.0, 5.0])
        without = p._plot_rect()[2]
        p.set_curve([1.0, 0.5], [10.0, 5.0], second=[8.0, 8.0])
        self.assertLess(p._plot_rect()[2], without)

    def test_the_second_series_is_zero_based_so_flat_reads_as_flat(self):
        # Auto-ranging a constant series would amplify float noise to fill the
        # panel and make "storage does not move" look like it moves.
        p = self._plot()
        p.set_curve([1.0, 0.5, 0.25], [10.0, 5.0, 3.0], second=[8.0, 8.0, 8.0])
        ys = [p._y2_of(v) for v in (8.0, 8.0, 8.0)]
        self.assertEqual(len(set(ys)), 1)

    def test_it_paints_with_both_series_and_a_rise_marker(self):
        from PyQt6.QtGui import QPixmap
        p = self._plot()
        p.set_curve([1.0, 0.5, 0.25], [10.0, 5.0, 3.0], knee=0.5,
                    second=[8.0, 8.0, 9.5], rise_below=0.25,
                    second_fmt=lambda v: f"{v:.1f} GB")
        p.render(QPixmap(p.size()))     # would raise on a bad paint path


class StorageCurveTests(unittest.TestCase):
    """The measured claim behind treating these as two levers."""

    def _reps(self):
        return [{"id": 0, "label": "all", "frac": (0.0, 0.0, 1.0, 1.0)}]

    def test_tracked_block_stops_storage_falling_with_scale(self):
        """The decision-relevant claim, and deliberately NOT "storage is flat":
        the block and the cell count round independently, so the curve jitters
        (19% on this geometry, non-monotone). What holds is that downsampling
        never buys disk -- reaching for this lever to save storage is reaching
        for the wrong knob."""
        from core.config import FlowConfig
        from core.scale_sweep import storage_curve
        scales = [1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1]
        s = storage_curve(self._reps(), 5312, 2988, scales,
                          FlowConfig(block_size=None), 24.0, 4, 100.0)
        at_full = s[0]
        self.assertGreaterEqual(min(s), 0.9 * at_full)
        self.assertGreater(max(s), at_full)     # some scales cost MORE

    def test_the_packed_atlas_can_absorb_the_rounding_entirely(self):
        # The 7-replicate layout happens to hold exactly 205 cells from 1.0 down
        # to 0.15, which is why "flat" looked true when first measured. It is a
        # property of that packing, not a general one.
        from core.config import FlowConfig
        from core.scale_sweep import storage_curve
        reps = [{"id": i, "label": f"r{i}",
                 "frac": (0.1 * i, 0.1, 0.1 * i + 0.08, 0.5)} for i in range(7)]
        s = storage_curve(reps, 5312, 2988, [1.0, 0.75, 0.5, 0.25],
                          FlowConfig(block_size=None), 24.0, 4, 100.0)
        self.assertLessEqual((max(s) - min(s)) / max(s), 0.05)

    def test_a_pinned_block_makes_storage_fall_with_scale(self):
        from core.config import FlowConfig
        from core.scale_sweep import storage_curve
        s = storage_curve(self._reps(), 5312, 2988, [1.0, 0.5, 0.25],
                          FlowConfig(block_size=64), 24.0, 4, 100.0)
        self.assertGreater(s[0], s[1])
        self.assertGreater(s[1], s[2])

    def test_a_rising_tail_is_found_rather_than_smoothed(self):
        # Past a point the tracked block stops dividing the scaled tile evenly
        # and partial edge cells push the count back UP -- more storage for less
        # resolution, which is a hard reason to stop and is invisible unmarked.
        from core.scale_sweep import storage_rises_below
        scales = [1.0, 0.5, 0.25, 0.15, 0.10]
        storage = [26.4, 26.4, 26.4, 26.4, 30.9]
        self.assertAlmostEqual(storage_rises_below(scales, storage), 0.10)


class SweepTests(_SurfaceTestCase):
    """The sweep end to end: real passes, real dialog, no detector."""

    def _pump(self, surface, timeout=120.0):
        """Pump the event loop until the worker has signalled in. Sleeps rather
        than spinning: a bare processEvents loop drains an empty queue far faster
        than the passes run and would time out on a working sweep."""
        deadline = time.monotonic() + timeout
        while surface._sweep_worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        self.assertIsNone(surface._sweep_worker, "sweep did not finish")

    def _sweep(self, surface, scales):
        surface._start_sweep(scales)
        self._pump(surface)

    def test_a_sweep_produces_a_row_and_a_cost_sample_per_scale(self):
        surface = self._surface()
        dlg = MagicMock()
        surface._dlg = dlg
        self._sweep(surface, [1.0, 0.5])
        self.assertEqual(dlg.add_sweep_row.call_count, 2)
        rows = [c.args[0] for c in dlg.add_sweep_row.call_args_list]
        self.assertEqual([r.scale for r in rows], [1.0, 0.5])
        # Each row is also a sample, keyed at the block a production run uses.
        self.assertEqual(set(surface._cost_samples), {(1.0, 64), (0.5, 32)})
        self.assertTrue(all(b > 1 for _s, b in surface._cost_samples))

    def test_the_sweep_needs_no_tuned_detector_or_selected_replicate(self):
        """Dropping the detector is what makes this true: the sweep only times
        extraction, so it can run the moment the window opens."""
        surface = self._surface()
        self.assertIsNone(surface._explorer)
        ok, why = surface._sweep_ready()
        self.assertTrue(ok, why)

    def test_the_sweep_is_what_moves_the_model_out_of_the_block_1_regime(self):
        """Before the sweep the only samples come from the live block=1 pixel
        cache, which overstates a batch run; after it, the model is fitted in the
        tracked (production) regime."""
        surface = self._surface()
        surface._record_cost_sample(CostSampleTests._cd(1.0, 1, 9.0))
        surface._record_cost_sample(CostSampleTests._cd(0.5, 1, 4.0))
        self.assertEqual(surface._cost_model()[1], 1)
        surface._dlg = MagicMock()
        self._sweep(surface, [1.0, 0.5])
        model, block = surface._cost_model()
        self.assertIsNone(block)            # tracked: no upper-bound caveat
        self.assertFalse(model.provisional)
        self.assertEqual(model.n_samples, 2)

    def test_rows_reach_a_real_dialog_and_redraw_the_frontier(self):
        from gui.downsample_dialog import DownsampleDialog
        surface = self._surface()
        w, h, fps, _fc = surface._dims
        dlg = DownsampleDialog([{"id": 0, "label": "all",
                                 "frac": (0.0, 0.0, 1.0, 1.0)}],
                               src_width=w, src_height=h, fps=fps,
                               current_scale=1.0, model=None)
        self.addCleanup(dlg.deleteLater)
        surface._dlg = dlg
        dlg.sweep_requested.connect(surface._start_sweep)
        self.assertEqual(dlg.plot._values, [])          # nothing measured yet
        dlg._on_run_sweep()
        self._pump(surface)
        self.assertIsNotNone(dlg._sweep_ref)
        self.assertAlmostEqual(dlg._sweep_ref.scale, 1.0, places=6)
        self.assertFalse(dlg._model.provisional)
        self.assertGreater(len(dlg.plot._values), 1)
        # The storage series rides along, which is the point of the plot.
        self.assertEqual(len(dlg.plot._second), len(dlg.plot._values))
        self.assertEqual(len(dlg.sweep._rows), 5)
        # Every row carries a corpus projection, including the reference. It
        # lands while only one scale has been timed, so its projection reads "—"
        # when first drawn; without the re-render at the end it would stay that
        # way on the one row a user most wants the number for.
        for label, cells in dlg.sweep._rows.items():
            self.assertNotEqual(cells[3].text(), "—", f"row {label}")

    def test_a_sweep_owns_the_decoder(self):
        surface = self._surface()
        surface._sweep_worker = MagicMock()
        surface.extract = LiveScalogramSurface.extract.__get__(surface)
        with patch("gui.explorers.live_scalogram_surface._LiveExtractWorker") as W:
            surface.extract()
            surface.process_whole_video()
        W.assert_not_called()
        self.assertIsNone(surface._proc_worker)
        surface._sweep_worker = None

    def test_a_failing_scale_does_not_abort_the_remaining_rows(self):
        surface = self._surface()
        dlg = MagicMock()
        surface._dlg = dlg
        real = live_scalogram_surface.measure_scale

        def flaky(video_path, cfg, reps, **kw):
            if abs(cfg.preprocess.downsample - 0.5) < 1e-9:
                raise RuntimeError("boom")
            return real(video_path, cfg, reps, **kw)

        with patch.object(live_scalogram_surface, "measure_scale", flaky):
            self._sweep(surface, [1.0, 0.5, 0.35])
        self.assertEqual(dlg.add_sweep_row.call_count, 2)
        dlg.sweep_failed.assert_called_once()
        self.assertAlmostEqual(dlg.sweep_failed.call_args.args[0], 0.5)


if __name__ == "__main__":
    unittest.main()
