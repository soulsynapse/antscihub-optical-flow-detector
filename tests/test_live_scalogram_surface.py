from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtWidgets import QApplication

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

    def test_re_recording_a_key_still_identifies_the_newest_block(self):
        # dict order keeps a re-recorded key at its ORIGINAL position, so the
        # newest sample cannot be read off insertion order.
        surface = self._surface()
        surface._record_cost_sample(self._cd(1.0, 32, 4.0))
        surface._record_cost_sample(self._cd(1.0, 64, 5.0))
        surface._record_cost_sample(self._cd(0.5, 32, 2.0))
        surface._record_cost_sample(self._cd(1.0, 32, 4.1))
        self.assertEqual(surface._last_cost_key, (1.0, 32))
        model, block = surface._cost_model()
        # Only the block-32 samples, so the fit is over two distinct scales.
        self.assertFalse(model.provisional)
        self.assertEqual(model.n_samples, 2)

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
        surface._record_cost_sample(self._cd(1.0, 1, 8.0))
        surface._record_cost_sample(self._cd(0.5, 1, 3.0))
        surface._record_cost_sample(self._cd(1.0, 64, 6.0))
        surface._record_cost_sample(self._cd(0.5, 64, 2.5))
        model, block = surface._cost_model()
        self.assertEqual(model.n_samples, 2)
        # The block-64 pair is newest, so its (cheaper) wall times drive the fit.
        self.assertAlmostEqual(model.seconds_per_frame(1.0) * 100, 6.0, places=6)


if __name__ == "__main__":
    unittest.main()
