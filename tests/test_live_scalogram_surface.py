from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QHideEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from gui.explorers import live_scalogram_surface
from core.replicate_home import replicate_dir
from core.scale_sweep import ScalePass
from gui.explorers.live_scalogram_surface import (_Busy, _Cancelled,
                                                  _StreamWorker,
                                                  LiveScalogramSurface)
from gui.tuning_store import tuning_path
from tests.test_channel_source import _write_moving_square


class _QtTestCase(unittest.TestCase):
    """Anything constructing a QObject needs the application to exist first, and
    these tests must not depend on some other module having created it."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])


class _SurfaceTestCase(_QtTestCase):
    """Shared fixture: one throwaway clip and a surface whose live pass is
    stubbed, so tests drive the control flow only."""

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

    def _drop_tuning(self):
        """The surface remembers its tuning in a sidecar next to the clip, and
        every test here shares one clip. Without this, knobs set by one test are
        restored into the next test's surface.

        Since Batch S slice 3 that sidecar has two halves -- the strip beside the
        video and the ``view`` inside the replicate's home -- so both go, and the
        home with them."""
        try:
            os.remove(tuning_path(self.video))
        except OSError:
            pass
        shutil.rmtree(replicate_dir(self.video, 0), ignore_errors=True)

    def _surface(self):
        self._drop_tuning()
        self.addCleanup(self._drop_tuning)
        reps = [{"id": 0, "label": "all", "frac": (0.0, 0.0, 1.0, 1.0)}]
        surface = LiveScalogramSurface(self.video, reps)
        self.addCleanup(self._destroy, surface)
        # Nothing runs on construction any more (Play commits the decoder), but
        # the restart path would, so it is stubbed out for the same reason.
        surface.start_stream = MagicMock()
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


class KnobReplanTests(_SurfaceTestCase):
    """Every preprocessing knob is upstream of the block grid, so a knob change
    replans the RUNNING pass and does nothing when none is running.

    This replaces the old per-pixel-cache behaviour. That cache existed so a
    Block change could re-reduce cached block=1 channels instead of paying a
    full re-extract; with the extract retired there is no window to re-run and
    nothing for it to save.
    """

    def test_a_knob_change_replans_a_running_pass(self):
        surface = self._surface()
        surface._stream_worker = MagicMock()
        surface.restart_stream = MagicMock()
        surface._on_knob_settled()
        surface.restart_stream.assert_called_once()
        # The replan names the settings it is restarting under. That readout is
        # the reason the note travels through restart_stream at all: set before
        # the stop, _request_stop overwrites it and it is never seen.
        (note,), _kw = surface.restart_stream.call_args
        self.assertIn("block", note)
        surface._stream_worker = None

    def test_a_knob_change_while_stopped_still_says_what_it_did(self):
        """Otherwise the strip keeps whatever the last pass left there and a
        knob moved while stopped reads as a dead control."""
        surface = self._surface()
        surface.status_lbl.setText("live pass stopped")
        surface.norm_combo.setCurrentText("zscore")
        surface._on_knob_settled()
        self.assertIn("zscore", surface.status_lbl.text())
        self.assertIn("Play", surface.status_lbl.text())

    def test_a_knob_change_does_nothing_when_no_pass_is_running(self):
        surface = self._surface()
        surface.restart_stream = MagicMock()
        surface._on_knob_settled()
        surface.restart_stream.assert_not_called()

    def test_both_debounces_land_on_the_one_replan(self):
        """Block and the rest still settle at different speeds -- Block is a
        single spin step, Downsample is dragged -- but they no longer differ in
        what they do, so a Block change must not take a separate path."""
        surface = self._surface()
        surface._stream_worker = MagicMock()
        surface.restart_stream = MagicMock()
        surface._debounce.timeout.emit()
        surface._block_debounce.timeout.emit()
        self.assertEqual(surface.restart_stream.call_count, 2)
        surface._stream_worker = None

    def test_a_replan_is_refused_while_the_whole_video_pass_owns_the_decoder(self):
        surface = self._surface()
        surface._proc_worker = MagicMock()
        surface.restart_stream = MagicMock()
        surface._on_knob_settled()
        surface.restart_stream.assert_not_called()
        surface._proc_worker = None


class DetectionArmingTests(_SurfaceTestCase):
    """The detector only becomes answerable once an explorer exists, and on the
    FIRST pass of a session that is after the pass has already started."""

    def test_the_first_pass_arms_detection_when_its_explorer_appears(self):
        """Regression. start_stream stamps the track before the pass, which is
        right for a restart but cannot answer on the first Play: there is no
        explorer, so the stamp stays None and the detect timer never starts.
        The strip then stays empty for the WHOLE run while the frontier readout
        looks healthy -- and it produces no dropped windows, so
        _DETECT_DROP_ALARM never fires either."""
        surface = self._surface()
        self.assertIsNone(surface._explorer)

        surface._stream_worker = MagicMock()
        surface._arm_detection()
        self.assertIsNone(surface._track.stamp)
        self.assertFalse(surface._detect_timer.isActive())

        # The first served window builds the explorer; from there the detector
        # is answerable and must be armed.
        stamp = MagicMock()
        surface._current_stamp = lambda: (stamp, (1, 1, 2, 2))
        surface._explorer = MagicMock()
        surface._arm_detection()
        self.assertIsNotNone(surface._track.stamp)
        self.assertTrue(surface._detect_timer.isActive())
        surface._detect_timer.stop()
        surface._stream_worker = None

    def test_arming_without_a_pass_does_not_start_the_request_timer(self):
        """The timer parks requests on the stream worker, so arming it with no
        worker would tick against None."""
        surface = self._surface()
        stamp = MagicMock()
        surface._current_stamp = lambda: (stamp, (1, 1, 2, 2))
        surface._explorer = MagicMock()
        surface._arm_detection()
        self.assertFalse(surface._detect_timer.isActive())


class ParkedNavigationTests(_SurfaceTestCase):
    """With no pass running the strip still navigates -- and a committed seek now
    LOADS the window there (a single bounded decode), so the paused view shows
    that point rather than wherever the last pass stopped. Play stays the only
    control that commits the decoder to a forward run to the end of the clip."""

    def _finish_preview(self, surface, timeout=120.0):
        """Run the in-flight preview worker to completion and deliver its result
        on the GUI thread. The `done` signal is queued, so it fires on the next
        processEvents after the thread finishes."""
        w = surface._preview_worker
        self.assertIsNotNone(w, "a paused seek should start a preview load")
        w.wait(int(timeout * 1000))
        self.app.processEvents()

    def test_a_committed_seek_while_stopped_loads_the_window(self):
        surface = self._surface()
        # The fixture clip is 40 frames at 20 fps, so the window length has to
        # leave somewhere to seek TO: at the 1 s default the start clamps to 0
        # and the move would be untestable rather than absent.
        surface.len_spin.setValue(0.5)
        surface._on_seek_committed(20)
        self.assertEqual(surface.start_slider.value(), 20)
        # Loads rather than merely parking: a preview worker starts, and the
        # decoder is NOT committed to a forward run (that is Play's job).
        self.assertIn("loading", surface.status_lbl.text())
        surface.start_stream.assert_not_called()
        self._finish_preview(surface)
        # The window at the seek point is on screen, the cursor sits on the
        # clicked frame, and the readout invites Play to run from here.
        self.assertIsNotNone(surface._explorer)
        self.assertEqual(surface._explorer.absolute_frame(), 20)
        self.assertIn("Play", surface.status_lbl.text())
        surface.start_stream.assert_not_called()

    def test_focusing_a_detection_while_stopped_loads_the_window(self):
        """Focusing a detection (releasing a click inside one) is a verb. It
        moves the slider AND loads the window centred on the event, so the plots
        show it for verification rather than staying on the previous span."""
        surface = self._surface()
        surface.len_spin.setValue(0.5)
        surface._focus_frame(20)
        self.assertIn("loading", surface.status_lbl.text())
        self._finish_preview(surface)
        self.assertIsNotNone(surface._explorer)
        # Centred on the event: the window starts half a length before it, so the
        # cursor lands on the event itself.
        self.assertEqual(surface._explorer.absolute_frame(), 20)
        self.assertIn("Play", surface.status_lbl.text())
        surface.start_stream.assert_not_called()

    def test_scrubbing_does_not_move_the_cursor_during_a_pass(self):
        """Under follow_center every served window re-pins the cursor to the
        span centre, so a seek placed by a drag survives ~100 ms and the cursor
        oscillates between the pointer and the centre for the whole drag."""
        surface = self._surface()
        surface._explorer = MagicMock()
        surface._stream_worker = MagicMock()
        surface._on_scrubbed(300)
        surface._explorer.seek_absolute.assert_not_called()
        surface._stream_worker = None
        # Stopped, the cursor is the only feedback there is, so it still moves.
        surface._on_scrubbed(300)
        surface._explorer.seek_absolute.assert_called_once_with(300)


class SaveDetectionsTests(_SurfaceTestCase):
    """Save detections -> labelled marks in the clip's sidecar (Batch R / T30).
    The detector proposes the spans; Save writes the current-settings ones as
    ground truth under a behaviour label the user is asked for."""

    def _armed(self, surface, intervals):
        """A surface whose track reports ``intervals`` under a real stamp."""
        from core.live_track import TrackStamp
        surface.fps = 30.0
        surface._track = MagicMock()
        surface._track.detected_intervals.return_value = intervals
        surface._explorer = MagicMock()
        surface._explorer.detection_params.return_value = {
            "value_band": (0.1, 0.9), "count_band": (5.0, 1e9),
            "detect_window": 3}
        stamp = TrackStamp(channel="change", freq_band_hz=(0.5, 5.0),
                           grid=(8, 8), region_index=0, region_blocks=29,
                           downsample=1.0, block_size=64)
        surface._current_stamp = MagicMock(return_value=(stamp, None))
        self.addCleanup(self._drop_marks)

    def _drop_marks(self):
        from gui.marks_store import marks_path
        try:
            os.remove(marks_path(self.video))
        except OSError:
            pass
        shutil.rmtree(replicate_dir(self.video, 0), ignore_errors=True)

    def _saved(self):
        """The marks as the REPLICATE holds them (Batch S slice 3). The fixture
        has one box, id 0, and the stamp is scoped to region index 0."""
        from gui.marks_store import load_marks
        return load_marks(self.video, replicate_id=0, legacy_region=0)

    def test_save_writes_current_detections_as_labelled_marks(self):
        from gui.marks_store import marks_path
        surface = self._surface()
        self._armed(surface, [(30, 60), (90, 105)])
        with patch.object(live_scalogram_surface, "QInputDialog") as dlg:
            dlg.getText.return_value = ("flying", True)
            surface._save_detections()
        self.assertTrue(os.path.exists(marks_path(self.video, replicate_id=0)))
        doc = self._saved()
        # 30 fps -> seconds; keyed by the region the detector was scoped to.
        self.assertEqual(doc["spans"]["0"],
                         [[1.0, 2.0, "flying"], [3.0, 3.5, "flying"]])
        self.assertEqual(doc["provenance"]["flying"]["channel"], "change")
        self.assertIn("flying", surface.status_lbl.text())

    def test_the_spans_go_to_the_home_not_the_per_video_file(self):
        """Two replicates saving the same label must not share a provenance
        entry -- the collision the per-video file could not avoid."""
        from gui.marks_store import load_marks
        surface = self._surface()
        self._armed(surface, [(30, 60)])
        with patch.object(live_scalogram_surface, "QInputDialog") as dlg:
            dlg.getText.return_value = ("flying", True)
            surface._save_detections()
        # The per-video file holds the palette and no spans of its own.
        self.assertEqual(load_marks(self.video)["spans"], {})

    def test_the_palette_is_written_per_video(self):
        """One behaviour is one colour across the whole clip, so the assignment
        lives beside the video rather than restarting inside each home."""
        from gui.marks_store import load_palette
        surface = self._surface()
        self._armed(surface, [(30, 60)])
        with patch.object(live_scalogram_surface, "QInputDialog") as dlg:
            dlg.getText.return_value = ("flying", True)
            surface._save_detections()
        self.assertIn("flying", load_palette(self.video))
        self.assertEqual(self._saved()["colors"], {})

    def test_provenance_records_the_rectangle_the_spans_describe(self):
        """Batch S slice 6 retires a geometry on a box move; a mark that cannot
        name its rectangle would go on being shown against a box that may have
        landed on a different animal."""
        surface = self._surface()
        self._armed(surface, [(30, 60)])
        with patch.object(live_scalogram_surface, "QInputDialog") as dlg:
            dlg.getText.return_value = ("flying", True)
            surface._save_detections()
        self.assertEqual(self._saved()["provenance"]["flying"]["frac"],
                         [0.0, 0.0, 1.0, 1.0])

    def test_cancelling_the_label_prompt_writes_nothing(self):
        from gui.marks_store import marks_path
        surface = self._surface()
        self._armed(surface, [(30, 60)])
        with patch.object(live_scalogram_surface, "QInputDialog") as dlg:
            dlg.getText.return_value = ("", False)
            surface._save_detections()
        self.assertFalse(os.path.exists(marks_path(self.video)))

    def test_a_blank_label_writes_nothing(self):
        from gui.marks_store import marks_path
        surface = self._surface()
        self._armed(surface, [(30, 60)])
        with patch.object(live_scalogram_surface, "QInputDialog") as dlg:
            dlg.getText.return_value = ("   ", True)
            surface._save_detections()
        self.assertFalse(os.path.exists(marks_path(self.video)))

    def test_no_current_detections_is_reported_not_written(self):
        from gui.marks_store import marks_path
        surface = self._surface()
        self._armed(surface, [])            # settings changed under the click
        with patch.object(live_scalogram_surface, "QInputDialog") as dlg:
            dlg.getText.return_value = ("flying", True)
            surface._save_detections()
        dlg.getText.assert_not_called()
        self.assertFalse(os.path.exists(marks_path(self.video)))
        self.assertIn("No current", surface.status_lbl.text())

    def test_the_prompt_prefills_the_last_used_label(self):
        surface = self._surface()
        self._armed(surface, [(30, 60)])
        with patch.object(live_scalogram_surface, "QInputDialog") as dlg:
            dlg.getText.return_value = ("wingbeat", True)
            surface._save_detections()
            self._armed(surface, [(30, 60)])
            surface._save_detections()
            # Second prompt is seeded with the first save's label.
            self.assertEqual(dlg.getText.call_args.kwargs.get("text"),
                             "wingbeat")


class ResumeWhereStoppedTests(_SurfaceTestCase):
    """Stopping parks the window at the playhead, so Play resumes there instead
    of snapping the cursor back to wherever the strip was last clicked."""

    def _stopped_at(self, surface, playhead: int):
        surface._explorer = MagicMock()
        surface._explorer.absolute_frame.return_value = playhead
        surface._stream_worker = MagicMock()
        surface.stop_stream()
        surface._on_stream_cancelled()

    def test_a_stop_parks_the_window_at_the_playhead(self):
        surface = self._surface()
        surface.len_spin.setValue(0.5)
        surface._on_seek_committed(4)             # the last click
        self._stopped_at(surface, 24)
        self.assertEqual(surface.start_slider.value(), 24)
        self.assertIn("stopped", surface.status_lbl.text())

    def test_parking_does_not_read_as_a_knob_edit(self):
        """The write goes through blockSignals: _on_window_changed would print
        a settings-changed status over the stop message in the same turn."""
        surface = self._surface()
        surface.len_spin.setValue(0.5)
        with patch.object(surface, "_on_window_changed") as changed:
            self._stopped_at(surface, 20)
        changed.assert_not_called()

    def test_a_restart_keeps_its_own_position(self):
        """Only a real stop reconciles the two. A restart already wrote the
        position it wants into the slider."""
        surface = self._surface()
        surface.len_spin.setValue(0.5)
        surface._on_seek_committed(10)
        surface._explorer = MagicMock()
        surface._explorer.absolute_frame.return_value = 30
        surface._stream_worker = MagicMock()
        surface.restart_stream()
        surface._on_stream_cancelled()
        self.assertEqual(surface.start_slider.value(), 10)

    def test_the_clamp_near_the_end_does_not_drag_the_cursor_back(self):
        """The window start clamps to leave a window's worth of clip, so near
        the end it lands behind the playhead -- the strip cursor must not follow
        it backwards."""
        surface = self._surface()
        surface.len_spin.setValue(0.5)
        surface.navigator.set_cursor(39)
        self._stopped_at(surface, 39)
        self.assertLess(surface.start_slider.value(), 39)
        self.assertEqual(surface.navigator.strip.cursor, 39)


class HiddenTabTests(_SurfaceTestCase):
    """Switching tabs must not leave a full-clip pass running, including via a
    restart that was armed moments earlier."""

    def test_hiding_disarms_a_pending_restart(self):
        """Regression. A knob or seek arms _restart_stream_at; hideEvent stops
        the worker; _on_stream_cancelled then consumes the flag WITHOUT checking
        visibility and starts another pass -- holding the decoder and a full
        ring on a tab nobody is looking at, which is what the stop exists to
        prevent."""
        surface = self._surface()
        surface._stream_worker = MagicMock()
        surface.restart_stream()
        self.assertIsNotNone(surface._restart_stream_at)

        surface.hideEvent(QHideEvent())
        self.assertIsNone(surface._restart_stream_at)

        surface._on_stream_cancelled()
        surface.start_stream.assert_not_called()


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
    """Play / Process become Stop while their own pass runs (todo 7)."""

    def test_buttons_toggle_to_stop_for_the_running_pass_only(self):
        surface = self._surface()
        surface._set_busy(_Busy.STREAM)
        self.assertEqual(surface.live_btn.text(), "Stop")
        self.assertTrue(surface.live_btn.isEnabled())
        self.assertFalse(surface.process_btn.isEnabled())

        surface._set_busy(_Busy.PROCESS)
        self.assertEqual(surface.process_btn.text(), "Stop")
        self.assertTrue(surface.process_btn.isEnabled())
        self.assertFalse(surface.live_btn.isEnabled())

        surface._set_busy(None)
        self.assertEqual(surface.live_btn.text(), live_scalogram_surface._LIVE_TEXT)
        self.assertEqual(surface.process_btn.text(), "Process whole video ▶")
        self.assertTrue(surface.live_btn.isEnabled())
        self.assertTrue(surface.process_btn.isEnabled())

    def test_clicking_stop_cancels_instead_of_starting_another_pass(self):
        surface = self._surface()
        surface._stream_worker = MagicMock()
        surface._on_live_clicked()
        surface._stream_worker.cancel.assert_called_once()
        surface.start_stream.assert_not_called()
        # Both stay disabled until the worker's `cancelled` actually lands.
        self.assertFalse(surface.live_btn.isEnabled())
        self.assertFalse(surface.process_btn.isEnabled())
        surface._stream_worker = None

    def test_space_starts_and_stops_the_live_pass(self):
        """Play IS the live pass: there is no separate playback of an already
        extracted window for Space to drive."""
        surface = self._surface()
        surface.toggle_playback()
        surface.start_stream.assert_called_once()

        surface._stream_worker = MagicMock()
        surface.toggle_playback()
        surface._stream_worker.cancel.assert_called_once()
        surface._stream_worker = None

    def test_a_restart_only_starts_once_the_stopped_pass_has_unwound(self):
        """The worker owns the decoder until it notices the cancel at its next
        frame, so a restart that started here would find it still held."""
        surface = self._surface()
        worker = MagicMock()
        surface._stream_worker = worker

        surface.restart_stream()
        worker.cancel.assert_called_once()
        surface.start_stream.assert_not_called()
        self.assertIsNotNone(surface._restart_stream_at)

        surface._on_stream_cancelled()
        surface.start_stream.assert_called_once()
        self.assertIsNone(surface._restart_stream_at)

    def test_a_real_stop_does_not_arm_a_restart(self):
        surface = self._surface()
        surface._stream_worker = MagicMock()
        surface.stop_stream()
        surface._on_stream_cancelled()
        surface.start_stream.assert_not_called()
        self.assertIn("stopped", surface.status_lbl.text())

    def test_a_pass_is_refused_while_the_whole_video_pass_owns_the_decoder(self):
        surface = self._surface()
        del surface.start_stream                 # exercise the real guard
        surface._proc_worker = MagicMock()

        surface.start_stream()

        self.assertIsNone(surface._stream_worker)
        surface._proc_worker = None


class CostSampleTests(_SurfaceTestCase):
    """Timing carried off a completed pass is what lets the downsample dialog
    price the lever from measurement instead of assertion.

    Every sample now arrives through the dialog's sweep. The surface used to
    contribute one per windowed extract and infer its regime from the resolved
    block; both went with the extract, so these drive `_on_sweep_row`, which
    states the regime outright from what the sweep was asked to run.
    """

    @staticmethod
    def _row(scale, block, wall, frames=100, spans=None, truncated=False):
        return ScalePass(scale=scale, block=block, grid=(4, 4), frames=frames,
                         wall=wall, spans=spans or {}, truncated=truncated)

    def _record(self, surface, scale, block, wall, regime=..., **kw):
        """One sweep row at a stated regime. ``regime`` defaults to "tracked"
        (None), which is what the `auto` block means and what production does."""
        surface._sweep_block_intent = None if regime is ... else regime
        surface._on_sweep_row(self._row(scale, block, wall, **kw))

    def test_sample_is_recorded_per_scale_and_block(self):
        surface = self._surface()
        self._record(surface, 1.0, 64, 4.0)
        self._record(surface, 0.5, 32, 2.0)
        self.assertEqual(set(surface._cost_samples), {(1.0, 64), (0.5, 32)})

    def test_a_pass_with_no_measured_wall_is_not_a_sample(self):
        # A zero-cost point drags the fitted decode floor toward zero, which is
        # what puts the knee where aggressive downsampling looks free.
        surface = self._surface()
        self._record(surface, 1.0, 64, 0.0)
        self._record(surface, 0.5, 32, 2.0, frames=0)
        self.assertEqual(surface._cost_samples, {})
        self.assertEqual(surface._cost_model(), (None, None))

    def test_a_truncated_pass_is_not_a_sample(self):
        """Its wall covers the frames it managed, so admitting it with the
        requested count understates seconds-per-frame -- the same direction of
        error as a zero-cost point."""
        surface = self._surface()
        self._record(surface, 1.0, 64, 4.0, truncated=True)
        self.assertEqual(surface._cost_samples, {})

    def test_re_recording_a_key_still_identifies_the_newest_regime(self):
        # dict order keeps a re-recorded key at its ORIGINAL position, so the
        # newest sample cannot be read off insertion order.
        surface = self._surface()
        self._record(surface, 1.0, 32, 4.0, regime=32)
        self._record(surface, 1.0, 32, 4.1, regime=32)
        self._record(surface, 0.5, 16, 2.0, regime=16)
        self.assertEqual(surface._last_cost_key, (0.5, 16))
        # Both are pinned, but to DIFFERENT blocks, so they are two regimes of
        # one sample each and neither can be fitted.
        model, _block = surface._cost_model()
        self.assertTrue(model.provisional)

    def test_model_stays_provisional_while_only_one_scale_is_measured(self):
        surface = self._surface()
        self._record(surface, 1.0, 64, 4.0, spans={"decode": 0.01})
        model, block = surface._cost_model()
        self.assertTrue(model.provisional)
        # ...and therefore declines to place a knee, which is the whole point:
        # one pass cannot see the decode floor.
        self.assertIsNone(model.knee_scale())

    def test_samples_at_different_blocks_are_not_fitted_together(self):
        surface = self._surface()
        self._record(surface, 0.5, 1, 9.0, regime=1)
        self._record(surface, 1.0, 64, 4.0)
        model, block = surface._cost_model()
        self.assertTrue(model.provisional)      # neither regime has two scales

    def test_fit_uses_the_regime_with_the_most_scales_not_the_newest(self):
        """Sweeps run at different pinned blocks accumulate across a session, so
        taking only the newest regime would leave the model provisional forever
        whenever the last sweep was a short one."""
        surface = self._surface()
        self._record(surface, 1.0, 1, 8.0, regime=1)
        self._record(surface, 0.5, 1, 3.0, regime=1)
        self._record(surface, 0.25, 64, 1.5, regime=64)   # newest, alone
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
        self._record(surface, 1.0, 1, 8.0, regime=1)
        self._record(surface, 0.5, 1, 3.0, regime=1)
        model, block = surface._cost_model()
        self.assertEqual(block, 1)
        self.assertFalse(model.provisional)

    def test_ties_between_regimes_go_to_the_newest(self):
        surface = self._surface()
        self._record(surface, 1.0, 1, 8.0, regime=1)     # pinned block 1
        self._record(surface, 0.5, 1, 3.0, regime=1)
        self._record(surface, 1.0, 64, 6.0)              # tracked (64x1.0)
        self._record(surface, 0.5, 32, 2.5)              # tracked (64x0.5)
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
            self._record(surface, scale, block, wall)
        model, block = surface._cost_model()
        self.assertFalse(model.provisional)
        self.assertEqual(model.n_samples, 3)
        # None, not a number: no single block describes these passes, and there
        # is no upper-bound caveat to make because this IS what production does.
        self.assertIsNone(block)
        self.assertIsNotNone(model.knee_scale())

    def test_a_pinned_block_sweep_reports_that_block(self):
        surface = self._surface()
        self._record(surface, 1.0, 16, 5.0, regime=16)
        self._record(surface, 0.5, 16, 2.0, regime=16)
        _model, block = surface._cost_model()
        self.assertEqual(block, 16)

    def test_a_pass_pinned_to_the_block_tracking_would_pick_stays_pinned(self):
        """The regime cannot be read off the resolved block. Pin Block to 64 and
        sweep: at scale 1.0 tracking would ALSO have chosen 64, so reading the
        regime back off the block would file the reference row as tracked, split
        the sweep into two groups and drop the widest, highest-leverage point
        from the fit. The sweep therefore states its intent."""
        surface = self._surface()
        for scale, wall in ((1.0, 6.0), (0.5, 3.0), (0.25, 2.0)):
            self._record(surface, scale, 64, wall, regime=64)
        model, block = surface._cost_model()
        self.assertEqual(block, 64)
        self.assertEqual(model.n_samples, 3)        # not 2


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

    def test_the_sweep_moves_the_model_out_of_a_pinned_regime(self):
        """A session can carry samples from an earlier sweep pinned to block 1,
        which overstates a batch run because it does no reduction. Running a
        tracked sweep must refit in the production regime rather than leaving
        the pinned numbers in force."""
        surface = self._surface()
        rec = CostSampleTests()._record
        rec(surface, 1.0, 1, 9.0, regime=1)
        rec(surface, 0.5, 1, 4.0, regime=1)
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
        """A sweep runs its own timed passes, so neither the live pass nor the
        whole-video commit may start underneath it."""
        surface = self._surface()
        surface._sweep_worker = MagicMock()
        del surface.start_stream            # exercise the real guard
        with patch("gui.explorers.live_scalogram_surface.LiveStreamWorker") as W:
            surface.start_stream()
            surface.process_whole_video()
        W.assert_not_called()
        self.assertIsNone(surface._stream_worker)
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


class ChannelGatingTests(_SurfaceTestCase):
    """The live pass computes what is selected, not every channel -- the 4.6x the
    'why is it slow' investigation turned up (see _channels_wanted)."""

    def test_default_wants_only_change_when_no_explorer_yet(self):
        surface = self._surface()
        want = surface._channels_wanted()
        self.assertIn("change", want)
        # change is the always-on precondition and also the default selection, so
        # a fresh surface with no view state computes exactly one channel.
        self.assertEqual(want, frozenset({"change"}))

    def test_selected_channel_from_saved_view_is_wanted(self):
        surface = self._surface()
        surface._pending_state = {"channel": "tensor speed"}
        want = surface._channels_wanted()
        self.assertIn("tensor_speed", want)
        self.assertIn("change", want)      # always carried; explorer needs it
        # But NOT the two flow-free channels nobody asked for.
        self.assertNotIn("appearance", want)
        self.assertNotIn("intensity", want)

    def test_all_channels_checkbox_wants_everything(self):
        surface = self._surface()
        surface.all_chan_chk.setChecked(True)
        self.assertEqual(surface._channels_wanted(),
                         frozenset(live_scalogram_surface.LIVE_CHANNELS))

    def test_all_channels_is_persisted_in_the_strip(self):
        surface = self._surface()
        surface.all_chan_chk.setChecked(True)
        self.assertTrue(surface._strip_values()["all_channels"])
        surface._apply_strip({"all_channels": False})
        self.assertFalse(surface.all_chan_chk.isChecked())

    def test_plan_wants_only_the_selected_channel(self):
        """The whole point, at the seam that matters: the ChannelPlan a live pass
        is built from carries the reduced set, so the stream never computes the
        flow solve for a channel nobody is reading."""
        surface = self._surface()
        surface._pending_state = {"channel": "change energy Jtt"}
        cfg = surface._build_cfg()
        _meta, plan, _cap = surface._stream_plan(cfg, 0)
        self.assertEqual(set(plan.want), {"change"})
        self.assertFalse(plan.need_flow)


class ProcessPlanTests(_SurfaceTestCase):
    """The gear beside Process: which spans a commit covers."""

    def test_default_plan_is_the_whole_clip_continuous(self):
        surface = self._surface()
        segs = surface._plan_segments()
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].start, 0)
        self.assertEqual(segs[0].stop, surface.frame_count)

    def test_from_here_starts_at_the_window(self):
        surface = self._surface()
        surface._process_settings = {"strategy": "from_here", "chunk_s": 30.0,
                                     "budget": 0.1, "skip_covered": False}
        surface.start_slider.setValue(5)
        segs = surface._plan_segments()
        self.assertEqual(segs[0].start, 5)

    def test_bisect_budget_samples_a_fraction(self):
        surface = self._surface()
        surface._process_settings = {"strategy": "bisect", "chunk_s": 0.2,
                                     "budget": 0.25, "skip_covered": False}
        segs = surface._plan_segments()
        total = sum(s.n for s in segs)
        self.assertLess(total, surface.frame_count)
        self.assertGreater(total, 0)

    def test_settings_round_trip_through_the_tuning_sidecar(self):
        surface = self._surface()
        surface._process_settings = {"strategy": "bisect", "chunk_s": 12.0,
                                     "budget": 0.2, "skip_covered": True}
        surface._save_tuning()
        reborn = LiveScalogramSurface(
            self.video, [{"id": 0, "label": "all", "frac": (0.0, 0.0, 1.0, 1.0)}])
        self.addCleanup(self._destroy, reborn)
        reborn.start_stream = MagicMock()
        self.assertEqual(reborn._process_settings["strategy"], "bisect")
        self.assertAlmostEqual(reborn._process_settings["budget"], 0.2)

    def test_process_bails_without_a_selected_replicate(self):
        """No stamp -> no idea what a segment would be computed under. The pass
        must refuse rather than write coverage under settings nobody chose."""
        surface = self._surface()
        surface._explorer = MagicMock()
        surface._explorer.detection_params.return_value = {"region_index": -1}
        with patch.object(live_scalogram_surface, "_ProcessWorker") as W:
            surface.process_whole_video()
        W.assert_not_called()
        self.assertIsNone(surface._proc_worker)


class PerReplicateTrackTest(_SurfaceTestCase):
    """Switching replicates is a handover, not a settings change.

    With one track per clip, selecting replicate 2 after replicate 1 wrote into
    the SAME (T,) arrays: the first animal's counts were overwritten frame for
    frame and only the stamp id changed, so the strip stayed plausible while the
    work was gone.
    """

    def _surface2(self):
        """Two replicates whose ids are deliberately not 0 and 1, so a test
        cannot pass by conflating a region index with a replicate id."""
        self._drop_tuning()
        self.addCleanup(self._drop_tuning)
        reps = [{"id": 9, "label": "b", "frac": (0.5, 0.0, 1.0, 1.0)},
                {"id": 5, "label": "a", "frac": (0.0, 0.0, 0.5, 1.0)}]
        surface = LiveScalogramSurface(self.video, reps)
        self.addCleanup(self._destroy, surface)
        surface.start_stream = MagicMock()
        return surface

    def test_region_index_maps_through_tile_order_not_position(self):
        s = self._surface2()
        # in_tile_order sorts by id, and the list above is in draw order.
        self.assertEqual(s._replicate_id_for(0), 5)
        self.assertEqual(s._replicate_id_for(1), 9)
        self.assertIsNone(s._replicate_id_for(2))

    def test_each_region_gets_its_own_track_object(self):
        s = self._surface2()
        s._activate_region(0)
        first = s._track
        s._activate_region(1)
        self.assertIsNot(s._track, first)
        s._activate_region(0)
        self.assertIs(s._track, first, "returning to a region must not re-create it")

    def test_a_region_switch_does_not_overwrite_the_other_regions_series(self):
        import numpy as np
        s = self._surface2()
        s._activate_region(0)
        s._track.count[100:150] = 7.0
        s._activate_region(1)
        s._track.count[100:150] = 3.0
        s._activate_region(0)
        np.testing.assert_allclose(s._track.count[100:150], 7.0)

    def test_a_switch_flushes_the_outgoing_track_to_its_own_folder(self):
        from core.live_track import TrackStamp
        from core.replicate_home import home_path
        import numpy as np
        s = self._surface2()
        s._activate_region(0)
        stamp = TrackStamp(channel="change", freq_band_hz=(1.0, 5.0),
                           grid=(4, 3), region_index=0, region_blocks=6,
                           downsample=1.0, block_size=8)
        s._track.set_stamp(stamp, region_grid=(2, 3,
                                               np.arange(6, dtype=np.int32) // 3,
                                               np.arange(6, dtype=np.int32) % 3))
        s._track.write(0, np.ones((20, 6), np.float32), trim=0)
        s._activate_region(1)          # the handover pays for the outgoing work
        self.assertTrue(os.path.exists(home_path(self.video, 5, ".track.npz")),
                        "replicate 5's track must land in replicate 5's folder")
        self.assertFalse(os.path.exists(home_path(self.video, 9, ".track.npz")))

    def test_nothing_is_written_before_a_region_is_chosen(self):
        """The placeholder track is all-uncovered, which is save_track's cue to
        DELETE a sidecar -- and with no replicate id the one it would delete is
        the legacy per-video file no replicate has adopted yet."""
        from gui.track_store import track_path
        s = self._surface2()
        with open(track_path(self.video), "wb") as f:
            f.write(b"not a real sidecar")
        self.addCleanup(lambda: os.path.exists(track_path(self.video))
                        and os.remove(track_path(self.video)))
        self.assertIsNone(s._active_region)
        self.assertFalse(s._save_track())
        self.assertTrue(os.path.exists(track_path(self.video)))

    def test_a_region_with_no_replicate_behind_it_is_never_written(self):
        """The error path must not reintroduce the bug. A stale remembered
        selection has no replicate id, and falling through to replicate_id=None
        would write one region's work straight back to the shared per-video
        file."""
        from gui.track_store import track_path
        s = self._surface2()
        s._activate_region(4)                     # only regions 0 and 1 exist
        self.assertIsNone(s._active_region, "must not activate a phantom region")
        s._active_region = 4                      # force the refusal path
        self.assertFalse(s._save_track())
        self.assertFalse(os.path.exists(track_path(self.video)))


if __name__ == "__main__":
    unittest.main()
