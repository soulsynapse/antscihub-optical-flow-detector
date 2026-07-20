"""The continuous live surface (todo.md Batch Q, slice 2).

Two halves, tested together because the contract between them is the point: the
explorer learns to accept a new span under a FIXED geometry, and the surface
drives it from a ``LiveStreamWorker`` at ~1 Hz.

The failures worth pinning here are all of one shape -- a plot that renders and
does not mean what it appears to. A one-frame window transformed as if it were a
time series; a scalogram cube built over a span the user has already left; a
cursor that looks stationary while drifting backwards through the clip. Each of
those was reachable in the first cut of this slice and each is silent.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

import numpy as np
from PyQt6.QtWidgets import QApplication

from core.channel_source import ChannelData
from gui.explorers import live_scalogram_surface as lss
from gui.explorers.live_scalogram_surface import _Busy, LiveScalogramSurface
from gui.explorers.scalogram_explorer import ScalogramExplorer
from tests.test_channel_source import _write_moving_square
from tests.test_view_state import _channel_data


def _slid(cd: ChannelData, window_start: int, T: int | None = None) -> ChannelData:
    """The same geometry over a later span -- what a trailing window becomes one
    tick later."""
    T = cd.channels["change"].shape[0] if T is None else T
    ny, nx = map(int, cd.meta["grid"])
    rng = np.random.default_rng(window_start)
    chans = {k: rng.random((T, ny, nx), dtype=np.float32) + 0.1
             for k in cd.channels}
    meta = {**cd.meta, "n_frames": T, "window_start": window_start}
    return ChannelData(meta=meta, channels=chans, window_start=window_start)


class _QtTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])


class SetChannelDataTests(_QtTestCase):
    """The explorer accepting a new span in place."""

    def _explorer(self, **kw):
        cd = _channel_data(**kw)
        ex = ScalogramExplorer.from_channel_data(
            cd, video_path=None, own_shortcuts=False, own_status=False)
        self.addCleanup(self._destroy, ex)
        return ex, cd

    def _destroy(self, ex):
        ex.close()
        ex.deleteLater()
        self.app.processEvents()

    def test_refuses_a_grid_change(self):
        """A geometry change is a REBUILD, not an update: the constructor
        re-derives grid, regions, block snap and cube budget together, and
        mutating a subset of those in place is how they drift apart."""
        ex, _ = self._explorer(ny=4, nx=4)
        with self.assertRaises(ValueError) as cm:
            ex.set_channel_data(_channel_data(ny=8, nx=8))
        self.assertIn("grid", str(cm.exception))

    def test_refuses_an_fps_change(self):
        """self.freqs and every cached cube are derived from fps, so accepting a
        new one would leave the frequency axis labelling the old rate."""
        ex, cd = self._explorer()
        other = _channel_data()
        other.meta["fps"] = 60.0
        with self.assertRaises(ValueError) as cm:
            ex.set_channel_data(other)
        self.assertIn("fps", str(cm.exception))

    def test_requires_the_change_channel(self):
        ex, cd = self._explorer()
        thin = _channel_data()
        del thin.channels["change"]
        with self.assertRaises(ValueError):
            ex.set_channel_data(thin)

    def test_a_followed_cursor_keeps_following(self):
        """The live default. follow_latest parks on the newest frame, and every
        slide after that must stay there rather than being carried by absolute
        index into the middle of the window."""
        ex, cd = self._explorer(T=64)
        ex.follow_latest()
        self.assertEqual(ex.frame, 63)
        ex.set_channel_data(_slid(cd, window_start=40))
        self.assertEqual(ex.frame, ex.T - 1)
        ex.set_channel_data(_slid(cd, window_start=90))
        self.assertEqual(ex.frame, ex.T - 1)

    def test_a_parked_cursor_holds_its_ABSOLUTE_frame(self):
        """Scrub away from the frontier and the cursor must stay on the video
        frame it was put on, not on the T-axis index -- those diverge the moment
        the window slides, and holding the index walks the cursor backwards
        through the clip at the eviction rate while appearing to sit still."""
        ex, cd = self._explorer(T=64)          # window_start 0, so abs == index
        ex._update_frame(50)
        self.assertEqual(ex.window_start, 0)
        ex.set_channel_data(_slid(cd, window_start=20))
        # Absolute frame 50 is index 30 of a window starting at 20.
        self.assertEqual(ex.window_start, 20)
        self.assertEqual(ex.frame, 30)

    def test_a_cursor_the_window_has_slid_past_is_clamped(self):
        ex, cd = self._explorer(T=64)
        ex._update_frame(10)
        ex.set_channel_data(_slid(cd, window_start=200))   # 10 is long gone
        self.assertEqual(ex.frame, 0)

    def test_the_cube_cache_is_dropped_and_the_generation_bumped(self):
        """Cubes are keyed by (region, channel) and carry no span, so one built
        over the previous window is indistinguishable from a current one."""
        ex, cd = self._explorer(T=64)
        ex._sg_cache[(0, ex.channel)] = np.zeros((4, 64, 4), np.float32)
        gen = ex._data_gen
        ex.set_channel_data(_slid(cd, window_start=8))
        self.assertEqual(len(ex._sg_cache), 0)
        self.assertGreater(ex._data_gen, gen)

    def test_a_cube_from_before_the_update_is_dropped_not_cached(self):
        """The race the generation exists for: a worker launched against the old
        span returns after the new one is in place. Caching it would serve a
        scalogram of a span the user has already left, for as long as the
        selection stays put -- and it would be blamed on the transform."""
        ex, cd = self._explorer(T=64)
        ex.active_region_index = 0
        ex._cube_gen = ex._data_gen                 # as _launch_worker stamps it
        ex.set_channel_data(_slid(cd, window_start=8))
        stale = np.zeros((len(ex.freqs), 64, 4), np.float32)
        with patch.object(ex, "_request_cube"):
            ex._on_cube_ready((0, ex.channel), stale)
        self.assertNotIn((0, ex.channel), ex._sg_cache)

    def test_the_frame_count_readout_is_not_left_on_the_first_window(self):
        ex, cd = self._explorer(T=64)
        self.assertIn("64 frames", ex._info_lbl.text())
        ex.set_channel_data(_slid(cd, window_start=8, T=32))
        self.assertIn("32 frames", ex._info_lbl.text())
        self.assertEqual(ex.scrub.maximum(), 31)

    def test_the_tuning_survives_the_update(self):
        """The whole reason this is an update and not a rebuild: bands are the
        tuning, and losing them once a second would make the surface unusable
        (the T17 failure, at 1 Hz)."""
        ex, cd = self._explorer(T=64)
        ex.count_w_plot.band_lo, ex.count_w_plot.band_hi = 3.0, float("inf")
        ex.scalo_plot.band_lo, ex.scalo_plot.band_hi = 1.0, 5.0
        ex.set_channel_data(_slid(cd, window_start=8))
        self.assertEqual(ex.count_w_plot.band_lo, 3.0)
        self.assertEqual((ex.scalo_plot.band_lo, ex.scalo_plot.band_hi), (1.0, 5.0))


class _SurfaceTestCase(_QtTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import tempfile
        cls._dir = tempfile.mkdtemp(prefix="live_stream_")
        cls.video = os.path.join(cls._dir, "moving.mp4")
        _write_moving_square(cls.video, n=60)

    def _surface(self):
        reps = [{"id": 0, "label": "all", "frac": (0.0, 0.0, 1.0, 1.0)}]
        with patch("gui.explorers.live_scalogram_surface.QTimer.singleShot"):
            s = LiveScalogramSurface(self.video, reps)
        self.addCleanup(self._destroy, s)
        s._show_live_window = MagicMock()
        return s

    def _destroy(self, s):
        s.close()
        s.deleteLater()
        self.app.processEvents()

    def _win(self, n_frames: int, token, first: int = 0) -> dict:
        return {"first": first, "frontier": first + n_frames, "token": token,
                "channels": {"change": np.zeros((n_frames, 1, 1), np.float32)}}


class StreamWiringTests(_SurfaceTestCase):

    def test_a_pass_already_owning_the_decoder_blocks_the_stream(self):
        s = self._surface()
        s._proc_worker = object()
        s.start_stream()
        self.assertIsNone(s._stream_worker)
        s._proc_worker = None

    def test_a_window_from_a_superseded_pass_is_ignored(self):
        """A request parked before a restart is serviced against the OLD island
        and arrives after the new one has begun. Rendering it would put frames
        from a different set of knobs on screen, looking merely stale."""
        s = self._surface()
        s._stream_worker = MagicMock()
        s._live_request_n = 40
        s._stream_token = 7
        s._on_window_ready(self._win(40, token=6))
        s._show_live_window.assert_not_called()
        s._on_window_ready(self._win(40, token=7))
        s._show_live_window.assert_called_once()

    def test_a_window_below_the_floor_is_not_drawn(self):
        """The first tick of a pass routinely lands one or two frames, and the
        explorer derives its whole time axis from what it is handed: a Morlet
        transform over a single sample renders and means nothing."""
        s = self._surface()
        s._stream_worker = MagicMock()
        s._stream_token = 1
        s._live_request_n = 200
        s._on_window_ready(self._win(1, token=1))
        s._on_window_ready(self._win(lss._MIN_LIVE_FRAMES - 1, token=1))
        s._show_live_window.assert_not_called()
        s._on_window_ready(self._win(lss._MIN_LIVE_FRAMES, token=1))
        s._show_live_window.assert_called_once()

    def test_a_deliberately_short_request_is_still_drawn(self):
        """The floor is min(requested, MIN_LIVE_FRAMES), never flat: a request
        shorter than the floor is a short window the user asked for, and
        refusing to draw it at all would be worse than drawing it late."""
        s = self._surface()
        s._stream_worker = MagicMock()
        s._stream_token = 1
        s._live_request_n = 12
        s._on_window_ready(self._win(12, token=1))
        s._show_live_window.assert_called_once()

    def test_every_terminal_signal_stops_the_request_timer(self):
        """The timer parks requests on the worker, so one surviving the worker
        would touch a thread on its way out."""
        for end, label in ((lambda s: s._on_stream_done({"n_frames": 9}), "done"),
                           (lambda s: s._on_stream_failed("boom"), "failed"),
                           (lambda s: s._on_stream_cancelled(), "cancelled")):
            with self.subTest(label):
                s = self._surface()
                s._stream_worker = MagicMock()
                s._stream_timer.start()
                end(s)
                self.assertFalse(s._stream_timer.isActive())
                self.assertIsNone(s._stream_worker)
                self.assertTrue(s.extract_btn.isEnabled())
                self.assertEqual(s.live_btn.text(), lss._LIVE_TEXT)

    def test_a_truncated_pass_does_not_read_as_complete(self):
        """FINDINGS.md section 15: the decoder stopping early is a data loss, and
        reporting it as a finished island is the claim that section exists
        about."""
        s = self._surface()
        s._stream_worker = MagicMock()
        s._on_stream_done({"n_frames": 120, "truncated": True})
        self.assertIn("early", s.status_lbl.text())
        self.assertNotIn("complete", s.status_lbl.text())

    def test_the_readout_names_which_side_of_realtime_the_pass_is_on(self):
        """The measured drop from ~80 fps to ~18 fps when `appearance` joins the
        pass is the difference between running ahead of playback and falling
        behind it, and a bare fps number does not say which."""
        s = self._surface()
        s._stream_worker = MagicMock()
        s._stream_worker.is_cancelled.return_value = False
        s._stream_plan_obj = MagicMock(start=0, n=1000)
        s.fps = 60.0
        s._on_advanced(0, 300, 18.0)
        self.assertIn("behind playback", s.status_lbl.text())
        s._on_advanced(0, 300, 200.0)
        self.assertNotIn("behind playback", s.status_lbl.text())

    def test_an_update_is_skipped_while_the_last_cube_is_still_building(self):
        """The livelock this backpressure exists for: the cube worker cannot be
        cancelled and will not relaunch while one is in flight, so pushing a new
        span every tick makes every cube arrive against a newer generation, get
        dropped, and relaunch -- the scalogram then never appears AT ALL, rather
        than appearing late."""
        s = self._surface()
        del s._show_live_window                      # use the real one
        s._explorer = MagicMock()
        s._explorer.is_building.return_value = True
        s._stream_meta = {"fps": 30.0, "grid": [1, 1]}
        s._show_live_window(self._win(64, token=1))
        s._explorer.set_channel_data.assert_not_called()
        s._explorer.is_building.return_value = False
        s._show_live_window(self._win(64, token=1))
        s._explorer.set_channel_data.assert_called_once()
        s._explorer = None

    def test_the_terminal_update_lands_even_while_a_cube_builds(self):
        s = self._surface()
        del s._show_live_window
        s._explorer = MagicMock()
        s._explorer.is_building.return_value = True
        s._stream_meta = {"fps": 30.0, "grid": [1, 1]}
        s._show_live_window(self._win(64, token=1), force=True)
        s._explorer.set_channel_data.assert_called_once()
        s._explorer = None

    def test_a_truncated_pass_marks_the_span_it_left_on_screen(self):
        """The status line is not enough: anything reading cd.meta -- detection
        included -- would see a clean window."""
        s = self._surface()
        del s._show_live_window
        s._explorer = MagicMock()
        s._explorer.is_building.return_value = False
        s._stream_meta = {"fps": 30.0, "grid": [1, 1]}
        s._stream_worker = MagicMock()
        s._live_window = self._win(64, token=1)
        s._on_stream_done({"n_frames": 64, "truncated": True})
        cd = s._explorer.set_channel_data.call_args[0][0]
        self.assertTrue(cd.meta["truncated"])
        s._explorer = None

    def test_hiding_the_tab_stops_the_live_pass(self):
        """A live pass runs to the end of the clip, so unlike the extract it does
        not self-terminate: left running it holds the decoder for minutes."""
        s = self._surface()
        s.show()                    # hideEvent does not fire on a never-shown widget
        self.app.processEvents()
        worker = MagicMock()
        s._stream_worker = worker
        s._stream_timer.start()
        s.hide()
        self.app.processEvents()
        worker.cancel.assert_called_once()
        self.assertFalse(s._stream_timer.isActive())
        s._stream_worker = None

    def test_stopping_halts_the_timer_before_the_worker_unwinds(self):
        """cancel() only sets a flag; the worker notices at its next frame. The
        timer must not keep parking requests on it in between."""
        s = self._surface()
        s._stream_worker = MagicMock()
        s._stream_timer.start()
        s.stop_stream()
        self.assertFalse(s._stream_timer.isActive())
        s._stream_worker = None

    def test_eviction_is_reported_rather_than_the_island_claiming_its_origin(self):
        """`advanced` publishes `start` precisely because past capacity the
        island is no longer [plan.start, frontier)."""
        s = self._surface()
        s._stream_worker = MagicMock()
        s._stream_worker.is_cancelled.return_value = False
        s._stream_plan_obj = MagicMock(start=0, n=10000)
        s._on_advanced(0, 500, 90.0)
        self.assertNotIn("dropped", s.status_lbl.text())
        s._on_advanced(400, 5000, 90.0)
        self.assertIn("dropped", s.status_lbl.text())


if __name__ == "__main__":
    unittest.main()
