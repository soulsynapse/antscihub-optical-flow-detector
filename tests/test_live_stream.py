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

from core.channel_source import ChannelData, LIVE_CHANNELS
from gui.explorers import live_scalogram_surface as lss
from gui.explorers.live_scalogram_surface import LiveScalogramSurface
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

    def test_follow_center_pins_the_cursor_to_the_middle_of_every_window(self):
        """The reading position sits half a window behind the frontier, so the
        plots carry computed footage on BOTH sides of the playhead. The centre
        has to be re-asserted per window: the span slides under it, so a cursor
        merely placed there once would be carried away by absolute index."""
        ex, cd = self._explorer(T=64)
        ex.follow_center()
        mid = (ex.T - 1) // 2
        self.assertEqual(ex.frame, mid)
        for start in (40, 90, 250):
            ex.set_channel_data(_slid(cd, window_start=start))
            self.assertEqual(ex.frame, (ex.T - 1) // 2)

    def test_follow_center_advances_the_ABSOLUTE_frame_as_the_window_slides(self):
        """The T-axis index is constant under follow_center, so `frame_moved`
        never fires -- and the whole-clip strip would sit still for the entire
        pass if it were the only route. absolute_frame is what actually moves."""
        ex, cd = self._explorer(T=64)
        ex.follow_center()
        seen = []
        for start in (0, 40, 90, 250):
            ex.set_channel_data(_slid(cd, window_start=start))
            seen.append(ex.absolute_frame())
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(len(set(seen)), len(seen))     # strictly advancing
        self.assertEqual(seen[-1], 250 + (ex.T - 1) // 2)

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

    def test_the_cube_cache_is_kept_across_the_update(self):
        """Cubes now OUTLIVE the span they were built over.

        They used to be cleared here, on the reasoning that a cube keyed only by
        (region, channel) cannot be told apart from a current one. That was right
        about the hazard and wrong about the remedy: once the per-frame plots run
        at 10 Hz against a transform that takes seconds, clearing means no cube
        ever survives long enough to be drawn and the densities stay empty for
        the whole pass. The span travels with the cube instead.
        """
        ex, cd = self._explorer(T=64)
        key = (0, ex.channel)
        ex._sg_cache[key] = np.zeros((4, 64, 4), np.float32)
        ex._sg_span[key] = (0, 64)
        gen = ex._data_gen
        ex.set_channel_data(_slid(cd, window_start=8))
        self.assertIn(key, ex._sg_cache)
        self.assertEqual(ex._sg_span[key], (0, 64))   # its OWN span, not the new one
        self.assertGreater(ex._data_gen, gen)

    def test_a_cube_from_before_the_update_is_tagged_with_its_own_span(self):
        """The race: a worker launched against the old span returns after the new
        one is in place. It is kept, but it must be recorded as covering the span
        it was actually transformed over -- that is what lets the density heatmap
        draw it in the right place instead of stretching it over the new span,
        which would misalign every column against the per-frame plots above it.
        """
        ex, cd = self._explorer(T=64)
        ex.active_region_index = 0
        ex._cube_gen = ex._data_gen                 # as _launch_worker stamps it
        ex._cube_span = (0, 64)                     # ...and the span it stamps
        ex.set_channel_data(_slid(cd, window_start=8))
        stale = np.zeros((len(ex.freqs), 64, 4), np.float32)
        with patch.object(ex, "_request_cube"):
            ex._on_cube_ready((0, ex.channel), stale)
        self.assertIn((0, ex.channel), ex._sg_cache)
        self.assertEqual(ex._sg_span[(0, ex.channel)], (0, 64))

    def test_a_lagging_cube_is_placed_at_its_own_span_not_stretched(self):
        """The registration the whole split turns on.

        These plots are stacked, share a widget width and are read down a
        column. A cube covering [0, 64) drawn edge-to-edge over a span that has
        grown to 128 would put its frame 40 above the traces' frame 80 -- an
        x-axis misregistration that invites reading a burst against the wrong
        value. It must be placed at its true offset with the rest hatched.
        """
        ex, cd = self._explorer(T=128)
        ex.active_region_index = 0
        key = (0, ex.channel)
        # A cube over only the first 64 frames of a 128-frame span.
        ex._sg_cache[key] = np.zeros((len(ex.freqs), 64, 4), np.float32)
        ex._sg_span[key] = (0, 64)
        ex._refresh_densities()
        dp = ex.density_plots[ex.channel]
        self.assertEqual(dp.matrix.shape[0], 64)     # not stretched to 128
        self.assertEqual(dp.axis_off, 0)
        self.assertEqual(dp.axis_total, 128)         # ...on the FULL axis

    def test_the_hatch_marks_the_span_not_the_frames_that_landed(self):
        """Coverage is a property of the SPAN, not of which columns happened to
        receive a frame.

        Deriving it from the frames breaks whenever the covered region holds
        fewer frames than it spans pixels -- 64 frames over 300 columns leave 236
        of them frameless -- and those columns were painted as unexamined,
        shredding real data into stripes. That is the hatch's own purpose
        inverted, and it is the normal case for any live window shorter than the
        plot is wide (~600 px), so a dense-span test cannot see it.
        """
        from gui.explorers.speed_explorer import DensityPlot
        w, h = 600, 120
        T, K, total = 64, 20, 128          # covers exactly the left half
        dp = DensityPlot("d")
        dp.set_matrix(np.random.default_rng(0).random((T, K), np.float32),
                      0, total)
        lo, hi = dp._data_range()
        img = dp._density_image(w, h, lo, hi)
        ptr = img.constBits()
        ptr.setsize(h * w * 3)
        a = np.frombuffer(ptr, np.uint8).reshape(h, w, 3).astype(int)
        # The hatch is blue-dominant; both the ramp and an empty bin are not.
        hatched = (a[:, :, 2] > a[:, :, 0] + 4).mean(axis=0) > 0.5
        self.assertEqual(hatched[:300].sum(), 0, "covered columns hatched")
        self.assertGreater(hatched[300:].sum(), 290, "uncovered columns bare")

    def test_a_cube_the_ring_has_partly_evicted_is_trimmed_not_slid(self):
        """window_start advances as the ring drops oldest frames, so a cube built
        earlier can start BEFORE the current span. Clamping that negative offset
        to zero would slide the whole matrix forward and draw old data over newer
        frames -- silently, and looking perfectly plausible."""
        ex, cd = self._explorer(T=64)
        ex.active_region_index = 0
        ex.set_channel_data(_slid(cd, window_start=32, T=64))
        key = (0, ex.channel)
        ex._sg_cache[key] = np.zeros((len(ex.freqs), 64, 4), np.float32)
        ex._sg_span[key] = (0, 64)          # covers absolute [0, 64); span is [32, 96)
        ex._refresh_densities()
        dp = ex.density_plots[ex.channel]
        # Only absolute [32, 64) overlaps: 32 frames, sitting at offset 0.
        self.assertEqual(dp.matrix.shape[0], 32)
        self.assertEqual(dp.axis_off, 0)
        self.assertEqual(dp.axis_total, 64)

    def test_a_retained_cube_does_not_stop_the_next_one_being_built(self):
        """Retention's own trap: once cubes outlive their span, "already cached"
        stops meaning "current". If a cache HIT short-circuits the request, the
        first cube of a live pass is the only one ever built and the density
        freezes at the opening window behind a forever-growing hatch -- which
        reads as slow progress, not as a stuck cache."""
        ex, cd = self._explorer(T=64)
        ex.active_region_index = 0
        key = (0, ex.channel)
        ex._sg_cache[key] = np.zeros((len(ex.freqs), 64, 4), np.float32)
        ex._sg_span[key] = (0, 64)
        self.assertTrue(ex._cube_is_current(key))
        # Patched across the update: set_channel_data rebuilds, which launches a
        # real worker, and _request_cube then serialises correctly against it --
        # so an unpatched probe afterwards measures the in-flight guard rather
        # than the staleness check this test is about.
        with patch.object(ex, "_launch_worker") as lw:
            # No build in flight: the constructor starts one, and _request_cube
            # serialises against it, so leaving it set would make this measure
            # the in-flight guard rather than the staleness check.
            ex._worker = None
            # The span moves on: the cached cube is now history, not the answer.
            ex.set_channel_data(_slid(cd, window_start=8, T=64))
            self.assertFalse(ex._cube_is_current(key))
            lw.assert_called_once_with(key)

    def test_the_live_path_does_not_re_percentile_the_overlay(self):
        """np.percentile is a full sort of T*B floats -- 53 ms at T=30k, B=377,
        on its own more than three times the rest of the fast path. Running it at
        10 Hz reintroduces exactly what _rebuild_selected_views froze it to
        avoid. A drifting normalization also makes the channel non-comparable by
        eye across time."""
        ex, cd = self._explorer(T=64)
        ex.active_region_index = 0
        ex.set_channel_data(_slid(cd, window_start=0, T=64))   # seeds it
        before = ex._ov_scale
        with patch("gui.explorers.scalogram_explorer.np.percentile") as pc:
            ex.set_channel_data(_slid(cd, window_start=8, T=64), live=True)
            pc.assert_not_called()
        self.assertEqual(ex._ov_scale, before)
        # ...but a non-live update still rescales.
        with patch("gui.explorers.scalogram_explorer.np.percentile",
                   return_value=1.0) as pc:
            ex.set_channel_data(_slid(cd, window_start=16, T=64))
            pc.assert_called()

    def test_the_first_live_update_still_sets_the_overlay_scale(self):
        """_ov_scale starts at EPS, so a pass whose every update took the live
        path would normalize the overlay by epsilon and saturate every block."""
        ex, cd = self._explorer(T=64)
        ex.active_region_index = 0
        ex._ov_scale_set = False
        ex.set_channel_data(_slid(cd, window_start=0, T=64), live=True)
        self.assertTrue(ex._ov_scale_set)
        self.assertGreater(ex._ov_scale, 0.0)

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


class HostedPlaybackTests(_QtTestCase):
    """Embedded in the live surface there is ONE play/pause on the tab and it
    belongs to the live pass. The video underneath is not something you start
    and stop -- it loops over whatever span is held, during a pass and after it.
    """

    def _explorer(self, own_shortcuts: bool):
        ex = ScalogramExplorer.from_channel_data(
            _channel_data(T=64), video_path=None,
            own_shortcuts=own_shortcuts, own_status=False)
        self.addCleanup(self._destroy, ex)
        return ex

    def _destroy(self, ex):
        ex.close()
        ex.deleteLater()
        self.app.processEvents()

    def test_hosted_playback_starts_itself_and_hides_its_button(self):
        ex = self._explorer(own_shortcuts=False)
        self.assertTrue(ex.playing)
        self.assertTrue(ex._timer.isActive())
        # isHidden, not isVisible: nothing here is ever shown, so isVisible is
        # False for both cases and would pass without the change.
        self.assertTrue(ex.play_btn.isHidden())

    def test_standalone_still_starts_paused_with_a_button(self):
        ex = self._explorer(own_shortcuts=True)
        self.assertFalse(ex.playing)
        self.assertFalse(ex._timer.isActive())
        self.assertFalse(ex.play_btn.isHidden())

    def test_hosted_declines_the_main_window_space_walk(self):
        """Regression: the explorer sits between the focus widget and the
        surface, so without the opt-out it wins the walk and Space toggles the
        video instead of the live pass."""
        self.assertFalse(self._explorer(own_shortcuts=False)
                         .space_toggles_playback)
        self.assertTrue(self._explorer(own_shortcuts=True)
                        .space_toggles_playback)


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
                self.assertTrue(s.process_btn.isEnabled())
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

    def test_an_update_is_no_longer_skipped_while_a_cube_builds(self):
        """The backpressure is GONE, and this pins why removing it was safe.

        It existed for a livelock: the cube worker cannot be cancelled, and a
        cube arriving after its span had moved used to be DISCARDED, so pushing a
        new span every tick meant every cube was dropped and relaunched and the
        scalogram never appeared at all. The discard is what has gone -- a cube
        is retained and tagged with the span it covers -- so a late cube is now
        late data correctly placed, and there is nothing to protect against.

        The gate was throttling the per-FRAME plots (~12 ms together at
        whole-clip length) to the rate of a per-BLOCK transform up to 500x slower
        that feeds none of them.
        """
        s = self._surface()
        del s._show_live_window                      # use the real one
        s._explorer = MagicMock()
        s._explorer.is_building.return_value = True
        # The explorer's grid must match the served window's, or _show_live_window
        # rebuilds instead of updating in place (a grid move can only be absorbed
        # by a rebuild -- see _put_on_screen). A live pass's windows all share the
        # explorer's grid, so the mock declares it.
        s._explorer.ny, s._explorer.nx = 1, 1
        s._stream_meta = {"fps": 30.0, "grid": [1, 1]}
        s._show_live_window(self._win(64, token=1))
        s._explorer.set_channel_data.assert_called_once()
        # ...and it goes down the live path, which skips the overlay percentile.
        self.assertIs(s._explorer.set_channel_data.call_args.kwargs["live"], True)
        s._explorer = None

    def test_the_terminal_update_lands_even_while_a_cube_builds(self):
        s = self._surface()
        del s._show_live_window
        s._explorer = MagicMock()
        s._explorer.is_building.return_value = True
        s._explorer.ny, s._explorer.nx = 1, 1       # match the served grid
        s._stream_meta = {"fps": 30.0, "grid": [1, 1]}
        s._show_live_window(self._win(64, token=1), force=True)
        s._explorer.set_channel_data.assert_called_once()
        s._explorer = None

    def test_a_grid_change_rebuilds_rather_than_crashing(self):
        """A Block/Downsample change replans the pass onto a new grid, and the
        first window of the restarted pass then reaches an explorer built for the
        OLD grid. set_channel_data refuses a grid move (it cannot remap regions
        or the cube budget in place), so _show_live_window must REBUILD instead
        -- the crash this replaced was that window hitting set_channel_data with
        a (113,115) grid against an (8,8) explorer."""
        s = self._surface()
        del s._show_live_window                      # use the real one
        s._explorer = MagicMock()
        s._explorer.ny, s._explorer.nx = 8, 8        # the old geometry
        s._stream_meta = {"fps": 30.0, "grid": [113, 115]}   # the new one
        s._swap_explorer = MagicMock()
        s._arm_detection = MagicMock()
        s._show_live_window(self._win(64, token=1))
        s._swap_explorer.assert_called_once()
        s._explorer.set_channel_data.assert_not_called()
        s._arm_detection.assert_called_once()
        s._explorer = None

    def test_a_truncated_pass_marks_the_span_it_left_on_screen(self):
        """The status line is not enough: anything reading cd.meta -- detection
        included -- would see a clean window."""
        s = self._surface()
        del s._show_live_window
        s._explorer = MagicMock()
        s._explorer.is_building.return_value = False
        s._explorer.ny, s._explorer.nx = 1, 1       # match the served grid
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


class OnDemandChannelSelectabilityTests(_QtTestCase):
    """A live pass carries only the selected channel (+ _ALWAYS_STREAM), so the
    others are absent until a replan. They must stay SELECTABLE anyway -- checking
    one is what tells the surface to compute it -- rather than being grayed into a
    catch-22 where a channel can never be picked because it is not carried and is
    not carried because it was never picked."""

    def _explorer(self, present, on_demand):
        base = _channel_data()
        cd = ChannelData(
            meta=base.meta,
            channels={k: v for k, v in base.channels.items() if k in present},
            window_start=base.window_start)
        ex = ScalogramExplorer.from_channel_data(
            cd, video_path=None, own_shortcuts=False, own_status=False,
            on_demand_channels=on_demand)
        self.addCleanup(self._destroy, ex)
        return ex, base   # the FULL data, for feeding a replan that lands a channel

    def _destroy(self, ex):
        ex.close()
        ex.deleteLater()
        self.app.processEvents()

    def _name(self, attr):
        from gui.explorers.scalogram_explorer import CHANNELS
        return next(n for n, (a, _c) in CHANNELS.items() if a == attr)

    def test_an_absent_on_demand_channel_stays_selectable(self):
        ex, _ = self._explorer(present=("change",), on_demand=LIVE_CHANNELS)
        cb = ex.chan_checks[self._name("appearance")]
        self.assertTrue(cb.isEnabled())
        self.assertIn("restarts the live pass", cb.toolTip())

    def test_a_carried_channel_reads_as_ready_to_detect(self):
        ex, _ = self._explorer(present=("change",), on_demand=LIVE_CHANNELS)
        cb = ex.chan_checks[self._name("change")]
        self.assertTrue(cb.isEnabled())
        self.assertIn("builds its cube", cb.toolTip())

    def test_a_channel_no_source_can_produce_stays_disabled(self):
        """The guard is not gone, only narrowed: a channel that is neither carried
        nor producible on demand is still a dead control and reads as one."""
        ex, _ = self._explorer(present=("change",), on_demand=())
        cb = ex.chan_checks[self._name("appearance")]
        self.assertFalse(cb.isEnabled())
        self.assertIn("not carried by this source", cb.toolTip())

    def test_a_replan_landing_the_channel_flips_the_tooltip(self):
        """The tooltip is a promise the pass has yet to keep; once the replanned
        window carries the channel, it becomes the plain build hint."""
        ex, base = self._explorer(present=("change",), on_demand=LIVE_CHANNELS)
        name = self._name("appearance")
        self.assertIn("restarts the live pass", ex.chan_checks[name].toolTip())
        ex.set_channel_data(base, live=True)          # now carries appearance
        self.assertTrue(ex.chan_checks[name].isEnabled())
        self.assertIn("builds its cube", ex.chan_checks[name].toolTip())


if __name__ == "__main__":
    unittest.main()
