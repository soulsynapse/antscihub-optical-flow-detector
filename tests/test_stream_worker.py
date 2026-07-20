"""Batch Q slice 1: the continuous worker that fills the ring.

What is worth pinning here is not that frames arrive -- Batch P's tests already
own that -- but the three properties the GUI is about to depend on and that no
other test can see:

1. The island the worker publishes matches the island the buffer holds, including
   after eviction, where ``start`` moves and a naive frontier reading would claim
   history the ring no longer has.
2. A window request is served on the worker thread and answered by signal, with
   absolute indices. This is the seam that keeps ``StreamBuffer``'s
   check-then-copy race out of the GUI's reach.
3. The final ``advanced`` is exact regardless of the publish throttle, and
   ``pass_meta`` is readable after ``done``. "Done" and "still processing" are
   different UI states and the throttle must not blur them.
"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer

from core.channel_source import live_channel_source
from core.config import PipelineConfig
from core.detection import region_blocks_and_grid
from core.wavelet import band_indices, default_freqs, morlet_band_power
from core.stream_buffer import MIN_CAPACITY, capacity_for_budget
from core.tensor_channels import extract_channels_live, plan_channel_stream
from gui.stream_worker import LiveStreamWorker
from tests.test_channel_source import _full_frame_replicate, _write_moving_square

N_FRAMES = 40
_WANT = frozenset({"intensity", "change"})

# The two tests that react to a pass *while it runs* need a clip that takes long
# enough to run. At 40 frames the whole pass is ~0.1 s -- under one publish
# interval -- so the only `advanced` is the final one and a GUI-thread reaction
# to it arrives after the loop has already ended. That is an artifact of a
# synthetic clip, not of the worker: on real footage a pass is seconds. Rather
# than slow every test, the timing-sensitive pair get their own longer video.
N_LONG = 400


def _app():
    return QCoreApplication.instance() or QCoreApplication([])


class LiveStreamWorkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()
        cls._dir = tempfile.mkdtemp(prefix="qstream_")
        cls.video = os.path.join(cls._dir, "moving.mp4")
        cls.long_video = os.path.join(cls._dir, "moving_long.mp4")
        _write_moving_square(cls.video, n=N_FRAMES)
        _write_moving_square(cls.long_video, n=N_LONG)
        cls.cfg = PipelineConfig()
        cls.meta = live_channel_source(cls.video, cls.cfg,
                                       _full_frame_replicate()).meta
        cls.long_meta = live_channel_source(cls.long_video, cls.cfg,
                                            _full_frame_replicate()).meta

    @classmethod
    def tearDownClass(cls):
        for path in (cls.video, cls.long_video):
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            os.rmdir(cls._dir)
        except OSError:
            pass

    def _run_worker(self, worker, on_advanced=None, timeout_ms=30000):
        """Drive a worker to completion on a real event loop, collecting what it
        emits. A live loop rather than calling ``_run`` directly, because queued
        delivery across the thread boundary is the thing under test."""
        loop = QEventLoop()
        seen = {"advanced": [], "windows": [], "failed": None, "cancelled": False}
        worker.advanced.connect(lambda s, f, r: (
            seen["advanced"].append((s, f, r)),
            on_advanced(worker, s, f) if on_advanced else None))
        worker.window_ready.connect(seen["windows"].append)
        seen["detected"] = []
        worker.detected.connect(seen["detected"].append)
        worker.failed.connect(lambda m: seen.__setitem__("failed", m))
        worker.cancelled.connect(lambda: seen.__setitem__("cancelled", True))
        worker.finished.connect(loop.quit)
        guard = QTimer()
        guard.setSingleShot(True)
        guard.timeout.connect(loop.quit)
        guard.start(timeout_ms)
        worker.start()
        loop.exec()
        worker.wait(5000)
        self.assertIsNone(seen["failed"], seen["failed"])
        return seen

    def _plan(self, **kw):
        kw.setdefault("want", _WANT)
        return plan_channel_stream(self.meta, **kw)

    # -- the island ----------------------------------------------------------
    def test_the_frontier_reaches_the_end_of_the_plan(self):
        plan = self._plan(start=0, n=N_FRAMES)
        w = LiveStreamWorker(self.video, plan, capacity=N_FRAMES + 8)
        seen = self._run_worker(w)
        self.assertEqual(seen["advanced"][-1][:2], (0, N_FRAMES))
        self.assertFalse(w.pass_meta["truncated"])
        self.assertEqual(w.pass_meta["n_frames"], N_FRAMES)

    def test_an_evicting_ring_publishes_a_start_that_moves(self):
        # The property a progress bar would get wrong: past capacity the island
        # is no longer [0, frontier). A surface drawing history back to the
        # island's origin would be drawing rows the ring has already dropped.
        cap = MIN_CAPACITY
        plan = self._plan(start=0, n=N_FRAMES)
        w = LiveStreamWorker(self.video, plan, capacity=cap)
        seen = self._run_worker(w)
        start, frontier, _ = seen["advanced"][-1]
        self.assertEqual(frontier, N_FRAMES)
        self.assertEqual(start, max(0, N_FRAMES - cap))
        for s, f, _ in seen["advanced"]:
            self.assertLessEqual(f - s, cap)       # never claims more than it has

    def test_capacity_is_sized_from_the_plan(self):
        # Batch P's stated reason for ChannelPlan existing: the ring and the pass
        # filling it must not compute the geometry separately.
        plan = self._plan(start=0, n=N_FRAMES)
        cap = capacity_for_budget(4 * 1024 ** 2, len(plan.want), plan.ny, plan.nx)
        w = LiveStreamWorker(self.video, plan, capacity=cap)
        seen = self._run_worker(w)
        self.assertEqual(seen["advanced"][-1][1], N_FRAMES)

    # -- window requests -----------------------------------------------------
    def test_a_requested_window_comes_back_with_absolute_indices_and_real_data(self):
        plan = plan_channel_stream(self.long_meta, start=4, n=200, want=_WANT)
        w = LiveStreamWorker(self.long_video, plan, capacity=256,
                             publish_hz=1e6)
        # Parked BEFORE start, so it is served at the first frame boundary
        # whatever the machine does with the two threads -- the alternative
        # (request from an `advanced` handler) races a pass that finishes in
        # ~0.1 s on a 1x1 grid, and would be flaky rather than wrong. The
        # re-request keeps exercising the mid-pass path opportunistically.
        w.request_latest(8, ["change"], token="tick")
        seen = self._run_worker(w, on_advanced=lambda worker, s, f: (
            worker.request_latest(8, ["change"], token="tick")))
        self.assertTrue(seen["windows"], "no window was served")
        want = extract_channels_live(self.long_video, self.long_meta,
                                     start=4, n=200, channels=["change"])["change"]
        for payload in seen["windows"]:
            arr = payload["channels"]["change"]
            first = payload["first"]
            self.assertEqual(payload["token"], "tick")
            self.assertEqual(first + len(arr), payload["frontier"])
            self.assertGreaterEqual(first, 4)
            np.testing.assert_array_equal(arr, want[first - 4:first - 4 + len(arr)])

    def test_the_newest_request_replaces_an_unserved_one(self):
        # No queue, on purpose: the consumer is a repaint timer, and a backlog
        # would redraw stale windows the user has already scrolled away from.
        plan = self._plan(start=0, n=N_FRAMES)
        w = LiveStreamWorker(self.video, plan, capacity=64)
        w.request_latest(4, ["change"], token="old")
        w.request_latest(6, ["change"], token="new")
        seen = self._run_worker(w)
        tokens = [p["token"] for p in seen["windows"]]
        self.assertNotIn("old", tokens)
        self.assertIn("new", tokens)

    def test_a_request_never_outruns_the_frontier(self):
        # latest() clamps to what has actually arrived, so an early request gets
        # a short window rather than zero rows padded to look like data.
        plan = self._plan(start=0, n=N_FRAMES)
        w = LiveStreamWorker(self.video, plan, capacity=64)
        w.request_latest(10_000, ["change"], token="huge")
        seen = self._run_worker(w)
        for payload in seen["windows"]:
            arr = payload["channels"]["change"]
            self.assertEqual(payload["first"], 0)
            self.assertEqual(len(arr), payload["frontier"])

    def test_channels_stay_aligned_across_a_multi_channel_request(self):
        plan = self._plan(start=0, n=N_FRAMES)
        w = LiveStreamWorker(self.video, plan, capacity=64)
        w.request_latest(6, ["intensity", "change"])
        seen = self._run_worker(w)
        for payload in seen["windows"]:
            lens = {k: len(v) for k, v in payload["channels"].items()}
            self.assertEqual(len(set(lens.values())), 1, lens)

    # -- completion and cancellation -----------------------------------------
    def test_the_final_advance_is_emitted_despite_the_throttle(self):
        # publish_hz low enough that the throttle would swallow every tick; the
        # unconditional final emission is what makes "done" exact.
        plan = self._plan(start=0, n=N_FRAMES)
        w = LiveStreamWorker(self.video, plan, capacity=64, publish_hz=1e-6)
        seen = self._run_worker(w)
        self.assertEqual(seen["advanced"][-1][:2], (0, N_FRAMES))

    def test_cancelling_ends_on_cancelled_and_not_done(self):
        plan = plan_channel_stream(self.long_meta, start=0, n=N_LONG, want=_WANT)
        w = LiveStreamWorker(self.long_video, plan, capacity=256)
        done = []
        w.done.connect(done.append)
        seen = self._run_worker(w, on_advanced=lambda worker, s, f: worker.cancel())
        self.assertTrue(seen["cancelled"])
        self.assertEqual(done, [])
        # Stopped short: the point of cancelling is that the rest is not decoded.
        self.assertLess(seen["advanced"][-1][1], N_LONG)
        # And the decoder is released, so the next pass can open the file.
        again = LiveStreamWorker(self.video, self._plan(start=0, n=6),
                                 capacity=64)
        self.assertEqual(self._run_worker(again)["advanced"][-1][1], 6)

    # -- detection runs on the worker thread ---------------------------------
    def _detect(self, worker, n=32, band=(0.5, 8.0)):
        worker.request_detect(n, "change", self.meta, 0,
                              default_freqs(float(self.meta["fps"])), band,
                              token="d")

    def test_a_detection_request_returns_band_power_over_region_columns(self):
        plan = self._plan(start=0, n=N_FRAMES)
        w = LiveStreamWorker(self.video, plan, capacity=64)
        self._detect(w)
        seen = self._run_worker(w)
        self.assertTrue(seen["detected"])
        msg = seen["detected"][-1]
        # (T, B) over the region's block columns, with an ABSOLUTE first index --
        # the accumulator places it on the video's axis, not the ring's.
        self.assertEqual(msg["band_power"].ndim, 2)
        ny, nx = (int(v) for v in self.meta["grid"])
        self.assertEqual(msg["band_power"].shape[1], ny * nx)
        self.assertEqual(msg["token"], "d")
        self.assertGreaterEqual(msg["first"], 0)
        self.assertLessEqual(msg["first"] + msg["band_power"].shape[0],
                             msg["frontier"])

    def test_the_transform_matches_the_committed_detector_exactly(self):
        # The property the whole surface rests on: navigate to a detection the
        # live pass found, re-run the committed path over the same frames, and
        # get the same numbers. If these drifted, verifying a detection would
        # disprove it.
        plan = self._plan(start=0, n=N_FRAMES)
        w = LiveStreamWorker(self.video, plan, capacity=N_FRAMES + 8)
        self._detect(w, n=N_FRAMES)
        msg = self._run_worker(w)["detected"][-1]
        cd = live_channel_source(self.video, self.cfg, _full_frame_replicate(),
                                 start=0, n=N_FRAMES)
        fps = float(self.meta["fps"])
        blocks, _grid = region_blocks_and_grid(
            cd.meta, np.asarray(cd.channels["change"], np.float32), 0)
        i, j = band_indices(default_freqs(fps), 0.5, 8.0)
        want = morlet_band_power(blocks, fps, default_freqs(fps), i, j)
        got = msg["band_power"]
        n = min(len(want), len(got))
        np.testing.assert_allclose(got[-n:], want[-n:], rtol=1e-4, atol=1e-4)

    def test_a_one_frame_window_is_not_transformed(self):
        # A single sample is not a time series; transforming it would return a
        # padding response that the accumulator would then record as examined.
        plan = self._plan(start=0, n=N_FRAMES)
        w = LiveStreamWorker(self.video, plan, capacity=64)
        self._detect(w, n=1)
        for msg in self._run_worker(w)["detected"]:
            self.assertGreaterEqual(msg["band_power"].shape[0], 2)

    def test_the_tail_is_examined_after_the_decode_loop_ends(self):
        # Without the final serve, the last frames of every pass would stay
        # unexamined purely because of where the request timer last fired.
        plan = self._plan(start=0, n=N_FRAMES)
        w = LiveStreamWorker(self.video, plan, capacity=N_FRAMES + 8)
        self._detect(w, n=16)
        seen = self._run_worker(w)
        last = seen["detected"][-1]
        self.assertEqual(last["frontier"], N_FRAMES)
        self.assertEqual(last["first"] + last["band_power"].shape[0], N_FRAMES)

    def test_an_unknown_channel_is_dropped_rather_than_raising(self):
        plan = self._plan(start=0, n=N_FRAMES)
        w = LiveStreamWorker(self.video, plan, capacity=64)
        w.request_detect(16, "not_a_channel", self.meta, 0,
                         default_freqs(float(self.meta["fps"])), (0.5, 8.0))
        seen = self._run_worker(w)
        self.assertEqual(seen["detected"], [])
        self.assertIsNone(seen["failed"])


if __name__ == "__main__":
    unittest.main()
