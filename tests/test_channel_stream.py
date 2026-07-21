"""Batch P: the extraction loop as a generator, so a live surface can render
frames as they arrive instead of extracting a window and blocking on it.

The load-bearing property is the first test's, and it is the same one Batch M and
T18 each had to pin: **the refactor must not move a number**. A streaming path
that is a little bit different from the windowed one would be worse than no
streaming path at all, because the windowed pass is what the sidecar, the cost
model and every recorded finding were measured against.

The second cluster pins the contract ``core.stream_buffer.StreamBuffer`` depends
on -- absolute, contiguous indices and a complete channel dict per frame -- since
that buffer rejects a non-contiguous append precisely so this generator cannot
silently produce a span whose time axis is a lie.
"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from core.channel_source import live_channel_source
from core.config import PipelineConfig
from core.stream_buffer import StreamBuffer
from core.tensor_channels import (CHANNELS, extract_channels_live,
                                  plan_channel_stream, stream_channel_planes)
from tests.test_channel_source import (_full_frame_replicate,
                                       _two_replicates, _write_moving_square)

N_FRAMES = 40


class ChannelStreamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.mkdtemp(prefix="chstream_")
        cls.video = os.path.join(cls._dir, "moving.mp4")
        _write_moving_square(cls.video, n=N_FRAMES)
        cls.cfg = PipelineConfig()
        cls.meta = live_channel_source(cls.video, cls.cfg,
                                       _full_frame_replicate()).meta

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove(cls.video)
            os.rmdir(cls._dir)
        except OSError:
            pass

    def _drain(self, plan, **kw):
        """Every yielded frame plus the generator's return value."""
        gen = stream_channel_planes(self.video, plan, **kw)
        frames = []
        while True:
            try:
                frames.append(next(gen))
            except StopIteration as stop:
                return frames, stop.value

    # -- equivalence -------------------------------------------------------
    def test_streamed_planes_are_bit_identical_to_the_windowed_array(self):
        # The whole justification for the refactor being safe. Exact equality,
        # not allclose: the streaming path runs the SAME arithmetic in the same
        # order, so any tolerance here would be hiding a real divergence.
        for start, n in ((0, N_FRAMES), (5, 12), (N_FRAMES - 3, 3)):
            with self.subTest(start=start, n=n):
                want = extract_channels_live(self.video, self.meta,
                                             start=start, n=n)
                plan = plan_channel_stream(self.meta, start=start, n=n)
                frames, meta = self._drain(plan)
                self.assertEqual(len(frames), want["meta"]["n_frames"])
                for i, planes in frames:
                    for k in plan.want:
                        np.testing.assert_array_equal(
                            planes[k], want[k][i - plan.start],
                            err_msg=f"channel {k} frame {i}")
                for key in ("fps", "block", "grid", "n_frames", "truncated",
                            "window_start", "channel_version", "approximated",
                            "channels_computed"):
                    self.assertEqual(meta[key], want["meta"][key], key)

    def test_equivalence_holds_across_tiles_and_channel_selection(self):
        # Two replicates exercise atlas packing: each tile writes its own bbox
        # into a plane the consumer never assembled, which is the one place a
        # per-frame dict could differ from a preallocated slab.
        meta = live_channel_source(self.video, self.cfg,
                                   _two_replicates()).meta
        for channels in (["change"], ["intensity", "change"],
                         ["appearance"], list(CHANNELS)):
            with self.subTest(channels=channels):
                want = extract_channels_live(self.video, meta, start=2, n=10,
                                             channels=channels)
                plan = plan_channel_stream(meta, start=2, n=10,
                                           want=frozenset(channels))
                frames, _ = self._drain(plan)
                for i, planes in frames:
                    self.assertEqual(set(planes), set(plan.want))
                    for k in plan.want:
                        np.testing.assert_array_equal(planes[k],
                                                      want[k][i - 2])

    # -- the StreamBuffer contract -----------------------------------------
    def test_indices_are_absolute_and_contiguous_from_the_plan_start(self):
        plan = plan_channel_stream(self.meta, start=7, n=9)
        frames, _ = self._drain(plan)
        self.assertEqual([i for i, _ in frames], list(range(7, 16)))

    def test_planes_append_straight_into_a_stream_buffer(self):
        # The integration this batch exists for: no adapter between the two.
        # StreamBuffer.append raises on a non-contiguous index or a wrong shape,
        # so this passing IS the shape/ordering assertion.
        plan = plan_channel_stream(self.meta, start=4, n=16,
                                   want=frozenset({"intensity", "change"}))
        buf = StreamBuffer(sorted(plan.want), plan.ny, plan.nx,
                           capacity=64, start=plan.start)
        for i, planes in stream_channel_planes(self.video, plan):
            buf.append(i, planes)
        self.assertEqual((buf.start, buf.frontier), (4, 20))
        want = extract_channels_live(self.video, self.meta, start=4, n=16,
                                     channels=["intensity", "change"])
        np.testing.assert_array_equal(buf.window(4, 20, "change"),
                                      want["change"])

    def test_every_wanted_channel_is_present_on_every_frame(self):
        # Including the first frame of a from-zero window, where there is no
        # previous frame and the motion channels are legitimately zero. A
        # consumer must never have to branch on which keys a frame carries --
        # StreamBuffer.append rejects a frame missing one outright.
        plan = plan_channel_stream(self.meta, start=0, n=4)
        frames, _ = self._drain(plan)
        for _, planes in frames:
            self.assertEqual(set(planes), set(plan.want))
            for k in plan.want:
                self.assertEqual(planes[k].shape, (plan.ny, plan.nx))
        self.assertTrue(np.all(frames[0][1]["change"] == 0))

    def test_a_window_not_starting_at_zero_measures_motion_on_its_first_frame(self):
        # The seed frame: the pass decodes one frame before the window so the
        # first STORED frame carries motion, and that seed is not yielded.
        plan = plan_channel_stream(self.meta, start=6, n=4)
        frames, _ = self._drain(plan)
        self.assertEqual(frames[0][0], 6)
        self.assertTrue(np.any(frames[0][1]["change"] > 0))

    # -- the plan ----------------------------------------------------------
    def test_the_plan_clamps_the_window_to_the_video(self):
        # The clamp lives in exactly one place now, which is the point: a buffer
        # sized from the plan and the pass filling it cannot disagree about how
        # many frames are coming.
        plan = plan_channel_stream(self.meta, start=N_FRAMES - 5, n=1000)
        self.assertEqual((plan.start, plan.n, plan.stop),
                         (N_FRAMES - 5, 5, N_FRAMES))
        frames, _ = self._drain(plan)
        self.assertEqual(len(frames), 5)

    def test_an_empty_window_yields_nothing_and_is_not_truncated(self):
        # "Asked for nothing and got nothing" must stay distinguishable from
        # "the decoder stopped early", since only the second is a data loss.
        plan = plan_channel_stream(self.meta, start=3, n=0)
        frames, meta = self._drain(plan)
        self.assertEqual(frames, [])
        self.assertEqual(meta["n_frames"], 0)
        self.assertFalse(meta["truncated"])

    def test_an_empty_channel_set_is_refused_by_the_plan(self):
        # ``live_channel_source`` already refuses this, but that guard is a layer
        # above and a streaming consumer calls the plan directly -- so without
        # this the new entry point routes around it, and a full-cost pass returns
        # frames with no channels in them.
        with self.assertRaises(ValueError):
            plan_channel_stream(self.meta, start=0, n=8, want=frozenset())

    def test_a_plan_compares_but_does_not_hash(self):
        # Equality is what a surface needs ("did the plan change? then restart").
        # Hashing is disabled deliberately rather than left to raise from inside
        # the tiles tuple, where the error names neither the class nor the caller.
        a = plan_channel_stream(self.meta, start=0, n=8)
        self.assertEqual(a, plan_channel_stream(self.meta, start=0, n=8))
        self.assertNotEqual(a, plan_channel_stream(self.meta, start=1, n=8))
        with self.assertRaisesRegex(TypeError, "ChannelPlan"):
            hash(a)

    # (That the plan decodes nothing needs no test: it takes no video path.)

    # -- partial consumption -----------------------------------------------
    def test_closing_early_releases_the_decoder(self):
        # The live surface abandons a pass on every seek, so an abandoned
        # generator must not leak the capture. If it did, this test would fail
        # on the reopen below rather than somewhere unrelated much later.
        plan = plan_channel_stream(self.meta, start=0, n=N_FRAMES)
        gen = stream_channel_planes(self.video, plan)
        first = [next(gen) for _ in range(3)]
        gen.close()
        self.assertEqual([i for i, _ in first], [0, 1, 2])
        again, _ = self._drain(plan_channel_stream(self.meta, start=0, n=4))
        self.assertEqual(len(again), 4)


if __name__ == "__main__":
    unittest.main()
