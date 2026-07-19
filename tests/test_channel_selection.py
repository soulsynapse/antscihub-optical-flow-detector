"""Per-channel extraction (T18): the whole-video commit detects on ONE channel,
so it should compute one.

The load-bearing property is the first test's: restricting the pass must not
change the numbers. A speed win that quietly perturbs the channel it kept would
be worse than the waste it replaces.
"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from core.channel_source import LIVE_CHANNELS, live_channel_source
from core.config import PipelineConfig
from core.tensor_channels import CHANNELS, extract_channels_live
from tests.test_channel_source import (_full_frame_replicate,
                                       _write_moving_square)


class ChannelSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.mkdtemp(prefix="chsel_")
        cls.video = os.path.join(cls._dir, "moving.mp4")
        _write_moving_square(cls.video)
        cls.cfg = PipelineConfig()
        cls.reps = _full_frame_replicate()

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove(cls.video)
            os.rmdir(cls._dir)
        except OSError:
            pass

    def _live(self, **kw):
        return live_channel_source(self.video, self.cfg, self.reps, **kw)

    def test_a_selected_channel_is_bit_identical_to_the_full_pass(self):
        full = self._live()
        for name in LIVE_CHANNELS:
            with self.subTest(channel=name):
                one = self._live(channels=[name])
                np.testing.assert_array_equal(one.channels[name],
                                              full.channels[name])

    def test_only_the_requested_channel_is_carried(self):
        cd = self._live(channels=["change"])
        self.assertEqual(set(cd.channels), {"change"})
        self.assertEqual(cd.available, {"change"})
        self.assertEqual(cd.meta["channels_computed"], ["change"])

    def test_a_pair_selects_both(self):
        cd = self._live(channels=["change", "tensor_speed"])
        self.assertEqual(set(cd.channels), {"change", "tensor_speed"})

    def test_unrequested_channels_are_zero_length_not_zero_filled(self):
        # Zero is a real measurable value on every one of these channels, so a
        # zero-FILLED placeholder reads as "measured, and nothing happened" --
        # the silent false negative this codebase keeps designing against. A
        # zero-LENGTH array cannot be mistaken for data, and costs no memory:
        # the full-length alternative is ~88 MB per channel per hour of footage.
        res = extract_channels_live(
            self.video, self._live().meta, start=0, n=8, channels=["change"])
        self.assertEqual(res["change"].shape[0], 8)
        for name in set(CHANNELS) - {"change"}:
            with self.subTest(channel=name):
                self.assertIn(name, res)               # key shape stays fixed
                self.assertEqual(res[name].shape[0], 0)
        self.assertEqual(res["meta"]["channels_computed"], ["change"])

    def test_default_still_computes_every_live_channel(self):
        cd = self._live()
        self.assertEqual(set(cd.channels), set(LIVE_CHANNELS))

    def test_an_empty_selection_is_refused(self):
        # Silently returning a source with no channels would surface much later
        # as an empty detection rather than as a bad request.
        with self.assertRaises(ValueError):
            self._live(channels=[])
        with self.assertRaises(ValueError):
            self._live(channels=["speed"])      # cache-only, not a live channel

    def test_change_skips_the_expensive_downstream_work(self):
        # The point of the exercise. Not a wall-clock assertion (too flaky for
        # CI); instead read the timing spans, which name the stages directly.
        meta = self._live().meta
        spans = lambda ch: set(extract_channels_live(
            self.video, meta, start=0, n=12,
            channels=[ch])["meta"]["timing"]["spans"])

        change = spans("change")
        self.assertNotIn("flow_solve", change)
        self.assertNotIn("appearance", change)
        self.assertNotIn("texture", change)
        self.assertIn("tensor_products", change)

        # tensor_speed does need the solve, so the gate is per-channel and not
        # just "skip everything but the first stage".
        self.assertIn("flow_solve", spans("tensor_speed"))

    def test_a_sweep_records_which_channels_it_priced(self):
        # A cost-model fit mixing a four-channel sample with a one-channel one
        # reads the difference as scale. The sample has to say which it was.
        from core.scale_sweep import measure_scale

        dims = (64, 48, 20.0, 40)
        full = measure_scale(self.video, self.cfg, self.reps, dims=dims,
                             start=0, n=12)
        one = measure_scale(self.video, self.cfg, self.reps, dims=dims,
                            start=0, n=12, channels=["change"])
        self.assertEqual(set(full.channels), set(LIVE_CHANNELS))
        self.assertEqual(one.channels, ("change",))


if __name__ == "__main__":
    unittest.main()
