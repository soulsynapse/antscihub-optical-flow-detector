"""The empirical sensitivity sweep: run the tuned detector at each scale.

These tests exercise the real path (a written video, a real structure-tensor
pass, the real detector) rather than mocking the extraction, because the claims
that matter here are about what the pass produces -- the block it resolves, the
grid it lands on, and whether a row is a legitimate cost sample -- and a mock
would assert those into existence instead of measuring them.
"""
from __future__ import annotations

import os
import tempfile
import unittest

import cv2
import numpy as np

from core.config import FlowConfig, PipelineConfig, PreprocessConfig
from core.evidence import (ScaleEvidence, measure_scale, reference_caveat,
                           sweep_scales)
from core.video import VideoSource


def _write_moving_square(path: str, n: int = 40, w: int = 128, h: int = 96) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
    if not writer.isOpened():
        raise unittest.SkipTest("no mp4v VideoWriter available in this environment")
    try:
        for i in range(n):
            frame = np.zeros((h, w, 3), np.uint8)
            x = 8 + 2 * i
            cv2.rectangle(frame, (x, 32), (x + 16, 60), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def _ev(scale, detected, grid=(4, 4), block=8, wall=1.0, frames=10):
    return ScaleEvidence(scale=scale, block=block, grid=grid, frames=frames,
                         wall=wall, window_start=0,
                         detected=np.asarray(detected, bool), intervals=[],
                         spans={"decode": 0.1})


class AgreementTests(unittest.TestCase):
    """kept/missed/added is set arithmetic over frame indices, deliberately NOT
    a score: a row that trades missed frames for added ones is detecting
    something else, and only the split makes that visible."""

    def test_identical_rows_keep_everything(self):
        ref = _ev(1.0, [0, 1, 1, 0, 1])
        self.assertEqual(_ev(0.5, [0, 1, 1, 0, 1]).agreement_with(ref),
                         (3, 0, 0))

    def test_lost_and_gained_frames_are_reported_separately(self):
        ref = _ev(1.0, [1, 1, 0, 0])
        self.assertEqual(_ev(0.5, [1, 0, 1, 0]).agreement_with(ref), (1, 1, 1))

    def test_a_row_that_finds_nothing_misses_everything(self):
        ref = _ev(1.0, [1, 1, 1])
        self.assertEqual(_ev(0.25, [0, 0, 0]).agreement_with(ref), (0, 3, 0))

    def test_ragged_lengths_compare_over_the_overlap(self):
        # A cancelled or short pass must not raise here; comparing the frames
        # both rows actually have is the honest reading.
        ref = _ev(1.0, [1, 1, 1, 1])
        self.assertEqual(_ev(0.5, [1, 0]).agreement_with(ref), (1, 1, 0))


class ReferenceCaveatTests(unittest.TestCase):
    """The worst thing this panel can do is award a passing grade for a vacuous
    test, and driving the real tool showed it does exactly that by default: a
    freshly built explorer has both bands wide open, the gate is on for every
    frame, and every scale then reports perfect agreement -- an unqualified
    argument for aggressive downsampling from a detector discriminating nothing.
    """

    def test_a_discriminating_reference_has_no_caveat(self):
        self.assertIsNone(reference_caveat(_ev(1.0, [0, 1, 1, 0])))

    def test_an_always_on_reference_is_refused(self):
        why = reference_caveat(_ev(1.0, [1, 1, 1, 1]))
        self.assertIsNotNone(why)
        self.assertIn("EVERY frame", why)

    def test_an_always_off_reference_is_refused(self):
        why = reference_caveat(_ev(1.0, [0, 0, 0]))
        self.assertIsNotNone(why)
        self.assertIn("nothing", why)

    def test_an_empty_reference_is_refused(self):
        self.assertIsNotNone(reference_caveat(_ev(1.0, [])))


class SweepScaleTests(unittest.TestCase):
    def test_reference_runs_first(self):
        self.assertEqual(sweep_scales(1.0, (1.0, 0.5, 0.25))[0], 1.0)

    def test_the_current_scale_is_always_included(self):
        # The settings being judged were tuned at it, so a sweep omitting it
        # would compare the user's real working scale against nothing.
        got = sweep_scales(0.42, (1.0, 0.75, 0.5, 0.25), limit=3)
        self.assertIn(0.42, got)
        self.assertEqual(got[0], 1.0)

    def test_rows_after_the_reference_descend(self):
        got = sweep_scales(1.0, (0.25, 1.0, 0.5, 0.75))
        self.assertEqual(got, sorted(got, reverse=True))

    def test_limit_is_respected(self):
        self.assertEqual(len(sweep_scales(1.0, (1.0, .75, .5, .35, .25, .15),
                                          limit=4)), 4)


class MeasureScaleTests(unittest.TestCase):
    """Drive a real pass. The point of each assertion is a property the dialog
    depends on, not the pass's numeric output."""

    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.mkdtemp(prefix="evid_")
        cls.video = os.path.join(cls._dir, "moving.mp4")
        _write_moving_square(cls.video)
        with VideoSource(cls.video) as src:
            info = src.info
        cls.dims = (info.width, info.height, float(info.fps),
                    int(info.frame_count))
        cls.reps = [{"id": 0, "label": "all", "frac": (0.0, 0.0, 1.0, 1.0)}]
        cls.params = {
            "channel_attr": "change", "region_index": 0,
            "freq_band_hz": (1.0, 5.0),
            "value_band": (0.0, float("inf")),
            "count_band": (0.0, float("inf")),
            "detect_window": 3, "centered": True,
        }

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove(cls.video)
            os.rmdir(cls._dir)
        except OSError:
            pass

    def _cfg(self, scale, block=None):
        return PipelineConfig(
            preprocess=PreprocessConfig(downsample=scale, normalize="off",
                                        denoise="off", registration="off",
                                        bg_subtract="off", mask_path=None),
            flow=FlowConfig(block_size=block))

    def _run(self, scale, block=None, n=24):
        return measure_scale(self.video, self._cfg(scale, block), self.reps,
                             dims=self.dims, start=0, n=n, region_index=0,
                             params=self.params)

    def test_a_pass_reports_the_scale_and_the_production_block(self):
        """The whole reason the sweep exists as a cost source: it resolves the
        block the way a batch run would instead of the live surface's block=1
        pixel cache, where block_reduce is 62% of the wall time."""
        ev = self._run(0.5)
        self.assertAlmostEqual(ev.scale, 0.5, places=6)
        self.assertEqual(ev.block, FlowConfig().resolve_block_size(0.5))
        self.assertGreater(ev.block, 1)

    def test_the_tracked_block_holds_the_grid_across_scales(self):
        """Comparability of the count band rests on this: the count is a number
        of blocks, so two rows are thresholded on the same quantity only while
        the grid is the same. `auto` is what makes that true."""
        a, b = self._run(1.0), self._run(0.5)
        self.assertEqual(a.grid, b.grid)

    def test_a_pinned_block_changes_the_grid_and_is_therefore_flagged(self):
        # The counterpart the panel must warn about rather than rank through.
        a, b = self._run(1.0, block=8), self._run(0.5, block=8)
        self.assertNotEqual(a.grid, b.grid)

    def test_a_row_is_a_usable_cost_sample(self):
        ev = self._run(1.0)
        s = ev.pass_sample()
        self.assertEqual(s.frames, ev.frames)
        self.assertGreater(s.wall, 0.0)
        self.assertGreater(s.seconds_per_frame, 0.0)
        self.assertAlmostEqual(s.scale, 1.0, places=6)

    def test_detection_covers_the_window_and_finds_the_moving_square(self):
        ev = self._run(1.0)
        self.assertEqual(ev.detected.shape[0], ev.frames)
        # Bands are wide open, so the gate should fire somewhere on a clip that
        # is nothing but motion; a silent all-zero row would mean the sweep is
        # measuring the detector's plumbing rather than the footage.
        self.assertGreater(ev.n_detected_frames, 0)
        self.assertGreater(ev.n_events, 0)

    def test_rows_from_different_scales_are_comparable(self):
        ref, low = self._run(1.0), self._run(0.35)
        kept, missed, added = low.agreement_with(ref)
        self.assertEqual(kept + missed, ref.n_detected_frames)
        self.assertEqual(kept + added, low.n_detected_frames)

    def test_the_detector_settings_are_used_unchanged(self):
        """A sweep run against different bands would answer a question the user
        did not ask, so this pins that params pass straight through."""
        params = dict(self.params, freq_band_hz=(2.0, 3.0))
        ev = measure_scale(self.video, self._cfg(1.0), self.reps,
                           dims=self.dims, start=0, n=24, region_index=0,
                           params=params)
        # An impossible count band cannot fire, which is only observable if the
        # band reached the detector.
        blocked = measure_scale(
            self.video, self._cfg(1.0), self.reps, dims=self.dims, start=0,
            n=24, region_index=0,
            params=dict(params, count_band=(1e9, 1e9 + 1)))
        self.assertEqual(blocked.n_detected_frames, 0)
        self.assertGreaterEqual(ev.n_detected_frames, 0)


if __name__ == "__main__":
    unittest.main()
