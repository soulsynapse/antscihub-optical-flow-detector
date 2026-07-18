"""Cacheless windowed channel extraction -- the seam that lets the scalogram
path run on a bare video with no flow cache."""
from __future__ import annotations

import os
import tempfile
import unittest

import cv2
import numpy as np

from core.config import FlowConfig, PipelineConfig, PreprocessConfig
from core.channel_source import (LIVE_CHANNELS, live_channel_source,
                                 reduce_channel_data, synth_live_meta)
from core.tensor_channels import extract_channels_live


def _write_moving_square(path: str, n: int = 40, w: int = 64, h: int = 48) -> None:
    """A white square translating across a black frame, so the structure tensor
    has real, reproducible motion to measure."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
    if not writer.isOpened():
        raise unittest.SkipTest("no mp4v VideoWriter available in this environment")
    try:
        for i in range(n):
            frame = np.zeros((h, w, 3), np.uint8)
            x = 4 + i                      # moves one px/frame to the right
            cv2.rectangle(frame, (x, 16), (x + 10, 30), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def _full_frame_replicate() -> list[dict]:
    return [{"id": 0, "label": "all", "frac": (0.0, 0.0, 1.0, 1.0)}]


def _two_replicates() -> list[dict]:
    # Side-by-side, non-overlapping: exercises atlas packing across tiles.
    return [{"id": 0, "label": "L", "frac": (0.0, 0.0, 0.5, 1.0)},
            {"id": 1, "label": "R", "frac": (0.5, 0.0, 1.0, 1.0)}]


class LiveChannelSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.mkdtemp(prefix="chsrc_")
        cls.video = os.path.join(cls._dir, "moving.mp4")
        _write_moving_square(cls.video)

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove(cls.video)
            os.rmdir(cls._dir)
        except OSError:
            pass

    def test_mid_clip_window_no_cache(self):
        cfg = PipelineConfig()
        cd = live_channel_source(self.video, cfg, _full_frame_replicate(),
                                 start=20, n=10)

        # Window length and offset are honoured, and the T axis is the window.
        self.assertEqual(cd.n_frames, 10)
        self.assertEqual(cd.window_start, 20)

        ny, nx = cd.meta["grid"]
        for name in LIVE_CHANNELS:
            self.assertIn(name, cd.channels)
            self.assertEqual(cd.channels[name].shape, (10, ny, nx))

        # The seed frame (frame 19) gives frame-0 of the window real motion, so
        # tensor_speed is populated across the whole window, not just from t=1.
        ts = cd.channels["tensor_speed"]
        self.assertGreater(float(ts.sum()), 0.0)
        self.assertGreater(float(ts[0].sum()), 0.0, "seed frame should give t=0 motion")
        self.assertTrue(np.isfinite(cd.channels["appearance"]).all())

        # A live source never carries the pipeline's cached flow speed.
        self.assertNotIn("speed", cd.available)

    def test_window_clamps_to_clip_end(self):
        cfg = PipelineConfig()
        cd = live_channel_source(self.video, cfg, _full_frame_replicate(),
                                 start=35, n=100)          # asks past the end
        self.assertEqual(cd.window_start, 35)
        self.assertEqual(cd.n_frames, 5)                   # 40 - 35
        self.assertEqual(cd.channels["change"].shape[0], 5)

    def test_block_rereduce_matches_direct_extraction(self):
        # A Block change re-reduces the cached pixel-level (block=1) channels; the
        # result must equal a fresh extraction at that block, block-for-block, for
        # both divisor and non-divisor blocks (partial edge cells) and >1 tile.
        for reps in (_full_frame_replicate(), _two_replicates()):
            pp = live_channel_source(self.video, PipelineConfig(
                flow=FlowConfig(block_size=1)), reps, start=5, n=12)
            for block in (2, 3, 5):
                cfg_n = PipelineConfig(flow=FlowConfig(block_size=block))
                reduced = reduce_channel_data(pp, cfg_n, reps)
                direct = live_channel_source(self.video, cfg_n, reps,
                                             start=5, n=12)
                self.assertEqual(list(reduced.meta["grid"]),
                                 list(direct.meta["grid"]),
                                 f"grid mismatch @ block {block}")
                for name in LIVE_CHANNELS:
                    np.testing.assert_allclose(
                        reduced.channels[name], direct.channels[name],
                        rtol=0, atol=1e-5,
                        err_msg=f"{name} @ block {block}, {len(reps)} tiles")

    def test_denoise_forced_off_flags_approximated(self):
        # Config asks for temporal denoise; the windowed path must force it off
        # (stateful, can't reproduce mid-clip) and flag the result approximated.
        cfg = PipelineConfig(
            preprocess=PreprocessConfig(denoise="gaussian"))
        meta = synth_live_meta(self.video, cfg, _full_frame_replicate(),
                               width=64, height=48, fps=20.0, frame_count=40)
        res = extract_channels_live(self.video, meta, start=10, n=8)
        self.assertTrue(res["meta"]["approximated"])
        self.assertEqual(res["meta"]["window_start"], 10)
        self.assertEqual(res["meta"]["n_frames"], 8)


if __name__ == "__main__":
    unittest.main()
