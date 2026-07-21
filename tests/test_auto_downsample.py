"""Scale is not derived from framing, so equivalent workflows agree.

This file used to be a shelved spec for ``resolve_replicate_downsample``: an
automatic scale resolved from the widest replicate rather than the source width.
That machinery is gone, and the correctness gap it existed to close is closed a
different and cheaper way -- see todo.md Batch K.

The gap was that ``downsample=None`` meant ``1300 / src_width``, so the physical
working scale was a function of how the camera happened to be framed. A
pre-cropped video and an equivalent uncropped replicate box resolved to
DIFFERENT scales, and their detection results were therefore not comparable.
Resolving from replicate geometry would have fixed that by making the scale
larger; defaulting to no downsampling at all fixes it by not rescaling either
one. At 1.0 the invariant is trivial, which is the point.

The tests below pin the invariant itself rather than any particular resolver, so
they survive the eventual organism-relative mode (pixels per body length) as long
as that mode keeps equivalent framings equivalent.
"""
from __future__ import annotations

import unittest

from core.channel_source import synth_live_meta
from core.config import (BASE_BLOCK_SOURCE_PX, FlowConfig, PipelineConfig,
                         PreprocessConfig)


class DefaultScaleTests(unittest.TestCase):
    def test_default_does_not_downsample(self):
        """The governing principle: the pipeline must not silently decide what
        is detectable. See core/config.py PreprocessConfig.downsample."""
        self.assertEqual(PreprocessConfig().resolve_downsample(), 1.0)

    def test_scale_is_independent_of_source_width(self):
        cfg = PreprocessConfig()
        for width in (0, 640, 1300, 5312, 8000):
            self.assertEqual(cfg.resolve_downsample(width), 1.0)

    def test_explicit_factor_is_honoured(self):
        self.assertEqual(
            PreprocessConfig(downsample=0.25).resolve_downsample(8000), 0.25)

    def test_precrop_and_equivalent_uncropped_box_resolve_identically(self):
        """The original correctness gap, now closed by construction.

        The same 2000-source-pixel-wide owned region, expressed once as a box in
        an 8000px frame and once as a pre-cropped 2000px video, must produce the
        same working geometry.
        """
        cfg = PipelineConfig()
        uncropped = [{"id": 0, "label": "animal", "frac": (0.25, 0.1, 0.5, 0.9)}]
        precropped = [{"id": 0, "label": "animal", "frac": (0.0, 0.0, 1.0, 1.0)}]

        boxed = synth_live_meta("uncropped.mp4", cfg, uncropped,
                                width=8000, height=3000, fps=30.0,
                                frame_count=100)
        cropped = synth_live_meta("precropped.mp4", cfg, precropped,
                                  width=2000, height=2400, fps=30.0,
                                  frame_count=100)

        self.assertEqual(boxed["downsample"], cropped["downsample"])
        self.assertEqual(boxed["block_size"], cropped["block_size"])
        self.assertEqual(boxed["replicate_tiles"][0]["work_width"],
                         cropped["replicate_tiles"][0]["work_width"])
        # 2000 source px, unscaled.
        self.assertEqual(boxed["replicate_tiles"][0]["work_width"], 2000)


class BlockTracksScaleTests(unittest.TestCase):
    """block_size tracks the scale so the two levers stay separable: moving
    Downsample must change compute WITHOUT moving the block grid, or a change in
    detection output cannot be attributed to either knob."""

    def test_auto_block_is_base_at_full_scale(self):
        self.assertEqual(FlowConfig().resolve_block_size(1.0),
                         BASE_BLOCK_SOURCE_PX)

    def test_auto_block_scales_with_the_working_scale(self):
        f = FlowConfig()
        self.assertEqual(f.resolve_block_size(0.5), 32)
        self.assertEqual(f.resolve_block_size(0.25), 16)
        # The old default pairing: 1300/5312 with a 16px block.
        self.assertEqual(f.resolve_block_size(0.2447), 16)

    def test_auto_block_never_degenerates_to_zero(self):
        self.assertEqual(FlowConfig().resolve_block_size(0.001), 1)

    def test_explicit_block_overrides_tracking(self):
        f = FlowConfig(block_size=16)
        self.assertEqual(f.resolve_block_size(1.0), 16)
        self.assertEqual(f.resolve_block_size(0.25), 16)

    def test_grid_is_scale_invariant_under_tracking(self):
        """The property the tracking exists for, measured end to end."""
        reps = [{"id": 0, "label": "a", "frac": (0.0, 0.0, 0.5, 0.5)}]
        grids = set()
        for scale in (1.0, 0.5, 0.25):
            cfg = PipelineConfig(
                preprocess=PreprocessConfig(downsample=scale))
            meta = synth_live_meta("v.mp4", cfg, reps, width=4096, height=2048,
                                   fps=30.0, frame_count=100)
            grids.add(tuple(meta["grid"]))
        self.assertEqual(len(grids), 1, f"grid moved with scale: {grids}")


if __name__ == "__main__":
    unittest.main()
