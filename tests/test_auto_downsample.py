"""Automatic downsampling follows owned replicate geometry, not video margins.

SHELVED SPEC -- the implementation this exercises does not exist yet.

An earlier attempt landed ``replicate_reference_width`` /
``resolve_replicate_downsample`` and had to be backed out: resolving the scale
from the widest replicate instead of the source width raises the factor roughly
4x linear (0.16 -> 0.65 on an 8000px clip owning a 2000px replicate), i.e. ~16x
the pixels through the per-pixel tensor solve, which made the live surface
unusable. Extraction already crops per tile, so the crop does not offset this --
the tile is exactly what gets scaled up.

The invariant below is still the one we want, and it is a *correctness* property,
not a quality preference: a pre-cropped video and an equivalent uncropped box
must resolve to the same physical scale, or detection results are not comparable
across those two workflows. Today they are not.

Unshelve when either (a) the extraction pass is fast enough to absorb the cost
(see todo.md Batch C/6), or (b) the per-replicate target width is measured down
from DEFAULT_TARGET_WIDTH to whatever detection actually needs -- halving it cuts
the cost to 4x rather than 16x and may be the cheaper fix. Measure before
optimizing.
"""
from __future__ import annotations

import unittest

try:
    from core.channel_source import synth_live_meta
    from core.config import PipelineConfig, PreprocessConfig
    from core.replicates import (build_layout, replicate_reference_width,
                                 resolve_replicate_downsample)
    _MISSING = None
except ImportError as e:                       # the shelved implementation
    _MISSING = str(e)


@unittest.skipIf(_MISSING is not None,
                 f"replicate-aware auto downsample is shelved: {_MISSING}")
class ReplicateAutoDownsampleTests(unittest.TestCase):
    def test_precrop_and_equivalent_uncropped_box_resolve_identically(self):
        cfg = PipelineConfig()
        uncropped = [
            {"id": 0, "label": "animal", "frac": (0.25, 0.1, 0.5, 0.9)},
        ]
        precropped = [
            {"id": 0, "label": "animal", "frac": (0.0, 0.0, 1.0, 1.0)},
        ]

        # The owned region is 2000 source pixels wide in both representations.
        boxed_scale = resolve_replicate_downsample(
            cfg.preprocess, uncropped, src_width=8000)
        cropped_scale = resolve_replicate_downsample(
            cfg.preprocess, precropped, src_width=2000)

        self.assertAlmostEqual(boxed_scale, 0.65)
        self.assertEqual(boxed_scale, cropped_scale)

        boxed_meta = synth_live_meta(
            "uncropped.mp4", cfg, uncropped, width=8000, height=3000,
            fps=30.0, frame_count=100)
        cropped_meta = synth_live_meta(
            "precropped.mp4", cfg, precropped, width=2000, height=2400,
            fps=30.0, frame_count=100)
        self.assertEqual(boxed_meta["downsample"], cropped_meta["downsample"])
        self.assertEqual(boxed_meta["replicate_tiles"][0]["work_width"], 1300)
        self.assertEqual(cropped_meta["replicate_tiles"][0]["work_width"], 1300)

    def test_widest_replicate_sets_one_common_physical_scale(self):
        reps = [
            {"id": 0, "label": "wide", "frac": (0.0, 0.0, 0.5, 0.5)},
            {"id": 1, "label": "narrow", "frac": (0.5, 0.5, 0.75, 1.0)},
        ]
        cfg = PreprocessConfig()

        self.assertEqual(replicate_reference_width(reps, 4000), 2000)
        scale = resolve_replicate_downsample(cfg, reps, 4000)
        self.assertAlmostEqual(scale, 0.65)

        layout = build_layout(reps, 4000, 2000, scale, block_size=16)
        self.assertEqual(layout.tiles[0].work_width, 1300)
        self.assertEqual(layout.tiles[1].work_width, 650)

    def test_explicit_factor_still_overrides_replicate_geometry(self):
        reps = [
            {"id": 0, "label": "small", "frac": (0.1, 0.1, 0.2, 0.2)},
        ]
        cfg = PreprocessConfig(downsample=0.25)
        self.assertEqual(resolve_replicate_downsample(cfg, reps, 8000), 0.25)

    def test_auto_does_not_upscale_a_small_replicate(self):
        reps = [
            {"id": 0, "label": "small", "frac": (0.0, 0.0, 0.1, 1.0)},
        ]
        cfg = PreprocessConfig()
        self.assertEqual(resolve_replicate_downsample(cfg, reps, 8000), 1.0)


if __name__ == "__main__":
    unittest.main()
