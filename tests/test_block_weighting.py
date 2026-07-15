"""Partial edge blocks must count by valid area, not as whole blocks.

build_layout rounds each replicate's block grid up, so the final row/column of a
box can be a thin sliver -- as little as one working pixel tall. Its speed value
is already averaged over only the valid pixels, but every downstream *area* count
(clump size, passing-block count, strength) used to treat that sliver as a full
block, which let a row of one-pixel edge blocks masquerade as a real clump. These
tests pin the geometry of the weight and the behaviour of the detection gate.
"""
from __future__ import annotations

import unittest

import numpy as np

from core.behavior import (Behavior, LogicNode, RangeLeaf, SpatialCriteria,
                           TemporalCriteria)
from core.replicates import block_weight_plane
from core.roi import ROI, roi_detection


class BlockWeightPlaneTests(unittest.TestCase):
    def test_partial_row_and_column_weighted_by_valid_area(self):
        # One tile, block 16. Working size 20 x 40:
        #   width  20 -> 2 cols, last col valid 4px  -> weight 4/16 = 0.25
        #   height 40 -> 3 rows, last row valid 8px  -> weight 8/16 = 0.5
        meta = {
            "grid": [3, 2],
            "block_size": 16,
            "replicate_tiles": [
                {"work_width": 20, "work_height": 40, "atlas_bbox": [0, 0, 3, 2]},
            ],
        }
        w = block_weight_plane(meta)
        np.testing.assert_allclose(w, [[1.0, 0.25],
                                       [1.0, 0.25],
                                       [0.5, 0.125]])

    def test_separator_cells_between_tiles_weigh_zero(self):
        meta = {
            "grid": [3, 2],
            "block_size": 16,
            "replicate_tiles": [
                {"work_width": 32, "work_height": 16, "atlas_bbox": [0, 0, 1, 2]},
                {"work_width": 32, "work_height": 16, "atlas_bbox": [2, 0, 3, 2]},
            ],
        }
        w = block_weight_plane(meta)
        np.testing.assert_array_equal(w[1], [0.0, 0.0])   # the separator row

    def test_legacy_full_frame_cache_weighs_every_block_one(self):
        w = block_weight_plane({"grid": [3, 4]})   # no replicate_tiles
        np.testing.assert_array_equal(w, np.ones((3, 4), np.float32))

    def test_missing_work_dims_default_to_full_blocks(self):
        # Older/partial meta without work_width/height must not crash and must
        # reproduce the pre-weighting behaviour (every block full).
        meta = {"grid": [1, 2], "block_size": 4,
                "replicate_tiles": [{"atlas_bbox": [0, 0, 1, 2]}]}
        np.testing.assert_array_equal(block_weight_plane(meta),
                                      np.ones((1, 2), np.float32))


class _Ctx:
    """Minimal FeatureContext: roi_detection needs n_frames, fps and get()."""
    def __init__(self, speed: np.ndarray, fps: float = 10.0):
        self._speed = speed.astype(np.float32)
        self.n_frames = speed.shape[0]
        self.fps = fps
        self.speed = self._speed

    def get(self, name: str) -> np.ndarray:
        if name != "speed":
            raise KeyError(name)
        return self._speed


class _Cache:
    def __init__(self, meta: dict):
        self.meta = meta


class RoiDetectionWeightingTests(unittest.TestCase):
    """A tile 5 blocks wide whose last row is a 1px sliver (weight 1/16).

    Frame 0: only that thin bottom row is above threshold across all 5 columns.
    Frame 1: a full top row is above threshold across all 5 columns.
    With min_blocks=3 the thin row is 5 * 1/16 = 0.3125 blocks of area and must
    NOT fire; the full row is 5 blocks and must fire.
    """
    def _setup(self):
        block = 16
        work_w, work_h = 80, 3 * block + 1      # 5 cols; 4 rows, last 1px tall
        meta = {
            "grid": [4, 5],
            "block_size": block,
            "downsample": 1.0,
            "replicate_tiles": [{
                "work_width": work_w, "work_height": work_h,
                "grid": [4, 5], "atlas_bbox": [0, 0, 4, 5],
            }],
        }
        speed = np.zeros((2, 4, 5), np.float32)
        speed[0, 3, :] = 1.0     # phantom sliver clump along the bottom edge
        speed[1, 0, :] = 1.0     # genuine full-block clump along the top
        ctx = _Ctx(speed)
        cache = _Cache(meta)
        roi = ROI(roi_id=1, frames=list(range(2)),
                  mask=np.ones((4, 5), dtype=bool), bbox=(0, 0, 4, 5))
        behavior = Behavior(
            name="fast",
            spec=LogicNode(op="and",
                           children=[RangeLeaf("speed", lo=0.5, hi=1e9)]),
            spatial=SpatialCriteria(min_blocks=3, min_fraction=0.0,
                                    merge_distance=0),
            criteria=TemporalCriteria(min_duration_s=0.0, max_gap_s=0.0,
                                      smooth_s=0.0),
        )
        return cache, ctx, behavior, roi

    def test_thin_edge_row_does_not_trip_the_clump_gate(self):
        cache, ctx, behavior, roi = self._setup()
        detected, _ = roi_detection(cache, ctx, behavior, roi)
        # The 5-sliver bottom row is 0.31 blocks of valid area, below min_blocks=3.
        self.assertFalse(bool(detected[0]))
        # The full top row is a genuine 5-block clump and still fires.
        self.assertTrue(bool(detected[1]))

    def test_unweighted_count_would_have_fired_on_the_sliver(self):
        # Guards the premise: without weighting the sliver row is 5 raw blocks,
        # which clears min_blocks=3 -- the exact false positive being fixed.
        _, _, _, roi = self._setup()
        raw_blocks_in_row = int(roi.mask[3].sum())
        self.assertGreaterEqual(raw_blocks_in_row, 3)


if __name__ == "__main__":
    unittest.main()
