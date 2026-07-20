"""Partial edge blocks must count by valid area, not as whole blocks.

build_layout rounds each replicate's block grid up, so the final row/column of a
box can be a thin sliver -- as little as one working pixel tall. Its speed value
is already averaged over only the valid pixels, but every downstream *area* count
used to treat that sliver as a full block, which let a row of one-pixel edge
blocks masquerade as a real clump. These tests pin the geometry of the weight.
"""
from __future__ import annotations

import unittest

import numpy as np

from core.replicates import block_weight_plane


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


if __name__ == "__main__":
    unittest.main()
