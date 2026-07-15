from __future__ import annotations

import tempfile
import unittest

import numpy as np

from core import cache as cache_mod
from core.config import FeatureConfig, PipelineConfig
from core.features import FeatureContext, cached_feature_names
from core.flow import (forward_backward_error, reduce_scalar_to_blocks)
from core.roi import ROI, rect_roi, roi_block_values


class _CacheStub:
    def __init__(self, downsample: float = 1.0):
        self.meta = {"downsample": downsample}


class StandardizationMathTests(unittest.TestCase):
    def test_forward_backward_error_is_zero_for_inverse_translation(self):
        forward = np.zeros((12, 12, 2), np.float32)
        backward = np.zeros_like(forward)
        forward[..., 0] = 1.0
        backward[..., 0] = -1.0
        error = forward_backward_error(forward, backward)
        np.testing.assert_allclose(error[:, :-1], 0.0, atol=1e-6)
        self.assertTrue(np.all(error[:, -1] == 12.0))

    def test_forward_backward_error_detects_inconsistent_flow(self):
        forward = np.zeros((8, 8, 2), np.float32)
        backward = np.zeros_like(forward)
        forward[..., 1] = 0.5
        error = forward_backward_error(forward, backward)
        self.assertAlmostEqual(float(np.median(error[:-1])), 0.5, places=5)

    def test_scalar_block_reductions(self):
        x = np.arange(64, dtype=np.float32).reshape(8, 8)
        mean = reduce_scalar_to_blocks(x, 4, "mean")
        p90 = reduce_scalar_to_blocks(x, 4, "p90")
        self.assertEqual(mean.shape, (2, 2))
        self.assertTrue(np.all(p90 >= mean))
        expected = np.percentile(
            x.reshape(2, 4, 2, 4).transpose(0, 2, 1, 3).reshape(2, 2, 16),
            90, axis=2)
        np.testing.assert_allclose(p90, expected)

    def test_median_features_leave_raw_flow_untouched(self):
        u = np.zeros((1, 5, 5), np.float32)
        u[0, 2, 2] = 100.0
        ctx = FeatureContext(u, np.zeros_like(u), np.abs(u), fps=10, block_size=1)
        filtered = ctx.get("median3_u")
        self.assertEqual(float(filtered[0, 2, 2]), 0.0)
        self.assertEqual(float(ctx.u[0, 2, 2]), 100.0)

    def test_texture_percentile_is_per_frame_and_continuous(self):
        base = np.zeros((2, 2, 2), np.float32)
        texture = np.array([[[0, 1], [2, 3]], [[30, 20], [10, 0]]], np.float32)
        ctx = FeatureContext(base, base, base, fps=10, block_size=1,
                             bands={"texture_min_eigen": texture})
        pct = ctx.get("texture_percentile")
        np.testing.assert_allclose(pct[0], [[0.25, 0.5], [0.75, 1.0]])
        np.testing.assert_allclose(pct[1], [[1.0, 0.75], [0.5, 0.25]])


class ReplicateStandardizationTests(unittest.TestCase):
    def setUp(self):
        speed = np.array([
            [[1, 2], [3, 4]],
            [[2, 4], [6, 8]],
            [[10, 20], [30, 40]],
            [[20, 40], [60, 80]],
        ], np.float32)
        self.ctx = FeatureContext(np.zeros_like(speed), np.zeros_like(speed), speed,
                                  fps=1.0, block_size=1)

    def test_baseline_p99_is_replicate_specific(self):
        roi = rect_roi(1, (0, 0, 1, 1), (2, 2), 4,
                       baseline_start_s=0.0, baseline_end_s=2.0)
        ratio = roi_block_values(_CacheStub(), self.ctx, roi,
                                 "speed_over_baseline_p99")
        expected_floor = np.percentile(self.ctx.speed[:2], 99)
        np.testing.assert_allclose(ratio, self.ctx.speed.reshape(4, 4) / expected_floor)

    def test_auto_noise_fallback_is_one_constant_per_replicate(self):
        roi = rect_roi(1, (0, 0, 1, 1), (2, 2), 4)
        ratio = roi_block_values(_CacheStub(), self.ctx, roi,
                                 "speed_over_auto_noise")
        floors = np.percentile(self.ctx.speed[1:].reshape(3, 4), 25, axis=1)
        expected = self.ctx.speed.reshape(4, 4) / np.percentile(floors, 99)
        np.testing.assert_allclose(ratio, expected)

    def test_physical_conversion_accounts_for_cache_downsampling(self):
        roi = rect_roi(1, (0, 0, 1, 1), (2, 2), 4,
                       pixels_per_mm=20.0, body_length_mm=2.0)
        mm_s = roi_block_values(_CacheStub(downsample=0.5), self.ctx, roi,
                                "speed_mm_s")
        bl_s = roi_block_values(_CacheStub(downsample=0.5), self.ctx, roi,
                                "speed_body_lengths_s")
        np.testing.assert_allclose(mm_s, self.ctx.speed.reshape(4, 4) / 10.0)
        np.testing.assert_allclose(bl_s, mm_s / 2.0)


class CacheContractTests(unittest.TestCase):
    def test_new_cache_is_unopenable_until_marked_complete(self):
        meta = {
            "fps": 10.0,
            "n_frames": 2,
            "block_size": 1,
            "grid": [1, 1],
            "features": ["u", "v", "speed"],
        }
        with tempfile.TemporaryDirectory() as root:
            cache = cache_mod.create_cache(root, "x", meta, backend="zarr")
            for name in meta["features"]:
                cache.create_array(name, (2, 1, 1), "float32")
                cache.write(name, 0, np.zeros((2, 1, 1), np.float32))
            cache.close()
            with self.assertRaises(cache_mod.IncompleteCacheError):
                cache_mod.open_cache(root, "x")
            cache.meta["complete"] = True
            cache.write_meta()
            opened = cache_mod.open_cache(root, "x")
            opened.close()

    def test_standardization_cache_features_are_explicit(self):
        cfg = PipelineConfig(features=FeatureConfig(
            cache_fb_error=True, cache_texture=True))
        names = cached_feature_names(cfg)
        self.assertIn("fb_error_p90", names)
        self.assertIn("texture_min_eigen", names)


if __name__ == "__main__":
    unittest.main()
