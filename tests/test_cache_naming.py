from __future__ import annotations

import unittest

from core.config import (Band, FeatureConfig, FlowConfig, PipelineConfig,
                         PreprocessConfig)
from gui.tab1_flow import _test_cache_suffix


class TestCacheNamingTests(unittest.TestCase):
    def test_suffix_describes_settings_and_preserves_fractional_duration(self):
        cfg = PipelineConfig(
            preprocess=PreprocessConfig(
                mask_path="mask.png",
                registration="phase",
                denoise="median",
                bg_subtract="mog2",
                downsample=0.25,
                normalize="clahe",
            ),
            flow=FlowConfig(backend="dis", block_size=8),
            features=FeatureConfig(
                bands=(Band(8.0, 20.0),),
                window_s=2.0,
                hop_s=0.5,
                cache_fb_error=True,
                cache_texture=True,
                dtype="float32",
                compression="lz4",
            ),
        )

        self.assertEqual(
            _test_cache_suffix(cfg, 10.5),
            "_test10p5s_dis_b8_ds0p25_regphase_denmedian_bgmog2_"
            "normclahe_band8to20_win2_hop0p5_f32_lz4_mask_fberr_texture",
        )
        self.assertNotEqual(
            _test_cache_suffix(cfg, 10.5),
            _test_cache_suffix(cfg, 10.25),
        )

    def test_suffix_makes_default_off_states_explicit(self):
        suffix = _test_cache_suffix(PipelineConfig(), 10.0)
        self.assertIn("_test10s_farneback_b16_dsauto", suffix)
        self.assertIn("_regoff_denoff_bgoff_normoff_", suffix)
        self.assertIn("_band12to25_win1_hop0p25_f16_zstd", suffix)


if __name__ == "__main__":
    unittest.main()
