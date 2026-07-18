"""The whole-video detector must agree with the window preview block-for-block.
These lock the shared math and the memory-bounded band power that make that true.
"""
from __future__ import annotations

import unittest

import numpy as np

from core.wavelet import band_indices, default_freqs, morlet_band_power, morlet_power
from core.detection import (detect_channel_region, detect_over_blocks,
                           inband_count, recompute_from_band_power,
                           region_blocks_and_grid, windowed_mean)


class MorletBandPowerTest(unittest.TestCase):
    def test_matches_full_cube_slice(self):
        rng = np.random.default_rng(0)
        fps = 30.0
        x = rng.standard_normal((300, 40)).astype(np.float32)
        freqs = default_freqs(fps)
        i, j = band_indices(freqs, 6.0, 12.0)
        full = morlet_power(x, fps, freqs)[i:j].sum(axis=0)
        # Small chunk exercises the block-chunking path too.
        band = morlet_band_power(x, fps, freqs, i, j, block_chunk=7)
        self.assertEqual(band.shape, (300, 40))
        np.testing.assert_allclose(band, full, rtol=1e-4, atol=1e-4)

    def test_1d_squeeze(self):
        fps = 30.0
        x = np.random.default_rng(1).standard_normal(200).astype(np.float32)
        freqs = default_freqs(fps)
        i, j = band_indices(freqs, 4.0, 9.0)
        full = morlet_power(x, fps, freqs)[i:j].sum(axis=0)
        band = morlet_band_power(x, fps, freqs, i, j)
        self.assertEqual(band.shape, (200,))
        np.testing.assert_allclose(band, full, rtol=1e-4, atol=1e-4)


class DetectionMathTest(unittest.TestCase):
    def test_inband_count(self):
        m = np.array([[1.0, 5.0, 9.0], [0.0, np.nan, 5.0]], np.float32)
        np.testing.assert_array_equal(inband_count(m, 4.0, 6.0), [1.0, 1.0])

    def test_windowed_mean_trailing_and_centered(self):
        count = np.array([0, 4, 0, 0], np.float32)
        np.testing.assert_allclose(windowed_mean(count, 1, True), count)
        # trailing W=2: t1 = (0+4)/2 = 2, t2 = (4+0)/2 = 2
        tr = windowed_mean(count, 2, centered=False)
        self.assertAlmostEqual(tr[1], 2.0)
        self.assertAlmostEqual(tr[2], 2.0)
        # centered W=3 at t=1 sees [0,1,2] -> mean(0,4,0) = 4/3
        ce = windowed_mean(count, 3, centered=True)
        self.assertAlmostEqual(ce[1], 4.0 / 3.0, places=5)


class DetectOverBlocksTest(unittest.TestCase):
    def test_pass_and_intervals(self):
        # A band-limited burst in the middle third should light up the gate.
        rng = np.random.default_rng(2)
        fps = 30.0
        T, B = 300, 12
        t = np.arange(T) / fps
        blocks = 0.05 * rng.standard_normal((T, B)).astype(np.float32)
        burst = (t > 4.0) & (t < 6.0)
        blocks[burst] += np.sin(2 * np.pi * 8.0 * t[burst])[:, None]
        freqs = default_freqs(fps)

        res = detect_over_blocks(
            blocks, fps, freqs, freq_band_hz=(6.0, 10.0),
            value_band=(0.05, float("inf")), count_band=(3.0, float("inf")),
            detect_window=int(fps), centered=True)

        self.assertEqual(res.band_power.shape, (T, B))
        self.assertGreater(float(res.gate.sum()), 0.0)
        # The detection should sit inside the burst, not at the quiet ends.
        on = np.flatnonzero(res.gate > 0.5)
        self.assertTrue((on / fps).min() > 3.0)
        self.assertTrue((on / fps).max() < 7.0)
        intervals = res.detected_intervals()
        self.assertTrue(len(intervals) >= 1)

    def test_recompute_matches_full_pass(self):
        rng = np.random.default_rng(3)
        fps = 30.0
        blocks = rng.standard_normal((200, 8)).astype(np.float32)
        freqs = default_freqs(fps)
        kw = dict(freq_band_hz=(5.0, 10.0), value_band=(0.5, 5.0),
                  count_band=(1.0, float("inf")), detect_window=15, centered=True)
        full = detect_over_blocks(blocks, fps, freqs, **kw)
        # Re-tuning off the retained band power must reproduce the same tracks.
        again = recompute_from_band_power(
            full.band_power, value_band=kw["value_band"],
            count_band=kw["count_band"], detect_window=kw["detect_window"],
            centered=kw["centered"])
        np.testing.assert_array_equal(full.count, again.count)
        np.testing.assert_array_equal(full.gate, again.gate)


class _StubChannelData:
    def __init__(self, arr, ny, nx, tiles=None):
        self.meta = {"fps": 30.0, "grid": [ny, nx], "n_frames": arr.shape[0],
                     "replicate_tiles": tiles or []}
        self.channels = {"tensor_speed": arr}
        self.window_start = 0


class DetectChannelRegionTest(unittest.TestCase):
    def test_whole_frame_matches_direct_blocks(self):
        fps = 30.0
        freqs = default_freqs(fps)
        T, ny, nx = 200, 3, 4
        arr = np.random.default_rng(4).standard_normal((T, ny, nx)).astype(np.float32)
        kw = dict(freq_band_hz=(5.0, 10.0), value_band=(0.5, 5.0),
                  count_band=(1.0, float("inf")), detect_window=15, centered=True)

        res = detect_channel_region(_StubChannelData(arr, ny, nx), 0,
                                    "tensor_speed", freqs=freqs, **kw)
        gy, gx = np.mgrid[0:ny, 0:nx]
        ref = detect_over_blocks(arr.reshape(T, ny * nx), fps, freqs,
                                 region_grid=(ny, nx, gy.ravel(), gx.ravel()), **kw)

        self.assertEqual(res.band_power.shape, (T, ny * nx))
        np.testing.assert_allclose(res.band_power, ref.band_power)
        np.testing.assert_array_equal(res.gate, ref.gate)
        np.testing.assert_array_equal(res.clump, ref.clump)

    def test_region_scoping_picks_the_right_tile(self):
        # Two stacked tiles in the atlas; scoping region 1 must read only its rows.
        ny, nx, T = 4, 2, 60
        arr = np.zeros((T, ny, nx), np.float32)
        arr[:, 2:4, :] = 3.0                         # only region 1 has signal
        tiles = [{"atlas_bbox": [0, 0, 2, 2]}, {"atlas_bbox": [2, 0, 4, 2]}]
        blocks, grid = region_blocks_and_grid(
            _StubChannelData(arr, ny, nx, tiles).meta, arr, 1)
        self.assertEqual(blocks.shape, (T, 4))
        self.assertTrue((blocks == 3.0).all())
        self.assertEqual(grid[0], 2)                 # dy
        self.assertEqual(grid[1], 2)                 # dx


if __name__ == "__main__":
    unittest.main()
