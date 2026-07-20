"""View-state round trip across a live re-extract rebuild (T17).

The live surface captures state off the old explorer and applies it to a freshly
built one. Anything missing from that dict silently reverts to a default on every
whole-video pass -- which is exactly how the two detection threshold bands were
being lost.
"""
from __future__ import annotations

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

import numpy as np
from PyQt6.QtWidgets import QApplication

from core.channel_source import ChannelData
from gui.explorers.scalogram_explorer import ScalogramExplorer


def _channel_data(T=64, ny=4, nx=4, block=64) -> ChannelData:
    rng = np.random.default_rng(0)
    meta = {
        "fps": 30.0,
        "grid": [ny, nx],
        "block_size": block,
        "n_frames": T,
        "replicate_tiles": [{"id": 0, "label": "r0", "atlas_bbox": [0, 0, ny, nx],
                             "frac": [0.0, 0.0, 1.0, 1.0]}],
    }
    channels = {name: rng.random((T, ny, nx), dtype=np.float32) + 0.1
                for name in ("change", "appearance", "tensor_speed", "intensity")}
    return ChannelData(meta=meta, channels=channels)


def _uneven_channel_data(scale: int, T=64) -> ChannelData:
    """Two replicates of DIFFERENT sizes, as build_layout really produces them
    (each tile is sized from its own frac box). ``scale`` multiplies both, the
    way halving the block size would."""
    big, small = 8 * scale, 2 * scale
    ny, nx = big + small, big
    meta = {
        "fps": 30.0,
        "grid": [ny, nx],
        "block_size": 64 // scale,
        "n_frames": T,
        "replicate_tiles": [
            {"id": 0, "label": "big", "atlas_bbox": [0, 0, big, big],
             "frac": [0.0, 0.0, 1.0, 0.8]},
            {"id": 1, "label": "small", "atlas_bbox": [big, 0, big + small, small],
             "frac": [0.0, 0.8, 0.25, 1.0]},
        ],
    }
    rng = np.random.default_rng(0)
    channels = {name: rng.random((T, ny, nx), dtype=np.float32) + 0.1
                for name in ("change", "appearance", "tensor_speed", "intensity")}
    return ChannelData(meta=meta, channels=channels)


class ViewStateRoundTripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _explorer(self, **kw):
        ex = ScalogramExplorer.from_channel_data(
            _channel_data(**kw), video_path=None, own_shortcuts=False,
            own_status=False)
        self.addCleanup(self._destroy, ex)
        return ex

    def _destroy(self, ex):
        ex.close()
        ex.deleteLater()
        self.app.processEvents()

    def test_detection_bands_survive_a_rebuild(self):
        old = self._explorer()
        old.channel = "change energy Jtt"
        old.density_plots["change energy Jtt"].band_lo = 0.25
        old.density_plots["change energy Jtt"].band_hi = 0.75
        old.count_w_plot.band_lo = 2.0
        old.count_w_plot.band_hi = 9.0

        new = self._explorer()
        new.apply_view_state(old.capture_view_state())

        self.assertEqual(new.detection_params()["value_band"], (0.25, 0.75))
        self.assertEqual(new.detection_params()["count_band"], (2.0, 9.0))

    def test_unselected_channels_keep_their_bands(self):
        old = self._explorer()
        old.density_plots["intensity"].band_lo = 1.5
        old.density_plots["intensity"].band_hi = 3.5

        new = self._explorer()
        new.apply_view_state(old.capture_view_state())

        dp = new.density_plots["intensity"]
        self.assertEqual((dp.band_lo, dp.band_hi), (1.5, 3.5))

    def test_an_unplaced_band_stays_unplaced(self):
        # Carrying None through as +/-inf would turn "never set" into an explicit
        # unbounded threshold and defeat set_band_active's lazy seeding.
        old = self._explorer()
        old.density_plots["appearance energy"].band_lo = None
        old.density_plots["appearance energy"].band_hi = None

        new = self._explorer()
        new.apply_view_state(old.capture_view_state())

        dp = new.density_plots["appearance energy"]
        self.assertIsNone(dp.band_lo)
        self.assertIsNone(dp.band_hi)

    def test_count_band_is_redenominated_by_a_block_change(self):
        """Batch N item 2. The count band is a RAW BLOCK COUNT, so a block
        change re-denominates it by the region-size ratio. Carried through
        untouched, a threshold tuned on a fine grid becomes unreachable on a
        coarse one and the detector silently stops firing."""
        old = self._explorer(ny=4, nx=4, block=64)          # 16 blocks/replicate
        old.count_w_plot.band_lo = 4.0
        old.count_w_plot.band_hi = 8.0

        new = self._explorer(ny=8, nx=8, block=32)          # 64 blocks/replicate
        note = new.apply_view_state(old.capture_view_state())

        self.assertEqual(new.detection_params()["count_band"], (16.0, 32.0))
        self.assertIsNotNone(note, "a self-changing threshold said nothing")
        self.assertIn("64", note)                            # names the new block
        self.assertIn("32", note)

    def test_conversion_denominates_against_the_tuned_replicate(self):
        """Replicate tiles are NOT uniform -- build_layout sizes each from its
        own frac box -- and a rebuild resets the selection. Measuring the new
        denominator against whatever the fresh explorer has selected would
        divide the tuned replicate's old count by a DIFFERENT replicate's new
        one and scale the threshold by the ratio of two unrelated boxes."""
        old = ScalogramExplorer.from_channel_data(
            _uneven_channel_data(scale=1), video_path=None,
            own_shortcuts=False, own_status=False)
        self.addCleanup(self._destroy, old)
        old.active_region_index = 1                  # the SMALL replicate: 2x2 = 4
        old.count_w_plot.band_lo, old.count_w_plot.band_hi = 4.0, float("inf")
        st = old.capture_view_state()
        self.assertEqual(st["count_denom"]["region_blocks"], 4)

        new = ScalogramExplorer.from_channel_data(
            _uneven_channel_data(scale=2), video_path=None,
            own_shortcuts=False, own_status=False)
        self.addCleanup(self._destroy, new)
        # A rebuild leaves nothing selected, which is exactly the trap: region 0
        # is 8x8 -> 16x16 and would give a factor of 4x the wrong way.
        self.assertLess(new.active_region_index, 0)
        new.apply_view_state(st)

        # The small replicate goes 2x2 = 4 -> 4x4 = 16, so the factor is 4.
        lo, _ = new.detection_params()["count_band"]
        self.assertAlmostEqual(lo, 16.0)

    def test_an_unchanged_block_leaves_the_count_band_alone(self):
        old = self._explorer(block=64)
        old.count_w_plot.band_lo, old.count_w_plot.band_hi = 2.0, 9.0
        new = self._explorer(block=64)
        note = new.apply_view_state(old.capture_view_state())
        self.assertEqual(new.detection_params()["count_band"], (2.0, 9.0))
        self.assertIsNone(note, "an untouched band announced a conversion")

    def test_an_unbounded_count_endpoint_does_not_rescale(self):
        """"Unbounded above" is not a count and has no denominator. Scaling inf
        is a no-op numerically, but the LOW end must still convert -- that is
        the endpoint that actually gates."""
        old = self._explorer(ny=4, nx=4, block=64)
        old.count_w_plot.band_lo = 3.0
        old.count_w_plot.band_hi = float("inf")

        new = self._explorer(ny=8, nx=8, block=32)
        new.apply_view_state(old.capture_view_state())

        lo, hi = new.detection_params()["count_band"]
        self.assertEqual(lo, 12.0)
        self.assertTrue(np.isinf(hi))

    def test_older_state_without_the_new_keys_still_applies(self):
        # A dict captured before this fix must not raise.
        new = self._explorer()
        new.apply_view_state({"channel": "change energy Jtt", "frame": 3, "sweep_win": 5,
                              "centered": True, "freq_band": (1.0, 4.0)})
        self.assertEqual(new.channel, "change energy Jtt")

    def test_a_state_without_a_denominator_carries_the_band_verbatim(self):
        """A dict captured before count_denom existed records no grid, so there
        is nothing to convert FROM. Guessing a factor and applying it to a tuned
        threshold is worse than leaving it: the user can see an unchanged number
        and re-tune, but cannot see a wrong conversion.

        No recorded block size also means no evidence the grid MOVED, so this
        stays silent -- unlike the case below, where it demonstrably did."""
        new = self._explorer(block=32)
        note = new.apply_view_state({"count_band": (5.0, 11.0)})
        self.assertEqual(new.detection_params()["count_band"], (5.0, 11.0))
        self.assertIsNone(note)

    def test_an_unconvertible_block_change_warns_instead_of_passing_silently(self):
        """The hazard this whole feature exists to prevent. The block moved, so
        the threshold is known to be wrong; if it also cannot be converted, the
        one unacceptable outcome is to carry it through without a word."""
        new = self._explorer(block=32)
        note = new.apply_view_state({
            "count_band": (5.0, 11.0),
            # Block moved, but the denominator is unusable.
            "count_denom": {"block_size": 64, "region_index": 0,
                            "region_blocks": 0},
        })
        self.assertEqual(new.detection_params()["count_band"], (5.0, 11.0))
        self.assertIsNotNone(note, "an invalidated threshold passed silently")
        self.assertIn("NOT", note)
        self.assertIn("re-tune", note)

    def test_a_missing_tuned_region_warns(self):
        """The tuned replicate is gone from the rebuilt source, so there is no
        comparable block count -- the same unconvertible case, reached the way
        it actually happens."""
        new = self._explorer(block=32)
        note = new.apply_view_state({
            "count_band": (5.0, 11.0),
            "count_denom": {"block_size": 64, "region_index": 7,
                            "region_blocks": 40},
        })
        self.assertEqual(new.detection_params()["count_band"], (5.0, 11.0))
        self.assertIsNotNone(note)

    def test_an_untuned_band_is_silent_across_a_block_change(self):
        """An unset/unbounded band carries no tuning, so a block change has
        invalidated nothing and there is nothing to warn about."""
        new = self._explorer(block=32)
        note = new.apply_view_state({
            "count_band": (None, float("inf")),
            "count_denom": {"block_size": 64, "region_index": 0,
                            "region_blocks": 0},
        })
        self.assertIsNone(note)


class ReplicateSwitchBandsTest(unittest.TestCase):
    """Switching replicates used to wipe every band to None, which made a
    threshold un-comparable across replicates by construction: you cannot ask
    whether one value separates behaviour in rep 25 and rep 26 if looking at
    rep 26 discards it."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _explorer(self):
        ex = ScalogramExplorer.from_channel_data(
            _uneven_channel_data(scale=1), video_path=None,
            own_shortcuts=False, own_status=False)
        self.addCleanup(self._destroy, ex)
        return ex

    def _destroy(self, ex):
        ex.close()
        ex.deleteLater()
        self.app.processEvents()

    def _select(self, ex, idx):
        for i in range(ex.region_combo.count()):
            if ex.region_combo.itemData(i) == idx:
                ex.region_combo.setCurrentIndex(i)
                return
        self.fail(f"no combo entry for region {idx}")

    def _await_matrix(self, ex, timeout=10.0):
        """Cubes build on a worker; spin the loop until the selected channel's
        matrix lands. A fixed pump count sits right on the edge of the build
        and makes these tests flaky."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            dp = ex._selected_density()
            if dp is not None and dp.matrix.size:
                return dp
        self.fail("cube never built within the timeout")

    def test_value_band_carries_to_an_unvisited_replicate(self):
        """A channel value band is an absolute per-block quantity -- 0.3 is 0.3
        in every replicate -- so it must survive the switch verbatim."""
        ex = self._explorer()
        ch = ex.channel
        self._select(ex, 0)
        ex.density_plots[ch].band_lo = 0.25
        ex.density_plots[ch].band_hi = 0.75

        self._select(ex, 1)
        self.assertEqual(ex.density_plots[ch].band(), (0.25, 0.75))

    def test_count_band_is_redenominated_by_the_block_count_ratio(self):
        """The count band is a RAW BLOCK COUNT and replicate tiles are not
        uniform, so carrying it verbatim would gate on a different fraction of
        the replicate than the user tuned."""
        ex = self._explorer()
        self._select(ex, 0)
        self.assertEqual(ex._region_block_count(0), 64)   # 8x8
        self.assertEqual(ex._region_block_count(1), 4)    # 2x2
        ex.count_w_plot.band_lo, ex.count_w_plot.band_hi = 8.0, 40.0

        self._select(ex, 1)
        # 64 -> 4 blocks is a factor of 1/16.
        self.assertEqual(ex.count_w_plot.band(), (0.5, 2.5))

    def test_revisiting_a_replicate_restores_its_own_bands(self):
        """Cache beats carry-forward: returning to a scope shows what you left
        there, not what you were last touching somewhere else."""
        ex = self._explorer()
        ch = ex.channel
        self._select(ex, 0)
        ex.density_plots[ch].band_lo, ex.density_plots[ch].band_hi = 0.25, 0.75
        ex.count_w_plot.band_lo, ex.count_w_plot.band_hi = 8.0, 40.0

        self._select(ex, 1)
        ex.density_plots[ch].band_lo, ex.density_plots[ch].band_hi = 0.4, 0.9

        self._select(ex, 0)
        self.assertEqual(ex.density_plots[ch].band(), (0.25, 0.75))
        self.assertEqual(ex.count_w_plot.band(), (8.0, 40.0))

        self._select(ex, 1)
        self.assertEqual(ex.density_plots[ch].band(), (0.4, 0.9))

    def test_unselected_channels_keep_their_bands_across_a_switch(self):
        ex = self._explorer()
        self._select(ex, 0)
        other = next(n for n in ex.density_plots if n != ex.channel)
        ex.density_plots[other].band_lo = 1.5
        ex.density_plots[other].band_hi = 3.5

        self._select(ex, 1)
        self.assertEqual(ex.density_plots[other].band(), (1.5, 3.5))

    def test_an_off_range_carried_band_warns(self):
        """A carried band can be correct and still land nowhere near this
        replicate's data, which reads as a broken detector rather than a
        tuning miss. Cubes arrive asynchronously, so the check has to run when
        the matrix lands, not when the region changes."""
        ex = self._explorer()
        ch = ex.channel
        self._select(ex, 0)
        self._await_matrix(ex)
        ex.density_plots[ch].band_lo, ex.density_plots[ch].band_hi = 50.0, 60.0

        self._select(ex, 1)
        dp = self._await_matrix(ex)
        self.assertFalse(ex.band_note.isHidden())
        self.assertIn("outside this one's range", ex.band_note.text())

        # Re-dragging into range is how the warning clears.
        lo = float(np.nanmin(dp.matrix))
        hi = float(np.nanmax(dp.matrix))
        dp.band_lo, dp.band_hi = lo, hi
        ex._on_density_band_committed(ch)
        self.assertTrue(ex.band_note.isHidden())

    def test_an_in_range_carried_band_is_silent(self):
        ex = self._explorer()
        ch = ex.channel
        self._select(ex, 0)
        dp = self._await_matrix(ex)
        dp.band_lo = float(np.nanmin(dp.matrix))
        dp.band_hi = float(np.nanmax(dp.matrix))

        self._select(ex, 1)
        self._await_matrix(ex)
        self.assertTrue(ex.band_note.isHidden())


if __name__ == "__main__":
    unittest.main()
