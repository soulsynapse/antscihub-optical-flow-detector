"""Per-video tuning memory and the Reset button.

Two behaviours that pull in opposite directions and therefore have to be tested
together: reopening a clip must find the knobs and the three detection bands as
they were left, and Reset must put every one of them back -- including the
per-scope band cache, which exists precisely so that bands survive things.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

import numpy as np
from PyQt6.QtWidgets import QApplication

from gui.explorers.live_scalogram_surface import LiveScalogramSurface
from gui.explorers.scalogram_explorer import ScalogramExplorer
from gui.tuning_store import load_tuning, save_tuning, tuning_path
from tests.test_channel_source import _write_moving_square
from tests.test_view_state import _channel_data


class _QtTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])


class TuningStoreTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp(prefix="tuning_store_")
        self.video = os.path.join(self._dir, "clip.mp4")
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for p in (tuning_path(self.video), self.video):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(self._dir)
        except OSError:
            pass

    def test_band_endpoints_round_trip_including_the_unbounded_ones(self):
        """A band endpoint is legitimately None ("never placed") or +/-inf
        ("unbounded on this side"), and the two mean different things -- the
        first is re-seeded lazily, the second is a threshold the user opened."""
        payload = {"view": {"count_band": [None, float("inf")],
                            "value_bands": {"change energy Jtt":
                                            [float("-inf"), 0.5]}}}
        save_tuning(self.video, payload)
        got = load_tuning(self.video)["view"]

        self.assertEqual(got["count_band"], [None, float("inf")])
        self.assertEqual(got["value_bands"]["change energy Jtt"],
                         [float("-inf"), 0.5])

    def test_numpy_scalars_do_not_defeat_the_write(self):
        save_tuning(self.video, {"view": {"frame": np.int64(7),
                                          "freq_band": [np.float32(1.5), None]}})
        got = load_tuning(self.video)["view"]

        self.assertEqual(got["frame"], 7)
        self.assertAlmostEqual(got["freq_band"][0], 1.5, places=5)

    def test_a_corrupt_sidecar_reads_as_no_tuning(self):
        with open(tuning_path(self.video), "w", encoding="utf-8") as f:
            f.write("{not json at all")

        self.assertEqual(load_tuning(self.video), {})

    def test_a_foreign_version_is_discarded_not_half_applied(self):
        with open(tuning_path(self.video), "w", encoding="utf-8") as f:
            f.write('{"version": 999, "strip": {"downsample": 0.1}}')

        self.assertEqual(load_tuning(self.video), {})

    def test_a_failed_write_leaves_the_previous_tuning_intact(self):
        """Encoding happens before the file is opened, so a value that cannot
        be serialised must not truncate a good sidecar into a corrupt one."""
        save_tuning(self.video, {"strip": {"downsample": 0.5}})
        save_tuning(self.video, {"strip": {"downsample": object()}})

        self.assertEqual(load_tuning(self.video)["strip"], {"downsample": 0.5})


class ResetTuningTests(_QtTestCase):
    """reset_tuning() on the explorer: the detection half of the Reset button."""

    def _explorer(self) -> ScalogramExplorer:
        ex = ScalogramExplorer.from_channel_data(
            _channel_data(), video_path=None, own_shortcuts=False,
            own_status=False)
        self.addCleanup(self._destroy, ex)
        return ex

    def _destroy(self, ex):
        ex.close()
        ex.deleteLater()
        self.app.processEvents()

    def _tuned(self) -> ScalogramExplorer:
        ex = self._explorer()
        ex.scalo_plot.band_lo, ex.scalo_plot.band_hi = 3.0, 12.0
        ex.count_w_plot.band_lo, ex.count_w_plot.band_hi = 2.0, 9.0
        ex.density_plots[ex.channel].band_lo = 0.25
        ex.density_plots[ex.channel].band_hi = 0.75
        ex.sweep_win = 41
        ex.sweep_win_slider.setValue(41)
        ex.centered_chk.setChecked(False)
        return ex

    def test_every_band_reopens_wide(self):
        ex = self._tuned()
        ex.reset_tuning()

        for plot in (ex.scalo_plot, ex.count_w_plot, ex.density_plots[ex.channel]):
            with self.subTest(plot=plot.title):
                self.assertEqual((plot.band_lo, plot.band_hi),
                                 (float("-inf"), float("inf")))

    def test_the_detection_window_returns_to_one_second(self):
        ex = self._tuned()
        ex.reset_tuning()

        self.assertEqual(ex.sweep_win, int(round(ex.fps)))
        self.assertEqual(ex.sweep_win_slider.value(), ex.sweep_win)
        self.assertTrue(ex.centered)
        self.assertTrue(ex.centered_chk.isChecked())

    def test_the_per_scope_band_cache_is_cleared_too(self):
        """The cache restores a band when you come back to a replicate. Left
        populated, a reset would undo itself the moment you switched scope."""
        ex = self._tuned()
        ex._value_band_cache[(0, ex.channel)] = (0.25, 0.75)
        ex._count_band_cache[0] = (2.0, 9.0)

        ex.reset_tuning()

        self.assertFalse(ex._value_band_cache)
        self.assertFalse(ex._count_band_cache)

    def test_what_you_are_looking_at_is_not_reset(self):
        """Channel and replicate say what is on screen, not what counts as a
        detection; resetting thresholds must not also move the view."""
        ex = self._tuned()
        ex.channel = "tensor speed"
        region = ex.active_region_index

        ex.reset_tuning()

        self.assertEqual(ex.channel, "tensor speed")
        self.assertEqual(ex.active_region_index, region)

    def test_a_reset_is_reported_as_a_tuning_change(self):
        ex = self._tuned()
        seen = []
        ex.tuning_changed.connect(lambda: seen.append(1))

        ex.reset_tuning()

        self.assertEqual(len(seen), 1)


class _SurfaceTestCase(_QtTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._dir = tempfile.mkdtemp(prefix="tuning_surface_")
        cls.video = os.path.join(cls._dir, "moving.mp4")
        _write_moving_square(cls.video)

    @classmethod
    def tearDownClass(cls):
        for p in (tuning_path(cls.video), cls.video):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(cls._dir)
        except OSError:
            pass

    def _drop_tuning(self):
        try:
            os.remove(tuning_path(self.video))
        except OSError:
            pass

    def _surface(self) -> LiveScalogramSurface:
        self.addCleanup(self._drop_tuning)
        reps = [{"id": 0, "label": "all", "frac": (0.0, 0.0, 1.0, 1.0)}]
        # singleShot patched out: constructing the surface must not kick off the
        # opening extract pass.
        with patch("gui.explorers.live_scalogram_surface.QTimer.singleShot"):
            surface = LiveScalogramSurface(self.video, reps)
        self.addCleanup(self._destroy, surface)
        surface._show_channel_data = MagicMock()
        surface.extract = MagicMock()
        return surface

    def _destroy(self, surface):
        surface.close()
        surface.deleteLater()
        self.app.processEvents()


class SurfaceMemoryTests(_SurfaceTestCase):
    def test_the_strip_reopens_on_what_was_left_there(self):
        self._drop_tuning()
        first = self._surface()
        first.start_slider.setValue(4)
        # Within the fixture clip's length; the spin box clamps above it, and a
        # clamped value would make this assert its own clamping.
        first.len_spin.setValue(1.25)
        first.ds_spin.setValue(0.5)
        first.block_spin.setValue(16)
        first.norm_combo.setCurrentText("clahe")
        first._save_tuning()

        second = self._surface()

        self.assertEqual(second.start_slider.value(), 4)
        self.assertAlmostEqual(second.len_spin.value(), 1.25)
        self.assertAlmostEqual(second.ds_spin.value(), 0.5)
        self.assertEqual(second.block_spin.value(), 16)
        self.assertEqual(second.norm_combo.currentText(), "clahe")

    def test_the_restore_does_not_arm_a_pass(self):
        """Applying five remembered knobs through their own signals would queue
        five re-extracts to arrive at one state."""
        self._drop_tuning()
        first = self._surface()
        first.ds_spin.setValue(0.5)
        first._save_tuning()

        second = self._surface()

        self.assertFalse(second._debounce.isActive())
        self.assertFalse(second._block_debounce.isActive())

    def test_the_bands_and_replicate_are_handed_to_the_first_explorer(self):
        self._drop_tuning()
        first = self._surface()
        first._explorer = MagicMock()
        first._explorer.capture_view_state.return_value = {
            "channel": "tensor speed", "sweep_win": 7,
            "count_band": [2.0, float("inf")]}
        first._explorer.active_region_index = 1
        first._save_tuning()

        second = self._surface()

        self.assertEqual(second._pending_state["channel"], "tensor speed")
        self.assertEqual(second._pending_state["sweep_win"], 7)
        self.assertEqual(second._pending_state["count_band"],
                         [2.0, float("inf")])
        self.assertEqual(second._pending_region, 1)

    def test_a_clip_never_tuned_opens_on_defaults(self):
        self._drop_tuning()
        surface = self._surface()

        self.assertIsNone(surface._pending_state)
        self.assertIsNone(surface._pending_region)
        self.assertEqual(surface._strip_values(), surface._strip_defaults)

    def test_a_committed_band_schedules_a_write(self):
        """The surface persists off the explorer's tuning_changed; without the
        relay, bands would only ever be saved when a strip knob moved."""
        surface = self._surface()
        surface._save_debounce.stop()
        explorer = ScalogramExplorer.from_channel_data(
            _channel_data(), video_path=None, own_shortcuts=False,
            own_status=False)
        self.addCleanup(explorer.deleteLater)
        explorer.tuning_changed.connect(surface._save_debounce.start)

        explorer.tuning_changed.emit()

        self.assertTrue(surface._save_debounce.isActive())


class SurfaceResetTests(_SurfaceTestCase):
    def test_reset_restores_the_strip_defaults(self):
        surface = self._surface()
        defaults = dict(surface._strip_defaults)
        surface.start_slider.setValue(4)
        surface.ds_spin.setValue(0.25)
        surface.block_spin.setValue(16)
        surface.norm_combo.setCurrentText("clahe")

        surface.reset_all()

        self.assertEqual(surface._strip_values(), defaults)

    def test_reset_runs_exactly_one_pass(self):
        surface = self._surface()
        surface.ds_spin.setValue(0.25)
        surface.norm_combo.setCurrentText("clahe")
        surface.extract.reset_mock()

        surface.reset_all()

        self.assertEqual(surface.extract.call_count, 1)

    def test_reset_clears_the_explorer_before_the_pass_captures_it(self):
        """extract() carries the explorer's view state across the rebuild, so a
        reset that landed after it would be undone by its own re-extract."""
        surface = self._surface()
        order = []
        surface._explorer = MagicMock()
        surface._explorer.reset_tuning.side_effect = lambda: order.append("reset")
        surface.extract.side_effect = lambda: order.append("extract")

        surface.reset_all()

        self.assertEqual(order, ["reset", "extract"])

    def test_reset_without_an_explorer_still_resets_the_strip(self):
        """The button is live from the moment the surface opens, before the
        first extract has produced anything to reset."""
        surface = self._surface()
        surface._explorer = None
        surface.ds_spin.setValue(0.25)

        surface.reset_all()

        self.assertEqual(surface._strip_values(), surface._strip_defaults)


if __name__ == "__main__":
    unittest.main()
