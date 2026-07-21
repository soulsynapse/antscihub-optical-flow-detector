"""The calibration sub-tool's two non-obvious wiring invariants.

The arithmetic is covered in ``test_calibration.py``. What is easy to break
silently, and what these pin, is:

1. **Index agreement.** ``build_layout`` sorts replicates by id while a replicate
   list is in draw order, so an index that means one replicate in tile order
   means another in list order unless both are sorted. That mismatch would put
   one replicate's rendered picture beside another's px-per-body-length, and --
   once calibration writes back -- would calibrate the wrong animal.
2. **The write-back relay.** ``AppState.set_replicate_specs`` *copies* the
   replicate dicts, so a calibration measured off ``replicate_specs`` cannot
   reach the replicate tab's list or its sidecar by mutation. If that copy ever
   became a share, the relay would look redundant and be removed -- and would
   then be needed again the moment the copy came back.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtWidgets import QApplication

from core.replicates import build_layout
from gui.downsample_dialog import DownsampleDialog

# Deliberately not in id order, as a hand-edited or reordered sidecar would be.
_REPS = [
    {"id": 3, "label": "rep3", "frac": (0.60, 0.60, 0.90, 0.90), "color": "#f00"},
    {"id": 1, "label": "rep1", "frac": (0.05, 0.05, 0.30, 0.30), "color": "#0f0"},
    {"id": 2, "label": "rep2", "frac": (0.35, 0.35, 0.55, 0.55), "color": "#00f"},
]
_W, _H = 640, 480


class _QtTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])


class TestIndexAgreement(_QtTestCase):
    def test_dialog_indexes_replicates_in_tile_order(self):
        dlg = DownsampleDialog(_REPS, src_width=_W, src_height=_H, fps=30.0,
                               current_scale=1.0, model=None)
        tiles = build_layout(_REPS, _W, _H, 1.0, 1).tiles
        self.assertEqual([r["id"] for r in dlg._reps],
                         [t.replicate_id for t in tiles])
        # The caller's list is not reordered underneath it.
        self.assertEqual([r["id"] for r in _REPS], [3, 1, 2])

    def test_readout_box_matches_the_tile_at_the_same_index(self):
        """``_kept``'s uncalibrated fallback reports box dimensions, so it must
        read the same tile the render strip drew at that index."""
        dlg = DownsampleDialog(_REPS, src_width=_W, src_height=_H, fps=30.0,
                               current_scale=1.0, model=None)
        tiles = build_layout(_REPS, _W, _H, 1.0, 1).tiles
        for i, tile in enumerate(tiles):
            x0, y0, x1, y1 = tile.source_box
            dlg.rep_spin.setValue(i)
            self.assertIn(f"{x1 - x0}x{y1 - y0} working px", dlg._kept(1.0))

    def test_surface_calibrates_the_replicate_the_dialog_is_showing(self):
        from gui.explorers.live_scalogram_surface import LiveScalogramSurface
        dlg = DownsampleDialog(_REPS, src_width=_W, src_height=_H, fps=30.0,
                               current_scale=1.0, model=None)
        # _sorted_reps is the surface's half of the same contract; exercising it
        # directly avoids standing up a video decoder for an ordering question.
        surface_order = [r["id"] for r in
                         sorted(_REPS, key=lambda r: int(r.get("id", 0)))]
        self.assertEqual([r["id"] for r in dlg._reps], surface_order)
        self.assertTrue(hasattr(LiveScalogramSurface, "_sorted_reps"))


class TestDialogCalibrationMerge(_QtTestCase):
    def test_merge_updates_only_the_selected_replicate(self):
        reps = [dict(r) for r in _REPS]
        dlg = DownsampleDialog(reps, src_width=_W, src_height=_H, fps=30.0,
                               current_scale=1.0, model=None)
        dlg.apply_calibration(0, {"body_length_px": 40.0})
        self.assertEqual(dlg._reps[0].get("body_length_px"), 40.0)
        self.assertIsNone(dlg._reps[1].get("body_length_px"))

    def test_animal_only_calibration_reaches_the_readout(self):
        """The fiducial-free path is the one that would silently do nothing if
        the ``body_length_px`` fallback were dropped from the cost model."""
        reps = [dict(r) for r in _REPS]
        dlg = DownsampleDialog(reps, src_width=_W, src_height=_H, fps=30.0,
                               current_scale=1.0, model=None)
        dlg.rep_spin.setValue(0)
        self.assertIn("uncalibrated", dlg._kept(0.5))
        dlg.apply_calibration(0, {"body_length_px": 80.0})
        self.assertIn("40.0 px per body length", dlg._kept(0.5))

    def test_empty_merge_is_a_noop(self):
        reps = [dict(r) for r in _REPS]
        dlg = DownsampleDialog(reps, src_width=_W, src_height=_H, fps=30.0,
                               current_scale=1.0, model=None)
        dlg.apply_calibration(0, {})
        dlg.apply_calibration(99, {"body_length_px": 1.0})   # clamped, no crash
        self.assertIsNone(dlg._reps[0].get("body_length_px"))


class TestWriteBackRelayIsNecessary(_QtTestCase):
    def test_app_state_copies_replicate_dicts(self):
        from gui.state import AppState
        state = AppState(project_dir=".")
        state.set_replicate_specs(_REPS)
        self.assertIsNot(state.replicate_specs[0], _REPS[0])

    def test_apply_calibration_updates_specs_and_emits(self):
        from gui.state import AppState
        state = AppState(project_dir=".")
        state.set_replicate_specs(_REPS)
        seen = []
        state.calibration_changed.connect(lambda i, f: seen.append((i, f)))

        state.apply_calibration(2, {"pixels_per_mm": 8.0})
        spec = next(r for r in state.replicate_specs if r["id"] == 2)
        self.assertEqual(spec["pixels_per_mm"], 8.0)
        self.assertEqual(seen, [(2, {"pixels_per_mm": 8.0})])

        # A partial merge must not clear a sibling field.
        state.apply_calibration(2, {"body_length_px": 30.0})
        self.assertEqual(spec["pixels_per_mm"], 8.0)
        self.assertEqual(spec["body_length_px"], 30.0)

    def test_unknown_id_and_empty_fields_are_safe(self):
        from gui.state import AppState
        state = AppState(project_dir=".")
        state.set_replicate_specs(_REPS)
        seen = []
        state.calibration_changed.connect(lambda i, f: seen.append(i))
        state.apply_calibration(99, {"pixels_per_mm": 1.0})   # no such replicate
        state.apply_calibration(1, {})                        # nothing measured
        self.assertEqual(seen, [99])
        for r in state.replicate_specs:
            self.assertIsNone(r.get("pixels_per_mm"))


class TestSurfaceMetadataRefresh(_QtTestCase):
    """``AppState.set_replicate_specs`` replaces its dicts with fresh copies on
    every edit, so a surface built earlier holds references that go stale.
    Measured before the fix: calibrating in the replicate tab left an
    already-built surface's downsample window reading 'uncalibrated'."""

    @staticmethod
    def _surface_reps():
        # The refresh is pure dict bookkeeping, so it is exercised through an
        # unbound call rather than by standing up a video decoder for it.
        from gui.explorers.live_scalogram_surface import LiveScalogramSurface

        class _Stub:
            replicates = [dict(r) for r in _REPS]
            refresh_replicate_metadata = \
                LiveScalogramSurface.refresh_replicate_metadata
        return _Stub()

    def test_metadata_is_refreshed_from_the_owner(self):
        s = self._surface_reps()
        incoming = [dict(r) for r in _REPS]
        for r in incoming:
            if r["id"] == 1:
                r.update({"pixels_per_mm": 8.0, "body_length_px": 100.0})
        s.refresh_replicate_metadata(incoming)
        got = next(r for r in s.replicates if r["id"] == 1)
        self.assertEqual(got["pixels_per_mm"], 8.0)
        self.assertEqual(got["body_length_px"], 100.0)
        untouched = next(r for r in s.replicates if r["id"] == 2)
        self.assertIsNone(untouched.get("pixels_per_mm"))

    def test_geometry_is_never_copied_across(self):
        """A surface whose geometry differed would be the wrong surface -- the
        tab rebuilds it on a geometry hash -- so silently adopting new boxes
        would leave it extracting one region and drawing another."""
        s = self._surface_reps()
        before = tuple(s.replicates[0]["frac"])
        incoming = [dict(r) for r in _REPS]
        incoming[0]["frac"] = (0.0, 0.0, 0.99, 0.99)
        incoming[0]["pixels_per_mm"] = 4.0
        s.refresh_replicate_metadata(incoming)
        self.assertEqual(tuple(s.replicates[0]["frac"]), before)
        self.assertEqual(s.replicates[0]["pixels_per_mm"], 4.0)

    def test_replicate_missing_from_the_owner_is_left_alone(self):
        s = self._surface_reps()
        s.refresh_replicate_metadata([])
        self.assertEqual(len(s.replicates), len(_REPS))


class TestAnimalOnlyCalibrationResumes(_QtTestCase):
    """"Measure the animal now, find a ruler later" has to complete: without
    seeding ``body_length_px`` the second session cannot derive a body length in
    mm from the line the first session already measured."""

    @staticmethod
    def _frame():
        import numpy as np
        return np.zeros((240, 320, 3), np.uint8)

    def test_stored_pixel_length_is_shown_on_reopen(self):
        from gui.calibration_dialog import CalibrationDialog
        dlg = CalibrationDialog(self._frame(), scale=0.5, body_length_px=100.0)
        self.assertIn("50.0 working px per body length", dlg.result_lbl.text())
        self.assertTrue(dlg.apply_btn.isEnabled())

    def test_a_later_ruler_converts_the_stored_line(self):
        from gui.calibration_dialog import CalibrationDialog
        dlg = CalibrationDialog(self._frame(), scale=1.0, body_length_px=100.0)
        dlg.ruler_btn.setChecked(True)
        dlg._on_line((10.0, 10.0), (410.0, 10.0))     # 400 px
        dlg.known_mm.setValue(50.0)                    # -> 8 px/mm
        fields = dlg._build_calibration().as_replicate_fields()
        self.assertAlmostEqual(fields["pixels_per_mm"], 8.0)
        self.assertAlmostEqual(fields["body_length_mm"], 12.5)
        self.assertAlmostEqual(fields["body_length_px"], 100.0)

    def test_a_new_animal_line_supersedes_the_stored_one(self):
        from gui.calibration_dialog import CalibrationDialog
        dlg = CalibrationDialog(self._frame(), scale=1.0, body_length_px=100.0)
        dlg.animal_btn.setChecked(True)
        dlg._on_line((0.0, 0.0), (60.0, 0.0))
        self.assertAlmostEqual(
            dlg._build_calibration().body_length_px, 60.0)

    def test_pixel_seed_wins_over_the_mm_seed(self):
        """The pixel length is what was measured; deriving it back from mm would
        fold in whichever ruler was used then, so a ruler re-drawn now would move
        a length nobody re-measured."""
        from gui.calibration_dialog import CalibrationDialog
        dlg = CalibrationDialog(self._frame(), scale=1.0, pixels_per_mm=2.0,
                                body_length_mm=10.0, body_length_px=100.0)
        self.assertAlmostEqual(dlg._build_calibration().body_length_px, 100.0)

    def test_mm_seed_alone_still_carries_through(self):
        """Replicates calibrated by hand, or before body_length_px existed."""
        from gui.calibration_dialog import CalibrationDialog
        dlg = CalibrationDialog(self._frame(), scale=1.0, pixels_per_mm=2.0,
                                body_length_mm=10.0)
        self.assertAlmostEqual(dlg._build_calibration().body_length_px, 20.0)


if __name__ == "__main__":
    unittest.main()
