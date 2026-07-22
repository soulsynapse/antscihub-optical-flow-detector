"""GUI adapter for the existing per-replicate pre-transcode machinery."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtWidgets import QApplication

from gui.replicate_split import ReplicateSplitWorker
from gui.state import AppState
from gui.tab2_replicates import Tab2Replicates
from gui.tab_live_preprocess import TabLivePreprocess

_APP = QApplication.instance() or QApplication([])

REPS = [{"id": 1, "label": "rep1", "frac": (0.1, 0.1, 0.4, 0.4),
         "color": "#ff0000", "processed": False}]


class SplitWorkerTest(unittest.TestCase):
    def test_worker_snapshots_geometry_and_relays_progress(self):
        reps = [dict(REPS[0])]
        seen = {}
        progress = []
        complete = []

        def build(video, passed_reps, out_dir, **kwargs):
            seen.update(video=video, reps=passed_reps, out_dir=out_dir,
                        kwargs=kwargs)
            kwargs["progress"](0.375)
            return "manifest"

        worker = ReplicateSplitWorker(
            "source.mp4", reps, "clips", quality="high", overwrite=False)
        reps[0]["frac"] = (0, 0, 1, 1)
        worker.progress.connect(progress.append)
        worker.complete.connect(complete.append)
        with patch("gui.replicate_split.build_pretranscode", side_effect=build):
            worker.run()

        self.assertEqual(seen["reps"][0]["frac"], (0.1, 0.1, 0.4, 0.4))
        self.assertEqual(seen["kwargs"]["clip_layout"], "home")
        self.assertEqual(seen["kwargs"]["quality"], "high")
        self.assertEqual(progress, [375])
        self.assertEqual(complete, ["manifest"])

    def test_requested_cancel_has_its_own_terminal_signal(self):
        cancelled = []

        def build(*_args, **kwargs):
            worker.cancel()
            if kwargs["should_cancel"]():
                from core.pretranscode import PretranscodeCancelled
                raise PretranscodeCancelled

        worker = ReplicateSplitWorker(
            "source.mp4", REPS, "clips", quality="high", overwrite=False)
        worker.cancelled.connect(lambda: cancelled.append(True))
        with patch("gui.replicate_split.build_pretranscode", side_effect=build):
            worker.run()
        self.assertEqual(cancelled, [True])


class SplitTabTest(unittest.TestCase):
    def test_controls_require_both_a_video_and_boxes(self):
        tab = Tab2Replicates(AppState())
        self.assertFalse(tab.split_btn.isEnabled())
        self.assertIn("Open a video", tab.split_status.text())

    def test_start_freezes_geometry_and_restores_it_after_the_thread(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        video = os.path.join(d.name, "source.mp4")
        open(video, "wb").close()
        tab = Tab2Replicates(AppState())
        tab.replicates = [dict(REPS[0])]
        transitions = []
        tab.split_running_changed.connect(transitions.append)

        class Manifest:
            clips = ()
            frame_count = 12

        def build(*_args, **kwargs):
            kwargs["progress"](0.5)
            return Manifest()

        with patch.object(Tab2Replicates, "_video_path", return_value=video), \
                patch("gui.replicate_split.build_pretranscode", side_effect=build):
            tab._refresh_split_status()
            tab._split_or_cancel()
            worker = tab._split_worker
            self.assertIsNotNone(worker)
            self.assertFalse(tab.video.isEnabled())
            self.assertFalse(tab.box_group.isEnabled())
            worker.wait(5000)
            _APP.processEvents()
            _APP.processEvents()

        self.assertIsNone(tab._split_worker)
        self.assertTrue(tab.video.isEnabled())
        self.assertTrue(tab.box_group.isEnabled())
        self.assertEqual(transitions, [True, False])

    def test_reentering_preprocessing_discovers_a_new_manifest(self):
        """Creating clips changes no geometry, so the surface is not rebuilt."""
        state = AppState()
        state.replicate_specs = [dict(REPS[0])]
        tab = TabLivePreprocess(state)
        surface = MagicMock()
        tab._surface = surface
        tab._sig = ("source.mp4", "geometry")
        with patch.object(tab, "_current_sig", return_value=tab._sig):
            tab._ensure_surface()
        surface.refresh_replicate_metadata.assert_called_once_with(
            state.replicate_specs)
        surface._sync_clip_availability.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
