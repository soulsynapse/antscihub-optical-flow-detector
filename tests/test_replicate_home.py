"""Per-replicate homes: the directory that owns one replicate's work.

The defects worth a test here are all quiet ones. A home named after a label
looks fine until someone renames a box; a home named ``rep01`` looks fine until
a second source lands in the folder; a recycled id looks fine until the new box
comes up already showing the dead animal's detections. None of those raise, and
none of them are visible in a screenshot -- so they are asserted rather than
inspected.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from core.replicate_home import (describe_home, ensure_home, home_name,
                                 home_path, replicate_dir, sync_homes)


class HomeNamingTest(unittest.TestCase):
    def test_the_home_is_named_by_id_and_prefixed_by_the_source_stem(self):
        self.assertEqual(home_name("/f/GX010047.MP4", 1), "GX010047_rep01")
        self.assertEqual(home_name("/f/GX010047.MP4", 12), "GX010047_rep12")

    def test_two_sources_in_one_folder_do_not_share_a_home(self):
        """The reason the stem is in the directory name at all.

        A bare ``rep01/`` beside both A.MP4 and B.MP4 would have A's first
        replicate and B's first replicate -- different animals, different
        geometry -- writing into the same place.
        """
        self.assertNotEqual(replicate_dir("/f/A.MP4", 1),
                            replicate_dir("/f/B.MP4", 1))

    def test_the_home_sits_beside_the_source(self):
        video = os.path.join("some", "footage", "GX010047.MP4")
        self.assertEqual(os.path.dirname(replicate_dir(video, 3)),
                         os.path.dirname(video))

    def test_artefacts_keep_the_stem_inside_the_home(self):
        p = home_path("/f/GX010047.MP4", 2, ".track.npz")
        self.assertEqual(os.path.basename(p), "GX010047.track.npz")
        self.assertEqual(os.path.basename(os.path.dirname(p)),
                         "GX010047_rep02")

    def test_the_label_is_not_an_input(self):
        """A rename cannot move a home, because no naming function can see one.

        Asserted structurally rather than by renaming something: the guarantee
        the design rests on is that the label is absent from the signature, the
        same call ``core.replicates._canonical_geometry`` made when it left
        labels out of the geometry hash.
        """
        import inspect
        for fn in (home_name, replicate_dir, ensure_home):
            self.assertNotIn("label", inspect.signature(fn).parameters,
                             f"{fn.__name__} must not depend on the label")


class HomeCreationTest(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.addCleanup(self._d.cleanup)
        self.video = os.path.join(self._d.name, "clip.mp4")

    def test_ensure_home_creates_it_and_is_idempotent(self):
        a = ensure_home(self.video, 1)
        b = ensure_home(self.video, 1)
        self.assertEqual(a, b)
        self.assertTrue(os.path.isdir(a))

    def test_ensure_home_returns_none_rather_than_raising(self):
        """Drawing a box must not become an error dialog. A home that cannot be
        created degrades to the artefacts not being written later, which the
        stores already report in their own terms."""
        self.assertIsNone(ensure_home("", 1))
        # A directory whose parent is a FILE cannot be made, on any platform.
        with open(self.video, "wb") as f:
            f.write(b"\0")
        self.assertIsNone(ensure_home(os.path.join(self.video, "inner.mp4"), 1))

    def test_sync_homes_covers_every_box_and_skips_malformed_ones(self):
        made = sync_homes(self.video, [{"id": 1}, {"id": 2}, {"nope": True}])
        self.assertEqual(len(made), 2)
        self.assertTrue(all(os.path.isdir(p) for p in made))

    def test_sync_homes_does_not_remove_a_home_whose_box_is_gone(self):
        """Additive only. A deleted box leaves its directory, which is what
        makes a monotonic next_id a safety property instead of a detail."""
        sync_homes(self.video, [{"id": 1}, {"id": 2}])
        sync_homes(self.video, [{"id": 1}])
        self.assertTrue(os.path.isdir(replicate_dir(self.video, 2)))


class DescribeHomeTest(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.addCleanup(self._d.cleanup)
        self.video = os.path.join(self._d.name, "clip.mp4")

    def _put(self, rid, suffix, blob=b"x"):
        ensure_home(self.video, rid)
        p = home_path(self.video, rid, suffix)
        mode = "wb" if isinstance(blob, bytes) else "w"
        with open(p, mode) as f:
            f.write(blob)
        return p

    def test_an_empty_or_missing_home_describes_nothing(self):
        self.assertEqual(describe_home(self.video, 9), [])
        ensure_home(self.video, 1)
        self.assertEqual(describe_home(self.video, 1), [])

    def test_no_video_never_walks_the_working_directory(self):
        self.assertEqual(describe_home("", 1), [])

    def test_it_counts_spans_and_labels_so_the_warning_is_actionable(self):
        self._put(1, ".marks.json", json.dumps({"spans": {
            "0": [[0.0, 1.0, "flying"], [2.0, 3.0, "flying"],
                  [4.0, 5.0, "grooming"]]}}))
        (only,) = describe_home(self.video, 1)
        self.assertIn("3 marked spans", only)
        self.assertIn("2 labels", only)

    def test_a_single_span_and_label_are_not_pluralised(self):
        self._put(1, ".marks.json",
                  json.dumps({"spans": {"0": [[0.0, 1.0, "flying"]]}}))
        (only,) = describe_home(self.video, 1)
        self.assertIn("1 marked span across 1 label", only)

    def test_an_unreadable_file_is_reported_not_skipped(self):
        """"Something here I could not open" is still what the user needs to
        know before overwriting it."""
        self._put(1, ".marks.json", "{not json")
        self.assertEqual(describe_home(self.video, 1),
                         ["a marks file (unreadable)"])

    def test_it_names_the_track_the_clip_and_the_tuning(self):
        self._put(1, ".track.npz", b"\0" * 2048)
        self._put(1, ".tuning.json", "{}")
        self._put(1, ".mkv", b"\0" * 4096)
        held = " | ".join(describe_home(self.video, 1))
        self.assertIn("detection track", held)
        self.assertIn("remembered tuning", held)
        self.assertIn("transcoded clip", held)

    def test_unexpected_files_are_counted_rather_than_listed(self):
        ensure_home(self.video, 1)
        for i in range(5):
            with open(os.path.join(replicate_dir(self.video, 1),
                                   f"scratch{i}.bin"), "wb") as f:
                f.write(b"x")
        self.assertEqual(describe_home(self.video, 1), ["5 other files"])


if __name__ == "__main__":
    unittest.main()
