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

from core.replicate_home import (GEOMETRY_NAME, describe_home, ensure_home,
                                 generation_dir, home_name, home_path,
                                 list_generations, next_generation,
                                 replicate_dir, restore_generation,
                                 retire_current, sync_homes)


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


class GenerationTest(unittest.TestCase):
    """Retiring a geometry, and the swap that brings one back.

    The defects here are the same shape as the rest of this file: a retire that
    leaves one file behind, or a restore that moves the results without the
    rectangle, produces a home that looks fine and attributes one rectangle's
    measurements to another.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = os.path.join(self.tmp.name, "GX010047.MP4")
        open(self.video, "wb").close()
        ensure_home(self.video, 1)
        self.addCleanup(self.tmp.cleanup)

    def _put(self, name: str, data="x"):
        p = os.path.join(replicate_dir(self.video, 1), name)
        mode = "wb" if isinstance(data, bytes) else "w"
        with open(p, mode) as f:
            f.write(data)
        return p

    def _root_files(self):
        d = replicate_dir(self.video, 1)
        return sorted(n for n in os.listdir(d)
                      if os.path.isfile(os.path.join(d, n)))

    def test_an_empty_home_retires_nothing(self):
        """Otherwise every nudge of a fresh box leaves a numbered directory
        recording nothing, and the history stops being readable."""
        self.assertIsNone(retire_current(self.video, 1, (0.1, 0.1, 0.2, 0.2)))
        self.assertEqual(list_generations(self.video, 1), [])

    def test_retiring_moves_every_root_file_and_records_the_rectangle(self):
        self._put("GX010047.track.npz", b"\0" * 512)
        self._put("GX010047.marks.json", "{}")
        self._put("scratch.bin", "note")
        gen = retire_current(self.video, 1, (0.1, 0.2, 0.3, 0.4))

        self.assertEqual(gen, 1)
        # Nothing left at the root: a file that stayed would describe the OLD
        # rectangle while sitting where the new one's results go.
        self.assertEqual(self._root_files(), [])
        (g,) = list_generations(self.video, 1)
        self.assertEqual(g["frac"], (0.1, 0.2, 0.3, 0.4))
        self.assertEqual(
            sorted(n for n in os.listdir(g["path"]) if n != GEOMETRY_NAME),
            ["GX010047.marks.json", "GX010047.track.npz", "scratch.bin"])

    def test_generations_number_from_the_listing_not_from_a_count(self):
        """A hand-removed directory must not let a later retirement reissue its
        number and inherit its place in the history."""
        for i in range(3):
            self._put("GX010047.marks.json", f"{{\"n\": {i}}}")
            retire_current(self.video, 1, (0.1, 0.1, 0.2, 0.2))
        self.assertEqual([g["gen"] for g in list_generations(self.video, 1)],
                         [1, 2, 3])

        import shutil
        shutil.rmtree(generation_dir(self.video, 1, 2))
        self.assertEqual(next_generation(self.video, 1), 4)
        self._put("GX010047.marks.json", "{}")
        self.assertEqual(retire_current(self.video, 1, (0.5, 0.5, 0.6, 0.6)), 4)

    def test_a_user_directory_is_not_mistaken_for_a_generation(self):
        os.makedirs(os.path.join(replicate_dir(self.video, 1), "old_notes"))
        self.assertEqual(list_generations(self.video, 1), [])
        self.assertEqual(next_generation(self.video, 1), 1)

    def test_restoring_swaps_and_returns_the_retired_rectangle(self):
        self._put("GX010047.marks.json", '{"spans": {"0": []}, "gen": 1}')
        retire_current(self.video, 1, (0.1, 0.1, 0.2, 0.2))
        self._put("GX010047.marks.json", '{"spans": {"0": []}, "gen": 2}')

        frac = restore_generation(self.video, 1, 1, (0.7, 0.7, 0.8, 0.8))

        # The rectangle comes back WITH the files -- that is the whole gesture.
        self.assertEqual(frac, (0.1, 0.1, 0.2, 0.2))
        with open(home_path(self.video, 1, ".marks.json")) as f:
            self.assertEqual(json.load(f)["gen"], 1)
        # What was current is retired in turn, so the swap is reversible.
        (g,) = list_generations(self.video, 1)
        self.assertEqual(g["frac"], (0.7, 0.7, 0.8, 0.8))
        self.assertFalse(os.path.exists(generation_dir(self.video, 1, 1)))

    def test_restoring_twice_returns_to_where_it_started(self):
        self._put("GX010047.marks.json", '{"gen": 1}')
        retire_current(self.video, 1, (0.1, 0.1, 0.2, 0.2))
        self._put("GX010047.marks.json", '{"gen": 2}')

        back = restore_generation(self.video, 1, 1, (0.7, 0.7, 0.8, 0.8))
        (g,) = list_generations(self.video, 1)
        again = restore_generation(self.video, 1, g["gen"], back)

        self.assertEqual(again, (0.7, 0.7, 0.8, 0.8))
        with open(home_path(self.video, 1, ".marks.json")) as f:
            self.assertEqual(json.load(f)["gen"], 2)

    def test_a_generation_with_no_rectangle_is_listed_but_refuses_restore(self):
        """Restoring results while leaving the box where it is recreates exactly
        the misattribution the retirement prevented."""
        self._put("GX010047.marks.json", "{}")
        retire_current(self.video, 1, (0.1, 0.1, 0.2, 0.2))
        os.remove(os.path.join(generation_dir(self.video, 1, 1), GEOMETRY_NAME))

        (g,) = list_generations(self.video, 1)
        self.assertIsNone(g["frac"])
        with self.assertRaises(ValueError):
            restore_generation(self.video, 1, 1, (0.7, 0.7, 0.8, 0.8))

    def test_a_missing_generation_raises_rather_than_silently_doing_nothing(self):
        with self.assertRaises(FileNotFoundError):
            restore_generation(self.video, 1, 9, (0.1, 0.1, 0.2, 0.2))

    def test_generations_are_named_as_kept_never_as_at_stake(self):
        """describe_home backs a dialog about what a MOVE risks. Retired
        generations are the one thing a move does not put at risk, so counting
        them among the "other files" would invert the warning."""
        self._put("GX010047.marks.json", "{}")
        retire_current(self.video, 1, (0.1, 0.1, 0.2, 0.2))
        self._put("GX010047.track.npz", b"\0" * 512)

        held = describe_home(self.video, 1)
        self.assertIn("1 retired geometry (kept)", held)
        self.assertFalse([h for h in held if "other file" in h])

    def test_a_generation_describes_its_own_contents(self):
        self._put("GX010047.track.npz", b"\0" * 2048)
        self._put("GX010047.marks.json",
                  json.dumps({"spans": {"0": [[0.0, 1.0, "flying"]]}}))
        retire_current(self.video, 1, (0.1, 0.1, 0.2, 0.2))

        (g,) = list_generations(self.video, 1)
        held = " | ".join(g["held"])
        self.assertIn("detection track", held)
        self.assertIn("1 marked span", held)
        # Its own geometry record is structure, not content.
        self.assertNotIn("other file", held)


if __name__ == "__main__":
    unittest.main()
