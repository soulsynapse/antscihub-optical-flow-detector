from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from gui.marks_store import (add_detection_spans, color_for, load_marks,
                             load_palette, marks_path, save_marks,
                             save_palette)


class MarksStoreTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp(prefix="marks_")
        self.video = os.path.join(self._dir, "clip.mp4")
        # rmtree rather than a file loop: marks now write into replicate home
        # SUBDIRECTORIES, which a loop over os.listdir cannot remove.
        self.addCleanup(shutil.rmtree, self._dir, True)

    def test_sidecar_sits_next_to_the_video(self):
        self.assertEqual(marks_path(self.video),
                         os.path.join(self._dir, "clip.marks.json"))
        self.assertIsNone(marks_path(""))

    def test_load_of_missing_file_is_an_empty_doc(self):
        doc = load_marks(self.video)
        self.assertEqual(doc, {"spans": {}, "colors": {}, "provenance": {}})

    def test_round_trip_writes_seconds_keyed_by_region(self):
        doc = load_marks(self.video)
        add_detection_spans(doc, 2, "flying", [(1.5, 2.0), (3.0, 3.25)],
                            provenance={"channel": "change"})
        wrote, note = save_marks(self.video, doc)
        self.assertTrue(wrote, note)

        reloaded = load_marks(self.video)
        self.assertEqual(reloaded["spans"]["2"],
                         [[1.5, 2.0, "flying"], [3.0, 3.25, "flying"]])
        # Provenance rides along, stamped with a save time.
        self.assertEqual(reloaded["provenance"]["flying"]["channel"], "change")
        self.assertIn("saved_utc", reloaded["provenance"]["flying"])

    def test_the_span_dialect_is_kept_even_in_a_per_replicate_file(self):
        """``{"spans": {rid: [[t0, t1, label]]}}``, region-keyed, in SECONDS.

        The region key is redundant inside a home that holds one region, and it
        is kept anyway: it is what lets the legacy read-through be a *filter*
        rather than a converter, and a converter is the thing that can be wrong
        about which region it is rewriting.
        """
        doc = load_marks(self.video)
        add_detection_spans(doc, 0, "still", [(0.0, 1.0)])
        save_marks(self.video, doc)
        raw = json.load(open(marks_path(self.video), encoding="utf-8"))
        self.assertIn("spans", raw)
        self.assertEqual(raw["spans"]["0"], [[0.0, 1.0, "still"]])

    def test_saving_spans_does_not_assign_a_colour(self):
        """The palette is per-CLIP and the spans are per-replicate, so folding
        the assignment into the span write would restart insertion order inside
        every home and give one behaviour a different colour per replicate."""
        doc = load_marks(self.video)
        add_detection_spans(doc, 0, "still", [(0.0, 1.0)])
        self.assertEqual(doc["colors"], {})

    def test_resaving_same_label_and_region_replaces_not_appends(self):
        doc = load_marks(self.video)
        add_detection_spans(doc, 1, "flying", [(1.0, 2.0)])
        add_detection_spans(doc, 1, "flying", [(5.0, 6.0)])
        self.assertEqual(doc["spans"]["1"], [[5.0, 6.0, "flying"]])

    def test_other_labels_and_regions_survive_a_resave(self):
        doc = load_marks(self.video)
        add_detection_spans(doc, 1, "flying", [(1.0, 2.0)])
        add_detection_spans(doc, 1, "still", [(3.0, 4.0)])
        add_detection_spans(doc, 2, "flying", [(7.0, 8.0)])
        # Re-saving flying in region 1 leaves still (same region) and flying
        # (other region) untouched.
        add_detection_spans(doc, 1, "flying", [(9.0, 10.0)])
        self.assertEqual(sorted(doc["spans"]["1"]),
                         [[3.0, 4.0, "still"], [9.0, 10.0, "flying"]])
        self.assertEqual(doc["spans"]["2"], [[7.0, 8.0, "flying"]])

    def test_spans_are_kept_sorted_by_start(self):
        doc = load_marks(self.video)
        add_detection_spans(doc, 0, "x", [(5.0, 6.0), (1.0, 2.0), (3.0, 4.0)])
        starts = [s[0] for s in doc["spans"]["0"]]
        self.assertEqual(starts, sorted(starts))

    def test_colors_are_stable_and_distinct_by_insertion_order(self):
        doc = {"spans": {}, "colors": {}, "provenance": {}}
        c1 = color_for(doc, "a")
        c2 = color_for(doc, "b")
        self.assertNotEqual(c1, c2)
        # Same label -> same colour, and it does not consume a new slot.
        self.assertEqual(color_for(doc, "a"), c1)
        self.assertEqual(len(doc["colors"]), 2)

    def test_corrupt_file_reads_as_empty_not_raise(self):
        with open(marks_path(self.video), "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(load_marks(self.video),
                         {"spans": {}, "colors": {}, "provenance": {}})


class PerReplicateMarksTests(unittest.TestCase):
    """Batch S slice 3: marks live in the replicate's home, and the legacy
    per-video corpus is read THROUGH rather than migrated."""

    def setUp(self):
        self._dir = tempfile.mkdtemp(prefix="marks_home_")
        self.video = os.path.join(self._dir, "clip.mp4")
        self.addCleanup(shutil.rmtree, self._dir, True)

    def _write_legacy(self):
        """A pre-homes sidecar holding two regions' spans and a palette."""
        doc = {"spans": {}, "colors": {}, "provenance": {}}
        add_detection_spans(doc, 1, "flying", [(1.0, 2.0)], {"channel": "change"})
        add_detection_spans(doc, 2, "still", [(5.0, 6.0)], {"channel": "change"})
        doc["colors"] = {"flying": "#4caf50", "still": "#2196f3"}
        save_marks(self.video, doc)
        return doc

    def test_marks_go_into_the_replicates_home(self):
        self.assertEqual(
            marks_path(self.video, replicate_id=2),
            os.path.join(self._dir, "clip_rep02", "clip.marks.json"))

    def test_a_home_read_falls_through_to_the_legacy_file_filtered(self):
        self._write_legacy()
        doc = load_marks(self.video, replicate_id=7, legacy_region=1)
        # Only region 1's spans, and only region 1's provenance.
        self.assertEqual(doc["spans"], {"1": [[1.0, 2.0, "flying"]]})
        self.assertIn("flying", doc["provenance"])
        self.assertNotIn("still", doc["provenance"])

    def test_the_read_through_never_touches_the_legacy_file(self):
        self._write_legacy()
        before = open(marks_path(self.video), encoding="utf-8").read()
        load_marks(self.video, replicate_id=7, legacy_region=1)
        load_marks(self.video, replicate_id=8, legacy_region=2)
        self.assertEqual(open(marks_path(self.video), encoding="utf-8").read(),
                         before)

    def test_a_home_with_marks_stops_reading_the_legacy_file(self):
        self._write_legacy()
        doc = load_marks(self.video, replicate_id=7, legacy_region=1)
        add_detection_spans(doc, 1, "flying", [(9.0, 9.5)])
        save_marks(self.video, doc, replicate_id=7)
        again = load_marks(self.video, replicate_id=7, legacy_region=1)
        self.assertEqual(again["spans"]["1"], [[9.0, 9.5, "flying"]])

    def test_without_a_legacy_region_nothing_is_inherited(self):
        """A marks file carries no stamp naming its region, so with no explicit
        index there is nothing to filter on -- and inheriting every region's
        spans would attribute one animal's labels to another."""
        self._write_legacy()
        self.assertEqual(load_marks(self.video, replicate_id=7)["spans"], {})

    def test_a_region_absent_from_the_legacy_file_reads_empty(self):
        self._write_legacy()
        self.assertEqual(
            load_marks(self.video, replicate_id=9, legacy_region=5)["spans"], {})

    def test_the_palette_survives_a_palette_write(self):
        """save_palette shares a file with a hand-curated corpus, so it must
        touch only ``colors``."""
        self._write_legacy()
        save_palette(self.video, {"flying": "#ffffff", "new": "#000000"})
        raw = json.load(open(marks_path(self.video), encoding="utf-8"))
        self.assertEqual(raw["colors"],
                         {"flying": "#ffffff", "new": "#000000"})
        self.assertEqual(raw["spans"]["1"], [[1.0, 2.0, "flying"]])
        self.assertEqual(raw["spans"]["2"], [[5.0, 6.0, "still"]])
        self.assertIn("flying", raw["provenance"])

    def test_a_palette_write_preserves_unknown_keys(self):
        save_marks(self.video, {"spans": {}, "colors": {}, "provenance": {},
                                "notes": "hand written"})
        save_palette(self.video, {"a": "#111111"})
        raw = json.load(open(marks_path(self.video), encoding="utf-8"))
        self.assertEqual(raw["notes"], "hand written")
        self.assertEqual(raw["colors"], {"a": "#111111"})

    def test_load_palette_reads_the_per_video_file(self):
        self._write_legacy()
        self.assertEqual(load_palette(self.video),
                         {"flying": "#4caf50", "still": "#2196f3"})


if __name__ == "__main__":
    unittest.main()
