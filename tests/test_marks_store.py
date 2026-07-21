from __future__ import annotations

import json
import os
import tempfile
import unittest

from gui.marks_store import (add_detection_spans, color_for, load_marks,
                             marks_path, save_marks)


class MarksStoreTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp(prefix="marks_")
        self.video = os.path.join(self._dir, "clip.mp4")

    def tearDown(self):
        for f in os.listdir(self._dir):
            os.remove(os.path.join(self._dir, f))
        os.rmdir(self._dir)

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

    def test_the_format_is_the_shelved_timelines(self):
        # marks_to_dict / set_marks_from_dict speak {"spans": {rid: [[t0,t1,lbl]]},
        # "colors": {...}} -- the rehomed widget must read what Save wrote.
        doc = load_marks(self.video)
        add_detection_spans(doc, 0, "still", [(0.0, 1.0)])
        save_marks(self.video, doc)
        raw = json.load(open(marks_path(self.video), encoding="utf-8"))
        self.assertIn("spans", raw)
        self.assertIn("colors", raw)
        self.assertIn("still", raw["colors"])
        self.assertEqual(raw["spans"]["0"], [[0.0, 1.0, "still"]])

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


if __name__ == "__main__":
    unittest.main()
