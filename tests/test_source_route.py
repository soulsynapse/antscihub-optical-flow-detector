"""Which pixels a pass reads, decided per call (Batch S slice 5).

These build a real cut with ffmpeg where they can, because the three outcomes
this module distinguishes -- no manifest, somebody else's manifest, our own but
stale -- are separated by facts on disk (a file's size, a source's signature, a
geometry hash) rather than by anything a mock would exercise.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from core import pretranscode as pt
from core.source_route import (CLIPS, SOURCE, find_manifest_path,
                               resolve_source)
from tests.test_pretranscode import REPS, _make_video, requires_ffmpeg


class _CutFixture(unittest.TestCase):
    """A real 2-replicate cut, so every route decision runs against real files."""

    @classmethod
    def setUpClass(cls):
        if shutil.which("ffmpeg") is None:
            raise unittest.SkipTest("ffmpeg not on PATH")
        cls.dir = tempfile.mkdtemp(prefix="route-")
        cls.video = os.path.join(cls.dir, "clip.mp4")
        _make_video(cls.video, n=8)
        cls.manifest = pt.build_pretranscode(cls.video, REPS, cls.dir,
                                             quality="standard")
        cls.manifest_path = pt.manifest_path_for(cls.video, cls.dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)


@requires_ffmpeg
class ResolveTests(_CutFixture):
    def test_a_cut_video_routes_to_its_clips(self):
        route = resolve_source(self.video, REPS)
        self.assertEqual(route.kind, CLIPS)
        self.assertEqual(route.reason, "")
        self.assertEqual(route.extract_kwargs["manifest"], self.manifest)
        self.assertTrue(os.path.isdir(route.extract_kwargs["clip_dir"]))

    def test_the_clip_is_found_beside_the_video_with_no_clip_dir_given(self):
        """Slice 4 put the clip in the replicate's home and the manifest beside
        the source, so a cut video needs no --manifest and no --clip-dir. That is
        the whole point of routing: the cut stopped being a mode."""
        self.assertEqual(find_manifest_path(self.video, None),
                         self.manifest_path)

    def test_an_uncut_video_routes_to_the_source_and_says_why(self):
        other = os.path.join(self.dir, "never-cut.mp4")
        _make_video(other, n=4)
        route = resolve_source(other, REPS)
        self.assertEqual(route.kind, SOURCE)
        self.assertIn("no pre-transcode manifest", route.reason)
        # Not an error. Clips are an optional speed-up and a corpus that was
        # never cut must still run.
        self.assertEqual(route.extract_kwargs, {"manifest": None,
                                                "clip_dir": None})

    def test_opting_out_routes_to_the_source_without_touching_the_disk(self):
        route = resolve_source(self.video, REPS, allow_clips=False)
        self.assertEqual(route.kind, SOURCE)
        self.assertIn("disabled", route.reason)

    def test_a_moved_box_RAISES_rather_than_falling_back(self):
        """Ruled 2026-07-21. A silent fallback would pay ~25x the decode AND
        cross a provenance boundary to do it, on footage the operator has
        evidently already cut once -- and say nothing about either."""
        moved = [dict(REPS[0], frac=[0.0, 0.0, 0.4, 0.5]), REPS[1]]
        with self.assertRaises(pt.PretranscodeError) as cm:
            resolve_source(self.video, moved)
        self.assertIn("geometry", str(cm.exception))

    def test_a_truncated_clip_RAISES(self):
        entry = self.manifest.clips[0]
        p = pt.clip_path(self.dir, entry.filename)
        keep = open(p, "rb").read()
        try:
            with open(p, "wb") as f:
                f.write(keep[:len(keep) // 2])
            with self.assertRaises(pt.PretranscodeError):
                resolve_source(self.video, REPS)
        finally:
            with open(p, "wb") as f:
                f.write(keep)

    def test_a_manifest_for_a_DIFFERENT_video_is_not_ours_and_does_not_raise(self):
        """Manifests are named from the video's basename, so two sources called
        GX010047.MP4 in different session directories map to one manifest path
        under a shared --clip-dir. verify_manifest cannot catch that: it
        validates the manifest against the source the MANIFEST names, which is
        present and unchanged. For the video in hand, no manifest exists."""
        sub = os.path.join(self.dir, "other-session")
        os.makedirs(sub, exist_ok=True)
        twin = os.path.join(sub, "clip.mp4")          # same basename, other bytes
        _make_video(twin, n=6)
        route = resolve_source(twin, REPS, clip_dir=self.dir)
        self.assertEqual(route.kind, SOURCE)
        self.assertIn("not from this video", route.reason)

    def test_an_EXPLICIT_manifest_for_a_different_video_raises(self):
        """Discovered is somebody else's file; named is a usage error, and the
        difference is that the operator asserted this one applies."""
        sub = os.path.join(self.dir, "explicit-session")
        os.makedirs(sub, exist_ok=True)
        twin = os.path.join(sub, "clip.mp4")
        _make_video(twin, n=6)
        with self.assertRaises(pt.PretranscodeError):
            resolve_source(twin, REPS, clip_dir=self.dir,
                           manifest_path=self.manifest_path)


@requires_ffmpeg
class SampleKindTests(_CutFixture):
    def test_a_source_route_reports_the_default_cost_sample_kind(self):
        route = resolve_source(self.video, REPS, allow_clips=False)
        self.assertEqual(route.sample_kind, "source")

    def test_a_clip_route_carries_the_provenance_key_not_a_bare_label(self):
        """Two cuts at different quality are different pixels and a different
        decode floor, so they must not share a cost fit either."""
        route = resolve_source(self.video, REPS)
        self.assertTrue(route.sample_kind.startswith("clips:"))
        self.assertIn(self.manifest.provenance_key(), route.sample_kind)


class RouteShapeTests(unittest.TestCase):
    """No decode needed: these pin the contract call sites read."""

    def test_extract_kwargs_always_has_both_keys(self):
        """So a call site reads **route.extract_kwargs unconditionally. The
        branch is what would drift -- it is the shape that lets one of the
        extraction paths quietly keep decoding the source."""
        src = resolve_source("nope.mp4", allow_clips=False)
        self.assertEqual(set(src.extract_kwargs), {"manifest", "clip_dir"})

    def test_a_source_route_describes_itself_for_a_status_line(self):
        src = resolve_source("nope.mp4", allow_clips=False)
        self.assertIn("source", src.describe())


if __name__ == "__main__":
    unittest.main()
