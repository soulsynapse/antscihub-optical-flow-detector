"""Batch O (T26): the container's frame count is a claim, not a measurement.

The bug this covers is not "a wrong number" -- it is *two different quantities
behind one name*. ``FINDINGS.md`` section 15: ``GX010047c2`` advertises 11328
frames and decodes 11308, which made the pre-transcode cut refuse the project's
own primary test footage and made every full-length headless pass over it trip
the truncation guard. The standing workaround was ``--allow-truncated``, i.e.
switching off the guard.

So most of what is asserted here is about *provenance* rather than arithmetic:
which number was used, whether anyone measured it, and what the operator is told
to do when it does not add up.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from core import framecount as fcm
from core import pretranscode as pt


requires_ffmpeg = unittest.skipUnless(shutil.which("ffmpeg"),
                                      "ffmpeg is not on PATH")


class _FakeManifest:
    """Only the three attributes ``resolve_frame_count`` reads."""

    def __init__(self, source_path, frame_count, container_frame_count=0):
        self.source_path = source_path
        self.frame_count = frame_count
        self.container_frame_count = container_frame_count


class _TmpVideo(unittest.TestCase):
    """A stand-in file. resolve_frame_count and the sidecar never decode -- they
    only hash bytes -- so a real video is not needed for the provenance logic and
    would make these tests slow and ffmpeg-dependent for nothing."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.video = os.path.join(self.dir, "clip.MP4")
        with open(self.video, "wb") as f:
            f.write(b"not really a video, but it has stable bytes" * 100)


class TestRecordRoundTrip(_TmpVideo):
    def test_write_read_round_trip(self):
        rec = fcm.build_record(self.video, container_frames=11328,
                               decoded_frames=11308, method="pretranscode-cut")
        path = fcm.write_record(rec)
        self.assertEqual(path, fcm.sidecar_path_for(self.video))
        back = fcm.read_record(path)
        self.assertEqual(back, rec)
        self.assertEqual(back.undecodable, 20)

    def test_version_mismatch_is_refused(self):
        path = fcm.write_record(
            fcm.build_record(self.video, container_frames=10, decoded_frames=10))
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        d["version"] = fcm.FRAMECOUNT_VERSION + 1
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f)
        with self.assertRaises(fcm.FrameCountError):
            fcm.read_record(path)

    def test_build_record_with_a_known_count_does_not_decode(self):
        """The cut hands over the count it already measured. If this decoded, the
        pre-transcode would pay a second full pass for a number in hand."""
        rec = fcm.build_record(self.video, container_frames=5, decoded_frames=4)
        self.assertEqual(rec.decoded_frames, 4)


class TestSidecarIdentity(_TmpVideo):
    def test_absent_sidecar_is_none_not_an_error(self):
        self.assertIsNone(fcm.load_sidecar(self.video))

    def test_valid_sidecar_loads(self):
        fcm.write_record(fcm.build_record(self.video, container_frames=100,
                                          decoded_frames=98))
        rec = fcm.load_sidecar(self.video)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.decoded_frames, 98)

    def test_sidecar_for_a_replaced_file_is_rejected(self):
        """Identity, not existence -- the rule core/shard.py's resume follows.

        A stale count is worse than no count, because it is *trusted*: it would
        be used as a coverage denominator for a file it never described.
        """
        fcm.write_record(fcm.build_record(self.video, container_frames=100,
                                          decoded_frames=98))
        with open(self.video, "wb") as f:
            f.write(b"a completely different file with a different length")
        self.assertIsNone(fcm.load_sidecar(self.video))

    def test_corrupt_sidecar_is_none_rather_than_fatal(self):
        """A corrupt cache entry must not fail a job that would otherwise have
        run correctly against an unverified count."""
        with open(fcm.sidecar_path_for(self.video), "w", encoding="utf-8") as f:
            f.write("{ this is not json")
        self.assertIsNone(fcm.load_sidecar(self.video))


class TestResolveFrameCount(_TmpVideo):
    def test_container_claim_is_the_unverified_fallback(self):
        r = fcm.resolve_frame_count(self.video, 11328)
        self.assertEqual(r.count, 11328)
        self.assertFalse(r.verified)
        self.assertEqual(r.source, fcm.SOURCE_CONTAINER)

    def test_sidecar_beats_the_container(self):
        fcm.write_record(fcm.build_record(self.video, container_frames=11328,
                                          decoded_frames=11308))
        r = fcm.resolve_frame_count(self.video, 11328)
        self.assertEqual(r.count, 11308)
        self.assertTrue(r.verified)
        self.assertEqual(r.source, fcm.SOURCE_SIDECAR)
        self.assertEqual(r.unverified_excess, 20)

    def test_manifest_beats_the_sidecar(self):
        fcm.write_record(fcm.build_record(self.video, container_frames=11328,
                                          decoded_frames=11000))
        man = _FakeManifest(self.video, 11308, container_frame_count=11328)
        r = fcm.resolve_frame_count(self.video, 11328, manifest=man)
        self.assertEqual(r.count, 11308)
        self.assertEqual(r.source, fcm.SOURCE_MANIFEST)

    def test_manifest_for_a_replaced_file_is_ignored(self):
        """Identity, not just path -- the same rule load_sidecar follows.

        Matching on path alone would make the BEST-trusted tier the least-checked
        one: a source re-exported in place keeps its path, and its stale count
        would come back verified=True and become the coverage denominator.
        """
        man = _FakeManifest(self.video, 11308, container_frame_count=11328)
        man.source_size = os.path.getsize(self.video)
        man.quick_sig = fcm.quick_signature(self.video)
        self.assertEqual(fcm.resolve_frame_count(self.video, 11328,
                                                 manifest=man).source,
                         fcm.SOURCE_MANIFEST)
        with open(self.video, "wb") as f:
            f.write(b"a different export at the same path")
        r = fcm.resolve_frame_count(self.video, 11328, manifest=man)
        self.assertEqual(r.source, fcm.SOURCE_CONTAINER)
        self.assertFalse(r.verified)

    def test_manifest_for_a_different_source_is_ignored(self):
        """A manifest describes one file. Reading its length as this file's would
        be the same class of error as the bug this batch exists to fix -- a
        number applied to something it does not measure."""
        man = _FakeManifest(os.path.join(self.dir, "other.MP4"), 999)
        r = fcm.resolve_frame_count(self.video, 11328, manifest=man)
        self.assertEqual(r.count, 11328)
        self.assertEqual(r.source, fcm.SOURCE_CONTAINER)

    def test_the_claim_is_never_rejected_for_disagreeing(self):
        """Both numbers are correct answers to different questions. The claim is
        not used when something better exists; it is not treated as corrupt."""
        r = fcm.resolve_frame_count(self.video, 11328)
        self.assertEqual(r.container_frames, 11328)


@requires_ffmpeg
class TestCountDecodableFrames(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.video = os.path.join(self.dir, "src.mp4")
        # A CFR synthetic: its container claim and its decodable count agree, so
        # this can only show the counter is RIGHT on a clean file. It cannot
        # reproduce the over-claiming source -- a synthetic has no undecodable
        # packets, which is precisely why the original bug survived Batch I's
        # testsrc fixtures. That gap is named here rather than papered over.
        import subprocess
        subprocess.run(
            [shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=3",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", self.video],
            check=True, **pt._popen_kwargs())

    def test_counts_a_clean_file_exactly(self):
        self.assertEqual(fcm.count_decodable_frames(self.video), 30)

    def test_agrees_with_the_container_on_a_cfr_source(self):
        *_, claimed = pt.probe_source(self.video)
        self.assertEqual(fcm.count_decodable_frames(self.video), claimed)

    def test_missing_file_is_an_error_not_a_zero(self):
        """Returning 0 would read as "this video has no frames" and silently
        produce an empty, detection-free pass."""
        with self.assertRaises(fcm.FrameCountError):
            fcm.count_decodable_frames(os.path.join(self.dir, "nope.mp4"))


if __name__ == "__main__":
    unittest.main()
