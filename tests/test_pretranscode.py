"""Tests for ROI pre-transcode.

The end-to-end tests build a small synthetic video with ffmpeg and cut it, so
they exercise the real filter graph rather than a mock of it -- the graph is
where the traps live (crop rounding, plane selection, rate handling), and a
mocked subprocess would assert only that we can format a string.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
import unittest.mock

import numpy as np

from core import pretranscode as pt
from core.replicates import build_layout, geometry_hash


HAS_FFMPEG = shutil.which("ffmpeg") is not None
requires_ffmpeg = unittest.skipUnless(HAS_FFMPEG, "ffmpeg not on PATH")

REPS = [
    {"id": 1, "label": "left", "frac": [0.0, 0.0, 0.5, 0.5]},
    {"id": 2, "label": "right", "frac": [0.5, 0.5, 1.0, 1.0]},
]


def _make_video(path: str, w: int = 160, h: int = 120, n: int = 24,
                fps: str = "24000/1001") -> None:
    """A deterministic clip with motion, so crops actually differ from one another.

    The lavfi source duration is derived from ``n`` rather than fixed: a hardcoded
    duration silently caps ``-frames:v``, so a caller asking for 600 frames would
    get 60 and any test depending on the clip's length would pass for the wrong
    reason.
    """
    from fractions import Fraction
    rate = float(Fraction(fps))
    duration = n / rate + 1.0 / rate      # a frame of slack against rounding
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi",
         "-i", f"testsrc=size={w}x{h}:rate={fps}:duration={duration:.4f}",
         "-frames:v", str(n), "-c:v", "libx264", "-crf", "18",
         "-pix_fmt", "yuv420p", path],
        check=True, capture_output=True)


def _gray(path: str, w: int, h: int) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True).stdout
    n = len(raw) // (w * h)
    return np.frombuffer(raw[:n * w * h], np.uint8).reshape(n, h, w)


class TestRational(unittest.TestCase):
    def test_ntsc_rates_are_exact_not_rounded(self):
        """The seek trap: 24.0 for 24000/1001 lands 3 frames early by frame 11000."""
        self.assertEqual(pt._as_rational(24000 / 1001), (24000, 1001))
        self.assertEqual(pt._as_rational(30000 / 1001), (30000, 1001))
        self.assertEqual(pt._as_rational(60000 / 1001), (60000, 1001))

    def test_integer_rates_stay_integers(self):
        self.assertEqual(pt._as_rational(25.0), (25, 1))
        self.assertEqual(pt._as_rational(30.0), (30, 1))

    def test_unknown_rate_falls_back_to_a_rational(self):
        num, den = pt._as_rational(17.5)
        self.assertAlmostEqual(num / den, 17.5, places=6)


class TestQualityPresets(unittest.TestCase):
    def test_default_is_a_known_preset(self):
        self.assertIn(pt.DEFAULT_QUALITY, pt.QUALITY_PRESETS)

    def test_only_the_lossless_preset_claims_bit_exactness(self):
        for name, p in pt.QUALITY_PRESETS.items():
            with self.subTest(name):
                self.assertEqual(p["lossless"], p["rms"] == 0.0)

    def test_every_preset_writes_gray(self):
        """The pipeline reads luma; a preset that kept chroma would mislead."""
        for name, p in pt.QUALITY_PRESETS.items():
            with self.subTest(name):
                args = p["args"]
                self.assertEqual(args[args.index("-pix_fmt") + 1], "gray")

    def test_unknown_quality_is_rejected(self):
        with self.assertRaises(pt.PretranscodeError):
            pt.clip_command("x.mp4", _tiles(), ["a.mkv", "b.mkv"], "ludicrous")


def _tiles():
    return build_layout(REPS, 160, 120, 1.0, 16).tiles


class TestFilterGraph(unittest.TestCase):
    def test_one_split_feeds_every_crop(self):
        """A single decode is the entire point; one ffmpeg run per replicate
        would multiply the cost this module exists to pay down."""
        graph, labels = pt.build_filter_graph(_tiles())
        self.assertEqual(len(labels), 2)
        self.assertIn("split=2", graph)
        self.assertEqual(graph.count("crop="), 2)

    def test_single_replicate_uses_null_not_split(self):
        graph, labels = pt.build_filter_graph(build_layout(
            REPS[:1], 160, 120, 1.0, 16).tiles)
        self.assertEqual(len(labels), 1)
        self.assertIn("null", graph)
        self.assertNotIn("split", graph)

    def test_crop_is_exact_so_odd_origins_do_not_shift(self):
        """Without exact=1 an odd x0/y0 rounds to an even boundary, and that
        sub-pixel shift is read by dense flow as real translation."""
        graph, _ = pt.build_filter_graph(_tiles())
        self.assertEqual(graph.count(":exact=1"), 2)

    def test_graph_never_scales(self):
        """Crop is lossless in geometry; scaling would bake a sensitivity
        decision into an artifact that costs a re-decode to regenerate."""
        graph, _ = pt.build_filter_graph(_tiles())
        self.assertNotIn("scale=", graph)

    def test_no_tiles_is_an_error(self):
        with self.assertRaises(pt.PretranscodeError):
            pt.build_filter_graph([])


class TestManifest(unittest.TestCase):
    def _man(self, **kw):
        base = dict(
            version=pt.PRETRANSCODE_VERSION, source_path="/v.mp4",
            source_size=10, source_sha256="a" * 64, quick_sig="b" * 32,
            src_width=160, src_height=120, fps_num=24000, fps_den=1001,
            frame_count=24, geometry_hash=geometry_hash(REPS),
            resolved_scale=1.0, scale_rule="scale-1.0-default",
            codec="libx264-crf12", lossless=False, quality="high",
            quality_rms=1.041,
            clips=(pt.ClipEntry(1, "left", (0, 0, 0.5, 0.5), (0, 0, 80, 60),
                                80, 60, "v__rep01.mkv"),))
        base.update(kw)
        return pt.Manifest(**base)

    def test_fps_is_stored_rational_and_reconstructs_exactly(self):
        m = self._man()
        self.assertEqual((m.fps_num, m.fps_den), (24000, 1001))
        self.assertAlmostEqual(m.fps, 24000 / 1001, places=12)

    def test_round_trips_through_json(self):
        m = self._man()
        blob = json.dumps(m.to_meta())
        self.assertEqual(pt.Manifest.from_meta(json.loads(blob)), m)

    def test_clip_lookup_is_by_id_not_position(self):
        m = self._man(clips=(
            pt.ClipEntry(7, "b", (0, 0, 1, 1), (0, 0, 8, 6), 8, 6, "b.mkv"),
            pt.ClipEntry(3, "a", (0, 0, 1, 1), (0, 0, 8, 6), 8, 6, "a.mkv")))
        self.assertEqual(m.clip_for(3).filename, "a.mkv")
        with self.assertRaises(KeyError):
            m.clip_for(99)

    def test_provenance_key_separates_quality(self):
        """Lossy coding perturbs the frame-to-frame quantity `change` measures,
        so two qualities are different measurements, not one cached result."""
        a = self._man(quality="high", codec="libx264-crf12")
        b = self._man(quality="standard", codec="libx264-crf18")
        self.assertNotEqual(a.provenance_key(), b.provenance_key())

    def test_provenance_key_separates_geometry_and_scale_rule(self):
        base = self._man()
        self.assertNotEqual(base.provenance_key(),
                            self._man(geometry_hash="deadbeef").provenance_key())
        self.assertNotEqual(base.provenance_key(),
                            self._man(scale_rule="organism-relative").provenance_key())

    def test_provenance_key_is_stable_across_equal_manifests(self):
        self.assertEqual(self._man().provenance_key(), self._man().provenance_key())

    def test_version_mismatch_is_refused_on_read(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            meta = self._man().to_meta()
            meta["version"] = pt.PRETRANSCODE_VERSION + 1
            with open(p, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            with self.assertRaises(pt.PretranscodeError):
                pt.read_manifest(p)


class TestClipLayout(unittest.TestCase):
    """Where a clip is written, and how a manifest's ``filename`` resolves."""

    def test_home_layout_names_the_replicates_home(self):
        """Same directory ``core.replicate_home`` puts the track and marks in --
        derived from HOME_FMT rather than re-spelled, so the two cannot drift."""
        from core.replicate_home import home_name
        rel = pt.clip_filename("GX010047", 1)
        self.assertEqual(rel, "GX010047_rep01/GX010047.mkv")
        self.assertEqual(rel.split("/")[0], home_name("x/GX010047.MP4", 1))

    def test_flat_layout_is_the_pre_batch_s_name(self):
        self.assertEqual(pt.clip_filename("GX010047", 1, "flat"),
                         "GX010047__rep01.mkv")

    def test_relative_paths_are_posix_on_every_platform(self):
        """The manifest is JSON and may be written on Windows and read on Linux,
        where a backslash is a legal filename character -- so a native separator
        would resolve there as one absurdly-named file rather than failing."""
        self.assertNotIn("\\", pt.clip_filename("v", 3))

    def test_unknown_layout_is_refused_by_name(self):
        with self.assertRaises(pt.PretranscodeError) as cm:
            pt.clip_filename("v", 1, "nested")
        self.assertIn("home", str(cm.exception))

    def test_clip_path_resolves_a_bare_basename(self):
        """Manifests cut before Batch S carry basenames and must still resolve;
        that is why widening ``filename`` needed no version bump."""
        self.assertEqual(pt.clip_path("out", "v__rep01.mkv"),
                         os.path.join("out", "v__rep01.mkv"))

    def test_clip_path_splits_the_relative_path_into_real_components(self):
        got = pt.clip_path("out", "v_rep01/v.mkv")
        self.assertEqual(got, os.path.join("out", "v_rep01", "v.mkv"))
        # The point of not using os.path.join on the whole string: on Windows
        # that leaves a mixed-separator path that opens fine and so passes every
        # test, while being unprintable and uncomparable.
        self.assertEqual(os.path.dirname(got), os.path.join("out", "v_rep01"))


class TestCliLayoutPlumbing(unittest.TestCase):
    """The CLI half of slice 4, without running a transcode."""

    def _man(self, names):
        return pt.Manifest(
            version=pt.PRETRANSCODE_VERSION, source_path="/v.mp4",
            source_size=10, source_sha256="a" * 64, quick_sig="b" * 32,
            src_width=160, src_height=120, fps_num=24, fps_den=1,
            frame_count=24, geometry_hash="h", resolved_scale=1.0,
            scale_rule="scale-1.0-default", codec="c", lossless=False,
            quality="high", quality_rms=1.0,
            clips=tuple(pt.ClipEntry(i + 1, "x", (0, 0, 1, 1), (0, 0, 8, 6),
                                     8, 6, n) for i, n in enumerate(names)))

    def test_default_layout_is_the_home(self):
        from cli.pretranscode import _build_parser
        self.assertEqual(_build_parser().parse_args(["v.mp4"]).layout,
                         pt.DEFAULT_CLIP_LAYOUT)
        self.assertEqual(pt.DEFAULT_CLIP_LAYOUT, "home")

    def test_superseded_names_clips_a_relayout_orphaned(self):
        """A flat->home re-cut leaves the old clips on disk, silently doubling
        the storage the report just quoted."""
        import tempfile
        from cli.pretranscode import _superseded
        with tempfile.TemporaryDirectory() as d:
            old = pt.clip_path(d, "v__rep01.mkv")
            open(old, "wb").close()
            new = pt.clip_path(d, "v_rep01/v.mkv")
            os.makedirs(os.path.dirname(new))
            open(new, "wb").close()
            got = _superseded(self._man(["v__rep01.mkv"]),
                              self._man(["v_rep01/v.mkv"]), d)
            self.assertEqual(got, [old])

    def test_superseded_is_empty_when_the_layout_did_not_change(self):
        import tempfile
        from cli.pretranscode import _superseded
        with tempfile.TemporaryDirectory() as d:
            man = self._man(["v_rep01/v.mkv"])
            self.assertEqual(_superseded(man, man, d), [])
            self.assertEqual(_superseded(None, man, d), [])


class _FakeProc:
    """Minimal Popen stand-in: canned stdout lines, no real child."""

    def __init__(self, lines, err=()):
        self.stdout = iter(lines)
        self.stderr = iter(err) if err else None
        self.returncode = 0
        self.waited = False

    def wait(self, timeout=None):
        self.waited = True
        return 0


class TestProgressParsing(unittest.TestCase):
    """Direct tests of the ``frame=`` parser.

    Driving this through ffmpeg is unreliable: it emits a progress block only
    about every 0.5 s, so any clip small enough for a fast test finishes inside
    a single block and reports nothing but the final frame. An end-to-end test
    of parsing would therefore pass whether or not parsing worked -- the exact
    vacuity this class exists to avoid.
    """

    def _run(self, lines, total=100, cancel=None):
        seen = []
        pt._pump_progress(_FakeProc(lines), total, seen.append, cancel)
        return seen

    def _stats(self, lines, total=100):
        _, stats = pt._pump_progress(_FakeProc(lines), total, None, None)
        return stats

    def test_frame_lines_become_fractions(self):
        self.assertEqual(self._run(["frame=25\n", "frame=50\n"]), [0.25, 0.5])

    def test_other_progress_keys_are_ignored(self):
        seen = self._run(["fps=30.0\n", "bitrate=N/A\n", "frame=10\n",
                          "progress=continue\n"])
        self.assertEqual(seen, [0.1])

    def test_non_numeric_frame_value_does_not_raise(self):
        """ffmpeg emits frame=N/A before the first frame is written."""
        self.assertEqual(self._run(["frame=N/A\n", "frame=5\n"]), [0.05])

    def test_fraction_is_clamped_at_one(self):
        """Reported frames can exceed a container's advertised count."""
        self.assertEqual(self._run(["frame=250\n"], total=100), [1.0])

    def test_zero_total_reports_no_fraction_but_still_counts(self):
        """A caller with no length estimate (core.framecount) passes total=0.

        There is no denominator to divide by, so no fraction is reported -- but
        the counting must keep working, because the count is the entire reason
        that caller runs at all.
        """
        self.assertEqual(self._run(["frame=5\n"], total=0), [])
        self.assertEqual(self._stats(["frame=5\n"], total=0).frames, 5)

    def test_cancel_raises_before_consuming_the_rest(self):
        with self.assertRaises(pt.PretranscodeCancelled):
            self._run(["frame=1\n"] * 10, cancel=lambda: True)

    def test_stderr_is_drained_and_returned(self):
        """It must be drained concurrently or a full pipe buffer deadlocks."""
        proc = _FakeProc(["frame=1\n"], err=["boom\n", "bang\n"])
        err, _ = pt._pump_progress(proc, 10, None, None)
        self.assertEqual("".join(err), "boom\nbang\n")
        self.assertTrue(proc.waited)

    # -- the stats half (Batch O) --------------------------------------------

    def test_last_frame_value_is_the_measured_length(self):
        """Not the max, not the first: ffmpeg's counter only ever advances, and
        the final block is what the run actually wrote."""
        self.assertEqual(
            self._stats(["frame=10\n", "frame=90\n", "progress=end\n"]).frames,
            90)

    def test_frames_is_none_when_nothing_was_reported(self):
        """Distinct from 0. A run that wrote zero frames and a run whose progress
        stream was never understood are different failures, and the cut refuses
        to write a manifest in either case rather than recording a length it did
        not measure."""
        self.assertIsNone(self._stats(["progress=end\n"]).frames)

    def test_dup_and_drop_are_the_retiming_signal(self):
        clean = self._stats(["frame=5\n", "dup_frames=0\n", "drop_frames=0\n"])
        self.assertFalse(clean.retimed)
        duped = self._stats(["frame=5\n", "dup_frames=3\n", "drop_frames=0\n"])
        self.assertTrue(duped.retimed)
        self.assertEqual((duped.dup, duped.drop), (3, 0))
        dropped = self._stats(["frame=5\n", "dup_frames=0\n", "drop_frames=2\n"])
        self.assertTrue(dropped.retimed)

    def test_na_values_do_not_abort_the_pump(self):
        """ffmpeg writes N/A for fields it cannot fill yet. Aborting on one would
        stop draining stderr, reintroducing the pipe-buffer deadlock."""
        stats = self._stats(["dup_frames=N/A\n", "frame=N/A\n", "frame=7\n",
                             "drop_frames=N/A\n"])
        self.assertEqual((stats.frames, stats.dup, stats.drop), (7, 0, 0))


@requires_ffmpeg
class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.d = tempfile.mkdtemp()
        self.src = os.path.join(self.d, "v.mp4")
        _make_video(self.src)
        self.out = os.path.join(self.d, "clips")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_cut_writes_a_clip_per_replicate_and_a_manifest(self):
        man = pt.build_pretranscode(self.src, REPS, self.out)
        self.assertEqual(len(man.clips), 2)
        for c in man.clips:
            self.assertTrue(os.path.isfile(pt.clip_path(self.out, c.filename)))
        self.assertTrue(os.path.isfile(pt.manifest_path_for(self.src, self.out)))

    # -- Batch O: the two halves of the replaced guard ------------------------

    def _forced_stats(self, frames=None, dup=0, drop=0, err=()):
        """Drive build_pretranscode's checks with a chosen progress result.

        ffmpeg genuinely re-timing or genuinely dying part way cannot be provoked
        on demand from a synthetic fixture, which is the same reason the original
        bug survived Batch I's tests. Forcing the parsed result exercises the
        branch that decides what to DO about it, which is what regressed.
        """
        real = pt._pump_progress

        def fake(proc, total, progress, cancel):
            _, stats = real(proc, total, progress, cancel)
            if frames is not None:
                stats.frames = frames
            stats.dup, stats.drop = dup, drop
            return list(err), stats
        return unittest.mock.patch.object(pt, "_pump_progress", fake)

    def test_retiming_is_refused_on_dup_or_drop(self):
        with self._forced_stats(dup=3):
            with self.assertRaises(pt.PretranscodeError) as cm:
                pt.build_pretranscode(self.src, REPS, self.out)
        self.assertIn("re-timed", str(cm.exception))

    def _over_claiming_source(self, excess=20):
        """Make the container advertise more frames than decode.

        This is GX010047c2 exactly: the claim is inflated while the decode is
        clean and complete. Patching the claim rather than the measured count
        models the real file, and leaves check (b) -- clips against what ffmpeg
        wrote -- agreeing, as it does on the real file.
        """
        real = pt.probe_source

        def fake(path):
            w, h, num, den, n = real(path)
            return w, h, num, den, n + excess
        return unittest.mock.patch.object(pt, "probe_source", fake)

    def test_clean_over_claim_is_accepted_and_recorded(self):
        """The legitimate case, and the whole reason a bare shortfall cannot be
        the trigger -- refusing it is the bug this batch fixed. GX010047c2
        over-claims by 20 while ffmpeg exits 0 with ZERO bytes of stderr,
        measured rather than assumed."""
        with self._over_claiming_source(20):
            man = pt.build_pretranscode(self.src, REPS, self.out)
        self.assertEqual(man.container_frame_count, man.frame_count + 20)
        for c in man.clips:
            self.assertEqual(c.frame_count, man.frame_count)

    def test_short_cut_with_ffmpeg_errors_is_refused(self):
        """The regression guard. A bare shortfall is legitimate (above), so the
        discriminator is the stderr ffmpeg emits when it actually fails. Without
        this, a decode dying at frame 5000 of 11328 and exiting 0 records 5000 as
        the source's true length, which resolve_frame_count then hands to the
        coverage guard as VERIFIED -- so 6308 unread frames are reported
        examined-and-clear."""
        with self._over_claiming_source(20), \
                self._forced_stats(err=["Error while decoding stream\n"]):
            with self.assertRaises(pt.PretranscodeError) as cm:
                pt.build_pretranscode(self.src, REPS, self.out)
        self.assertIn("short decode", str(cm.exception))
        self.assertFalse(os.path.exists(pt.manifest_path_for(self.src, self.out)))

    def test_frame_count_is_the_measurement_not_the_claim(self):
        man = pt.build_pretranscode(self.src, REPS, self.out)
        self.assertGreater(man.frame_count, 0)
        for c in man.clips:
            self.assertEqual(c.frame_count, man.frame_count)

    def test_the_cut_writes_a_framecount_sidecar(self):
        """The count is free here and nowhere else, so it is recorded beside the
        source rather than only inside the manifest."""
        from core import framecount as fcm
        man = pt.build_pretranscode(self.src, REPS, self.out)
        rec = fcm.load_sidecar(self.src)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.decoded_frames, man.frame_count)
        self.assertEqual(rec.method, "pretranscode-cut")

    def test_manifest_round_trips_from_disk(self):
        man = pt.build_pretranscode(self.src, REPS, self.out)
        self.assertEqual(
            pt.read_manifest(pt.manifest_path_for(self.src, self.out)), man)

    def test_clip_dimensions_match_the_replicate_box(self):
        man = pt.build_pretranscode(self.src, REPS, self.out)
        for c in man.clips:
            x0, y0, x1, y1 = c.source_box
            self.assertEqual((c.width, c.height), (x1 - x0, y1 - y0))
            arr = _gray(pt.clip_path(self.out, c.filename), c.width, c.height)
            self.assertGreater(len(arr), 0)
            self.assertEqual(arr.shape[1:], (c.height, c.width))

    def test_lossless_clip_is_bit_exact_against_a_direct_crop(self):
        """The claim the lossless preset makes, checked rather than asserted."""
        man = pt.build_pretranscode(self.src, REPS, self.out, quality="lossless")
        c = man.clip_for(1)
        x0, y0, x1, y1 = c.source_box
        ref_path = os.path.join(self.d, "ref.mkv")
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", self.src,
             "-vf", f"crop={x1-x0}:{y1-y0}:{x0}:{y0}:exact=1,format=gray",
             "-c:v", "ffv1", "-pix_fmt", "gray", ref_path],
            check=True, capture_output=True)
        got = _gray(pt.clip_path(self.out, c.filename), c.width, c.height)
        ref = _gray(ref_path, c.width, c.height)
        n = min(len(got), len(ref))
        self.assertGreater(n, 0)
        self.assertEqual(np.abs(got[:n].astype(int) - ref[:n].astype(int)).max(), 0)

    def test_clips_hold_different_pixels_for_different_boxes(self):
        """Guards the split/map pairing -- a mis-wired graph would write the same
        crop to every output, which the dimension checks alone would not catch."""
        man = pt.build_pretranscode(self.src, REPS, self.out, quality="lossless")
        a, b = man.clip_for(1), man.clip_for(2)
        ga = _gray(pt.clip_path(self.out, a.filename), a.width, a.height)
        gb = _gray(pt.clip_path(self.out, b.filename), b.width, b.height)
        self.assertFalse(np.array_equal(ga, gb))

    def test_refuses_to_clobber_existing_clips_unless_asked(self):
        pt.build_pretranscode(self.src, REPS, self.out)
        with self.assertRaises(pt.PretranscodeError):
            pt.build_pretranscode(self.src, REPS, self.out)
        pt.build_pretranscode(self.src, REPS, self.out, overwrite=True)

    def test_cut_writes_each_clip_into_its_replicates_home(self):
        """Slice 4: the clip joins the track, marks and tuning rather than
        introducing a layout of its own."""
        from core.replicate_home import home_name
        man = pt.build_pretranscode(self.src, REPS, self.out)
        for c in man.clips:
            self.assertEqual(c.filename,
                             f"{home_name(self.src, c.replicate_id)}/v.mkv")
            self.assertTrue(os.path.isfile(pt.clip_path(self.out, c.filename)))

    def test_flat_layout_still_cuts_and_verifies(self):
        man = pt.build_pretranscode(self.src, REPS, self.out, clip_layout="flat")
        self.assertEqual(sorted(c.filename for c in man.clips),
                         ["v__rep01.mkv", "v__rep02.mkv"])
        pt.verify_manifest(man, self.out, REPS)

    def test_layout_does_not_change_the_provenance_key(self):
        """It moves bytes, not pixels. Two cuts that differ only in where the
        same clip sits MUST compare as equal, or every clip-derived cache entry
        would be invalidated by a filesystem decision."""
        home = pt.build_pretranscode(self.src, REPS, self.out, quality="lossless")
        flat = pt.build_pretranscode(self.src, REPS, self.out, quality="lossless",
                                     clip_layout="flat", overwrite=True)
        self.assertNotEqual(home.clips[0].filename, flat.clips[0].filename)
        self.assertEqual(home.provenance_key(), flat.provenance_key())

    def test_a_failed_cut_removes_its_clips_but_not_the_home(self):
        """Batch S ruling 2: a home may already hold a curated corpus, so a
        cancelled or failed transcode must never take the directory with it."""
        with self._forced_stats(dup=1):
            with self.assertRaises(pt.PretranscodeError):
                pt.build_pretranscode(self.src, REPS, self.out)
        home = os.path.join(self.out, "v_rep01")
        self.assertTrue(os.path.isdir(home))
        self.assertEqual(os.listdir(home), [])

    def test_verify_accepts_a_fresh_cut(self):
        man = pt.build_pretranscode(self.src, REPS, self.out)
        pt.verify_manifest(man, self.out, REPS)

    def test_verify_rejects_moved_geometry(self):
        """A moved box reading a stale clip would attribute one region's pixels
        to another's detections."""
        man = pt.build_pretranscode(self.src, REPS, self.out)
        moved = [dict(REPS[0], frac=[0.0, 0.0, 0.4, 0.4]), REPS[1]]
        with self.assertRaises(pt.PretranscodeError):
            pt.verify_manifest(man, self.out, moved)

    def test_verify_rejects_a_missing_clip(self):
        man = pt.build_pretranscode(self.src, REPS, self.out)
        os.remove(pt.clip_path(self.out, man.clips[0].filename))
        with self.assertRaises(pt.PretranscodeError):
            pt.verify_manifest(man, self.out, REPS)

    def test_verify_rejects_a_changed_source(self):
        man = pt.build_pretranscode(self.src, REPS, self.out)
        _make_video(self.src, n=20)
        with self.assertRaises(pt.PretranscodeError):
            pt.verify_manifest(man, self.out, REPS)

    def test_progress_reaches_one_and_never_goes_backwards(self):
        seen = []
        pt.build_pretranscode(self.src, REPS, self.out, progress=seen.append)
        self.assertTrue(seen)
        self.assertEqual(seen[-1], 1.0)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in seen))
        # The transcode reports 0..0.98 and the provenance hash the remainder;
        # if either phase forgot its offset the sequence would dip.
        self.assertEqual(seen, sorted(seen))

    def test_cancel_removes_partial_clips(self):
        """A half-written clip that looked complete would be indistinguishable
        from a real one to every later pass."""
        with self.assertRaises(pt.PretranscodeCancelled):
            pt.build_pretranscode(self.src, REPS, self.out,
                                  should_cancel=lambda: True)
        leftovers = [f for f in os.listdir(self.out)] if os.path.isdir(self.out) else []
        self.assertEqual([f for f in leftovers if f.endswith(".mkv")], [])

    def test_cancel_is_not_a_keyboard_interrupt(self):
        """A real Ctrl-C must stay distinguishable from a requested cancel."""
        self.assertFalse(issubclass(pt.PretranscodeCancelled, KeyboardInterrupt))

    def test_ffmpeg_failure_surfaces_stderr(self):
        """The stderr drain also has to still deliver the diagnostic on failure."""
        bad = os.path.join(self.d, "not-a-video.mp4")
        with open(bad, "wb") as f:
            f.write(b"not video data" * 100)
        with self.assertRaises(pt.PretranscodeError) as cm:
            pt.build_pretranscode(bad, REPS, self.out)
        self.assertNotIn("(no stderr)", str(cm.exception))

    def test_source_rate_is_captured_rationally(self):
        man = pt.build_pretranscode(self.src, REPS, self.out)
        self.assertEqual((man.fps_num, man.fps_den), (24000, 1001))


if __name__ == "__main__":
    unittest.main()
