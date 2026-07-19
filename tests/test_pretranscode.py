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

    def test_zero_frame_count_does_not_divide_by_zero(self):
        self.assertEqual(self._run(["frame=5\n"], total=0), [1.0])

    def test_cancel_raises_before_consuming_the_rest(self):
        with self.assertRaises(pt.PretranscodeCancelled):
            self._run(["frame=1\n"] * 10, cancel=lambda: True)

    def test_stderr_is_drained_and_returned(self):
        """It must be drained concurrently or a full pipe buffer deadlocks."""
        proc = _FakeProc(["frame=1\n"], err=["boom\n", "bang\n"])
        err = pt._pump_progress(proc, 10, None, None)
        self.assertEqual("".join(err), "boom\nbang\n")
        self.assertTrue(proc.waited)


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
            self.assertTrue(os.path.isfile(os.path.join(self.out, c.filename)))
        self.assertTrue(os.path.isfile(pt.manifest_path_for(self.src, self.out)))

    def test_manifest_round_trips_from_disk(self):
        man = pt.build_pretranscode(self.src, REPS, self.out)
        self.assertEqual(
            pt.read_manifest(pt.manifest_path_for(self.src, self.out)), man)

    def test_clip_dimensions_match_the_replicate_box(self):
        man = pt.build_pretranscode(self.src, REPS, self.out)
        for c in man.clips:
            x0, y0, x1, y1 = c.source_box
            self.assertEqual((c.width, c.height), (x1 - x0, y1 - y0))
            arr = _gray(os.path.join(self.out, c.filename), c.width, c.height)
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
        got = _gray(os.path.join(self.out, c.filename), c.width, c.height)
        ref = _gray(ref_path, c.width, c.height)
        n = min(len(got), len(ref))
        self.assertGreater(n, 0)
        self.assertEqual(np.abs(got[:n].astype(int) - ref[:n].astype(int)).max(), 0)

    def test_clips_hold_different_pixels_for_different_boxes(self):
        """Guards the split/map pairing -- a mis-wired graph would write the same
        crop to every output, which the dimension checks alone would not catch."""
        man = pt.build_pretranscode(self.src, REPS, self.out, quality="lossless")
        a, b = man.clip_for(1), man.clip_for(2)
        ga = _gray(os.path.join(self.out, a.filename), a.width, a.height)
        gb = _gray(os.path.join(self.out, b.filename), b.width, b.height)
        self.assertFalse(np.array_equal(ga, gb))

    def test_refuses_to_clobber_existing_clips_unless_asked(self):
        pt.build_pretranscode(self.src, REPS, self.out)
        with self.assertRaises(pt.PretranscodeError):
            pt.build_pretranscode(self.src, REPS, self.out)
        pt.build_pretranscode(self.src, REPS, self.out, overwrite=True)

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
        os.remove(os.path.join(self.out, man.clips[0].filename))
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
