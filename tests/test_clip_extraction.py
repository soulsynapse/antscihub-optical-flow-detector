"""Extraction from pre-transcoded ROI clips.

**Clips are never bit-exact against the live crop, not even at ``lossless``**,
and these tests exist to pin down by how much rather than to pretend otherwise.
The live path converts the source's yuv luma straight to gray16le and keeps the
sub-8-bit precision that the limited->full range conversion produces; a clip
stores 8-bit gray, so it rounds by up to 0.494 grey levels before anything else
happens. ``lossless`` names the *encode*, not the route.

What is asserted instead is the structure the whole design rests on:

  * agreement degrades monotonically with quality (lossless < high < standard),
    so the knob means what it claims;
  * the differencing channels are one to two orders of magnitude more sensitive
    than ``intensity``, which is the measured form of the module's argument that
    lossy inter-frame coding perturbs exactly what ``change`` measures;
  * geometry and provenance are enforced, not advisory.

These build real clips with ffmpeg. A mocked decoder would assert that we can
format a filter graph, which is not where the traps are.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from core import pretranscode as pt
from core.channel_source import live_channel_source, resolve_clip_paths
from core.config import PipelineConfig
from core.tensor_channels import _tiles_from_meta, extract_channels_live


HAS_FFMPEG = shutil.which("ffmpeg") is not None
requires_ffmpeg = unittest.skipUnless(HAS_FFMPEG, "ffmpeg not on PATH")

# Odd frac boundaries deliberately: they land the crop on odd source pixels,
# which is where crop=...:exact=1 matters. An even-aligned box would pass even if
# that flag were dropped from one of the two graphs.
REPS = [
    {"id": 1, "label": "L", "frac": [0.0, 0.0, 0.37, 0.61]},
    {"id": 2, "label": "R", "frac": [0.41, 0.23, 0.93, 0.88]},
]


def _make_video(path: str, w: int = 160, h: int = 120, n: int = 32) -> None:
    """Deterministic footage with real motion, at an NTSC rate.

    ``24000/1001`` rather than a round rate so any code that rounds the frame
    rate on the way to a seek shows up as a window from the wrong place.
    """
    rate = 24000 / 1001
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc=size={w}x{h}:rate=24000/1001:"
               f"duration={(n + 1) / rate:.4f}",
         "-frames:v", str(n), "-c:v", "libx264", "-crf", "18",
         # -g 1 so the source itself seeks exactly, keeping this test about the
         # clip path rather than about the fixture's GOP structure.
         "-g", "1", "-pix_fmt", "yuv420p", path],
        check=True, capture_output=True)


@requires_ffmpeg
class ClipExtractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.mkdtemp(prefix="clipext_")
        cls.video = os.path.join(cls._dir, "src.mp4")
        _make_video(cls.video)
        cls._by_quality: dict = {}
        cls.man, cls.clip_dir = cls._clips("lossless")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._dir, ignore_errors=True)

    @classmethod
    def _clips(cls, quality: str):
        """``(manifest, dir)`` for ``quality``, cut once and reused.

        Memoized because a transcode per assertion would dominate the suite's
        runtime, and because re-cutting into a populated directory is refused
        outright -- correctly, since a half-replaced clip set is exactly what
        that guard exists to prevent.
        """
        hit = cls._by_quality.get(quality)
        if hit is None:
            d = os.path.join(cls._dir, f"clips_{quality}")
            hit = (pt.build_pretranscode(cls.video, REPS, d, quality=quality), d)
            cls._by_quality[quality] = hit
        return hit

    def _agreement(self, quality: str, start: int = 0, n: int = 12) -> dict:
        """Per-channel RMS difference clip-vs-live, normalized by channel scale.

        Normalized because the channels carry wildly different units --
        ``tensor_speed`` is px/s and ``change`` is squared grey levels -- so a
        raw RMS would compare nothing meaningful across them.

        The signed flow components ``u``/``v`` are normalized by the flow
        MAGNITUDE scale rather than their own max. A component that is ~0
        everywhere (``v`` on a horizontally-moving fixture) has a degenerate
        self-scale, and dividing a tiny decode difference by it manufactures a
        large relative error that reflects the fixture's geometry, not a
        divergence of the two decode routes. The physical question -- how big is
        the disagreement next to the motion being measured -- uses ``tensor_speed``.
        """
        cfg = PipelineConfig()
        live = live_channel_source(self.video, cfg, REPS, start=start, n=n)
        man, d = self._clips(quality)
        clips = live_channel_source(self.video, cfg, REPS, start=start, n=n,
                                    manifest=man, clip_dir=d)
        self.assertEqual(live.n_frames, clips.n_frames)
        self.assertEqual(set(live.channels), set(clips.channels))
        speed_scale = max(float(np.abs(live.channels["tensor_speed"]).max()),
                          1e-12)
        out = {}
        for name, a in live.channels.items():
            b = clips.channels[name]
            scale = (speed_scale if name in ("u", "v")
                     else max(float(np.abs(a).max()), 1e-12))
            out[name] = float(np.sqrt(np.mean((b - a) ** 2)) / scale)
        return out

    def test_lossless_clips_track_the_live_crop_closely(self):
        """Close, but deliberately not asserted equal -- see the module docstring.

        The bound is loose on purpose. It is a regression guard against the
        routes diverging structurally (a dropped ``exact=1``, a plane picked off
        the wrong stream, an off-by-one window), not a claimed error budget: the
        real budget belongs to footage, and this fixture is 160x120 ``testsrc``,
        about the least forgiving input there is for an 8-bit round trip.
        """
        agree = self._agreement("lossless")
        for name, rms in agree.items():
            self.assertLess(rms, 0.05, f"{name} drifted: rms/scale={rms:.4g}")

    def test_agreement_degrades_monotonically_with_quality(self):
        """The knob means what it claims, on the channels detection uses."""
        lo = self._agreement("lossless")
        hi = self._agreement("high")
        st = self._agreement("standard")
        for name in ("change", "appearance"):
            self.assertLess(lo[name], hi[name], f"{name}: lossless !< high")
            self.assertLess(hi[name], st[name], f"{name}: high !< standard")

    def test_differencing_channels_are_the_sensitive_ones(self):
        """``intensity`` is a block mean and averages the noise away; ``change``
        is squared frame differencing and is precisely what lossy inter-frame
        coding perturbs. Measured at ``standard``: ~0.044 vs ~0.0003 rms/scale.

        This is the empirical form of the argument for keeping ``quality`` in
        ``provenance_key``. If it ever inverts, that argument is wrong and the
        default should be revisited rather than the test relaxed.
        """
        agree = self._agreement("standard")
        self.assertGreater(agree["change"], 10 * agree["intensity"])

    def test_mid_clip_window_lands_in_the_same_place(self):
        """A rate rounded anywhere along the clip path yields a window of the
        right *length* from the wrong *place*, which a frame-zero test cannot
        catch. Agreement at a mid-clip start is what rules that out -- a
        misaligned window would disagree at signal scale, far above this bound.
        """
        agree = self._agreement("lossless", start=14, n=10)
        for name, rms in agree.items():
            self.assertLess(rms, 0.05, f"{name} drifted: rms/scale={rms:.4g}")

    def test_window_offset_is_reported(self):
        clips = live_channel_source(self.video, PipelineConfig(), REPS,
                                    start=14, n=10, manifest=self.man,
                                    clip_dir=self.clip_dir)
        self.assertEqual(clips.window_start, 14)
        self.assertEqual(clips.n_frames, 10)

    def test_provenance_lands_in_meta(self):
        cfg = PipelineConfig()
        cd = live_channel_source(self.video, cfg, REPS, start=0, n=4,
                                 manifest=self.man, clip_dir=self.clip_dir)
        self.assertEqual(cd.meta["clip_provenance"], self.man.provenance_key())
        self.assertEqual(cd.meta["clip_quality"], "lossless")
        # Absent, not None, on the source path -- the cache key distinguishes the
        # two by presence, so a stray key would silently rekey every old cache.
        live = live_channel_source(self.video, cfg, REPS, start=0, n=4)
        self.assertNotIn("clip_provenance", live.meta)

    def test_manifest_carries_the_exact_rate(self):
        """meta['fps'] comes from the manifest's rational, not a rounded float."""
        cfg = PipelineConfig()
        cd = live_channel_source(self.video, cfg, REPS, start=0, n=4,
                                 manifest=self.man, clip_dir=self.clip_dir)
        self.assertEqual((self.man.fps_num, self.man.fps_den), (24000, 1001))
        self.assertAlmostEqual(cd.meta["fps"], 24000 / 1001, places=12)

    def test_moved_geometry_is_refused(self):
        """A moved box must not read a stale clip -- that would attribute one
        region's pixels to another's detections."""
        moved = [dict(REPS[0]), {**REPS[1], "frac": [0.5, 0.3, 0.95, 0.9]}]
        with self.assertRaises(pt.PretranscodeError):
            live_channel_source(self.video, PipelineConfig(), moved, start=0,
                                n=4, manifest=self.man, clip_dir=self.clip_dir)

    def test_missing_clip_is_refused(self):
        d = os.path.join(self._dir, "partial")
        os.makedirs(d, exist_ok=True)
        shutil.copy(os.path.join(self.clip_dir, self.man.clips[0].filename), d)
        with self.assertRaises(pt.PretranscodeError):
            resolve_clip_paths(self.man, d, REPS,
                               _tiles_from_meta(self._meta()))

    def _truncated_clips(self, name: str, frames: int = 12) -> str:
        """A copy of the clip set with replicate 0's clip cut short."""
        d = os.path.join(self._dir, name)
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(self.clip_dir, d)
        victim = self.man.clips[0].filename
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", os.path.join(self.clip_dir, victim),
             "-frames:v", str(frames), "-c", "copy", os.path.join(d, victim)],
            check=True, capture_output=True)
        return d

    def test_truncated_clip_is_refused_before_any_decode(self):
        """First line of defence: the recorded size no longer matches.

        A clip cut short by a crash or a full disk used to pass verification
        outright, because verification checked that a clip existed and not how
        long it was -- so the problem was rediscovered at decode time, on every
        pass, forever. Recording ``size_bytes`` at the cut turns it into one
        cheap ``os.stat`` per clip and a hard error up front.
        """
        d = self._truncated_clips("truncated_verify")
        with self.assertRaises(pt.PretranscodeError):
            live_channel_source(self.video, PipelineConfig(), REPS, start=0,
                                n=28, manifest=self.man, clip_dir=d)

    def test_short_decode_yields_a_short_window_not_zeros(self):
        """Second line of defence, reached with verification deliberately bypassed.

        This still matters even though the size check now catches the ordinary
        case: it is what protects a clip set cut before ``size_bytes`` existed,
        and the ROI and full-frame decode paths, which have no manifest to check
        at all. The failure it prevents is a silent per-replicate false negative.
        ffmpeg's vstack, at its default ``shortest=0``, *repeats* the exhausted
        input's last frame rather than ending the stream -- so one replicate
        freezes (identical consecutive frames, change == 0, "nothing moved")
        while every other replicate looks perfectly healthy, and nothing anywhere
        reports it. With ``shortest=1`` the stream ends and the window is trimmed
        and flagged.

        Called through ``extract_channels_live`` with raw paths, which is the
        documented way past the manifest checks.
        """
        d = self._truncated_clips("truncated_decode")
        paths = [os.path.join(d, c.filename) for c in self.man.clips]
        meta = self._meta()
        res = extract_channels_live(self.video, meta, start=0, n=28,
                                    clip_paths=paths)
        n = int(res["meta"]["n_frames"])
        self.assertLess(n, 28)
        self.assertTrue(res["meta"]["truncated"])
        for name in ("change", "appearance", "intensity", "tensor_speed"):
            self.assertEqual(res[name].shape[0], n,
                             f"{name} was not trimmed with the window")
        # And no trailing all-zero block survived the trim.
        self.assertFalse(np.all(res["change"][-1] == 0),
                         "window still ends in zeros")

    def test_clip_frame_count_and_size_are_recorded(self):
        """Both are what make a short clip detectable before a pass reads it."""
        for c in self.man.clips:
            self.assertEqual(c.frame_count, self.man.frame_count)
            self.assertEqual(
                c.size_bytes,
                os.path.getsize(os.path.join(self.clip_dir, c.filename)))

    def test_clips_are_frame_aligned_with_the_source(self):
        """``-fps_mode passthrough`` at the cut, checked end to end.

        Without it ffmpeg's default CFR conversion may duplicate or drop frames
        to force a constant rate, and clip frame N stops being source frame N --
        after which every seek lands in the wrong place, drifting further the
        deeper into the clip you look. The build refuses such a cut outright, so
        reaching a manifest at all is the assertion; the counts confirm it.
        """
        for c in self.man.clips:
            self.assertEqual(c.frame_count, self.man.frame_count)

    def test_complete_pass_is_not_flagged_truncated(self):
        cd = live_channel_source(self.video, PipelineConfig(), REPS, start=0,
                                 n=12, manifest=self.man,
                                 clip_dir=self.clip_dir)
        self.assertEqual(cd.n_frames, 12)
        self.assertFalse(cd.meta.get("truncated"))

    def test_clip_size_mismatch_is_caught_from_the_manifest(self):
        """Caught for free from the recorded size, so the decoder need not probe.

        A wrongly-sized clip is the one mismatch the filter graph would absorb
        silently, by rescaling it to the tile's work size and returning
        plausible pixels from a different region.
        """
        import dataclasses
        bad = dataclasses.replace(self.man.clips[0], width=7, height=9)
        man = dataclasses.replace(self.man, clips=(bad,) + self.man.clips[1:])
        with self.assertRaises(ValueError):
            resolve_clip_paths(man, self.clip_dir, REPS,
                               _tiles_from_meta(self._meta()))

    def test_manifest_without_clip_dir_is_refused(self):
        with self.assertRaises(ValueError):
            live_channel_source(self.video, PipelineConfig(), REPS, start=0,
                                n=4, manifest=self.man)

    def _meta(self) -> dict:
        from core.channel_source import synth_live_meta
        return synth_live_meta(self.video, PipelineConfig(), REPS,
                               width=self.man.src_width,
                               height=self.man.src_height,
                               fps=self.man.fps,
                               frame_count=self.man.frame_count)


class ProvenanceKeyTest(unittest.TestCase):
    def test_quality_alone_changes_the_provenance_key(self):
        """Two cuts differing only in quality are different measurements: the
        change channel is squared frame differencing, which is precisely what
        lossy inter-frame coding perturbs."""
        common = dict(
            version=pt.PRETRANSCODE_VERSION, source_path="x", source_size=1,
            source_sha256="s", quick_sig="q", src_width=10, src_height=10,
            fps_num=24, fps_den=1, frame_count=5, geometry_hash="g",
            resolved_scale=1.0, scale_rule="r")
        a = pt.Manifest(**common, codec="libx264-crf12", lossless=False,
                        quality="high", quality_rms=1.041)
        b = pt.Manifest(**common, codec="libx264-crf18", lossless=False,
                        quality="standard", quality_rms=1.540)
        self.assertNotEqual(a.provenance_key(), b.provenance_key())


if __name__ == "__main__":
    unittest.main()
