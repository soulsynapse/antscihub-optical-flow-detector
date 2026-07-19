"""Batch J: the headless detection path (``core/batch.py`` + ``cli/detect.py``).

Two things here are load-bearing beyond "the code runs".

**The headless pass must agree with the GUI commit pass block-for-block.** The
whole tuned-preview/commit design rests on a detection found one way being
reproducible the other way; if they drift, the tool is untrustworthy at exactly
the moment it matters. ``test_matches_gui_worker_path`` runs the same two calls
the GUI worker runs and asserts array equality against ``run_video``.

**A short decode must fail the job.** Frames past a cut point are unexamined,
not examined-and-clear, and a fan-out of hundreds of jobs has nobody reading the
logs of the ones that "succeeded". The truncation tests pin that it raises by
default and only warns when explicitly allowed.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

import numpy as np

from core.batch import (BatchError, BatchResult, load_params, load_replicates,
                        normalize_params, params_to_json, run_video,
                        write_result)
from core.channel_source import live_channel_source
from core.config import PipelineConfig
from core.detection import detect_channel_region

HAS_FFMPEG = shutil.which("ffmpeg") is not None
requires_ffmpeg = unittest.skipUnless(HAS_FFMPEG, "ffmpeg not on PATH")

REPS = [
    {"id": 1, "label": "L", "frac": [0.0, 0.0, 0.37, 0.61]},
    {"id": 2, "label": "R", "frac": [0.41, 0.23, 0.93, 0.88]},
]

PARAMS = {
    "channel_attr": "change",
    "freq_band_hz": [0.5, 6.0],
    # Deliberately permissive so the fixture produces a non-empty gate: the
    # tests are about the plumbing and the agreement, not about whether testsrc
    # contains a behaviour.
    "value_band": [0.0, None],
    "count_band": [0, None],
    "detect_window": 5,
    "centered": True,
}


def _make_video(path: str, w: int = 160, h: int = 120, n: int = 32) -> None:
    rate = 24000 / 1001
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc=size={w}x{h}:rate=24000/1001:"
               f"duration={(n + 1) / rate:.4f}",
         "-frames:v", str(n), "-c:v", "libx264", "-crf", "18",
         "-g", "1", "-pix_fmt", "yuv420p", path],
        check=True, capture_output=True)


class ParamsTest(unittest.TestCase):
    """Band endpoints round-trip through JSON as ``null``, never ``Infinity``."""

    def test_null_endpoint_becomes_infinite(self):
        p = normalize_params(PARAMS)
        self.assertEqual(p["value_band"][0], 0.0)
        self.assertEqual(p["value_band"][1], math.inf)
        self.assertEqual(p["count_band"][1], math.inf)

    def test_missing_band_is_unbounded(self):
        p = normalize_params({"channel_attr": "change", "detect_window": 3})
        self.assertEqual(p["value_band"], (-math.inf, math.inf))

    def test_round_trip_is_standard_json(self):
        p = normalize_params(PARAMS)
        blob = json.dumps(params_to_json(p))       # would raise on a tuple/inf
        self.assertNotIn("Infinity", blob)
        self.assertEqual(normalize_params(json.loads(blob)), p)

    def test_rejects_unknown_channel(self):
        with self.assertRaises(BatchError):
            normalize_params({**PARAMS, "channel_attr": "speed"})

    def test_rejects_inverted_band(self):
        with self.assertRaises(BatchError):
            normalize_params({**PARAMS, "freq_band_hz": [6.0, 0.5]})

    def test_rejects_bad_window(self):
        with self.assertRaises(BatchError):
            normalize_params({**PARAMS, "detect_window": 0})

    def test_scalar_band_is_a_batch_error_not_a_typeerror(self):
        """A hand-edited params file is the expected input here, so a typo must
        surface as an operator message rather than a len() traceback."""
        with self.assertRaises(BatchError):
            normalize_params({**PARAMS, "value_band": 5})

    def test_unbounded_freq_lower_edge_is_allowed(self):
        """-inf is 'no lower bound', which band_indices handles; only a finite
        negative frequency is meaningless."""
        p = normalize_params({**PARAMS, "freq_band_hz": [None, 4.0]})
        self.assertEqual(p["freq_band_hz"][0], -math.inf)
        with self.assertRaises(BatchError):
            normalize_params({**PARAMS, "freq_band_hz": [-2.0, 4.0]})


class ReplicateLoadTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="batchreps_")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def _write(self, payload) -> str:
        p = os.path.join(self.dir, "r.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return p

    def test_accepts_both_sidecar_shapes(self):
        self.assertEqual(len(load_replicates(self._write(REPS))), 2)
        self.assertEqual(len(load_replicates(self._write({"replicates": REPS}))), 2)

    def test_overlapping_boxes_are_rejected(self):
        bad = [{"id": 1, "frac": [0.0, 0.0, 0.6, 0.6]},
               {"id": 2, "frac": [0.4, 0.4, 1.0, 1.0]}]
        with self.assertRaises(ValueError):
            load_replicates(self._write(bad))


class NoQtImportTest(unittest.TestCase):
    """The CLI must import on a node with no display. A transitive PyQt import
    would turn every headless job into a crash at startup, and the import chain
    is long enough (core -> gui helpers) that this is easy to break by accident."""

    def test_cli_imports_without_pyqt(self):
        code = ("import sys; import cli.detect; "
                "assert not [m for m in sys.modules if m.startswith('PyQt')], "
                "[m for m in sys.modules if m.startswith('PyQt')]")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, cwd=os.path.dirname(os.path.dirname(
                               os.path.abspath(__file__))))
        self.assertEqual(r.returncode, 0, r.stderr)


@requires_ffmpeg
class BatchRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.mkdtemp(prefix="batchrun_")
        cls.video = os.path.join(cls._dir, "src.mp4")
        _make_video(cls.video)
        cls.cfg = PipelineConfig()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._dir, ignore_errors=True)

    def _run(self, **kw) -> BatchResult:
        return run_video(self.video, REPS, self.cfg, PARAMS, **kw)

    def test_runs_every_replicate_by_default(self):
        res = self._run()
        self.assertEqual([r.replicate_id for r in res.regions], [1, 2])
        self.assertFalse(res.truncated)
        self.assertEqual(res.covered_frames, res.requested_frames)

    def test_region_selection_is_by_id_not_position(self):
        res = self._run(regions=[2])
        self.assertEqual([r.replicate_id for r in res.regions], [2])
        # ...and it resolves to the tile index, not to position 0 in the request.
        self.assertEqual(res.regions[0].region_index, 1)
        both = self._run()
        np.testing.assert_array_equal(res.regions[0].result.gate,
                                      both.regions[1].result.gate)

    def test_unknown_replicate_id_raises(self):
        with self.assertRaises(BatchError) as cm:
            self._run(regions=[99])
        self.assertIn("99", str(cm.exception))

    def test_matches_gui_worker_path(self):
        """The headless result equals what the GUI commit worker computes.

        This is the invariant the whole preview/commit split rests on, so it is
        asserted on the arrays rather than on interval counts -- a summary could
        agree while the series underneath had drifted.
        """
        res = self._run(regions=[1])
        # Exactly what _ProcessWorker._run does, inlined.
        cd = live_channel_source(self.video, self.cfg, REPS, start=0, n=None,
                                 channels=["change"])
        p = normalize_params(PARAMS)
        expect = detect_channel_region(
            cd, 0, "change", freq_band_hz=p["freq_band_hz"],
            value_band=p["value_band"], count_band=p["count_band"],
            detect_window=p["detect_window"], centered=p["centered"])
        got = res.regions[0].result
        for name in ("band_power", "count", "windowed", "gate", "clump"):
            np.testing.assert_array_equal(
                getattr(got, name), getattr(expect, name),
                err_msg=f"{name} drifted from the GUI commit path")

    def test_extraction_is_paid_once_for_many_regions(self):
        """Two regions must not cost two decodes. Timed rather than mocked: the
        claim is about wall clock, and a mock of the decoder would assert the
        call count while proving nothing about the cost."""
        one = self._run(regions=[1])
        both = self._run()
        self.assertEqual(len(both.regions), 2)
        # Extraction dominates; a second decode would roughly double it. Loose
        # bound because this is a 160x120 fixture where both terms are small.
        self.assertLess(both.extract_seconds, one.extract_seconds * 1.8)

    def test_start_offset_without_frame_count_is_not_truncation(self):
        """A --start run with no --frames must not trip the truncation guard.

        Regression: `requested` was the whole clip regardless of the offset, so
        a window starting at N could never reach it and every offset job aborted
        as truncated -- after paying for the entire extraction.
        """
        res = self._run(start=10)
        self.assertFalse(res.truncated)
        self.assertEqual(res.requested_frames, 22)     # 32-frame fixture, from 10
        self.assertEqual(res.covered_frames, 22)
        self.assertEqual(res.coverage, 1.0)

    def test_start_past_the_end_is_rejected(self):
        """Not an empty result: a zero-length pass is indistinguishable from
        'examined everything, found nothing'."""
        with self.assertRaises(BatchError) as cm:
            self._run(start=10_000)
        self.assertIn("10000", str(cm.exception))

    def test_duplicate_ids_are_collapsed(self):
        """The summary and the .npz are both keyed by replicate id, so a
        repeated id must not list a region twice while storing it once."""
        res = self._run(regions=[1, 1, 2])
        self.assertEqual([r.replicate_id for r in res.regions], [1, 2])

    def test_overlong_request_is_not_truncation(self):
        """Asking for more frames than the file holds is the end of the video,
        not a short decode. Counting the overshoot as missing coverage would
        fail every job that passed a generous --frames."""
        res = self._run(n=10_000)
        self.assertFalse(res.truncated)
        self.assertEqual(res.covered_frames, 32)
        self.assertEqual(res.coverage, 1.0)

    # The truncation POLICY is tested at this seam rather than by corrupting a
    # file. Whether the extractor correctly notices a short decode is already
    # covered against real ffmpeg output in tests/test_clip_extraction.py (the
    # vstack-freeze and zero-pad cases); what is under test here is what the
    # batch driver DOES about it, which is a branch on cd.meta['truncated'].
    def _short_source(self, keep: int = 12):
        real = live_channel_source

        def fake(*a, **kw):
            cd = real(*a, **kw)
            cd.channels = {k: v[:keep] for k, v in cd.channels.items()}
            cd.meta = {**cd.meta, "n_frames": keep, "truncated": True}
            return cd
        return unittest.mock.patch("core.batch.live_channel_source", fake)

    def test_truncated_pass_fails_by_default(self):
        with self._short_source():
            with self.assertRaises(BatchError) as cm:
                self._run()
        self.assertIn("UNEXAMINED", str(cm.exception))

    def test_truncated_pass_is_recorded_when_allowed(self):
        with self._short_source():
            res = self._run(allow_truncated=True)
        self.assertTrue(res.truncated)
        self.assertLess(res.coverage, 1.0)
        self.assertEqual(res.covered_frames, 12)
        self.assertTrue(any("UNEXAMINED" in w for w in res.warnings))
        self.assertTrue(res.to_summary()["coverage"]["truncated"])

    def test_summary_is_json_serializable_with_provenance(self):
        res = self._run()
        blob = json.dumps(res.to_summary())        # numpy ints would raise here
        d = json.loads(blob)
        self.assertEqual(d["provenance"]["channels_computed"], ["change"])
        self.assertIsNone(d["provenance"]["clip_provenance"])   # source, not clips
        self.assertEqual(d["params"]["channel_attr"], "change")
        self.assertIn("replicate_geometry_hash", d["provenance"])

    def test_write_result_emits_summary_and_series(self):
        res = self._run()
        out = os.path.join(self._dir, "out", "r.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        write_result(res, out)
        with open(out, encoding="utf-8") as f:
            d = json.load(f)
        npz = os.path.join(os.path.dirname(out), d["series_path"])
        with np.load(npz) as z:
            self.assertIn("r1_gate", z)
            self.assertIn("r2_windowed", z)
            self.assertNotIn("r1_band_power", z)     # opt-in, it is the big one
            np.testing.assert_array_equal(z["r1_gate"], res.regions[0].result.gate)
        self.assertFalse(d["series_includes_band_power"])

    def test_band_power_is_opt_in(self):
        res = self._run(regions=[1])
        out = os.path.join(self._dir, "bp.json")
        write_result(res, out, save_band_power=True)
        with np.load(os.path.splitext(out)[0] + ".npz") as z:
            self.assertEqual(z["r1_band_power"].shape,
                             res.regions[0].result.band_power.shape)


@requires_ffmpeg
class CliTest(unittest.TestCase):
    """Drive the argparse entry point as a subprocess -- exit codes are part of
    its contract, since a job runner reads them."""

    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.mkdtemp(prefix="batchcli_")
        cls.video = os.path.join(cls._dir, "src.mp4")
        _make_video(cls.video)
        # The sidecar convention: <stem>.rois.json next to the video.
        with open(os.path.splitext(cls.video)[0] + ".rois.json", "w") as f:
            json.dump({"replicates": REPS}, f)
        cls.params = os.path.join(cls._dir, "params.json")
        with open(cls.params, "w") as f:
            json.dump(PARAMS, f)
        cls.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._dir, ignore_errors=True)

    def _cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "cli.detect", *args],
            capture_output=True, text=True, cwd=self.root)

    def test_end_to_end_finds_the_sidecar(self):
        out = os.path.join(self._dir, "res.json")
        r = self._cli(self.video, "--params", self.params, "--out", out,
                      "--quiet")
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out, encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual([g["replicate_id"] for g in d["regions"]], [1, 2])
        self.assertTrue(d["replicate_path"].endswith(".rois.json"))

    def test_replicates_only_narrows(self):
        out = os.path.join(self._dir, "one.json")
        r = self._cli(self.video, "--params", self.params, "--out", out,
                      "--replicates-only", "2", "--quiet", "--no-series")
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out, encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual([g["replicate_id"] for g in d["regions"]], [2])
        self.assertNotIn("series_path", d)

    def test_missing_video_exits_1(self):
        r = self._cli(os.path.join(self._dir, "nope.mp4"), "--params",
                      self.params, "--quiet")
        self.assertEqual(r.returncode, 1)
        self.assertIn("no such video", r.stderr)

    def test_bad_replicate_id_exits_1(self):
        r = self._cli(self.video, "--params", self.params, "--quiet",
                      "--replicates-only", "77")
        self.assertEqual(r.returncode, 1)
        self.assertIn("77", r.stderr)

    def test_bad_params_exits_1(self):
        bad = os.path.join(self._dir, "bad.json")
        with open(bad, "w") as f:
            json.dump({**PARAMS, "channel_attr": "nope"}, f)
        r = self._cli(self.video, "--params", bad, "--quiet")
        self.assertEqual(r.returncode, 1)
        self.assertIn("nope", r.stderr)

    def test_start_offset_run_succeeds(self):
        """The CLI half of the truncation-guard regression."""
        out = os.path.join(self._dir, "off.json")
        r = self._cli(self.video, "--params", self.params, "--start", "10",
                      "--quiet", "--no-series", "--out", out)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out, encoding="utf-8") as f:
            self.assertFalse(json.load(f)["coverage"]["truncated"])

    def test_unreadable_manifest_exits_1_without_a_traceback(self):
        """PretranscodeError is a bare RuntimeError; the CLI must still turn it
        into an operator message, since manifest distrust is the likeliest way a
        clip-backed job legitimately fails."""
        bad = os.path.join(self._dir, "bad.pretranscode.json")
        with open(bad, "w") as f:
            json.dump({"version": 0}, f)         # a version the loader refuses
        r = self._cli(self.video, "--params", self.params, "--manifest", bad,
                      "--quiet")
        self.assertEqual(r.returncode, 1)
        self.assertNotIn("Traceback", r.stderr)
        self.assertTrue(r.stderr.startswith("error:"), r.stderr)

    def test_usage_error_exits_2(self):
        r = self._cli(self.video)                # --params is required
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
