"""Batch J, second slice: file-level partitioning (``core/shard.py`` + ``cli/run.py``).

Four properties here are the ones worth pinning, because each has a failure mode
that looks like success.

**The partition must cover the list exactly once.** A stride that dropped or
duplicated a video would produce a fan-out where some footage is silently never
examined -- and every shard still exits 0. ``test_partition_is_a_cover`` asserts
the union over all shards is the input, for a range of list/shard sizes
including more shards than videos.

**Resume must not return stale results.** Skipping on output existence alone
means a retuned band gets yesterday's detections back, with nothing saying so.
The skip tests pin that a changed param, a changed scale, and a changed block
size each defeat the skip.

**A failed video must not take its shard-mates with it, and must still fail the
shard.** Both halves matter: continuing salvages paid decode, and the non-zero
exit is what stops a job runner from recording the shard as clean.

Most tests mock ``run_video`` -- the extraction path is covered in
``test_batch_cli.py`` and re-decoding here would buy nothing. One end-to-end test
runs the real thing over two ffmpeg fixtures.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

from dataclasses import replace

from core.batch import BatchError, normalize_params, params_to_json
from core.config import PipelineConfig
from core.shard import (assign_output_names, expand_videos, find_manifest,
                        is_current, output_path_for, parse_shard, partition,
                        run_shard, skip_reason)

HAS_FFMPEG = shutil.which("ffmpeg") is not None
requires_ffmpeg = unittest.skipUnless(HAS_FFMPEG, "ffmpeg not on PATH")

REPS = [
    {"id": 1, "label": "L", "frac": [0.0, 0.0, 0.37, 0.61]},
    {"id": 2, "label": "R", "frac": [0.41, 0.23, 0.93, 0.88]},
]

PARAMS = {
    "channel_attr": "change",
    "freq_band_hz": [0.5, 6.0],
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


class ShardSpecTest(unittest.TestCase):
    def test_parses(self):
        self.assertEqual(parse_shard("0/1"), (0, 1))
        self.assertEqual(parse_shard("3/8"), (3, 8))

    def test_rejects_one_based_overrun(self):
        """The specific operator error this guard exists for: launching shards
        1..N under a 0-based scheme, which never runs shard 0 while every job
        reports success."""
        with self.assertRaises(BatchError) as cm:
            parse_shard("8/8")
        self.assertIn("0-based", str(cm.exception))

    def test_rejects_malformed(self):
        for bad in ("", "3", "3/", "a/8", "3/8/2", "-1/8", "3/0"):
            with self.assertRaises(BatchError):
                parse_shard(bad)


class PartitionTest(unittest.TestCase):
    def test_partition_is_a_cover(self):
        """Every video appears in exactly one shard, for any list/shard size."""
        for n_videos in (0, 1, 2, 7, 40):
            paths = [f"v{i:03d}.mp4" for i in range(n_videos)]
            for n_shards in (1, 2, 3, 8, 64):
                got = []
                for i in range(n_shards):
                    got.extend(partition(paths, i, n_shards))
                self.assertCountEqual(
                    got, paths,
                    f"{n_videos} videos over {n_shards} shards is not a cover")

    def test_more_shards_than_videos_is_empty_not_an_error(self):
        """A fan-out sized for the largest session must not crash on a small
        one; the surplus shards simply have nothing to do."""
        self.assertEqual(partition(["a.mp4"], 3, 8), [])

    def test_is_deterministic(self):
        paths = [f"v{i}.mp4" for i in range(20)]
        self.assertEqual(partition(paths, 2, 5), partition(paths, 2, 5))


class ExpandVideosTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="shardexp_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        for name in ("b.mp4", "a.mp4", "c.MP4", "notes.txt"):
            open(os.path.join(self.dir, name), "w").close()

    def test_glob_is_expanded_here(self):
        """Not left to the shell: PowerShell does not expand globs for native
        commands, so the pattern would arrive as a literal filename."""
        got = expand_videos([os.path.join(self.dir, "*.mp4")])
        self.assertEqual([os.path.basename(p) for p in got],
                         ["a.mp4", "b.mp4", "c.MP4"])
        self.assertNotIn("notes.txt", [os.path.basename(p) for p in got])

    def test_glob_case_is_platform_independent(self):
        """'*.mp4' must match 'c.MP4' everywhere, not just on Windows. A
        fan-out whose nodes disagree about the file list partitions different
        lists, leaving footage examined by nobody while every shard exits 0 --
        and GoPro's uppercase .MP4 is exactly the case that triggers it."""
        got = expand_videos([os.path.join(self.dir, "*.MP4")])
        self.assertEqual([os.path.basename(p) for p in got],
                         ["a.mp4", "b.mp4", "c.MP4"])

    def test_bracket_class_is_not_rewritten(self):
        """The case rewrite must leave an existing [...] class alone; nesting
        brackets would change what the class matches."""
        got = expand_videos([os.path.join(self.dir, "[ab].mp4")])
        self.assertEqual([os.path.basename(p) for p in got], ["a.mp4", "b.mp4"])

    def test_directory_is_scanned_for_video_suffixes(self):
        got = expand_videos([self.dir])
        self.assertEqual([os.path.basename(p) for p in got],
                         ["a.mp4", "b.mp4", "c.MP4"])

    def test_result_is_sorted_and_deduplicated(self):
        """The shard contract depends on every node deriving the same ordering
        from the same inputs, however the list was assembled."""
        a = os.path.join(self.dir, "a.mp4")
        got = expand_videos([os.path.join(self.dir, "b.mp4"), a, a])
        self.assertEqual([os.path.basename(p) for p in got], ["a.mp4", "b.mp4"])

    def test_separator_variants_are_one_video(self):
        """A Windows operator produces both 'd/a.mp4' and 'd\\a.mp4' without
        noticing; they must not become two jobs writing one output."""
        got = expand_videos([self.dir + "/a.mp4", self.dir + os.sep + "a.mp4"])
        self.assertEqual(len(got), 1)

    def test_list_file_with_comments(self):
        lf = os.path.join(self.dir, "list.txt")
        with open(lf, "w", encoding="utf-8") as f:
            f.write(f"# session 1\n{self.dir}/a.mp4\n\n{self.dir}/b.mp4\n")
        got = expand_videos([], list_file=lf)
        self.assertEqual([os.path.basename(p) for p in got], ["a.mp4", "b.mp4"])

    def test_pattern_matching_nothing_raises(self):
        """Silence here would mean a shard that examines no footage and exits
        0 -- the whole-session-missing failure."""
        with self.assertRaises(BatchError):
            expand_videos([os.path.join(self.dir, "*.mkv")])

    def test_directory_with_no_videos_raises(self):
        """Same hazard as the empty glob, and it was silent: a wrong-level
        directory (videos actually one folder deeper) contributed nothing and
        the run still reported success."""
        empty = os.path.join(self.dir, "empty")
        os.makedirs(empty)
        with self.assertRaises(BatchError):
            expand_videos([empty])

    def test_list_file_relative_paths_resolve_against_the_list_file(self):
        """Not the process CWD: a job runner launches from wherever it likes,
        and a list meaning different videos per launch directory gives each
        node a different partition."""
        sub = os.path.join(self.dir, "jobs")
        os.makedirs(sub)
        lf = os.path.join(sub, "list.txt")
        with open(lf, "w", encoding="utf-8") as f:
            f.write("../a.mp4\n../b.mp4\n")
        got = expand_videos([], list_file=lf)
        self.assertEqual([os.path.basename(p) for p in got],
                         ["a.mp4", "b.mp4"])
        for p in got:
            self.assertTrue(os.path.exists(p), f"{p} did not resolve")


class OutputNameTest(unittest.TestCase):
    """Basename-keyed outputs collide, and the collision fails silently: the
    loser is either overwritten or -- worse -- skipped as 'up to date' against
    the winner's summary, so its footage is never decoded and the run exits 0.
    Cameras number per card, so this is the normal multi-session case."""

    def test_same_stem_in_different_dirs_gets_distinct_outputs(self):
        vids = [os.path.join("f", "s1", "GX010047.MP4"),
                os.path.join("f", "s2", "GX010047.MP4")]
        names = assign_output_names(vids, "out")
        self.assertEqual(len(set(names.values())), 2)
        self.assertEqual({os.path.basename(p) for p in names.values()},
                         {"s1_GX010047.detections.json",
                          "s2_GX010047.detections.json"})

    def test_unique_stems_keep_the_plain_name(self):
        """The common case must keep the name cli/detect.py writes."""
        names = assign_output_names([os.path.join("f", "a.mp4"),
                                     os.path.join("f", "b.mp4")], "out")
        self.assertEqual({os.path.basename(p) for p in names.values()},
                         {"a.detections.json", "b.detections.json"})

    def test_deeper_collision_walks_further_up(self):
        vids = [os.path.join("f", "d1", "s", "v.mp4"),
                os.path.join("f", "d2", "s", "v.mp4")]
        names = assign_output_names(vids, "out")
        self.assertEqual(len(set(names.values())), 2)

    def test_names_do_not_depend_on_shard_membership(self):
        """Every node is handed the same list, so every node must derive the
        same mapping regardless of which slice it will run."""
        vids = sorted([os.path.join("f", "s1", "v.mp4"),
                       os.path.join("f", "s2", "v.mp4"),
                       os.path.join("f", "s3", "w.mp4")])
        full = assign_output_names(vids, "out")
        for i in range(3):
            self.assertEqual(assign_output_names(vids, "out"), full)
        self.assertEqual(len(set(full.values())), 3)

    def test_collision_is_caught_end_to_end(self):
        """The property that actually matters: two same-named videos in one run
        both get decoded, and neither is skipped against the other."""
        d = tempfile.mkdtemp(prefix="shardcol_")
        self.addCleanup(shutil.rmtree, d, True)
        vids = []
        for sub in ("s1", "s2"):
            os.makedirs(os.path.join(d, sub))
            p = os.path.join(d, sub, "GX010047.mp4")
            open(p, "w").close()
            with open(os.path.splitext(p)[0] + ".rois.json", "w",
                      encoding="utf-8") as f:
                json.dump(REPS, f)
            vids.append(p)
        calls = []

        def fake(video, *a, **k):
            calls.append(video)
            return _FakeResult()

        out_dir = os.path.join(d, "out")
        os.makedirs(out_dir)
        with unittest.mock.patch("core.shard.run_video", fake), \
             unittest.mock.patch("core.shard.write_result"):
            rep = run_shard(sorted(vids), PipelineConfig(),
                            normalize_params(PARAMS), out_dir=out_dir)
        self.assertEqual(len(calls), 2, "a colliding video was skipped")
        self.assertEqual(len(rep.ok), 2)
        self.assertEqual(len({o.out_path for o in rep.ok}), 2,
                         "two videos share one output path")


class OrderingTest(unittest.TestCase):
    """The partition is a stride over the sorted list, so the sort order is
    part of the shard contract and must not vary by platform."""

    def test_order_is_case_folded_not_platform_normcase(self):
        d = tempfile.mkdtemp(prefix="shardord_")
        self.addCleanup(shutil.rmtree, d, True)
        for n in ("B.mp4", "a.mp4", "C.mp4", "d.mp4"):
            open(os.path.join(d, n), "w").close()
        got = [os.path.basename(p) for p in expand_videos([d])]
        # os.path.normcase would give B,C,a,d on Linux and a,B,C,d on Windows.
        # Case-folded ordering gives the same answer on both.
        self.assertEqual(got, ["a.mp4", "B.mp4", "C.mp4", "d.mp4"])


class SkipTest(unittest.TestCase):
    """Resume is by output *identity*, not existence."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="shardskip_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.cfg = PipelineConfig()
        self.params = normalize_params(PARAMS)
        self.out = os.path.join(self.dir, "v.detections.json")
        self._write(self._summary())

    def _summary(self, **over):
        scale = self.cfg.preprocess.resolve_downsample()
        prov = {"downsample": scale,
                "block_size": self.cfg.flow.resolve_block_size(scale)}
        prov.update(over.pop("provenance", {}))
        cov = {"truncated": False, "start_frame": 0, "requested_n": None,
               "requested_frames": 500, "covered_frames": 500}
        cov.update(over.pop("coverage", {}))
        d = {"params": params_to_json(self.params), "provenance": prov,
             "coverage": cov}
        d.update(over)
        return d

    def _write(self, payload):
        with open(self.out, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def test_current_result_is_skipped(self):
        self.assertEqual(skip_reason(self.out, self.params, self.cfg),
                         "up to date")

    def test_missing_output_runs(self):
        os.remove(self.out)
        self.assertIsNone(skip_reason(self.out, self.params, self.cfg))

    def test_changed_params_defeat_the_skip(self):
        """The silent-wrong-result case: a retuned band must not return the old
        detections with nothing in the output saying so."""
        other = normalize_params({**PARAMS, "value_band": [0.5, None]})
        self.assertIsNone(skip_reason(self.out, other, self.cfg))

    def test_changed_scale_defeats_the_skip(self):
        cfg = replace(self.cfg,
                      preprocess=replace(self.cfg.preprocess, downsample=0.5))
        self.assertIsNone(skip_reason(self.out, self.params, cfg))

    def test_changed_block_size_defeats_the_skip(self):
        cfg = replace(self.cfg, flow=replace(self.cfg.flow, block_size=8))
        self.assertIsNone(skip_reason(self.out, self.params, cfg))

    def test_equivalent_config_still_skips(self):
        """downsample=None and downsample=1.0 resolve to the same scale, so
        neither may be called stale -- a false stale re-decodes hours of
        finished work for a difference that does not exist."""
        cfg = replace(self.cfg,
                      preprocess=replace(self.cfg.preprocess, downsample=1.0))
        self.assertEqual(skip_reason(self.out, self.params, cfg), "up to date")

    def test_truncated_result_is_rerun(self):
        """A resumed shard should converge on complete coverage, not inherit
        what the crashed run settled for."""
        self._write(self._summary(coverage={"truncated": True}))
        self.assertIsNone(skip_reason(self.out, self.params, self.cfg))

    def test_unreadable_summary_is_rerun_not_skipped(self):
        with open(self.out, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertIsNone(skip_reason(self.out, self.params, self.cfg))

    def test_missing_provenance_is_not_current(self):
        self.assertFalse(is_current({"params": params_to_json(self.params)},
                                    self.params, self.cfg))

    def test_short_smoke_test_is_not_reused_as_a_full_pass(self):
        """The case that motivated recording the raw request: a --frames 1000
        run and a full-length run agree on every resolved count once both
        complete, so only requested_n distinguishes them. Reusing the short one
        leaves everything past frame 1000 unexamined, not clear."""
        self._write(self._summary(coverage={"requested_n": 1000,
                                            "requested_frames": 1000,
                                            "covered_frames": 1000}))
        self.assertIsNone(skip_reason(self.out, self.params, self.cfg))
        # ...and is reused when the same short window is asked for again.
        self.assertEqual(
            skip_reason(self.out, self.params, self.cfg, frames=1000),
            "up to date")

    def test_full_pass_is_not_reused_for_a_narrower_request(self):
        self.assertIsNone(
            skip_reason(self.out, self.params, self.cfg, frames=100))

    def test_different_start_offset_is_not_reused(self):
        self.assertIsNone(
            skip_reason(self.out, self.params, self.cfg, start=500))

    def test_summary_without_a_recorded_window_is_rerun(self):
        """Results written before the window was recorded cannot be shown to
        answer this question, so they are recomputed rather than trusted."""
        s = self._summary()
        del s["coverage"]["requested_n"]
        self._write(s)
        self.assertIsNone(skip_reason(self.out, self.params, self.cfg))


class _FakeRegion:
    n_detected_frames = 3


class _FakeResult:
    regions = [_FakeRegion()]
    covered_frames = 32
    warnings: list = []


class RunShardTest(unittest.TestCase):
    """Failure isolation and the shard report, with the decode mocked out."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="shardrun_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.out_dir = os.path.join(self.dir, "out")
        os.makedirs(self.out_dir)
        self.videos = []
        for name in ("a.mp4", "b.mp4", "c.mp4"):
            p = os.path.join(self.dir, name)
            open(p, "w").close()
            with open(os.path.splitext(p)[0] + ".rois.json", "w",
                      encoding="utf-8") as f:
                json.dump(REPS, f)
            self.videos.append(p)
        self.params = normalize_params(PARAMS)

    def _run(self, run_video, **kw):
        with unittest.mock.patch("core.shard.run_video", run_video), \
             unittest.mock.patch("core.shard.write_result"):
            return run_shard(self.videos, PipelineConfig(), self.params,
                             out_dir=self.out_dir, **kw)

    def test_one_failure_does_not_stop_the_others(self):
        calls = []

        def fake(video, *a, **k):
            calls.append(video)
            if os.path.basename(video) == "b.mp4":
                raise BatchError("decode covered 9 of 32 requested frames")
            return _FakeResult()

        rep = self._run(fake)
        self.assertEqual(len(calls), 3, "the run stopped at the failure")
        self.assertEqual([len(rep.ok), len(rep.failed)], [2, 1])
        self.assertEqual([o.video_path for o in rep.failed], [self.videos[1]])

    def test_failed_videos_are_enumerated_in_the_report(self):
        """A failure must be visible as a name, not only as an absence among
        the successes."""
        def fake(video, *a, **k):
            raise BatchError("boom")
        rep = self._run(fake)
        self.assertEqual(rep.to_summary()["failed_videos"], self.videos)
        self.assertEqual(rep.to_summary()["counts"],
                         {"ok": 0, "skipped": 0, "failed": 3})

    def test_shard_slice_is_taken_from_the_whole_list(self):
        """Every shard is handed identical inputs and differs only in --shard;
        the slicing happens inside."""
        calls = []

        def fake(video, *a, **k):
            calls.append(os.path.basename(video))
            return _FakeResult()

        self._run(fake, shard=(1, 3))
        self.assertEqual(calls, ["b.mp4"])

    def test_missing_video_fails_only_that_video(self):
        self.videos.append(os.path.join(self.dir, "gone.mp4"))
        rep = self._run(lambda *a, **k: _FakeResult())
        self.assertEqual(len(rep.ok), 3)
        self.assertEqual(len(rep.failed), 1)

    def test_missing_sidecar_fails_only_that_video(self):
        os.remove(os.path.splitext(self.videos[1])[0] + ".rois.json")
        rep = self._run(lambda *a, **k: _FakeResult())
        self.assertEqual(len(rep.ok), 2)
        self.assertIn("replicate boxes", rep.failed[0].error)

    def test_missing_manifest_under_clip_dir_is_warned_not_silent(self):
        """Falling back to the source is ~25x the decode AND different pixels
        (FINDINGS.md section 10). An operator who named a clip dir must not
        discover that silently."""
        clips = os.path.join(self.dir, "clips")
        os.makedirs(clips)
        rep = self._run(lambda *a, **k: _FakeResult(), clip_dir=clips)
        self.assertEqual(len(rep.ok), 3)
        for o in rep.ok:
            self.assertFalse(o.used_clips)
            self.assertTrue(any("manifest" in w for w in o.warnings),
                            "source fallback was not recorded")
        self.assertIn("used_clips", rep.to_summary()["videos"][0])

    def test_no_clip_dir_means_no_fallback_warning(self):
        """Not asking for clips is not a fallback, so it must not warn."""
        rep = self._run(lambda *a, **k: _FakeResult())
        self.assertEqual([o.warnings for o in rep.ok], [[], [], []])

    def test_force_overrides_a_current_result(self):
        calls = []

        def fake(video, *a, **k):
            calls.append(video)
            return _FakeResult()

        # A summary that would otherwise be skipped.
        cfg = PipelineConfig()
        scale = cfg.preprocess.resolve_downsample()
        for v, out in assign_output_names(self.videos, self.out_dir).items():
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"params": params_to_json(self.params),
                           "provenance": {
                               "downsample": scale,
                               "block_size": cfg.flow.resolve_block_size(scale)},
                           "coverage": {"truncated": False, "start_frame": 0,
                                        "requested_n": None,
                                        "requested_frames": 32,
                                        "covered_frames": 32}}, f)

        rep = self._run(fake)
        self.assertEqual([len(rep.skipped), len(calls)], [3, 0])

        rep = self._run(fake, force=True)
        self.assertEqual([len(rep.ok), len(calls)], [3, 3])


class CliRunTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="shardcli_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.params_path = os.path.join(self.dir, "params.json")
        with open(self.params_path, "w", encoding="utf-8") as f:
            json.dump(PARAMS, f)

    def _argv(self, *extra):
        return [os.path.join(self.dir, "*.mp4"), "--params", self.params_path,
                "--out-dir", os.path.join(self.dir, "out"), "--quiet", *extra]

    def _touch_videos(self, *names):
        for n in names:
            p = os.path.join(self.dir, n)
            open(p, "w").close()
            with open(os.path.splitext(p)[0] + ".rois.json", "w",
                      encoding="utf-8") as f:
                json.dump(REPS, f)

    def test_bad_usage_is_exit_2_not_1(self):
        """A job runner must tell 'this command is wrong' (retrying on every
        node is pointless) from 'this footage failed' (it is not)."""
        from cli.run import main
        self._touch_videos("a.mp4")
        self.assertEqual(main(self._argv("--shard", "8/8")), 2)
        self.assertEqual(main(["--params", self.params_path, "--quiet"]), 2)

    def test_failed_video_is_exit_1(self):
        from cli.run import main
        self._touch_videos("a.mp4", "b.mp4")
        with unittest.mock.patch("core.shard.run_video",
                                 side_effect=BatchError("boom")):
            self.assertEqual(main(self._argv()), 1)

    def test_success_is_exit_0_and_writes_a_report(self):
        from cli.run import main
        self._touch_videos("a.mp4", "b.mp4")
        with unittest.mock.patch("core.shard.run_video",
                                 return_value=_FakeResult()), \
             unittest.mock.patch("core.shard.write_result"):
            self.assertEqual(main(self._argv("--shard", "0/2")), 0)
        rep = os.path.join(self.dir, "out", "shard-0-of-2.json")
        with open(rep, encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual(d["shard"], [0, 2])
        self.assertEqual(d["total_videos_in_list"], 2)
        self.assertEqual(d["counts"]["ok"], 1)

    def test_dry_run_decodes_nothing(self):
        from cli.run import main
        self._touch_videos("a.mp4", "b.mp4")
        with unittest.mock.patch("core.shard.run_video") as rv:
            self.assertEqual(main(self._argv("--dry-run")), 0)
        rv.assert_not_called()


class NoQtImportTest(unittest.TestCase):
    """The headless path must not pull Qt in transitively: on a display-less
    node that turns every job into a startup crash, and the core -> gui import
    chain makes it easy to break by accident. Same guard as
    ``test_batch_cli.py``, extended to the shard modules."""

    def test_importing_the_shard_cli_leaves_qt_unloaded(self):
        code = ("import sys; import cli.run, core.shard; "
                "mods=[m for m in sys.modules if 'PyQt' in m or m=='qtpy']; "
                "print(mods)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=os.path.dirname(
                                 os.path.dirname(os.path.abspath(__file__))))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "[]", out.stdout)


@requires_ffmpeg
class EndToEndTest(unittest.TestCase):
    """One real pass: two fixtures, two shards, the actual detector."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sharde2e_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.out_dir = os.path.join(self.dir, "out")
        for name in ("a.mp4", "b.mp4"):
            p = os.path.join(self.dir, name)
            _make_video(p)
            with open(os.path.splitext(p)[0] + ".rois.json", "w",
                      encoding="utf-8") as f:
                json.dump(REPS, f)
        self.params_path = os.path.join(self.dir, "params.json")
        with open(self.params_path, "w", encoding="utf-8") as f:
            json.dump(PARAMS, f)

    def _argv(self, shard):
        return [os.path.join(self.dir, "*.mp4"), "--params", self.params_path,
                "--out-dir", self.out_dir, "--shard", shard, "--quiet"]

    def test_two_shards_cover_both_videos(self):
        from cli.run import main
        self.assertEqual(main(self._argv("0/2")), 0)
        self.assertEqual(main(self._argv("1/2")), 0)
        for name in ("a.detections.json", "b.detections.json"):
            self.assertTrue(os.path.exists(os.path.join(self.out_dir, name)),
                            f"{name} was not produced by any shard")

    def test_rerun_skips_and_does_not_rewrite(self):
        from cli.run import main
        self.assertEqual(main(self._argv("0/2")), 0)
        out = os.path.join(self.out_dir, "a.detections.json")
        before = os.stat(out).st_mtime_ns
        self.assertEqual(main(self._argv("0/2")), 0)
        self.assertEqual(os.stat(out).st_mtime_ns, before,
                         "a current result was recomputed")


if __name__ == "__main__":
    unittest.main()
