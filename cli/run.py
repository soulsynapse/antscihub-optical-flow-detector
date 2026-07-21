"""``python -m cli.run`` -- run the tuned detector over a shard of a file list.

The fan-out half of Batch J (T24). ``cli/detect.py`` is one video; this is the
one a job runner launches N copies of::

    python -m cli.run "footage/*.MP4" --params tuned.json \\
        --shard 3/8 --out-dir results/

Every shard is given the *same* arguments and differs only in ``--shard i/N``
(``i`` is 0-based). Partitioning is a stride over the sorted list, so no shard
has to consult any other and a preempted one can simply be relaunched.

Videos may be given as literal paths, glob patterns (expanded here, because
PowerShell does not expand them for native commands), directories, or a
``--video-list`` file. Replicate boxes come from each video's ``.rois.json``
sidecar, and pre-transcoded ROI clips (Batch I) are picked up automatically from
each video's ``.pretranscode.json`` manifest unless ``--no-clips`` says not to.

Finished videos are skipped on a re-run, but only when the existing summary
records the same params and the same resolved scale and block size -- a retuned
band recomputes rather than silently returning yesterday's detections. ``--force``
recomputes regardless.

Exit codes are the job runner's contract: **0** every video in the shard
succeeded or was skipped, **1** at least one video failed (the others still ran
-- see the shard report), **2** bad usage.
"""
from __future__ import annotations

import argparse
import os
import sys

from core.batch import BatchError, config_from_overrides, load_params
from core.shard import (assign_output_names, expand_videos, parse_shard,
                        partition, run_shard, write_shard_report)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cli.run", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="*", help="video paths, glob patterns, or "
                                             "directories")
    p.add_argument("--video-list", help="file of video paths, one per line "
                                        "('#' comments allowed)")
    p.add_argument("--params", required=True,
                   help="tuned detector settings as JSON (see cli/detect.py)")
    p.add_argument("--shard", default="0/1", metavar="i/N",
                   help="this shard of N, i 0-based (default: 0/1, everything)")
    p.add_argument("--out-dir", default=".",
                   help="directory for per-video results and the shard report")
    p.add_argument("--report", help="shard report JSON path "
                   "(default: shard-<i>-of-<N>.json in --out-dir)")
    p.add_argument("--replicates", help="replicate box JSON applied to EVERY "
                   "video (default: each video's own .rois.json sidecar)")
    p.add_argument("--replicates-only", metavar="IDS",
                   help="comma-separated replicate ids to detect over")
    p.add_argument("--clip-dir", help="directory holding pre-transcode "
                   "manifests and clips (default: beside each video)")
    p.add_argument("--no-clips", action="store_true",
                   help="ignore pre-transcode manifests and decode the sources")
    p.add_argument("--start", type=int, default=0, help="first frame")
    p.add_argument("--frames", type=int, help="frame count (default: to the end)")
    p.add_argument("--downsample", type=float,
                   help="override the scale. NOT a free knob: it decides which "
                        "behaviours are detectable at all.")
    p.add_argument("--block-size", type=int, help="override the block size")
    p.add_argument("--allow-truncated", action="store_true",
                   help="accept a short decode instead of failing that video")
    p.add_argument("--force", action="store_true",
                   help="recompute videos whose results are already current")
    p.add_argument("--no-series", action="store_true",
                   help="write only the JSON summaries, no .npz per video")
    p.add_argument("--band-power", action="store_true",
                   help="also retain the (T,B) band power per region. Large.")
    p.add_argument("--dry-run", action="store_true",
                   help="list this shard's videos and exit, decoding nothing")
    p.add_argument("--quiet", action="store_true", help="no progress output")
    return p


def _printer(quiet: bool):
    """Per-video progress on stderr, so a redirected stdout stays clean."""
    def on_event(kind, p):
        if quiet:
            return
        if kind == "start":
            print(f"[{p['pos'] + 1}/{p['total']}] "
                  f"{os.path.basename(p['video'])}", file=sys.stderr, flush=True)
        elif kind == "skip":
            print(f"    skipped ({p['reason']})", file=sys.stderr, flush=True)
        elif kind == "fail":
            print(f"    FAILED: {p['error']}", file=sys.stderr, flush=True)
        elif kind == "done":
            o = p["outcome"]
            print(f"    {o.n_detected_frames} detected frame(s) over "
                  f"{o.covered_frames} examined in {o.seconds:.1f}s",
                  file=sys.stderr, flush=True)
            for w in o.warnings:
                print(f"    warning: {w}", file=sys.stderr, flush=True)
    return on_event


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if not args.videos and not args.video_list:
            raise BatchError("no videos: pass paths/patterns or --video-list")
        index, count = parse_shard(args.shard)
        cfg = config_from_overrides(args.downsample, args.block_size)
        params = load_params(args.params)
        videos = expand_videos(args.videos, list_file=args.video_list)
        if not videos:
            raise BatchError("the video list is empty")

        only = None
        if args.replicates_only:
            try:
                only = [int(s) for s in args.replicates_only.split(",")
                        if s.strip()]
            except ValueError:
                raise BatchError(f"--replicates-only wants integer ids, got "
                                 f"{args.replicates_only!r}")

        mine = partition(videos, index, count)
        if args.dry_run:
            # Output names are resolved here too, so a collision-driven rename
            # (s1_GX010047 rather than GX010047) is visible before a run rather
            # than discovered in the results directory afterwards.
            names = assign_output_names(videos, args.out_dir)
            # stdout, not stderr: this output is meant to be piped.
            for v in mine:
                print(f"{v}\t{names[v]}")
            print(f"shard {index}/{count}: {len(mine)} of {len(videos)} video(s)",
                  file=sys.stderr)
            return 0
        os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)
    except (BatchError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        # Usage, not a failed job: nothing was run, so this is exit 2. A job
        # runner must be able to tell "this command is wrong" (retrying is
        # pointless on every node) from "this footage failed" (it is not).
        return 2

    report = run_shard(
        videos, cfg, params, out_dir=args.out_dir, shard=(index, count),
        replicate_path=args.replicates, clip_dir=args.clip_dir,
        use_clips=not args.no_clips, regions=only,
        start=args.start, frames=args.frames,
        allow_truncated=args.allow_truncated, force=args.force,
        save_series=not args.no_series, save_band_power=args.band_power,
        on_event=_printer(args.quiet))

    report_path = args.report or os.path.join(
        args.out_dir, f"shard-{index}-of-{count}.json")
    try:
        write_shard_report(report, report_path)
    except OSError as e:
        # The work is done and the per-video summaries are on disk; losing the
        # report must not be reported as losing the run. But it is a failure --
        # the failed-video list lives there.
        print(f"error: could not write shard report: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"shard {index}/{count}: {len(report.ok)} ok, "
              f"{len(report.skipped)} skipped, {len(report.failed)} failed "
              f"in {report.seconds:.1f}s -> {report_path}", file=sys.stderr)
        for o in report.failed:
            print(f"  failed: {o.video_path}: {o.error}", file=sys.stderr)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
