"""``python -m cli.detect`` -- run the tuned detector over a video, headlessly.

The batch driver's first slice (Batch J / T24). Tuning happens in the explorer;
this is what runs the tuned settings over footage on a node with no display.

    python -m cli.detect VIDEO --params tuned.json --out results/GX0100.json

``--params`` is ``ScalogramExplorer.detection_params()`` as JSON, minus
``region_index``: this runs every replicate unless ``--replicates-only`` narrows
it. Unbounded band endpoints are ``null``.

    {
      "channel_attr": "change",
      "freq_band_hz": [0.5, 4.0],
      "value_band":   [0.002, null],
      "count_band":   [3, null],
      "detect_window": 90,
      "centered": true
    }

Boxes come from the video's ``.rois.json`` sidecar unless ``--replicates`` names
one. Pass ``--manifest`` to decode pre-transcoded ROI clips (Batch I) instead of
the source -- same math, roughly 25x less decode.

Exit codes are meant to be read by a job runner: **0** success, **1** the job
could not be trusted (bad params, moved boxes, short decode), **2** bad usage.
A short decode is a failure and not a warning on purpose -- see ``core/batch.py``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from core.batch import (BatchError, config_from_overrides,
                        default_replicate_path, load_params, load_replicates,
                        run_video, write_result)
from core.pretranscode import PretranscodeError, read_manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cli.detect", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="source video")
    p.add_argument("--params", required=True,
                   help="tuned detector settings as JSON (see module docstring)")
    p.add_argument("--out", help="summary JSON path "
                   "(default: <video stem>.detections.json in --out-dir)")
    p.add_argument("--out-dir", default=".", help="directory for --out's default")
    p.add_argument("--replicates", help="replicate box JSON "
                   "(default: the video's .rois.json sidecar)")
    p.add_argument("--replicates-only", metavar="IDS",
                   help="comma-separated replicate ids to detect over "
                        "(default: all). Ids, not positions.")
    p.add_argument("--manifest", help="pre-transcode manifest; decode ROI clips "
                                      "instead of the source")
    p.add_argument("--clip-dir", help="directory holding the manifest's clips "
                                      "(default: the manifest's own directory)")
    p.add_argument("--start", type=int, default=0, help="first frame")
    p.add_argument("--frames", type=int, help="frame count (default: to the end)")
    p.add_argument("--downsample", type=float,
                   help="override the scale. NOT a free knob: it decides which "
                        "behaviours are detectable at all, so it is left at the "
                        "config default unless you say otherwise.")
    p.add_argument("--block-size", type=int, help="override the block size")
    p.add_argument("--allow-truncated", action="store_true",
                   help="accept a short decode instead of failing. The frames "
                        "past the cut are unexamined, NOT examined-and-clear.")
    p.add_argument("--no-series", action="store_true",
                   help="write only the JSON summary, no .npz of per-frame series")
    p.add_argument("--band-power", action="store_true",
                   help="also store the (T,B) band power, which makes re-tuning "
                        "the value band and window possible without a fresh "
                        "pass. Large: hundreds of MB per region per hour.")
    p.add_argument("--quiet", action="store_true", help="no progress output")
    return p


def _progress_printer():
    """A coarse stderr progress line. stderr, so ``--out /dev/stdout`` style
    redirection of the summary stays clean."""
    state = {"t": 0.0}

    def tick(done, total=None):
        now = time.perf_counter()
        if now - state["t"] < 1.0:
            return
        state["t"] = now
        pct = f" ({done / total:.0%})" if total else ""
        print(f"\r  extracting: frame {done}{pct}", end="", file=sys.stderr,
              flush=True)
    return tick


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if not os.path.isfile(args.video):
            raise BatchError(f"no such video: {args.video}")
        cfg = config_from_overrides(args.downsample, args.block_size)
        params = load_params(args.params)

        rep_path = args.replicates or default_replicate_path(args.video)
        if not rep_path:
            raise BatchError(
                f"no replicate boxes: pass --replicates, or put a .rois.json "
                f"sidecar next to {args.video}")
        replicates = load_replicates(rep_path)

        only = None
        if args.replicates_only:
            try:
                only = [int(s) for s in args.replicates_only.split(",") if s.strip()]
            except ValueError:
                raise BatchError(f"--replicates-only wants integer ids, got "
                                 f"{args.replicates_only!r}")

        manifest = clip_dir = None
        if args.manifest:
            manifest = read_manifest(args.manifest)
            clip_dir = args.clip_dir or os.path.dirname(
                os.path.abspath(args.manifest))

        out = args.out or os.path.join(
            args.out_dir,
            os.path.splitext(os.path.basename(args.video))[0] + ".detections.json")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

        res = run_video(args.video, replicates, cfg, params,
                        regions=only, manifest=manifest, clip_dir=clip_dir,
                        start=args.start, n=args.frames,
                        allow_truncated=args.allow_truncated,
                        replicate_path=rep_path,
                        progress=None if args.quiet else _progress_printer())
        if not args.quiet:
            print(file=sys.stderr)
        write_result(res, out, save_series=not args.no_series,
                     save_band_power=args.band_power)
    # PretranscodeError is included by name: it is a bare RuntimeError, so it
    # matches neither BatchError nor OSError/ValueError, and every condition it
    # reports (moved boxes, a truncated clip, a re-cut source) is squarely the
    # "this job cannot be trusted" class that exit 1 exists for. Without it the
    # operator gets a traceback on the most likely manifest failure.
    except (BatchError, PretranscodeError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        total = sum(r.n_detected_frames for r in res.regions)
        fps = res.covered_frames / res.extract_seconds if res.extract_seconds else 0
        print(f"{os.path.basename(args.video)}: {len(res.regions)} region(s), "
              f"{total} detected frame(s) over {res.covered_frames} examined "
              f"({res.extract_seconds:.1f}s extract at {fps:.1f} fps, "
              f"{res.detect_seconds:.1f}s detect) -> {out}", file=sys.stderr)
        for w in res.warnings:
            print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
