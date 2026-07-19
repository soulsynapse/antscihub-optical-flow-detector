"""``python -m cli.count_frames`` -- measure how many frames a video decodes.

Batch O (T26). The container's frame count is a *claim* and the decoder's is a
*measurement*; ``core/framecount.py`` has the argument and ``FINDINGS.md``
section 15 the evidence. One file in four in this corpus over-claims --
``GX010047c2_02_17_26.MP4`` advertises 11328 frames and decodes 11308 -- and the
gap makes every full-length pass over it trip ``cli.detect``'s truncation guard.

    python -m cli.count_frames footage/*.MP4

Writes a ``.framecount.json`` sidecar beside each video. ``cli.detect`` and
``cli.run`` pick it up automatically and stop treating the container's number as
gospel; without one they fall back to the claim and say so in the result.

**This is a full decode and it is slow** -- minutes per multi-GB source. There is
no faster answer: "reached EOF" and "stopped early" are the same event to a
reader, so the only way to count decodable frames is to decode them. Pay it once
per file. If you are pre-transcoding ROI clips anyway (Batch I), do that instead
-- the cut measures the same number for free and writes the same sidecar.

Exit codes match the rest of the CLI: **0** all counted, **1** at least one file
could not be counted, **2** bad usage.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from core.batch import BatchError
from core.framecount import (FrameCountError, build_record, load_sidecar,
                             write_record)
from core.pretranscode import PretranscodeError, probe_source
from core.shard import expand_videos


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cli.count_frames", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+",
                   help="video paths or globs (quote globs on shells that expand them)")
    p.add_argument("--list-file", help="read additional paths, one per line")
    p.add_argument("--force", action="store_true",
                   help="re-count files that already have a valid sidecar. "
                        "Without this an up-to-date sidecar is left alone, since "
                        "the count is a property of the bytes and they have not "
                        "changed.")
    p.add_argument("--quiet", action="store_true", help="no progress output")
    return p


def _progress_printer(name: str):
    state = {"t": 0.0}

    def tick(frac):
        now = time.perf_counter()
        if now - state["t"] < 1.0:
            return
        state["t"] = now
        print(f"\r  {name}: {frac:.0%}", end="", file=sys.stderr, flush=True)
    return tick


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        videos = expand_videos(args.videos, list_file=args.list_file)
    # BatchError is what expand_videos raises for a pattern that matched nothing.
    # Without it here a mistyped glob exits 1 with a traceback instead of 2 with
    # a message -- and 1 means "a file could not be counted", which would tell a
    # job runner the wrong thing about a run that never started.
    except (BatchError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not videos:
        print("error: no videos matched", file=sys.stderr)
        return 2

    failures: list[tuple[str, str]] = []
    for path in videos:
        name = os.path.basename(path)
        if not args.force and load_sidecar(path) is not None:
            if not args.quiet:
                print(f"{name}: already counted", file=sys.stderr)
            continue
        try:
            *_, claimed = probe_source(path)
            rec = build_record(
                path, container_frames=claimed,
                progress=None if args.quiet else _progress_printer(name))
            out = write_record(rec)
        # A file that cannot be counted must not stop its list-mates, for the
        # same reason a failed video does not stop its shard (core/shard.py):
        # a batch of hundreds should report everything wrong with it in one run
        # rather than one thing per invocation.
        except (FrameCountError, PretranscodeError, OSError, ValueError) as e:
            failures.append((name, str(e)))
            if not args.quiet:
                print(f"\r{name}: FAILED -- {e}", file=sys.stderr)
            continue
        if not args.quiet:
            gap = rec.undecodable
            note = (f" ({gap} claimed frame(s) do not decode)" if gap
                    else " (container's claim was correct)")
            print(f"\r{name}: {rec.decoded_frames} frames{note} -> "
                  f"{os.path.basename(out)}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} file(s) could not be counted:", file=sys.stderr)
        for name, err in failures:
            print(f"  {name}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
