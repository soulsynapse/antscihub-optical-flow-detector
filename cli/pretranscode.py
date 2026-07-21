"""``python -m cli.pretranscode`` -- cut sources into per-replicate ROI clips.

Batch I (T23). ``core/pretranscode.py`` has the cut; this is the entry point that
runs it over footage. Until this file existed the cut was reachable only from the
test suite, while ``cli.run`` already *consumed* clips by default -- so the fast
path could be read but not written.

    python -m cli.pretranscode footage/*.MP4

Writes one clip per replicate **into that replicate's home** --
``<stem>_rep01/<stem>.mkv`` -- plus a ``<stem>.pretranscode.json`` manifest, all
**beside each video by default**, which is where ``core/shard.find_manifest``
looks. Move them with ``--out-dir`` and later passes need ``--clip-dir`` to match;
leave it alone and ``cli.run`` picks them up with no flag at all.

The home is the directory that already holds that replicate's track, marks and
tuning (``core/replicate_home.py``), so the cut adds a file to a directory that
exists rather than introducing a layout of its own. ``--layout flat`` restores
the old ``<stem>__rep<NN>.mkv`` naming; manifests written either way still read,
so nothing already cut needs re-cutting. **Note that ``--out-dir`` moves the
homes too** -- clips land in ``<out-dir>/<stem>_rep01/`` and no longer sit
alongside that replicate's other artefacts.

Boxes come from each video's ``.rois.json`` sidecar. A video with no sidecar is a
failure, not a skip -- there is no sensible default framing.

**What this buys, honestly.** Roughly 25x cheaper decode in isolation, which was
worth **1.06x** end to end on the one corpus measured (``FINDINGS.md`` section
16): the producer thread already hides decode behind the math, and at scale 1.0
the pass is math-bound. It still plainly wins on storage-local working sets, on
machines whose sources sit on slow or remote disk, and on any pass at a smaller
scale, where the math shrinks and the decode does not. Run it for those reasons,
not for single-process throughput.

**A cut also measures the decodable frame count for free**, because it decodes
the whole source anyway, and writes the same ``.framecount.json`` sidecar
``cli.count_frames`` would spend minutes per file producing. If you were going to
do both, do this one.

**Quality is a storage decision, not a speed one.** Decode cost tracks pixels,
not bytes, so the presets are within 12% of each other on speed and differ only
in size and fidelity. It is also not a knob to set casually: below ``lossless``
the clip's pixels differ from the source's, and ``change`` -- the detection
default -- measures precisely the frame-to-frame quantity that lossy inter-frame
coding perturbs. Clips cut at different quality carry different provenance keys
and are never silently comparable. Whether a given behaviour survives a given
preset is measurable end to end and is deliberately not assumed here.

Exit codes match the rest of the CLI: **0** all cut, **1** at least one video
could not be cut, **2** bad usage.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from core.batch import BatchError, default_replicate_path, load_replicates
from core.pretranscode import (CLIP_LAYOUTS, DEFAULT_CLIP_LAYOUT,
                               DEFAULT_QUALITY, QUALITY_PRESETS,
                               PretranscodeError, build_pretranscode,
                               clip_path, manifest_path_for, read_manifest,
                               verify_manifest)
from core.shard import expand_videos


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cli.pretranscode", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+",
                   help="video paths or globs (quote globs on shells that expand them)")
    p.add_argument("--list-file", help="read additional paths, one per line")
    p.add_argument("--out-dir",
                   help="directory for clips and manifests (default: beside each "
                        "video, which is where cli.run discovers them without a flag)")
    p.add_argument("--replicates",
                   help="replicate box JSON. Applies to ONE video only -- see the "
                        "error text if you pass it with several.")
    p.add_argument("--layout", default=DEFAULT_CLIP_LAYOUT,
                   choices=sorted(CLIP_LAYOUTS),
                   help=f"where each clip is written (default: "
                        f"{DEFAULT_CLIP_LAYOUT}). 'home' puts it in the "
                        "replicate's own directory beside its track and marks; "
                        "'flat' is the pre-Batch-S naming. Moves bytes, not "
                        "pixels -- a clip cut either way is the same clip.")
    p.add_argument("--quality", default=DEFAULT_QUALITY,
                   choices=sorted(QUALITY_PRESETS),
                   help=f"storage/fidelity preset (default: {DEFAULT_QUALITY}). "
                        "Barely affects decode speed; see the module docstring.")
    # No --block-size. build_pretranscode takes one, but it reaches only
    # build_layout's source_box rounding, which is block-independent (verified
    # across 16/32/64/128 on both aligned and ragged boxes: identical crops).
    # Nothing else from the layout reaches a clip or the manifest, so the flag
    # could not change any output -- and offering it would imply the cut has to
    # match the detection grid, which is exactly the thing it does not do.
    p.add_argument("--overwrite", action="store_true",
                   help="re-cut videos that already have clips. Without this an "
                        "up-to-date cut is left alone and a STALE one is an error "
                        "rather than a silent re-use or a silent replacement.")
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


def _clip_bytes(man) -> int:
    """Total clip size, from what the cut recorded rather than a re-stat.

    ``verify_manifest`` has already confirmed every clip is present at exactly
    this size, so hitting the filesystem again would answer the same question
    twice -- and a stat loop that skips what it cannot find would understate the
    total instead of failing."""
    return sum(int(c.size_bytes or 0) for c in man.clips)


def _clip_paths(man, out_dir: str) -> dict[str, str]:
    """Comparison key -> path as written, for every clip a manifest names."""
    return {os.path.normcase(os.path.abspath(clip_path(out_dir, c.filename))):
            clip_path(out_dir, c.filename) for c in man.clips}


def _superseded(prev, man, out_dir: str) -> list[str]:
    """Clips the PREVIOUS cut wrote that this one did not replace.

    Only ``--layout`` produces these, and it produces them silently: a re-cut
    from flat to home writes new files and leaves the old ones on disk, so a
    corpus quietly costs twice the storage the report just quoted. They are
    named rather than deleted -- ffmpeg's ``-y`` overwrote everything this cut
    actually claims, and a CLI that removes video files it was not asked about
    is a worse trade than a line of output.
    """
    if prev is None:
        return []
    current = _clip_paths(man, out_dir)
    return sorted(p for key, p in _clip_paths(prev, out_dir).items()
                  if key not in current and os.path.exists(p))


def _existing_state(video: str, out_dir: str,
                    reps: list[dict]) -> tuple[str, str]:
    """What is already sitting in ``out_dir`` for this video, and whether it can
    be trusted as a finished job.

    Resume is by identity, not existence -- the rule Batch J arrived at the hard
    way (``FINDINGS.md`` section 13). A manifest whose boxes have moved, whose
    source has changed, or whose clips are short is not a finished job, and
    treating it as one would hand every later pass stale pixels under a current
    name.

    **The manifest must also be THIS video's.** Manifests are named from the
    video's basename, so two sources called ``GX0100.MP4`` in different session
    directories map to one manifest path under a shared ``--out-dir`` -- the
    normal multi-session case. ``verify_manifest`` cannot catch it: it validates
    the manifest against the source *the manifest names*, which is present and
    unchanged, so it passes and the second video is reported "already cut"
    having never been examined. That is section 13's exact failure, reproduced
    here, and it is why identity is checked against the video in hand rather
    than inferred from the filename that led us to the file.

    Returns ``(kind, detail)`` with ``kind`` one of ``"none"``, ``"ok"``,
    ``"collision"``, ``"stale"``. A collision is kept distinct from staleness
    because ``--overwrite`` must NOT resolve it: "re-cut this video" and
    "clobber a different video's output" are different instructions, and only
    one of them was given.
    """
    path = manifest_path_for(video, out_dir)
    if not os.path.exists(path):
        return "none", ""
    try:
        man = read_manifest(path)
        if os.path.normcase(os.path.abspath(man.source_path)) != \
                os.path.normcase(os.path.abspath(video)):
            return "collision", (
                f"{os.path.basename(path)} belongs to {man.source_path}, not to "
                f"this video -- their basenames collide in this --out-dir. Cut "
                f"them to separate directories (the default is beside each "
                f"video, which cannot collide).")
        verify_manifest(man, out_dir, reps)
    except (PretranscodeError, OSError, ValueError) as e:
        return "stale", str(e)
    return "ok", ""


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        videos = expand_videos(args.videos, list_file=args.list_file)
    # BatchError included deliberately: expand_videos raises it for a pattern
    # that matched nothing, which is the single most likely thing an operator
    # gets wrong, and its own docstring says a CLI owes it an exit code rather
    # than a traceback.
    except (BatchError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not videos:
        print("error: no videos matched", file=sys.stderr)
        return 2
    # One sidecar cannot describe several sources. The boxes are fractional, so a
    # shared rig makes this look reasonable, and that is exactly why it is refused
    # rather than allowed: a wrong-but-plausible framing produces clips that
    # verify, decode, and attribute the wrong pixels to every detection.
    if args.replicates and len(videos) > 1:
        print(f"error: --replicates names one box set but {len(videos)} videos "
              "matched; run them one at a time, or let each video use its own "
              ".rois.json sidecar", file=sys.stderr)
        return 2

    failures: list[tuple[str, str]] = []
    for path in videos:
        name = os.path.basename(path)
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(path))
        try:
            rep_path = args.replicates or default_replicate_path(path)
            if not rep_path:
                raise BatchError("no replicate sidecar (.rois.json) beside it, "
                                 "and no --replicates given")
            reps = load_replicates(rep_path)

            kind, detail = _existing_state(path, out_dir, reps)
            # Checked before --overwrite is consulted: overwriting here would
            # destroy a DIFFERENT video's clips and report success for both.
            if kind == "collision":
                raise PretranscodeError(detail)
            if kind == "ok" and not args.overwrite:
                if not args.quiet:
                    print(f"{name}: already cut", file=sys.stderr)
                continue
            if kind == "stale" and not args.overwrite:
                raise PretranscodeError(
                    f"clips exist but cannot be trusted -- {detail}. Pass "
                    "--overwrite to re-cut them.")

            # Read before the cut replaces it: the only record of where the
            # previous cut put its clips is the manifest about to be rewritten.
            try:
                prev = read_manifest(manifest_path_for(path, out_dir))
            except (PretranscodeError, OSError, ValueError):
                prev = None

            man = build_pretranscode(
                path, reps, out_dir, quality=args.quality,
                clip_layout=args.layout, overwrite=args.overwrite,
                progress=None if args.quiet else _progress_printer(name))
        except KeyboardInterrupt:
            # build_pretranscode removes partial clips on the way out; a
            # half-written clip that looked complete is the failure this whole
            # module exists to avoid handing downstream.
            print(f"\r{name}: interrupted, partial clips removed", file=sys.stderr)
            return 1
        # A video that cannot be cut must not stop its list-mates, for the same
        # reason a failed video does not stop its shard (core/shard.py).
        except (PretranscodeError, BatchError, OSError, ValueError) as e:
            failures.append((name, str(e)))
            if not args.quiet:
                print(f"\r{name}: FAILED -- {e}", file=sys.stderr)
            continue

        if not args.quiet:
            # Measured storage, not a score. The standing decision in todo.md is
            # to show wall clock, bytes, or an event count on a named clip --
            # never a single number summarizing "how much worse".
            src = os.path.getsize(path)
            clips = _clip_bytes(man)
            pct = f"{100.0 * clips / src:.0f}%" if src else "n/a"
            print(f"\r{name}: {len(man.clips)} clip(s), {man.frame_count} frames, "
                  f"{clips / 1e6:.0f} MB ({pct} of source, ~{man.quality_rms:.3f} "
                  f"grey-level RMS) -> "
                  f"{os.path.basename(manifest_path_for(path, out_dir))}",
                  file=sys.stderr)
            old = _superseded(prev, man, out_dir)
            if old:
                print(f"  note: {len(old)} clip(s) from the previous cut are no "
                      f"longer referenced and still occupy disk; delete them by "
                      f"hand:", file=sys.stderr)
                for p in old:
                    print(f"    {p}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} video(s) could not be cut:", file=sys.stderr)
        for name, err in failures:
            print(f"  {name}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
