"""File-level job partitioning: the second half of Batch J (T24).

``core/batch.py`` runs one video. This runs *a shard of a file list*, which is
what turns the headless path into fan-out: a job runner launches N identical
commands differing only in ``--shard i/N``, and no coordinator, queue, or shared
filesystem lock is involved. Each node brings its own decode capacity, which is
the whole reason fan-out is the answer here rather than more threads
(``FINDINGS.md`` section 1: a single process is ~1.03x realtime at scale 1.0
because the work is math-bound, and the multi-process ceiling is set by decode).

Three decisions in here are load-bearing.

**Partition by stride, not by contiguous chunk.** ``sorted(paths)[i::N]``
interleaves; ``paths[i*k:(i+1)*k]`` would give each shard a contiguous run.
Footage is named by capture order and file size correlates strongly with
position within a session, so contiguous chunks systematically hand one shard
the long files and another the short ones. Stride spreads that. Both are
deterministic and stateless -- the property that actually matters, since a
preempted shard must be relaunchable without consulting the others -- but stride
is the one that balances.

**A failed video does not fail its shard-mates.** One corrupt file among forty
should not discard thirty-nine videos' worth of paid decode. Failures are
recorded per video, the run continues, and the *shard* exits non-zero at the
end. A job runner still sees a failed job; it just sees it after the work that
could be salvaged was.

**Resume is by output identity, not output existence.** Skipping any video whose
summary file exists would silently keep results computed under *different
params* -- the operator re-runs a shard with a retuned band and gets the old
detections back, with nothing in the output saying so. That is the
silent-wrong-result shape this codebase spends its effort refusing. So a skip
requires the existing summary to record the same params AND the same resolved
scale and block size; anything else is stale and is recomputed.
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
from dataclasses import dataclass, field

from core.batch import (BatchError, default_replicate_path, load_replicates,
                        params_to_json, run_video, write_result)
from core.config import PipelineConfig
from core.pretranscode import (PretranscodeError, manifest_path_for,
                               read_manifest)

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg")


def _ci_pattern(pat: str) -> str:
    """Rewrite a glob so it matches case-insensitively on every platform.

    ``glob`` inherits the filesystem's case rules, so ``"*.mp4"`` picks up
    ``GX0100.MP4`` on Windows and does not on Linux. That divergence is a shard
    hazard rather than a cosmetic one: a fan-out whose nodes disagree about the
    file list partitions a *different* list on each node, so some footage is
    examined by nobody -- and every shard still exits 0, because each one
    completed the list it saw. Camera-produced footage is exactly the case that
    triggers it, GoPro writing ``.MP4`` uppercase.

    Each alphabetic character becomes a two-case class. Two things are left
    alone. Characters inside an existing ``[...]`` class, since rewriting them
    would nest brackets and change what the class means. And any path segment
    with no glob magic in it -- a literal ``C:`` turned into ``[cC]:`` is not a
    drive letter any more, and literal directory names need no help because the
    filesystem already resolves them by its own case rules.
    """
    def segment(seg: str) -> str:
        if not any(c in seg for c in "*?["):
            return seg
        out, in_class = [], False
        for ch in seg:
            if in_class:
                out.append(ch)
                in_class = ch != "]"
            elif ch == "[":
                out.append(ch)
                in_class = True
            elif ch.isalpha():
                out.append(f"[{ch.lower()}{ch.upper()}]")
            else:
                out.append(ch)
        return "".join(out)

    return "".join(segment(p) for p in re.split(r"([\\/])", pat))

# Errors that mean "this job could not be trusted" rather than "this code has a
# bug". Same set cli/detect.py turns into exit 1, and for the same reason:
# PretranscodeError is a bare RuntimeError, so it matches none of the others by
# inheritance and has to be named.
JOB_ERRORS = (BatchError, PretranscodeError, OSError, ValueError)


# -- the file list -----------------------------------------------------------

def expand_videos(patterns: list[str], *, list_file: str | None = None) -> list[str]:
    """Resolve command-line video arguments into a sorted, de-duplicated list.

    Accepts literal paths, glob patterns, and directories (scanned one level for
    known video suffixes). Globs are expanded here rather than left to the shell
    because PowerShell -- the primary shell on this machine -- does not expand
    them for native commands at all, so ``"footage/*.MP4"`` would otherwise
    arrive as a literal filename that does not exist.
    """
    out: list[str] = []
    if list_file:
        # Relative entries resolve against the LIST FILE's directory, not the
        # process CWD. A job runner launches from wherever it likes, and a list
        # file that means different videos depending on the launch directory
        # would give each node a different partition -- the same
        # some-footage-examined-by-nobody failure the ordering rules below
        # exist to prevent.
        base = os.path.dirname(os.path.abspath(list_file))
        with open(list_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # '#' comments so a list file can carry notes about which
                # sessions it covers, and blank lines so it can be grouped.
                if line and not line.startswith("#"):
                    out.append(line if os.path.isabs(line)
                               else os.path.join(base, line))
    for pat in patterns:
        if os.path.isdir(pat):
            hits = [os.path.join(pat, e) for e in sorted(os.listdir(pat))
                    if e.lower().endswith(VIDEO_SUFFIXES)]
            # Loud for the same reason the glob branch is loud: a mistyped or
            # wrong-level directory that contributes nothing silently produces
            # a run that examines less footage than the operator asked for and
            # still exits 0.
            if not hits:
                raise BatchError(
                    f"directory holds no video files (looked for "
                    f"{', '.join(VIDEO_SUFFIXES)}, one level deep): {pat}")
            out.extend(hits)
        elif any(c in pat for c in "*?["):
            hits = glob.glob(_ci_pattern(pat))
            if not hits:
                raise BatchError(f"pattern matched no files: {pat}")
            out.extend(hits)
        else:
            out.append(pat)

    # De-duplicated with the FILESYSTEM's case rules (normcase), because that is
    # what decides whether two spellings are one file: on Windows 'a/b.MP4' and
    # 'a\\b.MP4' are one video and must not become two jobs writing one output,
    # while on Linux 'a.mp4' and 'A.mp4' are genuinely two files and collapsing
    # them would drop one silently.
    seen: set[str] = set()
    uniq: list[str] = []
    for p in out:
        key = os.path.normcase(os.path.normpath(os.path.abspath(p)))
        if key not in seen:
            seen.add(key)
            uniq.append(os.path.normpath(p))
    return sorted(uniq, key=_order_key)


def _order_key(path: str) -> tuple[str, str]:
    """A sort key that does not depend on the platform.

    ``os.path.normcase`` must NOT be used here even though it is right for
    de-duplication: it lowercases on Windows and is the identity on POSIX, so
    the same file list sorts differently on different nodes. ['B.mp4','a.mp4']
    sorts a,B on Windows and B,a on Linux, and since the partition is a stride
    over the sorted list, a Windows shard 0 and a Linux shard 1 then cover
    overlapping-but-incomplete slices -- some footage decoded twice, some
    examined by nobody, every shard exiting 0.

    Case-folded first so case is not the deciding factor anywhere, with the
    exact path as a tiebreak so paths differing only in case (possible on
    POSIX) still get a stable total order rather than an arbitrary one.
    """
    p = path.replace("\\", "/")
    return (p.casefold(), p)


def parse_shard(spec: str) -> tuple[int, int]:
    """``"i/N"`` -> ``(i, N)``, with ``i`` **0-based**.

    The off-by-one here is worth being loud about: an operator who launches
    shards 1..N under a 0-based scheme silently never runs shard 0, and its
    videos are simply missing from the results with every job reporting success.
    So ``i == N`` is rejected with a message that names the convention rather
    than being accepted as a wrap-around or an empty shard.
    """
    if spec.count("/") != 1:
        raise BatchError(f"--shard wants the form i/N (0-based), got {spec!r}")
    a, b = spec.split("/")
    try:
        i, n = int(a), int(b)
    except ValueError:
        raise BatchError(f"--shard wants two integers, got {spec!r}")
    if n < 1:
        raise BatchError(f"--shard N must be >= 1, got {n}")
    if not 0 <= i < n:
        raise BatchError(
            f"--shard index {i} is out of range for {n} shard(s): indices are "
            f"0-based, so use 0/{n} through {n - 1}/{n}")
    return i, n


def partition(paths: list[str], index: int, count: int) -> list[str]:
    """This shard's videos: a stride, so long files spread across shards rather
    than piling into whichever chunk holds the long session."""
    return list(paths[index::count])


# -- per-video inputs --------------------------------------------------------

def output_path_for(video_path: str, out_dir: str) -> str:
    """The un-disambiguated output path for one video: ``<stem>.detections.json``,
    matching ``cli/detect.py``'s default.

    Only safe when the stem is unique across the run. Use ``assign_output_names``
    for anything that processes a list -- see the collision it exists to stop.
    """
    return os.path.join(out_dir, output_stem(video_path) + ".detections.json")


def output_stem(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def assign_output_names(videos: list[str], out_dir: str) -> dict[str, str]:
    """Map every video to a distinct output path.

    Naming by basename alone is wrong for a fan-out and fails *silently* in the
    worst way. Cameras number files per card, so ``s1/GX010047.MP4`` and
    ``s2/GX010047.MP4`` in one run are the normal case, not a corner case --
    and both map to ``GX010047.detections.json``. Whichever runs second finds
    the first's summary, matches on params and scale, and is reported
    "skipped (up to date)": its footage is never decoded, and the run exits 0.
    If the two land on different nodes it is an overwrite instead. Either way a
    video's detections are missing with nothing saying so.

    Colliding stems are therefore extended leftward with parent directory
    components until unique, giving ``s1_GX010047`` / ``s2_GX010047``. Two
    properties matter and both come from taking the WHOLE list rather than one
    shard's slice: the result is independent of which shard a video falls in,
    and it is identical on every node, since each is handed the same list.
    Non-colliding stems are left alone, so the common case keeps the plain name
    ``cli/detect.py`` writes.
    """
    groups: dict[str, list[str]] = {}
    for v in videos:
        groups.setdefault(output_stem(v).casefold(), []).append(v)

    names: dict[str, str] = {}
    for group in groups.values():
        if len(group) == 1:
            names[group[0]] = output_stem(group[0])
            continue
        depth = 1
        while True:
            cand = {v: _stem_with_parents(v, depth) for v in group}
            if len({c.casefold() for c in cand.values()}) == len(group):
                break
            deeper = any(_stem_with_parents(v, depth + 1) != cand[v]
                         for v in group)
            if not deeper:
                # Paths exhausted and still identical -- only reachable when the
                # same file was named two ways the dedup did not catch. Refuse
                # rather than pick a winner: an arbitrary choice here is the
                # silent-overwrite this whole function exists to prevent.
                raise BatchError(
                    "cannot derive distinct output names for videos sharing a "
                    f"name: {', '.join(sorted(group))}")
            depth += 1
        names.update(cand)

    return {v: os.path.join(out_dir, n + ".detections.json")
            for v, n in names.items()}


def _stem_with_parents(video_path: str, depth: int) -> str:
    """``<parent-depth>_..._<stem>``, using the path as supplied.

    As supplied, not resolved: an absolute path resolved on one node can differ
    from another's (different mount points for the same share), and the names
    must agree across nodes.
    """
    parts = os.path.normpath(video_path).replace("\\", "/").split("/")
    parents = [p for p in parts[:-1] if p not in ("", ".", "..")]
    keep = parents[-depth:] if depth else []
    return "_".join(keep + [output_stem(video_path)])


def find_manifest(video_path: str, clip_dir: str | None) -> str | None:
    """A Batch I manifest for this video, if one is there to be found.

    A manifest belongs to exactly one source, so a shard cannot take a single
    ``--manifest`` the way ``cli/detect.py`` does; it discovers one per video by
    the naming convention ``build_pretranscode`` writes.
    """
    for d in ([clip_dir] if clip_dir else []) + [os.path.dirname(
            os.path.abspath(video_path))]:
        cand = manifest_path_for(video_path, d)
        if os.path.exists(cand):
            return cand
    return None


# -- resume ------------------------------------------------------------------

def _resolved_settings(cfg: PipelineConfig) -> tuple[float, int]:
    """The scale and block size a pass under ``cfg`` would actually record.

    Resolved rather than raw: ``downsample=None`` means 1.0 and a ``None`` block
    size tracks the scale, so comparing the raw config fields against a written
    summary's provenance would call a result stale over a difference that does
    not exist and re-decode hours of finished work for nothing.
    """
    scale = cfg.preprocess.resolve_downsample()
    return float(scale), int(cfg.flow.resolve_block_size(scale))


def is_current(summary: dict, params: dict, cfg: PipelineConfig, *,
               start: int = 0, frames: int | None = None) -> bool:
    """Whether an existing summary answers *this* job's question.

    Compares the tuned params, the resolved scale/block size, and the frame
    window. The window is part of the identity, not a detail: a shard run once
    with ``--frames 1000`` as a smoke test and then re-run full-length would
    otherwise match on params and scale, be skipped as "up to date", and leave
    the 1000-frame result standing as the video's detections -- everything past
    frame 1000 unexamined rather than clear, which is the silent false negative
    this codebase spends the most effort refusing.

    A summary with no recorded window is treated as not current, so results
    written before the window was recorded are recomputed rather than trusted.
    """
    if summary.get("params") != params_to_json(params):
        return False
    prov = summary.get("provenance") or {}
    scale, block = _resolved_settings(cfg)
    if prov.get("downsample") is None or prov.get("block_size") is None:
        return False
    if float(prov["downsample"]) != scale or int(prov["block_size"]) != block:
        return False

    cov = summary.get("coverage") or {}
    # `requested_n` is the RAW request (None = to the end), which the resolved
    # counts cannot reconstruct: a finished `--frames 1000` pass and a finished
    # to-the-end pass over a 1000-frame video agree on every other field.
    # "start_frame" absent means a summary written before the window was
    # recorded -- not current, so it is recomputed rather than trusted.
    if "start_frame" not in cov or "requested_n" not in cov:
        return False
    if int(cov["start_frame"]) != int(start):
        return False
    prev_n = cov["requested_n"]
    return (prev_n is None) == (frames is None) and (
        frames is None or int(prev_n) == int(frames))


def skip_reason(out_path: str, params: dict, cfg: PipelineConfig, *,
                start: int = 0, frames: int | None = None) -> str | None:
    """``None`` if this video must be run, else why it is being skipped.

    ``write_result`` is atomic, so an existing summary is a complete one and
    there is no torn-file case to defend against. What is left to check is
    whether it answers the same question.
    """
    if not os.path.exists(out_path):
        return None
    try:
        with open(out_path, encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, ValueError):
        # Unreadable is not "done". Recompute rather than skip: the cost is one
        # video, and the alternative is a missing result reported as a success.
        return None
    if not isinstance(summary, dict) or not is_current(
            summary, params, cfg, start=start, frames=frames):
        return None
    cov = summary.get("coverage") or {}
    # A truncated result is only ever written under --allow-truncated, but it is
    # still a partial answer. Re-running it is the safe default: a resumed shard
    # should converge on complete coverage, not inherit whatever the crashed run
    # settled for.
    if cov.get("truncated"):
        return None
    return "up to date"


# -- the run -----------------------------------------------------------------

@dataclass
class VideoOutcome:
    """What happened to one video. ``status`` is one of ``ok`` / ``skipped`` /
    ``failed``; a shard's exit code is decided by whether any is ``failed``."""
    video_path: str
    status: str
    out_path: str | None = None
    error: str | None = None
    seconds: float = 0.0
    n_detected_frames: int = 0
    covered_frames: int = 0
    # Which pixels this result came from. Recorded per video because below
    # `lossless` a clip-derived and a source-derived result are different
    # measurements of the same footage (FINDINGS.md section 10), and a run where
    # some videos found a manifest and others silently did not would otherwise
    # mix the two with nothing in the output distinguishing them.
    used_clips: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_summary(self) -> dict:
        d = {"video_path": self.video_path, "status": self.status,
             "seconds": round(self.seconds, 3)}
        if self.out_path:
            d["out_path"] = self.out_path
        if self.error:
            d["error"] = self.error
        if self.status == "ok":
            d["n_detected_frames"] = self.n_detected_frames
            d["covered_frames"] = self.covered_frames
            d["used_clips"] = self.used_clips
        if self.warnings:
            d["warnings"] = list(self.warnings)
        return d


@dataclass
class ShardReport:
    """The shard's own record, written beside the per-video results so a fan-out
    can be audited without opening every summary -- in particular so the videos
    that FAILED are enumerated somewhere, rather than being visible only as an
    absence among the successes."""
    shard_index: int
    shard_count: int
    outcomes: list[VideoOutcome]
    total_videos: int
    seconds: float = 0.0

    @property
    def failed(self) -> list[VideoOutcome]:
        return [o for o in self.outcomes if o.status == "failed"]

    @property
    def ok(self) -> list[VideoOutcome]:
        return [o for o in self.outcomes if o.status == "ok"]

    @property
    def skipped(self) -> list[VideoOutcome]:
        return [o for o in self.outcomes if o.status == "skipped"]

    def to_summary(self) -> dict:
        return {
            "shard": [self.shard_index, self.shard_count],
            "total_videos_in_list": self.total_videos,
            "counts": {"ok": len(self.ok), "skipped": len(self.skipped),
                       "failed": len(self.failed)},
            "seconds": round(self.seconds, 3),
            "failed_videos": [o.video_path for o in self.failed],
            "videos": [o.to_summary() for o in self.outcomes],
        }


def run_shard(videos: list[str], cfg: PipelineConfig, params: dict, *,
              out_dir: str = ".", shard: tuple[int, int] = (0, 1),
              replicate_path: str | None = None,
              clip_dir: str | None = None, use_clips: bool = True,
              regions: list[int] | None = None,
              start: int = 0, frames: int | None = None,
              allow_truncated: bool = False, force: bool = False,
              save_series: bool = True, save_band_power: bool = False,
              on_event=None) -> ShardReport:
    """Run this shard's slice of ``videos``, one video at a time.

    ``videos`` is the WHOLE list; the slice is taken here so that every shard is
    handed identical inputs and differs only in ``shard``. Never raises for a
    single video's failure -- inspect the returned report.

    ``on_event(kind, payload)`` is the progress hook (``start`` / ``done`` /
    ``skip`` / ``fail``), kept as a callback so this module stays free of both
    argparse and stdout.
    """
    index, count = shard
    # Derived from the WHOLE list, so a video's output name does not depend on
    # which shard it landed in and every node computes the same mapping.
    out_paths = assign_output_names(videos, out_dir)
    mine = partition(videos, index, count)
    outcomes: list[VideoOutcome] = []
    t_shard = time.perf_counter()

    def emit(kind, **payload):
        if on_event:
            on_event(kind, payload)

    for pos, video in enumerate(mine):
        out_path = out_paths[video]
        emit("start", video=video, pos=pos, total=len(mine))
        t0 = time.perf_counter()
        job_warnings: list[str] = []
        used_clips = False
        try:
            if not os.path.isfile(video):
                raise BatchError(f"no such video: {video}")
            if not force:
                why = skip_reason(out_path, params, cfg, start=start,
                                  frames=frames)
                if why:
                    outcomes.append(VideoOutcome(video, "skipped",
                                                 out_path=out_path))
                    emit("skip", video=video, reason=why)
                    continue

            rep_path = replicate_path or default_replicate_path(video)
            if not rep_path:
                raise BatchError(
                    f"no replicate boxes: pass --replicates, or put a "
                    f".rois.json sidecar next to {video}")
            replicates = load_replicates(rep_path)

            manifest = man_dir = None
            if use_clips:
                mp = find_manifest(video, clip_dir)
                if mp:
                    manifest = read_manifest(mp)
                    man_dir = clip_dir or os.path.dirname(os.path.abspath(mp))
                    used_clips = True
                elif clip_dir:
                    # Asked for clips by naming a clip dir and got none. Silence
                    # here means paying ~25x the decode AND crossing a
                    # provenance boundary (FINDINGS.md section 10) invisibly, so
                    # the fallback is recorded rather than merely happening.
                    job_warnings.append(
                        f"no pre-transcode manifest found for this video in "
                        f"{clip_dir} or beside it; decoded the SOURCE instead "
                        f"of ROI clips (slower, and not the same pixels)")

            res = run_video(video, replicates, cfg, params, regions=regions,
                            manifest=manifest, clip_dir=man_dir,
                            start=start, n=frames,
                            allow_truncated=allow_truncated,
                            replicate_path=rep_path)
            write_result(res, out_path, save_series=save_series,
                         save_band_power=save_band_power)
        except JOB_ERRORS as e:
            # Isolated per video on purpose. The alternative -- abort the shard
            # -- discards the decode already paid for on every video before this
            # one, and a restart repeats all of it.
            o = VideoOutcome(video, "failed", error=f"{type(e).__name__}: {e}",
                             seconds=time.perf_counter() - t0)
            outcomes.append(o)
            emit("fail", video=video, error=o.error)
            continue
        o = VideoOutcome(
            video, "ok", out_path=out_path, seconds=time.perf_counter() - t0,
            n_detected_frames=sum(r.n_detected_frames for r in res.regions),
            covered_frames=res.covered_frames, used_clips=used_clips,
            warnings=job_warnings + list(res.warnings))
        outcomes.append(o)
        emit("done", video=video, outcome=o)

    return ShardReport(shard_index=index, shard_count=count, outcomes=outcomes,
                       total_videos=len(videos),
                       seconds=time.perf_counter() - t_shard)


def write_shard_report(report: ShardReport, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report.to_summary(), f, indent=2, sort_keys=True)
    os.replace(tmp, path)
