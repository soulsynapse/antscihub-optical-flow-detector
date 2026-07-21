"""N-worker throughput for the headless batch path (T24, Batch J).

The figure Batch J actually needs. ``FINDINGS.md`` section 12 measured a single
process on ``rep3_intermittent_crop`` -- one replicate on a 512x512 working ROI --
and said in as many words not to extrapolate it to a node count. This measures
what a node really does: N concurrent ``cli.detect`` processes over real
multi-replicate footage at the scale-1.0 default.

**Why this is not a re-run of the section 1 multi-process table.** That one is
flat past ~2-4 workers, and it is a *decode* measurement -- one ffmpeg already
spreads decode across the whole box, so a second one has nothing left to take.
Under the scale-1.0 default the pass is math-bound instead (section 1's own
reversal), and the math is elementwise numpy in one process. The two tables
therefore measure different ceilings and the older one must not be quoted for
this question.

Method, and the three ways it could lie:

  * **Each worker gets its own frame window** (``--start i*stride``), not the
    same one. N workers over identical ranges share the OS page cache and read
    one video's bytes N times from RAM, which flatters every N > 1. Offsets cost
    nothing to use: seeking to frame 8000 measured 8.43 s against 8.60 s at
    frame 0, i.e. inside the noise.
  * **Two throughputs are reported, and they bracket the truth.** ``wall`` is
    total frames over the wall clock of the whole fan-out and includes each
    worker's interpreter+numpy startup, which is real cost an operator pays but
    is amortized over a real job's thousands of frames rather than this
    benchmark's hundreds. ``extract`` divides by the in-process extraction time,
    which excludes startup but still contains all contention, since it is
    measured while every other worker is running. A real long job approaches
    ``extract``; a short one gets ``wall``.
  * **Detection is not in the frame count's denominator by accident.** It is
    ~0.5% of the pass (0.047 s against 8.60 s on 7 replicates), so extraction
    throughput and job throughput are the same number here. That stops being
    true if a future channel makes the detector expensive.

Two known limits of the method, left in deliberately and stated so the numbers
are not read as tighter than they are. Both were found by review after the
measurement in ``FINDINGS.md`` section 14 was taken, and neither is worth
invalidating that measurement to fix, because both are quantified below as small
*for a math-bound pass*:

  * **Each level covers a different span of the video.** Offsets are ``i *
    stride``, so N=1 measures only the first ``frames`` and N=16 measures
    sixteen times that span. Speedup is against the N=1 baseline, so the
    baseline is one short segment at the head of the file. This is near-harmless
    at scale 1.0, where decode is ~0% of the pass and the tensor math costs the
    same per pixel regardless of content -- but **re-running this at low scale,
    where section 1 says the pass is decode-bound, would confound the speedup
    column**, because H.264 decode cost does vary with content. Fix then, not
    now: spread every level's workers across one fixed window.
  * **``extract_fps`` is a slight overestimate, in the direction that flatters
    scaling.** It divides by ``max(extract_seconds)``, but workers do not begin
    extracting at the same instant -- each pays its interpreter and numpy import
    first, staggered by spawn order and disk contention. The true window during
    which N extractions overlap is at least that long. The gap is visible in the
    output as ``wall_s`` minus ``slowest_extract_s`` (74.4 vs 73.3 at N=16, so
    ~1.5%) and grows with N. Closing it needs each worker to report absolute
    extraction start/end timestamps rather than a duration.

Usage:
    .venv\\Scripts\\python.exe scripts/bench_worker_scaling.py VIDEO \\
        --params tuned.json --workers 1,2,4,8,16 --frames 600
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _worker_cmd(video, params, out, start, frames, manifest, clip_dir):
    cmd = [sys.executable, "-m", "cli.detect", video,
           "--params", params, "--start", str(start), "--frames", str(frames),
           "--no-series", "--out", out, "--quiet"]
    if manifest:
        cmd += ["--manifest", manifest]
    if clip_dir:
        cmd += ["--clip-dir", clip_dir]
    return cmd


def run_level(video, params, n_workers, frames, stride, out_dir,
              manifest=None, clip_dir=None):
    """One fan-out level. Returns the level's record, or raises on any worker
    failure -- a level where some workers died is not a throughput datum, and
    silently averaging over the survivors would report a *higher* number for a
    *broken* run."""
    procs, outs = [], []
    t0 = time.perf_counter()
    for i in range(n_workers):
        out = os.path.join(out_dir, f"w{n_workers}_{i}.json")
        outs.append(out)
        # stderr is collected per worker into its own file rather than a pipe.
        # Draining N pipes one at a time deadlocks: the parent blocks in
        # procs[0].communicate() while workers 1..N-1 keep writing, and
        # core/timing.py logs a span line per pass to stderr that --quiet does
        # NOT suppress (only the progress printer is gated). At ~8 lines per
        # worker that stays under the pipe buffer, but the volume scales with
        # region count and any warning storm, and the failure is a silent hang.
        errf = open(os.path.join(out_dir, f"w{n_workers}_{i}.err"), "w+b")
        procs.append((subprocess.Popen(
            _worker_cmd(video, params, out, i * stride, frames,
                        manifest, clip_dir),
            cwd=REPO, stdout=subprocess.DEVNULL, stderr=errf), errf))
    errs = []
    for i, (p, errf) in enumerate(procs):
        rc = p.wait()
        errf.seek(0)
        err = errf.read()
        errf.close()
        if rc != 0:
            errs.append(f"worker {i} exited {rc}: "
                        f"{err.decode(errors='replace').strip()[-400:]}")
    wall = time.perf_counter() - t0
    if errs:
        raise RuntimeError(f"N={n_workers}: " + "; ".join(errs))

    # Coverage is re-checked here even though cli.detect already fails a
    # truncated pass: a worker that quietly examined fewer frames than asked
    # would divide real seconds by frames nobody decoded, and bias the result
    # toward "it scales" -- the dangerous direction for a node-count estimate.
    extract_s, covered = [], 0
    for out in outs:
        with open(out, encoding="utf-8") as f:
            d = json.load(f)
        cov = d["coverage"]
        if cov["truncated"] or cov["covered_frames"] != frames:
            raise RuntimeError(
                f"N={n_workers}: a worker covered {cov['covered_frames']} of "
                f"{frames} frames; throughput would be overstated")
        covered += cov["covered_frames"]
        extract_s.append(d["timing"]["extract_seconds"])

    slowest = max(extract_s)
    return {
        "workers": n_workers,
        "frames_each": frames,
        "total_frames": covered,
        "wall_s": round(wall, 2),
        "wall_fps": round(covered / wall, 1),
        # Frames over the slowest worker's extraction: the fan-out is only done
        # when its last member is, so the slowest -- not the mean -- is what a
        # job runner waits on.
        "extract_fps": round(covered / slowest, 1),
        "slowest_extract_s": round(slowest, 2),
        "fastest_extract_s": round(min(extract_s), 2),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--params", required=True)
    p.add_argument("--workers", default="1,2,4,8,16",
                   help="comma-separated worker counts")
    p.add_argument("--frames", type=int, default=600,
                   help="frames per worker")
    p.add_argument("--stride", type=int,
                   help="frame offset between workers "
                        "(default: --frames, i.e. non-overlapping windows)")
    p.add_argument("--manifest", help="pre-transcode manifest (Batch I clips)")
    p.add_argument("--clip-dir")
    p.add_argument("--fps", type=float,
                   help="source fps, to report a realtime multiple")
    p.add_argument("--out", help="write the table as JSON here")
    a = p.parse_args(argv)

    levels = [int(x) for x in a.workers.split(",") if x.strip()]
    # A zero or negative level would otherwise reach max() on an empty list and
    # die with "max() arg is an empty sequence" -- after the level had already
    # been announced as running, which reads as a crash mid-benchmark.
    bad = [n for n in levels if n < 1]
    if bad:
        p.error(f"--workers must all be >= 1, got {bad}")
    if not levels:
        p.error("--workers is empty")

    # Resolved against the CALLER's cwd, before the workers (which run with
    # cwd=REPO so that `-m cli.detect` resolves) get them. Left relative, a
    # video or params path would silently resolve against the repo root and
    # every worker would exit 2 -- correct only when the benchmark happens to be
    # launched from the repo root, which is how it was first run.
    video = os.path.abspath(a.video)
    params = os.path.abspath(a.params)
    manifest = os.path.abspath(a.manifest) if a.manifest else None
    clip_dir = os.path.abspath(a.clip_dir) if a.clip_dir else None

    stride = a.stride or a.frames
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in levels:
            print(f"N={n} ...", flush=True)
            row = run_level(video, params, n, a.frames, stride, tmp,
                            manifest, clip_dir)
            if a.fps:
                row["wall_realtime_x"] = round(row["wall_fps"] / a.fps, 2)
                row["extract_realtime_x"] = round(row["extract_fps"] / a.fps, 2)
            rows.append(row)
            print("   " + json.dumps(row), flush=True)

    base = rows[0]["extract_fps"] if rows else 0
    print("\n| workers | wall fps | extract fps | speedup vs N=1 | slowest worker |")
    print("|---|---|---|---|---|")
    for r in rows:
        sp = f"{r['extract_fps'] / base:.2f}x" if base else "-"
        print(f"| {r['workers']} | {r['wall_fps']} | {r['extract_fps']} | "
              f"{sp} | {r['slowest_extract_s']} s |")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"video": a.video, "frames_each": a.frames,
                       "levels": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
