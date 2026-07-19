"""How many frames a video actually decodes, as a recorded fact.

**The container's frame count is a claim; the decoder's is a measurement, and
this codebase spent a batch treating one as the other.** ``FINDINGS.md`` section
15 has the full diagnosis; the short form is that
``GX010047c2_02_17_26.MP4`` advertises 11328 video packets (``ffprobe
-count_packets`` agrees) and **decodes 11308**. Twenty frames are unrecoverable
in the source, before anything this project touches it. One file in four in the
corpus is like this, so it is neither a corpus-wide constant nor a quirk to
dismiss.

Two guards read that number and both fail on such a file:

  * the pre-transcode cut compared its clips' true counts against the source's
    *claim* and reported the difference as a re-timed stream (it was not);
  * ``core.batch.run_video`` computes ``available = frame_count - start``, so any
    pass reaching the true end of the video trips the truncation guard.

The second is the corrosive one. It is a false *positive* -- fail-closed, so no
silent false negative -- but the standing workaround for it is
``--allow-truncated``, which is the one habit that disarms the guard everywhere
else. A guard that cries wolf on ordinary footage is a guard that gets switched
off.

**A tolerance would be wrong twice over**: it is a magic constant, and it would
blind the guard to exactly the small re-timings it exists to catch. So the true
count has to become a recorded fact instead.

**There is no free way to learn it.** "Reached EOF" and "stopped early" are the
same event at read time -- a decoder that returns no frame does not say which it
was -- so the only way to count decodable frames is to decode them. This module
therefore does that once and writes the answer to a sidecar, and everything
downstream reads the sidecar rather than re-deriving it.

The layering: :func:`resolve_frame_count` prefers, in order, a Batch I manifest
(which measures the count for free during the cut), then this module's sidecar,
then the container's claim -- and it says which it used. A count from the last
source is marked ``verified=False`` and callers must treat it as an **unverified
upper bound**, because that is exactly what it is.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

# One-way, and it is the right direction: pre-transcode owns the ffmpeg plumbing
# and the file-identity signature, and this module is a second, much smaller
# consumer of both. build_pretranscode writes one of these records as a
# byproduct of the cut via a deliberately function-level import, so that this
# dependency stays acyclic at module scope.
from core.pretranscode import (_ffmpeg, _popen_kwargs, _pump_progress,
                               quick_signature)

FRAMECOUNT_VERSION = 1
FRAMECOUNT_SUFFIX = ".framecount.json"

# Where a resolved count came from, worst-trusted last. Recorded in results so a
# consumer can tell a coverage figure computed against a measured denominator
# from one computed against a container's guess.
SOURCE_MANIFEST = "manifest"
SOURCE_SIDECAR = "sidecar"
SOURCE_CONTAINER = "container"


class FrameCountError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameCountRecord:
    """A measured decodable-frame count for one file.

    ``quick_sig`` is the same head+tail+size signature the pre-transcode manifest
    uses: these sources run to 11 GB and a full re-hash on every read would cost
    more than the guard is worth, while a different or truncated file at the same
    path is caught in milliseconds. A record whose signature no longer matches is
    refused rather than used, because a stale count is worse than no count -- it
    would be *trusted*.
    """
    version: int
    video_path: str
    size_bytes: int
    quick_sig: str
    container_frames: int
    decoded_frames: int
    # How the count was obtained, so a record written as a byproduct of the cut
    # is distinguishable from one produced by a dedicated counting pass. They are
    # the same measurement; the provenance is recorded anyway because "who says
    # so" is the whole subject of this module.
    method: str = "ffmpeg-null-decode"

    def to_meta(self) -> dict:
        return {"version": self.version, "video_path": self.video_path,
                "size_bytes": self.size_bytes, "quick_sig": self.quick_sig,
                "container_frames": self.container_frames,
                "decoded_frames": self.decoded_frames, "method": self.method}

    @staticmethod
    def from_meta(d: dict) -> "FrameCountRecord":
        return FrameCountRecord(
            version=int(d["version"]),
            video_path=str(d["video_path"]),
            size_bytes=int(d["size_bytes"]),
            quick_sig=str(d["quick_sig"]),
            container_frames=int(d["container_frames"]),
            decoded_frames=int(d["decoded_frames"]),
            method=str(d.get("method", "ffmpeg-null-decode")))

    @property
    def undecodable(self) -> int:
        """Frames the container claims that do not decode. Zero on a clean file."""
        return max(0, self.container_frames - self.decoded_frames)


@dataclass(frozen=True)
class ResolvedCount:
    """A frame count plus whether anyone actually measured it.

    ``verified`` is not decoration. A caller sizing a coverage denominator with
    ``verified=False`` is dividing by a number no one has checked, and the
    difference decides whether a shortfall means "the decode stopped early"
    (a real, reportable gap) or "the container over-claimed" (nothing wrong at
    all). The two are indistinguishable without this flag, which is why every
    result records it.
    """
    count: int
    verified: bool
    source: str
    container_frames: int

    @property
    def unverified_excess(self) -> int:
        return max(0, self.container_frames - self.count)


def sidecar_path_for(video_path: str) -> str:
    return os.path.splitext(video_path)[0] + FRAMECOUNT_SUFFIX


def count_decodable_frames(video_path: str, *, total: int = 0, progress=None,
                           should_cancel=None) -> int:
    """Decode ``video_path`` to nothing and report how many frames came out.

    ``-f null`` runs the full decode and discards the output, which is the only
    honest way to answer the question -- and it is not cheap: minutes on an 11 GB
    source. That cost is why the answer is written to a sidecar rather than
    recomputed.

    ``-map 0:v:0`` pins the count to the first video stream. Without it a file
    carrying a second video stream (GoPro sources carry timecode and telemetry
    data streams, and some carry a thumbnail track) would have its frames counted
    across streams, producing a number that is not the frame count of anything.

    No filters, no ``-fps_mode``, and no ``-c:v``: a filter graph or a rate
    conversion could itself insert or drop frames, and this is meant to measure
    the source rather than a transform of it. The null muxer discards everything
    written to it, so naming an encoder would only add a per-frame encode whose
    output goes nowhere.

    ``total`` is the container's claimed length and is used **only** as the
    progress denominator -- never as an answer, which is the entire point of this
    module. Without it nothing is reported, and this decode runs for minutes on a
    multi-GB source: measured 1m36s on the 11 GB ``GX010047c2``, which reads as a
    hang, and a user who kills it there discards the whole pass.
    """
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin",
           "-i", video_path, "-map", "0:v:0",
           "-f", "null", "-", "-progress", "pipe:1", "-nostats"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, **_popen_kwargs())
    try:
        err, stats = _pump_progress(proc, int(total), progress, should_cancel)
    except BaseException:
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass
        raise
    if proc.returncode != 0:
        msg = "".join(err).strip() or "(no stderr)"
        raise FrameCountError(f"ffmpeg could not count {video_path}: {msg[:500]}")
    if stats.frames is None:
        raise FrameCountError(
            f"ffmpeg reported no frame count for {video_path}; it decoded "
            "nothing, or the progress stream was not understood")
    return int(stats.frames)


def build_record(video_path: str, *, container_frames: int,
                 decoded_frames: int | None = None,
                 method: str = "ffmpeg-null-decode",
                 progress=None, should_cancel=None) -> FrameCountRecord:
    """A record for ``video_path``, counting the frames unless told the answer.

    ``decoded_frames`` exists so the pre-transcode cut can hand over the count it
    already measured: ffmpeg reports the frames it encoded in the same pass, so
    the cut gets this for free and re-counting afterwards would be a second full
    decode for a number already in hand.
    """
    if decoded_frames is None:
        # The claim is the progress denominator and nothing else -- it is being
        # measured against, not trusted.
        decoded_frames = count_decodable_frames(
            video_path, total=int(container_frames), progress=progress,
            should_cancel=should_cancel)
    return FrameCountRecord(
        version=FRAMECOUNT_VERSION,
        video_path=os.path.abspath(video_path),
        size_bytes=os.path.getsize(video_path),
        quick_sig=quick_signature(video_path),
        container_frames=int(container_frames),
        decoded_frames=int(decoded_frames),
        method=method)


def write_record(rec: FrameCountRecord, path: str | None = None) -> str:
    """Write ``rec`` beside its video (or at ``path``). Atomic. Returns the path."""
    path = path or sidecar_path_for(rec.video_path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec.to_meta(), f, indent=2, sort_keys=True)
    os.replace(tmp, path)   # atomic, so a killed count leaves no half-written claim
    return path


def read_record(path: str) -> FrameCountRecord:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    ver = int(d.get("version", 0))
    if ver != FRAMECOUNT_VERSION:
        raise FrameCountError(
            f"frame-count record version {ver} != {FRAMECOUNT_VERSION}; re-count")
    return FrameCountRecord.from_meta(d)


def load_sidecar(video_path: str) -> FrameCountRecord | None:
    """The sidecar for ``video_path``, or ``None`` if there is no usable one.

    Every failure path returns ``None`` rather than raising: a missing, stale,
    unreadable or future-versioned record means "nobody has measured this file",
    which is a state the caller already has to handle. Raising instead would make
    a corrupt sidecar fail a job that would otherwise have run correctly against
    an unverified count.
    """
    p = sidecar_path_for(video_path)
    if not os.path.isfile(p):
        return None
    try:
        rec = read_record(p)
    except (OSError, ValueError, KeyError, FrameCountError):
        return None
    # Identity, not existence -- the same rule resume follows in core/shard.py. A
    # sidecar left behind by a since-replaced file would otherwise be trusted
    # over the container, which is worse than having no sidecar at all.
    try:
        if os.path.getsize(video_path) != rec.size_bytes:
            return None
        if quick_signature(video_path) != rec.quick_sig:
            return None
    except OSError:
        return None
    return rec


def _manifest_describes(manifest, video_path: str) -> bool:
    """Whether ``manifest`` was cut from the bytes currently at ``video_path``.

    Path equality is not enough, and matching only on it would make the
    best-trusted source the least-checked one. A source re-exported or
    re-stabilized in place keeps its path, so the stale manifest would still name
    it -- and its frame count, belonging to a different file, would be returned
    as *verified* and used as the coverage denominator. ``load_sidecar`` already
    checks size and signature for exactly this reason; this is the same rule
    applied to the tier above it.

    ``run_video`` does not call ``verify_manifest`` (that is the extraction
    path's job, and it needs the clip directory), so this cannot lean on it.
    Attributes are read defensively because a caller may pass any manifest-shaped
    object; a manifest that cannot prove identity simply is not used.
    """
    try:
        if os.path.abspath(getattr(manifest, "source_path", "")) != \
                os.path.abspath(video_path):
            return False
        size = getattr(manifest, "source_size", 0)
        if size and os.path.getsize(video_path) != int(size):
            return False
        sig = getattr(manifest, "quick_sig", "")
        if sig and quick_signature(video_path) != sig:
            return False
    except (OSError, TypeError, ValueError):
        return False
    return True


def resolve_frame_count(video_path: str, container_frames: int, *,
                        manifest=None) -> ResolvedCount:
    """The best available decodable-frame count for ``video_path``.

    Preference order, best first:

      1. **A Batch I manifest.** The cut decodes the whole source anyway and
         ffmpeg reports the frames it encoded, so the count is measured there for
         free and lands in ``Manifest.frame_count``. Only used when the manifest
         actually describes this file -- a manifest for a different source says
         nothing about this one.
      2. **This module's sidecar**, written by ``python -m cli.count_frames``.
      3. **The container's claim**, marked ``verified=False``. It is an upper
         bound: a container can advertise packets that do not decode, but frames
         cannot decode out of packets that are not there.

    The claim is never *rejected* for disagreeing with a measurement -- on
    ``GX010047c2`` it over-claims by 20 and both numbers are correct answers to
    different questions. It is simply not used when something better exists.
    """
    container = max(0, int(container_frames))
    if manifest is not None and _manifest_describes(manifest, video_path):
        n = int(getattr(manifest, "frame_count", 0) or 0)
        if n > 0:
            return ResolvedCount(count=n, verified=True, source=SOURCE_MANIFEST,
                                 container_frames=int(
                                     getattr(manifest, "container_frame_count",
                                             container) or container))
    rec = load_sidecar(video_path)
    if rec is not None and rec.decoded_frames > 0:
        return ResolvedCount(count=rec.decoded_frames, verified=True,
                             source=SOURCE_SIDECAR,
                             container_frames=rec.container_frames)
    return ResolvedCount(count=container, verified=False,
                         source=SOURCE_CONTAINER, container_frames=container)


__all__ = ["FRAMECOUNT_VERSION", "FRAMECOUNT_SUFFIX", "FrameCountError",
           "FrameCountRecord", "ResolvedCount", "build_record",
           "count_decodable_frames", "load_sidecar", "read_record",
           "resolve_frame_count", "sidecar_path_for", "write_record",
           "SOURCE_MANIFEST", "SOURCE_SIDECAR", "SOURCE_CONTAINER"]
