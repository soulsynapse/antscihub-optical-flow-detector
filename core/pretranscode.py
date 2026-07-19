"""ROI pre-transcode: cut a source once into per-replicate clips plus a manifest.

The live path already crops to replicate boxes, but H.264/HEVC decode is
whole-frame: the ``crop`` in :class:`core.video.ReplicateVideoSource`'s filter
graph runs *after* decode, so every pass still pays for all 15.9 M pixels of a
5.3K frame to keep the ~8% the replicates own. Pre-transcoding makes the stored
file small, so later decodes genuinely decode fewer pixels. Measured on
``GX010047c2`` (5312x2988 HEVC), 300 frames of one replicate:

    source HEVC + crop (today)   4.03 s    1.0x
    ffv1 lossless clip           0.16 s   25.4x

That ~25x is the entire point of this module, and it is **independent of the
codec** -- a lossy CRF 18 clip decoded at 28.6x, only 12% better. Decode cost
tracks pixels, not bytes. So there is no fidelity/speed trade here to agonize
over: take the lossless one and pay in storage.

**Quality is a recorded choice, not a constant.** Measured on the same clip,
luma error against a bit-exact reference and full-clip size for all 6 replicates:

    quality       grey-level RMS   clips     total disk
    lossless (ffv1)      0.000     131%        2.31x
    high    (crf 12)     1.041      24%        1.24x   <- default
    standard(crf 18)     1.540      11%        1.11x

``lossless`` names the **encode**, not the whole route. A clip stores 8-bit gray,
while the live crop converts the source's yuv luma straight to ``gray16le`` and
keeps the sub-8-bit precision the limited->full range conversion produces (81% of
luma values are not multiples of 257). So even a lossless clip differs from the
live crop by up to 0.494 grey levels of pure quantization -- below the 0.364 RMS
the live ROI path itself carries against a float reference. Nothing here needs
fixing, but no consumer may assume clip == live, and a test asserting it is
wrong. See ``FINDINGS.md`` section 10 for how that reaches each channel.

The default is ``high``. Bit-exactness was considered and deliberately not made
the default: these sources are *already* a lossy re-encode (the ``stab_`` files
are a stabilization generation removed from the sensor), so preserving an
already-degraded intermediate byte-for-byte buys less than it appears to, and it
costs 131% of the source in storage because lossless must encode inherited
compression noise verbatim -- noise does not compress. A result that flips
between CRF 12 and lossless was fragile to begin with, and is better surfaced
than hidden.

What survives that argument, and is why ``quality`` is recorded in the manifest
and folded into :meth:`Manifest.provenance_key`: the ``change`` channel is
``<I_t^2>``, squared frame differencing, and it is the detection default. Lossy
*inter-frame* coding perturbs exactly the frame-to-frame quantity that channel
measures, rather than degrading it generically. So the setting must never be
silently variable across clips being compared, and clips cut at different
quality must not compare as equal. Whether a given behaviour is robust across
these settings is now measurable end to end (extraction consumes the manifest via
``channel_source.live_channel_source``) -- measure it per behaviour rather than
assuming either way.

Lossless cost is also distributed counter-intuitively, worth not re-deriving:

    rep6-no-flying                 3.695 bpp   <- most expensive
    rep2-backlit-flying-whole-time 2.981 bpp
    rep1-mostly-flying-some-pause  1.921 bpp   <- among the cheapest

Motion is *cheaper*, because motion blur is smooth and compresses well, while a
still frame is mostly per-pixel noise. Do not estimate a corpus from one
replicate: rep3 alone reads 1.602 bpp and extrapolates to 86% of source, a third
under the true 131%.

**Crop only; never scale here.** Cropping discards pixels no replicate owns, so
it cannot change any detection result. Rescaling would bake a sensitivity
decision into an artifact that costs a full re-decode to regenerate, which the
governing principle forbids. The manifest records the resolved scale and the rule
that produced it so clips cut under different settings are never silently
compared, but the pixels themselves are always full resolution.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field, asdict

import cv2

from core.replicates import build_layout, geometry_hash, validate_replicates


# 2: -fps_mode passthrough on every output, plus per-clip frame_count/size_bytes.
# Bumped rather than defaulted-in because the flag can change WHICH FRAMES a clip
# holds on a source whose timestamps wobble -- a version-1 clip is not
# necessarily frame-aligned with its source, and provenance_key carries this
# field precisely so old and new cuts cannot compare as equal. Old manifests are
# refused by read_manifest, which is the intended outcome: re-cut.
PRETRANSCODE_VERSION = 2
MANIFEST_SUFFIX = ".pretranscode.json"

# The pipeline reads luma and discards chroma, so encoding gray is free
# scientifically and removes any question of which plane a consumer reads.
#
# On the lossless entry: -g 1 makes every frame a keyframe. FFV1 is intra-only
# for pixels regardless, but its adaptive context model resets on keyframes, so a
# larger GOP lets the model carry across frames -- worth ~4% size, at the cost of
# frames that are no longer independently decodable, which would surrender exact
# seeking. -context 1 is the part of that gain available without the trade
# (~0.6%). -slicecrc 1 costs nothing and makes silent corruption detectable.
#
# On the lossy entries: -g 1 is NOT used, because inter-frame prediction is where
# the compression comes from. Seeks therefore land on keyframe boundaries and
# decode forward, exactly as the source does today -- see the seek note in
# ``probe_source`` about why the rate must stay rational.
QUALITY_PRESETS: dict[str, dict] = {
    "lossless": {
        "args": ["-c:v", "ffv1", "-level", "3", "-g", "1", "-context", "1",
                 "-slices", "4", "-slicecrc", "1", "-pix_fmt", "gray"],
        "codec": "ffv1", "lossless": True,
        # Measured: 0.000 RMS, 131% of source, 2.31x total disk.
        "rms": 0.0,
    },
    "high": {
        "args": ["-c:v", "libx264", "-crf", "12", "-preset", "slow",
                 "-pix_fmt", "gray"],
        "codec": "libx264-crf12", "lossless": False,
        # Measured: 1.041 grey-levels RMS, 24% of source, 1.24x total disk.
        "rms": 1.041,
    },
    "standard": {
        "args": ["-c:v", "libx264", "-crf", "18", "-preset", "slow",
                 "-pix_fmt", "gray"],
        "codec": "libx264-crf18", "lossless": False,
        # Measured: 1.540 grey-levels RMS, 11% of source, 1.11x total disk.
        "rms": 1.540,
    },
}

DEFAULT_QUALITY = "high"


class PretranscodeError(RuntimeError):
    pass


class PretranscodeCancelled(Exception):
    """Raised when ``should_cancel`` asked for a stop.

    Its own type rather than ``KeyboardInterrupt``: a caller must be able to tell
    a user-requested cancel from a real interpreter interrupt, and a genuine
    Ctrl-C during a multi-minute transcode should not be reported as a clean
    cancel.
    """


@dataclass(frozen=True)
class ClipEntry:
    """One replicate's clip, and the geometry that produced it."""
    replicate_id: int
    label: str
    frac: tuple[float, float, float, float]
    source_box: tuple[int, int, int, int]   # x0, y0, x1, y1 in source pixels
    width: int
    height: int
    filename: str
    # Frames actually written, probed after the cut, and the file's size on disk.
    #
    # Both exist to make a short clip detectable BEFORE a pass reads it. Length is
    # what a truncated clip gets wrong, and it was the one property nothing
    # recorded: verify_manifest checked that a clip existed, so a clip cut short
    # by a crash or a full disk was rediscovered at decode time, per pass,
    # forever. ``frame_count`` is checked once at the cut against the source's
    # (which catches a CFR conversion inserting or dropping frames); ``size_bytes``
    # is what verify_manifest re-checks on every pass, because an os.stat costs
    # microseconds while re-probing 6 clips with VideoCapture measured 41.8 ms --
    # on a surface that starts a pass per knob edit. A truncation cannot shorten a
    # file without changing its size, so the cheap check is not the weaker one.
    #
    # Defaulted so a manifest written before these existed still loads; absent
    # values simply skip the checks rather than failing an older cut outright.
    frame_count: int = 0
    size_bytes: int = 0

    def to_meta(self) -> dict:
        return {
            "id": self.replicate_id,
            "label": self.label,
            "frac": list(self.frac),
            "source_box": list(self.source_box),
            "width": self.width,
            "height": self.height,
            "filename": self.filename,
            "frame_count": self.frame_count,
            "size_bytes": self.size_bytes,
        }

    @staticmethod
    def from_meta(d: dict) -> "ClipEntry":
        return ClipEntry(
            replicate_id=int(d["id"]),
            label=str(d.get("label", f"rep{d['id']}")),
            frac=tuple(float(v) for v in d["frac"]),
            source_box=tuple(int(v) for v in d["source_box"]),
            width=int(d["width"]),
            height=int(d["height"]),
            filename=str(d["filename"]),
            frame_count=int(d.get("frame_count", 0)),
            size_bytes=int(d.get("size_bytes", 0)),
        )


@dataclass(frozen=True)
class Manifest:
    """What a set of clips is, and what it may be compared against."""
    version: int
    source_path: str
    source_size: int
    source_sha256: str
    quick_sig: str
    src_width: int
    src_height: int
    # Rational, NEVER a float. FINDINGS.md section 3 trap 2: a seek that divides
    # a frame index by a rounded fps lands progressively earlier -- 24.0 for
    # 24000/1001 is 3 frames early by frame 11000, silently, yielding a window of
    # the right length from the wrong place. Storing num/den means a consumer can
    # reconstruct the exact rate rather than inheriting someone's rounding.
    fps_num: int
    fps_den: int
    frame_count: int
    geometry_hash: str
    # The scale this project would have resolved, and the rule that produced it.
    # Recorded, deliberately NOT applied: clips are always full-resolution crops.
    # Its purpose is to make clips cut under different settings non-comparable
    # rather than silently comparable.
    resolved_scale: float
    scale_rule: str
    codec: str
    lossless: bool
    # The named preset, kept alongside the resolved codec string so a manifest
    # says both what was asked for and what was actually run.
    quality: str
    # Measured luma error for this preset, in grey levels, against a bit-exact
    # reference. Carried so a consumer can state the artifact's own error budget
    # rather than re-deriving it -- and so a future preset cannot quietly change
    # what an old manifest claimed.
    quality_rms: float
    clips: tuple[ClipEntry, ...] = field(default_factory=tuple)

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    def clip_for(self, replicate_id: int) -> ClipEntry:
        for c in self.clips:
            if c.replicate_id == int(replicate_id):
                return c
        raise KeyError(f"no clip for replicate {replicate_id}")

    def to_meta(self) -> dict:
        d = asdict(self)
        d["clips"] = [c.to_meta() for c in self.clips]
        return d

    @staticmethod
    def from_meta(d: dict) -> "Manifest":
        return Manifest(
            version=int(d["version"]),
            source_path=str(d["source_path"]),
            source_size=int(d["source_size"]),
            source_sha256=str(d["source_sha256"]),
            quick_sig=str(d["quick_sig"]),
            src_width=int(d["src_width"]),
            src_height=int(d["src_height"]),
            fps_num=int(d["fps_num"]),
            fps_den=int(d["fps_den"]),
            frame_count=int(d["frame_count"]),
            geometry_hash=str(d["geometry_hash"]),
            resolved_scale=float(d["resolved_scale"]),
            scale_rule=str(d["scale_rule"]),
            codec=str(d["codec"]),
            lossless=bool(d["lossless"]),
            quality=str(d["quality"]),
            quality_rms=float(d["quality_rms"]),
            clips=tuple(ClipEntry.from_meta(c) for c in d.get("clips", ())),
        )

    def provenance_key(self) -> str:
        """Stable identity for everything that can change a clip's pixels.

        The flow cache key does not include the decoder or its bit depth
        (``FINDINGS.md`` section 3 trap 3), which is why caches built across that
        change had to be rebuilt by hand. Reading from a pre-transcoded clip is a
        third provenance axis on top of that one, so a consumer that caches
        anything derived from these clips must fold this in -- otherwise a result
        computed from a live crop and one computed from a clip cut under a
        different rule compare as equal.

        ``quality`` is in the key for the same reason and a sharper one: at any
        setting below lossless the clip's pixels differ from the source's, and
        the ``change`` channel measures precisely the frame-to-frame quantity
        that lossy inter-frame coding perturbs. Two runs at different quality are
        different measurements and must never share a cache entry.
        """
        blob = json.dumps({
            "version": self.version,
            "source_sha256": self.source_sha256,
            "geometry_hash": self.geometry_hash,
            "resolved_scale": round(self.resolved_scale, 12),
            "scale_rule": self.scale_rule,
            "codec": self.codec,
            "lossless": self.lossless,
            "quality": self.quality,
        }, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha1(blob).hexdigest()[:16]


def resolve_quality(quality: str) -> dict:
    """The preset for ``quality``, or a PretranscodeError naming the valid ones."""
    preset = QUALITY_PRESETS.get(quality)
    if preset is None:
        raise PretranscodeError(
            f"unknown quality {quality!r}; expected one of "
            f"{sorted(QUALITY_PRESETS)}")
    return preset


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FileNotFoundError("ffmpeg is not available on PATH")
    return exe


def _popen_kwargs() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def quick_signature(path: str) -> str:
    """Cheap identity: size plus the head and tail of the file.

    A consumer re-checks its source on every extraction, and the sources here run
    to 11 GB -- a full re-hash each time would cost more than the decode this
    module exists to save. Head+tail+size catches the realistic failure (a
    different or truncated file at the same path) in milliseconds. The full
    sha256 is recorded separately for provenance, where the cost is paid once.
    """
    span = 4 << 20
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(span))
        if size > span:
            f.seek(max(0, size - span))
            h.update(f.read(span))
    return h.hexdigest()[:32]


def full_sha256(path: str, *, progress=None, chunk: int = 8 << 20) -> str:
    """Whole-file digest, for the manifest's provenance record.

    Costs ~20 s on an 11 GB source, which is negligible beside the transcode it
    accompanies but far too slow to repeat on every read -- hence
    :func:`quick_signature` for the hot path.
    """
    size = max(1, os.path.getsize(path))
    h = hashlib.sha256()
    done = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            done += len(b)
            if progress is not None:
                progress(done / size)
    return h.hexdigest()


def probe_source(path: str) -> tuple[int, int, int, int, int]:
    """``(width, height, fps_num, fps_den, frame_count)`` from the container.

    The frame rate is recovered as an exact rational. OpenCV reports a float, so
    a rate like 24000/1001 arrives as 23.976023976..., and rounding it is the
    documented seek trap. Common broadcast rates are recognized exactly; anything
    else falls back to a rational approximation of the reported float, which is
    still better than storing the float and re-deriving it at each call site.
    """
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise PretranscodeError(f"cannot open {path}")
        w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        raise PretranscodeError(f"{path} reports no usable frame size")
    if fps <= 0:
        raise PretranscodeError(f"{path} reports no frame rate")
    num, den = _as_rational(fps)
    return w, h, num, den, max(0, n)


def _clip_frame_count(path: str) -> int:
    """Frames in a written clip, from the container.

    Split out from :func:`probe_source` rather than reusing it because a clip has
    no rate to recover and no geometry worth re-deriving -- and because
    ``probe_source`` raises when a rate is missing, which would turn a legitimate
    single-frame or rateless clip into a cut failure.
    """
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise PretranscodeError(f"cannot open clip {path}")
        return max(0, int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    finally:
        cap.release()


def _as_rational(fps: float) -> tuple[int, int]:
    """Exact rational for a reported frame rate, preferring known NTSC rates."""
    for base in (24, 25, 30, 48, 50, 60, 100, 120):
        # The 1000/1001 family: 23.976, 29.97, 59.94, ...
        if abs(fps - base * 1000.0 / 1001.0) < 1e-4:
            return base * 1000, 1001
        if abs(fps - float(base)) < 1e-6:
            return base, 1
    from fractions import Fraction
    fr = Fraction(fps).limit_denominator(100000)
    return fr.numerator, fr.denominator


def clip_filename(stem: str, replicate_id: int) -> str:
    return f"{stem}__rep{int(replicate_id):02d}.mkv"


def manifest_path_for(video_path: str, out_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(out_dir, stem + MANIFEST_SUFFIX)


def build_filter_graph(tiles) -> tuple[str, list[str]]:
    """One decode, N cropped gray outputs. Returns ``(graph, labels)``.

    Splitting inside a single filter graph is what makes this a *single* decode
    of the source; running ffmpeg once per replicate would multiply the one cost
    this module exists to pay down.
    """
    n = len(tiles)
    if n == 0:
        raise PretranscodeError("no replicate tiles to transcode")
    parts = []
    labels = []
    if n == 1:
        parts.append("[0:v]null[s0];")
    else:
        parts.append("[0:v]split=" + str(n)
                     + "".join(f"[s{i}]" for i in range(n)) + ";")
    for i, tile in enumerate(tiles):
        x0, y0, x1, y1 = tile.source_box
        # exact=1 forbids ffmpeg from rounding an odd-coordinate crop to an even
        # boundary. Without it an odd x0/y0 shifts the window by up to a whole
        # source pixel, and that sub-pixel offset -- inconsistent frame to frame
        # -- is read by dense flow as real translation. Same reasoning as
        # ReplicateVideoSource; the two must stay in agreement or a clip and a
        # live crop of the same box would not hold the same pixels.
        parts.append(f"[s{i}]crop={x1-x0}:{y1-y0}:{x0}:{y0}:exact=1,"
                     f"format=gray[c{i}];")
        labels.append(f"[c{i}]")
    return "".join(parts).rstrip(";"), labels


def clip_command(video_path: str, tiles, out_paths: list[str],
                 quality: str = DEFAULT_QUALITY) -> list[str]:
    """The full ffmpeg invocation: one input, one output per replicate.

    Note there is no ``scale`` in the graph and therefore no ``gray16le``. The
    trap that ``format=gray16le`` must follow ``scale`` (``FINDINGS.md`` section 3
    trap 1) applies where scaling happens, which is now downstream at consumption
    time -- these clips are full-resolution 8-bit crops, so a consumer scaling
    them into gray16 reproduces the existing high-precision path rather than
    inheriting a decision made here. Cropping is the only geometric operation
    this module performs.

    ``-fps_mode passthrough`` on every output, and it is load-bearing. FFmpeg's
    default output mode is ``cfr``, which *duplicates or drops* frames to force a
    constant rate on an input whose timestamps wobble. That would silently break
    the invariant every consumer of these clips rests on -- clip frame N is
    source frame N. ``ClipAtlasSource`` seeks by dividing a frame index by the
    manifest's rate, and the manifest records the SOURCE's frame count, so a cut
    that inserted or dropped even one frame yields a window of the right length
    from the wrong place, drifting further the deeper into the clip you look.
    Exactly the failure FINDINGS.md section 3 trap 2 describes for a rounded fps,
    reached by a different road. No effect on a genuinely CFR source, which is
    why a ``testsrc`` fixture cannot catch its absence; the ``stab_`` sources are
    a stabilization generation removed from the sensor and are not guaranteed to
    be one. Both atlas decoders already pass this flag on their own outputs --
    it was missing only here, at the one place that writes a file.
    """
    preset = resolve_quality(quality)
    graph, labels = build_filter_graph(tiles)
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin",
           "-i", video_path, "-filter_complex", graph]
    for label, out in zip(labels, out_paths):
        cmd += ["-map", label, "-fps_mode", "passthrough", *preset["args"], out]
    cmd += ["-progress", "pipe:1", "-nostats"]
    return cmd


def build_pretranscode(video_path: str, replicates: list[dict], out_dir: str, *,
                       resolved_scale: float = 1.0,
                       scale_rule: str = "scale-1.0-default",
                       block_size: int = 64,
                       quality: str = DEFAULT_QUALITY,
                       progress=None, should_cancel=None,
                       overwrite: bool = False) -> Manifest:
    """Cut ``video_path`` into per-replicate clips in ``out_dir`` and write a manifest.

    ``progress`` receives a 0..1 fraction; ``should_cancel`` is polled and, when
    it returns true, the ffmpeg child is terminated and partial clips removed --
    a half-written clip that looked complete would be indistinguishable from a
    real one to every later pass.
    """
    preset = resolve_quality(quality)
    validate_replicates(replicates)
    if not os.path.isfile(video_path):
        raise PretranscodeError(f"no such video: {video_path}")
    os.makedirs(out_dir, exist_ok=True)

    w, h, fps_num, fps_den, frame_count = probe_source(video_path)
    # block_size does not affect the pixels written; build_layout is used purely
    # for its source_box rounding, so a clip's crop is identical to the box the
    # live path would have read. Passing scale 1.0 keeps work_* == box size.
    layout = build_layout(replicates, w, h, 1.0, block_size)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_paths = [os.path.join(out_dir, clip_filename(stem, t.replicate_id))
                 for t in layout.tiles]
    existing = [p for p in out_paths if os.path.exists(p)]
    if existing and not overwrite:
        raise PretranscodeError(
            f"{len(existing)} clip(s) already exist in {out_dir}; "
            "pass overwrite=True to replace them")

    cmd = clip_command(video_path, layout.tiles, out_paths, quality)
    if overwrite:
        cmd.insert(1, "-y")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, **_popen_kwargs())
    # The transcode owns 0..0.98 and the provenance hash the remainder, so the
    # reported fraction never runs backwards between the two phases.
    frame_progress = (None if progress is None
                      else (lambda f: progress(0.98 * f)))
    try:
        err_lines = _pump_progress(proc, frame_count, frame_progress,
                                   should_cancel)
    except BaseException:
        _kill(proc)
        _cleanup(out_paths)
        raise
    if proc.returncode != 0:
        err = "".join(err_lines).strip() or "(no stderr)"
        _cleanup(out_paths)
        raise PretranscodeError(f"ffmpeg failed: {err[:500]}")

    missing = [p for p in out_paths if not os.path.exists(p)]
    if missing:
        _cleanup(out_paths)
        raise PretranscodeError(f"ffmpeg wrote no clip for {missing}")

    # Every clip must hold exactly as many frames as the source. This is the one
    # check that catches a frame-rate conversion having quietly re-timed the cut:
    # with -fps_mode passthrough it should be unreachable, but the consequence of
    # being wrong (clip frame N is no longer source frame N, so every later seek
    # lands in the wrong place) is silent and unbounded, so it is verified rather
    # than assumed. Both counts come from the same CAP_PROP_FRAME_COUNT
    # estimator, so they are comparable even where that estimator is approximate.
    # Paid once per cut, against a transcode measured in minutes.
    clip_frames = []
    for p in out_paths:
        n_clip = _clip_frame_count(p)
        if n_clip != frame_count:
            _cleanup(out_paths)
            raise PretranscodeError(
                f"{os.path.basename(p)} holds {n_clip} frames but the source "
                f"has {frame_count}; the cut re-timed the stream, so clip frame "
                "N is no longer source frame N and every later seek would land "
                "in the wrong place")
        clip_frames.append(n_clip)

    man = Manifest(
        version=PRETRANSCODE_VERSION,
        source_path=os.path.abspath(video_path),
        source_size=os.path.getsize(video_path),
        # Reported, not silent: this reads the whole source (~20-40 s on the
        # 11.7 GB clips), and a bar frozen at its last frame value reads as a
        # hang -- a user who kills it there discards a finished transcode. The
        # tail of the range is reserved for it so the reported fraction stays
        # monotonic.
        source_sha256=full_sha256(
            video_path,
            progress=None if progress is None
            else (lambda f: progress(0.98 + 0.02 * f))),
        quick_sig=quick_signature(video_path),
        src_width=w, src_height=h,
        fps_num=fps_num, fps_den=fps_den, frame_count=frame_count,
        geometry_hash=geometry_hash(replicates),
        resolved_scale=float(resolved_scale),
        scale_rule=str(scale_rule),
        codec=str(preset["codec"]), lossless=bool(preset["lossless"]),
        quality=str(quality), quality_rms=float(preset["rms"]),
        clips=tuple(
            ClipEntry(replicate_id=t.replicate_id, label=t.label, frac=t.frac,
                      source_box=t.source_box,
                      width=t.source_box[2] - t.source_box[0],
                      height=t.source_box[3] - t.source_box[1],
                      filename=os.path.basename(p),
                      frame_count=nf, size_bytes=os.path.getsize(p))
            for t, p, nf in zip(layout.tiles, out_paths, clip_frames)),
    )
    write_manifest(man, manifest_path_for(video_path, out_dir))
    if progress is not None:
        progress(1.0)
    return man


def _pump_progress(proc, frame_count: int, progress, should_cancel) -> list[str]:
    """Drain stdout for progress while a thread drains stderr. Returns stderr lines.

    **Both pipes must be drained concurrently.** ffmpeg writes progress to stdout
    and diagnostics to stderr; reading only stdout and deferring stderr until the
    process exits deadlocks as soon as ffmpeg emits one pipe buffer (~64 KB) of
    stderr, because ffmpeg then blocks writing while this side blocks reading.
    That is not hypothetical on these sources -- ``FINDINGS.md`` section 3 records
    mid-stream decode failures as a live concern, and an error-per-frame stream
    crosses 64 KB in seconds. It would hang the multi-minute whole-source runs
    this module exists for while never reproducing on a short test clip.
    """
    total = max(1, int(frame_count))
    err: list[str] = []
    pump = None
    if proc.stderr is not None:
        pump = threading.Thread(target=lambda: err.extend(proc.stderr),
                                daemon=True)
        pump.start()
    if proc.stdout is not None:
        for line in proc.stdout:
            if should_cancel is not None and should_cancel():
                raise PretranscodeCancelled("pre-transcode cancelled")
            if progress is not None and line.startswith("frame="):
                try:
                    progress(min(1.0, int(line.split("=", 1)[1]) / total))
                except ValueError:
                    pass
    proc.wait()
    if pump is not None:
        pump.join(timeout=10)
    return err


def _kill(proc) -> None:
    try:
        proc.kill()
        proc.wait(timeout=10)
    except Exception:
        pass
    finally:
        # Popen only closes these on __exit__ or collection, and a GUI that
        # cancels repeatedly would accumulate descriptors -- which on Windows can
        # also hold the partial clip locked, making _cleanup fail silently and
        # leave behind exactly the half-written file cancel exists to remove.
        for pipe in (proc.stdout, proc.stderr):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:
                pass


def _cleanup(paths: list[str]) -> None:
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


def write_manifest(man: Manifest, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(man.to_meta(), f, indent=2, sort_keys=True)
    os.replace(tmp, path)   # atomic, so a crash cannot leave a torn manifest


def read_manifest(path: str) -> Manifest:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    ver = int(d.get("version", 0))
    if ver != PRETRANSCODE_VERSION:
        raise PretranscodeError(
            f"manifest version {ver} != {PRETRANSCODE_VERSION}; re-cut the clips")
    return Manifest.from_meta(d)


def verify_manifest(man: Manifest, out_dir: str, replicates: list[dict] | None = None,
                    *, deep: bool = False) -> None:
    """Raise if these clips cannot be trusted for the given geometry.

    Checks the source is the same file, every clip is present *and unchanged in
    size*, and -- when ``replicates`` is given -- that the boxes still match the
    ones the clips were cut from. A moved box silently reading a stale clip would
    attribute one region's pixels to another's detections, so this is not
    advisory.

    The size check is what makes a truncated clip a hard error here rather than a
    surprise at decode time. It is deliberately size and not frame count: a file
    cannot lose frames without losing bytes, so ``os.stat`` catches the same
    thing that re-probing every clip with ``VideoCapture`` would, for
    microseconds instead of the measured 41.8 ms per 6 clips -- and this runs on
    every pass, on a surface that starts one per knob edit. Clips cut before the
    size was recorded carry 0 and skip the check rather than failing outright.
    """
    if replicates is not None and geometry_hash(replicates) != man.geometry_hash:
        raise PretranscodeError(
            "replicate geometry has changed since these clips were cut; "
            "re-run the pre-transcode")
    if not os.path.isfile(man.source_path):
        raise PretranscodeError(f"source is missing: {man.source_path}")
    if os.path.getsize(man.source_path) != man.source_size:
        raise PretranscodeError("source file size has changed since the cut")
    sig = full_sha256(man.source_path) if deep else quick_signature(man.source_path)
    if (man.source_sha256 if deep else man.quick_sig) != sig:
        raise PretranscodeError("source file contents have changed since the cut")
    for c in man.clips:
        p = os.path.join(out_dir, c.filename)
        if not os.path.isfile(p):
            raise PretranscodeError(f"clip is missing: {p}")
        if c.size_bytes and os.path.getsize(p) != c.size_bytes:
            raise PretranscodeError(
                f"clip {c.filename} is {os.path.getsize(p)} bytes but was cut at "
                f"{c.size_bytes}; it was truncated or replaced -- re-run the "
                "pre-transcode")
