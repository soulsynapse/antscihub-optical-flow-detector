"""Headless video access for the pipeline.

Deliberately Qt-free so the whole feature pipeline can run from a script or a
test without a display. The GUI wraps this in a QObject that adds signals.
"""
from __future__ import annotations

import hashlib
import os
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass

import cv2
import numpy as np

_DONE = object()


def prefetch(iterable, depth: int = 2):
    """Yield from ``iterable`` on a worker thread, running up to ``depth`` ahead.

    Decoding is pure producer work and OpenCV releases the GIL across
    ``VideoCapture.read``, so on the extraction pass it overlaps with the tensor
    math for free -- decode measured 26% of a 5.3K pass, and that is the share
    this hides rather than speeds up.

    ``depth`` stays small deliberately: a 5312x2988 BGR frame is ~48 MB, and at
    the default four can be alive at once (two queued, one the consumer still
    holds, one the producer has decoded but not handed off) -- about 190 MB.

    The consumer must close the generator (or exhaust it) before releasing
    whatever the producer reads from -- ``close()`` stops the thread and joins it,
    so a released VideoCapture can never be touched by a still-running decode.
    Exceptions raised by ``iterable`` are re-raised on the consumer's thread.
    """
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def put(item) -> bool:
        # Poll rather than block forever, so a consumer that walks away is
        # noticed even with the queue full and nobody draining it.
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce():
        try:
            for item in iterable:
                if not put(item):
                    return
        except Exception as exc:          # noqa: BLE001 - forwarded, not handled
            # Exception only: a KeyboardInterrupt or SystemExit re-raised inside
            # the consumer's loop would be indistinguishable from that loop's own
            # failure and would unwind through the extraction cancel machinery
            # instead of ending the process. Those leave via the finally below.
            put(exc)
        finally:
            # The consumer blocks in get() until it sees a terminator, so one must
            # go out on every exit path -- including the BaseException path that
            # deliberately forwards nothing. A put after stop is set is a no-op,
            # which is the already-abandoned case.
            put(_DONE)
            close = getattr(iterable, "close", None)
            if close is not None:
                close()

    th = threading.Thread(target=produce, daemon=True, name="ofd-prefetch")
    th.start()
    try:
        while True:
            item = q.get()
            if item is _DONE:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()
        # No timeout: the join is what makes the docstring's guarantee true, and
        # the caller acts on it by releasing the decoder the moment close()
        # returns. The producer cannot block indefinitely -- it only ever waits in
        # put(), which polls the stop flag -- so a timeout here could not rescue a
        # real hang, it could only hand back a capture still being read from.
        th.join()


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    video_hash: str

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    @property
    def nyquist_hz(self) -> float:
        return self.fps / 2.0

    def describe(self) -> str:
        return (
            f"{os.path.basename(self.path)}: {self.width}x{self.height} @ "
            f"{self.fps:.2f} fps, {self.duration_s:.1f} s "
            f"({self.frame_count} frames), Nyquist {self.nyquist_hz:.1f} Hz"
        )


def hash_video(path: str, sample_bytes: int = 1 << 20) -> str:
    """Cheap content hash: file size plus the head and tail of the file.

    Hashing a 4 GB clip in full would take longer than the flow computation it
    is meant to key. Size + 1 MB head + 1 MB tail is enough to distinguish any
    two videos we will realistically see, and it is stable across renames.
    """
    size = os.path.getsize(path)
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(sample_bytes))
        if size > 2 * sample_bytes:
            f.seek(-sample_bytes, os.SEEK_END)
            h.update(f.read(sample_bytes))
    return h.hexdigest()[:16]


class VideoSource:
    """Sequential + random access to frames, with the metadata the UI needs."""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {path}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        self.info = VideoInfo(
            path=path,
            fps=float(fps),
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            video_hash=hash_video(path),
        )
        self._pos = 0

    def seek(self, frame_idx: int) -> None:
        frame_idx = max(0, min(frame_idx, self.info.frame_count - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._pos = frame_idx

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._pos += 1
        return frame

    def frame_at(self, frame_idx: int) -> np.ndarray | None:
        """Random access, with a fast path for sequential reads.

        Seeking is expensive on long-GOP footage (H.264/HEVC from a GoPro): the
        decoder has to jump to the preceding keyframe and decode forward, which
        can be dozens of frames of work for a one-frame step. Playback and
        frame-stepping ask for frame N+1 immediately after frame N, so detect
        that and just decode the next frame -- which is what the decoder is
        already positioned to do. Without this, playback runs at a few frames per
        second no matter how fast the display code is.
        """
        if frame_idx == self._pos:
            return self.read()
        self.seek(frame_idx)
        return self.read()

    def iter_frames(self, start: int = 0, count: int | None = None):
        """Yield (frame_idx, bgr_frame) sequentially. Sequential decode only --
        never seek per frame, that is orders of magnitude slower on long GOPs."""
        self.seek(start)
        n = self.info.frame_count - start if count is None else count
        for i in range(n):
            frame = self.read()
            if frame is None:
                return
            yield start + i, frame

    def time_to_frame(self, t_s: float) -> int:
        return int(round(t_s * self.info.fps))

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


def _tile_tail(work_width: int, work_height: int, atlas_width: int) -> str:
    """The scale/format/pad tail every atlas tile shares, whatever fed it.

    Kept in one place because the two sources that use it -- a live crop out of
    the whole source frame and a pre-transcoded clip of the same box -- are
    expected to produce the *same numbers* for the same replicate. Any
    divergence here would surface as a clip and a live crop disagreeing, which
    is precisely the comparison the pre-transcode exists to make trustworthy.

    16-bit, and the format conversion must come AFTER the scale. Area-averaging
    a 4x4 patch of 8-bit samples yields a value that does not fit in 8 bits, and
    rounding it there was the largest single error this path carried. Measured
    against a float reference on a 73x76 tile: 8-bit out 0.528 grey-levels RMS,
    gray16 out 0.364 (full-frame OpenCV, which quantizes the same way, is
    0.321). Putting format=gray16le BEFORE the scale instead measures 3.80, far
    worse than either: swscale then converts first and scales a gray16 input
    rather than keeping its high-precision intermediate.
    """
    return (f"scale={work_width}:{work_height}:flags=area,"
            f"format=gray16le,pad={atlas_width}:ih:0:0")


class _AtlasStream:
    """Reader for a vertically packed gray16le atlas arriving on an FFmpeg pipe.

    :class:`ReplicateVideoSource` and :class:`ClipAtlasSource` differ only in the
    filter graph they hand FFmpeg. Everything downstream of the pipe -- framing,
    the uint16 -> 0..255 float rescale, teardown -- is shared here so the two
    cannot drift apart in the ways that would matter: a half-read frame, or a
    different grey convention between a clip and a live crop of the same box.

    Subclasses set ``width``, ``height``, ``slices``, ``n_frames``, ``start``
    and ``_what`` (a label for error messages), then call :meth:`_spawn`.
    """

    _what = "ROI decode"

    def _spawn(self, cmd: list[str]) -> None:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
        self.frame_bytes = self.width * self.height * 2   # gray16le

    def crop(self, atlas: np.ndarray, replicate_id: int) -> np.ndarray:
        """This tile's pixels, float32 on the usual 0-255 grey scale.

        The atlas arrives as uint16 (see the filter graph); rescaling here rather
        than at each call site keeps every consumer on one convention, and the
        float32 is what Preprocessor would have produced anyway.
        """
        tile = atlas[self.slices[int(replicate_id)]]
        return tile.astype(np.float32) * (255.0 / 65535.0)

    def crop_native(self, atlas: np.ndarray, replicate_id: int) -> np.ndarray:
        """This tile as the decoder's native gray16 view.

        Most consumers need the 0..255 float convention from :meth:`crop`.
        Z-score normalization is invariant to a positive input scale, however,
        so the tensor extractor can normalize this view directly and avoid a
        full-plane uint16 -> float32 conversion before immediately applying a
        second affine transform.  Keeping this as a separate method makes that
        exception explicit instead of weakening ``crop``'s public contract.
        """
        return atlas[self.slices[int(replicate_id)]]

    def iter_frames(self):
        if self.proc.stdout is None:
            return
        for i in range(self.n_frames):
            chunks = []
            remaining = self.frame_bytes
            while remaining:
                chunk = self.proc.stdout.read(remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            if remaining:
                rc = self.proc.wait()
                detail = self.proc.stderr.read().decode(errors="replace") \
                    if self.proc.stderr is not None else ""
                if rc != 0:
                    raise RuntimeError(
                        f"FFmpeg {self._what} failed at frame "
                        f"{self.start + i}: {detail.strip()}")
                return
            frame = np.frombuffer(b"".join(chunks), dtype="<u2").reshape(
                self.height, self.width)
            yield self.start + i, frame
        rc = self.proc.wait()
        if rc != 0:
            detail = self.proc.stderr.read().decode(errors="replace") \
                if self.proc.stderr is not None else ""
            raise RuntimeError(f"FFmpeg {self._what} failed: {detail.strip()}")

    def release(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc.stderr is not None:
            proc.stderr.close()
        self.proc = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class ReplicateVideoSource(_AtlasStream):
    """Sequential FFmpeg stream containing only scaled replicate-owned pixels.

    FFmpeg decodes the compressed frame once, then crops, converts to grayscale,
    downsamples, and vertically packs the exact replicate boxes before crossing
    the process boundary. On 5.3K footage this avoids materializing a 48 MB BGR
    frame in Python merely to retain ~8% of it.

    ``start`` opens the stream at an arbitrary frame instead of frame zero, which
    is what the live windowed extraction needs. The seek is an *input* seek
    (``-ss`` before ``-i``), so FFmpeg jumps to the preceding keyframe and decodes
    forward internally rather than handing us the keyframe -- accurate, and the
    only reason the windowed path can afford this decoder at all. ``iter_frames``
    yields absolute frame indices, so a caller cannot silently lose the offset.
    """

    def __init__(self, path: str, layout, n_frames: int, *, start: int = 0,
                 fps: float | None = None):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise FileNotFoundError("ffmpeg is not available on PATH")
        self.path = path
        self.layout = layout
        self.n_frames = int(n_frames)
        self.start = max(0, int(start))
        if self.start:
            # Frame index -> timestamp, so the seek is only as accurate as the
            # frame rate it divides by, and the error grows with the index. A
            # caller passing 23.98 for 24000/1001 lands 1 frame early by frame
            # 11000 and 3 frames early with a bare 24.0 -- silently, as a window
            # of the right length from the wrong place. The container's own value
            # is authoritative and costs ~8 ms to read, so it wins over anything
            # passed in; ``fps`` is only the fallback for a container that does
            # not report one.
            fps = self._container_fps() or fps
            if not fps or fps <= 0:
                raise ValueError(
                    "a frame rate is required to seek to a non-zero start frame")
        self.width = max(t.work_width for t in layout.tiles)
        self.height = sum(t.work_height for t in layout.tiles)
        self.slices: dict[int, tuple[slice, slice]] = {}

        if len(layout.tiles) == 1:
            split = "[0:v]null[v0];"
        else:
            split = f"[0:v]split={len(layout.tiles)}" + \
                "".join(f"[v{i}]" for i in range(len(layout.tiles))) + ";"
        filters = []
        outputs = []
        y = 0
        for i, tile in enumerate(layout.tiles):
            x0, y0, x1, y1 = tile.source_box
            filters.append(
                # exact=1 forbids FFmpeg from silently rounding an odd-coordinate
                # crop to an even boundary. Without it, an odd x0/y0 shifts the
                # window by up to a whole source pixel, and that sub-pixel offset
                # -- inconsistent frame to frame -- is read by dense flow as real
                # translation, which per-frame CLAHE at the box edge then amplifies.
                f"[v{i}]crop={x1-x0}:{y1-y0}:{x0}:{y0}:exact=1,"
                + _tile_tail(tile.work_width, tile.work_height, self.width)
                + f"[p{i}]")
            outputs.append(f"[p{i}]")
            self.slices[tile.replicate_id] = (
                slice(y, y + tile.work_height),
                slice(0, tile.work_width),
            )
            y += tile.work_height
        stack = (f"{outputs[0]}null[out]" if len(outputs) == 1 else
                 "".join(outputs) + f"vstack=inputs={len(outputs)}[out]")
        graph = split + ";".join(filters) + ";" + stack
        seek = ["-ss", f"{self.start / float(fps):.6f}"] if self.start else []
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", *seek, "-i", path,
            "-filter_complex", graph, "-map", "[out]",
            "-frames:v", str(self.n_frames), "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "gray16le", "-",
        ]
        self._spawn(cmd)

    def _container_fps(self) -> float | None:
        """The container's own frame rate, at full precision, or None."""
        cap = cv2.VideoCapture(self.path)
        try:
            if not cap.isOpened():
                return None
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            return fps if fps > 0 else None
        finally:
            cap.release()


class ClipAtlasSource(_AtlasStream):
    """The same atlas, assembled from pre-transcoded per-replicate clips.

    Each clip is already the crop, so this graph drops the ``crop`` filter and
    keeps everything after it identical (see :func:`_tile_tail`). What changes is
    where the decoder's work goes: ``ReplicateVideoSource`` still decodes every
    pixel of the source frame and throws ~92% away, because H.264/HEVC decode is
    whole-frame and the crop runs *after* it. Here the stored files are small, so
    the decoder genuinely decodes only replicate pixels -- ~25x faster on
    ``GX010047c2`` (``FINDINGS.md`` section 10).

    One FFmpeg process with N inputs, not N processes: the atlas has to arrive as
    one frame anyway for the consumer's ``crop`` to index into, and vstacking
    inside the graph is how it gets there without N pipes to interleave by hand.

    ``clip_paths`` aligns with ``layout.tiles`` by position. Each clip's stored
    frame size must equal the tile's ``source_box``; a mismatch means the clips
    were cut from different geometry and is raised rather than scaled away.
    Scaling it away is the dangerous outcome, and it is what would happen by
    default -- the ``scale`` in the graph would quietly resample a wrongly-sized
    clip to the tile's work size and hand back plausible pixels belonging to a
    different region.

    ``verify_sizes=False`` skips that probe for a caller that already knows the
    sizes are right. It is worth an option because the probe is not cheap
    relative to what this class exists to save: opening a ``VideoCapture`` per
    clip measured **41.8 ms for 6 clips**, against a ~160 ms clip decode and a
    1.4 ms manifest verification. ``channel_source.resolve_clip_paths`` checks
    the same thing against the manifest for free, so the live surface -- which
    starts a pass on every knob edit -- passes False.
    """

    _what = "clip decode"

    def __init__(self, clip_paths: list[str], layout, n_frames: int, *,
                 start: int = 0, fps: float | None = None,
                 verify_sizes: bool = True):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise FileNotFoundError("ffmpeg is not available on PATH")
        tiles = list(layout.tiles)
        if not tiles:
            raise ValueError("clip decode needs at least one tile")
        if len(clip_paths) != len(tiles):
            raise ValueError(
                f"{len(clip_paths)} clips for {len(tiles)} tiles")
        self.paths = list(clip_paths)
        self.layout = layout
        self.n_frames = int(n_frames)
        self.start = max(0, int(start))
        # Required only where it is used -- the seek -- matching
        # ReplicateVideoSource. Demanding it unconditionally made two sibling
        # decoders disagree about their preconditions for no reason: a caller
        # reading a clip from frame zero needs no rate at all.
        if self.start and (not fps or fps <= 0):
            raise ValueError(
                "a frame rate is required to seek to a non-zero start frame")
        self.width = max(t.work_width for t in tiles)
        self.height = sum(t.work_height for t in tiles)
        self.slices: dict[int, tuple[slice, slice]] = {}

        # Input seek per clip, from the SOURCE rate. Clips are cut without
        # re-timing, so clip frame N is source frame N; the caller passes the
        # manifest's rational rate rather than a rounded float for the reason in
        # ReplicateVideoSource's seek note -- the error grows with the index.
        seek = ([] if not self.start
                else ["-ss", f"{self.start / float(fps):.6f}"])
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
        for p in self.paths:
            cmd += [*seek, "-i", p]

        filters = []
        outputs = []
        y = 0
        for i, tile in enumerate(tiles):
            x0, y0, x1, y1 = tile.source_box
            if verify_sizes:
                cw, ch = _clip_frame_size(self.paths[i])
                if (cw, ch) != (x1 - x0, y1 - y0):
                    raise ValueError(
                        f"clip {os.path.basename(self.paths[i])} is {cw}x{ch} "
                        f"but its replicate box is {x1-x0}x{y1-y0}; the clips "
                        "were cut from different geometry -- re-run the "
                        "pre-transcode")
            filters.append(f"[{i}:v]"
                           + _tile_tail(tile.work_width, tile.work_height,
                                        self.width)
                           + f"[p{i}]")
            outputs.append(f"[p{i}]")
            self.slices[tile.replicate_id] = (
                slice(y, y + tile.work_height),
                slice(0, tile.work_width),
            )
            y += tile.work_height
        # shortest=1, and it is load-bearing. vstack defaults to shortest=0,
        # which does NOT end the stream when an input runs out -- it repeats that
        # input's last frame while the others advance. Here the inputs are
        # separate files, so a single clip cut short by a crash or a full disk
        # would silently *freeze* its replicate: consecutive identical frames,
        # hence change == 0, read downstream as "measured, and nothing moved" for
        # that replicate alone while every other replicate looks fine. A
        # per-replicate false negative with no symptom anywhere. Ending the
        # stream instead surfaces it as a short window, which _stream_channels
        # trims and flags.
        #
        # No effect on the healthy case: a cut writes every clip in one pass, so
        # they share a frame count. (ReplicateVideoSource needs no such flag --
        # its inputs are one split of one decode and cannot differ in length.)
        stack = (f"{outputs[0]}null[out]" if len(outputs) == 1 else
                 "".join(outputs) + f"vstack=inputs={len(outputs)}:shortest=1[out]")
        graph = ";".join(filters) + ";" + stack
        cmd += [
            "-filter_complex", graph, "-map", "[out]",
            "-frames:v", str(self.n_frames), "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "gray16le", "-",
        ]
        self._spawn(cmd)


def _clip_frame_size(path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open clip {path}")
        return (int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
                int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    finally:
        cap.release()
