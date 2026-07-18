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

    def frame_to_time(self, frame_idx: int) -> float:
        return frame_idx / self.info.fps

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


class ReplicateVideoSource:
    """Sequential FFmpeg stream containing only scaled replicate-owned pixels.

    FFmpeg decodes the compressed frame once, then crops, converts to grayscale,
    downsamples, and vertically packs the exact replicate boxes before crossing
    the process boundary. On 5.3K footage this avoids materializing a 48 MB BGR
    frame in Python merely to retain ~8% of it.
    """

    def __init__(self, path: str, layout, n_frames: int):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise FileNotFoundError("ffmpeg is not available on PATH")
        self.path = path
        self.layout = layout
        self.n_frames = int(n_frames)
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
                f"scale={tile.work_width}:{tile.work_height}:flags=area,"
                f"format=gray,pad={self.width}:ih:0:0[p{i}]")
            outputs.append(f"[p{i}]")
            self.slices[tile.replicate_id] = (
                slice(y, y + tile.work_height),
                slice(0, tile.work_width),
            )
            y += tile.work_height
        stack = (f"{outputs[0]}null[out]" if len(outputs) == 1 else
                 "".join(outputs) + f"vstack=inputs={len(outputs)}[out]")
        graph = split + ";".join(filters) + ";" + stack
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path,
            "-filter_complex", graph, "-map", "[out]",
            "-frames:v", str(self.n_frames), "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ]
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
        self.frame_bytes = self.width * self.height

    def crop(self, atlas: np.ndarray, replicate_id: int) -> np.ndarray:
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
                        f"FFmpeg ROI decode failed at frame {i}: {detail.strip()}")
                return
            frame = np.frombuffer(b"".join(chunks), dtype=np.uint8).reshape(
                self.height, self.width)
            yield i, frame
        rc = self.proc.wait()
        if rc != 0:
            detail = self.proc.stderr.read().decode(errors="replace") \
                if self.proc.stderr is not None else ""
            raise RuntimeError(f"FFmpeg ROI decode failed: {detail.strip()}")

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
