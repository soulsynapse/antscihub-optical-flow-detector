"""Headless video access for the pipeline.

Deliberately Qt-free so the whole feature pipeline can run from a script or a
test without a display. The GUI wraps this in a QObject that adds signals.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import cv2
import numpy as np


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
