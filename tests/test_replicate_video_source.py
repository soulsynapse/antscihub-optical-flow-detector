"""FFmpeg ROI decode: the windowed seek, the atlas layout, and the fallback.

This decoder had no coverage at all before the live tensor path started using it.
The seek is the part worth guarding hardest: an off-by-one there does not fail
loudly, it silently misaligns every extracted channel against the frame the
scrubber is showing.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import core.tensor_channels as tc
from core.channel_source import synth_live_meta
from core.config import PipelineConfig
from core.video import ReplicateVideoSource, VideoSource

FPS = 20.0
N_FRAMES = 40
W, H = 64, 48

needs_ffmpeg = unittest.skipIf(shutil.which("ffmpeg") is None,
                               "ffmpeg is not on PATH")


def _write_ramp_video(path: str) -> None:
    """Each frame is a flat grey level equal to its own index times four.

    Flat frames make the seek test unambiguous: the decoded value *is* the frame
    number, so a frame offset shows up as a fixed error rather than as a subtle
    change in texture that a correlation might forgive.
    """
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise unittest.SkipTest("no mp4v VideoWriter available in this environment")
    try:
        for i in range(N_FRAMES):
            writer.write(np.full((H, W, 3), i * 4, np.uint8))
    finally:
        writer.release()


def _write_moving_square(path: str) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise unittest.SkipTest("no mp4v VideoWriter available in this environment")
    try:
        for i in range(N_FRAMES):
            frame = np.zeros((H, W, 3), np.uint8)
            cv2.rectangle(frame, (4 + i, 16), (14 + i, 30), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def _two_replicates() -> list[dict]:
    return [{"id": 0, "label": "L", "frac": (0.0, 0.0, 0.5, 1.0)},
            {"id": 1, "label": "R", "frac": (0.5, 0.0, 1.0, 1.0)}]


def _roi_layout_for(tiles):
    return tc._roi_layout(tiles)


def _tiles(video: str, replicates) -> tuple[list[dict], float]:
    cfg = PipelineConfig()
    with VideoSource(video) as src:
        info = src.info
    meta = synth_live_meta(video, cfg, replicates, width=info.width,
                           height=info.height, fps=info.fps,
                           frame_count=info.frame_count)
    return tc._tiles_from_meta(meta), info.fps


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.mkdtemp(prefix="roivid_")
        cls.ramp = os.path.join(cls._dir, "ramp.mp4")
        cls.square = os.path.join(cls._dir, "square.mp4")
        _write_ramp_video(cls.ramp)
        _write_moving_square(cls.square)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._dir, ignore_errors=True)


@needs_ffmpeg
class SeekTest(_Base):
    def test_start_zero_yields_every_frame_in_order(self):
        tiles, fps = _tiles(self.ramp, _two_replicates())
        roi, frames = tc._open_roi(self.ramp, tiles, 0, N_FRAMES, fps)
        self.assertIsNotNone(roi)
        try:
            idxs = [i for i, _ in frames]
        finally:
            roi.release()
        self.assertEqual(idxs, list(range(N_FRAMES)))

    def test_seek_is_frame_accurate(self):
        """The frame decoded at start S must be S, not the preceding keyframe."""
        tiles, fps = _tiles(self.ramp, _two_replicates())
        for start in (0, 1, 5, 17, 33):
            with self.subTest(start=start):
                roi, frames = tc._open_roi(self.ramp, tiles, start, 2, fps)
                self.assertIsNotNone(roi)
                try:
                    idx, atlas = next(frames)
                    value = float(roi.crop(atlas, 0).mean())
                finally:
                    roi.release()
                # Absolute indexing: a caller must not have to re-add the offset.
                self.assertEqual(idx, start)
                # Frame content identifies the frame; tolerance covers the mp4v
                # round trip, and is far tighter than the 4 levels between frames.
                self.assertAlmostEqual(value, start * 4, delta=1.5)

    def test_seek_ignores_a_rounded_fps_from_the_caller(self):
        """A caller's rounded frame rate must not shift the window.

        The seek divides the frame index by the frame rate, so the error grows
        with the index: on 24000/1001 footage a caller passing 24.0 lands three
        frames early by frame 11000. The container's value is read instead, so a
        deliberately wrong fps here must change nothing.
        """
        tiles, _ = _tiles(self.ramp, _two_replicates())
        start = N_FRAMES - 6
        seen = []
        for supplied in (FPS, FPS * 1.10, FPS * 0.90):
            roi, frames = tc._open_roi(self.ramp, tiles, start, 1, supplied)
            self.assertIsNotNone(roi)
            try:
                idx, atlas = next(frames)
                seen.append((idx, round(float(roi.crop(atlas, 0).mean()))))
            finally:
                roi.release()
        self.assertEqual(len(set(seen)), 1, f"fps changed the window: {seen}")
        self.assertEqual(seen[0][0], start)
        self.assertAlmostEqual(seen[0][1], start * 4, delta=2.0)

    def test_seek_without_any_usable_frame_rate_is_refused(self):
        """No container fps and none supplied -> refuse, do not seek to zero."""
        tiles, _ = _tiles(self.ramp, _two_replicates())
        with mock.patch.object(ReplicateVideoSource, "_container_fps",
                               return_value=None):
            with self.assertRaises(ValueError):
                ReplicateVideoSource(self.ramp, _roi_layout_for(tiles), 5,
                                     start=10, fps=None)


@needs_ffmpeg
class AtlasTest(_Base):
    def test_crop_returns_the_right_tile_in_grey_levels(self):
        """Each replicate's crop must match a straight OpenCV crop of the same box."""
        replicates = _two_replicates()
        tiles, fps = _tiles(self.square, replicates)
        roi, frames = tc._open_roi(self.square, tiles, 3, 1, fps)
        self.assertIsNotNone(roi)
        try:
            _, atlas = next(frames)
            crops = [roi.crop(atlas, ti) for ti in range(len(tiles))]
        finally:
            roi.release()

        with VideoSource(self.square) as src:
            frame = src.frame_at(3)
        for ti, (t, got) in enumerate(zip(tiles, crops)):
            with self.subTest(tile=ti):
                x0, y0, x1, y1 = t["source_box"]
                want = cv2.cvtColor(
                    cv2.resize(frame[y0:y1, x0:x1],
                               (t["work_width"], t["work_height"]),
                               interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2GRAY).astype(np.float32)
                self.assertEqual(got.shape, want.shape)
                self.assertEqual(got.dtype, np.float32)
                # Both quantize an area-average, by different kernels, so they
                # agree closely rather than exactly -- see the batch notes: the
                # ROI path measures closer to a float reference than this one is.
                self.assertLess(float(np.abs(got - want).mean()), 6.0)
                # At frame 3 the square sits entirely in the left tile, so the
                # right one is flat and has no correlation to test -- checking it
                # anyway would assert on a 0/0 nan.
                if want.std() > 1.0:
                    self.assertGreater(
                        float(np.corrcoef(got.ravel(), want.ravel())[0, 1]), 0.99)

    def test_crop_is_on_the_0_255_scale(self):
        """gray16 arrives as uint16; consumers downstream assume 0-255 floats."""
        tiles, fps = _tiles(self.ramp, _two_replicates())
        roi, frames = tc._open_roi(self.ramp, tiles, 30, 1, fps)
        self.assertIsNotNone(roi)
        try:
            _, atlas = next(frames)
            self.assertEqual(atlas.dtype, np.dtype("<u2"))
            got = roi.crop(atlas, 0)
        finally:
            roi.release()
        self.assertAlmostEqual(float(got.mean()), 120.0, delta=2.0)

    def test_tiles_do_not_overlap_in_the_atlas(self):
        """Distinct source content must land in distinct atlas rows."""
        video = os.path.join(self._dir, "halves.mp4")
        writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"),
                                 FPS, (W, H))
        if not writer.isOpened():
            self.skipTest("no mp4v VideoWriter available")
        try:
            frame = np.zeros((H, W, 3), np.uint8)
            frame[:, W // 2:] = 200          # right half bright, left half black
            for _ in range(6):
                writer.write(frame)
        finally:
            writer.release()

        tiles, fps = _tiles(video, _two_replicates())
        roi, frames = tc._open_roi(video, tiles, 0, 1, fps)
        self.assertIsNotNone(roi)
        try:
            _, atlas = next(frames)
            left = float(roi.crop(atlas, 0).mean())
            right = float(roi.crop(atlas, 1).mean())
        finally:
            roi.release()
        self.assertLess(left, 40.0)
        self.assertGreater(right, 160.0)


class FallbackTest(_Base):
    def test_missing_ffmpeg_falls_back_rather_than_raising(self):
        tiles, fps = _tiles(self.ramp, _two_replicates())
        with mock.patch("core.video.shutil.which", return_value=None):
            roi, frames = tc._open_roi(self.ramp, tiles, 0, 4, fps)
        self.assertIsNone(roi)
        self.assertIsNone(frames)

    def test_decoder_that_fails_on_first_frame_falls_back(self):
        """Construction succeeding is not proof the graph runs -- a decoder that
        dies on contact must fall back, not kill the extraction pass."""
        tiles, fps = _tiles(self.ramp, _two_replicates())
        real = ReplicateVideoSource.iter_frames

        def boom(self):
            raise RuntimeError("FFmpeg ROI decode failed")
            yield  # pragma: no cover - generator marker

        with mock.patch.object(ReplicateVideoSource, "iter_frames", boom):
            roi, frames = tc._open_roi(self.ramp, tiles, 0, 4, fps)
        self.assertIsNone(roi)
        self.assertIsNone(frames)
        self.assertIs(ReplicateVideoSource.iter_frames, real)


@needs_ffmpeg
class ExtractionParityTest(_Base):
    def test_roi_and_full_frame_extraction_agree(self):
        """The two decode paths must produce the same channels, mid-clip."""
        cfg = PipelineConfig()
        with VideoSource(self.square) as src:
            info = src.info
        meta = synth_live_meta(self.square, cfg, _two_replicates(),
                               width=info.width, height=info.height,
                               fps=info.fps, frame_count=info.frame_count)

        roi_out = tc.extract_channels_live(self.square, meta, start=8, n=16)
        real = tc._open_roi
        tc._open_roi = lambda *a, **k: (None, None)
        try:
            full_out = tc.extract_channels_live(self.square, meta, start=8, n=16)
        finally:
            tc._open_roi = real

        for name in ("intensity", "change", "tensor_speed"):
            with self.subTest(channel=name):
                a = np.asarray(roi_out[name], np.float32)
                b = np.asarray(full_out[name], np.float32)
                self.assertEqual(a.shape, b.shape)
                if not b.any():
                    continue
                self.assertGreater(
                    float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), 0.95)

    def test_window_start_is_reported_absolutely(self):
        cfg = PipelineConfig()
        with VideoSource(self.square) as src:
            info = src.info
        meta = synth_live_meta(self.square, cfg, _two_replicates(),
                               width=info.width, height=info.height,
                               fps=info.fps, frame_count=info.frame_count)
        out = tc.extract_channels_live(self.square, meta, start=12, n=10)
        self.assertEqual(out["meta"]["window_start"], 12)
        self.assertEqual(out["meta"]["n_frames"], 10)
        # Locate the square by its brightest block column and check it sits where
        # frame 12 puts it, not where frame 0 would. Absolute grey level cannot
        # be used for this: the default zscore normalize removes the DC entirely
        # (and zeroes a flat frame outright), so only *position* survives it.
        block = int(meta["block_size"])
        col = int(np.argmax(out["intensity"][0].sum(axis=0)))
        self.assertAlmostEqual(col * block, 4 + 12, delta=block)


if __name__ == "__main__":
    unittest.main()
