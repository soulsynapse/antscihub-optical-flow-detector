"""Shared application state.

One object owns the video, the config, and the replicate-geometry list. Tabs talk
to each other only through its signals -- no tab holds a reference to another tab.
That is what keeps "publish geometry in one tab and refresh downstream" from
turning into a web of cross-tab calls.
"""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from core.config import PipelineConfig
from core.video import VideoSource

# Width at which frames are decoded down for display. See display_frame().
DISPLAY_MAX_W = 1280


class AppState(QObject):
    video_loaded = pyqtSignal()
    rois_changed = pyqtSignal()
    frame_changed = pyqtSignal(int)
    status = pyqtSignal(str)
    request_tab = pyqtSignal(int)
    # (replicate id, {field: value}) -- a calibration measured somewhere other
    # than the replicate tab. The tab owns the list and its sidecar, so it is
    # what persists this; state only relays and keeps its own specs in step.
    calibration_changed = pyqtSignal(int, object)
    # (replicate id) -- that replicate's fileset has just been retired into an
    # old_NNN/ generation, because its box moved. Anything holding results for it
    # IN MEMORY must drop them WITHOUT flushing: a flush would land old-rectangle
    # data back at the home root the retire just emptied, i.e. under the new
    # rectangle. Narrower than rois_changed on purpose -- that fires for any box
    # edit, and discarding every replicate's in-memory work when one box moved
    # would throw away results the move did not invalidate.
    replicate_retired = pyqtSignal(int)

    def __init__(self, project_dir: str = "."):
        super().__init__()
        self.project_dir = project_dir

        self.source: VideoSource | None = None
        self.cfg = PipelineConfig()

        # Geometry-only replicate specs shared across tabs -- the live detection
        # path works from these plus the atlas tiles, no cache-backed ROI objects.
        self.replicate_specs: list[dict] = []
        self.selected_roi: int | None = None

        self.current_frame = 0
        # Keep the most recently decoded source frame as well as its cheap
        # overview. Focused replicate views crop this full-resolution frame first,
        # avoiding the grainy result of enlarging a ~100 px-wide overview crop.
        self._full_frame: np.ndarray | None = None
        self._full_idx: int = -1
        self._disp_frame: np.ndarray | None = None
        self._disp_idx: int = -1

    # -- video ---------------------------------------------------------------

    def load_video(self, path: str) -> None:
        if self.source is not None:
            self.source.release()
        self.source = VideoSource(path)
        self.current_frame = 0
        self._full_frame = None
        self._full_idx = -1
        self._disp_frame = None
        self._disp_idx = -1

        # ROIs and manual marks are scoped to a specific video (they live in
        # sidecar files next to it, see video_sidecar). Switching videos must
        # therefore drop the previous clip's boxes from memory, or they would
        # ghost onto the new clip. The tabs reload the new video's sidecars off
        # video_loaded.
        self.replicate_specs = []
        self.selected_roi = None

        # Propose a band the footage can actually resolve. A 15-30 Hz default on
        # 60 fps footage would sit on the Nyquist limit and alias.
        #
        # INERT as written: this is the only writer of ``cfg.features.bands`` and
        # there is no reader -- the explorer's band comes from the user dragging
        # on an axis that ``core.wavelet.default_freqs`` has already capped at
        # 0.45*fps, which is what actually keeps a band below Nyquist. Kept
        # because the proposal is the right seed IF the picker is ever wired to
        # it; do not cite it as the aliasing guard until then.
        band = self.cfg.features.suggest_band(self.source.info.fps)
        self.cfg = self.cfg.with_band(band)

        self.video_loaded.emit()
        self.status.emit(self.source.info.describe())

    def video_sidecar(self, kind: str) -> str | None:
        """Path to a per-video sidecar file, living next to the video and named
        after it -- e.g. ``.../Foo.mp4`` -> ``.../Foo.rois.json`` for kind
        ``"rois"``. Returns None when no video is loaded.

        This is what scopes ROIs and manual marks to one clip: open a different
        video and you get a different sidecar, so nothing carries over from a
        previous clip. It is deliberately keyed on the video PATH (not its
        content hash) so a human can see which JSON belongs to which video in
        the folder, and moving the clip carries its annotations with it.
        """
        if self.source is None:
            return None
        base, _ = os.path.splitext(self.source.info.path)
        return f"{base}.{kind}.json"

    @property
    def fps(self) -> float:
        return self.source.info.fps if self.source else 30.0

    @property
    def has_video(self) -> bool:
        return self.source is not None

    def set_replicate_specs(self, replicates: list[dict]) -> None:
        """Publish replicate geometry to the other tabs."""
        self.replicate_specs = [{**r, "frac": tuple(r["frac"])}
                                for r in replicates]

    def apply_calibration(self, replicate_id: int, fields: dict) -> None:
        """Merge measured calibration onto one replicate and relay it.

        ``set_replicate_specs`` copies the dicts, so a calibration measured off
        ``replicate_specs`` (the live preprocessing surface works from those)
        would otherwise die with the widget that measured it. This updates the
        published copy and emits, leaving the replicate tab -- which owns the
        authoritative list and its per-video sidecar -- to persist it.

        Only the keys actually measured are merged, so a partial calibration
        never clears a field set by hand.
        """
        if not fields:
            return
        for rep in self.replicate_specs:
            if int(rep.get("id", -1)) == int(replicate_id):
                rep.update(fields)
                break
        self.calibration_changed.emit(int(replicate_id), dict(fields))

    # -- frame ---------------------------------------------------------------

    def display_frame(self, idx: int,
                      focus_frac: tuple[float, float, float, float] | None = None
                      ) -> np.ndarray | None:
        """The current frame, decoded once and downscaled for display.

        This MUST be the only place the video is decoded for the UI. Tabs 2 and 3
        each own a VideoPanel and both are connected to frame_changed, so both
        want the same frame on every step. If they each call VideoSource.frame_at
        themselves, the first decode leaves the decoder positioned at idx+1, the
        second asks for idx again, and the sequential fast path misses -- turning
        every frame step into a full keyframe seek (~400 ms on this GoPro
        footage). Caching the decoded frame here collapses that back to one
        sequential read.

        Downscaling here too: the source is 16 megapixels and the widget is ~700
        px wide, so compositing overlays at full resolution is pure waste.
        """
        import cv2

        if self.source is None:
            return None
        if self._full_idx != idx or self._full_frame is None:
            frame = self.source.frame_at(idx)
            if frame is None:
                return None
            self._full_frame = frame
            self._full_idx = idx
            h, w = frame.shape[:2]
            if w > DISPLAY_MAX_W:
                s = DISPLAY_MAX_W / w
                frame = cv2.resize(
                    frame, (DISPLAY_MAX_W, max(1, int(round(h * s)))),
                    interpolation=cv2.INTER_AREA)
            self._disp_frame = frame
            self._disp_idx = idx

        if focus_frac is None:
            return self._disp_frame

        x0, y0, x1, y1 = map(float, focus_frac)
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"Invalid display focus rectangle: {focus_frac}")
        full = self._full_frame
        h, w = full.shape[:2]
        sx0 = max(0, min(w - 1, int(round(x0 * w))))
        sy0 = max(0, min(h - 1, int(round(y0 * h))))
        sx1 = max(sx0 + 1, min(w, int(round(x1 * w))))
        sy1 = max(sy0 + 1, min(h, int(round(y1 * h))))
        crop = full[sy0:sy1, sx0:sx1]
        ch, cw = crop.shape[:2]
        if cw > DISPLAY_MAX_W:
            s = DISPLAY_MAX_W / cw
            crop = cv2.resize(
                crop, (DISPLAY_MAX_W, max(1, int(round(ch * s)))),
                interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(crop)

    def set_frame(self, idx: int) -> None:
        n = self.source.info.frame_count if self.source else 1
        idx = max(0, min(int(idx), n - 1))
        if idx != self.current_frame:
            self.current_frame = idx
            self.frame_changed.emit(idx)
