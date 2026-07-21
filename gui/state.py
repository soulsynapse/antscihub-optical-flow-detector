"""Shared application state.

One object owns the video, the config, the open cache and the ROI/replicate
list. Tabs talk to each other only through its signals -- no tab holds a
reference to another tab. That is what keeps "expand the cache in Preprocessing
& Flow and refresh downstream" from turning into a web of cross-tab calls.
"""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from core import cache as cache_mod
from core.config import PipelineConfig
from core.features import FeatureContext
from core.roi import ROI
from core.video import VideoSource

# Width at which frames are decoded down for display. See display_frame().
DISPLAY_MAX_W = 1280


class AppState(QObject):
    video_loaded = pyqtSignal()
    cache_opened = pyqtSignal()
    rois_changed = pyqtSignal()
    frame_changed = pyqtSignal(int)
    status = pyqtSignal(str)
    request_tab = pyqtSignal(int)
    # (replicate id, {field: value}) -- a calibration measured somewhere other
    # than the replicate tab. The tab owns the list and its sidecar, so it is
    # what persists this; state only relays and keeps its own specs in step.
    calibration_changed = pyqtSignal(int, object)

    def __init__(self, project_dir: str = "."):
        super().__init__()
        self.project_dir = project_dir
        self.cache_root = os.path.join(project_dir, ".cache")

        self.source: VideoSource | None = None
        self.cfg = PipelineConfig()
        self.cache = None
        self.ctx: FeatureContext | None = None

        self.rois: list[ROI] = []
        # Geometry-only replicate specs are shared so Preprocessing & Flow can run
        # ROI-first cache before any cache-backed ROI objects exist.
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
        # (roi_id, feature) -> full-length mean time series over the ROI's blocks.
        self._ts_cache: dict[tuple[int, str], np.ndarray] = {}

    # How band power is aggregated over a replicate box: "max" preserves a
    # small, localized oscillation (a wing occupying a few blocks in a big box);
    # "mean" and "p90" are steadier but dilute it. Max is the right default for
    # detecting small behaviors in hand-drawn regions.
    band_reduce: str = "max"

    def roi_series(self, roi, feature: str) -> np.ndarray:
        """Memoized per-ROI feature series.

        Recomputing this is O(n_frames x roi_blocks) and the inspector asks for
        it on every frame change. On the full clip that is a 30,600-frame reduction
        per feature per scrub step -- it would make scrubbing unusable. The result
        depends only on (roi, feature) and the cache, all of which are fixed
        between re-extractions, so it is safe to hold.

        Band-power features are computed ON DEMAND for whatever band the name
        encodes, and aggregated over the box's blocks by MAX rather than mean --
        see core.roi.roi_band_power for why. Every other feature is the mean over
        the box's blocks.
        """
        from core.roi import parse_band_feature, roi_band_power, roi_time_series
        key = (roi.roi_id, feature, self.band_reduce)
        hit = self._ts_cache.get(key)
        if hit is None:
            band = parse_band_feature(feature)
            if band is not None:
                hit = roi_band_power(self.cache, self.ctx, roi, band[0], band[1],
                                     reduce=self.band_reduce)
            else:
                hit = roi_time_series(self.cache, self.ctx, roi, feature)
            self._ts_cache[key] = hit
        return hit

    def invalidate_series(self) -> None:
        self._ts_cache.clear()

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

        # The previously open cache belongs to the OLD video (caches are keyed by
        # video hash and only ever opened by hand in Preprocessing & Flow). If kept,
        # new clip's boxes get rebuilt onto the old grid and Tab 3 shows traces
        # computed from the old video's flow under the new video -- a stale
        # detection. Drop it; the user re-caches the new clip as usual.
        if self.cache is not None:
            self.cache.close()
        self.cache = None
        self.ctx = None

        # ROIs and manual marks are scoped to a specific video (they live in
        # sidecar files next to it, see video_sidecar). Switching videos must
        # therefore drop the previous clip's ROIs/traces from memory, or they
        # would ghost onto the new clip until it is re-cached. The tabs reload
        # the new video's sidecars off video_loaded / cache_opened.
        self.rois = []
        self.replicate_specs = []
        self.selected_roi = None
        self.invalidate_series()

        # Propose a band the footage can actually resolve. A 15-30 Hz default on
        # 60 fps footage would sit on the Nyquist limit and alias.
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
        if self.cache is not None:
            return self.cache.fps
        return self.source.info.fps if self.source else 30.0

    @property
    def has_video(self) -> bool:
        return self.source is not None

    @property
    def has_cache(self) -> bool:
        return self.cache is not None and self.ctx is not None

    def set_replicate_specs(self, replicates: list[dict]) -> None:
        """Publish replicate geometry and invalidate a mismatched ROI-first cache."""
        from core.replicates import geometry_hash
        self.replicate_specs = [{**r, "frac": tuple(r["frac"])}
                                for r in replicates]
        if self.cache is not None and \
                self.cache.meta.get("processing_scope") == "replicate":
            expected = geometry_hash(self.replicate_specs) \
                if self.replicate_specs else None
            if expected != self.cache.meta.get("replicate_geometry_hash"):
                self.cache.close()
                self.cache = None
                self.ctx = None
                self.rois = []
                self.invalidate_series()
                self.status.emit(
                    "Replicate geometry changed; run Test/Full again to build "
                    "a matching ROI-first cache.")

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

    # -- cache ---------------------------------------------------------------

    def open_cache(self, key: str) -> None:
        if self.cache is not None:
            self.cache.close()
        self.cache = None
        self.ctx = None
        cache = cache_mod.open_cache(self.cache_root, key)
        try:
            if cache.meta.get("processing_scope") == "replicate":
                from core.replicates import geometry_hash
                if not self.replicate_specs:
                    raise ValueError(
                        "This is a per-replicate cache, but the current video "
                        "has no replicate boxes. Load its ROI layout first.")
                current = geometry_hash(self.replicate_specs)
                cached = cache.meta.get("replicate_geometry_hash")
                if current != cached:
                    raise ValueError(
                        "This cache was built for different replicate geometry. "
                        "Restore those boxes or run a new cache for the current "
                        "layout.")
            extras = {
                name: cache.read(name)
                for name in cache.feature_names
                if name not in ("u", "v", "speed")
            }
            ctx = FeatureContext(
                u=cache.read("u"),
                v=cache.read("v"),
                speed=cache.read("speed"),
                fps=cache.fps,
                block_size=cache.block_size,
                bands=extras,
                band_window_s=float(cache.meta.get("band_window_s", 1.0)),
                band_hop_s=float(cache.meta.get("band_hop_s", 0.25)),
                regions=[tuple(t["atlas_bbox"])
                         for t in cache.meta.get("replicate_tiles", [])] or None,
            )
        except Exception:
            cache.close()
            raise
        self.cache = cache
        self.ctx = ctx

        self.rois = []
        self.selected_roi = None
        self.current_frame = 0
        self._ts_cache.clear()

        self.cache_opened.emit()
        self.status.emit(
            f"Cache open: {self.cache.n_frames} frames, "
            f"{len(self.cache.meta.get('replicate_tiles', [])) or 1} region(s), "
            f"{self.cache.grid[0]}x{self.cache.grid[1]} packed blocks, "
            f"{cache_mod.human_bytes(self.cache.size_on_disk())} on disk"
        )

    @property
    def band_features(self) -> list[str]:
        if not self.cache:
            return []
        return [n for n in self.cache.feature_names if n.startswith("bandpower_")]

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
        n = self.cache.n_frames if self.cache else (
            self.source.info.frame_count if self.source else 1)
        idx = max(0, min(int(idx), n - 1))
        if idx != self.current_frame:
            self.current_frame = idx
            self.frame_changed.emit(idx)

    def roi_by_id(self, roi_id: int) -> ROI | None:
        return next((r for r in self.rois if r.roi_id == roi_id), None)
