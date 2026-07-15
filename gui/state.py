"""Shared application state.

One object owns the video, the config, the open cache, the filter state, the ROI
list and the behavior library. The three tabs talk to each other only through its
signals -- no tab holds a reference to another tab. That is what keeps "expand the
cache in Tab 1 and everything downstream refreshes" from turning into a web of
cross-tab calls.
"""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from core import cache as cache_mod
from core.behavior import Behavior, BehaviorLibrary
from core.config import PipelineConfig
from core.features import FeatureContext
from core.filters import FeatureSampler, FilterState
from core.roi import ROI, ROIParams
from core.video import VideoSource

# Width at which frames are decoded down for display. See display_frame().
DISPLAY_MAX_W = 1280


class AppState(QObject):
    video_loaded = pyqtSignal()
    cache_opened = pyqtSignal()
    filter_changed = pyqtSignal()
    rois_changed = pyqtSignal()
    behaviors_changed = pyqtSignal()
    frame_changed = pyqtSignal(int)
    status = pyqtSignal(str)
    request_tab = pyqtSignal(int)

    def __init__(self, project_dir: str = "."):
        super().__init__()
        self.project_dir = project_dir
        self.cache_root = os.path.join(project_dir, ".cache")

        self.source: VideoSource | None = None
        self.cfg = PipelineConfig()
        self.cache = None
        self.ctx: FeatureContext | None = None
        self.sampler: FeatureSampler | None = None

        self.filter = FilterState()
        self.roi_params = ROIParams()
        self.rois: list[ROI] = []
        self.selected_roi: int | None = None

        self.library = BehaviorLibrary(os.path.join(project_dir, "behaviors"))
        self.behaviors: list[Behavior] = []
        self.selected_behavior: str | None = None
        # (roi_id, behavior_name) -> cleaned boolean trace
        self.traces: dict[tuple[int, str], np.ndarray] = {}
        # (roi_id, behavior_name) -> per-frame strength (fraction of blocks), 0..1
        self.strengths: dict[tuple[int, str], np.ndarray] = {}

        self.current_frame = 0
        self._disp_frame: np.ndarray | None = None
        self._disp_idx: int = -1
        # (roi_id, feature) -> full-length mean time series over the ROI's blocks.
        self._ts_cache: dict[tuple[int, str], np.ndarray] = {}
        # (roi_id, feature) -> per-block value matrix (T, k) over the ROI.
        self._block_cache: dict[tuple[int, str], np.ndarray] = {}

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

    def roi_blocks(self, roi, feature: str) -> np.ndarray:
        """Memoized per-block value matrix (T, k) for a box+feature. This is what
        the range editor plots and what detection thresholds -- one source, so the
        plot and the ethogram cannot disagree."""
        from core.roi import roi_block_values
        key = (roi.roi_id, feature)
        hit = self._block_cache.get(key)
        if hit is None:
            hit = roi_block_values(self.cache, self.ctx, roi, feature)
            self._block_cache[key] = hit
        return hit

    def invalidate_series(self) -> None:
        self._ts_cache.clear()
        self._block_cache.clear()

    # -- video ---------------------------------------------------------------

    def load_video(self, path: str) -> None:
        if self.source is not None:
            self.source.release()
        self.source = VideoSource(path)
        self.current_frame = 0
        self._disp_frame = None
        self._disp_idx = -1

        # The previously open cache belongs to the OLD video (caches are keyed by
        # video hash and only ever opened by hand in Tab 1). If we keep it, the
        # new clip's boxes get rebuilt onto the old grid and Tab 3 shows traces
        # computed from the old video's flow under the new video -- a stale
        # detection. Drop it; the user re-caches the new clip in Tab 1 as usual.
        if self.cache is not None:
            self.cache.close()
        self.cache = None
        self.ctx = None
        self.sampler = None

        # ROIs and manual marks are scoped to a specific video (they live in
        # sidecar files next to it, see video_sidecar). Switching videos must
        # therefore drop the previous clip's ROIs/traces from memory, or they
        # would ghost onto the new clip until it is re-cached. The tabs reload
        # the new video's sidecars off video_loaded / cache_opened.
        self.rois = []
        self.selected_roi = None
        self.traces = {}
        self.strengths = {}
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

    # -- cache ---------------------------------------------------------------

    def open_cache(self, key: str) -> None:
        if self.cache is not None:
            self.cache.close()
        self.cache = None
        self.ctx = None
        self.sampler = None
        cache = cache_mod.open_cache(self.cache_root, key)
        try:
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
            )
        except Exception:
            cache.close()
            raise
        self.cache = cache
        self.ctx = ctx
        self.sampler = FeatureSampler(self.cache, self.ctx)

        self.rois = []
        self.traces = {}
        self.selected_roi = None
        self.current_frame = 0
        self._ts_cache.clear()
        # Filter ranges are keyed by feature name; a band from a previously opened
        # cache would reference a feature this cache does not have. Drop them.
        self.filter.ranges.clear()

        self.cache_opened.emit()
        self.status.emit(
            f"Cache open: {self.cache.n_frames} frames, "
            f"{self.cache.grid[0]}x{self.cache.grid[1]} blocks, "
            f"{cache_mod.human_bytes(self.cache.size_on_disk())} on disk"
        )

    @property
    def band_features(self) -> list[str]:
        if not self.cache:
            return []
        return [n for n in self.cache.feature_names if n.startswith("bandpower_")]

    def available_features(self) -> list[str]:
        """Everything the current cache can serve, cached or derived."""
        from core.features import REGISTRY
        from core.roi import roi_feature_available
        if not self.cache:
            return []
        out = list(self.cache.feature_names)
        available = set(out)

        # Add derived features only when all of their cache-time dependencies are
        # actually present. This keeps texture_percentile out of old caches that
        # have no texture map, while ordinary u/v/speed derivatives remain free.
        changed = True
        while changed:
            changed = False
            for name, spec in REGISTRY.items():
                if spec.kind != "derived" or name in available:
                    continue
                if all(dep in available for dep in spec.deps):
                    out.append(name)
                    available.add(name)
                    changed = True

        for name, spec in REGISTRY.items():
            if spec.kind == "roi" and roi_feature_available(name, self.rois):
                out.append(name)
        seen, uniq = set(), []
        for n in out:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        return uniq

    # -- frame ---------------------------------------------------------------

    def display_frame(self, idx: int) -> np.ndarray | None:
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
        if self._disp_idx == idx and self._disp_frame is not None:
            return self._disp_frame

        frame = self.source.frame_at(idx)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w > DISPLAY_MAX_W:
            s = DISPLAY_MAX_W / w
            frame = cv2.resize(frame, (DISPLAY_MAX_W, max(1, int(round(h * s)))),
                               interpolation=cv2.INTER_AREA)
        self._disp_frame = frame
        self._disp_idx = idx
        return frame

    def set_frame(self, idx: int) -> None:
        n = self.cache.n_frames if self.cache else (
            self.source.info.frame_count if self.source else 1)
        idx = max(0, min(int(idx), n - 1))
        if idx != self.current_frame:
            self.current_frame = idx
            self.frame_changed.emit(idx)

    def time_s(self) -> float:
        return self.current_frame / self.fps

    # -- behaviors -----------------------------------------------------------

    def reload_behaviors(self) -> None:
        self.behaviors = self.library.load_all()
        self.behaviors_changed.emit()

    def recompute_traces(self) -> None:
        """Evaluate every behavior against every replicate, per block.

        Detection now runs over the blocks inside each box (min-clump / merge /
        fraction spatial criteria), not just the box's aggregate series, so both
        the binary trace and a per-frame strength (fraction of blocks passing)
        are produced. Strength drives the graded ethogram.
        """
        from core.roi import roi_detection
        if not self.has_cache:
            return
        self.traces = {}
        self.strengths = {}
        for b in self.behaviors:
            for roi in self.rois:
                try:
                    detected, strength = roi_detection(self.cache, self.ctx, b, roi)
                except (KeyError, ValueError):
                    continue
                self.traces[(roi.roi_id, b.name)] = detected
                self.strengths[(roi.roi_id, b.name)] = strength
        self.behaviors_changed.emit()

    def roi_by_id(self, roi_id: int) -> ROI | None:
        return next((r for r in self.rois if r.roi_id == roi_id), None)
