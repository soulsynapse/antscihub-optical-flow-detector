"""ROI discovery: turn a per-frame boolean match mask into stable, persistent
regions.

A raw filter mask is noisy and flickery -- individual blocks pop in and out frame
to frame. An ROI is what survives three successive requirements:

  1. Spatial: morphological open/close, then a minimum connected-component area.
     Kills salt-and-pepper blocks and closes small holes inside a real region.
  2. Temporal: the region must persist for at least min_duration_s. A wingbeat
     lasting 3 frames is flow noise; one lasting 0.5 s is a behavior.
  3. Identity: components are tracked across frames by spatial overlap, so a
     region that moves keeps one ID rather than becoming a new ROI every frame.

ID stability matters beyond bookkeeping: the handoff requires per-ROI notes to
survive a filter retune. We therefore match new ROIs against the previous ROI set
by spatial overlap when re-extracting, and reuse the old ID where they agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np





@dataclass
class ROI:
    roi_id: int
    # Frame indices where this ROI is present (its "active" frames).
    frames: list[int] = field(default_factory=list)
    # Union mask over the block grid, for the thumbnail and for export.
    mask: np.ndarray | None = None
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # y0, x0, y1, x1 in blocks
    note: str = ""
    # Analysis-time standardization metadata. Pixel calibration is expressed in
    # SOURCE pixels/mm; cache downsampling is accounted for during conversion.
    baseline_start_s: float | None = None
    baseline_end_s: float | None = None
    pixels_per_mm: float | None = None
    body_length_mm: float | None = None
    # Source-frame ownership boundary. Atlas bbox/mask are internal cache
    # coordinates in ROI-first caches; the GUI and exports use this fraction.
    source_frac: tuple[float, float, float, float] | None = None

    def duration_s(self, fps: float) -> float:
        return len(self.frames) / fps

    def first_s(self, fps: float) -> float:
        return min(self.frames) / fps if self.frames else 0.0

    def last_s(self, fps: float) -> float:
        return max(self.frames) / fps if self.frames else 0.0

    def to_dict(self, fps: float) -> dict:
        return {
            "roi_id": self.roi_id,
            "bbox_blocks": list(self.bbox),
            "n_frames": len(self.frames),
            "duration_s": self.duration_s(fps),
            "first_s": self.first_s(fps),
            "last_s": self.last_s(fps),
            "note": self.note,
            "baseline_start_s": self.baseline_start_s,
            "baseline_end_s": self.baseline_end_s,
            "pixels_per_mm": self.pixels_per_mm,
            "body_length_mm": self.body_length_mm,
            "source_frac": list(self.source_frac) if self.source_frac else None,
        }


def rect_roi(roi_id: int, frac_box: tuple[float, float, float, float],
             grid: tuple[int, int], n_frames: int, label: str = "",
             baseline_start_s: float | None = None,
             baseline_end_s: float | None = None,
             pixels_per_mm: float | None = None,
             body_length_mm: float | None = None) -> ROI:
    """Build a rectangular ROI from a box given in frame fractions (x0,y0,x1,y1).

    Used by the manual replicate tab: the user draws a box over the video, and it
    becomes an ROI with a rectangular mask over the block grid, present for the
    whole clip. Tab 3 then treats it exactly like an auto-discovered ROI, so the
    behavior classifier does not care whether regions were drawn or found.
    """
    ny, nx = grid
    x0, y0, x1, y1 = frac_box
    bx0, bx1 = int(round(x0 * nx)), int(round(x1 * nx))
    by0, by1 = int(round(y0 * ny)), int(round(y1 * ny))
    bx0, bx1 = max(0, min(bx0, nx - 1)), max(1, min(bx1, nx))
    by0, by1 = max(0, min(by0, ny - 1)), max(1, min(by1, ny))
    if bx1 <= bx0:
        bx1 = bx0 + 1
    if by1 <= by0:
        by1 = by0 + 1

    mask = np.zeros((ny, nx), dtype=bool)
    mask[by0:by1, bx0:bx1] = True
    roi = ROI(roi_id=roi_id, frames=list(range(n_frames)), mask=mask,
              bbox=(by0, bx0, by1, bx1), note=label,
              baseline_start_s=baseline_start_s,
              baseline_end_s=baseline_end_s,
              pixels_per_mm=pixels_per_mm,
              body_length_mm=body_length_mm,
              source_frac=tuple(frac_box))
    return roi


def packed_rect_roi(roi_id: int,
                    source_frac: tuple[float, float, float, float],
                    atlas_bbox: tuple[int, int, int, int],
                    atlas_grid: tuple[int, int], n_frames: int,
                    label: str = "", baseline_start_s: float | None = None,
                    baseline_end_s: float | None = None,
                    pixels_per_mm: float | None = None,
                    body_length_mm: float | None = None) -> ROI:
    """Materialize one exact replicate tile in a packed ROI-first cache."""
    ny, nx = atlas_grid
    y0, x0, y1, x1 = atlas_bbox
    if not (0 <= y0 < y1 <= ny and 0 <= x0 < x1 <= nx):
        raise ValueError(f"Invalid packed bbox {atlas_bbox} for grid {atlas_grid}.")
    mask = np.zeros((ny, nx), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return ROI(
        roi_id=roi_id, frames=list(range(n_frames)), mask=mask,
        bbox=(y0, x0, y1, x1), note=label,
        baseline_start_s=baseline_start_s,
        baseline_end_s=baseline_end_s,
        pixels_per_mm=pixels_per_mm,
        body_length_mm=body_length_mm,
        source_frac=tuple(source_frac),
    )




ROI_FEATURES = {
    "speed_over_baseline_p99",
    "speed_over_auto_noise",
    "speed_mm_s",
    "net_speed_mm_s",
    "speed_body_lengths_s",
}




def _roi_feature_plane(cache, ctx, roi: ROI, feature: str) -> np.ndarray:
    """A whole-grid feature plane with replicate-specific scaling applied.

    Scaling the whole plane (then cropping it to the box) preserves the spatial
    shape needed by connected-component criteria while deriving the denominator
    only from the replicate that owns it.
    """
    if feature not in ROI_FEATURES:
        return ctx.get(feature)

    base_name = "net_speed" if feature == "net_speed_mm_s" else "speed"
    base = np.asarray(ctx.get(base_name), dtype=np.float32)
    ys, xs = np.where(roi.mask)
    if ys.size == 0:
        return np.zeros_like(base)
    inside = base[:, ys, xs]

    if feature == "speed_over_baseline_p99":
        if roi.baseline_start_s is None or roi.baseline_end_s is None or \
                roi.baseline_end_s <= roi.baseline_start_s:
            raise ValueError(
                f"Replicate {roi.roi_id} has no valid quiescent baseline.")
        i0 = max(0, int(np.floor(roi.baseline_start_s * ctx.fps)))
        i1 = min(ctx.n_frames, int(np.ceil(roi.baseline_end_s * ctx.fps)))
        if i1 <= i0:
            raise ValueError(
                f"Replicate {roi.roi_id}'s quiescent baseline is outside the clip.")
        noise = float(np.percentile(inside[i0:i1], 99))
        return (base / max(noise, 1e-6)).astype(np.float32)

    if feature == "speed_over_auto_noise":
        floor_by_frame = np.percentile(inside, 25, axis=1).astype(np.float32)
        # Frame 0 is an alignment sentinel with defined-zero flow, not a noise
        # observation. Collapse the remaining background series to one constant
        # per replicate so real temporal changes are not divided away.
        observed = floor_by_frame[1:] if floor_by_frame.size > 1 else floor_by_frame
        noise = float(np.percentile(observed, 99)) if observed.size else 0.0
        return (base / max(noise, 1e-6)).astype(np.float32)

    if roi.pixels_per_mm is None or roi.pixels_per_mm <= 0:
        raise ValueError(
            f"Replicate {roi.roi_id} has no source-pixels/mm calibration.")
    working_px_per_mm = roi.pixels_per_mm * float(cache.meta.get("downsample", 1.0))
    mm_s = base / max(working_px_per_mm, 1e-12)
    if feature == "speed_body_lengths_s":
        if roi.body_length_mm is None or roi.body_length_mm <= 0:
            raise ValueError(
                f"Replicate {roi.roi_id} has no body-length calibration.")
        mm_s = mm_s / roi.body_length_mm
    return mm_s.astype(np.float32)


def roi_time_series(cache, ctx, roi: ROI, feature: str) -> np.ndarray:
    """Mean of a feature over the ROI's blocks, per frame. (T,) float32.

    Averaged over the ROI's union mask -- not just its active frames -- so the
    series is defined everywhere and you can see what the feature was doing
    before and after the behavior, which is the whole point of the inspector.
    """
    vals = roi_block_values(cache, ctx, roi, feature)
    if vals.shape[1] == 0:
        return np.zeros(ctx.n_frames, np.float32)
    return vals.mean(axis=1, dtype=np.float32).astype(np.float32)


def _band_windows_to_frames(cache, ctx, n_win: int, n_frames: int) -> np.ndarray:
    """Nearest-window index for each frame, so a window-axis feature can be read
    on the frame axis without shifting it in time."""
    fps = ctx.fps
    hop = max(1, int(round(float(cache.meta.get("band_hop_s", 0.25)) * fps)))
    win = max(4, int(round(float(cache.meta.get("band_window_s", 1.0)) * fps)))
    centers = np.arange(n_win) * hop + win // 2
    idx = np.searchsorted(centers, np.arange(n_frames))
    idx = np.clip(idx, 0, n_win - 1)
    prev = np.clip(idx - 1, 0, n_win - 1)
    take_prev = np.abs(centers[prev] - np.arange(n_frames)) <= \
        np.abs(centers[idx] - np.arange(n_frames))
    return np.where(take_prev, prev, idx)


def roi_band_power(cache, ctx, roi: ROI, lo_hz: float, hi_hz: float,
                   reduce: str = "max") -> np.ndarray:
    """Band power for an arbitrary band over a replicate box, per frame.

    The per-block band power is computed ONCE for the whole grid by
    FeatureContext.get (memoized); here we slice the box's blocks and reduce.
    MAX by default, deliberately: a wingbeating insect occupies a handful of
    blocks in a tube-sized box, and averaging over the box buries it. Because the
    overlay thresholds the very same per-block plane, a box's ethogram bar being
    on (max >= threshold) means exactly that at least one of its blocks is lit on
    the video -- the two views cannot disagree.
    """
    ys, xs = np.where(roi.mask)
    if ys.size == 0:
        return np.zeros(ctx.n_frames, np.float32)

    plane = ctx.get(f"bandpower_{lo_hz:g}-{hi_hz:g}Hz")   # (n_win, ny, nx)
    cols = np.asarray(plane[:, ys, xs], dtype=np.float32)  # (n_win, k)
    if reduce == "mean":
        agg = cols.mean(axis=1)
    elif reduce == "p90":
        agg = np.percentile(cols, 90, axis=1)
    else:
        agg = cols.max(axis=1)

    if plane.shape[0] == ctx.n_frames:
        return agg.astype(np.float32)
    idx = _band_windows_to_frames(cache, ctx, plane.shape[0], ctx.n_frames)
    return agg[idx].astype(np.float32)


def parse_band_feature(name: str):
    from core.features import _parse_band
    return _parse_band(name)




def roi_block_values(cache, ctx, roi: ROI, feature: str) -> np.ndarray:
    """Every block's value inside the box, per frame: (T, k) float32.

    This is the distribution that detection actually thresholds, so it -- not the
    box mean -- is what the range editor must show. Setting a range on the mean is
    the classic trap: at the moment the animal moves, the mean is dragged down by
    all the still background blocks, so a range picked off the mean plot catches
    the background everywhere and misses the few blocks that are genuinely fast.

    Computed over the WHOLE grid (via ctx.get) then sliced to the box's blocks.
    A bbox-restricted computation was tried for speed, but the spatial gradient
    features (divergence, curl) develop large boundary artifacts at the box edge,
    which corrupted their distribution -- so this stays whole-grid for
    correctness.
    """
    ys, xs = np.where(roi.mask)
    if ys.size == 0:
        return np.zeros((ctx.n_frames, 0), np.float32)
    arr = _roi_feature_plane(cache, ctx, roi, feature)
    if arr.shape[0] != ctx.n_frames:
        idx = _band_windows_to_frames(cache, ctx, arr.shape[0], ctx.n_frames)
        vals = arr[idx][:, ys, xs]
    else:
        vals = arr[:, ys, xs]
    return np.asarray(vals, dtype=np.float32)




def roi_psd(cache, ctx, roi: ROI, feature: str = "speed"
            ) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density of an ROI's mean feature series. (freqs, psd)."""
    x = roi_time_series(cache, ctx, roi, feature)
    n = x.size
    if n < 8:
        return np.zeros(0), np.zeros(0)
    x = x - x.mean()
    w = np.hanning(n)
    spec = np.fft.rfft(x * w)
    psd = (np.abs(spec) ** 2) / (ctx.fps * np.sum(w ** 2))
    freqs = np.fft.rfftfreq(n, d=1.0 / ctx.fps)
    return freqs, psd.astype(np.float32)




