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

import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ROIParams:
    min_area_blocks: int = 2

    # Morphological opening defaults to OFF, and this is deliberate.
    #
    # Opening with radius r erodes with a (2r+1)x(2r+1) element, so it deletes
    # any component smaller than that element -- radius 1 destroys everything
    # under 3x3 blocks. On the reference footage a wingbeating grasshopper
    # occupies 1 to 4 blocks, so open_radius=1 removed 100% of the matched
    # regions. That is not a tuning accident: "small spatial extent" is part of
    # the very signature this tool exists to find, and opening is hostile to it.
    #
    # Opening is also mostly redundant here. It exists to kill salt-and-pepper
    # noise, but each block is already a 16x16-pixel average, so single-pixel
    # noise is long gone -- and min_area_blocks removes small components anyway,
    # without also shrinking the large ones. Turn it on only if the overlay is
    # visibly speckled with isolated blocks you know are spurious.
    open_radius: int = 0

    # Closing is safe: it fills small holes inside a region without shrinking it.
    close_radius: int = 1

    min_duration_s: float = 0.5
    # Bridge dropouts shorter than this when deciding whether a region persists.
    max_gap_s: float = 0.2

    # Two components in consecutive frames are the same ROI if their bounding
    # boxes overlap by at least this IoU...
    track_iou: float = 0.2
    # ...or, failing that, if their centroids are within this many blocks.
    #
    # The IoU test alone cannot track small regions. A 1-block component that
    # moves by one block has an IoU of exactly 0 with its own previous position,
    # so every frame would start a new ROI and nothing would ever meet the
    # minimum-duration test. Centroid distance degrades gracefully as regions get
    # small, which is the regime that matters here.
    track_max_dist_blocks: float = 3.0


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
              body_length_mm=body_length_mm)
    return roi


def _morph(mask: np.ndarray, open_r: int, close_r: int) -> np.ndarray:
    m = mask.astype(np.uint8)
    if open_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * open_r + 1, 2 * open_r + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * close_r + 1, 2 * close_r + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m.astype(bool)


def _iou(a: tuple, b: tuple) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    ih, iw = max(0, iy1 - iy0), max(0, ix1 - ix0)
    inter = ih * iw
    if inter == 0:
        return 0.0
    area_a = (ay1 - ay0) * (ax1 - ax0)
    area_b = (by1 - by0) * (bx1 - bx0)
    return inter / float(area_a + area_b - inter)


def _centroid(bbox: tuple) -> tuple[float, float]:
    y0, x0, y1, x1 = bbox
    return (0.5 * (y0 + y1), 0.5 * (x0 + x1))


def _centroid_dist(a: tuple, b: tuple) -> float:
    ay, ax = _centroid(a)
    by, bx = _centroid(b)
    return float(np.hypot(ay - by, ax - bx))


def _match_score(a: tuple, b: tuple, params: "ROIParams") -> float:
    """How strongly two components in consecutive frames look like the same
    region. Returns a score in (0, 1], or 0 for no match.

    IoU is preferred where it is meaningful, but it collapses for the small
    regions this tool targets -- a 1-block region that moves one block has zero
    overlap with itself. So we fall back to centroid proximity, scored so that a
    coincident centroid beats a distant one but never outranks a genuine overlap.
    """
    iou = _iou(a, b)
    if iou >= params.track_iou:
        return 1.0 + iou          # a real overlap always wins
    d = _centroid_dist(a, b)
    if d <= params.track_max_dist_blocks:
        return 1.0 - d / (params.track_max_dist_blocks + 1e-6)
    return 0.0


def extract_rois(masks: np.ndarray, fps: float, params: ROIParams,
                 previous: list[ROI] | None = None,
                 progress=None) -> list[ROI]:
    """masks: (T, ny, nx) bool. Returns tracked ROIs meeting all criteria."""
    n_t = masks.shape[0]
    tracks: dict[int, ROI] = {}
    next_id = 1
    # Live tracks: id -> (bbox, last_seen_frame)
    live: dict[int, tuple[tuple, int]] = {}
    max_gap_frames = max(0, int(round(params.max_gap_s * fps)))

    for t in range(n_t):
        m = _morph(masks[t], params.open_radius, params.close_radius)
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(
            m.astype(np.uint8), connectivity=8)

        comps = []
        for lab in range(1, n_lab):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area < params.min_area_blocks:
                continue
            x0 = stats[lab, cv2.CC_STAT_LEFT]
            y0 = stats[lab, cv2.CC_STAT_TOP]
            w = stats[lab, cv2.CC_STAT_WIDTH]
            h = stats[lab, cv2.CC_STAT_HEIGHT]
            comps.append(((y0, x0, y0 + h, x0 + w), labels == lab))

        # Retire tracks that have been gone longer than the gap tolerance.
        for tid in [i for i, (_, last) in live.items()
                    if t - last > max_gap_frames]:
            del live[tid]

        used: set[int] = set()
        for bbox, comp_mask in comps:
            best_id, best_score = None, 0.0
            for tid, (prev_bbox, _) in live.items():
                if tid in used:
                    continue
                score = _match_score(bbox, prev_bbox, params)
                if score > best_score:
                    best_id, best_score = tid, score

            if best_id is None:
                best_id = next_id
                next_id += 1
                tracks[best_id] = ROI(roi_id=best_id,
                                      mask=np.zeros_like(comp_mask, dtype=bool))
            used.add(best_id)
            live[best_id] = (bbox, t)

            roi = tracks[best_id]
            roi.frames.append(t)
            roi.mask |= comp_mask

        if progress and t % 200 == 0:
            progress(t, n_t)

    # Temporal criterion: drop anything too short-lived to be a behavior.
    min_frames = max(1, int(round(params.min_duration_s * fps)))
    kept = [r for r in tracks.values() if len(r.frames) >= min_frames]

    for r in kept:
        ys, xs = np.where(r.mask)
        if ys.size:
            r.bbox = (int(ys.min()), int(xs.min()),
                      int(ys.max()) + 1, int(xs.max()) + 1)

    kept.sort(key=lambda r: (-len(r.frames), r.roi_id))

    if previous:
        kept = _reassign_stable_ids(kept, previous)
    return kept


def _reassign_stable_ids(new: list[ROI], previous: list[ROI]) -> list[ROI]:
    """Give a new ROI the ID of the previous ROI it most overlaps.

    Retuning a filter should not renumber everything and orphan the user's notes.
    Greedy best-overlap matching is enough here -- the alternative (Hungarian) is
    not worth the dependency for a handful of regions.
    """
    taken: set[int] = set()
    prev_by_id = {p.roi_id: p for p in previous}
    scores = []
    for i, n in enumerate(new):
        for p in previous:
            # Same reasoning as the frame-to-frame tracker: bbox IoU alone cannot
            # re-identify a small region, so a nearby centroid also counts as a
            # match. A retune shifts a region's extent slightly; it does not
            # teleport it.
            s = _iou(n.bbox, p.bbox)
            if s <= 0.3:
                d = _centroid_dist(n.bbox, p.bbox)
                s = (1.0 - d / 3.0) * 0.3 if d <= 3.0 else 0.0
            if s > 0:
                scores.append((s, i, p.roi_id))
    scores.sort(reverse=True)

    assigned: dict[int, int] = {}
    for s, i, pid in scores:
        if i in assigned or pid in taken:
            continue
        assigned[i] = pid
        taken.add(pid)

    max_id = max([p.roi_id for p in previous] + [0])
    for i, n in enumerate(new):
        if i in assigned:
            n.roi_id = assigned[i]
            n.note = prev_by_id[assigned[i]].note
        else:
            max_id += 1
            n.roi_id = max_id
    return new


ROI_FEATURES = {
    "speed_over_baseline_p99",
    "speed_over_auto_noise",
    "speed_mm_s",
    "net_speed_mm_s",
    "speed_body_lengths_s",
}


def roi_feature_available(feature: str, rois: list[ROI]) -> bool:
    """Whether every current replicate has metadata required by ``feature``."""
    if feature not in ROI_FEATURES or not rois:
        return False
    if feature == "speed_over_auto_noise":
        return True
    if feature == "speed_over_baseline_p99":
        return all(r.baseline_start_s is not None and
                   r.baseline_end_s is not None and
                   r.baseline_end_s > r.baseline_start_s for r in rois)
    if feature in ("speed_mm_s", "net_speed_mm_s"):
        return all(r.pixels_per_mm is not None and r.pixels_per_mm > 0
                   for r in rois)
    return all(r.pixels_per_mm is not None and r.pixels_per_mm > 0 and
               r.body_length_mm is not None and r.body_length_mm > 0
               for r in rois)


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


def behavior_block_mask(cache, ctx, behavior, frame_idx: int) -> np.ndarray:
    """(ny, nx) bool: blocks where EVERY leaf of the behavior holds at this frame.

    This is the per-block detection the video overlay paints. It evaluates the
    behavior's flat AND of feature ranges directly on the per-block feature
    planes -- the same planes the ROI series are aggregated from -- so what you
    see lit on the video is exactly what drives the ethogram.
    """
    from core.behavior import RangeLeaf

    leaves: list = []

    def walk(node):
        if isinstance(node, RangeLeaf):
            if node.enabled:
                leaves.append(node)
        else:
            for c in getattr(node, "children", []):
                walk(c)

    walk(behavior.spec)
    ny, nx = ctx.speed.shape[1], ctx.speed.shape[2]
    m = np.ones((ny, nx), dtype=bool)
    for leaf in leaves:
        arr = ctx.get(leaf.feature)
        if arr.shape[0] != ctx.n_frames:            # window-axis feature
            w = int(_band_windows_to_frames(cache, ctx, arr.shape[0],
                                            ctx.n_frames)[frame_idx])
            plane = arr[w]
        else:
            plane = arr[min(frame_idx, arr.shape[0] - 1)]
        plane = np.asarray(plane, dtype=np.float32)
        m &= (plane >= leaf.lo) & (plane <= leaf.hi)
    return m


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


def roi_detection(cache, ctx, behavior, roi: ROI
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (detected, strength) for a behavior over one replicate box.

    Instead of asking "does the box's aggregate cross threshold", this counts how
    many blocks inside the box pass every feature range at each frame, applies the
    behavior's spatial criteria (a minimum clump size, optionally after merging
    nearby blocks), and only then the temporal criteria.

    Returns:
        detected : (T,) bool   -- final behavior trace, post temporal criteria.
        strength : (T,) float  -- fraction of the box's blocks passing, in [0,1],
                                  for the graded ethogram. This is the raw signal,
                                  before the block-count / duration thresholds.

    The strength is what makes the tool general: for a wingbeat it is "how much of
    the tube is oscillating", for an ant crossing it is "how much of the gate is
    covered by a moving clump" -- same knob, different behavior.
    """
    from core.behavior import RangeLeaf, apply_temporal

    n_t = ctx.n_frames
    y0, x0, y1, x1 = roi.bbox
    sub_mask = roi.mask[y0:y1, x0:x1]
    total_blocks = int(sub_mask.sum())
    if total_blocks == 0:
        return np.zeros(n_t, bool), np.zeros(n_t, np.float32)

    leaves = [c for c in behavior.spec.children
              if isinstance(c, RangeLeaf) and c.enabled]
    if not leaves:
        return np.zeros(n_t, bool), np.zeros(n_t, np.float32)

    # (T, bh, bw) boolean: every block in the box that passes all ranges, per
    # frame. Whole-grid feature planes cropped to the bbox (see roi_block_values
    # for why not a bbox-restricted computation).
    passing = np.ones((n_t, y1 - y0, x1 - x0), dtype=bool)
    for leaf in leaves:
        arr = _roi_feature_plane(cache, ctx, roi, leaf.feature)
        if arr.shape[0] != n_t:
            idx = _band_windows_to_frames(cache, ctx, arr.shape[0], n_t)
            sub = np.asarray(arr[idx][:, y0:y1, x0:x1], np.float32)
        else:
            sub = np.asarray(arr[:, y0:y1, x0:x1], np.float32)
        passing &= (sub >= leaf.lo) & (sub <= leaf.hi)
    passing &= sub_mask[None, :, :]

    sp = behavior.spatial
    counts = passing.reshape(n_t, -1).sum(axis=1).astype(np.float32)
    strength = counts / max(1, total_blocks)

    # Spatial gate: largest merged clump must reach min_blocks.
    need_cc = sp.min_blocks > 1 or sp.merge_distance > 0
    if need_cc:
        kernel = None
        if sp.merge_distance > 0:
            r = sp.merge_distance
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        largest = np.zeros(n_t, np.int32)
        for t in range(n_t):
            pm = passing[t]
            if not pm.any():
                continue
            m = pm.astype(np.uint8)
            if kernel is not None:
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            n_lab, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if n_lab > 1:
                largest[t] = int(stats[1:, cv2.CC_STAT_AREA].max())
        clump_ok = largest >= sp.min_blocks
    else:
        clump_ok = counts >= 1

    raw = clump_ok & (counts >= sp.min_blocks) & \
        (strength >= sp.min_fraction)
    detected = apply_temporal(raw, ctx.fps, behavior.criteria)
    return detected, strength.astype(np.float32)


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


def blocks_to_pixels(mask: np.ndarray, block: int, out_w: int, out_h: int
                     ) -> np.ndarray:
    """Upscale a block-grid mask to full frame pixels for overlay drawing."""
    m = (mask.astype(np.uint8) * 255)
    return cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_NEAREST)


def save_rois(path: str, rois: list[ROI], fps: float, block: int,
              write_masks: bool = True) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"fps": fps, "block_size": block,
               "rois": [r.to_dict(fps) for r in rois]}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    if write_masks:
        mask_dir = os.path.splitext(path)[0] + "_masks"
        os.makedirs(mask_dir, exist_ok=True)
        for r in rois:
            if r.mask is not None:
                cv2.imwrite(os.path.join(mask_dir, f"roi_{r.roi_id:03d}.png"),
                            r.mask.astype(np.uint8) * 255)
