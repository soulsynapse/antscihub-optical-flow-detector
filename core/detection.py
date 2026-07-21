"""Shared detection math for the tensor / scalogram path.

The scalogram explorer computes a detection incrementally across several methods:
band power -> in-band block count -> windowed mean over the detection window D ->
positive gate -> largest connected clump. The whole-video "commit" pass has to
produce the SAME numbers block-for-block -- otherwise you navigate to a detection
found over the whole clip, re-open its window to verify, and see a different
result, which would make the tool untrustworthy at exactly the moment it matters.

So the formulas live here as pure functions that BOTH the explorer and the
whole-video pass call. The only heavy part -- the Morlet band power -- is provided
memory-bounded by ``core.wavelet.morlet_band_power`` (chunked over blocks, no full
(F,T,B) cube), so the same detector runs over a 10 s window or a whole clip.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.timing import Timer
from core.wavelet import band_indices, default_freqs, morlet_band_power


def region_bbox(meta: dict, region_index: int) -> tuple[int, int, int, int]:
    """A region's ``(y0, x0, y1, x1)`` block bounding box from a cache-shaped
    meta. One region covering the whole grid when there are no replicate tiles."""
    tiles = meta.get("replicate_tiles")
    if tiles:
        return tuple(int(v) for v in tiles[region_index]["atlas_bbox"])
    ny, nx = (int(v) for v in meta["grid"])
    return 0, 0, ny, nx


def region_grid_from_meta(meta: dict, region_index: int):
    """``(n_blocks, (dy, dx, gy, gx))`` for a region, WITHOUT the data.

    Split out so a caller that needs the clump grid and the block count before
    any frames exist -- the live accumulator, which must stamp a span with what
    its count band is denominated in -- does not have to fabricate an array to
    ask. :func:`region_blocks_and_grid` is the same derivation with the slice.
    """
    y0, x0, y1, x1 = region_bbox(meta, region_index)
    dy, dx = y1 - y0, x1 - x0
    gy, gx = np.mgrid[0:dy, 0:dx]
    return dy * dx, (dy, dx, gy.ravel(), gx.ravel())


def region_blocks_and_grid(meta: dict, channel_arr: np.ndarray,
                           region_index: int):
    """A region's (T, B) block columns plus its (dy, dx, gy, gx) clump grid, from
    a cache-shaped meta. Column order and 0-based grid match the explorer's
    _scope_blocks / _make_snap, so a whole-clip pass indexes blocks identically."""
    T = channel_arr.shape[0]
    y0, x0, y1, x1 = region_bbox(meta, region_index)
    blocks = channel_arr[:, y0:y1, x0:x1].reshape(T, -1)
    _n, grid = region_grid_from_meta(meta, region_index)
    return blocks, grid


def inband_count(m: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """# blocks whose band power lies in [lo, hi] per frame. ``m`` is (T, B)."""
    m = np.asarray(m)
    inband = (m >= lo) & (m <= hi) & np.isfinite(m)
    return inband.sum(axis=1).astype(np.float32)


def window_bounds(T: int, W: int, centered: bool):
    """Per-frame prefix-sum bounds for a detection window of W frames; centered
    [t-D/2, t+D/2] or trailing [t-D+1, t]. Windows truncate at the clip edges;
    the returned effective length keeps the means honest there."""
    t = np.arange(T)
    if centered:
        lo = np.maximum(0, t - W // 2)
        hi = np.minimum(T, t + (W - W // 2))
    else:
        hi = t + 1
        lo = np.maximum(0, hi - W)
    return hi, lo, (hi - lo).astype(np.float32)


def windowed_mean(count: np.ndarray, W: int, centered: bool) -> np.ndarray:
    """Centered/trailing mean of a per-frame series over W frames, so a 1-frame
    spike of N blocks dilutes to N/W and cannot fake a sustained event."""
    count = np.asarray(count, np.float32)
    T = count.shape[0]
    if T == 0 or W <= 1:
        return count.astype(np.float32)
    c = np.concatenate([[0.0], np.cumsum(count, dtype=np.float64)])
    hi, lo, neff = window_bounds(T, W, centered)
    return ((c[hi] - c[lo]) / neff).astype(np.float32)


def detect_gate(windowed: np.ndarray, blo: float, bhi: float) -> np.ndarray:
    """Positive detection: windowed in-band count within [blo, bhi]."""
    windowed = np.asarray(windowed)
    return ((windowed >= blo) & (windowed <= bhi)).astype(np.float32)


def rescale_count_band(band, old_blocks: int, new_blocks: int):
    """Carry a ``count_band`` across a change of block grid.

    ``inband_count`` returns a RAW BLOCK COUNT and ``detect_gate`` compares it
    straight against the band endpoints -- nothing normalizes by how many blocks
    the region holds. So a band tuned on one grid means something different on
    another by exactly the ratio of region sizes: ~29 blocks per replicate at
    block 64 against ~377 at block 16, so ``[20, inf)`` is a sane threshold on
    one and unreachable on the other (FINDINGS.md section 5). The detector does
    not fail or clamp when that happens; it silently stops firing, which reads
    as "nothing happened".

    The factor is ``new_blocks / old_blocks`` measured from the ACTUAL region
    block counts, not from the block-size ratio squared. The two disagree
    whenever the grid does not divide the working frame evenly -- the ragged
    last row and column are dropped, and by a different amount at each block
    size -- and the actual count is the quantity the detector really compares
    against.

    Preserved deliberately:

    * ``None`` endpoints stay ``None``. They mean "never placed", which the
      widget seeds lazily; converting one would invent an explicit threshold.
    * Infinite endpoints stay infinite. "Unbounded above" is not a count and
      does not re-denominate.
    * Zero endpoints survive the multiply unchanged, which is correct: "at
      least none" is grid-independent.

    Returns the band unchanged when either side of the ratio is unusable, so a
    caller can pass a degenerate geometry without special-casing it.
    """
    lo, hi = band
    if old_blocks <= 0 or new_blocks <= 0 or old_blocks == new_blocks:
        return (lo, hi)
    factor = float(new_blocks) / float(old_blocks)

    def conv(v):
        if v is None or not np.isfinite(v):
            return v
        return float(v) * factor

    return (conv(lo), conv(hi))


def largest_clump_per_frame(m: np.ndarray, lo: float, hi: float, dy: int, dx: int,
                            gy: np.ndarray, gx: np.ndarray) -> np.ndarray:
    """Largest in-band, 8-connected clump per frame on the block grid. ``m`` is
    (T, B) for one region; gy/gx map its block columns to 0-based grid cells."""
    import cv2
    m = np.asarray(m)
    T = m.shape[0]
    out = np.zeros(T, np.float32)
    if m.size == 0 or m.shape[1] != gy.size:
        return out
    # One grid reused across frames rather than a fresh (dy, dx) allocation per
    # frame: this loop runs once per frame over a whole clip, and gy/gx cover
    # every cell, so each used frame overwrites the whole grid anyway. The
    # per-frame connected-components call is the irreducible cost; the array
    # churn around it is not.
    grid = np.zeros((dy, dx), np.uint8)
    passing_all = (m >= lo) & (m <= hi) & np.isfinite(m)
    for t in range(T):
        passing = passing_all[t]
        if not passing.any():
            continue
        grid[gy, gx] = passing.view(np.uint8)
        n_lab, _, stats, _ = cv2.connectedComponentsWithStats(
            grid, connectivity=8)
        if n_lab > 1:
            out[t] = float(stats[1:, cv2.CC_STAT_AREA].max())
    return out


@dataclass
class DetectionResult:
    """One detector run over a channel's block columns. ``band_power`` (T, B) is
    retained so value-band and detection-window re-tuning is instant; changing the
    frequency band or channel requires a fresh pass (the band sum is baked in)."""
    band_power: np.ndarray        # (T, B)
    count: np.ndarray             # (T,)  # blocks in the value band
    windowed: np.ndarray          # (T,)  windowed mean over D
    gate: np.ndarray              # (T,)  positive detection (0/1)
    clump: np.ndarray             # (T,)  largest connected clump area
    freq_band_hz: tuple
    value_band: tuple
    count_band: tuple
    detect_window: int
    centered: bool
    window_start: int = 0

    def detected_intervals(self) -> list[tuple[int, int]]:
        """Contiguous [start, end) frame runs where the gate is on (frame indices
        are absolute: window_start is added back)."""
        g = np.asarray(self.gate) > 0.5
        if not g.any():
            return []
        edges = np.diff(np.concatenate([[0], g.view(np.int8), [0]]))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        return [(int(s + self.window_start), int(e + self.window_start))
                for s, e in zip(starts, ends)]


def recompute_from_band_power(band_power: np.ndarray, *, value_band, count_band,
                              detect_window, centered, region_grid=None,
                              freq_band_hz=(float("-inf"), float("inf")),
                              window_start=0) -> DetectionResult:
    """Cheap re-tune: value band / detection window / count band change without a
    fresh transform, operating on retained (T, B) band power."""
    tm = Timer("recompute_from_band_power")
    bp = np.asarray(band_power, np.float32)
    lo, hi = value_band
    with tm.span("count_window_gate"):
        count = inband_count(bp, lo, hi)
        windowed = windowed_mean(count, detect_window, centered)
        blo, bhi = count_band
        gate = detect_gate(windowed, blo, bhi)
    if region_grid is not None and bp.size:
        dy, dx, gy, gx = region_grid
        # Per-frame connected components: a Python loop over T, so it is the one
        # part of the cheap re-tune path that is not obviously cheap.
        with tm.span("clump"):
            clump = largest_clump_per_frame(bp, lo, hi, dy, dx, gy, gx)
    else:
        clump = np.zeros(bp.shape[0], np.float32)
    tm.log(min_seconds=0.05, T=bp.shape[0] if bp.ndim else 0,
           B=bp.shape[1] if bp.ndim > 1 else 0)
    return DetectionResult(
        band_power=bp, count=count, windowed=windowed, gate=gate, clump=clump,
        freq_band_hz=tuple(freq_band_hz), value_band=(lo, hi),
        count_band=(blo, bhi), detect_window=int(detect_window),
        centered=bool(centered), window_start=int(window_start))


def detect_channel_region(channel_data, region_index: int, channel_attr: str, *,
                          freq_band_hz, value_band, count_band, detect_window,
                          centered, freqs=None, block_chunk: int = 512
                          ) -> DetectionResult:
    """Run the detector over ONE region of an already-extracted ChannelData -- the
    whole-clip commit path. ``channel_data`` is duck-typed (needs ``.meta`` and
    ``.channels``), so this stays cache/GUI-agnostic. ``freqs`` defaults to the
    explorer's ``default_freqs(fps)`` so the pass and the preview share the bank."""
    meta = channel_data.meta
    fps = float(meta["fps"])
    if freqs is None:
        freqs = default_freqs(fps)
    arr = np.asarray(channel_data.channels[channel_attr], np.float32)
    blocks, region_grid = region_blocks_and_grid(meta, arr, region_index)
    return detect_over_blocks(
        blocks, fps, freqs, freq_band_hz=freq_band_hz, value_band=value_band,
        count_band=count_band, detect_window=detect_window, centered=centered,
        region_grid=region_grid,
        window_start=int(getattr(channel_data, "window_start", 0)),
        block_chunk=block_chunk)


def detect_over_blocks(blocks: np.ndarray, fps: float, freqs: np.ndarray, *,
                       freq_band_hz, value_band, count_band, detect_window,
                       centered, region_grid=None, window_start=0,
                       block_chunk: int = 512) -> DetectionResult:
    """Full detector over one region's per-block channel columns (T, B): Morlet
    band power (memory-bounded), then the shared count/window/gate/clump chain."""
    tm = Timer("detect_over_blocks")
    blocks = np.asarray(blocks, np.float32)
    flo, fhi = freq_band_hz
    i, j = band_indices(freqs, flo, fhi)
    with tm.span("morlet_band_power"):
        bp = morlet_band_power(blocks, fps, freqs, i, j, block_chunk=block_chunk)
    # The count/gate/clump chain is part of this pass's cost, so it is timed here
    # and logged after -- logging before the call would report the transform alone
    # and hide the per-frame clump loop, which is the expensive half over a whole
    # clip. The inner Timer still logs separately for the cheap re-tune path.
    with tm.span("recompute"):
        res = recompute_from_band_power(
            bp, value_band=value_band, count_band=count_band,
            detect_window=detect_window, centered=centered,
            region_grid=region_grid, freq_band_hz=(flo, fhi),
            window_start=window_start)
    tm.log(T=blocks.shape[0] if blocks.ndim else 0,
           B=blocks.shape[1] if blocks.ndim > 1 else 0,
           freqs=len(freqs), band=f"{i}:{j}")
    return res
