"""Filter state and cross-filtered histograms -- the interaction model of the
whole tool.

The requirement is Lightroom's: every feature gets a histogram with a draggable
range band, and moving the range on histogram A immediately redraws histogram B
to show only the data that survives A. That is what makes multi-feature tuning
tractable, because you can see whether the intersection is actually separable
from the noise or whether you are just carving up one big blob.

Doing that literally -- re-reading the cache on every mouse-move -- is hopeless:
the full clip is 111 million block-frames. So we take a fixed random sample of
block-frames ONCE (a few million), materialize every opted-in feature over that
sample into a dense float32 matrix, and answer every histogram and intersection
query from RAM. A drag then costs a handful of numpy comparisons over a few
million rows, which is sub-frame.

The sample is drawn once and reused, deliberately. If it were redrawn per query,
the histograms would shimmer as the user dragged, and small populations (which is
exactly what a rare behavior is) would flicker in and out of existence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.features import REGISTRY, FeatureContext

# Enough that a behavior occupying 0.01% of block-frames still lands ~200 samples
# in the histogram, which is plenty to see a mode. Costs ~8 MB per feature.
DEFAULT_SAMPLE_SIZE = 2_000_000
DEFAULT_BINS = 128


@dataclass
class Range:
    lo: float
    hi: float
    enabled: bool = True

    def contains(self, arr: np.ndarray) -> np.ndarray:
        return (arr >= self.lo) & (arr <= self.hi)


@dataclass
class FilterState:
    """The set of active feature ranges. Shared by Tabs 2 and 3."""
    ranges: dict[str, Range] = field(default_factory=dict)

    def active(self) -> dict[str, Range]:
        return {k: r for k, r in self.ranges.items() if r.enabled}

    def set_range(self, name: str, lo: float, hi: float) -> None:
        if name in self.ranges:
            self.ranges[name].lo = lo
            self.ranges[name].hi = hi
        else:
            self.ranges[name] = Range(lo, hi)

    def to_dict(self) -> dict:
        return {k: {"lo": float(r.lo), "hi": float(r.hi), "enabled": r.enabled}
                for k, r in self.ranges.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "FilterState":
        return cls(ranges={k: Range(v["lo"], v["hi"], v.get("enabled", True))
                           for k, v in d.items()})


class FeatureSampler:
    """A fixed random sample of block-frames, with features materialized on it.

    Features are added lazily: a feature is only computed over the sample when
    the user actually opts it into the histogram panel, which keeps opening a
    cache cheap even when the registry is large.
    """

    def __init__(self, cache, ctx: FeatureContext,
                 sample_size: int = DEFAULT_SAMPLE_SIZE, seed: int = 0):
        self.cache = cache
        self.ctx = ctx
        ny, nx = cache.grid
        n_t = ctx.n_frames
        total = n_t * ny * nx

        rng = np.random.default_rng(seed)
        n = min(sample_size, total)
        if total > sample_size:
            # Sample WITH replacement, deliberately. rng.choice(total, replace=False)
            # builds a full permutation of `total` internally -- on the reference
            # clip that is 111 million int64s, i.e. ~900 MB allocated just to draw
            # 2 million indices. Sampling with replacement is O(n) and, at a
            # sampling fraction of 2%, produces a duplicate rate of ~1% -- utterly
            # irrelevant to a histogram's shape.
            flat = rng.integers(0, total, size=n, dtype=np.int64)
        else:
            flat = np.arange(total, dtype=np.int64)
        self.t_idx = (flat // (ny * nx)).astype(np.int32)
        rem = flat % (ny * nx)
        self.y_idx = (rem // nx).astype(np.int32)
        self.x_idx = (rem % nx).astype(np.int32)

        self._cols: dict[str, np.ndarray] = {}
        self._edges: dict[str, np.ndarray] = {}

    @property
    def n(self) -> int:
        return self.t_idx.size

    def column(self, name: str) -> np.ndarray:
        """The feature's values over the sample, computed on first request."""
        if name in self._cols:
            return self._cols[name]
        arr = self.ctx.get(name)
        if arr.shape[0] != self.ctx.n_frames:
            # A band-power / window-axis feature. Map each sampled frame to its
            # nearest STFT window rather than assuming the axes line up -- they
            # do not, and silently indexing with a frame number here would read
            # the wrong window and quietly corrupt every histogram built on it.
            w_idx = self.cache.band_frame_index(self.t_idx)
            col = arr[w_idx, self.y_idx, self.x_idx]
        else:
            col = arr[self.t_idx, self.y_idx, self.x_idx]
        col = np.asarray(col, dtype=np.float32)
        self._cols[name] = col
        return col

    def edges(self, name: str, bins: int = DEFAULT_BINS) -> np.ndarray:
        """Histogram bin edges, clipped to the feature's suggested percentiles.

        Flow magnitude distributions are violently heavy-tailed: a raw min/max
        axis puts 99.9% of the data in the leftmost bin and spends the rest of
        the plot on a handful of outlier blocks. The registry declares sensible
        clip percentiles per feature; we honour them.
        """
        key = f"{name}:{bins}"
        if key in self._edges:
            return self._edges[key]
        col = self.column(name)
        spec = REGISTRY.get(name)
        lo_p, hi_p = spec.clip_pct if spec else (0.5, 99.5)
        lo = float(np.percentile(col, lo_p))
        hi = float(np.percentile(col, hi_p))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(col.min()), float(col.max()) or 1.0
            if hi <= lo:
                hi = lo + 1.0
        e = np.linspace(lo, hi, bins + 1, dtype=np.float32)
        self._edges[key] = e
        return e

    def mask_excluding(self, state: FilterState, exclude: str | None) -> np.ndarray:
        """Boolean mask over the sample: rows passing every active range EXCEPT
        `exclude`. This is what cross-filtering means -- histogram B shows what
        survives every other filter, so you can see where to put B's own range."""
        act = state.active()
        m = np.ones(self.n, dtype=bool)
        for name, r in act.items():
            if name == exclude:
                continue
            m &= r.contains(self.column(name))
        return m

    def mask_all(self, state: FilterState) -> np.ndarray:
        return self.mask_excluding(state, exclude=None)

    def histogram(self, name: str, state: FilterState,
                  bins: int = DEFAULT_BINS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (edges, total_counts, filtered_counts).

        total_counts is the unconditional distribution, drawn dim in the
        background so the user never loses sight of the full population.
        filtered_counts is the cross-filtered distribution, drawn bright.
        """
        col = self.column(name)
        e = self.edges(name, bins)
        total, _ = np.histogram(col, bins=e)
        m = self.mask_excluding(state, exclude=name)
        filtered, _ = np.histogram(col[m], bins=e)
        return e, total, filtered

    def match_fraction(self, state: FilterState) -> float:
        if not state.active():
            return 1.0
        return float(self.mask_all(state).mean())


def frame_mask(cache, ctx: FeatureContext, state: FilterState,
               frame_idx: int) -> np.ndarray:
    """The (ny, nx) boolean overlay for one frame: blocks passing every range.

    Used to paint the live "matched pixels" overlay in Tab 2. Computed from the
    full cache rather than the sample, because an overlay has to be exact.
    """
    ny, nx = cache.grid
    m = np.ones((ny, nx), dtype=bool)
    for name, r in state.active().items():
        arr = ctx.get(name)
        if arr.shape[0] != ctx.n_frames:
            w = int(cache.band_frame_index(frame_idx))
            plane = arr[min(w, arr.shape[0] - 1)]
        else:
            plane = arr[min(frame_idx, arr.shape[0] - 1)]
        m &= r.contains(np.asarray(plane, dtype=np.float32))
    return m


def all_frame_masks(cache, ctx: FeatureContext, state: FilterState) -> np.ndarray:
    """(T, ny, nx) boolean mask over the whole clip. Used by ROI extraction."""
    n_t = ctx.n_frames
    ny, nx = cache.grid
    m = np.ones((n_t, ny, nx), dtype=bool)
    for name, r in state.active().items():
        arr = ctx.get(name)
        if arr.shape[0] != n_t:
            w_idx = cache.band_frame_index(np.arange(n_t))
            arr = arr[np.clip(w_idx, 0, arr.shape[0] - 1)]
        m &= r.contains(np.asarray(arr, dtype=np.float32))
    return m
