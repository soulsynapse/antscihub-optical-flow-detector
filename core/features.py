"""Declarative feature registry.

Adding a feature is a matter of registering it here, not editing the UI. The
inspectors and Behavior Classification build their feature choices from the
registry, so a new entry becomes available automatically.

Every feature declares whether it is CACHED (written during processing) or
DERIVED (recomputed on demand from cached arrays, memoized in RAM for
the session). The rule for which is which:

  Cache it if it is fundamental, or expensive to recompute from scratch.
  Derive it if it is a cheap transform of something already cached.

Concretely: the block-mean flow vector (u, v) and block-mean speed are cached
because they require the optical flow pass, which is the expensive step and can
never be recovered from anything else. Band-power is cached because an STFT over
the whole clip is not something you can afford to redo on every histogram drag.
Everything else -- angle, coherence, net flow, divergence, curl, rolling stats,
dominant frequency, spectral flatness -- is arithmetic on those arrays and is
computed lazily, so it costs zero disk and is available in Tabs 2/3 without
re-running the flow pass.

All arrays are (T, ny, nx): time-major, so a time slice is contiguous. This
matches the on-disk chunking, which is also time-major.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import cv2
import numpy as np

Kind = Literal["cached", "derived", "roi"]
Domain = Literal["spatial", "temporal"]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    units: str
    kind: Kind
    # "spatial" features have a value per block per frame and are meaningful as
    # a spatial overlay. "temporal" features characterize a time series and only
    # mean something over a window (Behavior Classification's domain).
    domain: Domain
    compute: Callable[["FeatureContext"], np.ndarray] | None = None
    deps: tuple[str, ...] = ()
    help: str = ""
    # Suggested histogram clipping percentiles; flow magnitude distributions are
    # extremely heavy-tailed, so a raw min/max axis wastes the whole plot on
    # outliers and leaves the signal in the leftmost pixel column.
    clip_pct: tuple[float, float] = (0.5, 99.5)


class FeatureContext:
    """The cached arrays a derived feature is allowed to read.

    Holds a (possibly partial) time slice so a view can compute a derived feature
    over just the frames it is showing, rather than the whole clip.
    """

    def __init__(self, u: np.ndarray, v: np.ndarray, speed: np.ndarray,
                 fps: float, block_size: int,
                 bands: dict[str, np.ndarray] | None = None,
                 band_times_s: np.ndarray | None = None,
                 t0: int = 0,
                 band_window_s: float = 1.0, band_hop_s: float = 0.25,
                 regions: list[tuple[int, int, int, int]] | None = None):
        self.u = u.astype(np.float32, copy=False)
        self.v = v.astype(np.float32, copy=False)
        self.speed = speed.astype(np.float32, copy=False)
        self.fps = fps
        self.block_size = block_size
        self.bands = bands or {}
        self.band_times_s = band_times_s
        self.t0 = t0
        # STFT parameters for on-demand band power, matched to the cache so a
        # custom band lands on the same window axis as any cached band.
        self.band_window_s = band_window_s
        self.band_hop_s = band_hop_s
        self.regions = regions or [(0, 0, speed.shape[1], speed.shape[2])]
        self.n_band_windows = next(
            (arr.shape[0] for arr in self.bands.values()), None)
        self._memo: dict[str, np.ndarray] = {}

    @property
    def n_frames(self) -> int:
        return self.u.shape[0]

    def times_s(self) -> np.ndarray:
        """Frame times in seconds. A method, not a property, to match
        FeatureCacheBase.times_s() -- the two are used interchangeably by
        callers and a silent property/method mismatch is a trap."""
        return (np.arange(self.n_frames) + self.t0) / self.fps

    def get(self, name: str) -> np.ndarray:
        """Fetch a feature by name, computing and memoizing it if derived.

        A feature that is present on disk always wins over its derived
        implementation, even if the registry marks it "derived". That is how the
        opt-in cache expansions pay off: caching a derived feature during the
        processing pass does not change what it *means*, only where it comes from.
        """
        if name == "u":
            return self.u
        if name == "v":
            return self.v
        if name == "speed":
            return self.speed
        if name in self._memo:
            return self._memo[name]
        if name in self.bands:
            return self.bands[name]

        # Band power for an arbitrary band, computed per block on demand from the
        # cached speed series and memoized. This is what makes band power behave
        # like every other derived feature -- you can ask for any pass-band in
        # Behavior Classification without having cached it. The per-block plane
        # returned here is the single source of truth: the video overlay thresholds it,
        # and a replicate's ROI series is the max of it over the box, so a box's
        # ethogram bar is on exactly when one of its blocks is lit on the video.
        band = _parse_band(name)
        if band is not None:
            arr = self._compute_block_band_power(*band)
            self._memo[name] = arr
            return arr

        spec = REGISTRY.get(name)
        if spec is None:
            raise KeyError(f"Unknown feature: {name}")
        if spec.kind == "cached":
            raise KeyError(
                f"Feature '{name}' can only come from the cache, and this cache "
                f"does not contain it. Re-run Preprocessing & Flow with it enabled."
            )
        if spec.kind == "roi":
            raise KeyError(
                f"Feature '{name}' needs replicate-specific metadata and can "
                "only be evaluated inside a replicate box."
            )
        arr = spec.compute(self)
        self._memo[name] = arr
        return arr

    def _compute_block_band_power(self, lo_hz: float, hi_hz: float) -> np.ndarray:
        """Per-block band power (n_win, ny, nx) for [lo_hz, hi_hz] from speed.

        Row-chunked so it never materializes the win-expanded array for the whole
        grid at once -- the same memory discipline as the offline band-power pass,
        because at small block sizes the grid is large.
        """
        T, ny, nx = self.speed.shape
        fps = self.fps
        win = max(4, int(round(self.band_window_s * fps)))
        hop = max(1, int(round(self.band_hop_s * fps)))
        if T < win:
            win = max(4, T // 2)
            hop = max(1, win // 4)
        n_win = 1 + (T - win) // hop
        freqs = np.fft.rfftfreq(win, d=1.0 / fps)
        band = (freqs >= lo_hz) & (freqs <= hi_hz)
        w = np.hanning(win).astype(np.float32)
        norm = fps * float(np.sum(w ** 2))
        df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

        out = np.zeros((n_win, ny, nx), np.float32)
        rows = max(1, min(ny, 20_000_000 // max(1, n_win * win * nx)))
        for r0 in range(0, ny, rows):
            r1 = min(ny, r0 + rows)
            flat = np.ascontiguousarray(
                self.speed[:, r0:r1, :].reshape(T, (r1 - r0) * nx))
            s0, s1 = flat.strides
            wins = np.lib.stride_tricks.as_strided(
                flat, shape=(n_win, win, flat.shape[1]),
                strides=(s0 * hop, s0, s1), writeable=False)
            seg = (wins - wins.mean(axis=1, keepdims=True)) * w[None, :, None]
            psd = (np.abs(np.fft.rfft(seg, axis=1)).astype(np.float32) ** 2) / np.float32(norm)
            bp = psd[:, band, :].sum(axis=1) * df if band.any() \
                else np.zeros((n_win, flat.shape[1]), np.float32)
            out[:, r0:r1, :] = bp.reshape(n_win, r1 - r0, nx)
        return out


# -- derived feature implementations -----------------------------------------

_EPS = 1e-6


def _parse_band(name: str) -> tuple[float, float] | None:
    """'bandpower_12-24Hz' -> (12.0, 24.0), else None."""
    if not name.startswith("bandpower_") or not name.endswith("Hz"):
        return None
    try:
        lo, hi = name[len("bandpower_"):-2].split("-")
        return float(lo), float(hi)
    except (ValueError, IndexError):
        return None


def _net_speed(ctx: FeatureContext) -> np.ndarray:
    """Magnitude of the block-mean flow vector: net translation of the block."""
    return np.hypot(ctx.u, ctx.v)


def _angle(ctx: FeatureContext) -> np.ndarray:
    """Direction of net flow, in degrees [0, 360).

    Computed from the mean vector, which is the magnitude-weighted circular mean
    -- the only correct way to average direction over a block.
    """
    a = np.degrees(np.arctan2(ctx.v, ctx.u))
    return np.mod(a, 360.0).astype(np.float32)


def _coherence(ctx: FeatureContext) -> np.ndarray:
    """|mean vector| / mean|vector|, in [0, 1].

    1 = every pixel in the block moves the same direction (rigid translation).
    0 = motion that cancels out within the block (oscillation, e.g. a wingbeat,
    or a boundary between two things moving oppositely).
    """
    return (_net_speed(ctx) / (ctx.speed + _EPS)).clip(0.0, 1.0).astype(np.float32)


def _median3(arr: np.ndarray, regions) -> np.ndarray:
    """Spatial 3x3 median independently inside each replicate tile."""
    out = np.zeros_like(arr, dtype=np.float32)
    for y0, x0, y1, x1 in regions:
        sub = np.asarray(arr[:, y0:y1, x0:x1], np.float32)
        for t in range(arr.shape[0]):
            out[t, y0:y1, x0:x1] = cv2.medianBlur(sub[t], 3)
    return out


def _median3_u(ctx: FeatureContext) -> np.ndarray:
    return _median3(ctx.u, ctx.regions)


def _median3_v(ctx: FeatureContext) -> np.ndarray:
    return _median3(ctx.v, ctx.regions)


def _median3_speed(ctx: FeatureContext) -> np.ndarray:
    return _median3(ctx.speed, ctx.regions)


def _median3_net_speed(ctx: FeatureContext) -> np.ndarray:
    return np.hypot(ctx.get("median3_u"), ctx.get("median3_v")).astype(np.float32)


def _texture_percentile(ctx: FeatureContext) -> np.ndarray:
    """Per-frame empirical percentile of block texture, in [0, 1].

    This turns the handoff's ``texture < q_alpha(frame)`` rule into an ordinary,
    tunable range constraint: keep ``texture_percentile >= alpha``.  It remains
    continuous; no alpha or binary mask is baked into the cache.
    """
    texture = np.asarray(ctx.get("texture_min_eigen"), dtype=np.float32)
    out = np.zeros_like(texture, dtype=np.float32)
    for y0, x0, y1, x1 in ctx.regions:
        flat = texture[:, y0:y1, x0:x1].reshape(texture.shape[0], -1)
        n = flat.shape[1]
        if n == 0:
            continue
        for t in range(flat.shape[0]):
            ordered = np.sort(flat[t])
            # Right-sided empirical CDF matches the strict mask rule x < q_alpha:
            # values tied at the quantile itself are retained rather than split.
            ranks = np.searchsorted(ordered, flat[t], side="right") / n
            out[t, y0:y1, x0:x1] = ranks.reshape(y1 - y0, x1 - x0)
    return out


def _divergence(ctx: FeatureContext) -> np.ndarray:
    """du/dx + dv/dy on the block grid. Positive = expansion (looming/spreading)."""
    out = np.zeros_like(ctx.u, dtype=np.float32)
    for y0, x0, y1, x1 in ctx.regions:
        u = ctx.u[:, y0:y1, x0:x1]
        v = ctx.v[:, y0:y1, x0:x1]
        du_dx = np.gradient(u, axis=2) if u.shape[2] > 1 else np.zeros_like(u)
        dv_dy = np.gradient(v, axis=1) if v.shape[1] > 1 else np.zeros_like(v)
        out[:, y0:y1, x0:x1] = du_dx + dv_dy
    return out


def _curl(ctx: FeatureContext) -> np.ndarray:
    """dv/dx - du/dy on the block grid. Nonzero = rotation."""
    out = np.zeros_like(ctx.v, dtype=np.float32)
    for y0, x0, y1, x1 in ctx.regions:
        u = ctx.u[:, y0:y1, x0:x1]
        v = ctx.v[:, y0:y1, x0:x1]
        dv_dx = np.gradient(v, axis=2) if v.shape[2] > 1 else np.zeros_like(v)
        du_dy = np.gradient(u, axis=1) if u.shape[1] > 1 else np.zeros_like(u)
        out[:, y0:y1, x0:x1] = dv_dx - du_dy
    return out


def _rolling(arr: np.ndarray, win: int, fn: str) -> np.ndarray:
    """Centered rolling statistic along the time axis, via cumulative sums."""
    win = max(1, int(win))
    if win == 1:
        return arr if fn == "mean" else np.zeros_like(arr)
    pad = win // 2
    padded = np.pad(arr, ((pad, win - 1 - pad), (0, 0), (0, 0)), mode="edge")
    cs = np.cumsum(padded, axis=0, dtype=np.float64)
    cs = np.concatenate([np.zeros((1,) + arr.shape[1:]), cs], axis=0)
    mean = (cs[win:] - cs[:-win]) / win
    if fn == "mean":
        return mean.astype(np.float32)
    cs2 = np.cumsum(padded.astype(np.float64) ** 2, axis=0)
    cs2 = np.concatenate([np.zeros((1,) + arr.shape[1:]), cs2], axis=0)
    meansq = (cs2[win:] - cs2[:-win]) / win
    var = np.maximum(meansq - mean ** 2, 0.0)
    return np.sqrt(var).astype(np.float32)


def _rolling_mean_speed(ctx: FeatureContext) -> np.ndarray:
    return _rolling(ctx.speed, int(round(0.5 * ctx.fps)), "mean")


def _rolling_std_speed(ctx: FeatureContext) -> np.ndarray:
    """Temporal variability of speed. A block containing a wingbeat has high
    rolling std and near-zero net flow; a walking ant has both high."""
    return _rolling(ctx.speed, int(round(0.5 * ctx.fps)), "std")


# Spatial percentile that defines the "still scene" level a frame is normalized
# against. 40 sits safely inside the background: on this footage most blocks are
# static PVC, so p40 lands on tube, not on the animal.
_REL_BG_PCT = 40.0
# Additive floor (px/s) so a frame with almost no motion anywhere does not divide
# by ~0 and manufacture huge ratios. Small relative to a typical background
# (~5-15 px/s here); it only bites in degenerate near-still frames.
_REL_BG_FLOOR = 1.0


def _rel_speed(ctx: FeatureContext) -> np.ndarray:
    """Rolling mean speed divided by the frame's own background motion level.

    This is the feature that carries ONE threshold across replicates of different
    brightness. Raw speed cannot: dense flow magnitude scales with image contrast,
    so a backlit tube produces 2-3x less flow for the same wingbeat, and no fixed
    px/s cutoff spans both a bright and a dim replicate. Dividing each frame by its
    own background (the spatial p40 across the grid, which is dominated by the
    static scene) cancels that contrast factor: the numerator and denominator scale
    together, so the ratio -- "how much faster is this block than the still parts
    of the scene" -- is dimensionless and comparable everywhere. Flying reads as a
    few-x elevation (~3-4x) whether the animal is bright or dim; a still replicate
    never rises above ~1x its own background.

    It is a ratio, not a z-score, on purpose. A per-block temporal z-score would
    inflate a genuinely-still replicate's own noise into false detections (the
    denominator there is the block's own tiny variance); the spatial-background
    ratio keeps a still scene near 1 because its numerator never leaves background.

    Background is spatial and per-frame, so this is well-defined on the partial
    time slices the UI shows -- no dependence on window length or clip position.
    """
    rm = ctx.get("rolling_mean_speed")            # (T, ny, nx)
    T = rm.shape[0]
    out = np.zeros_like(rm, dtype=np.float32)
    for y0, x0, y1, x1 in ctx.regions:
        sub = rm[:, y0:y1, x0:x1]
        bg = np.percentile(sub.reshape(T, -1), _REL_BG_PCT,
                           axis=1).astype(np.float32)
        denom = bg + np.float32(_REL_BG_FLOOR)
        out[:, y0:y1, x0:x1] = sub / denom[:, None, None]
    return out


def _welch_psd(x: np.ndarray, fps: float, nperseg: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """PSD along axis 0 of a (T, ...) array, Hann-windowed, no segment averaging.

    Kept deliberately simple (single segment, detrended) so it can run on the
    whole block grid at once without a Python loop over blocks.
    """
    T = x.shape[0]
    n = min(nperseg, T)
    if n < 4:
        return np.zeros(0), np.zeros((0,) + x.shape[1:])
    seg = x[:n] - x[:n].mean(axis=0, keepdims=True)
    w = np.hanning(n).astype(np.float32)
    seg = seg * w.reshape((-1,) + (1,) * (x.ndim - 1))
    spec = np.fft.rfft(seg, axis=0)
    psd = (np.abs(spec) ** 2) / (fps * np.sum(w ** 2))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    return freqs, psd.astype(np.float32)


def _dominant_freq(ctx: FeatureContext) -> np.ndarray:
    """Frequency of peak power in each block's speed time series, in Hz.

    A single value per block for the whole slice -- broadcast back over time so
    it has the same shape as the other spatial features.
    """
    freqs, psd = _welch_psd(ctx.speed, ctx.fps, nperseg=ctx.speed.shape[0])
    if freqs.size == 0:
        return np.zeros_like(ctx.speed)
    # Ignore DC: every block has a nonzero mean speed and it always wins.
    peak = np.argmax(psd[1:], axis=0) + 1
    dom = freqs[peak].astype(np.float32)
    return np.broadcast_to(dom, ctx.speed.shape).copy()


def _spectral_flatness(ctx: FeatureContext) -> np.ndarray:
    """Geometric mean / arithmetic mean of the PSD, in [0, 1].

    Near 0 = a sharp spectral peak (a periodic behavior like a wingbeat).
    Near 1 = a flat, noise-like spectrum. This is the feature that separates
    "periodic" from merely "fast".
    """
    _, psd = _welch_psd(ctx.speed, ctx.fps, nperseg=ctx.speed.shape[0])
    if psd.shape[0] == 0:
        return np.zeros_like(ctx.speed)
    p = psd[1:] + _EPS
    gm = np.exp(np.mean(np.log(p), axis=0))
    am = np.mean(p, axis=0)
    flat = (gm / (am + _EPS)).clip(0.0, 1.0).astype(np.float32)
    return np.broadcast_to(flat, ctx.speed.shape).copy()


def _direction_oscillation(ctx: FeatureContext) -> np.ndarray:
    """Lag-1 autocorrelation of the flow direction, in [-1, 1].

    Strongly negative = the direction reverses every frame, which is the
    signature of a reciprocating motion (a wing, a fanning leg) sampled near its
    own frequency. Computed on the unit vector to stay circular-safe.
    """
    n = np.hypot(ctx.u, ctx.v) + _EPS
    ux, uy = ctx.u / n, ctx.v / n
    dot = ux[1:] * ux[:-1] + uy[1:] * uy[:-1]
    out = np.zeros_like(ctx.speed)
    out[1:] = dot
    out[0] = dot[0] if dot.shape[0] else 0.0
    return _rolling(out, int(round(0.5 * ctx.fps)), "mean")


# -- the registry ------------------------------------------------------------

def _spec(**kw) -> FeatureSpec:
    return FeatureSpec(**kw)


REGISTRY: dict[str, FeatureSpec] = {}


def register(spec: FeatureSpec) -> None:
    REGISTRY[spec.name] = spec


for _s in [
    _spec(name="u", label="Flow u (net, horizontal)", units="px/s", kind="cached",
          domain="spatial", clip_pct=(1.0, 99.0),
          help="Horizontal component of the block's mean flow vector. Signed."),
    _spec(name="v", label="Flow v (net, vertical)", units="px/s", kind="cached",
          domain="spatial", clip_pct=(1.0, 99.0),
          help="Vertical component of the block's mean flow vector. Signed."),
    _spec(name="speed", label="Speed (mean |flow|)", units="px/s", kind="cached",
          domain="spatial",
          help="Mean of per-pixel flow magnitude within the block: total motion "
               "energy, regardless of direction. High for anything moving. "
               "Compare against net_speed to tell translation from oscillation."),
    _spec(name="fb_error_p90", label="Forward/backward error (block p90)",
          units="working px/frame", kind="cached", domain="spatial",
          help="90th percentile within each block of the exact pixelwise "
               "||F(x)+B(x+F(x))|| residual. Lower is more trustworthy. This is "
               "continuous and unthresholded; choose the acceptable error in "
               "the behavior tree (<=1 working pixel/frame is a starting point). "
               "Requires roughly 2x flow compute."),
    _spec(name="texture_min_eigen", label="Texture strength (min eigenvalue)",
          units="OpenCV response", kind="cached", domain="spatial",
          help="Block mean of cv2.cornerMinEigenVal on the preprocessed frame. "
               "Low values have weak two-dimensional texture and unreliable "
               "flow. Prefer texture_percentile for cross-frame thresholds."),

    _spec(name="net_speed", label="Net flow magnitude", units="px/s",
          kind="derived", domain="spatial", compute=_net_speed, deps=("u", "v"),
          help="Magnitude of the mean flow vector: how far the block as a whole "
               "translates. A wingbeat has high speed but LOW net_speed, because "
               "the up-stroke and down-stroke cancel."),
    _spec(name="angle", label="Flow direction", units="deg", kind="derived",
          domain="spatial", compute=_angle, deps=("u", "v"), clip_pct=(0.0, 100.0),
          help="Direction of net flow, 0-360 deg. Meaningless where net_speed is "
               "near zero -- filter on net_speed first."),
    _spec(name="coherence", label="Angular coherence", units="0-1", kind="derived",
          domain="spatial", compute=_coherence, deps=("u", "v", "speed"),
          clip_pct=(0.0, 100.0),
          help="|mean vector| / mean |vector|. 1 = the whole block moves as one "
               "(translation). 0 = motion that cancels within the block "
               "(oscillation). Low coherence + high speed is the wingbeat corner."),
    _spec(name="median3_u", label="Median 3x3 flow u", units="px/s",
          kind="derived", domain="spatial", compute=_median3_u, deps=("u",),
          help="A visible analysis-time 3x3 median on horizontal block flow. "
               "Raw u remains available for comparison."),
    _spec(name="median3_v", label="Median 3x3 flow v", units="px/s",
          kind="derived", domain="spatial", compute=_median3_v, deps=("v",),
          help="A visible analysis-time 3x3 median on vertical block flow. "
               "Raw v remains available for comparison."),
    _spec(name="median3_speed", label="Median 3x3 mean speed", units="px/s",
          kind="derived", domain="spatial", compute=_median3_speed,
          deps=("speed",),
          help="Spatial median of the cached scalar mean-speed field. This is "
               "separate from median3_net_speed because mean pixel speed and "
               "magnitude of the mean vector are not interchangeable."),
    _spec(name="median3_net_speed", label="Median 3x3 net flow magnitude",
          units="px/s", kind="derived", domain="spatial",
          compute=_median3_net_speed, deps=("u", "v"),
          help="Magnitude after separately median-filtering u and v on the "
               "block grid. Compare with raw net_speed to see what was removed."),
    _spec(name="texture_percentile", label="Texture percentile within frame",
          units="0-1", kind="derived", domain="spatial",
          compute=_texture_percentile, deps=("texture_min_eigen",),
          clip_pct=(0.0, 100.0),
          help="Per-frame empirical percentile of cached texture strength. "
               "Keeping >=0.25 implements the suggested q25 low-texture mask "
               "without baking alpha into the cache."),
    _spec(name="divergence", label="Divergence", units="1/s", kind="derived",
          domain="spatial", compute=_divergence, deps=("u", "v"),
          help="du/dx + dv/dy. Positive = expansion, negative = contraction."),
    _spec(name="curl", label="Curl", units="1/s", kind="derived",
          domain="spatial", compute=_curl, deps=("u", "v"),
          help="dv/dx - du/dy. Nonzero where the flow field rotates."),
    _spec(name="rolling_mean_speed", label="Rolling mean speed", units="px/s",
          kind="derived", domain="spatial", compute=_rolling_mean_speed,
          deps=("speed",),
          help="Speed smoothed over a 0.5 s centered window. Suppresses "
               "single-frame flow noise."),
    _spec(name="rolling_std_speed", label="Rolling std of speed", units="px/s",
          kind="derived", domain="spatial", compute=_rolling_std_speed,
          deps=("speed",),
          help="Temporal variability of speed over 0.5 s. High for anything that "
               "starts, stops, or oscillates; near zero for steady motion."),
    _spec(name="rel_speed", label="Relative speed (x background)", units="x",
          kind="derived", domain="spatial", compute=_rel_speed,
          deps=("speed",),
          help="Rolling mean speed divided by the frame's own background motion "
               "(spatial p40). Dimensionless, so ONE threshold works across "
               "replicates of different brightness -- unlike raw speed, whose "
               "scale changes with contrast. Flying is a few-x elevation (~3-4x) "
               "whether the animal is bright or backlit; a still region stays near "
               "1x. This is the feature to threshold when a speed cutoff won't "
                "generalize across your boxes."),

    # ROI-derived features are evaluated in core.roi because their denominator
    # or physical calibration belongs to a particular replicate, not the whole
    # frame. They are still registry entries so the behavior editor can expose
    # them when the selected boxes carry the required metadata.
    _spec(name="speed_over_baseline_p99",
          label="Speed / quiescent baseline p99", units="x baseline p99",
          kind="roi", domain="spatial", deps=("speed",),
          help="Per-block speed divided by this replicate's 99th-percentile "
               "speed during its explicitly selected quiescent baseline. A "
               "threshold of 3x or 5x is directly interpretable."),
    _spec(name="speed_over_auto_noise",
          label="Speed / automatic replicate noise", units="x auto noise",
          kind="roi", domain="spatial", deps=("speed",),
          help="Fallback when no clean baseline exists. For each frame, measure "
               "the spatial p25 across this replicate; its temporal p99 becomes "
               "ONE fixed reference for the whole replicate. This adapts across "
               "boxes without renormalizing behavior separately every frame."),
    _spec(name="speed_mm_s", label="Speed", units="mm/s", kind="roi",
          domain="spatial", deps=("speed",),
          help="Cached speed converted at analysis time using this replicate's "
               "source-pixels/mm calibration. Raw pixel-space speed is unchanged."),
    _spec(name="net_speed_mm_s", label="Net flow magnitude", units="mm/s",
          kind="roi", domain="spatial", deps=("net_speed",),
          help="Net flow converted at analysis time using this replicate's "
               "source-pixels/mm calibration."),
    _spec(name="speed_body_lengths_s", label="Speed", units="body lengths/s",
          kind="roi", domain="spatial", deps=("speed",),
          help="Mean speed normalized by source-pixels/mm and this replicate's "
               "body length. Both metadata values remain editable."),

    _spec(name="dominant_freq", label="Dominant frequency", units="Hz",
          kind="derived", domain="temporal", compute=_dominant_freq, deps=("speed",),
          clip_pct=(0.0, 100.0),
          help="Frequency of the largest peak in the block's speed spectrum, DC "
               "excluded. Trustworthy only well below Nyquist (fps/2)."),
    _spec(name="spectral_flatness", label="Spectral flatness", units="0-1",
          kind="derived", domain="temporal", compute=_spectral_flatness,
          deps=("speed",), clip_pct=(0.0, 100.0),
          help="Geometric/arithmetic mean of the spectrum. Near 0 = one sharp "
               "periodic peak. Near 1 = broadband noise. This separates a "
               "periodic behavior from something merely fast."),
    _spec(name="direction_oscillation", label="Direction oscillation index",
          units="-1 to 1", kind="derived", domain="temporal",
          compute=_direction_oscillation, deps=("u", "v"), clip_pct=(0.0, 100.0),
          help="Lag-1 autocorrelation of flow direction. Strongly negative = the "
               "direction reverses frame to frame, the signature of a "
               "reciprocating motion."),
]:
    register(_s)


def register_band(band_name: str, lo_hz: float, hi_hz: float) -> None:
    """Register a cached band-power feature. Called when a cache is opened, so
    the registry reflects whatever bands that cache actually holds."""
    register(_spec(
        name=band_name,
        label=f"Band power {lo_hz:g}-{hi_hz:g} Hz",
        units="(px/s)^2/Hz",
        kind="cached", domain="temporal",
        help=(f"Power in the speed spectrum between {lo_hz:g} and {hi_hz:g} Hz, "
              f"computed per block on a sliding window. This is the primary "
              f"feature for periodic behaviors: a grasshopper wingbeat at ~20 Hz "
              f"shows up as a bright, spatially small, low-coherence blob."),
    ))


def cached_feature_names(cfg) -> list[str]:
    """The arrays a pass with this config will write to disk."""
    names = ["u", "v", "speed"]
    names += [b.label() for b in cfg.features.bands]
    if cfg.features.cache_coherence:
        names.append("coherence")
    if cfg.features.cache_divergence_curl:
        names += ["divergence", "curl"]
    if cfg.features.cache_spectral_flatness:
        names.append("spectral_flatness")
    if cfg.features.cache_direction_oscillation:
        names.append("direction_oscillation")
    if cfg.features.cache_fb_error:
        names.append("fb_error_p90")
    if cfg.features.cache_texture:
        names.append("texture_min_eigen")
    return names


