"""A pluggable channel-testing lab, decoupled from core.detection.

Two layers so candidate channels are cheap to add and cheap to run:

  * BASE FIELDS -- the expensive, shared per-block time series a decode+flow pass
    produces (intensity, change, appearance, tensor_speed). Extracted once and
    cached to .npy by extract_base_fields.py; a channel never re-decodes.
  * CHANNEL -- a registered pure function (fields, meta) -> (T, B). The temporal
    filter is PART of the channel, not assumed to be Morlet, so a butter+filtfilt
    band-energy channel or a Welch feature is a first-class channel, not a special
    case. A channel declares which base fields it needs; the store loads only those.

The harness applies the SAME reductions (a mean control + tail statistics) and the
SAME span-level AUC to every registered channel, so a candidate is scored on equal
footing. Adding a channel is one function + @channel(...).

AUC = P[a random flying sample outranks a random not-flying sample]; 0.5 chance.
Span-level (mean score per flying bout vs per still bout) is the honest number
because frames within a bout are autocorrelated.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt
from scipy.stats import rankdata

from core.detection import largest_clump_per_frame, windowed_mean
from core.wavelet import band_indices, default_freqs, morlet_band_power

# ------------------------------------------------------------------ base fields
BASE_FIELDS = ("intensity", "change", "appearance", "tensor_speed")


class FieldStore:
    """Lazy per-block base fields (T, B) + geometry, from a t31_extract cache dir.

    Fields are memmapped and materialized on first use, then held; a whole-clip
    field is ~1.6 GB, so a caller sweeping many channels over a few base fields
    pays the read once. ``block_grid`` is the (ny, nx, gy, gx) a clump statistic
    needs, derived from the cached meta."""

    def __init__(self, cache_dir: str):
        self.dir = cache_dir
        with open(os.path.join(cache_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.fps = float(self.meta["fps"])
        self.ny, self.nx = (int(v) for v in self.meta["grid"])
        self._cache: dict[str, np.ndarray] = {}

    def field(self, name: str) -> np.ndarray:
        """(T, B) per-block time series for one base field."""
        if name not in self._cache:
            arr = np.load(os.path.join(self.dir, f"chan_{name}.npy"),
                          mmap_mode="r")
            self._cache[name] = np.asarray(arr).reshape(arr.shape[0], -1)
        return self._cache[name]

    @property
    def block_grid(self):
        gy, gx = np.mgrid[0:self.ny, 0:self.nx]
        return self.ny, self.nx, gy.ravel(), gx.ravel()


# --------------------------------------------------------------------- registry
@dataclass(frozen=True)
class Channel:
    name: str
    needs: tuple[str, ...]
    fn: Callable[[dict, dict], np.ndarray]   # (fields, meta) -> (T, B)

    def compute(self, store: FieldStore) -> np.ndarray:
        fields = {k: store.field(k) for k in self.needs}
        out = self.fn(fields, store.meta)
        return np.asarray(out, np.float32)


REGISTRY: dict[str, Channel] = {}


def channel(name: str, needs: tuple[str, ...]):
    """Decorator: register a candidate channel. ``needs`` names its base fields."""
    def deco(fn):
        REGISTRY[name] = Channel(name=name, needs=tuple(needs), fn=fn)
        return fn
    return deco


def reset_registry():
    REGISTRY.clear()


# ----------------------------------------------------------- temporal filters
# Reusable building blocks a channel body calls. Each maps a per-block (T, B)
# field to another (T, B) field; the choice of filter is what a channel IS.

def morlet_band(field: np.ndarray, fps: float, band: tuple[float, float]
                ) -> np.ndarray:
    """Morlet scalogram power summed over [lo, hi] Hz, per block per frame."""
    freqs = default_freqs(fps)
    i, j = band_indices(freqs, band[0], band[1])
    return morlet_band_power(np.asarray(field, np.float32), fps, freqs, i, j)


def butter_band_energy(field: np.ndarray, fps: float, band: tuple[float, float],
                       order: int = 4, smooth_frames: int = 4,
                       col_chunk: int = 2048) -> np.ndarray:
    """Zero-phase Butterworth band-pass energy, per block per frame.

    The butter analogue of ``morlet_band``: filtfilt the block's time series in
    [lo, hi] Hz (zero phase, so no group-delay smear across a flight onset),
    square, then smooth over ``smooth_frames`` to get instantaneous band power on
    the same footing as the wavelet's. Chunked over block columns because
    ``filtfilt`` upcasts to float64 internally (a whole (T, B) at once is ~3 GB)."""
    nyq = 0.5 * fps
    lo, hi = band[0] / nyq, min(band[1] / nyq, 0.999)
    b, a = butter(order, [lo, hi], btype="band")
    x = np.asarray(field, np.float32)
    T, B = x.shape
    out = np.empty((T, B), np.float32)
    for c0 in range(0, B, col_chunk):
        c1 = min(B, c0 + col_chunk)
        seg = filtfilt(b, a, x[:, c0:c1], axis=0)
        e = (seg * seg).astype(np.float32)
        out[:, c0:c1] = uniform_filter1d(e, smooth_frames, axis=0, mode="nearest")
    return out


# ------------------------------------------------------------------ reductions
# (T, B) channel field -> (T,) per-frame series. A mean CONTROL plus tails; the
# harness scores all of them so the mean-vs-tail contrast is visible per channel.
_GLOBAL_PCTILE = 90.0
_DETECT_WIN = 4


def _reductions(field: np.ndarray, store: FieldStore) -> dict[str, np.ndarray]:
    thr = float(np.nanpercentile(field, _GLOBAL_PCTILE))
    finite = np.isfinite(field)
    count = ((field >= thr) & finite).sum(axis=1).astype(np.float32)
    ny, nx, gy, gx = store.block_grid
    clump = largest_clump_per_frame(field, thr, np.inf, ny, nx, gy, gx)
    return {
        "mean (control)": np.nanmean(field, axis=1),
        "p99 (tail)": np.nanpercentile(field, 99, axis=1),
        "count>P90 win (tail)": windowed_mean(count, _DETECT_WIN, True),
        "clump win (tail)": windowed_mean(clump, _DETECT_WIN, True),
    }


# ---------------------------------------------------------------- labels + AUC
def auc(scores: np.ndarray, pos: np.ndarray):
    scores = np.asarray(scores, float)
    pos = np.asarray(pos, bool)
    ok = np.isfinite(scores)
    scores, pos = scores[ok], pos[ok]
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def flying_intervals(marks_path: str, fps: float, T: int):
    with open(marks_path) as f:
        spans = json.load(f)["spans"]["0"]
    iv = []
    for a, b, _ in spans:
        f0, f1 = max(0, int(round(a * fps))), min(T, int(round(b * fps)))
        if f1 > f0:
            iv.append((f0, f1))
    return sorted(iv)


def _labels(iv, T):
    lab = np.zeros(T, bool)
    for f0, f1 in iv:
        lab[f0:f1] = True
    return lab


def _still_bouts(iv, T, min_len):
    gaps, prev = [], 0
    for f0, f1 in iv:
        if f0 - prev >= min_len:
            gaps.append((prev, f0))
        prev = max(prev, f1)
    if T - prev >= min_len:
        gaps.append((prev, T))
    return gaps


def _bout_scores(series, bouts, erode):
    out = []
    for f0, f1 in bouts:
        s, e = f0 + erode, f1 - erode
        if e <= s:
            s, e = f0, f1
        seg = series[s:e]
        seg = seg[np.isfinite(seg)]
        out.append(seg.mean() if seg.size else np.nan)
    return np.array(out, float)


def validate(store: FieldStore, marks_path: str, *, guard_s=0.15,
             min_gap_s=0.30, channels: list[str] | None = None):
    """Score every registered channel (or the named subset) against the corpus.

    Returns a list of (channel, statistic, auc_frame, auc_span) and, for the
    single best statistic per channel, its per-frame series (for plotting)."""
    fps = store.fps
    names = channels if channels is not None else list(REGISTRY)
    T = store.field(REGISTRY[names[0]].needs[0]).shape[0]
    iv = flying_intervals(marks_path, fps, T)
    lab = _labels(iv, T)
    guard = int(round(guard_s * fps))
    keep = np.ones(T, bool)
    for f0, f1 in iv:
        keep[max(0, f0 - guard):min(T, f0 + guard)] = False
        keep[max(0, f1 - guard):min(T, f1 + guard)] = False
    bouts = _still_bouts(iv, T, int(round(min_gap_s * fps)))
    erode = guard

    rows, best_series = [], {}
    for name in names:
        ch = REGISTRY[name]
        field = ch.compute(store)
        reds = _reductions(field, store)
        best = None
        for st, series in reds.items():
            af = auc(series[keep], lab[keep])
            fly = _bout_scores(series, iv, erode)
            still = _bout_scores(series, bouts, erode)
            asp = auc(np.concatenate([fly, still]),
                      np.concatenate([np.ones(len(fly), bool),
                                      np.zeros(len(still), bool)]))
            rows.append((name, st, af, asp))
            if best is None or abs(asp - 0.5) > abs(best[1] - 0.5):
                best = (st, asp, series)
        best_series[name] = best
        del field
    return {"rows": rows, "best_series": best_series, "lab": lab, "iv": iv,
            "fps": fps, "T": T, "n_still_bouts": len(bouts)}
