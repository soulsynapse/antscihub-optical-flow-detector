"""Channels that are a function of an already-computed per-block series.

The extraction principle this file exists to honour: *cache what needs the
decode; derive what is a function of cached series.* Everything here reads a
``(T, ny, nx)`` channel that ``tensor_channels`` already produced and returns
another one, so nothing in this module costs a frame.

``occupancy`` is the load-bearing one. Every channel the tensor produces is a
motion or gradient energy, so a motionless animal and empty substrate are the
same point in feature space -- and "still" is a behaviour, not an absence.
Occupancy is the only read that says an animal is *present* without requiring it
to move, and it is what converts the extensive channels (which scale with how
much animal is in the block) into intensive ones (per unit animal). Without that
normalisation the dominant axis of any embedding built on these channels is
"how much animal is in view", which is a nuisance variable.
"""
from __future__ import annotations

import warnings

import numpy as np

# Polarity of the animal against its background, which sets both which tail of
# the intensity distribution is background and which sign of deviation counts.
#
#   "darker"   animal darker than substrate (front-lit on a pale arena). The
#              background is the BRIGHT tail, so it is estimated high.
#   "lighter"  animal brighter than substrate (backlit -- and
#              `rep2-backlit-flying-whole-time` is exactly this). Background is
#              the DARK tail.
#   "abs"      unknown or mixed. Median background, unsigned deviation. Safe
#              default; costs sensitivity because substrate flicker in either
#              direction now reads as occupancy.
POLARITIES = ("abs", "darker", "lighter")

# Where to put the background percentile for each polarity. Offset from the
# median rather than at an extreme: the estimate has to survive the animal
# sitting in the block for a large minority of the window (see the breakdown
# note on `background_level`) without chasing the brightest single frame, which
# is noise, not background.
_BG_PERCENTILE = {"abs": 50.0, "darker": 75.0, "lighter": 25.0}


def background_level(intensity: np.ndarray, polarity: str = "abs",
                     percentile: float | None = None) -> np.ndarray:
    """Per-block background intensity, ``(T, ny, nx) -> (ny, nx)`` float32.

    A percentile over time per block, NOT a mean: the mean is dragged by the
    animal every frame it is present and smears it into the background, which is
    the failure that makes frame-difference-against-a-mean produce trails. A
    percentile ignores the animal entirely as long as it is in the minority.

    **The breakdown point is the whole story here.** With ``polarity="abs"`` the
    median tolerates the animal occupying a block for up to 50% of the window;
    the signed polarities buy margin by estimating from the correct tail
    (75th/25th), tolerating ~75%. Past the limit the "background" becomes the
    animal, and the two polarities then fail differently -- a distinction that
    decides how visible the failure is:

      * signed -- the animal now sits AT the background, so its deviation is
        zero, and the substrate deviates on the side rectification discards. The
        block reports **no animal in any frame**: a total, silent loss.
      * ``"abs"`` -- the substrate survives as an unsigned deviation, so the
        channel **inverts** (quiet where the animal is, loud where it is not)
        rather than vanishing. Wrong, but visibly wrong.

    Neither is recoverable by choosing a different percentile. The window must be
    long relative to how long the animal holds still in one place -- an
    organism-relative constant, not a frame-relative one -- and the response to a
    short window is a longer one or a borrowed ``background``, not a re-tune.

    """
    if polarity not in POLARITIES:
        raise ValueError(f"polarity must be one of {POLARITIES}, got {polarity!r}")
    arr = np.asarray(intensity, np.float32)
    if arr.ndim != 3:
        raise ValueError(f"intensity must be (T, ny, nx), got shape {arr.shape}")
    if arr.shape[0] == 0:
        return np.zeros(arr.shape[1:], np.float32)
    q = _BG_PERCENTILE[polarity] if percentile is None else float(percentile)
    # nanpercentile, not percentile: a masked or out-of-tile block reduces to NaN
    # (core.flow.reduce_scalar_to_blocks preserves masked-input semantics), and
    # propagating that NaN into the background would silently void the block for
    # every frame rather than for the frames that were actually masked.
    #
    # A block masked for the WHOLE window is a different case and does come back
    # NaN, which then makes its occupancy NaN for every frame -- correct, since
    # there is no evidence to form a background from. Suppressed with
    # catch_warnings rather than errstate: numpy raises the all-NaN case through
    # the warnings module, so errstate (which governs only the floating-point
    # flags) would look like a guard here while doing nothing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bg = np.nanpercentile(arr, q, axis=0)
    return np.asarray(bg, np.float32)


def occupancy(intensity: np.ndarray, polarity: str = "abs",
              percentile: float | None = None,
              background: np.ndarray | None = None) -> np.ndarray:
    """How much animal is in each block, ``(T, ny, nx) -> (T, ny, nx)`` float32.

    Deviation of block intensity from its own background level, rectified to the
    side the animal is on. Extensive: it scales with the fraction of the block
    the animal covers, which is what makes it the right denominator for turning
    an energy channel into a per-animal one (see :func:`intensive`).

    ``background`` lets a level computed over a long window be applied to a short
    one. That is the only correct way to read occupancy on a window shorter than
    the animal's stillness bouts -- recomputing the percentile inside the short
    window re-introduces the breakdown documented on :func:`background_level`.
    It is mutually exclusive with ``percentile``, which would otherwise be
    accepted and silently ignored: borrowing a background is exactly when a
    caller is most likely to also be tuning the percentile, and having that
    tuning quietly do nothing is worse than an error.
    """
    arr = np.asarray(intensity, np.float32)
    if arr.ndim != 3:
        raise ValueError(f"intensity must be (T, ny, nx), got shape {arr.shape}")
    if background is not None and percentile is not None:
        raise ValueError("pass either `background` or `percentile`, not both; "
                         "a supplied background makes `percentile` a no-op")
    bg = (background_level(arr, polarity, percentile) if background is None
          else np.asarray(background, np.float32))
    if bg.shape != arr.shape[1:]:
        raise ValueError(f"background {bg.shape} does not match blocks "
                         f"{arr.shape[1:]}")
    d = arr - bg[None]
    if polarity == "abs":
        return np.abs(d)
    # Rectified, not absolute: with a known polarity, a deviation on the wrong
    # side is substrate change (a shadow, a lighting drift), not the animal, and
    # counting it would put occupancy where there is none.
    return np.maximum(d if polarity == "lighter" else -d, 0.0).astype(np.float32)


def _masked_divide(a: np.ndarray, b: np.ndarray, floor: float) -> np.ndarray:
    """``a / b`` where ``b > floor``, NaN elsewhere, dividing only under the mask.

    ``np.divide(..., where=, out=)`` rather than ``np.where(mask, a / b, nan)``:
    the latter evaluates the division over every element including the ones it is
    about to discard, and this runs on the live recompute path over a ring that
    can be a gigabyte.
    """
    mask = b > floor
    out = np.full(a.shape, np.nan, np.float32)
    np.divide(a, b, out=out, where=mask)
    return out


def intensive(extensive: np.ndarray, occ: np.ndarray, *,
              floor: float = 0.0) -> np.ndarray:
    """An extensive channel per unit animal, ``(T, ny, nx)`` -> same, float32.

    ``change``, ``appearance`` and the strain reads all scale with how much
    animal is in the block, so their raw values conflate "moving vigorously" with
    "more of the animal is here". Dividing by occupancy separates those.

    Blocks at or below ``floor`` occupancy return **NaN, not zero**. Zero would
    read downstream as a confident "no activity per unit animal" for a block that
    holds no animal to normalise by, which is the silent-false-negative failure
    this codebase designs against; NaN forces the consumer to decide.

    **``floor`` is an absolute occupancy, and it deliberately does not default to
    a quantile of the data.** An earlier version used a fraction of the median
    occupancy in the array passed in, which makes the channel a function of the
    window it was computed over: under live streaming the ring grows and then
    slides, so the floor moves every recompute and a block sitting near it
    flickers between a value and NaN with no input having changed -- and a
    threshold tuned on a 10 s window means something else on the whole clip. That
    is ``FINDINGS.md`` section 5's ``rescale_count_band`` trap in a new place. If
    you want a noise-floor rather than "any occupancy at all", derive it once from
    a stable reference (a quiet region's occupancy on a long window) and pass the
    same number everywhere.
    """
    ext = np.asarray(extensive, np.float32)
    o = np.asarray(occ, np.float32)
    if ext.shape != o.shape:
        raise ValueError(f"shape mismatch: extensive {ext.shape} vs occupancy "
                         f"{o.shape}")
    return _masked_divide(ext, o, float(floor))


def ratio(numer: np.ndarray, denom: np.ndarray, *, floor: float = 0.0,
          clip: tuple | None = None) -> np.ndarray:
    """A conjunction channel, ``numer / denom``, elementwise.

    These exist because a *linear* method in the concatenated channel space
    cannot construct a conjunction. "High change AND low speed" (a wingbeat in
    place) and "high change AND high speed" (running) are the same point in
    ``change``, and PCA cannot build the product that separates them -- so the
    product has to be handed to it as its own channel.

    ``clip`` defaults to **None**, i.e. unbounded. Only some of these ratios are
    bounded: ``appearance / change`` is in [0, 1] by construction, since
    ``flow_residual`` rectifies at zero and cannot exceed ``tt`` -- pass
    ``clip=(0.0, 1.0)`` there to absorb float error at tiny magnitudes, and take
    no log and no distribution band on it per the detection-channel design.
    ``tensor_speed / sqrt(change)`` and ``strain / speed`` are **not** bounded by
    1, and clipping them by default would silently collapse every block above
    unity onto the same value -- destroying exactly the dynamic range the channel
    was added for. Boundedness is a claim the caller makes, not a default.

    ``floor`` carries the same warning as :func:`intensive`: absolute, never a
    quantile of the array passed in.
    """
    a = np.asarray(numer, np.float32)
    b = np.asarray(denom, np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    out = _masked_divide(a, b, float(floor))
    if clip is not None:
        out = np.clip(out, clip[0], clip[1])
    return out.astype(np.float32)
