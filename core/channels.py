"""A pluggable derived-channel registry over the tensor path's base fields.

The extraction in :mod:`core.tensor_channels` produces the expensive per-block
base fields a decode+tensor pass measures (``intensity``, ``change``,
``appearance``, ``tensor_speed``, and the signed flow ``u``/``v``). A DERIVED
channel is a pure function of those already-computed fields -- it never
re-decodes. This module is the registry of them.

  * CHANNEL -- a registered function ``(fields, meta) -> (T, ny, nx)`` plus the
    set of base fields it declares it ``needs``. The temporal/spatial filter is
    PART of the channel, not assumed to be any one transform, so a butter/Welch
    band-energy channel and a spatial velocity-gradient channel are first-class
    on the same footing. Register with ``@channel(name, needs=(...))``.
  * The fields a channel receives are ``(T, ny, nx)`` -- the same shape a
    ``core.channel_source.ChannelData`` carries -- so a derived channel that
    needs spatial structure (the velocity gradient takes spatial derivatives)
    has it. ``evaluate`` loads only the declared ``needs``.

Promoted from ``scripts/channel_lab.py`` (which now imports from here); the lab
adds the offline corpus/AUC scoring on top and adapts its flattened ``(T, B)``
FieldStore to this ``(T, ny, nx)`` contract.

The velocity-gradient channels decompose the block velocity gradient ``∇v`` into
its three 2-D invariant parts -- divergence (trace), shear (deviatoric strain,
the symmetric traceless part), and vorticity (the antisymmetric part). ``∇v`` is
translation-invariant by construction, so it measures posture/configural change
with no tracker and no body frame. Where the retired flow cache computed only
divergence and curl per region, this computes the full gradient tensor, per
atlas region so a spatial derivative never crosses a replicate separator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt

from core.wavelet import band_indices, default_freqs, morlet_band_power


# --------------------------------------------------------------------- registry
@dataclass(frozen=True)
class Channel:
    name: str
    needs: tuple[str, ...]
    fn: Callable[[dict, dict], np.ndarray]   # (fields, meta) -> (T, ny, nx)

    def compute(self, fields: dict, meta: dict) -> np.ndarray:
        got = {k: fields[k] for k in self.needs}
        return np.asarray(self.fn(got, meta), np.float32)


REGISTRY: dict[str, Channel] = {}


def channel(name: str, needs: Iterable[str]):
    """Decorator: register a derived channel. ``needs`` names its base fields."""
    def deco(fn):
        REGISTRY[name] = Channel(name=name, needs=tuple(needs), fn=fn)
        return fn
    return deco


def reset_registry() -> None:
    REGISTRY.clear()


def evaluate(fields: dict, meta: dict, names: Iterable[str]) -> dict:
    """Compute the named derived channels from ``fields`` (base field -> (T,ny,nx)).

    Raises ``KeyError`` for an unknown channel or a channel whose declared base
    fields are absent, so a caller can never silently derive from missing data.
    """
    out: dict[str, np.ndarray] = {}
    for name in names:
        ch = REGISTRY.get(name)
        if ch is None:
            raise KeyError(f"unknown derived channel {name!r}; "
                           f"registered: {sorted(REGISTRY)}")
        missing = [k for k in ch.needs
                   if k not in fields or fields[k] is None]
        if missing:
            raise KeyError(
                f"derived channel {name!r} needs base fields {missing}, "
                f"which the source did not provide")
        out[name] = ch.compute(fields, meta)
    return out


def needs_for(names: Iterable[str]) -> set[str]:
    """The union of base fields the named derived channels declare they need.

    Lets a caller (the live surface) translate a request for a derived channel
    into the set of primitives its extraction pass must actually compute."""
    out: set[str] = set()
    for name in names:
        ch = REGISTRY.get(name)
        if ch is not None:
            out |= set(ch.needs)
    return out


# ----------------------------------------------------------- temporal filters
# Reusable building blocks a channel body calls. Each maps a per-block (T, ny,
# nx) field to another (T, ny, nx) field; the choice of filter is what a channel
# IS. Both accept any (T, ...) shape -- they flatten the trailing block axes,
# filter along time, and restore the shape -- so the same helper serves this
# module's (T, ny, nx) fields and the lab's flattened (T, B) ones.

def morlet_band(field: np.ndarray, fps: float,
                band: tuple[float, float]) -> np.ndarray:
    """Morlet scalogram power summed over [lo, hi] Hz, per block per frame."""
    x = np.asarray(field, np.float32)
    T = x.shape[0]
    rest = x.shape[1:]
    freqs = default_freqs(fps)
    i, j = band_indices(freqs, band[0], band[1])
    out = morlet_band_power(x.reshape(T, -1), fps, freqs, i, j)
    return out.reshape((T,) + rest)


def butter_band_energy(field: np.ndarray, fps: float, band: tuple[float, float],
                       order: int = 4, smooth_frames: int = 4,
                       col_chunk: int = 2048) -> np.ndarray:
    """Zero-phase Butterworth band-pass energy, per block per frame.

    The butter analogue of ``morlet_band``: filtfilt the block's time series in
    [lo, hi] Hz (zero phase, so no group-delay smear across a flight onset),
    square, then smooth over ``smooth_frames`` to get instantaneous band power on
    the same footing as the wavelet's. Chunked over block columns because
    ``filtfilt`` upcasts to float64 internally (a whole field at once is ~3 GB)."""
    x = np.asarray(field, np.float32)
    T = x.shape[0]
    rest = x.shape[1:]
    x = x.reshape(T, -1)
    nyq = 0.5 * fps
    lo, hi = band[0] / nyq, min(band[1] / nyq, 0.999)
    b, a = butter(order, [lo, hi], btype="band")
    B = x.shape[1]
    out = np.empty((T, B), np.float32)
    for c0 in range(0, B, col_chunk):
        c1 = min(B, c0 + col_chunk)
        seg = filtfilt(b, a, x[:, c0:c1], axis=0)
        e = (seg * seg).astype(np.float32)
        out[:, c0:c1] = uniform_filter1d(e, smooth_frames, axis=0, mode="nearest")
    return out.reshape((T,) + rest)


# ------------------------------------------------------- velocity gradient ∇v
def _regions(meta: dict) -> list[tuple[int, int, int, int]]:
    """Atlas (y0, x0, y1, x1) boxes a spatial derivative must stay inside.

    One box per replicate tile so ``np.gradient`` never crosses the padding
    between two replicates packed into the same atlas -- a difference taken
    across that seam is meaningless. Falls back to the whole grid for a
    non-replicate field."""
    tiles = meta.get("replicate_tiles")
    if not tiles:
        ny, nx = map(int, meta["grid"])
        return [(0, 0, ny, nx)]
    return [tuple(int(v) for v in t["atlas_bbox"]) for t in tiles]


def _grad_uv(fields: dict, meta: dict):
    """Per-region spatial gradients of the block flow: du/dx, du/dy, dv/dx, dv/dy.

    Each is ``(T, ny, nx)`` in (px/s)/block. Axis 2 is x, axis 1 is y (atlas row/
    col). A region only one cell wide/tall in an axis has no resolvable gradient
    there, so that component is left zero rather than fabricated."""
    u = np.asarray(fields["u"], np.float32)
    v = np.asarray(fields["v"], np.float32)
    ux = np.zeros_like(u); uy = np.zeros_like(u)
    vx = np.zeros_like(v); vy = np.zeros_like(v)
    for y0, x0, y1, x1 in _regions(meta):
        us, vs = u[:, y0:y1, x0:x1], v[:, y0:y1, x0:x1]
        if us.shape[2] > 1:
            ux[:, y0:y1, x0:x1] = np.gradient(us, axis=2)
            vx[:, y0:y1, x0:x1] = np.gradient(vs, axis=2)
        if us.shape[1] > 1:
            uy[:, y0:y1, x0:x1] = np.gradient(us, axis=1)
            vy[:, y0:y1, x0:x1] = np.gradient(vs, axis=1)
    return ux, uy, vx, vy


@channel("vel_divergence", needs=("u", "v"))
def _vel_divergence(fields: dict, meta: dict) -> np.ndarray:
    """du/dx + dv/dy -- the trace of ∇v. Signed: positive = local expansion
    (looming/spreading), negative = contraction."""
    ux, _uy, _vx, vy = _grad_uv(fields, meta)
    return ux + vy


@channel("vel_vorticity", needs=("u", "v"))
def _vel_vorticity(fields: dict, meta: dict) -> np.ndarray:
    """dv/dx - du/dy -- the antisymmetric part of ∇v. Signed: sign is the sense
    of rotation, magnitude the local rotation rate."""
    _ux, uy, vx, _vy = _grad_uv(fields, meta)
    return vx - uy


@channel("vel_shear", needs=("u", "v"))
def _vel_shear(fields: dict, meta: dict) -> np.ndarray:
    """Deviatoric strain-rate magnitude -- the symmetric traceless part of ∇v.

    ``sqrt((du/dx - dv/dy)^2 + (du/dy + dv/dx)^2)``: nonnegative, translation-
    AND rotation-invariant, so it isolates pure deformation (stretch + shear,
    with the isotropic expansion removed to ``vel_divergence``). Proportional to
    the deviatoric rate-of-strain tensor's Frobenius norm (the sqrt(2) constant
    is irrelevant to a tail threshold). This is the configural read: a rigid
    body translating or rotating gives zero; a limb extending or a body arching
    lights it up."""
    ux, uy, vx, vy = _grad_uv(fields, meta)
    return np.sqrt((ux - vy) ** 2 + (uy + vx) ** 2)


# The velocity-gradient channels as a group, in decomposition order. A caller
# wiring them into a UI or a scoring sweep names the group rather than restating
# the members (and so cannot drift from the registry).
VELOCITY_GRADIENT = ("vel_divergence", "vel_shear", "vel_vorticity")
