"""The 3D spatiotemporal structure tensor as one representation for motion,
texture and temporal change.

Optical flow, spatial texture strength and temporal change energy are three
reads of a single per-block object: the windowed second moment of the
spatiotemporal gradient (I_x, I_y, I_t). This module computes that object and
the three reads, so the pipeline can cache one thing and every explorer can
derive its channel from it.

    J = < g g^T >_window ,   g = (I_x, I_y, I_t)

stored as its 6 unique components in the fixed order used everywhere here:

    (xx, yy, tt, xy, xt, yt)

What each read recovers:

  * flow      -- solve the 2x2 spatial system  [[xx, xy], [xy, yy]] v = -[xt, yt].
                 This is Lucas-Kanade: the gradient-method flow IS the tensor.
                 Degenerate (small spatial determinant) exactly where the aperture
                 problem bites -- textureless regions -- and returns zero there.
  * texture   -- the smaller eigenvalue of the spatial 2x2 block, the Shi-Tomasi /
                 min-eigen cornerness: how well-conditioned the flow solve is.
  * change    -- tt directly: the derivative energy <I_t^2>. Fast-weighted, so it
                 is the flicker/appearance channel, NOT amplitude variance. The DC
                 level the gradient discards (hence amplitude variance) must come
                 from a separately cached mean intensity.

Gradients use central differences for I_x, I_y and a one-frame forward difference
for I_t, so flow comes out in pixels/frame -- the same units the flow backends
produce before the px/frame -> px/s conversion at block-reduction time.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

# Fixed component order. Import this rather than hard-coding strings so a cache
# schema and an explorer can never disagree about which plane is which.
COMPONENTS = ("xx", "yy", "tt", "xy", "xt", "yt")


# Which raw gradients each component is built from. Read by ``tensor_products``
# to skip a spatial gradient no requested component needs -- ``np.gradient`` over
# a full-resolution plane is not cheap, and the change-only pass needs none of it.
_NEEDS_SPATIAL = frozenset({0, 1, 3, 4, 5})     # anything with an ix or iy in it
_NEEDS_TEMPORAL = frozenset({2, 4, 5})          # anything with an it in it


def tensor_products(prev: np.ndarray, curr: np.ndarray,
                    components: Iterable[int] | None = None) -> list:
    """Pixelwise outer-product planes of the spatiotemporal gradient.

    Returns a length-6 list of (H, W) float32 planes in ``COMPONENTS`` order. The
    windowing that turns these raw products into a rank-informative tensor is the
    downstream block reduction (mean over each block), so this stays a pure
    pointwise step.

    ``components`` restricts the work to the listed indices; the rest come back
    as **``None``**, not as zero planes. A reader that was not expecting a
    partial tensor then fails with a ``TypeError`` on first arithmetic, whereas
    zeros would flow through ``flow_from_tensor`` and ``spatial_min_eigen`` as a
    plausible-looking answer -- the silent-false-negative failure this codebase
    designs against everywhere else (``FINDINGS.md`` sections 10, 11).

    This matters because the components are not equally priced and no consumer
    needs all six. ``change`` is ``tt`` alone: one squared temporal difference,
    no spatial gradient at all. Computing the other five, and blurring them,
    was ~81% of a change-only pass on 5.3K footage.
    """
    if prev.shape != curr.shape or curr.ndim != 2:
        raise ValueError("prev and curr must be the same HxW grayscale frame")
    want = frozenset(range(6) if components is None else components)
    if not want <= frozenset(range(6)):
        raise ValueError(f"component indices must be in 0..5, got {sorted(want)}")
    c = curr.astype(np.float32, copy=False)
    out: list = [None] * 6
    if not want:
        return out
    if want & _NEEDS_TEMPORAL:
        it = c - prev.astype(np.float32, copy=False)
    if want & _NEEDS_SPATIAL:
        iy, ix = np.gradient(c)         # np.gradient returns d/drow, d/dcol
    if 0 in want:
        out[0] = ix * ix
    if 1 in want:
        out[1] = iy * iy
    if 2 in want:
        out[2] = it * it
    if 3 in want:
        out[3] = ix * iy
    if 4 in want:
        out[4] = ix * it
    if 5 in want:
        out[5] = iy * it
    return out


def _unpack(J) -> tuple[np.ndarray, ...]:
    """The six components as float32 planes, from a (6, H, W) array or a
    length-6 sequence.

    A ``None`` component is rejected here rather than allowed to propagate: these
    readers each combine several components, so a partial tensor reaching one of
    them means the caller under-requested, and the useful place to say so is at
    the read, naming the component.
    """
    if len(J) != 6:
        raise ValueError("J must have 6 components on axis 0, in COMPONENTS order")
    missing = [COMPONENTS[i] for i in range(6) if J[i] is None]
    if missing:
        raise ValueError(f"tensor is missing component(s) {', '.join(missing)}; "
                         f"this read needs all six")
    return tuple(np.asarray(J[i], np.float32) for i in range(6))


def flow_from_tensor(J: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Block flow (u, v) in px/frame from a block-level tensor.

    ``J`` is (6, ny, nx). Solves the 2x2 spatial system per block; blocks whose
    spatial determinant is below ``eps`` (no resolvable texture) get zero flow,
    which is the honest answer under the aperture problem rather than a divide-by-
    near-zero explosion.
    """
    xx, yy, tt, xy, xt, yt = _unpack(J)
    det = xx * yy - xy * xy
    safe = np.abs(det) > eps
    inv = np.where(safe, 1.0 / np.where(safe, det, 1.0), 0.0)
    u = -(yy * xt - xy * yt) * inv
    v = -(xx * yt - xy * xt) * inv
    return np.stack([u.astype(np.float32), v.astype(np.float32)], axis=-1)


def spatial_min_eigen(J: np.ndarray) -> np.ndarray:
    """Smaller eigenvalue of the spatial 2x2 tensor per block (Shi-Tomasi).

    High where both spatial gradient directions carry energy (a corner/texture,
    where flow is well-posed); ~0 along an edge (aperture) or in a flat region.
    """
    xx, yy, tt, xy, xt, yt = _unpack(J)
    tr = xx + yy
    disc = np.sqrt(np.maximum((xx - yy) ** 2 + 4.0 * xy * xy, 0.0))
    return (0.5 * (tr - disc)).astype(np.float32)


def temporal_energy(J) -> np.ndarray:
    """Change energy <I_t^2> per block: the tt component. Fast-weighted.

    Reads component 2 directly rather than through ``_unpack``: this is the one
    read that needs a single component, so it must stay valid on the partial
    tensor a change-only pass builds.
    """
    if J[2] is None:
        raise ValueError("tensor is missing component tt")
    return np.asarray(J[2], np.float32)


def flow_residual(J: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Change energy that NO single motion can explain: the brightness-constancy
    residual, per block.

    The gradient-method flow minimises the constraint residual; at its optimum the
    leftover energy is  R = J_tt - b^T M^-1 b, with M the spatial 2x2 block and
    b = (xt, yt). Coherent translation of texture drives R -> 0 (motion accounts
    for the change); flicker, translucency and appear/disappear drive R high (the
    change is real but no displacement explains it). Where M is singular -- no
    texture, so no motion could explain anything -- the entire J_tt is residual.

    This is the channel that isolates appearance change from motion change, which
    raw J_tt (total change energy) does not: it is the same residual r in
    I_t = -grad(I).v + r that separates a moving edge from a backlit wing.
    """
    xx, yy, tt, xy, xt, yt = _unpack(J)
    det = xx * yy - xy * xy
    safe = np.abs(det) > eps
    inv = np.where(safe, 1.0 / np.where(safe, det, 1.0), 0.0)
    explained = (yy * xt * xt - 2.0 * xy * xt * yt + xx * yt * yt) * inv
    r = np.where(safe, tt - explained, tt)
    return np.maximum(r, 0.0).astype(np.float32)


def brightness_residual_field(prev: np.ndarray, curr: np.ndarray,
                              flow_uv: np.ndarray) -> np.ndarray:
    """Per-pixel brightness-constancy residual against an EXTERNAL flow field.

    r = I_t + grad(I).v , with ``flow_uv`` the (H, W, 2) forward flow in px/frame
    from whatever backend you trust (Farneback/DIS/RAFT). Unlike the tensor's own
    first-order residual, this is not limited to small-displacement linearisation:
    where a good pyramidal flow genuinely explains the motion, r -> 0, so what
    survives is appearance change no displacement accounts for -- the flicker /
    occlusion / backlit-wing signal. Square and block-reduce for a residual-energy
    channel.

    This is the clean extraction the main pipeline should cache: it reuses the
    flow it already computes, and needs only the current frame's spatial gradient.
    """
    if prev.shape != curr.shape or curr.ndim != 2:
        raise ValueError("prev and curr must be the same HxW grayscale frame")
    if flow_uv.shape[:2] != curr.shape or flow_uv.shape[-1] != 2:
        raise ValueError("flow_uv must be HxWx2 matching the frame")
    c = curr.astype(np.float32, copy=False)
    iy, ix = np.gradient(c)
    it = c - prev.astype(np.float32, copy=False)
    return (it + ix * flow_uv[..., 0] + iy * flow_uv[..., 1]).astype(np.float32)


def flow_coherence(J: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """How plane-like the local spatiotemporal energy is, in [0, 1].

    Derived from the 2x2 spatial tensor's eigenvalue ratio: ~1 when one spatial
    orientation dominates (a well-defined moving edge/texture -> trustworthy
    flow), ~0 when spatial energy is isotropic (corner or noise). This is the
    per-tensor sibling of the block flow's angular coherence.
    """
    xx, yy, tt, xy, xt, yt = _unpack(J)
    tr = xx + yy
    disc = np.sqrt(np.maximum((xx - yy) ** 2 + 4.0 * xy * xy, 0.0))
    lam1 = 0.5 * (tr + disc)
    lam2 = 0.5 * (tr - disc)
    denom = lam1 + lam2
    return np.where(denom > eps, (lam1 - lam2) / np.maximum(denom, eps),
                    0.0).astype(np.float32)
