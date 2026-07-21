"""Pluggable dense optical flow backends behind one interface.

**Mostly dormant.** These backends served the retired flow cache. The live
detection path solves flow from the structure tensor instead
(``core.structure_tensor.flow_from_tensor``), and the only thing imported from
this module today is ``reduce_scalar_to_blocks``, used by
``core.tensor_channels``. ``FlowConfig.backend`` still accepts
farneback/dis/raft, but nothing on the tensor path consults it -- do not read
that field as a claim that the solver is selectable.

Every backend takes two preprocessed grayscale float32 frames and returns a
HxWx2 float32 flow field in pixels/frame (u = horizontal, v = vertical).
Conversion to physical units (px/s) happens once, at block-reduction time,
because everything downstream is in seconds and Hz, never frames.

Availability is probed rather than assumed: RAFT needs torch + a GPU, and
selecting it without one must fail with a clear message rather than silently
falling back to something slower and different.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from core.config import FlowConfig


class FlowBackend(ABC):
    name: str = "base"

    @abstractmethod
    def compute(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Return HxWx2 float32 flow in pixels/frame."""

    def close(self) -> None:
        pass


class FarnebackBackend(FlowBackend):
    """Dense polynomial-expansion flow. CPU. The default.

    Adequate for coarse-scale, high-amplitude motion (wingbeat, walking). It
    over-smooths small structures, so antennal motion or grooming will be
    under-resolved -- use DIS or RAFT for those.
    """
    name = "farneback"

    def __init__(self, cfg: FlowConfig):
        self.cfg = cfg

    def compute(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        c = self.cfg
        return cv2.calcOpticalFlowFarneback(
            prev.astype(np.uint8), curr.astype(np.uint8), None,
            c.fb_pyr_scale, c.fb_levels, c.fb_winsize,
            c.fb_iterations, c.fb_poly_n, c.fb_poly_sigma, 0,
        )


class DISBackend(FlowBackend):
    """Dense Inverse Search. CPU, faster than Farneback at similar or better
    quality, and noticeably better on subtle motion."""
    name = "dis"

    _PRESETS = {
        0: cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        1: cv2.DISOPTICAL_FLOW_PRESET_FAST,
        2: cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }

    def __init__(self, cfg: FlowConfig):
        preset = self._PRESETS.get(cfg.dis_preset, cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.dis = cv2.DISOpticalFlow_create(preset)

    def compute(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        return self.dis.calc(prev.astype(np.uint8), curr.astype(np.uint8), None)


class RAFTBackend(FlowBackend):
    """RAFT (recurrent all-pairs field transforms) via torchvision. GPU.

    Best available quality on fine motion. Requires torch with CUDA; we refuse
    to run it on CPU because a full clip would take days.
    """
    name = "raft"

    def __init__(self, cfg: FlowConfig):
        import torch
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

        if not torch.cuda.is_available():
            raise RuntimeError(
                "RAFT requires a CUDA GPU and none is available. "
                "Use the DIS backend for the best CPU-only quality."
            )
        self.torch = torch
        self.device = torch.device("cuda")
        self.iters = cfg.raft_iters
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
        self.model = self.model.eval().to(self.device)

    def _prep(self, gray: np.ndarray):
        torch = self.torch
        # RAFT wants 3-channel, normalized to [-1, 1], and dims divisible by 8.
        h, w = gray.shape
        ph, pw = (-h) % 8, (-w) % 8
        img = np.clip(gray, 0, 255).astype(np.float32) / 255.0
        img = np.stack([img] * 3, axis=0)
        t = torch.from_numpy(img)[None].to(self.device)
        if ph or pw:
            t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="replicate")
        return t * 2.0 - 1.0, h, w

    def compute(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        torch = self.torch
        a, h, w = self._prep(prev)
        b, _, _ = self._prep(curr)
        with torch.no_grad():
            flows = self.model(a, b, num_flow_updates=self.iters)
        flow = flows[-1][0].permute(1, 2, 0).cpu().numpy()
        return flow[:h, :w].astype(np.float32)

    def close(self) -> None:
        if getattr(self, "torch", None) is not None:
            self.torch.cuda.empty_cache()


_BACKENDS = {
    "farneback": FarnebackBackend,
    "dis": DISBackend,
    "raft": RAFTBackend,
}


def gpu_available() -> bool:
    """True if RAFT can actually run. Probed once at startup by the UI."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False




def create_backend(cfg: FlowConfig) -> FlowBackend:
    if cfg.backend not in _BACKENDS:
        raise ValueError(f"Unknown flow backend: {cfg.backend}")
    return _BACKENDS[cfg.backend](cfg)


# -- flow diagnostics -------------------------------------------------------

def forward_backward_error(forward: np.ndarray,
                           backward: np.ndarray) -> np.ndarray:
    """Pixelwise forward/backward residual in working pixels per frame.

    ``forward`` maps frame t -> t+1 and ``backward`` maps t+1 -> t.  The
    backward vector must therefore be sampled at x + F(x) before the two vectors
    are added.  Pixels whose forward endpoint leaves the image are assigned a
    large finite error rather than NaN/inf so downstream histograms remain usable.

    This is deliberately a continuous diagnostic.  The absolute/relative
    rejection threshold belongs at analysis time and is never baked into the raw
    flow field.
    """
    if forward.shape != backward.shape or forward.ndim != 3 \
            or forward.shape[2] != 2:
        raise ValueError("forward and backward flow must both have shape HxWx2")

    h, w = forward.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    map_x = xx + forward[..., 0].astype(np.float32, copy=False)
    map_y = yy + forward[..., 1].astype(np.float32, copy=False)
    inside = (map_x >= 0.0) & (map_x <= w - 1) & \
             (map_y >= 0.0) & (map_y <= h - 1)

    bx = cv2.remap(backward[..., 0].astype(np.float32, copy=False),
                   map_x, map_y, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    by = cv2.remap(backward[..., 1].astype(np.float32, copy=False),
                   map_x, map_y, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    err = np.hypot(forward[..., 0] + bx, forward[..., 1] + by).astype(np.float32)
    cap = np.float32(max(h, w))
    err[~inside] = cap
    return np.minimum(err, cap).astype(np.float32, copy=False)


def _ragged_cells(values: np.ndarray, block: int, h: int, w: int,
                  ny: int, nx: int):
    """Yield ``(by, bx, cell)`` for the partial cells at the bottom/right edge.

    When ``include_partial``, ny/nx use ceiling division, so the last row and
    column can be short. Those cells are the only ones that fall outside the
    regular (ny, block, nx, block) grid, and both the mean and p90 reductions
    need exactly the same set -- keeping the geometry in one place stops a fix to
    edge handling from being applied to one statistic and not the other.
    """
    full_y, full_x = h // block, w // block
    for by in range(ny):
        for bx in range(nx):
            if by < full_y and bx < full_x:
                continue
            yield by, bx, values[by * block:min(h, (by + 1) * block),
                                 bx * block:min(w, (bx + 1) * block)]


def _block_mean(values: np.ndarray, block: int, h: int, w: int,
                ny: int, nx: int) -> np.ndarray:
    """Block-mean without padding or transposing the full plane.

    This is the extraction hot path -- five calls per frame per tile, measured at
    14% of a 5.3K pass and 32% of a small one. The generic route below pads the
    whole field with NaN, transposes it (which forces a copy at the following
    reshape), and runs ``nanmean``: three full-size temporaries for what is a
    strided sum. Reshaping to (ny, block, nx, block) and reducing axes 1 and 3
    needs none of them. Only the ragged last row/column fall outside the regular
    grid, and there are at most ny + nx of those cells.

    Those edge cells are reduced as three SLABS rather than one at a time. The
    per-cell loop was correct but paid numpy's fixed per-call overhead on cells
    of a few thousand pixels: a 349x321 replicate tile at block 64 has 25 regular
    cells and **11** ragged ones, so the loop ran 11 tiny ``mean`` calls per tile
    per frame -- 24 200 of them in a 200-frame pass, and most of this function's
    cost. The geometry makes the slabs possible: ``include_partial`` uses ceiling
    division, so there is at most ONE short row and ONE short column, and every
    cell in the bottom strip shares a height (every cell in the right strip, a
    width). Only the corner is genuinely alone. 1.7-3x, and reducing the same
    elements per cell as before.
    """
    out = np.empty((ny, nx), np.float32)
    full_y, full_x = h // block, w // block
    rem_y, rem_x = h - full_y * block, w - full_x * block
    # Gated on ny/nx exceeding the regular grid, NOT on rem_y/rem_x being
    # nonzero: without include_partial, ny == full_y even when the plane has a
    # remainder, and the short row is meant to be dropped. Writing out[full_y]
    # then would be an IndexError.
    if full_y and full_x:
        core = values[:full_y * block, :full_x * block]
        out[:full_y, :full_x] = core.reshape(
            full_y, block, full_x, block).mean(axis=(1, 3), dtype=np.float32)
    if ny > full_y and full_x:              # bottom strip: rem_y tall, block wide
        out[full_y, :full_x] = values[full_y * block:, :full_x * block].reshape(
            rem_y, full_x, block).mean(axis=(0, 2), dtype=np.float32)
    if nx > full_x and full_y:              # right strip: block tall, rem_x wide
        out[:full_y, full_x] = values[:full_y * block, full_x * block:].reshape(
            full_y, block, rem_x).mean(axis=(1, 2), dtype=np.float32)
    if ny > full_y and nx > full_x:         # the single corner cell
        out[full_y, full_x] = values[full_y * block:, full_x * block:].mean(
            dtype=np.float32)
    return out


def reduce_scalar_to_blocks(values: np.ndarray, block: int,
                            statistic: str = "mean",
                            include_partial: bool = False) -> np.ndarray:
    """Reduce one pixelwise scalar field to the same block grid as flow.

    ``p90`` is used for forward/backward error so a bad minority is not hidden by
    a quiet block mean.  Structure tensor strength uses ``mean`` because it is a
    continuous amount of local image evidence, not an outlier diagnostic.
    """
    if values.ndim != 2:
        raise ValueError("scalar field must have shape HxW")
    h, w = values.shape
    ny = ((h + block - 1) // block) if include_partial else h // block
    nx = ((w + block - 1) // block) if include_partial else w // block
    if ny == 0 or nx == 0:
        raise ValueError(
            f"Block size {block} is larger than the scalar field ({w}x{h}).")
    if include_partial and statistic == "p90":
        # Use the fast partition implementation for every complete cell. Only
        # the final row/column are ragged, so compute those few small percentiles
        # directly instead of sending the whole plane through nanpercentile.
        out = np.empty((ny, nx), np.float32)
        full_y, full_x = h // block, w // block
        if full_y and full_x:
            out[:full_y, :full_x] = reduce_scalar_to_blocks(
                values[:full_y * block, :full_x * block], block, "p90")
        for by, bx, cell in _ragged_cells(values, block, h, w, ny, nx):
            out[by, bx] = np.percentile(cell, 90)
        return out

    if statistic == "mean":
        out = _block_mean(values, block, h, w, ny, nx)
        if not np.isnan(out).any():
            return out
        # NaN reached the output. Either the field itself carries NaN, or (when
        # include_partial) a padded cell was all-NaN -- which ceiling division
        # makes impossible, so it is the field. Fall through to the nanmean
        # route below, which skips NaN per cell the way this path cannot.

    if include_partial:
        x = np.pad(values.astype(np.float32, copy=False),
                   ((0, ny * block - h), (0, nx * block - w)),
                   constant_values=np.nan)
    else:
        x = values[:ny * block, :nx * block]
    cells = x.reshape(ny, block, nx, block).transpose(0, 2, 1, 3)
    cells = cells.reshape(ny, nx, block * block)
    if statistic == "mean":
        if include_partial:
            return np.nanmean(cells, axis=2, dtype=np.float32).astype(np.float32)
        return cells.mean(axis=2, dtype=np.float32).astype(np.float32)
    if statistic == "p90":
        # np.percentile promotes/intermediates aggressively and drove the 5.3K
        # validation pass above 1 GB RSS. There are only block*block values, so
        # select the two adjacent order statistics directly and linearly
        # interpolate exactly as numpy's default percentile method does.
        rank = 0.9 * (cells.shape[2] - 1)
        lo, hi = int(np.floor(rank)), int(np.ceil(rank))
        part = np.partition(cells, (lo, hi), axis=2)
        if lo == hi:
            return part[..., lo].astype(np.float32)
        frac = np.float32(rank - lo)
        return (part[..., lo] * (1.0 - frac) +
                part[..., hi] * frac).astype(np.float32)
    raise ValueError(f"Unknown block statistic: {statistic}")


# -- block reduction ---------------------------------------------------------

def reduce_to_blocks(flow: np.ndarray, block: int, fps: float,
                     include_partial: bool = False
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a per-pixel flow field to per-block statistics, in px/s.

    Returns (u_mean, v_mean, speed_mean), each (n_by, n_bx) float32.

    We store the mean VECTOR (u, v) and the mean SPEED separately rather than
    storing "magnitude and angle" directly, for two reasons:

    1. Averaging angle over a block is a circular-mean bug: a block whose pixels
       point at 1 degree and 359 degrees averages to 180, the exact opposite of
       the truth. Averaging the vector components and taking atan2 afterwards is
       the correct circular mean, weighted by magnitude.

    2. The pair (|mean vector|, mean speed) is strictly more informative than
       either alone, and their ratio is angular coherence:

           coherence = |mean(v)| / mean(|v|)  in [0, 1]

       ~1 means every pixel in the block moves the same way (translation);
       ~0 means the block has lots of motion that cancels out (oscillation).
       A wingbeat is exactly the low-coherence, high-speed, low-net-flow case.
       This makes coherence, net flow, and angle all free derived features
       instead of paid cache expansions.
    """
    h, w = flow.shape[:2]
    n_by = ((h + block - 1) // block) if include_partial else h // block
    n_bx = ((w + block - 1) // block) if include_partial else w // block
    if n_by == 0 or n_bx == 0:
        raise ValueError(
            f"Block size {block} is larger than the working frame ({w}x{h}). "
            f"Lower the block size or raise the downsample factor."
        )

    if include_partial:
        f = np.pad(flow.astype(np.float32, copy=False),
                   ((0, n_by * block - h), (0, n_bx * block - w), (0, 0)),
                   constant_values=np.nan)
    else:
        # Full-frame legacy mode retains its historical crop semantics.
        f = flow[: n_by * block, : n_bx * block]
    u = f[..., 0].reshape(n_by, block, n_bx, block)
    v = f[..., 1].reshape(n_by, block, n_bx, block)

    reduce = np.nanmean if include_partial else np.mean
    u_mean = reduce(u, axis=(1, 3))
    v_mean = reduce(v, axis=(1, 3))
    speed_mean = reduce(np.sqrt(u * u + v * v), axis=(1, 3))

    # px/frame -> px/s. Time is in seconds everywhere downstream.
    return (u_mean * fps).astype(np.float32), \
           (v_mean * fps).astype(np.float32), \
           (speed_mean * fps).astype(np.float32)
