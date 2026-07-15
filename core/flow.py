"""Pluggable dense optical flow backends behind one interface.

Every backend takes two preprocessed grayscale float32 frames and returns a
HxWx2 float32 flow field in pixels/frame (u = horizontal, v = vertical).
Conversion to physical units (px/s) happens once, at block-reduction time,
because the cache is in seconds and Hz, never frames.

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


def backend_status() -> dict[str, str]:
    """Per-backend availability message, for the Tab 1 backend selector."""
    status = {
        "farneback": "Available (CPU). Fast; coarse motion.",
        "dis": "Available (CPU). Better on subtle motion.",
    }
    try:
        import torch  # noqa: F401
    except ImportError:
        status["raft"] = ("Unavailable: PyTorch is not installed. "
                          "Install torch + torchvision with CUDA to enable.")
        return status
    if gpu_available():
        status["raft"] = "Available (GPU). Best on fine motion."
    else:
        status["raft"] = ("Unavailable: PyTorch is installed but no CUDA GPU was "
                          "detected. RAFT on CPU is too slow to be usable.")
    return status


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
    large finite error rather than NaN/inf so cached histograms remain usable.

    This is deliberately a continuous diagnostic.  The absolute/relative
    rejection threshold belongs at analysis time and is not baked into the raw
    cached flow.
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


def reduce_scalar_to_blocks(values: np.ndarray, block: int,
                            statistic: str = "mean") -> np.ndarray:
    """Reduce one pixelwise scalar field to the same block grid as flow.

    ``p90`` is used for forward/backward error so a bad minority is not hidden by
    a quiet block mean.  Structure tensor strength uses ``mean`` because it is a
    continuous amount of local image evidence, not an outlier diagnostic.
    """
    if values.ndim != 2:
        raise ValueError("scalar field must have shape HxW")
    h, w = values.shape
    ny, nx = h // block, w // block
    if ny == 0 or nx == 0:
        raise ValueError(
            f"Block size {block} is larger than the scalar field ({w}x{h}).")
    x = values[:ny * block, :nx * block]
    cells = x.reshape(ny, block, nx, block).transpose(0, 2, 1, 3)
    cells = cells.reshape(ny, nx, block * block)
    if statistic == "mean":
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

def reduce_to_blocks(flow: np.ndarray, block: int, fps: float
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
    n_by, n_bx = h // block, w // block
    if n_by == 0 or n_bx == 0:
        raise ValueError(
            f"Block size {block} is larger than the working frame ({w}x{h}). "
            f"Lower the block size or raise the downsample factor."
        )

    # Crop the ragged right/bottom edge rather than padding it: a partial block
    # would have a different pixel count and bias its own statistics.
    f = flow[: n_by * block, : n_bx * block]
    u = f[..., 0].reshape(n_by, block, n_bx, block)
    v = f[..., 1].reshape(n_by, block, n_bx, block)

    u_mean = u.mean(axis=(1, 3))
    v_mean = v.mean(axis=(1, 3))
    speed_mean = np.sqrt(u * u + v * v).mean(axis=(1, 3))

    # px/frame -> px/s. Time is in seconds everywhere downstream.
    return (u_mean * fps).astype(np.float32), \
           (v_mean * fps).astype(np.float32), \
           (speed_mean * fps).astype(np.float32)
