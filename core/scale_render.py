"""Render what a replicate actually looks like to the pipeline at each scale.

The downsample dialog prices the scale lever in wall time and storage, and
deliberately refuses to score whether a coarser scale still *detects* a
behaviour (see ``core/scale_sweep.py`` for why the empirical detection panel was
removed). That leaves a real gap: the user is asked to give up resolution with
nothing showing them what resolution they are giving up. This module fills it
with the one thing that is neither a measurement of sensitivity nor an invented
score -- the actual image the per-pixel stages receive.

Two fields, and the second is the one that matters
---------------------------------------------------
* the **working grey image**, straight out of :class:`~core.preprocess.Preprocessor`
  at that scale -- literally the array the structure-tensor products are built
  from, not an approximation of it;
* the **frame-difference field** ``|I(t+1) - I(t)|``, computed *after* the
  downsample, in that order, because the order is the whole point. ``change`` is
  ``<I_t^2>`` per block (``core/tensor_channels.py``), so this is the per-pixel
  quantity the detector's band power is built out of.

Downsampling averages pixels **before** the differencing, so the difference field
dims as the scale falls. That is exactly the mechanism behind the finding that
retired the detection panel: per-block band power falls with scale, so a fixed
absolute threshold catches less whether or not the behaviour is still resolved.
Showing the mechanism is honest and useful; reporting the frames it costs a
particular tuning was not.

Why the display range is shared, and why there is no number
------------------------------------------------------------
Every tile is normalised against the **reference** (largest) scale's range, never
against its own. Auto-contrasting each tile would make every scale look equally
vivid and hide the amplitude loss completely -- the same class of error as the
withdrawn ``sig_corr`` reading, where a per-scale normalisation made a real
difference disappear into a flattering aggregate.

For the same reason nothing here returns a scalar. An "RMS change retained: 61%"
would be measured, would look authoritative, and would be read as a sensitivity
score -- which it is not, since dimmer is not the same as unresolved and the fix
for dimmer is re-tuning the threshold. The images carry the information; the
caller states the caveat in prose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.config import PreprocessConfig
from core.preprocess import Preprocessor
from core.video import VideoSource


@dataclass(frozen=True)
class ScaleRender:
    """One replicate box rendered at one working scale.

    ``gray`` and ``change`` are uint8 at *working* resolution, i.e. genuinely
    smaller at a smaller scale. They must be displayed at a common size with
    NEAREST upscaling: smooth interpolation would reintroduce detail the
    pipeline no longer has, and shrinking the tile instead would read as "the
    picture got smaller" rather than "the picture got coarser".
    """
    scale: float
    width: int
    height: int
    gray: np.ndarray
    change: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.scale:.2f}"

    @property
    def size_label(self) -> str:
        return f"{self.width}x{self.height} px"


def _preprocess_cfg(base: PreprocessConfig | None, scale: float) -> PreprocessConfig:
    """The pipeline's preprocessing at ``scale``, with the fitted stages off.

    ``registration`` and ``bg_subtract`` are forced off exactly as the tensor
    path forces them off (``core/tensor_channels.py``): they are stateful across
    frames, and a two-frame render cannot reproduce that state. ``normalize`` is
    carried through, because it is per-frame and does change what the solve sees.
    """
    return PreprocessConfig(
        downsample=float(scale),
        normalize=(base.normalize if base else "off"),
        denoise="off", registration="off", bg_subtract="off", mask_path=None)


def render_box_at_scales(video_path: str, box: tuple[int, int, int, int],
                         frame_idx: int, scales,
                         *, base_cfg: PreprocessConfig | None = None,
                         gap: int = 1) -> list[ScaleRender]:
    """Render one replicate box at each of ``scales``, from one pair of frames.

    ``box`` is ``(x0, y0, x1, y1)`` in SOURCE pixels -- a tile's ``source_box``.
    ``gap`` is the frame step used for the difference field and must match what
    extraction differences across (1: consecutive frames).

    Decodes the pair once and reuses it for every scale, so the cost is one seek
    regardless of how many scales are asked for. The seek dominates on long-GOP
    footage, which is why this takes a frame index rather than being called per
    scale.
    """
    scales = [float(s) for s in scales]
    if not scales:
        return []

    x0, y0, x1, y1 = (int(v) for v in box)
    with VideoSource(video_path) as src:
        idx = max(0, min(int(frame_idx), max(0, src.info.frame_count - 1 - gap)))
        a = src.frame_at(idx)
        b = src.frame_at(idx + gap)
    if a is None:
        raise IOError(f"Could not read frame {frame_idx} of {video_path}")
    if b is None:
        # A window ending on the last frame is ordinary, not an error: fall back
        # to differencing the frame against itself so the grey half still
        # renders, rather than failing the whole panel.
        b = a

    crop_a = a[y0:y1, x0:x1]
    crop_b = b[y0:y1, x0:x1]
    if crop_a.size == 0:
        raise ValueError(f"Replicate box {box} is empty for this source")

    bw, bh = x1 - x0, y1 - y0
    fields: list[tuple[float, np.ndarray, np.ndarray]] = []
    for s in scales:
        pre = Preprocessor(_preprocess_cfg(base_cfg, s), bw, bh)
        ga = pre.apply(crop_a)
        gb = pre.apply(crop_b)
        fields.append((s, ga, np.abs(gb - ga)))

    # The reference is the LARGEST scale asked for, not necessarily 1.0 -- the
    # caller may be comparing a narrower range, and normalising against a scale
    # that was never rendered would put the whole strip on an invisible axis.
    ref = max(fields, key=lambda f: f[0])[2]
    hi = float(np.percentile(ref, 99.5)) if ref.size else 0.0
    if hi <= 1e-6:
        # A static pair: nothing moved anywhere, at any scale. Normalising
        # against ~0 would amplify quantisation noise into a convincing-looking
        # difference field showing motion that is not there.
        hi = 1.0

    out = []
    for s, gray, diff in fields:
        out.append(ScaleRender(
            scale=s, width=gray.shape[1], height=gray.shape[0],
            gray=np.clip(gray, 0, 255).astype(np.uint8),
            change=np.clip(diff * (255.0 / hi), 0, 255).astype(np.uint8)))
    return out


def fit_box_to(box: tuple[int, int, int, int], max_px: int
               ) -> tuple[int, int, int, int]:
    """Shrink a source box about its centre so its longest side is ``max_px``.

    A replicate box can be the whole frame (the uncropped fallback), and
    rendering 5312x2988 at seven scales to fill a 160 px thumbnail wastes the
    decode and, worse, makes the strip useless: at that reduction every scale
    looks identical because the display is the bottleneck, not the working
    resolution. Cropping to a centre window keeps the comparison at a
    magnification where the scale difference is actually visible.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = x1 - x0, y1 - y0
    if max(w, h) <= max_px:
        return x0, y0, x1, y1
    k = max_px / float(max(w, h))
    nw, nh = max(2, int(round(w * k))), max(2, int(round(h * k)))
    cx, cy = x0 + w // 2, y0 + h // 2
    nx0 = max(x0, cx - nw // 2)
    ny0 = max(y0, cy - nh // 2)
    return nx0, ny0, min(x1, nx0 + nw), min(y1, ny0 + nh)
