"""Time a real extraction pass at each candidate scale, and price its storage.

This is what feeds the downsample dialog's frontier. It replaces the empirical
*detection* sweep that briefly lived here (see todo.md Batch M): running the
tuned detector at each scale answered "does this particular absolute threshold
still trip", which is a fact about the current tuning rather than about the
organism, and the measured result was a monotone frame loss with zero added
frames at every scale -- the signature of threshold drift, not of lost structure.
A table that looks measured and does not mean what it appears to is the failure
the plan names explicitly, so the detector came out and the timing stayed.

Why the passes run at the PRODUCTION block, not block=1
--------------------------------------------------------
The live surface extracts at ``block_size=1`` whenever the per-pixel cache fits,
so that a Block change is a cheap re-reduce. That is right for tuning and wrong
for measurement: at block=1 no reduction happens, and ``block_reduce`` was
measured at 62% of such a pass against ~15% at block 64 -- 11.0 s against 4.2 s
for the same scale on GX010047c2. A model fitted to those passes overstates a
batch run by ~2.6x. These passes resolve the block the way a batch run would, so
each is a legitimate :class:`~core.cost_model.PassSample`.

Storage is arithmetic, not measured -- and it does NOT fall with scale
----------------------------------------------------------------------
Cell counts come from the real layout, so no pass is needed to price storage --
but it is reported per scale anyway, because what it does is the point. With the
``auto`` block (the default) the block tracks the scale, so the grid is
scale-invariant *by construction* and downsampling buys no storage at all. That
is the two-lever claim made visible -- ``downsample`` is the compute lever,
``block_size`` is the storage lever -- and it is counterintuitive enough that
showing it is worth more than showing the time curve alone.

"Scale-invariant by construction" is not the same as constant in practice, and
the difference is measured, not theoretical. ``round(64 * s)`` and
``ceil(dim * s / block)`` round independently, so the cell count jitters:

* 7-replicate 5312x2988 layout: exactly 205 cells, 26.4 GB/100 h, at every scale
  from 1.0 down to 0.15 -- then 240 cells (30.9 GB) at 0.10. Atlas packing
  happens to absorb the rounding here.
* single full-frame replicate, same source: 539 GB at 1.0/0.75/0.50, **564 GB**
  at 0.35, 498 GB at 0.15, **615 GB** at 0.10. A 19% spread, not monotone.

So the honest claim is not "flat" but "does not reliably fall, and sometimes
rises". A user downsampling to save disk can end up paying MORE. That is worth
stating plainly rather than smoothing, and :func:`storage_rises_below` finds the
rising tail so the plot can mark it instead of letting it read as a glitch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.channel_source import live_channel_source
from core.cost_model import PassSample, atlas_cells, storage_bytes_per_hour


@dataclass(frozen=True)
class ScalePass:
    """One timed extraction pass at a candidate scale."""
    scale: float
    block: int
    grid: tuple[int, int]
    frames: int
    wall: float
    spans: dict[str, float] = field(default_factory=dict)
    approximated: bool = False
    # The decode ended before the requested window did, so ``frames`` counts what
    # was actually computed. Carried rather than inferred because a consumer
    # comparing scales cannot otherwise tell a cheap scale from a short pass.
    truncated: bool = False

    @property
    def seconds_per_frame(self) -> float:
        return self.wall / self.frames if self.frames > 0 else 0.0

    @property
    def usable(self) -> bool:
        """Whether this pass can enter a fit.

        A pass with no measured wall time would enter as a zero-cost point,
        pulling the fitted decode floor toward zero and putting the knee where
        aggressive downsampling looks free -- the one direction the cost model
        must never err in.

        A truncated pass is excluded for the same asymmetry. Its wall time covers
        the frames it managed, so admitting it with the requested frame count
        would understate seconds-per-frame and, again, move the knee toward
        "downsampling is free". Dropping the point loses one sample; keeping it
        biases every fit built from the sweep.
        """
        return self.frames > 0 and self.wall > 0.0 and not self.truncated

    def pass_sample(self) -> PassSample:
        return PassSample(scale=self.scale, frames=self.frames, wall=self.wall,
                          spans=dict(self.spans))


def measure_scale(video_path: str, cfg, replicates: list[dict], *, dims,
                  start: int, n: int, progress=None) -> ScalePass:
    """Extract ``[start, start+n)`` at ``cfg``'s scale and time it.

    The block is whatever ``cfg`` resolves it to, which for the ``auto`` default
    tracks the scale. No detector runs: this measures cost, and cost is all it
    claims to measure.
    """
    w, h, fps, fc = dims
    cd = live_channel_source(video_path, cfg, replicates, start=start, n=n,
                             width=w, height=h, fps=fps, frame_count=fc,
                             progress=progress)
    t = (cd.meta.get("timing") or {})
    ny, nx = cd.meta["grid"]
    scale = cfg.preprocess.resolve_downsample(w)
    return ScalePass(
        scale=float(t.get("scale", scale)),
        block=int(t.get("block", cfg.flow.resolve_block_size(scale))),
        grid=(int(ny), int(nx)),
        # ``cd.n_frames``, NOT the timer's ``frames``: the timer logs the window
        # that was *requested*, while the wall time it logs alongside covers only
        # the frames the decode actually delivered. Pairing the two divides real
        # seconds by imagined frames, which understates cost by exactly the
        # truncation ratio -- a 20-of-64 pass would price this scale at a third
        # of its true per-frame cost, and this number feeds the knee.
        frames=int(cd.n_frames),
        wall=float(t.get("wall", 0.0)),
        spans=dict(t.get("spans") or {}),
        approximated=bool(cd.approximated),
        truncated=bool(cd.meta.get("truncated", False)))


def storage_curve(replicates: list[dict], src_width: int, src_height: int,
                  scales, flow, fps: float, n_channels: int,
                  corpus_hours: float) -> list[float]:
    """Projected cache bytes for ``corpus_hours`` at each scale.

    Pure arithmetic over the real layout -- no pass required. ``flow`` resolves
    the block per scale, so this reports what the CURRENT block setting does:
    flat under ``auto``, falling under a pinned block.
    """
    out = []
    for s in scales:
        cells = atlas_cells(replicates, src_width, src_height, s,
                            flow.resolve_block_size(s))
        out.append(storage_bytes_per_hour(cells, fps, n_channels) * corpus_hours)
    return out


def storage_rises_below(scales, storage) -> float | None:
    """The largest scale below which storage stops falling, or None.

    Under a tracked block the curve is flat and this returns None. It fires on
    the rounding tail: once the tracked block is small enough that it no longer
    divides the scaled tile, partial edge cells push the count back up, so the
    user pays MORE storage for less resolution. Marked rather than smoothed,
    because an unexplained rising tail reads as a plotting glitch and a silently
    smoothed one hides a real reason to stop downsampling.
    """
    pairs = sorted(zip(scales, storage), reverse=True)      # descending scale
    worst = None
    for i in range(1, len(pairs)):
        if pairs[i][1] > pairs[i - 1][1] * 1.001:
            worst = pairs[i][0]
    return worst


def sweep_scales(current: float, candidates, *, reference: float = 1.0,
                 limit: int = 5) -> list[float]:
    """Which scales to time, reference first and descending after.

    The reference runs first so a sweep the user stops early is still readable
    against full resolution. ``current`` is always included even when it is not a
    candidate: it is the scale the project would actually run at.
    """
    want = [float(reference)]
    if abs(float(current) - float(reference)) > 1e-9:
        want.append(float(current))
    for c in sorted((float(c) for c in candidates), reverse=True):
        if len(want) >= limit:
            break
        if all(abs(c - v) > 1e-9 for v in want):
            want.append(c)
    return want[:1] + sorted(want[1:], reverse=True)
