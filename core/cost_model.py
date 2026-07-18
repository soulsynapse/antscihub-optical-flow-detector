"""What the downsample and block-size levers actually cost, from measured spans.

Batch K made downsampling opt-in and off by default, which is right but leaves a
user with a knob and no way to price it. The predictable response to a bare
"downsample?" control is *"I don't want my data to be worse, so I won't touch
it"* -- an avoidance, not a decision, and one that quietly makes large projects
infeasible for no examined reason. This module is the measured half of the answer
(see ``gui/downsample_dialog.py`` for the presented half): given timing spans from
passes that actually ran, it projects wall time and storage for a corpus at any
scale, and locates the knee beyond which resolution is given up for nothing.

Nothing here is fitted to a clip and then generalized. Every number a caller gets
back is either arithmetic over known geometry (storage) or an extrapolation from
spans this machine measured on this footage (time), and a model built from a
single pass reports ``provisional=True`` so the UI can say so.

The cost structure, which is what makes a knee exist at all
-----------------------------------------------------------
H.264 decode is whole-frame, and the ROI crop/scale are post-decode filter
stages, so decode costs the same at every scale -- measured at ~3.8 s per 479
frames of 5312x2988 regardless of crop (todo.md, ROI decode batch). Everything
downstream is per-pixel and therefore scales with the pixel count, i.e. with
``scale ** 2``. So::

    seconds_per_frame(s) = F + M * s**2

with ``F`` a fixed decode floor and ``M`` the per-pixel work at scale 1.0. The
whole shape of the decision follows from ``F`` being irreducible by this lever.

Why the knee is where the math cost equals the decode floor
-----------------------------------------------------------
"Knee" is usually eyeballed, which is exactly the kind of authoritative-looking
number todo.md's Batch M warns against. Here it has a closed form, because the
question a user is actually asking is an elasticity: *does 1% less resolution buy
me at least 1% less time?* With ``t(s) = F + M s**2``::

    E(s) = (dt/ds)(s/t) = 2 M s**2 / (F + M s**2)

``E(s) = 1`` exactly when ``M s**2 == F``, giving ``s* = sqrt(F / M)``. Above the
knee a unit of resolution buys more than a unit of time; below it the per-pixel
work has already shrunk under the decode floor and further downsampling shaves a
shrinking remainder of a fixed cost. That matches the measured behaviour the
sweep found and could not explain at the time: 1300 -> 650 px is a 4x pixel cut
for ~13% wall time.

A single pass cannot locate the knee, and this is measured, not cautionary
--------------------------------------------------------------------------
``core.video.prefetch`` runs decode on its own thread, so the ``decode`` span
measures *waiting for a frame*, not decode cost. When the overlap works it reads
essentially zero -- and then a one-sample split sees no floor at all. Measured on
``GX010047c2`` (7 replicates, 120 frames, scale 1.0), the ``decode`` span was
0.01 s of a 4.30 s pass (0%), giving::

    one pass:  floor  0.04 ms/frame   per-pixel 35.77 ms/frame   knee 0.05
    fitted(3): floor 11.79 ms/frame   per-pixel 24.09 ms/frame   knee 0.70

The one-pass model is not slightly biased, it is wrong by ~300x on the floor, and
wrong in the direction that tells a user aggressive downsampling is free. Its
curve is no better: it under-predicts the cost at scale 0.25 by ~5.7x. So a
provisional model must NOT be presented as a frontier and :meth:`knee_scale`
returns ``None`` for one. Fit over passes at two or more distinct scales instead
-- against the same clip the fitted form reproduces measured wall time to within
±2% across a 4x scale range, which is what justifies the quadratic at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.replicates import build_layout

# Spans whose cost does not move with the working scale. ``decode`` is here
# because H.264 decode is whole-frame and the ROI crop/scale run after it;
# ``progress_cb`` re-enters the GUI thread and is unrelated to pixel count.
FIXED_SPANS = frozenset({"decode", "progress_cb"})

# Spans that run over working pixels, hence scale with ``scale ** 2``.
# ``block_reduce`` belongs here despite writing a scale-invariant grid: it is the
# input pixels it reads that scale, not the cells it writes.
PER_PIXEL_SPANS = frozenset({
    "preprocess", "tensor_products", "tensor_blur", "flow_solve",
    "appearance", "texture",
})

BYTES_PER_CELL = 4          # float32 channel planes


@dataclass(frozen=True)
class PassSample:
    """One measured extraction pass, as the input to a fit.

    ``spans`` is a ``Timer.totals`` dict. ``wall`` is preferred over the span sum
    when present, because spans overlap and un-timed work exists between them.
    """
    scale: float
    frames: int
    wall: float
    spans: dict[str, float] = field(default_factory=dict)

    @property
    def seconds_per_frame(self) -> float:
        return self.wall / self.frames if self.frames > 0 else 0.0


@dataclass(frozen=True)
class CostModel:
    """``seconds_per_frame(s) = fixed_s + per_pixel_s * s**2``, per replicate set.

    Both coefficients are per frame and specific to the machine, footage and
    replicate layout they were measured on. They are not portable constants and
    must not be cached across videos.
    """
    fixed_s: float              # decode floor, per frame
    per_pixel_s: float          # per-pixel work per frame at scale 1.0
    provisional: bool           # True when inferred from one pass by span class
    n_samples: int = 1

    # -- construction --------------------------------------------------------
    @classmethod
    def from_spans(cls, spans: dict[str, float], frames: int,
                   scale: float, wall: float) -> "CostModel":
        """Split ONE pass into fixed and per-pixel halves by span name.

        Used to populate the dialog immediately, before any measured sweep has
        run. Unclassified spans (and the un-timed remainder between spans) are
        attributed to the per-pixel half: that is the conservative choice here,
        since over-attributing to per-pixel raises the knee and so errs toward
        keeping resolution. See the module docstring on why ``fixed_s`` from a
        single prefetched pass is a lower bound.
        """
        if frames <= 0 or scale <= 0:
            return cls(fixed_s=0.0, per_pixel_s=0.0, provisional=True)
        fixed = sum(t for k, t in spans.items() if k in FIXED_SPANS)
        fixed_pf = fixed / frames
        # Take the residual from wall rather than summing the per-pixel spans, so
        # work that no span covers is priced instead of vanishing.
        rest_pf = max(0.0, wall / frames - fixed_pf)
        return cls(fixed_s=fixed_pf,
                   per_pixel_s=rest_pf / (scale ** 2),
                   provisional=True)

    @classmethod
    def fit(cls, samples: list[PassSample]) -> "CostModel":
        """Least-squares fit of both coefficients over passes at ≥2 scales.

        This is the model to prefer: it needs no span classification and no
        assumption about what prefetch hid, because it reads the two coefficients
        off how wall time actually moved with scale. Falls back to
        :meth:`from_spans` when the samples do not span two distinct scales,
        since the fit is then underdetermined.
        """
        usable = [s for s in samples if s.frames > 0 and s.scale > 0]
        scales = {round(s.scale, 6) for s in usable}
        if len(usable) < 2 or len(scales) < 2:
            if not usable:
                return cls(0.0, 0.0, provisional=True)
            s = max(usable, key=lambda p: p.scale)
            return cls.from_spans(s.spans, s.frames, s.scale, s.wall)

        # Regress y = a + b*x with x = scale**2, y = seconds per frame.
        xs = [s.scale ** 2 for s in usable]
        ys = [s.seconds_per_frame for s in usable]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        b = sxy / sxx if sxx > 0 else 0.0
        a = my - b * mx
        # Neither coefficient can be negative physically; noise across a narrow
        # scale range can still drive one there, and a negative floor would put
        # the knee at a NaN. Clamp and re-derive the other from the mean point so
        # the model still reproduces the data it was given.
        if b < 0:
            b, a = 0.0, my
        if a < 0:
            a = 0.0
            b = my / mx if mx > 0 else 0.0
        return cls(fixed_s=a, per_pixel_s=b, provisional=False, n_samples=n)

    # -- projection ----------------------------------------------------------
    def seconds_per_frame(self, scale: float) -> float:
        return self.fixed_s + self.per_pixel_s * (float(scale) ** 2)

    def hours_for_corpus(self, scale: float, corpus_hours: float,
                         fps: float, workers: int = 1) -> float:
        """Wall-clock hours to process ``corpus_hours`` of footage at ``scale``.

        ``workers`` divides linearly, which holds only where the work is
        math-bound. At scale 1.0 it is (~80% math, todo.md Batch K), but a
        heavily downsampled run is decode-bound and measured flat past ~2-4
        workers, so a caller passing ``workers > 1`` should not present the low
        end of the curve as reachable. The dialog projects at ``workers=1``.
        """
        frames = corpus_hours * 3600.0 * fps
        return self.seconds_per_frame(scale) * frames / 3600.0 / max(1, workers)

    def realtime_factor(self, scale: float, fps: float) -> float:
        """Footage seconds processed per wall second. <1 is slower than realtime."""
        spf = self.seconds_per_frame(scale)
        return (1.0 / (spf * fps)) if spf > 0 and fps > 0 else float("inf")

    def knee_scale(self, min_scale: float = 0.05) -> float | None:
        """Scale where per-pixel work equals the decode floor; None if undefined.

        Returns None when:

        * the model is ``provisional`` -- a one-pass split cannot see the decode
          floor behind prefetch at all, and the knee it computes is off by
          ~300x in the direction that makes downsampling look free (see the
          module docstring for the measurement). Refusing to answer is the only
          safe behaviour, since the caller cannot tell a bad knee from a good one;
        * the floor is zero, so there is nothing for the curve to flatten against;
        * the knee falls at or above 1.0, meaning decode already dominates at
          full resolution. Then *every* scale is on the flat part and the honest
          readout is "this lever buys almost nothing here", not a marker.
        """
        if self.provisional or self.fixed_s <= 0 or self.per_pixel_s <= 0:
            return None
        s = math.sqrt(self.fixed_s / self.per_pixel_s)
        if s >= 1.0:
            return None
        return max(min_scale, s)

    def elasticity(self, scale: float) -> float:
        """d(log t)/d(log s). >1: resolution is cheap here. <1: past the knee."""
        t = self.seconds_per_frame(scale)
        if t <= 0:
            return 0.0
        return 2.0 * self.per_pixel_s * (scale ** 2) / t


# -- storage ----------------------------------------------------------------
def boxes_from_tiles(tiles) -> list[tuple[int, int]]:
    """``(width, height)`` in source pixels from tile dicts or ``ReplicateTile``s.

    The two extraction paths carry tiles differently -- ``core/tensor_channels.py``
    builds plain dicts, ``core/replicates.build_layout`` returns dataclasses -- so
    the conversion lives here rather than forcing the cost model to know either.
    """
    out = []
    for t in tiles:
        box = t["source_box"] if isinstance(t, dict) else t.source_box
        x0, y0, x1, y1 = box
        out.append((x1 - x0, y1 - y0))
    return out


def grid_cells(boxes: list[tuple[int, int]], scale: float, block: int) -> int:
    """Block-grid cells over every replicate box at a given scale and block.

    ``boxes`` are ``(width, height)`` in SOURCE pixels. Matches
    ``core/replicates.build_layout``: scale first, then ceil into blocks, so
    partial edge blocks are retained exactly as extraction retains them. This is
    arithmetic over known geometry, not an estimate.
    """
    total = 0
    for w, h in boxes:
        ww = max(1, int(round(w * scale)))
        wh = max(1, int(round(h * scale)))
        total += math.ceil(wh / block) * math.ceil(ww / block)
    return total


def atlas_cells(replicates: list[dict], src_width: int, src_height: int,
                scale: float, block: int) -> int:
    """Block cells actually ALLOCATED per frame, via the real layout.

    Not the same as :func:`grid_cells` over the boxes: channels are stored as
    ``(T, ny, nx)`` over the packed atlas, which pads every tile out to the
    widest one and inserts separator rows. On a 7-replicate 5312x2988 clip that
    is 205 cells against 175 summed over the tiles -- a ~17% under-count if the
    boxes are used for a storage figure. Storage projections must use this.
    """
    if not replicates:
        w = max(1, int(round(src_width * scale)))
        h = max(1, int(round(src_height * scale)))
        return math.ceil(h / block) * math.ceil(w / block)
    return build_layout(replicates, src_width, src_height, scale,
                        block).block_cells


def storage_bytes_per_hour(cells: int, fps: float, n_channels: int) -> float:
    """Cache bytes per hour of footage, from an allocated cell count.

    Independent of ``scale`` when the block tracks it, which is the measured
    claim behind treating these as two levers: ``block_size`` is close to a pure
    storage knob and ``downsample`` a pure compute one.
    """
    return cells * n_channels * BYTES_PER_CELL * fps * 3600.0


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def format_duration(hours: float) -> str:
    """Wall-clock durations here span minutes to months; pick a readable unit."""
    if hours < 1.0 / 60:
        return f"{hours * 3600:.0f} s"
    if hours < 1.0:
        return f"{hours * 60:.0f} min"
    if hours < 48.0:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} d"


def working_px_per_body_length(pixels_per_mm: float | None,
                               body_length_mm: float | None,
                               scale: float) -> float | None:
    """Organism-relative resolution at a given scale, or None if uncalibrated.

    Follows the convention already established at ``core/roi.py:418`` --
    calibration is stored in SOURCE px/mm and the working value is derived by
    multiplying through the scale -- so this reads the same fields exports and
    ``speed_body_lengths_s`` already read rather than inventing new ones.
    """
    if not pixels_per_mm or not body_length_mm:
        return None
    return float(pixels_per_mm) * float(body_length_mm) * float(scale)
