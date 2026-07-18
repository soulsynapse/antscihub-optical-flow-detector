"""Run the detector at several working scales and report what it actually found.

``core/cost_model.py`` prices the downsample lever. This module is the other half
of Batch M, and the more important one: it is the mechanism by which Batch K's
*"whether a coarser scale still resolves your behaviour must be demonstrated per
behaviour and per species, never assumed"* stops being an instruction nobody can
act on and becomes something a user executes in a minute. Without it the decision
window argues feasibility (time, storage, a frontier) far more strongly than it
argues sensitivity, and an asymmetric window biases toward downsampling -- exactly
the silent degradation the opt-in default exists to prevent.

What a run is
-------------
For each candidate scale: extract the SAME window at that scale, run the SAME
tuned detector over the SAME region, and report the events. Everything returned
is a raw count or a set of frame indices on a named clip. There is deliberately
no summary score (todo.md Batch M; the withdrawn ``sig_corr`` reading is the
cautionary case -- an aggregate that looked authoritative and did not mean what
it appeared to).

Why the passes run at the PRODUCTION block, not block=1
--------------------------------------------------------
The live surface extracts at ``block_size=1`` whenever the per-pixel cache fits,
so that a Block change is a cheap re-reduce. That is right for tuning and wrong
for measurement: at block=1 no reduction happens, and ``block_reduce`` was
measured at 62% of such a pass against ~15% at block 64. A cost model fitted to
those passes describes a much costlier pass than a production run. These passes
therefore resolve the block the way a batch run would (``resolve_block_size``),
which makes each of them a legitimate :class:`~core.cost_model.PassSample` as
well as an evidence row -- the sweep that answers "do I still catch my events"
is the same sweep that prices the lever honestly, so both are paid for once.

What this measurement can and cannot separate
---------------------------------------------
It answers the operational question exactly: *if I downsample, do the settings I
just tuned still fire on this clip?* It does NOT decompose a lost event into
"the signal is gone" versus "my absolute threshold drifted". Both value band and
count band are absolute, and downsampling averages pixels before differencing, so
per-block band power moves with scale even when the underlying motion is intact.
The count band is comparable across scales only while the block grid is (i.e.
while the block tracks the scale, which is the ``auto`` default); a pinned block
changes the number of cells the count is taken over and the comparison is then
between two different quantities. :attr:`ScaleEvidence.grid` is recorded per row
so the caller can check that rather than assume it, and a caller must surface the
mismatch instead of ranking rows through it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.channel_source import live_channel_source
from core.cost_model import PassSample
from core.detection import detect_channel_region
from core.wavelet import default_freqs


@dataclass(frozen=True)
class ScaleEvidence:
    """One scale's pass: what it cost, and what the detector found in it."""
    scale: float
    block: int
    grid: tuple[int, int]           # atlas block grid the pass produced
    frames: int
    wall: float
    window_start: int
    detected: np.ndarray            # (T,) bool, the detector's gate
    intervals: list[tuple[int, int]]
    spans: dict[str, float] = field(default_factory=dict)
    approximated: bool = False

    @property
    def n_events(self) -> int:
        return len(self.intervals)

    @property
    def n_detected_frames(self) -> int:
        return int(np.count_nonzero(self.detected))

    def pass_sample(self) -> PassSample:
        """This pass as a cost-model sample. Legitimate precisely because the
        block was resolved as production would resolve it (see module docstring)."""
        return PassSample(scale=self.scale, frames=self.frames, wall=self.wall,
                          spans=dict(self.spans))

    def agreement_with(self, ref: "ScaleEvidence") -> tuple[int, int, int]:
        """``(kept, missed, added)`` frames against a reference row.

        Set arithmetic over frame indices, not a score: ``kept`` is reference
        frames this scale also flagged, ``missed`` reference frames it did not,
        ``added`` frames it flagged that the reference did not. ``added`` is not
        a failure by itself -- a coarser field is smoother and can cross a
        threshold the full-resolution one hovered under -- but a row that mostly
        trades missed frames for added ones is detecting something else, which is
        visible here and would be invisible in an event count alone.
        """
        a, b = np.asarray(ref.detected, bool), np.asarray(self.detected, bool)
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
        return (int(np.count_nonzero(a & b)), int(np.count_nonzero(a & ~b)),
                int(np.count_nonzero(~a & b)))


def measure_scale(video_path: str, cfg, replicates: list[dict], *, dims,
                  start: int, n: int, region_index: int, params: dict,
                  progress=None) -> ScaleEvidence:
    """Extract ``[start, start+n)`` at ``cfg``'s scale and run the tuned detector.

    ``cfg`` carries the scale; the block is whatever ``cfg`` resolves it to, which
    for the ``auto`` default tracks the scale and holds the grid fixed in source
    pixels. ``params`` is ``ScalogramExplorer.detection_params()`` so the sweep
    reproduces exactly the settings the preview was tuned to -- a sweep run
    against different bands would answer a question the user did not ask.
    """
    w, h, fps, fc = dims
    cd = live_channel_source(video_path, cfg, replicates, start=start, n=n,
                             width=w, height=h, fps=fps, frame_count=fc,
                             progress=progress)
    res = detect_channel_region(
        cd, region_index, params["channel_attr"],
        freqs=default_freqs(fps), freq_band_hz=params["freq_band_hz"],
        value_band=params["value_band"], count_band=params["count_band"],
        detect_window=params["detect_window"], centered=params["centered"])
    t = (cd.meta.get("timing") or {})
    ny, nx = cd.meta["grid"]
    return ScaleEvidence(
        scale=float(t.get("scale", cfg.preprocess.resolve_downsample(w))),
        block=int(t.get("block", cfg.flow.resolve_block_size(
            cfg.preprocess.resolve_downsample(w)))),
        grid=(int(ny), int(nx)),
        frames=int(t.get("frames", cd.n_frames)),
        wall=float(t.get("wall", 0.0)),
        window_start=int(cd.window_start),
        detected=np.asarray(res.gate, np.float32) > 0.5,
        intervals=res.detected_intervals(),
        spans=dict(t.get("spans") or {}),
        approximated=bool(cd.approximated))


def reference_caveat(ref: ScaleEvidence) -> str | None:
    """Why this reference row cannot support a comparison, or None if it can.

    Found by driving the tool rather than by reasoning about it, and it is the
    worst failure the panel can have. A freshly built explorer has both bands
    wide open, so the gate is on for every frame of the window; every scale then
    reports perfect agreement, and five rows of "lost nothing" appear -- an
    unqualified argument for aggressive downsampling, produced by a detector
    that is not discriminating anything. The all-quiet case is the mirror image:
    nothing to lose, so nothing can be shown to survive.

    Neither is a bug in the sweep, which is why the sweep must say so itself. A
    vacuous comparison that looks like a passing result is worse than no panel,
    because the user came here specifically to be told whether it is safe.
    """
    n = int(np.asarray(ref.detected).size)
    hit = ref.n_detected_frames
    if n == 0:
        return "the reference pass produced no frames"
    if hit == 0:
        return ("the detector found nothing at full resolution, so there is "
                "nothing for a coarser scale to lose — tune it until it fires "
                "on the behaviour, then re-run")
    if hit == n:
        return ("the detector fires on EVERY frame at full resolution, so every "
                "scale will agree with it perfectly and the rows below carry no "
                "information — narrow the value and count bands until the gate "
                "distinguishes your behaviour from the rest of the window, then "
                "re-run")
    return None


def sweep_scales(current: float, candidates, *, reference: float = 1.0,
                 limit: int = 5) -> list[float]:
    """Which scales to run, reference first and descending after.

    The reference must run first: every other row is read against it, and a
    sweep the user stops halfway is then still interpretable. ``current`` is
    always included even when it is not a candidate -- the settings being judged
    were tuned at it, so omitting it would compare the user's actual working
    scale to nothing.
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
