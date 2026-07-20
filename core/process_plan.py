"""How the whole-video pass spends its time: which spans, and in what order.

The commit used to have exactly one shape -- decode frame 0 to the end, then
detect. That is the right shape when you intend to pay for the whole clip, and
the wrong one whenever you do not, which on multi-hour footage is most of the
time. Its failure is not slowness but BIAS: stopping a continuous pass early
leaves you having examined the beginning of the clip and nothing else, so the
detections you have are a sample of the first N minutes rather than a sample of
the video. Every strategy here except ``continuous`` exists to fix that.

Nothing in this module decodes, transforms or detects. It turns a clip length
and a choice into an ordered list of :class:`Segment`, which is what makes the
orders testable without a video and what keeps the worker a plain loop.

**Order is the product, not just the contents.** ``bisect`` returns the same
spans ``uniform`` would for a full budget; what it sells is that any PREFIX of
the list is already spread over the whole clip. Stopping it after four of
sixteen chunks leaves four spans at roughly 0, 1/2, 1/4 and 3/4 of the way in,
so coverage refines progressively and a partial run is still an unbiased look at
the whole video. That is the property the "process a fraction of the clip"
request is really asking for, and it is why the budget is a stopping rule rather
than a different plan.

**Segments are half-open ``[start, stop)`` in absolute frames** and never
overlap. The cone-of-influence padding a transform needs is deliberately NOT
folded in here: it belongs to whoever runs the transform, it depends on the
frequency band rather than on the schedule, and a plan whose segments silently
overlapped would make coverage arithmetic wrong everywhere it is checked.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Strategy ids, and the labels/blurbs the settings dialog renders. Kept together
# so adding one means touching this table and :func:`plan_segments`, and nothing
# in the GUI.
CONTINUOUS = "continuous"
FROM_HERE = "from_here"
BISECT = "bisect"
UNIFORM = "uniform"
GAPS = "gaps"

STRATEGIES = (
    (CONTINUOUS, "Continuous — the whole clip, start to end",
     "One pass from frame 0 to the end. Complete and in order, and the only "
     "strategy that guarantees no gaps. Stopping it early leaves you having "
     "examined the beginning of the clip and nothing else."),
    (FROM_HERE, "From here to the end",
     "Like continuous, but starts at the window position instead of frame 0. "
     "For finishing a clip you have already worked through the front of."),
    (BISECT, "Binary splits — subsample the whole clip, refining",
     "Chunks the clip and visits them in bisecting order: the middle, then the "
     "quarters, then the eighths. Any point you stop at, the part examined is "
     "spread evenly over the whole video rather than bunched at the front — so "
     "a budget of a tenth of the clip gives you a tenth sampled throughout. "
     "This is the one to use to find out what a long video contains."),
    (UNIFORM, "Uniform sample — evenly spaced chunks, in order",
     "The same spans a full binary-split budget covers, but visited front to "
     "back. Use it when you want the sample laid down in time order; use "
     "binary splits when you might stop early."),
    (GAPS, "Fill gaps — only what has not been examined",
     "Processes just the spans the strip shows as uncovered or stale under the "
     "current settings, skipping everything already done. Cheap after a "
     "restored session or a partial run."),
)

# Strategies whose spans are a SAMPLE of the clip rather than all of it. The
# surface warns for these: a clip processed this way has stretches nobody
# looked at, and a detection count from it is not "what the video contains".
SAMPLING = frozenset({BISECT, UNIFORM})

# Strategies the budget applies to. Elsewhere it is meaningless -- a continuous
# pass with a budget is just a shorter continuous pass, which the window start
# already expresses.
BUDGETED = frozenset({BISECT, UNIFORM})

DEFAULT_STRATEGY = CONTINUOUS
DEFAULT_CHUNK_S = 30.0
DEFAULT_BUDGET = 0.10


@dataclass(frozen=True)
class Segment:
    """One contiguous span to process, in absolute frames, half-open."""
    start: int
    stop: int

    @property
    def n(self) -> int:
        return max(0, self.stop - self.start)


def plan_segments(strategy: str, *, n_frames: int, fps: float, cursor: int = 0,
                  chunk_s: float = DEFAULT_CHUNK_S,
                  budget: float = DEFAULT_BUDGET,
                  gaps=None, covered=None, min_frames: int = 2) -> list[Segment]:
    """The ordered spans ``strategy`` would process.

    ``budget`` is a fraction of the clip and applies only to the sampling
    strategies (:data:`BUDGETED`); ``gaps`` is the ``[start, stop)`` list a
    :class:`~core.live_track.WholeVideoTrack` reports and is required by
    ``GAPS``. ``min_frames`` drops spans too short to carry a time series --
    two frames is the floor everything downstream already assumes, and a
    one-frame segment would be decoded, transformed and recorded as examined
    while meaning nothing.

    ``covered`` is an optional per-frame boolean mask of footage already
    examined under the settings now in force. It changes only the sampling
    strategies, and it changes them the way "process 10% more of the clip" has
    to mean once part of the clip is behind you: the budget is spent on chunks
    that still hold unexamined frames, in the same refining order, rather than
    chosen across the whole clip and then cancelled against coverage afterwards.
    That afterward-intersection is what let a re-run of a partly-done sampling
    plan shrink to nothing while most of the clip sat unexamined. With
    ``covered=None`` the sampling plans are unchanged.
    """
    n_frames = max(0, int(n_frames))
    if n_frames < min_frames:
        return []
    if strategy == CONTINUOUS:
        return [Segment(0, n_frames)]
    if strategy == FROM_HERE:
        start = min(max(0, int(cursor)), max(0, n_frames - min_frames))
        return [Segment(start, n_frames)]
    if strategy == GAPS:
        return [Segment(int(a), int(b)) for a, b in (gaps or [])
                if int(b) - int(a) >= min_frames]
    if strategy in (BISECT, UNIFORM):
        chunks = _chunks(n_frames, fps, chunk_s, min_frames)
        keep = _budgeted(len(chunks), budget)
        order = (_bisect_order(len(chunks)) if strategy == BISECT
                 else list(range(len(chunks))))
        if covered is None:
            return [chunks[i] for i in order[:keep]]
        return _budget_uncovered(chunks, order, keep, covered, min_frames)
    raise ValueError(f"unknown strategy {strategy!r}; "
                     f"pick from {[s for s, _, _ in STRATEGIES]}")


def _budget_uncovered(chunks: list[Segment], order: list[int], keep: int,
                      covered, min_frames: int) -> list[Segment]:
    """Spend a chunk budget on the parts of each chunk still unexamined.

    Same visit ``order`` and same budget (``keep`` chunks) as the plain path,
    with two differences: a chunk already fully examined contributes nothing and
    does not count against the budget, and a partly-examined one contributes
    only its uncovered runs. So ``keep`` chunks' worth of NEW footage is planned,
    spread over the clip in refining order, however much of it is already done.
    """
    cov = np.asarray(covered, bool)
    out: list[Segment] = []
    used = 0
    for i in order:
        if used >= keep:
            break
        runs = _uncovered_runs(cov, chunks[i].start, chunks[i].stop, min_frames)
        if runs:
            out.extend(runs)
            used += 1
    return out


def _uncovered_runs(covered: np.ndarray, a: int, b: int,
                    min_frames: int) -> list[Segment]:
    """The ``[start, stop)`` sub-spans of ``[a, b)`` where ``covered`` is False,
    dropping any shorter than ``min_frames``."""
    a, b = max(0, int(a)), min(len(covered), int(b))
    if b <= a:
        return []
    free = ~np.asarray(covered[a:b], bool)
    if not free.any():
        return []
    edges = np.diff(np.concatenate([[0], free.view(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1) + a
    stops = np.flatnonzero(edges == -1) + a
    return [Segment(int(s), int(e)) for s, e in zip(starts, stops)
            if int(e) - int(s) >= min_frames]


def coverage_note(strategy: str, segments: list[Segment], n_frames: int,
                  fps: float) -> str:
    """A one-line honest description of what a plan will and will not examine.

    Written here rather than in the dialog because it is the same sentence the
    status line needs after the pass, and because the thing it has to get right
    -- that a sampling pass leaves footage unexamined -- is a property of the
    plan, not of the widget that launched it.
    """
    if not segments:
        return "nothing to process"
    total = sum(s.n for s in segments)
    frac = 100.0 * total / max(1, n_frames)
    mins = total / max(fps, 1e-6) / 60.0
    body = (f"{len(segments)} span{'s' if len(segments) != 1 else ''}, "
            f"{mins:.1f} min of footage ({frac:.0f}% of the clip)")
    if strategy in SAMPLING:
        return (f"{body} — spread across the whole video; the other "
                f"{100 - frac:.0f}% will NOT be examined")
    return body


def _chunks(n_frames: int, fps: float, chunk_s: float,
            min_frames: int) -> list[Segment]:
    """The clip cut into equal spans, the last one short. Contiguous and
    exhaustive, so a full-budget sample is a full pass."""
    size = max(min_frames, int(round(float(chunk_s) * max(fps, 1e-6))))
    out = [Segment(a, min(a + size, n_frames))
           for a in range(0, n_frames, size)]
    # A trailing sliver shorter than a time series is folded into its
    # predecessor rather than dropped -- dropping it would leave the end of
    # every clip permanently unexaminable by this route.
    if len(out) > 1 and out[-1].n < min_frames:
        last = out.pop()
        out[-1] = Segment(out[-1].start, last.stop)
    return [s for s in out if s.n >= min_frames]


def _budgeted(k: int, budget: float) -> int:
    """How many of ``k`` chunks a budget buys. At least one whenever there is
    anything at all -- a budget that rounds to zero spans is a pass that runs,
    reports success and examines nothing."""
    if k <= 0:
        return 0
    frac = float(budget)
    if not (frac > 0.0):
        return k                     # 0 or negative reads as "no budget set"
    return max(1, min(k, int(round(frac * k))))


def _bisect_order(k: int) -> list[int]:
    """Chunk indices in progressively-refining order: 0, k/2, k/4, 3k/4, ...

    This is the base-2 van der Corput (radical inverse) sequence scaled to
    ``k``, which is the standard construction for a low-discrepancy ordering --
    every prefix is close to evenly spread, which is precisely the property that
    makes stopping early unbiased. Collisions (two ``j`` mapping to the same
    chunk) are skipped, and anything the sequence has not reached by the time
    every index is accounted for is appended, so the result is always a
    permutation of ``range(k)`` however the rounding falls.
    """
    if k <= 0:
        return []
    seen: set[int] = set()
    order: list[int] = []
    j = 0
    # Bounded rather than "until done": the sequence does cover every index, but
    # tying a loop's termination to that argument means a rounding change turns
    # into a hang. The tail append below makes the bound safe to be wrong.
    limit = 4 * k + 64
    while len(order) < k and j < limit:
        idx = int(_radical_inverse_2(j) * k)
        if idx < k and idx not in seen:
            seen.add(idx)
            order.append(idx)
        j += 1
    order.extend(i for i in range(k) if i not in seen)
    return order


def _radical_inverse_2(i: int) -> float:
    """``i`` written in binary, reflected about the point. 0, .5, .25, .75, ..."""
    f, r = 0.5, 0.0
    while i > 0:
        r += f * (i & 1)
        i >>= 1
        f *= 0.5
    return r
