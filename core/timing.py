"""Named-span timing for one processing pass, logged as a single line.

Extraction and detection are deep loops with several plausible hot spots (frame
decode, preprocessing, the per-pixel tensor solve, block reduction, the Morlet
transform). Guessing which one dominates is how you end up parallelizing the
cheap part, so this exists to produce the numbers BEFORE any optimization work.

Usage is one ``Timer`` per pass, spans named inside it, one log line at the end::

    tm = Timer("extract_channels_live")
    for frame in frames:
        with tm.span("decode"):
            ...
    tm.log(frames=n)

Design constraints that shaped this:

* Spans are entered tens of thousands of times per pass (per frame, per tile), so
  a span is a preallocated object reused across entries -- no generator, no
  dataclass, no dict churn per entry. Cost is ~2 ``perf_counter`` calls.
* Consequently a span name must NOT nest inside itself; the reused object holds a
  single start time. Distinct names may nest freely (their totals then overlap,
  which the log line makes visible by summing over 100%).
* Timing is on by default -- the point is to see the numbers during normal use --
  and is disabled with ``OFD_TIMING=0`` for benchmark runs that must not pay even
  the counter calls.
"""
from __future__ import annotations

import logging
import os
import sys
import time

LOGGER_NAME = "ofd.timing"


def _logger() -> logging.Logger:
    """Module logger, self-configured to stderr if the app set up no handlers.

    The app has no logging configuration of its own; without this, timing lines
    would vanish under the root logger's default WARNING level.
    """
    log = logging.getLogger(LOGGER_NAME)
    # Level is set independently of the handler check: if anything else attached a
    # handler first, the logger would otherwise stay NOTSET, inherit root's
    # WARNING, and silently drop every timing line.
    if log.level == logging.NOTSET:
        log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
        log.propagate = False   # no double-print if the app configures root later
    return log


def timing_enabled() -> bool:
    return os.environ.get("OFD_TIMING", "1") not in ("0", "false", "False", "")


class _Span:
    """One reusable name slot. Not reentrant -- see the module docstring."""
    __slots__ = ("_totals", "_counts", "_name", "_t0")

    def __init__(self, totals: dict, counts: dict, name: str):
        self._totals, self._counts, self._name = totals, counts, name
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self._t0
        self._totals[self._name] += dt
        self._counts[self._name] += 1
        return False


class _NullSpan:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_SPAN = _NullSpan()


class Timer:
    """Accumulates named spans for one pass; ``log()`` emits one summary line.

    Construct with ``enabled=False`` (or set ``OFD_TIMING=0``) to get an object
    with the same API whose spans and log are no-ops, so call sites never need a
    conditional.
    """
    __slots__ = ("label", "enabled", "totals", "counts", "_spans", "_t0")

    def __init__(self, label: str, enabled: bool | None = None):
        self.label = label
        self.enabled = timing_enabled() if enabled is None else bool(enabled)
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._spans: dict[str, _Span] = {}
        self._t0 = time.perf_counter()

    def span(self, name: str):
        """Context manager accumulating elapsed time under ``name``."""
        if not self.enabled:
            return _NULL_SPAN
        s = self._spans.get(name)
        if s is None:
            self.totals[name] = 0.0
            self.counts[name] = 0
            s = self._spans[name] = _Span(self.totals, self.counts, name)
        return s

    def add(self, name: str, seconds: float) -> None:
        """Record a span measured elsewhere (e.g. inside a callback)."""
        if not self.enabled:
            return
        self.totals[name] = self.totals.get(name, 0.0) + float(seconds)
        self.counts[name] = self.counts.get(name, 0) + 1

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def format(self, **extra) -> str:
        """``label 3.20s [n=900] | tensor 1.80s 56% ...``, spans slowest first.

        Percentages are of wall time, so nested or overlapping span names sum to
        more than 100% -- that is the honest reading, not a bug to normalize away.
        """
        wall = self.elapsed
        parts = [f"{self.label} {wall:.2f}s"]
        if extra:
            parts.append("[" + " ".join(f"{k}={v}" for k, v in extra.items()) + "]")
        head = " ".join(parts)
        if not self.totals:
            return head
        ranked = sorted(self.totals.items(), key=lambda kv: -kv[1])
        pct = (lambda t: 100.0 * t / wall) if wall > 0 else (lambda t: 0.0)
        body = "  ".join(f"{k} {t:.2f}s {pct(t):.0f}%" for k, t in ranked)
        return f"{head} | {body}"

    def log(self, min_seconds: float = 0.0, **extra) -> None:
        """Emit the summary line, unless the pass finished under ``min_seconds``.

        The threshold is for passes that run on every knob drag: a re-tune that
        takes 4 ms is not the bottleneck anyone is hunting, and logging it dozens
        of times a second would bury the lines that matter.
        """
        if self.enabled and self.elapsed >= min_seconds:
            _logger().info(self.format(**extra))
