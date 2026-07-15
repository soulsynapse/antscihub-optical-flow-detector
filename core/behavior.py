"""Behavior definitions: a small logical tree over feature ranges, plus the
temporal criteria that turn an instantaneous match into a bout.

The spec is a tree of AND/OR nodes with range leaves rather than a flat AND.
The flat case is what you will use 90% of the time, but the tree is what lets you
express signatures a flat AND cannot:

    wingbeat = AND(
        OR( band_power[18-22 Hz] high,        # fundamental
            band_power[38-44 Hz] high ),      # or its second harmonic
        coherence low,                        # motion cancels within the block
        net_speed low,                        # the animal is not translating
    )

An animal whose fundamental drops out of band on one stroke but whose harmonic
does not is still wingbeating. A flat AND would lose it.

Evaluation is two-stage, and the order is not arbitrary:

  1. Evaluate the tree per frame  -> a raw boolean trace per ROI.
  2. Apply temporal criteria      -> gap-fill, then minimum duration.

Gap-filling BEFORE the duration test is what makes the duration test meaningful:
a 2-second behavior interrupted by two dropped frames is one 2-second bout, not
three short ones that each fail a 0.5 s minimum.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

SPEC_VERSION = 1


# -- the spec tree -----------------------------------------------------------

@dataclass
class RangeLeaf:
    feature: str
    lo: float
    hi: float
    # A disabled leaf keeps its range (so the all-measures panel can show every
    # measure with its set band) but does not participate in the AND. This is how
    # the user scans all measures and toggles the ones that separate the behavior.
    enabled: bool = True

    kind: Literal["leaf"] = "leaf"

    def evaluate(self, series: dict[str, np.ndarray]) -> np.ndarray:
        x = series[self.feature]
        return (x >= self.lo) & (x <= self.hi)

    def features(self) -> set[str]:
        return {self.feature}

    def to_dict(self) -> dict:
        return {"kind": "leaf", "feature": self.feature,
                "lo": float(self.lo), "hi": float(self.hi),
                "enabled": self.enabled}

    def describe(self) -> str:
        return f"{self.feature} in [{self.lo:g}, {self.hi:g}]"


@dataclass
class LogicNode:
    op: Literal["and", "or"]
    children: list = field(default_factory=list)

    kind: Literal["node"] = "node"

    def _active_children(self) -> list:
        return [c for c in self.children
                if not (isinstance(c, RangeLeaf) and not c.enabled)]

    def evaluate(self, series: dict[str, np.ndarray]) -> np.ndarray:
        kids = self._active_children()
        if not kids:
            n = len(next(iter(series.values()))) if series else 0
            return np.ones(n, dtype=bool)
        out = kids[0].evaluate(series)
        for c in kids[1:]:
            out = (out & c.evaluate(series)) if self.op == "and" \
                else (out | c.evaluate(series))
        return out

    def features(self) -> set[str]:
        s: set[str] = set()
        for c in self._active_children():
            s |= c.features()
        return s

    def to_dict(self) -> dict:
        return {"kind": "node", "op": self.op,
                "children": [c.to_dict() for c in self.children]}

    def describe(self, indent: int = 0) -> str:
        pad = "  " * indent
        lines = [f"{pad}{self.op.upper()}"]
        for c in self.children:
            if isinstance(c, LogicNode):
                lines.append(c.describe(indent + 1))
            else:
                lines.append("  " * (indent + 1) + c.describe())
        return "\n".join(lines)


def node_from_dict(d: dict):
    if d.get("kind") == "leaf":
        return RangeLeaf(feature=d["feature"], lo=d["lo"], hi=d["hi"],
                         enabled=d.get("enabled", True))
    return LogicNode(op=d.get("op", "and"),
                     children=[node_from_dict(c) for c in d.get("children", [])])


# -- spatial criteria --------------------------------------------------------

@dataclass
class SpatialCriteria:
    """How many blocks inside a replicate must pass the feature ranges, and how,
    for a frame to count as a detection.

    This is what generalizes the tool beyond "is this box oscillating": with it
    you can say "at least N blocks, touching, must pass" -- e.g. an ant crossing a
    line is a small connected clump of moving blocks, exactly like the minimum
    area + merge distance of the original colour detector.
    """
    # Minimum passing blocks (in a single merged clump) for the frame to fire.
    min_blocks: int = 1
    # Alternatively/additionally, a minimum FRACTION of the replicate's blocks.
    min_fraction: float = 0.0
    # Blocks within this many blocks of each other are treated as one clump
    # before the min_blocks test (morphological close). 0 = touching only.
    merge_distance: int = 0


# -- temporal criteria -------------------------------------------------------

@dataclass
class TemporalCriteria:
    min_duration_s: float = 0.3
    max_gap_s: float = 0.15
    # Smoothing applied to the raw boolean trace before thresholding, in seconds.
    # 0 disables it.
    smooth_s: float = 0.0


def _fill_gaps(trace: np.ndarray, max_gap: int) -> np.ndarray:
    """Close runs of False shorter than max_gap that sit between two True runs."""
    if max_gap <= 0 or trace.size == 0:
        return trace
    out = trace.copy()
    i = 0
    n = trace.size
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        # Only bridge an INTERIOR gap. A leading or trailing run of False is not
        # a dropout, it is the behavior not having started (or having ended), and
        # filling it would invent bouts at the clip boundaries.
        if i > 0 and j < n and (j - i) <= max_gap:
            out[i:j] = True
        i = j
    return out


def _drop_short(trace: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1 or trace.size == 0:
        return trace
    out = trace.copy()
    i = 0
    n = trace.size
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i
        while j < n and out[j]:
            j += 1
        if (j - i) < min_len:
            out[i:j] = False
        i = j
    return out


def apply_temporal(trace: np.ndarray, fps: float,
                   crit: TemporalCriteria) -> np.ndarray:
    """Raw per-frame boolean -> cleaned behavior trace."""
    t = trace.astype(bool)
    if crit.smooth_s > 0:
        win = max(1, int(round(crit.smooth_s * fps)))
        k = np.ones(win, dtype=np.float32) / win
        t = np.convolve(t.astype(np.float32), k, mode="same") >= 0.5
    t = _fill_gaps(t, int(round(crit.max_gap_s * fps)))
    t = _drop_short(t, int(round(crit.min_duration_s * fps)))
    return t


# -- bouts -------------------------------------------------------------------

@dataclass
class Bout:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def trace_to_bouts(trace: np.ndarray, fps: float) -> list[Bout]:
    bouts: list[Bout] = []
    i, n = 0, trace.size
    while i < n:
        if not trace[i]:
            i += 1
            continue
        j = i
        while j < n and trace[j]:
            j += 1
        bouts.append(Bout(start_s=i / fps, end_s=j / fps))
        i = j
    return bouts


# -- the behavior ------------------------------------------------------------

@dataclass
class Behavior:
    name: str
    color: str = "#ff4488"
    spec: LogicNode = field(default_factory=lambda: LogicNode(op="and"))
    criteria: TemporalCriteria = field(default_factory=TemporalCriteria)
    spatial: SpatialCriteria = field(default_factory=SpatialCriteria)
    notes: str = ""
    version: int = SPEC_VERSION

    def features(self) -> set[str]:
        return self.spec.features()

    def evaluate(self, series: dict[str, np.ndarray], fps: float) -> np.ndarray:
        raw = self.spec.evaluate(series)
        return apply_temporal(raw, fps, self.criteria)

    def evaluate_raw(self, series: dict[str, np.ndarray]) -> np.ndarray:
        return self.spec.evaluate(series)

    def constraint_status(self, series: dict[str, np.ndarray],
                          frame_idx: int) -> list[tuple[str, bool, float]]:
        """Which leaves are met at one instant, for the "why isn't this an X?"
        panel. Returns (description, met, actual_value) per leaf."""
        out: list[tuple[str, bool, float]] = []

        def walk(node):
            if isinstance(node, RangeLeaf):
                if not node.enabled:
                    return   # disabled measures are not part of the behavior
                x = series.get(node.feature)
                if x is None or frame_idx >= len(x):
                    out.append((node.describe(), False, float("nan")))
                    return
                val = float(x[frame_idx])
                out.append((node.describe(), node.lo <= val <= node.hi, val))
            else:
                for c in node.children:
                    walk(c)

        walk(self.spec)
        return out

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "name": self.name,
            "color": self.color,
            "notes": self.notes,
            "spec": self.spec.to_dict(),
            "criteria": {
                "min_duration_s": self.criteria.min_duration_s,
                "max_gap_s": self.criteria.max_gap_s,
                "smooth_s": self.criteria.smooth_s,
            },
            "spatial": {
                "min_blocks": self.spatial.min_blocks,
                "min_fraction": self.spatial.min_fraction,
                "merge_distance": self.spatial.merge_distance,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Behavior":
        return cls(
            name=d["name"],
            color=d.get("color", "#ff4488"),
            notes=d.get("notes", ""),
            spec=node_from_dict(d.get("spec", {"kind": "node", "op": "and"})),
            criteria=TemporalCriteria(**d.get("criteria", {})),
            spatial=SpatialCriteria(**d.get("spatial", {})),
            version=d.get("version", SPEC_VERSION),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Behavior":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class BehaviorLibrary:
    """Behaviors on disk, one JSON per behavior so a single "wingbeat.json" can
    move between projects on its own."""

    def __init__(self, directory: str):
        self.dir = directory
        os.makedirs(directory, exist_ok=True)

    def list(self) -> list[str]:
        return sorted(f[:-5] for f in os.listdir(self.dir) if f.endswith(".json"))

    def load(self, name: str) -> Behavior:
        return Behavior.load(os.path.join(self.dir, f"{name}.json"))

    def load_all(self) -> list[Behavior]:
        out = []
        for n in self.list():
            try:
                out.append(self.load(n))
            except Exception:
                continue
        return out

    def save(self, b: Behavior) -> None:
        b.save(os.path.join(self.dir, f"{b.name}.json"))

    def delete(self, name: str) -> None:
        p = os.path.join(self.dir, f"{name}.json")
        if os.path.exists(p):
            os.remove(p)


def default_wingbeat(fps: float, band_feature: str) -> Behavior:
    """The bundled example, built against whatever band the cache actually holds.

    The three constraints are the physical signature of a beating wing, and each
    one is doing distinct work:

      band power HIGH  -- there is energy at the wingbeat frequency.
      coherence LOW    -- the motion cancels within the block, because the wing
                          goes up and comes back down. This is what separates a
                          wingbeat from an animal simply walking fast.
      net_speed LOW    -- the animal is not translating. This rejects a walking
                          ant whose legs also produce periodic flow.

    Thresholds are starting points, not truth. The band-power threshold in
    particular is in (px/s)^2/Hz and therefore scales with resolution and frame
    rate -- it MUST be retuned on the histogram for any new footage, which is
    exactly what Tab 3 is for.
    """
    return Behavior(
        name="wingbeat",
        color="#ffcc33",
        notes=("Bundled example. High band power at the wingbeat frequency, low "
               "angular coherence (the stroke reverses within the block), and low "
               "net flow (the animal is not walking). Retune the band-power "
               "threshold on the histogram for your footage -- it is in "
               "(px/s)^2/Hz and scales with resolution."),
        spec=LogicNode(op="and", children=[
            RangeLeaf(feature=band_feature, lo=50.0, hi=float("inf")),
            RangeLeaf(feature="coherence", lo=0.0, hi=0.5),
            RangeLeaf(feature="net_speed", lo=0.0, hi=15.0),
        ]),
        criteria=TemporalCriteria(min_duration_s=0.3, max_gap_s=0.15),
    )
