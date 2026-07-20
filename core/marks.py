"""Manual ground-truth spans: the corpus the detector is judged against.

A mark says "between these two frames, this replicate was doing this". That is
the only thing in this pipeline a human asserts directly, and everything
supervised downstream -- validating occupancy and the ratio channels (todo.md
T31), fitting per-behaviour signatures (T32) -- consumes it as label truth. So
this module is deliberately stricter than a drawing tool needs to be.

**Marks are keyed by replicate ID, not by region index.** The index is
positional: delete replicate 2 and every region after it shifts down one, so
index-keyed marks would silently re-attribute one animal's behaviour to
another. Nothing would error and the corpus would just be wrong. IDs come from
the replicate sidecar and survive reordering. ``None`` is a real key -- the
whole-frame region, which is what a clip with no replicates drawn presents.

**Frames, not seconds.** The whole surface is frame-indexed (``seek_absolute``,
the track arrays, the strip's ``_frame_at``), and a span stored in seconds has
to be multiplied by an fps to be used, which is one more place for a wrong fps
to land silently. ``fps`` is recorded alongside for display and export only.

**Three invariants, all about not handing T32 contradictory rows:**

1. *No frame carries two labels for one replicate.* A new span CARVES ITSELF
   OUT of every other label on that replicate rather than overlapping it. This
   is how you correct a mislabel -- re-mark the stretch and the old label
   yields. Retaining both would put one frame in two training classes, which no
   consumer would notice and every consumer would be poisoned by.
2. *Same-label spans that touch or overlap MERGE.* Otherwise laying a span down
   in two drags produces two bouts where there was one, and bout counts and
   durations -- which is what an ethogram is read for -- come out wrong.
3. *Labels are normalized on entry.* ``marks.json`` in this repo holds
   ``"Flying"``, ``"Flying "`` and ``"Flying  "`` as three distinct labels
   because the old picker was a free-text combo box. Whitespace is stripped and
   collapsed here so that cannot recur; the *picker* additionally refuses a
   case-insensitive near-duplicate, which is policy and lives in the GUI.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

# Fallback colours for labels, assigned by a STABLE hash of the name.
#
# Not ``hash()``: Python salts string hashing per process (PYTHONHASHSEED), so
# the shelved tab's ``abs(hash(label)) % len(palette)`` handed the same label a
# different colour on every launch. crc32 is stable across runs and machines,
# which is what "this behaviour is always blue" requires.
PALETTE = ["#ff4488", "#4ac6ff", "#ffd24a", "#6ee06e", "#c78bff",
           "#ff9d3a", "#00d0b0", "#ff6d6d"]

_WS = re.compile(r"\s+")


def normalize_label(label: str) -> str:
    """Collapse a typed label to its canonical form; ``""`` if it is not one."""
    return _WS.sub(" ", str(label or "").strip())


def label_color(label: str) -> str:
    """A stable colour for a label, identical across runs and machines."""
    return PALETTE[zlib.crc32(normalize_label(label).encode()) % len(PALETTE)]


@dataclass(frozen=True)
class Mark:
    """One labelled span, ``[start, end)`` in absolute video frames."""
    start: int
    end: int
    label: str

    def __post_init__(self):
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "end", int(self.end))
        object.__setattr__(self, "label", normalize_label(self.label))

    @property
    def n_frames(self) -> int:
        return max(0, self.end - self.start)

    def duration_s(self, fps: float) -> float:
        return self.n_frames / max(float(fps), 1e-6)

    def contains(self, frame: int) -> bool:
        return self.start <= int(frame) < self.end


class MarkSet:
    """Every mark for one clip, grouped by replicate id.

    Not a plain dict because the invariants above have to hold after every
    edit, and a caller appending to a list would bypass all three.
    """

    def __init__(self, n_frames: int = 0, fps: float = 30.0):
        self.n_frames = int(n_frames)
        self.fps = float(fps)
        self._by_rep: dict[int | None, list[Mark]] = {}

    # -- reading -------------------------------------------------------------
    def marks_for(self, replicate_id: int | None) -> list[Mark]:
        """This replicate's marks, ordered by start. A copy: callers iterate
        these while editing, and mutating the live list mid-walk is how the
        carve-out below would corrupt itself."""
        return list(self._by_rep.get(replicate_id, ()))

    def replicates(self) -> list[int | None]:
        return [r for r in self._by_rep if self._by_rep[r]]

    def labels(self) -> list[str]:
        """Every label in use anywhere in this clip, sorted. This is what the
        picker offers, so it is clip-scoped rather than replicate-scoped: a
        label you used on replicate 1 must be reachable on replicate 2 without
        retyping it (retyping is what forked "Flying" three ways)."""
        return sorted({m.label for ms in self._by_rep.values() for m in ms})

    def at(self, replicate_id: int | None, frame: int) -> Mark | None:
        for m in self._by_rep.get(replicate_id, ()):
            if m.contains(frame):
                return m
        return None

    def is_empty(self) -> bool:
        return not any(self._by_rep.values())

    def total_frames(self, label: str | None = None) -> int:
        lab = normalize_label(label) if label is not None else None
        return sum(m.n_frames for ms in self._by_rep.values() for m in ms
                   if lab is None or m.label == lab)

    def summary(self) -> dict[str, tuple[int, int]]:
        """``{label: (n_spans, n_frames)}`` across the whole clip.

        The corpus readout. T30's complaint is that ``marks.json`` holds one
        span and no animal-absent spans at all, so what a labelling session
        needs on screen is not "you have marks" but how many of each, which is
        the difference between a corpus and a demonstration.
        """
        out: dict[str, list[int]] = {}
        for ms in self._by_rep.values():
            for m in ms:
                row = out.setdefault(m.label, [0, 0])
                row[0] += 1
                row[1] += m.n_frames
        return {k: (v[0], v[1]) for k, v in sorted(out.items())}

    # -- editing -------------------------------------------------------------
    def add(self, replicate_id: int | None, start: int, end: int,
            label: str) -> Mark | None:
        """Lay a span down, enforcing all three invariants. Returns the mark as
        it ended up (merged with its neighbours), or None if it was not a span.

        Order matters: carve the OTHER labels first, then merge same-label
        neighbours. Merging first would grow the new span and then carve a
        larger hole than the user actually drew.
        """
        lab = normalize_label(label)
        lo, hi = int(min(start, end)), int(max(start, end))
        lo = max(0, lo)
        hi = min(hi, self.n_frames) if self.n_frames else hi
        if not lab or hi <= lo:
            return None

        kept: list[Mark] = []
        for m in self._by_rep.get(replicate_id, ()):
            if m.label == lab:
                kept.append(m)                    # merged in the next pass
                continue
            kept.extend(_subtract(m, lo, hi))     # invariant 1
        merged = Mark(lo, hi, lab)
        rest: list[Mark] = []
        for m in kept:
            if m.label == lab and m.start <= merged.end and merged.start <= m.end:
                merged = Mark(min(merged.start, m.start),      # invariant 2
                              max(merged.end, m.end), lab)
            else:
                rest.append(m)
        rest.append(merged)
        self._by_rep[replicate_id] = sorted(rest, key=lambda m: m.start)
        return merged

    def remove_at(self, replicate_id: int | None, frame: int) -> Mark | None:
        """Delete whichever span covers ``frame``. Returns it, or None."""
        hit = self.at(replicate_id, frame)
        if hit is None:
            return None
        self._by_rep[replicate_id] = [
            m for m in self._by_rep[replicate_id] if m is not hit]
        return hit

    def clear(self, replicate_id: int | None = ..., label: str | None = None):
        """Drop marks. Default clears everything; pass a replicate and/or a
        label to narrow it. ``...`` rather than ``None`` as the "all
        replicates" sentinel because None IS a replicate key here."""
        reps = (list(self._by_rep) if replicate_id is ...
                else [replicate_id])
        lab = normalize_label(label) if label else None
        for r in reps:
            if r not in self._by_rep:
                continue
            self._by_rep[r] = ([m for m in self._by_rep[r] if m.label != lab]
                               if lab else [])

    def rename_label(self, old: str, new: str) -> int:
        """Merge one label into another everywhere. Returns spans touched.

        This is the repair path for a corpus that already forked -- the five
        labels in ``marks.json`` are two behaviours -- and it re-runs the merge
        invariant, because two spans that were adjacent under different names
        become one bout under the same name.
        """
        o, n = normalize_label(old), normalize_label(new)
        if not o or not n or o == n:
            return 0
        touched = 0
        for rep, ms in list(self._by_rep.items()):
            hits = [m for m in ms if m.label == o]
            if not hits:
                continue
            touched += len(hits)
            self._by_rep[rep] = [m for m in ms if m.label != o]
            for m in hits:
                self.add(rep, m.start, m.end, n)
        return touched

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "n_frames": int(self.n_frames),
            "fps": float(self.fps),
            # Keys are strings because JSON object keys always are; the
            # whole-frame region's None becomes "" so it round-trips distinctly
            # from replicate 0, which is a real and different thing.
            "spans": {("" if rep is None else str(rep)):
                      [[m.start, m.end, m.label] for m in ms]
                      for rep, ms in sorted(
                          self._by_rep.items(),
                          key=lambda kv: (kv[0] is not None, kv[0]))
                      if ms},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MarkSet":
        d = d or {}
        ms = cls(int(d.get("n_frames", 0) or 0),
                 float(d.get("fps", 30.0) or 30.0))
        for rep, spans in (d.get("spans") or {}).items():
            key: int | None
            if rep in ("", None):
                key = None
            else:
                try:
                    key = int(rep)
                except (TypeError, ValueError):
                    continue
            for s in spans or ():
                try:
                    ms.add(key, int(s[0]), int(s[1]), str(s[2]))
                except (TypeError, ValueError, IndexError):
                    continue        # one bad span must not lose the file
        return ms


# -- persistence ------------------------------------------------------------
#
# Marks live next to the video and are scoped to it -- ``.../Foo.mp4`` ->
# ``.../Foo.marks.json`` -- the same convention ROIs, replicates and tuning
# already use (AppState.video_sidecar, core.framecount, gui.tuning_store). That
# scoping is the one behaviour the retired tab got right and the easiest to
# lose: opening a different clip must load ITS marks or none, never inherit the
# previous clip's. Keyed on path rather than content hash for the reasons stated
# in video_sidecar -- a human can see which JSON belongs to which clip.
#
# Unlike tuning, these are DATA. A tuning sidecar that fails to load costs one
# re-drag; a marks sidecar that fails to load costs hours of annotation. So the
# failure policy is the opposite: read errors are REPORTED, not swallowed, and a
# write goes through a temp file and a replace so an interrupted save cannot
# truncate the previous session's corpus.

SUFFIX = ".marks.json"

# Bumped when a stored payload can no longer be read as it was written.
#
# Version 1 is the FIRST versioned format. The retired behaviour tab wrote this
# same filename with no version at all, spans in SECONDS, and keys that were ROI
# ids -- see load_marks, which refuses those rather than guessing. Reading a
# seconds-based span as frames would turn a 4.5 s mark into frame 4: a silently
# wrong corpus, which is worse than no corpus.
VERSION = 1


class MarksFormatError(Exception):
    """A marks sidecar exists but cannot be read as this format. Carries the
    path so the caller can tell the user which file was left alone."""

    def __init__(self, path: str, why: str):
        super().__init__(f"{path}: {why}")
        self.path = path
        self.why = why


def marks_path(video_path: str) -> str:
    import os
    base, _ = os.path.splitext(video_path)
    return base + SUFFIX


def load_marks(video_path: str, n_frames: int = 0,
               fps: float = 30.0) -> MarkSet:
    """This clip's marks, or an empty set when it has none.

    Raises ``MarksFormatError`` when a file is present but not readable as this
    format -- including the retired tab's unversioned seconds-based layout. The
    file is left untouched, so a user who has one can convert it deliberately;
    silently starting empty would look identical to "this clip is unannotated"
    and the next save would overwrite the very thing that was misread.
    """
    import json
    import os
    path = marks_path(video_path)
    if not video_path or not os.path.exists(path):
        return MarkSet(n_frames, fps)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise MarksFormatError(path, f"could not be read ({e})") from e
    if not isinstance(data, dict):
        raise MarksFormatError(path, "is not a marks file")
    ver = data.get("version")
    if ver != VERSION:
        raise MarksFormatError(
            path,
            f"was written in an older format (version {ver!r}, this build "
            f"reads {VERSION}). Its spans are in seconds and keyed by ROI id, "
            f"which cannot be converted to frames and replicate ids without "
            f"guessing. It has been left alone.")
    ms = MarkSet.from_dict(data)
    # The clip is the authority on its own length and rate; a sidecar copied
    # beside a re-encode would otherwise keep asserting the old ones.
    if n_frames:
        ms.n_frames = int(n_frames)
    if fps:
        ms.fps = float(fps)
    return ms


def save_marks(video_path: str, marks: MarkSet) -> None:
    """Write this clip's marks. Raises on failure -- deliberately.

    ``save_tuning`` swallows everything because it runs off a keystroke
    debounce and the loss is a remembered window. This is annotation: a user
    who has marked twenty spans onto a read-only volume needs to find out now,
    not when they reopen the clip.
    """
    import json
    import os
    if not video_path:
        return
    path = marks_path(video_path)
    blob = json.dumps({**marks.to_dict(), "version": VERSION}, indent=2)
    # Encode first, then write to a sibling temp and replace: an interrupted
    # write must not leave a truncated file where the corpus was, because a
    # truncated file is indistinguishable from a corrupt one and would take the
    # previous session down with it.
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(blob)
    os.replace(tmp, path)


def _subtract(m: Mark, lo: int, hi: int) -> list[Mark]:
    """``m`` with ``[lo, hi)`` removed: 0, 1 or 2 pieces."""
    if m.end <= lo or hi <= m.start:
        return [m]                                  # disjoint
    out = []
    if m.start < lo:
        out.append(Mark(m.start, lo, m.label))
    if hi < m.end:
        out.append(Mark(hi, m.end, m.label))
    return out                                      # fully covered -> []


__all__ = ["Mark", "MarkSet", "MarksFormatError", "PALETTE", "SUFFIX",
           "VERSION", "label_color", "load_marks", "marks_path",
           "normalize_label", "save_marks"]
