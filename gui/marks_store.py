"""Per-video ground-truth marks: the labelled-span sidecar.

The corpus that T30/T31/T32 need is written HERE. A mark is a labelled span of a
region -- "region 2 was Flying from 1.5 s to 2.0 s" -- and the file is the
``.marks.json`` sibling of the clip, the same per-video scoping convention as
``state.video_sidecar`` and ``gui/track_store``: open a different clip and you
get a different sidecar, so one clip's labels never ghost onto another.

**The format is the shelved timeline's, on purpose.** ``gui/_shelved/timeline.py``
reads and writes ``{"spans": {rid: [[t0, t1, label], ...]}, "colors": {...}}`` with
spans in SECONDS keyed by region id. Batch R rehomes that widget onto this surface,
and it must load what "Save detections" wrote without a converter, so this file
speaks that dialect verbatim and only ADDS a ``provenance`` block (which the
timeline ignores, being an unknown key).

**Provenance is why these marks can validate a detector rather than merely
describe one.** Each label carries the detector settings that produced its
spans -- channel, frequency band, value/count band, detection window, geometry.
Without it a saved corpus is a set of spans with no record of what fired them, and
T32 (fit a signature per label) cannot know which config's output it is learning
to reproduce. It is keyed by label because one label is one behaviour is one
detector configuration; saving the same label from a second region overwrites the
provenance with that region's config, which is the honest record of "the last
settings used for this behaviour".

Reads are best-effort, as everywhere else here: a corrupt or missing sidecar means
"no marks yet", never a refusal. Writes go through a temp-and-rename so an
interrupted save cannot truncate a corpus that took real curation to build.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

SUFFIX = ".marks.json"

# A small fixed cycle, assigned to new labels in order. Deliberately not random:
# reopening a clip should show Flying in the same colour it had, and colour is
# derived from insertion order, which is stable across sessions because the
# sidecar preserves it.
_PALETTE = ["#4caf50", "#2196f3", "#ff9800", "#e91e63",
            "#9c27b0", "#00bcd4", "#cddc39", "#ff5722"]


def marks_path(video_path: str) -> str | None:
    if not video_path:
        return None
    base, _ = os.path.splitext(video_path)
    return base + SUFFIX


def load_marks(video_path: str) -> dict:
    """The clip's marks doc, or an empty one. Never raises for a bad file."""
    path = marks_path(video_path)
    if not path or not os.path.exists(path):
        return {"spans": {}, "colors": {}, "provenance": {}}
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return {"spans": {}, "colors": {}, "provenance": {}}
    doc.setdefault("spans", {})
    doc.setdefault("colors", {})
    doc.setdefault("provenance", {})
    return doc


def save_marks(video_path: str, doc: dict) -> tuple[bool, str]:
    """Write ``doc`` as the clip's marks sidecar. Returns ``(wrote, note)``."""
    path = marks_path(video_path)
    if not path:
        return False, ""
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        os.replace(tmp, path)
    except OSError as e:
        return False, f"could not save marks: {e}"
    return True, ""


def color_for(doc: dict, label: str) -> str:
    """This label's colour, assigning the next palette entry if it is new.
    Mutates ``doc['colors']`` so the assignment persists on the next save."""
    colors = doc.setdefault("colors", {})
    if label not in colors:
        colors[label] = _PALETTE[len(colors) % len(_PALETTE)]
    return colors[label]


def add_detection_spans(doc: dict, region_index: int, label: str,
                        spans_s: list[tuple[float, float]],
                        provenance: dict | None = None) -> dict:
    """Fold a fresh set of detections into ``doc`` and return it.

    Re-saving the SAME label for the SAME region REPLACES that pairing's spans
    rather than appending -- a second "Save detections" after nudging a knob is
    the corrected answer to the same question, not a second answer to add to the
    first. Spans of other labels in the region, and other regions entirely, are
    untouched. ``spans_s`` are ``(t0, t1)`` in seconds.
    """
    rid = str(int(region_index))
    color_for(doc, label)
    kept = [s for s in doc["spans"].get(rid, []) if s[2] != label]
    kept.extend([float(t0), float(t1), label] for (t0, t1) in spans_s)
    kept.sort(key=lambda s: s[0])
    doc["spans"][rid] = kept
    if provenance is not None:
        doc.setdefault("provenance", {})[label] = {
            **provenance, "saved_utc": datetime.now(timezone.utc).isoformat()}
    return doc


__all__ = ["SUFFIX", "add_detection_spans", "color_for", "load_marks",
           "marks_path", "save_marks"]
