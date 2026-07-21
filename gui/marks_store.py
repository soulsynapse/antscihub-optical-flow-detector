"""Per-replicate ground-truth marks: the labelled-span sidecar.

The corpus that T30/T31/T32 need is written HERE. A mark is a labelled span of a
region -- "region 2 was Flying from 1.5 s to 2.0 s".

**Marks belong to a REPLICATE, for ``gui/track_store``'s reason.** A span is one
region's answer, so it lives in that replicate's home (``core.replicate_home``):
``.../Foo.mp4`` replicate 2 -> ``.../Foo_rep02/Foo.marks.json``. Until Batch S
slice 3 every region wrote into one ``Foo.marks.json``, which cost less than the
track collision did (spans are keyed by region, so they did not overwrite each
other) but cost something real anyway: ``provenance`` is keyed by LABEL alone, so
saving "flying" from region 3 replaced region 2's detector configuration with
region 3's. The old docstring called that "the honest record of the last settings
used for this behaviour"; it was honest only because there was one file to be
honest in. Per-replicate provenance dissolves the caveat rather than changing it.

**Legacy marks are read through, never migrated.** A home with no marks file
falls back to the per-video sidecar, FILTERED to its own region index -- filter,
never reshape. Two reasons, and the second is the load-bearing one:

* The 152-bout rep3 corpus (T30) lives in a per-video file that took real
  curation. Reading it where it lies cannot damage it; a migration can.
* Unlike a track, a mark carries **no stamp naming its region**. ``load_track``
  can adopt a legacy sidecar because the file says which region it describes; a
  marks file says only "region index 2", and an index is a position in
  ``core.replicates.in_tile_order`` that coincides with a replicate id only while
  ids run 1..N with nothing deleted. So a real migration would have to *infer*
  the mapping it is rewriting the file under, which is the inference slice 2
  exists to forbid. Read-through leaves the inference reversible.

The legacy file goes inert for a region the moment that region's home has marks
of its own -- the same "next save writes the home copy" rule ``load_track`` uses.

**``colors`` stays per-video, and stays in that same legacy file.** The palette is
assigned by insertion order so that reopening a clip shows Flying in the colour it
had, and one behaviour label should be one colour across every replicate in the
clip -- that is a per-CLIP display contract, not a per-replicate one. Because the
file it lives in may also be a curated corpus, :func:`save_palette` is a
load-modify-save that touches **only** the ``colors`` key and preserves
``spans``, ``provenance`` and any unknown key verbatim.

**The format is the retired timeline's, on purpose**, and per-replicate files keep
speaking it: ``{"spans": {rid: [[t0, t1, label], ...]}, ...}`` with spans in
SECONDS keyed by region id, even though a home holds exactly one region. Keeping
the region key is what lets the legacy read-through be a filter rather than a
converter, and a converter is the thing that can be wrong.

**Provenance is why these marks can validate a detector rather than merely
describe one.** Each label carries the detector settings that produced its
spans -- channel, frequency band, value/count band, detection window, geometry,
and the box ``frac`` itself. Without it a saved corpus is a set of spans with no
record of what fired them, and T32 (fit a signature per label) cannot know which
config's output it is learning to reproduce.

``frac`` is there for Batch S slice 6: a box move retires a geometry, and marks
labelled against the old rectangle stay valid *for that rectangle* and for no
other. That is a sharper claim than staleness. A moved box can land on a
different animal or on empty plate, so a span carried forward as current is not
merely measured under settings that changed -- it **attributes one animal's
labelled behaviour to another**, which is the failure ``track_store``'s
per-replicate paths exist to prevent, and which no amount of graying-out fixes.
A mark that cannot say which rectangle it describes inherits the legacy corpus's
own defect, so ``frac`` is recorded from the first per-replicate write rather
than retrofitted. It lives in ``provenance`` and nowhere else -- a second
top-level copy would be the diverging-state failure T11 and T17 were both fixed
by deleting.

Reads are best-effort, as everywhere else here: a corrupt or missing sidecar means
"no marks yet", never a refusal. Writes go through a temp-and-rename so an
interrupted save cannot truncate a corpus that took real curation to build.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from core.replicate_home import ensure_home, home_path

SUFFIX = ".marks.json"

# A small fixed cycle, assigned to new labels in order. Deliberately not random:
# reopening a clip should show Flying in the same colour it had, and colour is
# derived from insertion order, which is stable across sessions because the
# sidecar preserves it.
_PALETTE = ["#4caf50", "#2196f3", "#ff9800", "#e91e63",
            "#9c27b0", "#00bcd4", "#cddc39", "#ff5722"]


def _empty() -> dict:
    return {"spans": {}, "colors": {}, "provenance": {}}


def marks_path(video_path: str, replicate_id: int | None = None) -> str | None:
    """Where one replicate's marks live; the legacy per-video path when
    ``replicate_id`` is None. See the module note on why both exist."""
    if not video_path:
        return None
    if replicate_id is None:
        base, _ = os.path.splitext(video_path)
        return base + SUFFIX
    return home_path(video_path, replicate_id, SUFFIX)


def _read(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    doc.setdefault("spans", {})
    doc.setdefault("colors", {})
    doc.setdefault("provenance", {})
    return doc


def load_marks(video_path: str, replicate_id: int | None = None,
               legacy_region: int | None = None) -> dict:
    """This replicate's marks doc, or an empty one. Never raises for a bad file.

    With no ``replicate_id`` this reads the per-video file whole, which is what
    the palette and any whole-clip reader want.

    With one, and no marks in that replicate's home, the per-video file is read
    THROUGH and filtered to ``legacy_region``'s spans -- never rewritten, never
    reshaped. The returned doc is the ordinary shape, still keyed by region, so a
    caller cannot tell a read-through from a home read and cannot accidentally
    persist one as the other: :func:`save_marks` always writes the home.

    ``legacy_region`` is a region INDEX and is passed separately from the
    replicate id on purpose -- ``track_store.load_track`` gives the full argument,
    and it is sharper here, because a marks file carries no stamp naming its
    region at all. Without an explicit index there is nothing to filter on, so a
    home with no marks correctly reads as empty rather than inheriting every
    region's spans.
    """
    path = marks_path(video_path, replicate_id)
    if not path:
        return _empty()
    if os.path.exists(path):
        return _read(path) or _empty()
    if replicate_id is None or legacy_region is None:
        return _empty()

    legacy_path = marks_path(video_path)
    if not legacy_path or not os.path.exists(legacy_path):
        return _empty()
    legacy = _read(legacy_path)
    if legacy is None:
        return _empty()
    rid = str(int(legacy_region))
    spans = legacy["spans"].get(rid)
    if not spans:
        return _empty()
    # Provenance is label-keyed and pre-dates per-replicate files, so a label
    # this region never saved may still be present, written by another region.
    # Only labels this region actually has spans for are carried, which is the
    # most that can be claimed without the mapping the module note refuses to
    # infer. Colours are the palette and are deliberately left to load_palette.
    labels = {s[2] for s in spans if isinstance(s, list) and len(s) > 2}
    prov = {k: v for k, v in legacy["provenance"].items() if k in labels}
    return {"spans": {rid: spans}, "colors": {}, "provenance": prov}


def save_marks(video_path: str, doc: dict,
               replicate_id: int | None = None) -> tuple[bool, str]:
    """Write ``doc`` as this replicate's marks sidecar. Returns ``(wrote, note)``.

    Always writes the home when given a ``replicate_id``, including when the doc
    it is saving came from a legacy read-through -- that is how the legacy file
    goes inert for a region without ever being touched.
    """
    path = marks_path(video_path, replicate_id)
    if not path:
        return False, ""
    try:
        if replicate_id is not None:
            # Normally present from the moment the box was drawn; a hand-tidied
            # or imported tree can leave it missing, and finding that out as a
            # failed write costs curation rather than compute.
            ensure_home(video_path, replicate_id)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        os.replace(tmp, path)
    except OSError as e:
        return False, f"could not save marks: {e}"
    return True, ""


def load_palette(video_path: str) -> dict:
    """The clip's label -> colour map. Per-video; see the module note."""
    path = marks_path(video_path)
    if not path or not os.path.exists(path):
        return {}
    doc = _read(path)
    return dict(doc["colors"]) if doc else {}


def save_palette(video_path: str, colors: dict) -> None:
    """Persist the clip's palette, touching ONLY the ``colors`` key.

    Load-modify-save rather than a whole-doc write, because the file this shares
    is the legacy per-video marks sidecar and may hold a hand-curated corpus
    (T30's 152 bouts). ``spans``, ``provenance`` and any key this module does not
    know about survive verbatim.

    Silent on failure, in ``tuning_store.save_tuning``'s spirit and for its
    reason: a lost colour assignment costs one palette entry, and this runs from
    the GUI thread during a save the user asked for something else from.
    """
    path = marks_path(video_path)
    if not path:
        return
    doc = _read(path) if os.path.exists(path) else None
    if doc is None:
        doc = _empty()
    doc["colors"] = dict(colors)
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass


def color_for(doc: dict, label: str) -> str:
    """This label's colour, assigning the next palette entry if it is new.
    Mutates ``doc['colors']`` so the assignment persists on the next save.

    ``doc`` here is whatever the caller is treating as the palette -- for a
    per-replicate save that is the per-video palette from :func:`load_palette`,
    wrapped, not the replicate's own doc. Insertion order is the assignment rule,
    so feeding it a per-replicate doc would restart the cycle in every home and
    give one behaviour a different colour per replicate.
    """
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

    **Does not assign a colour.** It used to, and could not once the palette went
    per-video and the spans per-replicate: assigning into ``doc`` would write the
    palette into the replicate's home, where insertion order restarts and one
    behaviour ends up a different colour per replicate. The caller pairs this
    with :func:`color_for` over the palette from :func:`load_palette`.
    """
    rid = str(int(region_index))
    kept = [s for s in doc["spans"].get(rid, []) if s[2] != label]
    kept.extend([float(t0), float(t1), label] for (t0, t1) in spans_s)
    kept.sort(key=lambda s: s[0])
    doc["spans"][rid] = kept
    if provenance is not None:
        doc.setdefault("provenance", {})[label] = {
            **provenance, "saved_utc": datetime.now(timezone.utc).isoformat()}
    return doc


__all__ = ["SUFFIX", "add_detection_spans", "color_for", "load_marks",
           "load_palette", "marks_path", "save_marks", "save_palette"]
