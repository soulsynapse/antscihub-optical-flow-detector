"""Where one replicate's work lives on disk.

A replicate is a region of a source video, and everything measured about it --
its detection track, its marks, its tuning, and (optionally) a transcoded clip of
just that region -- belongs together. Until this existed those artefacts were all
per-VIDEO sidecars, which is correct for the box list and wrong for everything
else: two replicates in one clip wrote to the same ``.track.npz``, so processing
the second destroyed the first, silently, at a cost of minutes of decode.

The home is a directory beside the source::

    footage/
      GX010047.MP4               <- source
      GX010047.rois.json         <- per-VIDEO: the boxes, their labels, next_id
      GX010047_rep01/
        GX010047.track.npz
        GX010047.marks.json
        GX010047.tuning.json
        GX010047.mkv             <- only if the replicate was cut out
      GX010047_rep02/ ...

**Named by id, never by label.** ``core.replicates._canonical_geometry`` already
ruled that "labels/calibration do not invalidate flow" -- a label is display
metadata, and the geometry hash excludes it deliberately. Naming a directory
after it would make free user text load-bearing: renaming would become a
directory move with every path inside it shifting, and two boxes called "control"
would be a filesystem collision rather than a cosmetic one. With an id-named home
a rename stays what it is, a string edit in the box sidecar.

**Stem-prefixed, because a folder may hold more than one source.** ``rep01/``
alone would be ambiguous the moment ``A.MP4`` and ``B.MP4`` sit side by side,
both with replicates; ``A_rep01`` and ``B_rep01`` cannot be confused, and sort
next to the source they belong to. The files inside keep the stem too, so one
carried out of its directory is still identifiable.

**Homes exist whether or not the source is ever cut up.** The directory is the
replicate's identity on disk, created when the box is drawn; a clip is one
optional thing inside it. That is what lets the transcode be a pure speed-up
rather than a mode: nothing about the layout changes when you skip it.

**Nothing here ever deletes a home.** Deleting a box in the GUI leaves its
directory alone, and the box list's ``next_id`` is monotonic, so no later box can
reclaim a retired id and silently inherit a dead animal's measurements. The cost
is a stale directory the user can remove by hand; the alternative is destroying a
curated corpus on a keystroke.

**Generations: the id means "this animal", a generation means "this rectangle".**
Moving a box does not invalidate the work under it, it *supersedes* it. The
superseded fileset moves down into ``old_NNN/`` beside a ``geometry.json``
recording the rectangle it was measured against::

    GX010047_rep01/
      GX010047.track.npz        <- CURRENT geometry, unnumbered
      GX010047.marks.json
      old_002/
        geometry.json           <- {"frac": [...], "retired_at": ...}
        GX010047.track.npz      <- measured on the rectangle in geometry.json
        GX010047.marks.json

Retiring rather than keeping is the point. A moved box can land on a different
animal, or on nothing, so marks carried forward as current would attribute one
animal's labelled behaviour to another -- the failure class the per-region track
split exists to prevent, not the milder staleness that ``TrackStamp`` already
catches. Retiring rather than *deleting* is the other half: a 2% nudge must not
take a hand-curated corpus with it.

**The current generation is unnumbered.** It sits at the home root and is given a
number only when it is superseded. That is what keeps the three stores out of
this entirely: they write to the home root via :func:`home_path`, and retiring
moves the fileset out from under them without their ever learning a generation
exists.

**The generation counter is DERIVED (``max(existing) + 1``), never persisted**,
and this is the one place that departs from ``next_id``. ``next_id`` has to be
stored because a deleted box leaves *no trace* in the box sidecar, so the counter
is the only memory of it. A retired generation leaves a **directory**, which is
itself the trace -- the filesystem enforces monotonicity for free. The hazard is
the same one at a smaller scale (a reissued generation would adopt a dead
rectangle's marks), so it must be read from the listing and never from a count.
"""
from __future__ import annotations

import datetime
import json
import os
import re

# Suffix appended to the source stem to name a home. Zero-padded to two digits so
# a directory listing sorts rep02 before rep10; ids past 99 simply get wider,
# which sorts imperfectly but stays unambiguous.
HOME_FMT = "{stem}_rep{rid:02d}"

# Suffix -> how to describe a file of that kind to a human, in the dialogs that
# have to say what a directory holds before the user does something to its box.
# Kept here rather than in the GUI because the set of things that live in a home
# is this module's business.
_KIND_ORDER = (".track.npz", ".marks.json", ".tuning.json",
               ".mkv", ".mp4", ".avi")

# A retired generation's directory name. Zero-padded to three so a listing sorts,
# and matched by a regex rather than by prefix so a user directory called
# "old_notes" is never mistaken for one and swept into the numbering.
GEN_FMT = "old_{gen:03d}"
_GEN_RE = re.compile(r"^old_(\d+)$")

# The retired rectangle, beside the files measured against it. This is the ONLY
# record of that geometry: the box sidecar holds current boxes and has no
# generation concept, and a copy there would be a second statement of one fact.
GEOMETRY_NAME = "geometry.json"


def home_name(video_path: str, replicate_id: int) -> str:
    """Directory name (not path) of a replicate's home."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return HOME_FMT.format(stem=stem, rid=int(replicate_id))


def replicate_dir(video_path: str, replicate_id: int) -> str:
    """Absolute-ish path to a replicate's home. Does not create it."""
    return os.path.join(os.path.dirname(video_path),
                        home_name(video_path, replicate_id))


def home_path(video_path: str, replicate_id: int, suffix: str) -> str:
    """Path to one artefact inside a home, e.g. ``suffix=".track.npz"``.

    The stem is repeated inside the directory on purpose: a file carried out of
    its home for inspection should still say which clip it came from.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(replicate_dir(video_path, replicate_id), stem + suffix)


def ensure_home(video_path: str, replicate_id: int) -> str | None:
    """Create a replicate's home if it is missing; return its path, or None.

    Best-effort in the same spirit as ``tuning_store.save_tuning``: this runs
    from the GUI thread the instant a box is drawn, and a read-only directory
    must not turn drawing a box into an error dialog. A missing home degrades to
    the artefacts simply not being written later, which the stores already report
    in their own terms.
    """
    if not video_path:
        return None
    path = replicate_dir(video_path, replicate_id)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return None
    return path


def sync_homes(video_path: str, replicates: list[dict]) -> list[str]:
    """Ensure every current box has a home. Returns the paths that now exist.

    Idempotent, and deliberately additive only: boxes that were deleted keep
    their directories (see the module note). Called after any edit to the box
    list, so a layout imported from another clip materialises its homes at once
    rather than one at a time as each replicate is first processed.
    """
    made = []
    for rep in replicates:
        try:
            rid = int(rep["id"])
        except (KeyError, TypeError, ValueError):
            continue
        path = ensure_home(video_path, rid)
        if path is not None:
            made.append(path)
    return made


def generation_dir(video_path: str, replicate_id: int, gen: int) -> str:
    """Path to one retired generation inside a home. Does not create it."""
    return os.path.join(replicate_dir(video_path, replicate_id),
                        GEN_FMT.format(gen=int(gen)))


def list_generations(video_path: str, replicate_id: int) -> list[dict]:
    """Every retired generation of a replicate, oldest first.

    Each entry is ``{"gen", "path", "frac", "retired_at", "held"}``, where
    ``frac`` is the rectangle it was measured against (None when the geometry
    file is missing or unreadable) and ``held`` is :func:`describe_home`'s
    phrasing applied to that generation.

    A generation whose ``frac`` will not load is still returned. It cannot be
    drawn, but it can be listed and restored, and silently dropping it would
    make a corpus disappear from the one surface that exists to show the user it
    is still there.
    """
    if not video_path:
        return []
    home = replicate_dir(video_path, replicate_id)
    try:
        names = os.listdir(home)
    except OSError:
        return []
    out = []
    for name in names:
        m = _GEN_RE.match(name)
        if not m or not os.path.isdir(os.path.join(home, name)):
            continue
        path = os.path.join(home, name)
        frac, retired_at = _read_geometry(path)
        out.append({"gen": int(m.group(1)), "path": path, "frac": frac,
                    "retired_at": retired_at,
                    "held": _describe_dir(path, _stem(video_path))})
    out.sort(key=lambda g: g["gen"])
    return out


def next_generation(video_path: str, replicate_id: int) -> int:
    """The number the next retirement takes: one past the highest that exists.

    Derived from the directory listing on purpose -- see the module note. Taking
    ``len(generations) + 1`` instead would reissue a number the moment a user
    removed an old directory by hand, and the reissued generation would adopt the
    removed rectangle's place in the history.
    """
    gens = [g["gen"] for g in list_generations(video_path, replicate_id)]
    return max(gens, default=0) + 1


def retire_current(video_path: str, replicate_id: int,
                   frac: tuple[float, float, float, float],
                   *, label: str = "") -> int | None:
    """Move the home's current fileset into a fresh ``old_NNN/``.

    Returns the generation number, or None when there was nothing to retire.
    ``frac`` is the rectangle the retired files were measured against -- the
    box's position *before* the move, which the caller still holds and this
    module has no way to know.

    **Nothing is retired when the home root holds no files.** Otherwise every
    nudge of a freshly drawn box would leave an empty numbered directory, and the
    history the user is meant to be able to read would fill with generations that
    record nothing. This mirrors the delete prompt, which likewise only speaks up
    when the home actually holds something.

    **Every regular file at the root moves, not a known-suffix whitelist.** A
    file a user or a future slice put in the home describes the old rectangle
    exactly as the track does, and leaving it behind would silently re-attribute
    it to the new one. Subdirectories stay: those are the other generations.
    """
    home = replicate_dir(video_path, replicate_id)
    try:
        names = [n for n in os.listdir(home)
                 if os.path.isfile(os.path.join(home, n))]
    except OSError:
        return None
    if not names:
        return None

    gen = next_generation(video_path, replicate_id)
    dest = generation_dir(video_path, replicate_id, gen)
    os.makedirs(dest, exist_ok=True)
    # Geometry first. A crash between the moves and the geometry write would
    # leave a generation whose rectangle is unknown -- files that cannot be
    # interpreted, which is worse than an empty directory beside intact ones.
    _write_geometry(dest, frac, label)
    for name in names:
        os.replace(os.path.join(home, name), os.path.join(dest, name))
    return gen


def restore_generation(video_path: str, replicate_id: int, gen: int,
                       current_frac: tuple[float, float, float, float],
                       *, label: str = "") -> tuple[float, float, float, float]:
    """Bring a retired generation back to the home root; return its rectangle.

    A swap, not a second mechanism: the fileset currently at the root is retired
    first (taking a new number), then ``gen``'s files move up and its now-empty
    directory is removed. So restoring is a move in the other direction, and
    restoring twice returns you to where you started with nothing lost either
    way.

    The returned ``frac`` is what the caller must write back onto the box -- the
    whole point of the gesture is that the detections come back *with* their
    rectangle. Raises ``FileNotFoundError`` if that generation is not there, and
    ``ValueError`` if it records no usable rectangle: restoring files while
    leaving the box where it is would recreate exactly the misattribution the
    retirement prevented.
    """
    src = generation_dir(video_path, replicate_id, gen)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"no generation {gen} in {replicate_dir(video_path, replicate_id)}")
    frac, _ = _read_geometry(src)
    if frac is None:
        raise ValueError(
            f"generation {gen} does not record the rectangle it was measured "
            "against, so its results cannot be re-aimed")

    # Retire what is current BEFORE moving anything up, or the two filesets
    # collide in one directory and the survivor is whichever moved last.
    retire_current(video_path, replicate_id, current_frac, label=label)
    home = replicate_dir(video_path, replicate_id)
    for name in os.listdir(src):
        if name == GEOMETRY_NAME:
            continue
        os.replace(os.path.join(src, name), os.path.join(home, name))
    # The directory goes only once it is empty of everything but its own
    # geometry record -- which has just been superseded by the box itself.
    try:
        os.remove(os.path.join(src, GEOMETRY_NAME))
        os.rmdir(src)
    except OSError:
        # Something else is in there. Leaving the husk is the safe failure: it
        # is visible, and it never costs data.
        pass
    return frac


def _write_geometry(gen_path: str, frac, label: str) -> None:
    doc = {"frac": [float(v) for v in frac],
           "retired_at": datetime.datetime.now().isoformat(timespec="seconds")}
    if label:
        doc["label"] = str(label)
    with open(os.path.join(gen_path, GEOMETRY_NAME), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def _read_geometry(gen_path: str) -> tuple[tuple | None, str]:
    try:
        with open(os.path.join(gen_path, GEOMETRY_NAME), encoding="utf-8") as f:
            doc = json.load(f)
        frac = tuple(float(v) for v in doc["frac"])
        if len(frac) != 4:
            return None, ""
        return frac, str(doc.get("retired_at", ""))
    except (OSError, ValueError, TypeError, KeyError):
        return None, ""


def describe_home(video_path: str, replicate_id: int) -> list[str]:
    """Human phrases for what a replicate's home holds; empty when it holds
    nothing that matters.

    Exists so the dialogs guarding a move or a delete can say *what* is at stake
    instead of asserting that something is. A warning that names "a detection
    track covering this clip and 9 marked spans across 2 labels" is a decision
    the user can actually make; "existing data will be lost" is not.

    Never raises and never blocks: an unreadable file is described by its
    presence and size rather than skipped, because "there is something here I
    could not open" is still information the user needs before overwriting it.
    """
    if not video_path:
        # No video loaded: replicate_dir would resolve against the process's cwd
        # and could list an unrelated "_rep01" that happens to sit there.
        return []
    out = _describe_dir(replicate_dir(video_path, replicate_id),
                        _stem(video_path))
    # Retired generations are named separately from the current fileset, and
    # never fall into the "other files" count -- they are the one thing in a home
    # a move does NOT put at risk, so listing them among what is at stake would
    # invert the dialog's meaning.
    gens = list_generations(video_path, replicate_id)
    if gens:
        out.append(f"{len(gens)} retired geometr"
                   f"{'ies' if len(gens) != 1 else 'y'} (kept)")
    return out


def _describe_dir(d: str, stem: str) -> list[str]:
    """What one directory of a home holds -- current root or a generation."""
    try:
        names = set(os.listdir(d))
    except OSError:
        return []
    out: list[str] = []
    for suffix in _KIND_ORDER:
        name = stem + suffix
        if name not in names:
            continue
        full = os.path.join(d, name)
        if suffix == ".track.npz":
            out.append(f"a detection track ({_size(full)})")
        elif suffix == ".marks.json":
            out.append(_describe_marks(full))
        elif suffix == ".tuning.json":
            out.append("remembered tuning (window, scale, bands)")
        else:
            out.append(f"a transcoded clip of this replicate ({_size(full)})")
    # Anything else a user or a future slice put here. Counted rather than named
    # so the list cannot grow unbounded in a message box. Generation directories
    # and a generation's own geometry record are structure, not content.
    known = {stem + s for s in _KIND_ORDER} | {GEOMETRY_NAME}
    extra = len([n for n in names if n not in known
                 and not (_GEN_RE.match(n) and os.path.isdir(os.path.join(d, n)))])
    if extra:
        out.append(f"{extra} other file{'s' if extra != 1 else ''}")
    return out


def _stem(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def _describe_marks(path: str) -> str:
    """"12 marked spans across 3 labels", or a fallback when it will not parse."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        spans = doc.get("spans") or {}
        n = sum(len(v) for v in spans.values() if isinstance(v, list))
        labels = {s[2] for v in spans.values() if isinstance(v, list)
                  for s in v if len(s) > 2}
    except (OSError, ValueError, TypeError, IndexError):
        return "a marks file (unreadable)"
    if not n:
        return "an empty marks file"
    return (f"{n} marked span{'s' if n != 1 else ''} across "
            f"{len(labels)} label{'s' if len(labels) != 1 else ''}")


def _size(path: str) -> str:
    try:
        n = os.path.getsize(path)
    except OSError:
        return "size unknown"
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.0f} MB"
    return f"{max(1, n // 1024)} KB"


__all__ = ["GEN_FMT", "GEOMETRY_NAME", "HOME_FMT", "describe_home",
           "ensure_home", "generation_dir", "home_name", "home_path",
           "list_generations", "next_generation", "replicate_dir",
           "restore_generation", "retire_current", "sync_homes"]
