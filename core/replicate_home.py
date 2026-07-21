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
"""
from __future__ import annotations

import json
import os

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
    d = replicate_dir(video_path, replicate_id)
    try:
        names = set(os.listdir(d))
    except OSError:
        return []

    stem = os.path.splitext(os.path.basename(video_path))[0]
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
    # so the list cannot grow unbounded in a message box.
    known = {stem + s for s in _KIND_ORDER}
    extra = len([n for n in names if n not in known])
    if extra:
        out.append(f"{extra} other file{'s' if extra != 1 else ''}")
    return out


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


__all__ = ["HOME_FMT", "describe_home", "ensure_home", "home_name", "home_path",
           "replicate_dir", "sync_homes"]
