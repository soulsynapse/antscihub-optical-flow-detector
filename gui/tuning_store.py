"""Memory for the live tuning surface, split per-video and per-replicate.

Reopening a clip should find it as you left it: the same window, the same
working scale and block, and the same three detection bands. Those are not
derivable from the video and are expensive to re-place by hand, so they are
written to a sidecar next to it -- ``.../Foo.mp4`` -> ``.../Foo.tuning.json``,
the same convention ROIs, replicates and the frame count already use (see
``AppState.video_sidecar`` and ``core.framecount``). Keyed on the video PATH
rather than its content hash for the same reasons stated there: a human can see
which JSON belongs to which clip, and moving the clip carries its tuning along.

**But not all of it is per-video, and the line is `view`.** Batch S slice 3
splits the payload:

===================  ==========  ==================================================
key                  lives       why
===================  ==========  ==================================================
``strip``            per-video   downsample, block, normalize, window: properties
                                 of the SOURCE, and a single pass runs at one scale
``process``          per-video   whole-video pass settings
``region_index``     per-video   which replicate was last selected -- a fact about
                                 the clip's session, not about any one replicate
``view``             per-home    channel, freq band, value band, count band,
                                 detect window, selection
===================  ==========  ==================================================

``view`` is exactly ``core.live_track.TrackStamp``'s invalidation set, which is
the argument for the split rather than a coincidence. ``count_band`` settles it
alone: ``core.detection.rescale_count_band`` and ``count_denom`` exist because
``inband_count`` produces a RAW block count that the detector compares without
normalizing by region size, so the same band means something ~13x different
between a 29-block region and a 377-block one (todo.md Batch N item 2). A
per-video count band is therefore not merely untidy -- it is a threshold that
silently stops firing when you switch replicates.

A per-replicate ``view`` lives in the replicate's home, beside its track and its
marks: ``.../Foo_rep02/Foo.tuning.json``.

**Legacy sidecars are offered to one region, once.** A pre-split file carries
``view`` next to ``region_index``, and that index says which replicate the view
describes -- so ``load_tuning`` hands it to that replicate and to no other,
``track_store.load_track``'s discipline. A pre-split file with no
``region_index`` names no region and its ``view`` is dropped rather than guessed
at: the cost of a miss is re-placing three bands, and the cost of a wrong guess
is a detector threshold from another animal's geometry.

Every read is best-effort. A corrupt, truncated or hand-edited sidecar means
"no remembered tuning" and the surface opens on its defaults -- losing a
remembered window is a nuisance, refusing to open the video is a failure, and
these are display settings, not data.

Band endpoints are legitimately +/-inf ("unbounded on this side"), which is not
strict JSON but round-trips exactly through Python's encoder and decoder. That
is deliberate: the alternative is a sentinel that ``None`` (never placed) is
already spoken for, and the reader below re-checks the shape anyway.
"""
from __future__ import annotations

import json
import os

from core.replicate_home import ensure_home, home_path

SUFFIX = ".tuning.json"

# Bumped when a stored payload can no longer be read the way it was written.
# A mismatch is discarded silently rather than migrated: the cost of losing a
# remembered window is one re-drag, and stale keys applied to new semantics is
# exactly the class of silent-wrong-threshold this file is meant to avoid.
VERSION = 1


def tuning_path(video_path: str, replicate_id: int | None = None) -> str:
    """Where one replicate's tuning lives; the per-video path when
    ``replicate_id`` is None. Both exist -- see the module note's table."""
    if replicate_id is None:
        base, _ = os.path.splitext(video_path)
        return base + SUFFIX
    return home_path(video_path, replicate_id, SUFFIX)


def _read(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != VERSION:
        return {}
    return data


def load_tuning(video_path: str, replicate_id: int | None = None,
                legacy_region: int | None = None) -> dict:
    """Remembered tuning; ``{}`` when there is none usable.

    With no ``replicate_id`` this is the per-video half (``strip``, ``process``,
    ``region_index``). With one it is that replicate's half (``view``), falling
    back ONCE to a pre-split per-video file -- but only when that file's own
    ``region_index`` is ``legacy_region``, so a view placed against one
    replicate's geometry cannot be handed to another's.

    ``legacy_region`` is a region INDEX, not a replicate id, and is passed
    separately for ``track_store.load_track``'s reason: the index is a position
    in ``core.replicates.in_tile_order``, so the two coincide only while ids run
    1..N with nothing deleted.
    """
    if not video_path:
        return {}
    data = _read(tuning_path(video_path, replicate_id))
    if replicate_id is None or data:
        return data
    if legacy_region is None:
        return {}
    legacy = _read(tuning_path(video_path))
    view = legacy.get("view")
    if not view:
        return {}
    saved_region = legacy.get("region_index")
    if saved_region is None or int(saved_region) != int(legacy_region):
        # Another replicate's view, or a file that names no region at all.
        # Neither is this one's to restore; see the module note.
        return {}
    return {"view": view}


def save_tuning(video_path: str, payload: dict,
                replicate_id: int | None = None) -> None:
    """Write ``payload`` as a tuning sidecar. Never raises.

    Failure is silent by design: this is called from a debounce on the GUI
    thread while the user tunes, and a read-only directory or a full disk must
    not interrupt that with a dialog per keystroke. The consequence of a failed
    write is that the next session opens on defaults.
    """
    if not video_path:
        return
    try:
        if replicate_id is not None:
            ensure_home(video_path, replicate_id)
        # Encoded whole before the file is opened. Encoding straight into the
        # handle would leave a half-written sidecar behind when some value
        # partway through does not serialise -- and a truncated file is
        # indistinguishable from a corrupt one, so the failure would take the
        # PREVIOUS session's good tuning with it.
        #
        # default=float rather than a custom encoder: everything stored here is
        # a number, a string, or a container of those, and the numbers arrive as
        # numpy scalars often enough (block counts, band endpoints read off
        # arrays) that letting one through unhandled would fail the write.
        blob = json.dumps({**payload, "version": VERSION}, indent=2,
                          default=float)
        with open(tuning_path(video_path, replicate_id), "w",
                  encoding="utf-8") as f:
            f.write(blob)
    except (OSError, TypeError, ValueError):
        pass


__all__ = ["SUFFIX", "VERSION", "load_tuning", "save_tuning", "tuning_path"]
