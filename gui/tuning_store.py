"""Per-video memory for the live tuning surface.

Reopening a clip should find it as you left it: the same window, the same
working scale and block, and the same three detection bands. Those are not
derivable from the video and are expensive to re-place by hand, so they are
written to a sidecar next to it -- ``.../Foo.mp4`` -> ``.../Foo.tuning.json``,
the same convention ROIs, replicates and the frame count already use (see
``AppState.video_sidecar`` and ``core.framecount``). Keyed on the video PATH
rather than its content hash for the same reasons stated there: a human can see
which JSON belongs to which clip, and moving the clip carries its tuning along.

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

SUFFIX = ".tuning.json"

# Bumped when a stored payload can no longer be read the way it was written.
# A mismatch is discarded silently rather than migrated: the cost of losing a
# remembered window is one re-drag, and stale keys applied to new semantics is
# exactly the class of silent-wrong-threshold this file is meant to avoid.
VERSION = 1


def tuning_path(video_path: str) -> str:
    base, _ = os.path.splitext(video_path)
    return base + SUFFIX


def load_tuning(video_path: str) -> dict:
    """Remembered tuning for ``video_path``; ``{}`` when there is none usable."""
    if not video_path:
        return {}
    try:
        with open(tuning_path(video_path), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != VERSION:
        return {}
    return data


def save_tuning(video_path: str, payload: dict) -> None:
    """Write ``payload`` as ``video_path``'s tuning sidecar. Never raises.

    Failure is silent by design: this is called from a debounce on the GUI
    thread while the user tunes, and a read-only directory or a full disk must
    not interrupt that with a dialog per keystroke. The consequence of a failed
    write is that the next session opens on defaults.
    """
    if not video_path:
        return
    try:
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
        with open(tuning_path(video_path), "w", encoding="utf-8") as f:
            f.write(blob)
    except (OSError, TypeError, ValueError):
        pass


__all__ = ["SUFFIX", "VERSION", "load_tuning", "save_tuning", "tuning_path"]
