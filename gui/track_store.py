"""Per-video memory for the whole-video detection track.

The sibling of ``gui/tuning_store``, and for the same reason with a much larger
stake. That one remembers where the knobs were; this one remembers *the result of
running them*. A whole-video pass is minutes of decode, and until this existed
the track lived only in ``LiveScalogramSurface._track`` -- so hiding the tab,
editing a replicate (which rebuilds the surface), or reopening the clip tomorrow
discarded it, and the only way back to the same answer was to spend the same
minutes again.

**A track belongs to one REPLICATE, not to the video.** ``TrackStamp`` carries
``region_index`` and ``region_blocks`` precisely because a track is one region's
answer, but this file used to derive a single path from the video -- so running
replicate 2 and then replicate 3 wrote both into ``Foo.track.npz`` and the second
destroyed the first, silently, at a cost of minutes of decode each time. Tracks
now live in the replicate's home (``core.replicate_home``): ``.../Foo.mp4``
replicate 2 -> ``.../Foo_rep02/Foo.track.npz``.

Passing no ``replicate_id`` still resolves the old per-video path. That is not a
compatibility shim for its own sake -- it is how a sidecar written before homes
existed is found and adopted, ONCE, by the replicate it actually belongs to. See
:func:`load_track`: the legacy file names its own region in its stamp, so it is
offered to that region and to no other. A file that cannot say which replicate it
describes is ignored rather than guessed at.

Keyed on the video PATH either way, so a human can see which file belongs to
which clip and moving the clip carries its work along.

**A restore never launders stale work as current.** The stamp registry travels
with the arrays, so a frame restored under settings that no longer apply comes
back marked stale -- gray on the strip -- exactly as it would have been had it
been computed this session and then had the channel changed underneath it. That
is the whole reason ``WholeVideoTrack`` stamps per frame rather than per track,
and it is what makes restoring safe: the worst case is that the strip shows you
gray, not that it shows you a number computed for a different question.

**Band power is optional and capped.** It is the one per-BLOCK array retained,
and its size is ``T * B * 4`` -- ~45 MB at 30k frames and 377 blocks, but 1.6 GB
at 30k frames and 12,995 blocks, which is a real geometry this project uses
(block 4 on a 462x456 crop). Above :data:`BP_DISK_CAP` it is dropped from the
sidecar rather than written. The consequence is bounded and already a documented
mode: without retained band power a *value band* re-tune needs a fresh pass,
while navigation, gating and coverage all still work.

Reads are best-effort in the same way ``tuning_store``'s are -- a corrupt or
outdated sidecar means "no remembered track", never a refusal to open the clip.
Writes are not silent, though: unlike a remembered window, losing this costs
real compute, so the caller is told and says so in the status line.
"""
from __future__ import annotations

import json
import os

import numpy as np

from core.live_track import WholeVideoTrack
from core.replicate_home import ensure_home, home_path

SUFFIX = ".track.npz"

# Bumped when a stored payload can no longer be read the way it was written.
# Discarded rather than migrated, for the reason tuning_store gives: the cost of
# a miss is a re-pass, and the cost of a wrong migration is a strip asserting
# coverage under settings that never produced it.
VERSION = 1

# Largest band-power array written to disk. Everything else in the payload is
# per-FRAME and together under a megabyte at clip length, so this cap is
# effectively the whole file-size policy.
#
# 256 MB rather than the 1 GB the in-memory cap allows: this is written on every
# tab hide, and a gigabyte per hide would turn a tab switch into a visible stall
# for a convenience the user did not ask for at that moment.
BP_DISK_CAP = 256 * 1024 ** 2

# Keys stored as arrays; everything else goes through one JSON blob so the
# schema lives in one readable place instead of across a dozen npz keys.
_ARRAYS = ("count", "clump", "gate", "stamp_id")


def track_path(video_path: str, replicate_id: int | None = None) -> str:
    """Where one replicate's track lives; the legacy per-video path when
    ``replicate_id`` is None. See the module note on why both exist."""
    if replicate_id is None:
        base, _ = os.path.splitext(video_path)
        return base + SUFFIX
    return home_path(video_path, replicate_id, SUFFIX)


def save_track(video_path: str, track: WholeVideoTrack,
               replicate_id: int | None = None) -> tuple[bool, str]:
    """Write ``track`` as this replicate's sidecar.

    Returns ``(wrote, note)``. The note is non-empty when something the user
    would want to know happened -- band power declined for size, or the write
    failed outright. Reported rather than swallowed because this is not a
    display setting: a failed write means the next session re-runs a pass that
    had already been paid for, and finding that out then is worse than now.
    """
    if not video_path:
        return False, ""
    state = track.to_state()
    note = ""
    bp = state.pop("band_power", None)
    valid = state.pop("bp_valid", None)
    arrays = {k: np.asarray(state.pop(k)) for k in _ARRAYS}
    grid = state.pop("region_grid", None)

    # Nothing computed -> nothing to remember, and an empty sidecar would be
    # indistinguishable from a stale one on the next read.
    if not (arrays["stamp_id"] != 0).any():
        _remove(video_path, replicate_id)
        return False, ""

    if bp is not None:
        nbytes = int(np.asarray(bp).nbytes)
        if nbytes <= BP_DISK_CAP:
            arrays["band_power"] = np.asarray(bp, np.float32)
            arrays["bp_valid"] = np.asarray(valid, bool)
        else:
            note = (f"detection track saved, but its {nbytes / 1024 ** 3:.1f} GB "
                    f"of band power was too large to store — re-tuning the "
                    f"value band next session will need another pass")
    if grid is not None:
        dy, dx, gy, gx = grid
        state["region_grid_dims"] = [int(dy), int(dx)]
        arrays["region_grid_y"] = np.asarray(gy, np.int32)
        arrays["region_grid_x"] = np.asarray(gx, np.int32)

    try:
        blob = json.dumps({**state, "version": VERSION}, default=float)
    except (TypeError, ValueError) as e:
        return False, f"detection track could not be encoded: {e}"
    try:
        # Uncompressed: the payload is float32 detection energy, which compresses
        # poorly, and this runs on the GUI thread from hideEvent. Measured
        # priorities are the other way round from tuning_store's tiny JSON.
        #
        # Written to a temp path and renamed, so an interrupted write cannot
        # replace a good sidecar with a truncated one -- the failure mode that
        # would silently cost the user the pass this file exists to preserve.
        path = track_path(video_path, replicate_id)
        # The home normally exists from the moment the box was drawn, but a
        # layout imported into a read-only tree, or one a user tidied by hand,
        # can leave it missing -- and discovering that as a failed write costs
        # the pass this file exists to preserve.
        if replicate_id is not None:
            ensure_home(video_path, replicate_id)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, meta=np.frombuffer(blob.encode("utf-8"), np.uint8),
                     **arrays)
        os.replace(tmp, path)
    except OSError as e:
        return False, f"detection track could not be saved: {e}"
    return True, note


def load_track(video_path: str, n_frames: int, fps: float,
               replicate_id: int | None = None,
               legacy_region: int | None = None
               ) -> tuple[WholeVideoTrack | None, str]:
    """Read this replicate's remembered track, or ``(None, note)``.

    When the replicate has no track of its own, a legacy per-video sidecar is
    offered to it -- but only if that file SAYS it is this one's, by carrying
    ``legacy_region`` as the ``region_index`` of the stamp it was written under.
    A pre-homes sidecar holds exactly one region's work (that was the bug), so at
    most one replicate can claim it and the rest correctly see nothing. A file
    with no stamp names no region and is left alone: adopting it would be a
    guess, and the thing being guessed at is which animal the numbers describe.

    ``legacy_region`` is passed separately because a replicate ID and a region
    INDEX are not the same number -- the index is a position in
    ``core.replicates.in_tile_order``, which sorts by id, so they coincide only
    when ids happen to run 1..N with nothing deleted. Deriving one from the other
    here is exactly the shortcut that would adopt a neighbour's track.

    The legacy file is not deleted or moved on adoption. The next save writes the
    home copy, and the original goes inert the moment the home exists -- which is
    preferable to a migration that can half-succeed.

    ``n_frames`` and ``fps`` are checked against the sidecar rather than trusted
    from it. They are the clip's identity as far as a per-frame series is
    concerned: a recrop, a re-encode at another rate, or a different container
    reporting a different count all mean the stored indices point at different
    footage, and a track whose frame 9000 is not this clip's frame 9000 is worse
    than no track. (See the standing note in MEMORY: robustness to a standard
    ffmpeg recrop is required, so this must *detect* the change, not assume it
    cannot happen.)
    """
    if not video_path:
        return None, ""
    path = track_path(video_path, replicate_id)
    if not os.path.exists(path):
        if replicate_id is None or legacy_region is None:
            return None, ""
        legacy = track_path(video_path)
        if not os.path.exists(legacy):
            return None, ""
        track, note = _read(legacy, n_frames, fps)
        if track is None or track.stamp is None:
            return None, note
        if int(track.stamp.region_index) != int(legacy_region):
            # Someone else's work, or nobody's. Not this replicate's to show.
            return None, ""
        return track, (note and note + " (from a pre-replicate-folder sidecar)")
    return _read(path, n_frames, fps)


def _read(path: str, n_frames: int, fps: float
          ) -> tuple[WholeVideoTrack | None, str]:
    """Parse one sidecar. The identity checks live here so the legacy path gets
    exactly the same scrutiny as the home one -- an old file is more likely to
    be stale, not less."""
    try:
        with np.load(path, allow_pickle=False) as data:
            state = json.loads(bytes(data["meta"]).decode("utf-8"))
            if state.get("version") != VERSION:
                return None, ""
            if int(state["n_frames"]) != int(n_frames):
                return None, (
                    f"a remembered detection track was found but it covers "
                    f"{state['n_frames']} frames and this clip has {n_frames} "
                    f"— ignored")
            if abs(float(state["fps"]) - float(fps)) > 1e-6:
                return None, (
                    "a remembered detection track was found but it was computed "
                    "at a different frame rate — ignored")
            for k in _ARRAYS:
                state[k] = data[k]
            if "band_power" in data.files:
                state["band_power"] = data["band_power"]
                state["bp_valid"] = data["bp_valid"]
            dims = state.pop("region_grid_dims", None)
            if dims is not None and "region_grid_y" in data.files:
                state["region_grid"] = (int(dims[0]), int(dims[1]),
                                        data["region_grid_y"],
                                        data["region_grid_x"])
            track = WholeVideoTrack.from_state(state)
    except (OSError, ValueError, KeyError, TypeError) as e:
        # Every malformed shape from_state rejects lands here too, which is the
        # intent: it raises so that this one place decides what a bad sidecar
        # means, and it means "no remembered track".
        return None, f"a remembered detection track could not be read ({e})"
    covered = float(track.covered.mean()) * 100.0
    return track, (f"restored an earlier detection track — {covered:.0f}% of "
                   f"the clip already examined")


def _remove(video_path: str, replicate_id: int | None = None) -> None:
    try:
        os.remove(track_path(video_path, replicate_id))
    except OSError:
        pass


__all__ = ["BP_DISK_CAP", "SUFFIX", "VERSION", "load_track", "save_track",
           "track_path"]
