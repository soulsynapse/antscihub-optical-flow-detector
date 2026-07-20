"""The whole-video detection track: a per-frame series that fills progressively.

The live surface used to paint a trailing window and nothing else. This holds the
other half of the redesign (todo.md, "the live axis is the WHOLE VIDEO"): one
array per series at CLIP length, written into as windows arrive, so navigating
around a clip accumulates parts of a single picture instead of restarting one.

Three things make this affordable and honest, and each is load-bearing.

**Per-FRAME series only.** ``gate``, ``count`` and ``clump`` are (T,) -- ~120 KB
each at 30k frames -- so they carry the whole-video axis for free. The (F, T, B)
cube does NOT and never lives here; that is what the ring buffer bounds. The one
per-BLOCK array retained is ``band_power`` (T, B), because retaining it is what
makes a threshold re-tune instant instead of a re-stream, and at ~377 blocks per
replicate it is ~45 MB at 30k frames. :func:`band_power_bytes` prices it and
``retain_band_power=False`` declines it, so a geometry that does not fit degrades
to "re-tuning needs a refill" rather than to an out-of-memory error.

**Coverage is not zero.** An unwritten frame and a frame computed as quiet are
different claims, and a strip that paints them the same makes "nothing happened
here" indistinguishable from "nobody looked here" -- the standing failure this
codebase designs against (FINDINGS.md section 10, traps 7 and 8). So coverage is
carried explicitly in ``stamp_id``, never inferred from a zero.

**Coverage means the transform could SEE those frames.** Writes are trimmed by
the cone of influence (:func:`coi_trim`) before they land, so the padding response
at a window's edges is not written at all; a later overlapping window, for which
those same frames are interior rather than edge, supplies them. This is what
closes T35 -- the detector consuming the cone the display fades -- and it closes
it by never producing the contaminated numbers rather than by masking them
afterwards.

**Staleness is per frame, not per track.** ``stamp_id`` records WHICH settings
produced each frame, so changing the channel or the frequency band leaves the
earlier work on screen as visibly stale rather than blanking it. The split is
deliberate: a :class:`TrackStamp` covers only what the retained band power was
computed under (channel, frequency band, geometry, region). The cheap thresholds
-- value band, count band, detection window -- are NOT in it, because
:meth:`WholeVideoTrack.set_value_band` and :meth:`derive_gate` recompute those
over everything already covered. Putting them in the stamp would gray out the
whole clip on a count-band nudge and make the user re-stream to get it back.

Nothing here is Qt-aware and nothing here decodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.detection import (detect_gate, inband_count, largest_clump_per_frame,
                            windowed_mean)
from core.wavelet import coi_efolding_s

# ``stamp_id`` value meaning "never computed". A real stamp is always >= 1, so a
# zeroed array reads as fully uncovered, which is the correct initial claim.
UNCOVERED = 0


@dataclass(frozen=True)
class TrackStamp:
    """What a stretch of retained band power was computed UNDER.

    Only the fields that invalidate the BAND POWER belong here. Changing any of
    them means the retained (T, B) rows no longer answer the question being
    asked, so the frames they produced become stale and must be recomputed.
    Everything downstream of the band power -- the value band, the count band,
    the detection window -- is deliberately absent, because those re-derive from
    retained rows without a fresh transform.

    ``region_blocks`` is carried even though ``region_index`` implies it on any
    one grid, because it is what the count band is denominated in: two stamps
    with the same region index and different block sizes are not comparable, and
    the strip needs to say so.
    """
    channel: str
    freq_band_hz: tuple[float, float]
    grid: tuple[int, int]              # (ny, nx) block grid
    region_index: int
    region_blocks: int
    downsample: float
    block_size: int | None             # None = the grid tracked the scale

    def __post_init__(self):
        # Normalized so two stamps that mean the same thing hash the same: these
        # arrive from spin boxes and config dataclasses as ints, floats and
        # numpy scalars interchangeably, and an int/float mismatch on one field
        # would silently mark every frame stale on a no-op edit.
        object.__setattr__(self, "freq_band_hz",
                           (float(self.freq_band_hz[0]),
                            float(self.freq_band_hz[1])))
        object.__setattr__(self, "grid", (int(self.grid[0]), int(self.grid[1])))
        object.__setattr__(self, "region_index", int(self.region_index))
        object.__setattr__(self, "region_blocks", int(self.region_blocks))
        object.__setattr__(self, "downsample", float(self.downsample))
        object.__setattr__(self, "block_size",
                           None if self.block_size is None else int(self.block_size))


def band_power_bytes(n_frames: int, n_blocks: int) -> int:
    """Footprint of retaining whole-clip band power at ``(T, B)`` float32.

    Exposed so the caller prices the choice against a real budget instead of
    discovering it as a MemoryError partway through a pass. ~45 MB at 30k frames
    and 377 blocks; ~5 GB if someone points a region at a whole 5.3K frame at
    block 8, which is exactly the case that must decline rather than try.
    """
    return int(n_frames) * int(n_blocks) * 4


def coi_trim(freq_band_hz, fps: float) -> int:
    """Frames to discard at EACH end of a transformed window, in samples.

    The cone of influence is widest at the band's LOWEST frequency, so that is
    what sets the trim: at 24 fps a 0.5 Hz band contaminates ~33 frames at each
    edge (tau = 1.369/f seconds, from :func:`core.wavelet.coi_efolding_s`, not
    from the 1.46/f that circulated in the plan). Trimming by the widest cone
    means the higher frequencies in the band are trimmed more than they need to
    be -- accepted, because the alternative is a per-frequency ragged coverage
    mask for a series that has already been summed over the band, and the frames
    given up are supplied by the next overlapping window anyway.
    """
    flo = float(np.min(np.asarray(freq_band_hz, float)))
    if not np.isfinite(flo) or flo <= 0:
        return 0
    return int(np.ceil(coi_efolding_s(np.array([flo]))[0] * float(fps)))


@dataclass
class WholeVideoTrack:
    """Clip-length detection series with per-frame coverage and staleness.

    The series are always ``n_frames`` long and always readable; ``stamp_id`` is
    the sole authority on which entries mean anything. Reading ``gate`` without
    consulting it is the bug this class is shaped to make obvious.
    """
    n_frames: int
    fps: float
    n_blocks: int = 0
    retain_band_power: bool = True

    # Per-frame series over the whole clip.
    count: np.ndarray = field(init=False)      # blocks in the value band
    clump: np.ndarray = field(init=False)      # largest connected clump area
    gate: np.ndarray = field(init=False)       # positive detection (0/1)
    stamp_id: np.ndarray = field(init=False)   # UNCOVERED, or a stamp's id

    def __post_init__(self):
        T = int(self.n_frames)
        if T < 0:
            raise ValueError(f"n_frames must be >= 0, got {T}")
        self.n_frames = T
        self.fps = float(self.fps)
        self.count = np.zeros(T, np.float32)
        self.clump = np.zeros(T, np.float32)
        self.gate = np.zeros(T, np.float32)
        self.stamp_id = np.full(T, UNCOVERED, np.int32)
        # Retained (T, B) rows for the CURRENT stamp only, plus which of them are
        # filled. A previous stamp's rows are dropped outright on a stamp change
        # -- B itself moves with the grid, so they cannot be kept side by side,
        # and the per-frame series are what survive to be shown as stale.
        self._bp: np.ndarray | None = None
        self._bp_valid = np.zeros(T, bool)
        self._stamps: dict[TrackStamp, int] = {}
        self._current: TrackStamp | None = None
        self._current_id = UNCOVERED
        # Detector settings the per-frame series currently reflect.
        self._value_band = (0.0, float("inf"))
        self._count_band = (0.0, float("inf"))
        self._detect_window = 1
        self._centered = True
        self._region_grid = None

    # -- stamp / settings ----------------------------------------------------
    @property
    def stamp(self) -> TrackStamp | None:
        return self._current

    @property
    def current_id(self) -> int:
        return self._current_id

    def set_stamp(self, stamp: TrackStamp, region_grid=None) -> bool:
        """Declare what subsequent writes are computed under.

        Returns whether the stamp actually changed. On a change the retained band
        power is dropped (its B no longer matches) but the per-frame series and
        ``stamp_id`` are NOT: those are what let the strip keep showing earlier
        work grayed rather than blanking the clip, which is the whole point of
        stamping per frame instead of per track.

        A stamp seen before gets its old id back, so returning to a previous
        channel makes the frames it produced current again rather than leaving
        them stale forever. Their band power is gone, so re-tuning a threshold
        over them still needs a refill -- but they are honest history, not
        garbage, and the detector settings that produced them are unchanged.
        """
        self._region_grid = region_grid
        if stamp == self._current:
            return False
        self._current = stamp
        if stamp not in self._stamps:
            self._stamps[stamp] = len(self._stamps) + 1     # ids start at 1
        self._current_id = self._stamps[stamp]
        self.n_blocks = int(stamp.region_blocks)
        self._bp = None
        self._bp_valid[:] = False
        return True

    def set_detector(self, *, value_band, count_band, detect_window: int,
                     centered: bool) -> None:
        """Apply threshold settings, re-deriving whatever they invalidate.

        The two halves cost very different things and are separated on purpose:

        * ``value_band`` feeds :func:`inband_count` and
          :func:`largest_clump_per_frame`, so changing it means a per-frame pass
          over every retained band-power row -- and the clump half is a Python
          loop over connected components, the expensive part of a re-tune.
        * ``count_band`` and ``detect_window`` touch only the windowed mean and
          the gate, which are a cumsum and a comparison over (T,).

        So a count-band drag is free at whole-clip length and a value-band drag
        is not. Callers should debounce the latter; nothing here does it for
        them, because a core object that swallowed edits would make the cost
        invisible rather than absent.
        """
        vb = (float(value_band[0]), float(value_band[1]))
        redo_value = vb != self._value_band
        self._value_band = vb
        self._count_band = (float(count_band[0]), float(count_band[1]))
        self._detect_window = max(1, int(detect_window))
        self._centered = bool(centered)
        if redo_value:
            self._rederive_value_series()
        self._derive_gate()

    # -- writing -------------------------------------------------------------
    def write(self, first: int, band_power: np.ndarray, *, trim: int = 0,
              trim_head: int | None = None, trim_tail: int | None = None) -> tuple[int, int]:
        """Write transformed rows for the window starting at absolute ``first``.

        ``band_power`` is ``(w, B)`` as :func:`core.wavelet.morlet_band_power`
        returns it. ``trim`` frames are dropped from EACH end before anything
        lands -- see :func:`coi_trim` -- so what is recorded as covered is only
        what the transform could actually see. Returns the absolute ``[a, b)``
        actually written, which is empty when the window is shorter than its own
        cone.

        ``trim_head`` / ``trim_tail`` override ``trim`` at one end. The head
        override is what lets a window that begins at frame 0 keep its opening
        frames: the record genuinely starts there, so those frames are at the
        edge of the DATA rather than at the edge of an arbitrary cut, and
        discarding them would leave the first ~33 frames of every clip
        permanently unexamined. The same applies at the true end of the clip.
        """
        bp = np.asarray(band_power, np.float32)
        if bp.ndim != 2:
            raise ValueError(f"band_power must be (w, B), got {bp.shape}")
        w, B = bp.shape
        if self._current is None:
            raise RuntimeError("set_stamp() before write(): an unstamped write "
                               "could not be told apart from a stale one")
        if B != self.n_blocks:
            raise ValueError(
                f"band_power has {B} blocks but the stamp says "
                f"{self.n_blocks}; the grid moved without a new stamp")
        head = trim if trim_head is None else trim_head
        tail = trim if trim_tail is None else trim_tail
        a = int(first) + max(0, int(head))
        b = int(first) + w - max(0, int(tail))
        # Clip to the track. A window may legitimately run past the last frame
        # when the decoder reports a different count than the container header.
        a, b = max(0, a), min(self.n_frames, b)
        if b <= a:
            return (a, a)
        lo, hi = a - int(first), b - int(first)
        rows = bp[lo:hi]
        if self.retain_band_power:
            if self._bp is None:
                self._bp = np.zeros((self.n_frames, B), np.float32)
            self._bp[a:b] = rows
            self._bp_valid[a:b] = True
        # Per-frame and memoryless in the detection window, so they are computed
        # once here for the rows that arrived rather than over the whole clip on
        # every tick -- which is what keeps the cost of a live pass flat in the
        # length of the island instead of growing with it.
        vlo, vhi = self._value_band
        self.count[a:b] = inband_count(rows, vlo, vhi)
        self.clump[a:b] = self._clump_of(rows)
        self.stamp_id[a:b] = self._current_id
        self._derive_gate()
        return (a, b)

    def _clump_of(self, rows: np.ndarray) -> np.ndarray:
        if self._region_grid is None or rows.size == 0:
            return np.zeros(rows.shape[0], np.float32)
        dy, dx, gy, gx = self._region_grid
        vlo, vhi = self._value_band
        return largest_clump_per_frame(rows, vlo, vhi, dy, dx, gy, gx)

    # -- deriving ------------------------------------------------------------
    def _rederive_value_series(self) -> None:
        """Recompute ``count`` and ``clump`` from retained band power.

        Only over rows this track still HOLDS. Frames covered under an older
        stamp keep the values they were computed with: they are already marked
        stale, and silently recomputing a subset of them under the new value band
        would leave the strip mixing two definitions with nothing to say which
        was which.
        """
        if self._bp is None or not self._bp_valid.any():
            return
        vlo, vhi = self._value_band
        idx = np.flatnonzero(self._bp_valid)
        # Contiguous runs, so the clump loop sees whole spans rather than one
        # frame at a time; the components are per frame either way, but the call
        # overhead across a 30k-frame clip is not nothing.
        for a, b in _runs(idx):
            rows = self._bp[a:b]
            self.count[a:b] = inband_count(rows, vlo, vhi)
            self.clump[a:b] = self._clump_of(rows)

    def _derive_gate(self) -> None:
        """Windowed mean and gate, computed PER CONTIGUOUS CURRENT RUN.

        Per run, not over the whole array: ``windowed_mean`` is a cumsum, so
        running it across an uncovered gap would average real counts together
        with the zeros standing in for frames nobody examined, and produce a
        gate that dips near every gap edge for no reason in the data. Per run,
        each island's window truncates at its own boundary exactly as it does at
        the clip's -- which is what ``window_bounds`` already does honestly.

        CURRENT rather than merely covered, and this is the subtle half. A stale
        frame's ``count`` was computed under the value band in force when it was
        written; re-gating it against a count band chosen later would combine
        two settings that never coexisted and present the result as that
        stretch's detection. It would also make the gray bars visibly move as
        you tune knobs that provably do not apply to them, which is precisely
        the "silently correct and silently wrong look identical" failure the
        stamp exists to prevent. So a stale frame keeps the gate it was computed
        with, frozen, until a pass re-examines it.
        """
        current = self.current
        if current.any():
            self.gate[current] = 0.0
            blo, bhi = self._count_band
            for a, b in _runs(np.flatnonzero(current)):
                windowed = windowed_mean(self.count[a:b], self._detect_window,
                                         self._centered)
                self.gate[a:b] = detect_gate(windowed, blo, bhi)

    # -- reading -------------------------------------------------------------
    @property
    def covered(self) -> np.ndarray:
        """Frames computed under ANY stamp. Includes stale ones -- a stale frame
        was examined, it was just examined under different settings."""
        return self.stamp_id != UNCOVERED

    @property
    def current(self) -> np.ndarray:
        """Frames computed under the stamp now in force."""
        return self.stamp_id == self._current_id

    @property
    def stale(self) -> np.ndarray:
        """Covered, but under settings no longer in force."""
        return self.covered & ~self.current

    def coverage_fraction(self) -> float:
        if self.n_frames == 0:
            return 0.0
        return float(self.current.sum()) / self.n_frames

    def detected_intervals(self, *, current_only: bool = True
                           ) -> list[tuple[int, int]]:
        """Contiguous ``[start, end)`` runs where the gate is on.

        ``current_only`` because navigation should step through detections the
        current settings actually produced. A stale detection is still shown on
        the strip -- it happened, under settings you can see are different -- but
        stepping into one silently would present it as an answer to the question
        being asked now.
        """
        g = (self.gate > 0.5)
        if current_only:
            g = g & self.current
        if not g.any():
            return []
        edges = np.diff(np.concatenate([[0], g.view(np.int8), [0]]))
        return [(int(s), int(e)) for s, e in
                zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]

    def gaps(self, a: int = 0, b: int | None = None) -> list[tuple[int, int]]:
        """Uncovered-or-stale spans within ``[a, b)`` -- what still needs a pass.
        The complement of :attr:`current`, in the form a scheduler wants."""
        b = self.n_frames if b is None else int(b)
        a = max(0, int(a))
        b = min(self.n_frames, b)
        if b <= a:
            return []
        return [(int(s + a), int(e + a))
                for s, e in _runs(np.flatnonzero(~self.current[a:b]))]


def _runs(idx: np.ndarray) -> list[tuple[int, int]]:
    """Sorted indices -> contiguous ``[start, end)`` spans."""
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]]) + 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]
