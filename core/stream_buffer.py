"""A bounded, append-only ring of per-block channel frames: the backing store for
continuous (rather than windowed) live processing.

The live surface used to extract a fixed window and block until it finished. This
holds the other model: a worker streams forward from wherever the user is, and the
GUI renders whatever has arrived. Two decisions shape everything here.

**One island, not many.** Seeking forward past the frontier abandons the current
span and restarts at the new position, rather than backfilling a gap. So the
buffer always represents exactly ONE contiguous run of absolute video frames --
``[start, start + count)`` -- and there is no gap list, no partial-coverage query
and no second worker competing for the decoder. Seeking backward inside the
retained span is instant; seeking before it is just another restart.

**Bounded, always.** Capacity is frames, fixed at construction from a byte budget.
Past it the oldest frame is dropped. This is what keeps a fine block grid usable:
at block 16 a 30k-frame clip is ~1.8 GB for two channels, which is not a buffer,
it is an out-of-memory error with extra steps. Scrolling back past the ring start
reprocesses rather than reading stale rows.

Nothing here is Qt-aware and nothing here decodes. It is the shared contract
between the extraction worker that appends and the plots that read, so both can
be tested without a video or an event loop.

**Threading discipline -- this class is NOT internally synchronised.** Reads and
writes must happen on one thread. The intended pattern is that the extraction
worker owns the buffer and appends to it, and publishes ``frontier`` through a
QUEUED signal; the GUI thread then asks the worker for a window rather than
touching the buffer directly. The failure this rules out is subtle and would be
blamed on the wavelet: :meth:`window` checks its bounds and then copies, so an
``append`` that evicts between those two steps returns rows belonging to a
different point in the clip, silently and only under load. A lock is deliberately
not taken here -- it would make every read look safe while leaving the
check-then-copy race intact for any caller that held indices across calls.
"""
from __future__ import annotations

import numpy as np

# Below this the ring cannot hold enough history for the slowest wavelet scale to
# mean anything, and a "live" plot of a band that is entirely cone-of-influence is
# worse than no plot. Callers that cannot afford it should coarsen the grid rather
# than accept a ring this short.
MIN_CAPACITY = 64


def capacity_for_budget(budget_bytes: int, n_channels: int, ny: int, nx: int,
                        itemsize: int = 4) -> int:
    """How many frames fit in ``budget_bytes``. Never below :data:`MIN_CAPACITY`,
    so a budget too small to be useful produces a ring that is honestly too small
    rather than one that silently holds three frames."""
    per_frame = max(1, int(n_channels) * int(ny) * int(nx) * int(itemsize))
    return max(MIN_CAPACITY, int(budget_bytes) // per_frame)


class StreamBuffer:
    """Ring of ``(ny, nx)`` block frames per channel over one contiguous span.

    Absolute video frame indices are the only currency in this API. The ring
    position of a frame is an internal detail, because exposing it would let a
    caller hold an index across an eviction and read the wrong frame -- the exact
    class of bug this buffer exists in the middle of.
    """

    def __init__(self, channels, ny: int, nx: int, capacity: int,
                 start: int = 0):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.channels = tuple(channels)
        if not self.channels:
            raise ValueError("at least one channel is required")
        self.ny, self.nx = int(ny), int(nx)
        self.capacity = int(capacity)
        self._buf = {c: np.zeros((self.capacity, self.ny, self.nx), np.float32)
                     for c in self.channels}
        self._start = int(start)     # absolute index of the oldest retained frame
        self._count = 0

    # -- span --------------------------------------------------------------
    @property
    def start(self) -> int:
        """Absolute index of the oldest retained frame."""
        return self._start

    @property
    def count(self) -> int:
        """Frames currently retained."""
        return self._count

    @property
    def frontier(self) -> int:
        """Absolute index one past the newest retained frame -- i.e. the next
        frame the worker is expected to append."""
        return self._start + self._count

    @property
    def evicted(self) -> bool:
        """Whether the ring has wrapped and is dropping history. Surfaces in the
        UI as the retained span no longer reaching the island's origin."""
        return self._count >= self.capacity

    def covers(self, a: int, b: int) -> bool:
        """Is ``[a, b)`` entirely retained? Empty/inverted ranges are False, not
        vacuously True: a plot asking for nothing wants to be told it has nothing
        rather than to render an empty array as data."""
        return b > a and a >= self._start and b <= self.frontier

    # -- writing -----------------------------------------------------------
    def reset(self, start: int) -> None:
        """Abandon the island and begin a new one at absolute frame ``start``.

        The buffers are NOT zeroed: nothing can read a row until ``count`` grows
        past it, so clearing would be pure cost on the one operation the user
        feels directly (a seek). ``count`` is the sole authority on validity.
        """
        self._start = int(start)
        self._count = 0

    def append(self, index: int, frame: dict) -> None:
        """Append the block frame at absolute index ``index``.

        Rejects a non-contiguous index rather than silently starting a new span.
        A worker that has skipped a frame has a bug, and letting it write here
        would produce a buffer whose ``[start, frontier)`` claim is a lie --
        every downstream time axis, wavelet transform and detection window reads
        that span as uniformly sampled. Call :meth:`reset` to move deliberately.
        """
        if index != self.frontier:
            raise ValueError(
                f"non-contiguous append: frame {index} but frontier is "
                f"{self.frontier}; call reset() to start a new span")
        # Validate EVERY plane before writing ANY. On a full ring
        # ``index % capacity`` is the oldest *retained* frame's slot, so a
        # validate-as-you-go loop that fails on the second channel would have
        # already overwritten the first -- leaving a frame inside the valid span
        # holding one channel from this frame and the rest from the evicted one.
        # That is a silent cross-channel misalignment, and the span would still
        # claim to be intact. Two passes cost nothing next to the tensor work.
        for c in self.channels:
            plane = frame.get(c)
            if plane is None:
                raise KeyError(f"frame is missing channel {c!r}")
            if plane.shape != (self.ny, self.nx):
                raise ValueError(f"channel {c!r} has shape {plane.shape}, "
                                 f"expected {(self.ny, self.nx)}")
        pos = index % self.capacity
        for c in self.channels:
            self._buf[c][pos] = frame[c]
        if self._count < self.capacity:
            self._count += 1
        else:
            self._start += 1        # evict oldest; span slides forward

    # -- reading -----------------------------------------------------------
    def window(self, a: int, b: int, channel: str) -> np.ndarray:
        """A contiguous ``(b - a, ny, nx)`` copy of ``[a, b)`` for one channel.

        A copy, not a view: the span may straddle the ring seam, and a consumer
        (the wavelet transform above all) needs one array whose time axis is
        actually contiguous. Returning a view where it happens not to wrap and a
        copy where it does would make aliasing depend on the buffer's fill state,
        which is the kind of intermittent behaviour that costs a day to find.
        """
        if channel not in self._buf:
            raise KeyError(f"unknown channel {channel!r}")
        if not self.covers(a, b):
            raise ValueError(f"[{a}, {b}) is not retained; buffer holds "
                             f"[{self._start}, {self.frontier})")
        lo, hi = a % self.capacity, ((b - 1) % self.capacity) + 1
        buf = self._buf[channel]
        if lo < hi:
            return buf[lo:hi].copy()
        return np.concatenate([buf[lo:], buf[:hi]])     # straddles the seam

    def latest(self, n: int, channel: str) -> tuple:
        """The trailing ``min(n, count)`` frames as ``(array, first_index)``.

        The workhorse read: the live scalogram recomputes over a bounded trailing
        window rather than over a growing one, because the wavelet transform is
        O(T log T) and re-running it over the whole island on every tick does not
        stay interactive. The returned ``first_index`` is absolute, so the caller
        can place the result on the video's time axis without knowing the ring.
        """
        n = min(int(n), self._count)
        if n <= 0:
            return np.zeros((0, self.ny, self.nx), np.float32), self.frontier
        a = self.frontier - n
        return self.window(a, self.frontier, channel), a

    def as_channel_dict(self, a: int, b: int) -> dict:
        """``{channel: (b - a, ny, nx)}`` over ``[a, b)`` -- the shape
        ``ChannelData.channels`` wants, so a live span can be handed to the same
        detection and derivation code a windowed extraction feeds."""
        return {c: self.window(a, b, c) for c in self.channels}
