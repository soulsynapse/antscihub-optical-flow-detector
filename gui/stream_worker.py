"""Qt worker threads for the streaming passes, and the continuous one Batch Q
is built on.

:class:`StreamWorker` is the shared base every pass on the live surface uses --
it lived in ``live_scalogram_surface`` until this module existed, and moved here
so :class:`LiveStreamWorker` could reuse it without importing a 1500-line GUI
file (and without that file importing back).

:class:`LiveStreamWorker` is the new one. It owns a
:class:`core.stream_buffer.StreamBuffer` and fills it from Batch P's
``stream_channel_planes``, publishing progress instead of a result. The buffer
is deliberately not internally synchronised (see its module docstring), so the
threading contract here is the whole point of the class:

**The buffer is touched on the worker thread only.** ``request_latest`` is
called from the GUI thread and does nothing but park a request under a mutex.
The decode loop picks it up between frames, copies the window on its own thread,
and hands it back by queued signal. That is why there is no ``buffer`` property:
exposing it would put the check-then-copy race in :meth:`StreamBuffer.window`
back within reach of any caller, and the resulting symptom -- wavelet rows
belonging to a different point in the clip, only under load -- would be blamed on
the transform rather than on the read.

**Newest request wins; there is no queue.** The consumer is a repaint timer, so
a backlog of stale windows is strictly worse than a dropped one: every entry in
it would be redrawn and immediately superseded, and the plot would lag further
behind the further behind it already was.

**The DETECTOR runs here too, not on the GUI thread.** ``request_detect`` parks a
detection request the same way, and the transform runs between decoded frames.
Two reasons, and the second is the load-bearing one. The transform is
O(F.T log T.B), so running it in a signal handler would stutter the UI at exactly
the rate the live view updates. More importantly the display path DROPS windows
while the explorer is still transforming the previous one -- correct for a
display, fatal for an accumulator, because the whole-video track would then be
left with holes wherever the GUI happened to be busy, and a hole is indis-
tinguishable from a stretch the detector examined and found quiet. Detection
coverage must not depend on how the display is keeping up.
"""
from __future__ import annotations

import contextlib
import math
import time

import numpy as np
from PyQt6.QtCore import QMutex, QMutexLocker, QThread, pyqtSignal

from core.detection import region_blocks_and_grid
from core.stream_buffer import StreamBuffer
from core.tensor_channels import stream_channel_planes
from core.wavelet import band_indices, morlet_band_power

# How often the frontier is published, at most. Per-frame emission would post ~80
# queued events a second to paint a number that is only legible at a few hertz.
# The final frontier is always emitted regardless of this, so "done" is exact.
_PUBLISH_HZ = 10.0

# Settling time of the reported processing rate, in SECONDS of wall clock.
#
# Deliberately a time constant and not a per-frame alpha. A fixed per-frame alpha
# makes the settling time scale with the frame rate, which is backwards for this
# particular readout: the rate exists to tell the user the frontier is falling
# behind playback, and the slow configuration is the one that needs to say so
# fastest. Measured on GX010047c2 (todo.md Batch Q), `intensity`+`change` runs
# ~80 fps and adding `appearance` drops it to ~18 fps -- so a per-frame alpha
# would have announced the 18 fps case, the one that matters, four times slower
# than the 80 fps case it does not need to announce at all.
_RATE_TAU_S = 0.25


class Cancelled(Exception):
    """Raised inside a worker's progress callback to unwind a cancelled pass."""


class StreamWorker(QThread):
    """Base for the streaming passes, giving all of them a uniform cancel path.

    ``cancel()`` is called from the GUI thread and only sets a flag; the worker
    notices at its next progress tick (every 20 frames in ``_stream_channels``)
    and raises, which unwinds through that function's ``finally`` and releases
    the decoder. Each pass therefore ends on exactly one of ``done`` / ``failed``
    / ``cancelled``, so the GUI has a single place to restore its buttons.
    """
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    progress = pyqtSignal(int, int)  # (frames done, total) during extraction

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def is_cancelled(self) -> bool:
        return self._cancel

    def _tick(self, done: int, total: int):
        """Progress callback handed to the extractor; doubles as the cancel poll
        since it is the only place the long pass calls back into us."""
        if self._cancel:
            raise Cancelled
        self.progress.emit(done, total)

    def run(self):
        try:
            self._run()
        except Cancelled:
            self.cancelled.emit()
        except Exception as e:                     # surface any extraction error
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _run(self):
        raise NotImplementedError


class LiveStreamWorker(StreamWorker):
    """Fill a ring buffer forward from ``plan.start``, publishing as it goes.

    One instance is one island (todo.md Batch Q): a seek past the frontier
    abandons this worker and starts another at the new position, rather than
    backfilling. So nothing here knows about gaps, and the span the buffer
    reports is always exactly one contiguous run.

    ``capacity`` is in frames and comes from the caller, computed from the same
    ``plan`` this worker runs -- see :func:`core.stream_buffer.capacity_for_budget`.
    Sizing it here from a byte budget instead would put the geometry arithmetic in
    two places, which is the duplication ``ChannelPlan`` exists to remove.
    """
    # (buffer start, frontier, processed frames per second). The rate is here
    # rather than derivable by the GUI because only this thread sees the frame
    # timings; a rate measured from signal arrivals would measure the event loop.
    advanced = pyqtSignal(int, int, float)
    # {"first": absolute index of row 0, "channels": {name: (T, ny, nx)},
    #  "frontier": int, "token": whatever the requester passed}
    window_ready = pyqtSignal(object)
    # {"first": absolute index of row 0, "band_power": (T, B), "frontier": int,
    #  "token": ...}. The transform only -- the value/count bands and the
    #  detection window are NOT applied here, because those re-derive from
    #  retained band power without a fresh pass and belong wherever that
    #  retention lives (core.live_track). Emitting a finished gate would bake a
    #  threshold into the accumulator and force a re-stream to change it.
    detected = pyqtSignal(object)

    def __init__(self, video_path: str, plan, capacity: int, *,
                 clip_paths=None, publish_hz: float = _PUBLISH_HZ, parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._plan = plan
        self._capacity = int(capacity)
        self._clip_paths = clip_paths
        self._min_publish_dt = 1.0 / max(publish_hz, 1e-6)
        self._lock = QMutex()
        self._pending = None            # the one outstanding window request
        self._pending_detect = None     # the one outstanding detection request
        # Kept so the post-loop detection can re-run the last request's
        # parameters; see _serve_detect(final=True).
        self._last_detect = None
        self._rate = 0.0
        # Written on the worker thread before the final signal, read by the GUI
        # after it. Carries `truncated`, which is the difference between "the
        # clip ended" and "the decoder stopped early" -- a data loss the surface
        # must not render as a completed island.
        self.pass_meta: dict | None = None

    # -- called from the GUI thread ------------------------------------------
    def request_latest(self, n: int, channels, token=None) -> None:
        """Ask for the trailing ``n`` frames of ``channels``, served on the
        worker thread and returned by :attr:`window_ready`.

        Returns nothing: the answer cannot be produced here without reading the
        buffer from the wrong thread, which is the one thing this class exists
        to prevent. A request parked while another is still unserved replaces it.
        """
        with QMutexLocker(self._lock):
            self._pending = (int(n), tuple(channels), token)

    def request_detect(self, n: int, channel: str, meta: dict, region_index: int,
                       freqs, freq_band_hz, token=None) -> None:
        """Ask for Morlet band power over the trailing ``n`` frames of one
        channel's region columns, served on the worker thread.

        Parked rather than queued, exactly like :meth:`request_latest`: a
        backlog of transforms would be work done on spans the frontier has
        already left behind, while the newest span went unexamined.

        ``n`` should overlap the previous request by at least twice the cone of
        influence. Consecutive non-overlapping windows would leave a
        permanently-unexamined seam at every join, because the accumulator drops
        each window's contaminated edges -- and a seam that no later window
        covers is a gap the strip is obliged to paint as unexamined forever.
        """
        with QMutexLocker(self._lock):
            self._pending_detect = (int(n), str(channel), meta,
                                    int(region_index), np.asarray(freqs, float),
                                    tuple(freq_band_hz), token)

    # -- worker thread -------------------------------------------------------
    def _run(self):
        plan = self._plan
        buf = StreamBuffer(sorted(plan.want), plan.ny, plan.nx,
                           capacity=self._capacity, start=plan.start)
        gen = stream_channel_planes(self._video_path, plan,
                                    clip_paths=self._clip_paths)
        last_publish = 0.0
        prev_t = time.monotonic()
        # Driven by hand rather than with a `for` so the generator's RETURN value
        # (the pass metadata, delivered on StopIteration) is not thrown away --
        # `truncated` in particular. `closing` releases the decoder on the cancel
        # path, where the generator is abandoned partway through.
        with contextlib.closing(gen):
            while True:
                if self._cancel:
                    raise Cancelled
                try:
                    i, planes = next(gen)
                except StopIteration as stop:
                    self.pass_meta = stop.value
                    break
                buf.append(i, planes)

                now = time.monotonic()
                dt = now - prev_t
                prev_t = now
                if dt > 0:
                    inst = 1.0 / dt
                    # alpha from elapsed time, so the time constant is _RATE_TAU_S
                    # regardless of how fast frames arrive.
                    alpha = 1.0 - math.exp(-dt / _RATE_TAU_S)
                    self._rate = (inst if self._rate <= 0.0 else
                                  self._rate + alpha * (inst - self._rate))
                if now - last_publish >= self._min_publish_dt:
                    last_publish = now
                    self._publish(buf)
                self._serve_pending(buf)
                self._serve_detect(buf)

        # Always publish the final state, whatever the throttle last allowed:
        # the difference between "processing" and "done" is this emission, and
        # dropping it would leave the surface claiming the frontier is short of
        # where the pass actually reached.
        self._publish(buf)
        self._serve_pending(buf)
        # And always run one last detection over the tail. Without it the final
        # seconds of every pass stay unexamined: the last request was serviced
        # some frames back, and its trailing cone was trimmed off on the way in.
        self._serve_detect(buf, final=True)
        self.done.emit(self.pass_meta)

    def _publish(self, buf: StreamBuffer) -> None:
        """Announce the island. ``progress`` is emitted alongside ``advanced``
        because every other worker in this module has one and the live surface
        already connects it -- a streaming worker that inherited the signal and
        left it silent would look like a stalled pass rather than a signal
        nobody wired."""
        self.advanced.emit(buf.start, buf.frontier, self._rate)
        self.progress.emit(buf.frontier - self._plan.start, self._plan.n)

    def _serve_pending(self, buf: StreamBuffer) -> None:
        with QMutexLocker(self._lock):
            req = self._pending
            self._pending = None
        if req is None:
            return
        n, channels, token = req
        out, first = {}, buf.frontier
        for c in channels:
            out[c], first = buf.latest(n, c)
        if not any(v.shape[0] for v in out.values()):
            # Nothing has arrived yet (reachable at the tail of the pass when the
            # plan clamped to zero frames). Emitting a zero-row window would hand
            # a consumer an empty time axis that looks exactly like a measured
            # one -- the same "empty is not vacuously valid" rule StreamBuffer
            # .covers enforces. The request stays dropped, not deferred: the next
            # tick asks again.
            return
        self.window_ready.emit({"first": first, "channels": out,
                                "frontier": buf.frontier, "token": token})

    def _serve_detect(self, buf: StreamBuffer, final: bool = False) -> None:
        """Transform the trailing detection window, on this thread.

        ``final`` re-uses the last request's parameters after the decode loop has
        ended, so the tail of the pass is examined rather than left as a gap
        whose only cause was where the timer happened to last fire.
        """
        with QMutexLocker(self._lock):
            req = self._pending_detect
            self._pending_detect = None
            if req is None and final:
                req = self._last_detect
            if req is not None:
                self._last_detect = req
        if req is None:
            return
        n, channel, meta, region_index, freqs, band, token = req
        if channel in buf.channels:
            planes, first = buf.latest(n, channel)
        else:
            # A DERIVED channel (the velocity gradient) is not streamed into the
            # buffer; build it here from the base fields the pass DID stream (u,
            # v), which are present because _channels_wanted requested them when
            # this channel was selected. Same window, so latest() returns each
            # base aligned to the same first index.
            from core.channels import REGISTRY, evaluate, needs_for
            base = sorted(needs_for({channel}))
            if channel not in REGISTRY or not base or \
                    any(b not in buf.channels for b in base):
                return
            fields, first = {}, None
            for b in base:
                fields[b], first = buf.latest(n, b)
            planes = evaluate(fields, meta, [channel])[channel]
        if planes.shape[0] < 2:
            # One frame is not a time series. Transforming it would return a
            # padding response and the accumulator would record it as examined.
            return
        blocks, _grid = region_blocks_and_grid(meta, planes, region_index)
        i, j = band_indices(freqs, band[0], band[1])
        bp = morlet_band_power(blocks, float(meta["fps"]), freqs, i, j)
        self.detected.emit({"first": int(first), "band_power": bp,
                            "frontier": buf.frontier, "token": token})
