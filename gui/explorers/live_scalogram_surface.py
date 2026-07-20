"""The live tuning surface: preprocessing knobs feeding the scalogram directly.

This is the dissolved preprocessing tab. Instead of configuring a flow pass and
waiting for a cache, you press Play and watch the structure-tensor scalogram /
detection stack respond as you drag downsample, block size, and normalization.
There is no flow solve and no cache write.

**There is exactly one pass, and Play runs it.** The surface used to have two:
a bounded windowed *extract* that decoded a fixed span and returned a result,
and a *live* pass that runs forward to the end of the clip publishing a moving
frontier. The extract has been retired. It existed because the live pass did
not, and once the live pass could build and feed the explorer itself, keeping
both meant two ways to fill the same widgets, two definitions of "the current
window", and a seek that silently chose between them. Play (or Space) starts
and stops the one pass; every other control either replans it or does nothing
until it is running.

The playhead sits at the MIDDLE of the served window, not at the frontier --
see ScalogramExplorer.follow_center. The pass runs ahead of the reading
position by half a window, so every plot shows computed footage on both sides
of the cursor and the data scrolls under a fixed landmark.

Because ``downsample`` and ``block_size`` change the block *geometry* (not just
pixels), a knob change replans the pass and rebuilds the ScalogramExplorer from
a fresh ChannelData -- the explorer's constructor already derives every
geometry-dependent structure consistently, which is far safer than mutating
grid/regions/cube-cache in place. The tuning context you care about (selected
channel, cursor, detection window, frequency band) is captured off the old
explorer and re-applied to the new one.

``normalize`` is a per-frame pixel op (z-score is ~invariant for tensor_speed,
reshapes change/intensity); ``block_size`` is a geometry op whose expensive
per-pixel tensor solve is actually block-size independent. That independence
used to fund a block=1 pixel cache, so a Block change could re-reduce instead of
re-extracting; it went with the extract, since a live pass replans rather than
re-runs a window and there is nothing for such a cache to save.
"""
from __future__ import annotations

import time
from dataclasses import replace
from enum import Enum

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                             QPushButton, QSpinBox, QVBoxLayout, QWidget)

from core.channel_source import (LIVE_CHANNELS, ChannelData,
                                 live_channel_source, synth_live_meta)
from core.config import FlowConfig, PipelineConfig, PreprocessConfig
from core.cost_model import CostModel, PassSample
from core.scale_sweep import measure_scale
from core.replicates import build_layout, in_tile_order

# Fields that define a replicate's geometry. A surface is rebuilt outright when
# these change (the tab keys it on a geometry hash), so refreshing metadata must
# never copy them across -- see refresh_replicate_metadata.
_GEOMETRY_KEYS = frozenset({"frac", "id"})
from core.detection import detect_channel_region, region_grid_from_meta
from core.live_track import (TrackStamp, WholeVideoTrack, band_power_bytes,
                             coi_trim)
from core.stream_buffer import MIN_CAPACITY, capacity_for_budget
from core.sysmem import budget_bytes
from core.tensor_channels import plan_channel_stream
from core.video import VideoSource
from core.wavelet import default_freqs
from gui.explorers.detection_timeline import DetectionNavigator
from gui.explorers.scalogram_explorer import ScalogramExplorer
# The worker base moved to gui/stream_worker.py so the continuous worker could
# share it without importing this module. The base and Cancelled are re-bound to
# the private names the rest of this file, and its tests, already use.
from gui.stream_worker import Cancelled as _Cancelled
from gui.stream_worker import LiveStreamWorker
from gui.stream_worker import StreamWorker as _StreamWorker
from gui.tuning_store import load_tuning, save_tuning

# Downsampling is opt-in and off by default (todo.md Batch K), so there is no
# longer an "auto" sentinel on the scale knob -- 1.0 is a real, and the default,
# value. This is the smallest scale the spinbox offers.
_MIN_DS = 0.05

# Block spinbox sentinel: 0 means "track the scale", i.e. leave block_size None
# and let FlowConfig hold the grid fixed in source pixels.
_AUTO_BLOCK = 0

# Idle labels for the two action buttons; each becomes "Stop" while its own
# pass runs (see _set_busy).
_PROCESS_TEXT = "Process whole video ▶"
_LIVE_TEXT = "Play ▶"

# How often the GUI asks the stream worker for a trailing window.
#
# 10 Hz, raised from 1.0 once the per-plot costs were actually measured
# (FINDINGS §24). The old value assumed the consumer was an O(T log T) transform
# over the window; it is not. The plots this tick drives are all per-FRAME --
# pooled mean ~3 ms, pooled Morlet ~8 ms, trace and sweep ~1 ms, together ~12 ms
# at T=30k -- because the Morlet here transforms a single (T,) pooled series and
# carries none of the B factor. The genuinely expensive thing, the per-BLOCK
# (F,T,B) cube at 0.4-6 s, is NOT driven by this tick: it runs at its own pace on
# its own thread and its density heatmap is drawn at whatever span it has
# actually covered.
#
# The ~8x headroom is deliberate and load-bearing: §24's budget holds only while
# the per-block window stays bounded. If a future change lets it grow toward clip
# length the fast path reaches ~155 ms and this rate fails SILENTLY -- no error,
# just a surface that stops keeping up.
_WINDOW_REQUEST_HZ = 10.0

# Fewest frames a served window may have before it is put on screen.
#
# The worker only refuses a window of ZERO rows, which is one frame short of the
# guard this needs: the first tick of a pass routinely lands 1-2 frames, and the
# explorer derives its whole time axis from whatever it is handed -- a Morlet
# transform over a single sample, a detection window D of 1, a scrub with one
# position. All of that renders, and none of it means anything, which is the
# failure mode this codebase keeps naming (a plausible-looking plot of nothing).
# The value is core.stream_buffer.MIN_CAPACITY for the same stated reason: below
# it there is not enough history for the slowest wavelet scale to mean anything.
#
# Applied as min(requested, floor), never as a flat floor -- a request SHORTER
# than this is a deliberately short window, and refusing to draw it at all would
# be worse than drawing it late.
_MIN_LIVE_FRAMES = MIN_CAPACITY

# How often a detection window is requested from the stream worker. Slower than
# the display tick because the transform is the expensive half and runs on the
# decode thread: asking faster does not fill the track faster, it just steals
# frames from the frontier.
_DETECT_REQUEST_HZ = 0.5

# How much a detection window must overlap the previous one, as a multiple of
# the cone of influence trimmed off each end. The accumulator discards `coi`
# frames at both edges of every window, so consecutive requests that only just
# touch would leave a 2*coi seam unexamined at every join -- a permanent gap
# whose only cause is the request cadence. Two cones of overlap means each seam
# lands in the interior of the next window.
_DETECT_OVERLAP_COI = 2.5

# Whole-clip band power is retained so a threshold re-tune is instant rather than
# a re-stream (todo.md: the value/count bands and the detection window are
# deliberately NOT part of a TrackStamp). It is ~45 MB at 30k frames and 377
# blocks, which is nothing -- but a region pointed at a whole 5.3K frame at a
# fine block runs to gigabytes, so it is priced against this cap and declined
# rather than attempted.
_TRACK_BP_CAP = 1024 ** 3

# Consecutive dropped detection windows before the surface stops treating them
# as incidental and says the pass is producing nothing. One drop is ordinary (a
# geometry change races a request in flight); a run of them means the stream and
# the tuned selection disagree, and the symptom -- a strip that never fills
# while progress looks healthy -- is indistinguishable from a quiet clip.
_DETECT_DROP_ALARM = 3

# Fraction of free RAM the ring buffer may take, and its floor/cap. The ring IS
# the only copy of the streamed channels, so there is no peak multiplier to
# allow for here.
_RING_BUDGET_FRACTION = 0.25
_RING_BUDGET_FLOOR = 512 * 1024 ** 2
_RING_BUDGET_CAP = 8 * 1024 ** 3

# Longest side of the source crop the render strip works from. A replicate box
# can be the whole frame, and squeezing 5312 px into a ~108 px tile makes every
# scale look identical -- the display becomes the bottleneck instead of the
# working resolution, which is the one thing the strip exists to show. Cropping
# to a centre window keeps the comparison at a magnification where the
# difference between scales is visible.
_RENDER_MAX_PX = 420


class _Busy(Enum):
    """Which pass owns the decoder. Named rather than bare strings because
    _set_busy silently disables both buttons for any value it does not
    recognise, which a mistyped literal would make unrecoverable."""
    PROCESS = "process"
    SWEEP = "sweep"
    STREAM = "stream"


class _RenderWorker(_StreamWorker):
    """Render one replicate box at every candidate scale, off the GUI thread.

    Cheap in work (one frame pair, then a resize per scale) and expensive in
    latency: seeking long-GOP footage decodes forward from the preceding
    keyframe, which is easily a second on 5.3K. Doing that on the GUI thread
    would freeze the dialog on open and again on every replicate change.

    Unlike the streaming passes this does NOT contend for the decoder the others
    hold -- it opens its own ``VideoSource``, reads two frames and closes -- so
    it is deliberately not gated behind ``_sweep_ready``. Blocking it while a
    live pass runs would leave the strip empty in the ordinary case of opening
    the window straight after starting one.
    """

    def __init__(self, video_path, box, frame_idx, scales, pre_cfg, parent=None):
        super().__init__(parent)
        self._args = (video_path, box, frame_idx, list(scales), pre_cfg)

    def _run(self):
        from core.scale_render import fit_box_to, render_box_at_scales
        video_path, box, frame_idx, scales, pre_cfg = self._args
        renders = render_box_at_scales(
            video_path, fit_box_to(box, _RENDER_MAX_PX), frame_idx, scales,
            base_cfg=pre_cfg)
        self.done.emit(renders)


class _ProcessWorker(_StreamWorker):
    """The whole-video commit: stream the WHOLE clip's channels once, then run the
    tuned detector over the selected region. No flow, no cache -- just the
    detection track. This is the one expensive pass, paid after tuning."""
    phase = pyqtSignal(str)          # a coarse phase change (e.g. detection start)

    def __init__(self, video_path, cfg, replicates, dims, region_index, params,
                 parent=None):
        super().__init__(parent)
        self._args = (video_path, cfg, replicates, dims, region_index, params)
        # ``(covered, requested)`` when the decode ended early, else None. Read by
        # the done handler; see _run.
        self.short: tuple[int, int] | None = None

    def _run(self):
        video_path, cfg, reps, dims, region_index, params = self._args
        w, h, fps, fc = dims
        # ONLY the channel being detected on. The whole-clip pass used to compute
        # all four regardless, and the other three were pure waste -- nothing
        # downstream of here reads them, since detect_channel_region takes a
        # single channel_attr. On the detection default (``change``) this skips
        # the flow solve, the appearance residual and the min-eigen; the live
        # preview deliberately still computes all four, because there a channel
        # toggle must stay instant instead of triggering a replan.
        cd = live_channel_source(
            video_path, cfg, reps, start=0, n=None, width=w, height=h,
            fps=fps, frame_count=fc, channels=[params["channel_attr"]],
            progress=self._tick)
        # A decode that ended early yields a SHORT track, not a padded one, so
        # the detector below runs over less video than the user asked for and
        # every "no detection here" past the cut point is unexamined rather than
        # examined-and-clear. The result object carries no notion of coverage, so
        # the shortfall is recorded on the worker for the done handler to report
        # -- silently returning a partial pass as a whole-video pass is the exact
        # false negative the truncation trim exists to prevent, and trimming
        # without reporting only moves where it hides.
        self.short = (int(cd.n_frames), int(fc)) if cd.meta.get("truncated") \
            else None
        # Extraction (the whole-clip per-pixel tensor stream) is done; the band
        # power + gate that follow are a much smaller slice of the wall time,
        # but the phase note keeps the status honest across the handoff.
        # detect_channel_region has no progress hook, so this is the last cancel
        # point -- a stop during detection lands here rather than mid-detector.
        if self._cancel:
            raise _Cancelled
        self.phase.emit("running detector")
        res = detect_channel_region(
            cd, region_index, params["channel_attr"],
            freqs=default_freqs(fps), freq_band_hz=params["freq_band_hz"],
            value_band=params["value_band"], count_band=params["count_band"],
            detect_window=params["detect_window"], centered=params["centered"])
        if self._cancel:
            raise _Cancelled
        self.done.emit(res)


class _SweepWorker(_StreamWorker):
    """The downsample dialog's sweep: one timed extraction pass per scale.

    Each pass is a cost sample taken at the block a production run would use --
    a production run reduces to the display block as it goes, so a sweep row is
    the one place that prices a pass in isolation, where ``block_reduce`` is
    62% of the wall time against ~15% in production (11.0 s against 4.2 s at
    scale 1.0, measured).

    No detector runs. An earlier version ran the tuned detector at each scale and
    reported events kept and lost, which read as sensitivity evidence but
    measured threshold drift -- see ``core/scale_sweep.py``. Dropping it also
    means the sweep needs no tuned explorer and no selected replicate, so it can
    run the moment the window opens.

    A scale that fails does not abort the sweep: a row reporting the error is
    more useful than losing the scales that would have run after it.
    """
    row = pyqtSignal(object)               # a completed ScalePass
    row_failed = pyqtSignal(float, str)
    scale_started = pyqtSignal(float, int, int)     # (scale, index, total)

    def __init__(self, video_path, cfg, replicates, dims, start, n, scales,
                 parent=None):
        super().__init__(parent)
        self._args = (video_path, cfg, replicates, dims, start, n, list(scales))

    def _run(self):
        video_path, cfg, reps, dims, start, n, scales = self._args
        for i, s in enumerate(scales):
            if self._cancel:
                raise _Cancelled
            self.scale_started.emit(float(s), i + 1, len(scales))
            # Only the scale moves; the block comes from the user's own flow
            # config, so `auto` tracks it and a pinned block stays pinned.
            cfg_s = replace(cfg, preprocess=replace(cfg.preprocess,
                                                    downsample=float(s)))
            try:
                sp = measure_scale(video_path, cfg_s, reps, dims=dims,
                                   start=start, n=n, progress=self._tick)
            except _Cancelled:
                raise
            except Exception as e:
                self.row_failed.emit(float(s), f"{type(e).__name__}: {e}")
                continue
            self.row.emit(sp)
        self.done.emit(None)


class LiveScalogramSurface(QWidget):
    # A calibration measured in the downsample window. This surface works from
    # AppState's *copies* of the replicate dicts, so it cannot persist one
    # itself; the owning tab relays this to AppState, which the replicate tab --
    # owner of the authoritative list and its sidecar -- writes to disk.
    calibration_changed = pyqtSignal(int, object)   # (replicate id, fields)

    def __init__(self, video_path: str, replicates: list[dict],
                 base_cfg: PipelineConfig | None = None, parent=None,
                 frame_provider=None):
        super().__init__(parent)
        self.video_path = video_path
        self.replicates = list(replicates)
        # Where the rest of the app is parked in the clip, if the owner can tell
        # us. Read on demand rather than tracked: the surface only cares at the
        # moment the user asks to inherit that position.
        self._frame_provider = frame_provider
        cfg = base_cfg or PipelineConfig()

        with VideoSource(video_path) as src:
            info = src.info
        self._dims = (info.width, info.height, float(info.fps),
                      int(info.frame_count))
        self.fps = float(info.fps)
        self.frame_count = int(info.frame_count)

        self._explorer: ScalogramExplorer | None = None
        self._proc_worker: _ProcessWorker | None = None
        # Tuning remembered for THIS clip (see gui/tuning_store). Loaded before
        # the strip is built so the controls come up on it.
        self._saved = load_tuning(video_path)
        # Seeded here so the FIRST live pass opens on the bands that were left
        # here, and consumed by the first _swap_explorer.
        self._pending_state: dict | None = self._saved.get("view") or None
        # Replicate selection travels alongside, not inside, the view state: a
        # rebuild deliberately resets the selection, and only the reopen case
        # should restore it. Consumed by the first _swap_explorer.
        self._pending_region = self._saved.get("region_index")
        # Last view state we managed to capture, for saves that land while the
        # explorer is being rebuilt (there is no live one to read then).
        self._saved_view: dict | None = self._pending_state
        # Progress-readout timing/context, set when each pass starts.
        self._proc_t0 = 0.0
        self._proc_ctx = ""

        # Timed passes, feeding the downsample dialog's cost model. Keyed by
        # (scale, block) because the two are not interchangeable costs: a pass at
        # block=1 does no reduction at all, so mixing those samples with
        # block-reduced ones into one fit would regress across two variables and
        # attribute both to the scale.
        self._cost_samples: dict[tuple[float, int], PassSample] = {}
        # The regime each sample was taken in, keyed the same way: None when the
        # block tracked the scale, the pinned working block otherwise. Recorded
        # from the config that ran rather than inferred by comparing the block to
        # what tracking would have produced -- those coincide at some scale for
        # any pinned block (64 at scale 1.0, 32 at 0.5), and inferring would then
        # file that one pass in the wrong regime and drop it from the fit.
        self._sample_regime: dict[tuple[float, int], int | None] = {}
        self._last_cost_key: tuple[float, int] | None = None
        # The downsample dialog's empirical sweep, while one is running. Held on
        # the surface rather than in the dialog because it owns the decoder and
        # must not overlap the live pass or the whole-video commit.
        self._sweep_worker: _SweepWorker | None = None
        self._sweep_block_intent: int | None = None
        # The dialog's render strip. Its own decoder handle, so it does not
        # contend with the passes above and is not tracked by _Busy.
        self._render_worker: _RenderWorker | None = None
        self._dlg = None

        # -- the continuous pass (todo.md Batch Q) ---------------------------
        # This one runs forward to the end of the clip and
        # publishes a frontier instead of returning a result. It owns the decoder
        # for as long as it runs, so it is tracked by _Busy like the others.
        self._stream_worker: LiveStreamWorker | None = None
        self._ring_budget = budget_bytes(_RING_BUDGET_FRACTION,
                                         _RING_BUDGET_FLOOR, _RING_BUDGET_CAP)
        # The last trailing window the worker served, as it handed it over:
        # {"first", "channels", "frontier", "token"}. Slice 3's scalogram reads
        # this; for now it is what proves the request/serve round trip is live
        # rather than plumbed-but-untravelled, via the status line.
        self._live_window: dict | None = None
        self._live_request_n = 0
        self._stream_truncated = False
        # Bumped on every (re)start. A window request is serviced on the worker
        # thread and arrives by queued signal, so one parked before a restart can
        # land after it -- carrying frames from the abandoned island. The token
        # is what lets the handler tell that case from a current one; without it
        # the surface would render a span from a different set of knobs and look
        # merely stale rather than wrong.
        self._stream_token = 0
        self._stream_plan_obj = None
        self._stream_meta: dict | None = None
        # Where a live pass should resume after a seek stopped it. Consumed by
        # _on_stream_cancelled; None means the stop was a real stop.
        self._restart_stream_at: int | None = None
        # Consecutive detection windows refused for a geometry mismatch; see
        # _DETECT_DROP_ALARM. Reset by any write that lands.
        self._detect_drops = 0
        # Drives request_latest. Started with the pass and stopped with it, so a
        # request is never parked on a worker that is on its way out.
        self._stream_timer = QTimer(self)
        self._stream_timer.setInterval(int(1000 / _WINDOW_REQUEST_HZ))
        self._stream_timer.timeout.connect(self._request_live_window)

        # -- the whole-video detection track ---------------------------------
        # The live axis is the WHOLE VIDEO, not the trailing window: navigating
        # around the clip accumulates parts of one picture. Created here rather
        # than on the first pass so the strip is a working seeker the moment the
        # tab opens, and so a stamp change never has to reconcile "no track yet"
        # with "a track under other settings".
        self._track = WholeVideoTrack(n_frames=self.frame_count, fps=self.fps)
        self._track_grid = None          # (dy, dx, gy, gx) for the clump pass
        # Coalesces value-band drags. Unlike the count band, a value-band change
        # re-runs the per-frame connected-components loop over every retained
        # row -- the one part of a re-tune that is not free at clip length.
        self._retune_debounce = QTimer(self)
        self._retune_debounce.setSingleShot(True)
        self._retune_debounce.setInterval(200)
        self._retune_debounce.timeout.connect(self._on_retune_settled)
        # Drives request_detect, separately from the display tick.
        self._detect_timer = QTimer(self)
        self._detect_timer.setInterval(int(1000 / _DETECT_REQUEST_HZ))
        self._detect_timer.timeout.connect(self._request_detect_window)

        # Coalesce rapid knob edits into a single pass restart on settle. Created
        # before the strip because the controls connect to it.
        #
        # Every knob here is upstream of the block grid, so there is nothing to
        # patch in place: the running pass was planned against the old geometry
        # and has to be replanned. Block used to be the exception -- it could
        # re-reduce a cached block=1 extract -- but that cache existed only to
        # avoid a re-extract, and with the extract path gone Block is just
        # another knob that replans the pass.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(500)
        self._debounce.timeout.connect(self._on_knob_settled)

        self._block_debounce = QTimer(self)
        self._block_debounce.setSingleShot(True)
        self._block_debounce.setInterval(250)
        self._block_debounce.timeout.connect(self._on_knob_settled)

        # Persist the tuning on settle. Longer than either compute debounce
        # because nothing waits on it: the point is one write per edit rather
        # than one per keystroke, and a write that lands after the replan
        # has already started costs nothing.
        self._save_debounce = QTimer(self)
        self._save_debounce.setSingleShot(True)
        self._save_debounce.setInterval(1000)
        self._save_debounce.timeout.connect(self._save_tuning)

        # Focusable so a committed spin-box edit has somewhere to hand focus
        # back to: the main window's Space handler walks up from the focus
        # widget, and this is the level that carries toggle_playback().
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        root = QVBoxLayout(self)
        root.addWidget(self._build_strip(cfg))

        self._host = QWidget()
        self._host_lay = QVBoxLayout(self._host)
        self._host_lay.setContentsMargins(0, 0, 0, 0)
        self._placeholder = QLabel(
            "Press Play ▶ (or Space) to run the live scalogram from the window "
            "start.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color:#8ab; padding:40px;")
        self._host_lay.addWidget(self._placeholder)
        root.addWidget(self._host, 1)

        # The whole-clip strip. ALWAYS visible: it is this tab's position
        # control, not a report of a finished pass, so hiding it until a commit
        # lands would leave the clip with no seeker for the whole tuning
        # session. It carries the clip's length from the start and fills in as
        # detection reaches each part.
        self.navigator = DetectionNavigator()
        self.navigator.set_span(self.frame_count, self.fps)
        self.navigator.focus_requested.connect(self._focus_frame)
        self.navigator.scrubbed.connect(self._on_scrubbed)
        self.navigator.seek_committed.connect(self._on_seek_committed)
        root.addWidget(self.navigator)
        # Nothing runs on show. The old windowed extract auto-fired here because
        # it was bounded -- seconds, then done. A live pass runs to the end of
        # the clip and holds the decoder, so starting one just because a tab
        # became visible is not the same trade. The strip is a working seeker
        # from frame zero regardless; Play is what commits the decoder.

    # -- config strip --------------------------------------------------------
    def _build_strip(self, cfg: PipelineConfig) -> QWidget:
        # A bare container, not a titled QGroupBox: the title + frame inset cost
        # ~1cm of vertical space to say what the tab label already says.
        box = QWidget()
        outer = QVBoxLayout(box)
        outer.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()

        row.addWidget(QLabel("Window start"))
        self.start_slider = QSpinBox()
        self.start_slider.setRange(0, max(0, self.frame_count - 2))
        self.start_slider.setSingleStep(max(1, int(self.fps)))
        self._commit_on_enter(self.start_slider)
        self.start_slider.valueChanged.connect(self._on_window_changed)
        row.addWidget(self.start_slider)
        self.start_lbl = QLabel("0.00 s")
        self.start_lbl.setMinimumWidth(70)
        row.addWidget(self.start_lbl)

        # Carries the playhead over from the Replicates tab: the usual way a
        # window gets chosen is by scrubbing there until something interesting is
        # on screen, and retyping that frame here is the step that loses it.
        self.inherit_btn = QPushButton("Inherit")
        self.inherit_btn.setToolTip(
            "Set the window start to where the Replicates tab is parked.")
        self.inherit_btn.clicked.connect(self._inherit_start)
        self.inherit_btn.setVisible(self._frame_provider is not None)
        row.addWidget(self.inherit_btn)

        row.addWidget(QLabel("Length"))
        self.len_spin = QDoubleSpinBox()
        max_len = min(60.0, self.frame_count / max(self.fps, 1e-6))
        self.len_spin.setRange(0.2, max(0.2, max_len))
        self.len_spin.setValue(min(10.0, max(0.2, max_len)))
        self.len_spin.setSuffix(" s")
        self._commit_on_enter(self.len_spin)
        self.len_spin.valueChanged.connect(self._on_window_changed)
        row.addWidget(self.len_spin)

        row.addSpacing(12)
        row.addWidget(QLabel("Downsample"))
        self.ds_spin = QDoubleSpinBox()
        self.ds_spin.setRange(_MIN_DS, 1.0)
        self.ds_spin.setSingleStep(0.05)
        self.ds_spin.setDecimals(3)
        self.ds_spin.setValue(cfg.preprocess.resolve_downsample())
        self.ds_spin.setToolTip(
            "Working scale. 1.0 (no downsampling) is the default and is NOT a "
            "quality setting to trade away lightly: downsampling decides what "
            "is detectable, and whether a coarser scale still resolves your "
            "behaviour has to be demonstrated for that behaviour and species. "
            "It is offered because it is often what makes a large project "
            "computationally feasible — it shortens every per-pixel stage.")
        self._commit_on_enter(self.ds_spin)
        self.ds_spin.valueChanged.connect(self._debounce.start)
        self.ds_spin.valueChanged.connect(self._sync_block_auto_text)
        row.addWidget(self.ds_spin)

        # The knob alone gets refused on principle (see gui/downsample_dialog.py):
        # this is where its cost and its consequence are made legible.
        self.ds_help_btn = QPushButton("…")
        self.ds_help_btn.setFixedWidth(26)
        self.ds_help_btn.setToolTip(
            "What downsampling gains and costs: projected time and storage for "
            "your corpus at each scale, priced from passes measured on this "
            "machine, with the knee marked.")
        self.ds_help_btn.clicked.connect(self._open_downsample_dialog)
        row.addWidget(self.ds_help_btn)

        row.addWidget(QLabel("Block"))
        self.block_spin = QSpinBox()
        self.block_spin.setRange(_AUTO_BLOCK, 64)
        self.block_spin.setValue(cfg.flow.block_size or _AUTO_BLOCK)
        self.block_spin.setToolTip(
            "Working pixels per block. Sets the scalogram grid (and cube "
            "memory); the per-pixel tensor solve is block-size independent, so "
            "a change here replans the running pass. On auto the block "
            "tracks the scale, holding the "
            "grid fixed in source pixels so that moving Downsample changes "
            "compute only — not localization.")
        self._commit_on_enter(self.block_spin)
        self.block_spin.valueChanged.connect(self._block_debounce.start)
        row.addWidget(self.block_spin)
        self._sync_block_auto_text()

        row.addWidget(QLabel("Normalize"))
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["off", "zscore", "clahe"])
        self.norm_combo.setCurrentText(cfg.preprocess.normalize)
        self.norm_combo.setToolTip(
            "Upstream per-frame pixel op (replans the pass). z-score is ~invariant "
            "for tensor_speed; reshapes change/intensity. CLAHE has a known "
            "replicate-edge artifact.")
        self.norm_combo.currentTextChanged.connect(self._debounce.start)
        row.addWidget(self.norm_combo)

        # Everything to its left, plus the detection window and the three
        # threshold bands in the explorer below, back to how this surface opens
        # on a clip it has never seen. Sits at the end of the knob run rather
        # than with Play/Process: it is the last member of that group, not a
        # third action on the video.
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setToolTip(
            "Restore the defaults: window, downsample, block and normalize "
            "above, and — in the panel below — the detection window D and all "
            "three detection bands (frequency, channel value, block count). "
            "Replans a running pass once. The selected channel and "
            "replicate are kept.")
        self.reset_btn.clicked.connect(self.reset_all)
        row.addWidget(self.reset_btn)

        # The defaults the button restores, snapshotted from the values just
        # built -- BEFORE any remembered tuning is applied over them, so Reset
        # goes to the program's defaults and not to whatever this clip happened
        # to be left on last session.
        self._strip_defaults = self._strip_values()
        self._apply_strip(self._saved.get("strip") or {})
        # Wired after the restore: the restore is not an edit and must not
        # schedule a write of what it just read.
        for sig in (self.start_slider.valueChanged, self.len_spin.valueChanged,
                    self.ds_spin.valueChanged, self.block_spin.valueChanged,
                    self.norm_combo.currentTextChanged):
            sig.connect(lambda *_: self._save_debounce.start())

        # Two stacked, right-aligned status lines sit inline to the left of the
        # action buttons and take the row's slack: the live / whole-video compute
        # on top, and the hosted explorer's graph-compute line just below it.
        # Darker orange + larger so they read against the light control strip.
        status_css = ("color:#c2691a; font-family:Consolas; font-size:12px; "
                      "font-weight:600;")
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(status_css)
        self.status_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.graph_status_lbl = QLabel("")
        self.graph_status_lbl.setStyleSheet(status_css)
        self.graph_status_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        statuscol = QVBoxLayout()
        statuscol.setSpacing(1)
        statuscol.addWidget(self.status_lbl)
        statuscol.addWidget(self.graph_status_lbl)
        row.addLayout(statuscol, 1)
        row.addSpacing(8)
        self.live_btn = QPushButton(_LIVE_TEXT)
        self.live_btn.setToolTip(
            "Run the surface forward from the window start to the end of the "
            "clip (Space does the same). The playhead sits half a window behind "
            "the frontier, so the plots show computed footage on both sides of "
            "it. The readout reports the achieved rate against playback: below "
            "1.0× the frontier is falling behind, and the retained history is "
            "bounded by a ring buffer, so a long run drops its oldest frames.")
        self.live_btn.clicked.connect(self._on_live_clicked)
        row.addWidget(self.live_btn)
        self.process_btn = QPushButton(_PROCESS_TEXT)
        self.process_btn.setToolTip(
            "Run the tuned detector over the WHOLE clip (one streaming pass, no "
            "flow, no cache) and navigate the detections below.")
        self.process_btn.clicked.connect(self._on_process_clicked)
        row.addWidget(self.process_btn)
        outer.addLayout(row)

        self._sync_window_label()
        # denoise is deliberately absent: it is stateful from frame zero and
        # cannot be reproduced for an arbitrary mid-clip window (forced off).
        return box

    def _commit_on_enter(self, spin):
        """Make a typed spin box commit on Enter, not per keystroke.

        Keyboard tracking off means valueChanged fires once the edit is
        committed (Enter, focus-out, or a step), so typing "0.25" no longer
        arms three replans on the way through 0, 0.2, 0.25. On commit the box
        also hands focus back to the surface: while its line edit holds focus
        it eats Space, and Space is the play/pause the user reaches for next.
        """
        spin.setKeyboardTracking(False)
        spin.editingFinished.connect(self._release_edit_focus)

    def _release_edit_focus(self):
        # editingFinished also fires on focus-out, where the box no longer has
        # focus and something else just took it. Only Enter leaves the sender
        # focused, and only then should focus come back here.
        spin = self.sender()
        if spin is not None and spin.hasFocus():
            self.setFocus(Qt.FocusReason.OtherFocusReason)

    def _on_window_changed(self, *_):
        self._sync_window_label()
        self._debounce.start()

    def _inherit_start(self):
        """Move the window start to the app's current playhead frame."""
        if self._frame_provider is None:
            return
        frame = int(self._frame_provider())
        # Clamp rather than ignore an out-of-range playhead: the spin box range
        # stops two frames short of the end, and landing on the last usable
        # window is a better answer than silently doing nothing.
        frame = max(self.start_slider.minimum(),
                    min(self.start_slider.maximum(), frame))
        # setValue is a no-op when it matches, so a second press costs nothing;
        # when it differs, valueChanged carries the replan as usual.
        self.start_slider.setValue(frame)

    def _sync_window_label(self, scrub: int | None = None):
        """The window-start readout. ``scrub`` shows a position being dragged on
        the strip WITHOUT moving the slider: the drag has not committed, and
        writing it into the slider would replan the pass per pixel of drag."""
        if scrub is not None:
            self.start_lbl.setText(f"{scrub / self.fps:.2f} s ⟵")
            return
        self.start_lbl.setText(f"{self.start_slider.value() / self.fps:.2f} s")

    def _sync_block_auto_text(self):
        """Show what "auto" currently resolves to, since it moves with the scale."""
        tracked = FlowConfig(block_size=None).resolve_block_size(
            float(self.ds_spin.value()))
        self.block_spin.setSpecialValueText(f"auto ({tracked})")

    # -- remembered tuning / reset -------------------------------------------
    def _strip_values(self) -> dict:
        """The strip's knobs as plain numbers, for saving and for Reset."""
        return {"start": int(self.start_slider.value()),
                "length_s": float(self.len_spin.value()),
                "downsample": float(self.ds_spin.value()),
                "block": int(self.block_spin.value()),
                "normalize": self.norm_combo.currentText()}

    def _apply_strip(self, values: dict) -> None:
        """Push ``values`` onto the strip without arming a replan.

        Signals are blocked throughout: every setter here is wired to a
        debounce, so applying five of them live would queue five passes to
        arrive at one state. Callers run the single pass themselves.

        Out-of-range values are clamped by the widgets (a remembered window
        start from a longer clip, a block above this build's ceiling), and an
        unknown normalize mode is dropped rather than forced -- a sidecar is a
        file on disk and may not have been written by this version.
        """
        widgets = (self.start_slider, self.len_spin, self.ds_spin,
                   self.block_spin, self.norm_combo)
        for w in widgets:
            w.blockSignals(True)
        try:
            if "start" in values:
                self.start_slider.setValue(int(values["start"]))
            if "length_s" in values:
                self.len_spin.setValue(float(values["length_s"]))
            if "downsample" in values:
                self.ds_spin.setValue(float(values["downsample"]))
            if "block" in values:
                self.block_spin.setValue(int(values["block"]))
            norm = values.get("normalize")
            if norm is not None and self.norm_combo.findText(str(norm)) >= 0:
                self.norm_combo.setCurrentText(str(norm))
        except (TypeError, ValueError):
            pass                # a malformed sidecar leaves the strip as it was
        finally:
            for w in widgets:
                w.blockSignals(False)
        # Both labels read off knobs that just moved underneath them, and
        # "auto" in particular resolves against the scale.
        self._sync_window_label()
        self._sync_block_auto_text()

    def reset_all(self) -> None:
        """Reset button: the strip to its defaults and the explorer's detection
        tuning with it, then replan a running pass onto them.

        The explorer is reset FIRST because a restart captures its view state to
        carry across the rebuild -- reset after, and the pass now in flight would
        restore the bands this just cleared.
        """
        if self._explorer is not None:
            self._explorer.reset_tuning()
        self._apply_strip(self._strip_defaults)
        self._save_debounce.start()
        if self._stream_worker is not None:
            self.restart_stream("reset to defaults · restarting…")
        else:
            self.status_lbl.setText("reset to defaults")

    def _save_tuning(self) -> None:
        """Write this clip's tuning sidecar (best-effort; see gui/tuning_store)."""
        self._save_debounce.stop()
        if self._explorer is not None:
            self._saved_view = self._explorer.capture_view_state()
            region = self._explorer.active_region_index
        else:
            region = self._pending_region
        save_tuning(self.video_path, {"strip": self._strip_values(),
                                      "view": self._saved_view,
                                      "region_index": region})

    # -- config assembly -----------------------------------------------------
    def _build_cfg(self) -> PipelineConfig:
        block = int(self.block_spin.value())
        return PipelineConfig(
            preprocess=PreprocessConfig(
                downsample=float(self.ds_spin.value()),
                normalize=self.norm_combo.currentText(),
                denoise="off", registration="off", bg_subtract="off",
                mask_path=None),
            flow=FlowConfig(
                block_size=None if block == _AUTO_BLOCK else block),
        )

    def _window(self) -> tuple[int, int]:
        start = int(self.start_slider.value())
        n = max(2, int(round(self.len_spin.value() * self.fps)))
        n = min(n, self.frame_count - start)
        return start, n

    # -- button state --------------------------------------------------------
    def _set_busy(self, which: _Busy | None):
        """Reflect which pass is running in the two action buttons: the running
        one becomes Stop, the other is disabled (only one pass owns the decoder
        at a time). ``None`` restores the idle labels."""
        processing = which is _Busy.PROCESS
        streaming = which is _Busy.STREAM
        self.process_btn.setText("Stop" if processing else _PROCESS_TEXT)
        self.live_btn.setText("Stop" if streaming else _LIVE_TEXT)
        self.process_btn.setEnabled(which is None or processing)
        self.live_btn.setEnabled(which is None or streaming)

    def _on_process_clicked(self):
        if self._proc_worker is not None:
            self._request_stop(self._proc_worker, "whole-video pass")
        else:
            self.process_whole_video()

    def _request_stop(self, worker, what: str):
        """Ask a running pass to unwind. It stops at its next progress tick, so
        both buttons stay disabled until the worker's ``cancelled`` lands."""
        worker.cancel()
        self.process_btn.setEnabled(False)
        self.live_btn.setEnabled(False)
        self.status_lbl.setText(f"stopping the {what}…")

    def _on_knob_settled(self):
        """A preprocessing knob settled. Every one of them is upstream of the
        block grid, so a running pass has to be replanned; with nothing running
        the new settings are simply what the next Play will use.

        Both debounces land here. They still differ in settle time -- Block is
        a single spin step and 250 ms is enough, whereas Downsample is dragged
        -- but they no longer differ in what they do.

        Both branches SAY which settings are now in force. The idle one is not
        cosmetic: without it the strip keeps whatever the last pass left there
        ("live pass stopped"), so a knob moved while stopped gives no feedback
        at all and reads as a dead control.
        """
        self._debounce.stop()
        self._block_debounce.stop()
        cfg = self._build_cfg()
        where = (f"block {self._resolved_block(cfg)}, "
                 f"norm {cfg.preprocess.normalize}")
        if self._stream_worker is not None:
            self.restart_stream(f"settings changed — restarting at {where}…")
        else:
            self.status_lbl.setText(f"{where} · press Play ▶ to run from here")

    # -- the continuous pass (todo.md Batch Q) -------------------------------
    def _on_live_clicked(self):
        if self._stream_worker is not None:
            self.stop_stream()
        else:
            self.start_stream()

    def restart_stream(self, note: str = "") -> None:
        """Bring a RUNNING pass onto new settings, or onto a new position.

        Two steps, not one, and the second happens in _on_stream_cancelled: the
        worker owns the decoder until it notices the cancel at its next frame,
        so starting here would find it still held. ``_restart_stream_at`` is the
        flag that tells that handler this stop was a move rather than a stop.

        The position always comes from the start slider, which every caller
        writes first -- there is deliberately no second way to set it.

        ``note`` is written AFTER the stop, not before. ``stop_stream`` ends in
        ``_request_stop``, which overwrites the status with "stopping the…", so
        a caller that set its own message first would have it replaced in the
        same event-loop turn and never seen -- which is what happened to the
        block/normalize readout this parameter now carries.
        """
        if self._stream_worker is None:
            return
        self.stop_stream()
        self._restart_stream_at = int(self.start_slider.value())
        if note:
            self.status_lbl.setText(note)

    def stop_stream(self):
        """Ask the live pass to unwind. The timer stops HERE rather than in
        _end_stream alone: the worker only notices the cancel at its next frame,
        and until then the timer would keep parking requests on a thread that is
        on its way out."""
        if self._stream_worker is None:
            return
        self._stream_timer.stop()
        self._request_stop(self._stream_worker, "live pass")

    def _stream_plan(self, cfg: PipelineConfig, start: int):
        """Geometry, window and ring capacity for a live pass. Decode-free, which
        is the whole reason ``plan_channel_stream`` exists: the ring is sized from
        the SAME object the pass runs on, so the clamp of the window against the
        clip is computed once rather than once here and once in the worker."""
        w, h, fps, fc = self._dims
        meta = synth_live_meta(self.video_path, cfg, self.replicates,
                               width=w, height=h, fps=fps, frame_count=fc)
        # n=None: forward to the end of the clip. A live pass has no window --
        # that is the point of it -- and the ring, not
        # the plan, is what bounds memory.
        plan = plan_channel_stream(meta, start=start, n=None,
                                   want=frozenset(LIVE_CHANNELS))
        capacity = capacity_for_budget(self._ring_budget, len(plan.want),
                                       plan.ny, plan.nx)
        return meta, plan, capacity

    def start_stream(self):
        if (self._proc_worker is not None or self._sweep_worker is not None
                or self._stream_worker is not None):
            return                          # another pass owns the decoder
        self._debounce.stop()
        self._block_debounce.stop()
        start = int(self.start_slider.value())
        cfg = self._build_cfg()
        meta, plan, capacity = self._stream_plan(cfg, start)
        if plan.n < 2:
            self.status_lbl.setText("nothing left to stream from here")
            return
        # Carry the tuning across the rebuild: a live pass rebuilds the explorer,
        # and without this the bands revert to defaults on every start (T17, and
        # the reason capture_view_state exists).
        if self._explorer is not None:
            self._pending_state = self._explorer.capture_view_state()
        self._stream_meta = meta
        self._stream_plan_obj = plan
        self._stream_token += 1
        self._live_window = None
        self._stream_truncated = False
        self._set_busy(_Busy.STREAM)
        self.status_lbl.setText(
            f"live from {start / self.fps:.2f} s · ring holds "
            f"{capacity / max(self.fps, 1e-6):.0f} s at "
            f"{plan.ny}×{plan.nx} blocks…")
        self._stream_worker = LiveStreamWorker(
            self.video_path, plan, capacity, parent=self)
        self._stream_worker.advanced.connect(self._on_advanced)
        self._stream_worker.window_ready.connect(self._on_window_ready)
        self._stream_worker.detected.connect(self._on_detected)
        self._stream_worker.done.connect(self._on_stream_done)
        self._stream_worker.failed.connect(self._on_stream_failed)
        self._stream_worker.cancelled.connect(self._on_stream_cancelled)
        self._stream_worker.finished.connect(self._stream_worker.deleteLater)
        self._stream_worker.start()
        self._stream_timer.start()
        # Stamp the track with what this pass will be computed under BEFORE any
        # window can arrive, so no write can land unstamped or against the
        # previous pass's settings. On the FIRST pass of a session this cannot
        # answer yet (no explorer, so no tuned bands); _show_live_window calls
        # it again the moment it builds one.
        self._arm_detection()

    def _arm_detection(self) -> None:
        """Stamp the track for the settings now in force and start the detector.

        Two callers, deliberately: ``start_stream`` runs it before the pass so
        no window can land against the previous pass's settings, and
        ``_show_live_window`` runs it again when it builds the explorer, which
        is the first moment the detector is answerable at all. Idempotent --
        ``_sync_track`` is a no-op when the stamp has not moved, and restarting
        a running QTimer only resets its interval.
        """
        if self._sync_track():
            self.status_lbl.setText(
                self.status_lbl.text() + " · earlier detection shown gray "
                "(different settings)")
        if self._track.stamp is None:
            self.status_lbl.setText(
                self.status_lbl.text() + " · display only — select a "
                "replicate to accumulate detection")
        elif self._stream_worker is not None:
            # Only while a pass is running: the timer parks requests on the
            # stream worker, so arming it without one would tick against None.
            self._detect_timer.start()

    def _request_live_window(self):
        """Ask for the trailing window the scalogram is recomputed over.

        Bounded, never the growing island: ``morlet_band_power`` is
        O(F·T log T·B), so re-running it over a 30k-frame island every tick is
        not interactive. The length is the same one the window knob already
        carries, so the window knob still means what it says.
        """
        if self._stream_worker is None:
            return
        n = max(2, int(round(self.len_spin.value() * self.fps)))
        # Remembered so the arriving window can be measured against what was
        # actually asked for, rather than against the knob's value now -- the two
        # differ if the knob moved while the request was in flight.
        self._live_request_n = n
        self._stream_worker.request_latest(n, sorted(LIVE_CHANNELS),
                                           self._stream_token)

    def _on_advanced(self, start: int, frontier: int, rate: float):
        """Frontier readout, in the terms that let the user judge whether the
        surface is keeping up.

        The realtime RATIO is the point, not the raw fps: the measured drop from
        ~80 fps to ~18 fps when `appearance` joins the pass is the difference
        between running ahead of playback and falling behind it, and a bare fps
        number does not say which side of 1.0 it is on. ``start`` is reported
        (not assumed to be the plan's) because past capacity the ring has dropped
        history, and claiming the island still reaches its origin would be the
        stale-shown-as-current failure this whole surface is built against.
        """
        if self._stream_worker is None or self._stream_worker.is_cancelled():
            return          # keep the "stopping…" note
        plan = self._stream_plan_obj
        ratio = rate / max(self.fps, 1e-6)
        behind = " — behind playback" if rate > 0 and ratio < 1.0 else ""
        held = (frontier - start) / max(self.fps, 1e-6)
        dropped = (" · oldest dropped"
                   if plan is not None and start > plan.start else "")
        pct = (100.0 * (frontier - plan.start) / max(1, plan.n)
               if plan is not None else 0.0)
        self.status_lbl.setText(
            f"live · frame {frontier} ({pct:.0f}%) · holding {held:.1f} s"
            f"{dropped} · {rate:.0f} fps ({ratio:.2f}× realtime){behind}")

    def _on_window_ready(self, win: dict):
        """A trailing window came back from the worker thread.

        The token check is the whole guard: a request parked before a restart is
        serviced against the OLD island and arrives after the new one has begun,
        so rendering it would put frames from a different set of knobs on screen
        looking merely stale. Dropping it is safe because the next tick asks
        again against the current worker.
        """
        if self._stream_worker is None:
            return
        if win.get("token") != self._stream_token:
            return
        arrived = int(next(iter(win["channels"].values())).shape[0])
        if arrived < min(self._live_request_n, _MIN_LIVE_FRAMES):
            # Still filling. Dropped rather than drawn short: see
            # _MIN_LIVE_FRAMES. The frontier readout is already reporting the
            # fill, so the user is not left wondering why nothing has appeared.
            return
        self._live_window = win
        self._show_live_window(win)

    def _on_stream_done(self, pass_meta):
        self._end_stream()
        n = int((pass_meta or {}).get("n_frames", 0))
        if (pass_meta or {}).get("truncated"):
            # Not the same as reaching the end: the decoder stopped early and
            # this island is shorter than the clip. Saying "complete" here is
            # exactly the claim FINDINGS.md section 15 was written about.
            self._stream_truncated = True
            if self._live_window is not None:
                # Re-render the window that is already on screen, now that the
                # flag is known: the status line alone would leave the
                # ChannelData (and anything reading its meta, detection
                # included) asserting a clean pass.
                self._show_live_window(self._live_window, force=True)
            self.status_lbl.setText(
                f"live pass stopped early at {n} frames — the decoder ended "
                f"before the clip did")
            return
        self.status_lbl.setText(f"live pass complete · {n} frames")

    def _on_stream_failed(self, msg: str):
        self._end_stream()
        # A pending seek dies with the pass. Only _on_stream_cancelled consumes
        # the flag, so leaving it set here would arm a restart that fires at the
        # NEXT stop -- at a position the user chose minutes earlier, against a
        # pass they deliberately ended.
        self._restart_stream_at = None
        self.status_lbl.setText(f"live pass failed: {msg}")

    def _on_stream_cancelled(self):
        self._end_stream()
        # A seek during a live pass stops the island and starts another where the
        # user landed. Resumed HERE rather than at the seek, because the worker
        # only notices the cancel at its next frame and a start before that would
        # find the decoder still held.
        if self._restart_stream_at is not None:
            self._restart_stream_at = None
            self.status_lbl.setText("live pass moved — restarting here")
            self.start_stream()
            return
        self.status_lbl.setText("live pass stopped")

    def _end_stream(self):
        """One place the three terminal signals converge, so the timer cannot
        outlive the worker it parks requests on."""
        self._stream_timer.stop()
        self._detect_timer.stop()
        self._stream_worker = None
        self._set_busy(None)

    # -- the whole-video detection track -------------------------------------
    def _current_stamp(self) -> tuple[TrackStamp | None, tuple | None]:
        """``(stamp, region_grid)`` for the settings now in force, or
        ``(None, None)`` when the detector is not yet answerable.

        Not answerable means no explorer (no pass has run, so there
        are no tuned bands) or no selected replicate. Returning None rather than
        a placeholder stamp is deliberate: a placeholder would let writes land
        against settings nobody chose, and the strip would report coverage for a
        detector that was never configured.
        """
        if self._explorer is None:
            return None, None
        params = self._explorer.detection_params()
        idx = int(params["region_index"])
        if idx < 0:
            return None, None
        meta = self._explorer.meta
        n_blocks, grid = region_grid_from_meta(meta, idx)
        cfg = self._build_cfg()
        stamp = TrackStamp(
            channel=params["channel_attr"],
            freq_band_hz=params["freq_band_hz"],
            grid=tuple(int(v) for v in meta["grid"]),
            region_index=idx,
            region_blocks=n_blocks,
            downsample=cfg.preprocess.downsample,
            block_size=cfg.flow.block_size)
        return stamp, grid

    def _sync_track(self, repaint: bool = True) -> bool:
        """Push the current settings onto the track.

        Returns whether the STAMP moved -- i.e. whether everything already
        computed just became stale. The caller reports that; this does not,
        because the same call is made from a knob handler (where the user needs
        telling) and from the start of a pass (where they do not).
        """
        stamp, grid = self._current_stamp()
        if stamp is None:
            return False
        moved = self._track.set_stamp(stamp, region_grid=grid)
        if moved:
            # Retention is priced per stamp because B moves with the grid and
            # the region: a band the surface could afford to retain at block 64
            # is a different object at block 16.
            fits = band_power_bytes(self.frame_count,
                                    stamp.region_blocks) <= _TRACK_BP_CAP
            self._track.retain_band_power = fits
            self._track_grid = grid
        self._sync_track_detector(repaint=False)
        if repaint:
            self._repaint_track()
        return moved

    def _sync_track_detector(self, repaint: bool = True) -> None:
        """Re-derive the track from the threshold knobs, without a fresh pass.

        This is the half of the redesign that makes tuning usable at clip
        length: the value band, the count band and the detection window are NOT
        part of the stamp, so moving one re-derives every frame already covered
        instead of graying the clip out and demanding a re-stream.
        """
        if self._explorer is None or self._track.stamp is None:
            return
        p = self._explorer.detection_params()
        self._track.set_detector(
            value_band=p["value_band"], count_band=p["count_band"],
            detect_window=p["detect_window"], centered=p["centered"])
        if repaint:
            self._repaint_track()

    def on_tuning_changed(self) -> None:
        """A detection knob moved. Debounced because the value band's half of the
        re-derive is a per-frame connected-components loop over every retained
        row -- cheap per frame, not cheap across 30k of them at drag rate."""
        self._retune_debounce.start()

    def _on_retune_settled(self) -> None:
        """Apply a settled tuning change, and SAY SO when it invalidated work.

        A knob that quietly grays out ten minutes of processing is the same
        class of surprise as one that quietly changes a threshold's meaning
        (T17, and the count-band re-denomination note): correct behaviour still
        has to be visible. Threshold moves take the other branch and say
        nothing, because nothing was lost -- they re-derived.
        """
        moved = self._sync_track()
        if not moved:
            return
        stale = int(self._track.stale.sum())
        if stale:
            self.status_lbl.setText(
                f"settings changed — {stale / max(self.fps, 1e-6):.0f} s of "
                f"earlier detection is now shown gray (computed under the "
                f"previous channel/band/geometry) and needs another pass")

    def _repaint_track(self) -> None:
        self.navigator.set_track(self._track)

    def _track_write(self, first: int, band_power, **trim) -> bool:
        """Write into the track, refusing a geometry that does not match.

        The one place both producers -- the live pass and the commit -- go
        through, because both can be handed band power whose block count belongs
        to a different grid: the commit builds its own ChannelData, and a live
        transform can be serviced after the geometry moved. ``write`` raises on
        that, which in a Qt signal handler is a hard crash rather than an error,
        so the mismatch is turned into a reported False here instead.

        Returns whether anything was recorded. Callers MUST say so when it is
        False: no coverage and no detections look the same on the strip, and
        that is exactly the confusion the coverage mask exists to prevent.
        """
        if self._track.stamp is None:
            return False
        bp = np.asarray(band_power, np.float32)
        if bp.ndim != 2 or bp.shape[1] != self._track.n_blocks:
            return False
        a, b = self._track.write(int(first), bp, **trim)
        return b > a

    def _request_detect_window(self):
        """Park a detection request on the stream worker.

        The window is the display window widened by the overlap the accumulator
        needs: every write loses a cone of influence at each end, so requests
        that merely abut would leave an unexamined seam at every join. Widening
        the request is the cheapest place to fix that -- the alternative is a
        gap list and a backfill worker for gaps this surface created itself.
        """
        if self._stream_worker is None or self._explorer is None:
            return
        stamp = self._track.stamp
        if stamp is None:
            return
        base = max(2, int(round(self.len_spin.value() * self.fps)))
        coi = coi_trim(stamp.freq_band_hz, self.fps)
        n = base + int(_DETECT_OVERLAP_COI * 2 * coi)
        self._stream_worker.request_detect(
            n, stamp.channel, self._stream_meta, stamp.region_index,
            default_freqs(self.fps), stamp.freq_band_hz, self._stream_token)

    def _on_detected(self, msg: dict):
        """Band power for a trailing window came back. Write it COI-trimmed.

        The token check is the same guard ``_on_window_ready`` needs and for a
        sharper reason: a transform parked before a restart is serviced against
        the OLD island, so writing it would record coverage under the current
        stamp for frames computed under different knobs -- stale data laundered
        into current, which is worse than either showing it stale or not at all.
        """
        if self._stream_worker is None or msg.get("token") != self._stream_token:
            return
        stamp = self._track.stamp
        if stamp is None:
            return
        bp = msg["band_power"]
        first = int(msg["first"])
        coi = coi_trim(stamp.freq_band_hz, self.fps)
        # The clip's true ends are edges of the DATA, not of an arbitrary cut,
        # so the transform is as trustworthy there as it can ever be and the
        # frames are kept. Trimming them anyway would leave the first and last
        # ~1.4 s of every clip permanently unexaminable.
        head = 0 if first <= 0 else coi
        tail = 0 if first + bp.shape[0] >= self.frame_count else coi
        if not self._track_write(first, bp, trim_head=head, trim_tail=tail):
            # A single drop is ordinary -- the geometry moved between the
            # request and its service, and there is no correct reshape. A
            # SUSTAINED run of them is not: it means the stream's meta and the
            # explorer's disagree about the grid, and the symptom is a strip
            # that simply never fills while the pass reports healthy progress.
            self._detect_drops += 1
            if self._detect_drops == _DETECT_DROP_ALARM:
                self.status_lbl.setText(
                    "live detection is producing nothing — the streamed block "
                    "geometry does not match the tuned selection; stop and "
                    "restart before trusting this strip")
            return
        self._detect_drops = 0
        self._repaint_track()

    # -- the strip as the clip's seeker --------------------------------------
    def _on_scrubbed(self, frame: int):
        """Dragging the strip. Cheap consumers only -- the hosted explorer's
        cursor if that frame is in the span it holds, and the readout. The
        expensive move waits for the release; a pass restart per pixel of drag
        would make the seeker unusable.

        The explorer cursor is NOT moved while a pass is running. Under
        follow_center it cannot hold: every served window re-pins the cursor to
        the span centre, so a seek placed here survives ~100 ms and the cursor
        oscillates between the pointer and the centre for the whole drag. The
        drag position is already shown in the window-start readout, and the
        release restarts the pass there.
        """
        self._sync_window_label(scrub=frame)
        if self._explorer is not None and self._stream_worker is None:
            self._explorer.seek_absolute(frame)

    def _on_seek_committed(self, frame: int):
        """Released the strip somewhere with no detection under it: go there.

        A live pass RESTARTS at the new position rather than continuing, and the
        track keeps everything it already holds -- which is the reversal the
        redesign records. Skipping around builds up parts of one whole-video
        picture; the per-frame series that carry it are cheap enough that
        abandoning earlier segments buys nothing.

        With nothing running the seek parks the window and SAYS so. It does not
        start a pass: Play is the only control that commits the decoder, and a
        drag on a seeker bar is not a request to spend minutes of decode. But it
        must not be silent either -- the plots still show the last span, so
        without the readout the strip looks broken rather than parked.
        """
        n = max(2, int(round(self.len_spin.value() * self.fps)))
        start = int(np.clip(frame, 0, max(0, self.frame_count - n)))
        self.start_slider.blockSignals(True)
        self.start_slider.setValue(start)
        self.start_slider.blockSignals(False)
        self._sync_window_label()
        self.navigator.set_cursor(frame)
        if self._stream_worker is not None:
            self.restart_stream("live pass moving…")
        else:
            self.status_lbl.setText(
                f"parked at {start / self.fps:.2f} s · press Play ▶ to run "
                f"from here")

    def _live_channel_data(self, win: dict) -> ChannelData:
        """Wrap a served trailing window as the ChannelData the explorer and the
        detector already read, so a live span arrives through the one interface
        they have.

        ``n_frames`` is the WINDOW's length, not the clip's -- the explorer's T
        axis is the span it was handed, exactly as in ``live_channel_source``.
        """
        chans = win["channels"]
        first = int(win["first"])
        any_arr = next(iter(chans.values()))
        meta = {**self._stream_meta,
                "n_frames": int(any_arr.shape[0]),
                "window_start": first,
                # Carried for the same reason live_channel_source carries it: a
                # consumer that sees only a length cannot tell a short window
                # from a decode that ended early. Unknown until the pass ends,
                # so it is False during the run and the final window is
                # re-rendered once the answer is in (see _on_stream_done).
                "truncated": self._stream_truncated,
                "channels_computed": sorted(chans)}
        plan = self._stream_plan_obj
        return ChannelData(
            meta=meta, channels=chans, window_start=first,
            approximated=bool(plan.approximated) if plan is not None else False)

    def _show_live_window(self, win: dict, force: bool = False) -> None:
        """Put a served window on screen: build the explorer on the first one,
        then update it in place.

        In place, not rebuilt: ``_swap_explorer`` constructs a fresh
        ScalogramExplorer and re-applies the captured view state, which is right
        for a knob change (the geometry moved) and wrong at 1 Hz -- it would
        discard the scalogram cube cache, reset the selected replicate, and
        replace the widget under the user's cursor once a second.

        NO LONGER gated on ``is_building()``. That gate existed because a cube
        arriving after its span had moved was DISCARDED, so updating faster than
        the transform meant every cube was dropped and relaunched and the
        scalogram never appeared at all (§23 trap 3). The discard is what has
        gone: a cube is now retained with the span it was transformed over, and
        its density heatmap is drawn at that span with the uncovered remainder
        hatched. A cube landing late is therefore late data correctly placed
        rather than wrong data, and there is nothing left to protect against.

        Dropping the gate is the point of the change. It was throttling the
        per-FRAME plots -- trace, pooled scalogram, detection sweep, ~12 ms
        together at whole-clip length -- to the rate of a per-BLOCK transform
        that is up to 500x slower and feeds none of them.
        """
        cd = self._live_channel_data(win)
        if self._explorer is None:
            self._swap_explorer(cd)
            # Pinned to the MIDDLE of the served window, and it stays there: the
            # playhead is a fixed landmark with computed footage on both sides
            # of it, rather than a marker riding the ragged right edge. See
            # ScalogramExplorer.follow_center.
            self._explorer.follow_center()
            # The detector only becomes ANSWERABLE here. start_stream stamps the
            # track before the pass starts, which is right for every restart --
            # but on the first pass of a session there is no explorer yet, so
            # _current_stamp returns None, the track goes unstamped and the
            # detect timer never starts. The strip then stays empty for the
            # whole run while progress looks healthy: the exact shape
            # _DETECT_DROP_ALARM exists to catch, arriving by a route that
            # produces no drops to count. Stamping here closes it.
            self._arm_detection()
            return
        self._explorer.set_channel_data(cd, live=not force)
        # The strip's cursor is driven by the explorer's frame_moved, which only
        # fires when the frame CHANGES. Under follow_center the centre index is
        # constant while the window slides, so the absolute frame moves without
        # the index moving -- and the strip would sit still through the whole
        # pass. Push it explicitly.
        self.navigator.set_cursor(self._explorer.absolute_frame())

    def _cost_model(self) -> tuple[CostModel | None, int | None]:
        """Fit within one regime; returns ``(model, fixed block or None)``.

        Samples must not be mixed across regimes -- a block=1 pass does no
        reduction at all and is much slower than a production pass. Picking the
        regime with the most distinct scales finds the fit if one exists
        anywhere; ties go to the newest.

        Every sample now comes from the dialog's own sweep (see _on_sweep_row).
        The surface used to contribute one per windowed extract, so the dialog
        often opened with a model already fitted; with the extract retired it
        opens empty until a sweep is run. That is a real loss of convenience and
        not of correctness -- the sweep's rows were always the better samples,
        since they resolve the block as production does.

        The second element tells the dialog which regime the numbers came from:
        a fixed block (``1`` in particular, which the dialog labels as an upper
        bound) or ``None`` for the tracked regime, which needs no caveat because
        it is what production does.
        """
        if not self._cost_samples:
            return None, None
        groups: dict[int | None, list[PassSample]] = {}
        for key, s in self._cost_samples.items():
            groups.setdefault(self._sample_regime.get(key), []).append(s)
        newest = (self._sample_regime.get(self._last_cost_key)
                  if self._last_cost_key else object())
        block, samples = max(groups.items(),
                             key=lambda kv: (len({s.scale for s in kv[1]}),
                                             kv[0] == newest))
        return CostModel.fit(samples), block

    def _open_downsample_dialog(self):
        from gui.downsample_dialog import DownsampleDialog
        w, h, fps, _fc = self._dims
        model, model_block = self._cost_model()
        dlg = DownsampleDialog(
            self.replicates, src_width=w, src_height=h, fps=fps,
            current_scale=float(self.ds_spin.value()),
            model=model, model_block=model_block, flow=self._build_cfg().flow,
            n_channels=len(LIVE_CHANNELS), parent=self)
        dlg.sweep_requested.connect(self._start_sweep)
        dlg.sweep_cancelled.connect(self._stop_sweep)
        dlg.render_requested.connect(self._start_render)
        dlg.calibrate_requested.connect(self._open_calibration)
        ok, why = self._sweep_ready()
        dlg.set_sweep_available(ok, why)
        self._dlg = dlg
        try:
            accepted = dlg.exec()
        finally:
            # A sweep still running when the window closes has nowhere to report
            # to, and it holds the decoder the next pass needs. Unwind it here
            # rather than letting it race the reopened dialog.
            self._stop_sweep(wait=True)
            self._stop_render(wait=True)
            self._dlg = None
            # Parented to the surface, so without this each open leaves a live
            # dialog behind whose Run button is still wired to _start_sweep.
            dlg.deleteLater()
        if accepted and dlg.chosen_scale is not None:
            # setValue fires valueChanged, which starts the replan debounce,
            # so choosing a scale here applies exactly as dragging the spin does.
            self.ds_spin.setValue(float(dlg.chosen_scale))

    # -- the dialog's calibration sub-tool -----------------------------------
    def _sorted_reps(self) -> list[dict]:
        return in_tile_order(self.replicates)

    def refresh_replicate_metadata(self, replicates: list[dict]) -> None:
        """Re-read non-geometry replicate fields from the owner.

        Geometry changes rebuild this whole surface (the tab keys it on a
        geometry hash), but calibration and baselines do not -- and
        ``AppState.set_replicate_specs`` *replaces* its dicts with fresh copies
        on every edit, so this surface's references go stale silently. Measured:
        calibrating in the replicate tab left an already-built surface's
        downsample window reporting the replicate as uncalibrated for the rest
        of the session.

        Refreshing all metadata rather than calibration alone, because the same
        staleness applies to every non-geometry field and a calibration-only
        patch would have to be rewritten for the next one. Geometry is
        deliberately NOT copied across: if it differed, this surface would be
        the wrong one and the tab would have rebuilt it.
        """
        incoming = {int(r["id"]): r for r in replicates if "id" in r}
        for rep in self.replicates:
            src = incoming.get(int(rep.get("id", -1)))
            if src is None:
                continue
            rep.update({k: v for k, v in src.items() if k not in _GEOMETRY_KEYS})

    def _open_calibration(self, rep_index: int):
        reps = self._sorted_reps()
        if not reps:
            return
        i = min(max(0, int(rep_index)), len(reps) - 1)
        rep = reps[i]
        box = self._replicate_box(i)
        start, n = self._window()
        # Decoded on the GUI thread: it is one seek, user-initiated, and a modal
        # window follows immediately, so the freeze is bounded and legible in a
        # way a spinner over an empty dialog would not be.
        with VideoSource(self.video_path) as src:
            frame = src.frame_at(start + n // 2)
        if frame is None:
            # Same failure the render strip reports, and the same place to
            # report it: the dialog already has a status line for "this frame
            # could not be decoded".
            if self._dlg is not None:
                self._dlg.render_failed(
                    f"frame {start + n // 2} could not be decoded")
            return
        from gui.calibration_dialog import CalibrationDialog
        dlg = CalibrationDialog(
            frame, box=box, label=str(rep.get("label", "")),
            pixels_per_mm=rep.get("pixels_per_mm"),
            body_length_mm=rep.get("body_length_mm"),
            body_length_px=rep.get("body_length_px"),
            scale=float(self.ds_spin.value()), parent=self._dlg or self)
        try:
            if dlg.exec() and dlg.calibration is not None:
                fields = dlg.calibration.as_replicate_fields()
                if fields:
                    if self._dlg is not None:
                        self._dlg.apply_calibration(i, fields)
                    self.calibration_changed.emit(int(rep["id"]), fields)
        finally:
            dlg.deleteLater()

    # -- the dialog's render strip -------------------------------------------
    def _replicate_box(self, index: int):
        """The source-pixel box of one replicate, as the pass sees it.

        Through ``build_layout`` at scale 1.0 rather than off the replicate dict
        directly, so the strip renders the same box the pass crops -- a box the
        layout clamps or rounds must not be shown unclamped, or the strip
        describes a crop that never runs.
        """
        layout = build_layout(self.replicates, self._dims[0], self._dims[1],
                              scale=1.0, block_size=1)
        tiles = list(layout.tiles)
        if not tiles:
            return None
        return tiles[min(max(0, index), len(tiles) - 1)].source_box

    def _start_render(self, rep_index: int, scales):
        box = self._replicate_box(int(rep_index))
        if box is None:
            if self._dlg is not None:
                self._dlg.render_failed("this source has no replicate boxes")
            return
        # Supersede rather than queue: the user stepping through replicates would
        # otherwise wait out every seek they skipped past.
        self._stop_render(wait=True)
        start, n = self._window()
        self._render_worker = _RenderWorker(
            self.video_path, box, start + n // 2, scales,
            self._build_cfg().preprocess, self)
        self._render_worker.done.connect(self._on_render_done)
        self._render_worker.failed.connect(self._on_render_failed)
        self._render_worker.finished.connect(self._render_worker.deleteLater)
        self._render_worker.start()

    def _stop_render(self, wait: bool = False):
        w = self._render_worker
        if w is None:
            return
        w.cancel()
        self._render_worker = None
        if wait:
            # There is no cancel point inside a two-frame decode, so this waits
            # out the seek rather than interrupting it. Bounded and short, and
            # the alternative is a thread writing into a closed dialog.
            w.wait()

    def _on_render_done(self, renders):
        # A superseded worker still runs to completion (a two-frame decode has no
        # cancel point) and still emits. _stop_render has already cleared the
        # slot, so identity is what distinguishes the live result from the stale
        # one -- without it a fast second request is overwritten by the first.
        if self.sender() is not self._render_worker:
            return
        self._render_worker = None
        if self._dlg is None:
            return
        first = renders[0] if renders else None
        note = ""
        if first is not None:
            note = (f"Frame {self._window()[0] + self._window()[1] // 2} of "
                    f"replicate {self._dlg.rep_spin.value()}, "
                    f"{first.size_label} at full resolution.")
        self._dlg.set_renders(renders, note)

    def _on_render_failed(self, msg: str):
        if self.sender() is not self._render_worker:
            return
        self._render_worker = None
        if self._dlg is not None:
            self._dlg.render_failed(msg)

    # -- the dialog's timing sweep -------------------------------------------
    def _sweep_ready(self) -> tuple[bool, str]:
        """Whether a sweep can run, and why not when it cannot.

        Short, because the sweep only times the decode+solve: it needs no tuned
        detector and no selected replicate, so it can run the moment the window
        opens. Only a decoder conflict or a degenerate window can stop it.
        """
        if self._proc_worker is not None or self._stream_worker is not None:
            return False, "Another pass is running; wait for it to finish."
        if self._window()[1] < 2:
            return False, "The window is too short at this start position."
        return True, ""

    def _start_sweep(self, scales):
        ok, why = self._sweep_ready()
        if not ok or self._sweep_worker is not None:
            if self._dlg is not None:
                self._dlg.sweep_finished(why or "a sweep is already running")
            return
        self._debounce.stop()               # an armed knob edit must not cut in
        self._block_debounce.stop()
        start, n = self._window()
        cfg = self._build_cfg()
        # Every row of this sweep runs under one block intent, so record it once
        # here rather than re-deriving it per row from the resolved block.
        self._sweep_block_intent = cfg.flow.block_size
        self._set_busy(_Busy.SWEEP)
        self._sweep_worker = _SweepWorker(
            self.video_path, cfg, self.replicates, self._dims, start, n,
            scales, self)
        self._sweep_worker.row.connect(self._on_sweep_row)
        self._sweep_worker.row_failed.connect(self._on_sweep_row_failed)
        self._sweep_worker.scale_started.connect(self._on_sweep_scale)
        self._sweep_worker.done.connect(self._on_sweep_done)
        self._sweep_worker.failed.connect(self._on_sweep_failed)
        self._sweep_worker.cancelled.connect(self._on_sweep_cancelled)
        self._sweep_worker.finished.connect(self._sweep_worker.deleteLater)
        self._sweep_worker.start()

    def _stop_sweep(self, wait: bool = False):
        w = self._sweep_worker
        if w is None:
            return
        w.cancel()
        if wait:
            # Only on teardown: the worker unwinds at its next progress tick, and
            # blocking the GUI thread for that is preferable to leaving it
            # decoding into a dialog that no longer exists.
            w.wait()
            self._sweep_worker = None
            self._set_busy(None)

    def _on_sweep_scale(self, scale: float, i: int, total: int):
        self.status_lbl.setText(
            f"sweep {i}/{total} · timing a pass at scale {scale:.2f}…")

    def _on_sweep_row(self, sp):
        # The sweep's passes resolve the block as production would, so each row
        # is the cost sample the live surface cannot produce: this is what moves
        # the frontier out of the block=1 regime, live, while the dialog is open.
        # A pass with no measured wall time is dropped rather than entered as a
        # zero-cost point, which would pull the fitted decode floor toward zero
        # and put the knee where downsampling looks free.
        if sp.usable:
            key = (round(float(sp.scale), 6), int(sp.block))
            self._cost_samples[key] = sp.pass_sample()
            self._sample_regime[key] = self._sweep_block_intent
            self._last_cost_key = key
        if self._dlg is not None:
            # Model first: the row prints a corpus projection read off the model,
            # so refitting after would leave the newest row a step behind.
            self._dlg.set_model(*self._cost_model())
            self._dlg.add_sweep_row(sp)

    def _on_sweep_row_failed(self, scale: float, msg: str):
        if self._dlg is not None:
            self._dlg.sweep_failed(scale, msg)

    def _on_sweep_done(self, _res):
        self._sweep_worker = None
        self._set_busy(None)
        self.status_lbl.setText("sweep done")
        if self._dlg is not None:
            self._dlg.sweep_finished(
                "Done. Times are measured on this machine and this footage and "
                "are not portable; storage is exact arithmetic over the layout.")

    def _on_sweep_failed(self, msg: str):
        self._sweep_worker = None
        self._set_busy(None)
        if self._dlg is not None:
            self._dlg.sweep_finished(f"sweep failed: {msg}")

    def _on_sweep_cancelled(self):
        self._sweep_worker = None
        self._set_busy(None)
        self.status_lbl.setText("sweep stopped")
        if self._dlg is not None:
            self._dlg.sweep_finished(
                "Stopped. The rows that finished are still valid — each is an "
                "independent timed pass, not a partial result.")

    def _resolved_block(self, cfg: PipelineConfig) -> int:
        """The working block size a pass will actually use (auto tracks scale)."""
        return cfg.flow.resolve_block_size(
            cfg.preprocess.resolve_downsample(self._dims[0]))

    def _swap_explorer(self, cd):
        if self._placeholder is not None:
            self._placeholder.setParent(None)
            self._placeholder.deleteLater()
            self._placeholder = None
        old = self._explorer
        new = ScalogramExplorer.from_channel_data(
            cd, video_path=self.video_path, own_shortcuts=False,
            own_status=False, parent=self._host)
        self._host_lay.addWidget(new)
        self._explorer = new
        # The explorer no longer renders its own status line; mirror it into the
        # strip's second orange line. A direct relay (not a signal) so the
        # explorer can force a synchronous repaint of it before blocking work.
        new.set_status_relay(self.graph_status_lbl)
        new.tuning_changed.connect(self._save_debounce.start)
        # The track follows the tuning: threshold moves re-derive over retained
        # coverage, and a channel/band/region move re-stamps it so earlier work
        # goes visibly stale instead of being silently reused or blanked.
        new.tuning_changed.connect(self.on_tuning_changed)
        # The strip is the clip's seeker, so its cursor follows playback and
        # scrubbing in the hosted explorer, not only pass results.
        new.frame_moved.connect(self.navigator.set_cursor)
        if self._pending_region is not None:
            # Reopening the clip: put the selection back before the bands land,
            # so the count band's re-denomination measures against the replicate
            # it was tuned on. Consumed once -- a later rebuild is a rebuild and
            # keeps the reset-to-selection-view behaviour it always had.
            new.select_region(int(self._pending_region))
            self._pending_region = None
        if self._pending_state is not None:
            note = new.apply_view_state(self._pending_state)
            self._pending_state = None
            # A Block change re-denominates the detection threshold (it is a raw
            # block count). The conversion is the right thing to do, but a tuned
            # number that changes itself has to say so -- silently correct and
            # silently wrong look identical from here.
            if note:
                self.status_lbl.setText(note)
        if old is not None:
            old.close()                             # releases source + event filter
            old.setParent(None)
            old.deleteLater()

    # -- whole-video commit --------------------------------------------------
    def process_whole_video(self):
        if (self._proc_worker is not None or self._sweep_worker is not None
                or self._stream_worker is not None):
            return                                  # a pass is already running
        if self._explorer is None:
            self.status_lbl.setText(
                "press Play ▶ and tune the detector first")
            return
        params = self._explorer.detection_params()
        if params["region_index"] < 0:
            self.status_lbl.setText("select a replicate before processing")
            return
        cfg = self._build_cfg()
        self._set_busy(_Busy.PROCESS)
        flo, fhi = params["freq_band_hz"]
        self._proc_t0 = time.monotonic()
        self._proc_ctx = (f"{flo:.2f}–{fhi:.2f} Hz on {params['channel_attr']}")
        self.status_lbl.setText(
            f"processing whole video ({self.frame_count} frames) · "
            f"{self._proc_ctx}… starting the one expensive pass")
        self._proc_worker = _ProcessWorker(
            self.video_path, cfg, self.replicates, self._dims,
            params["region_index"], params, self)
        self._proc_worker.done.connect(self._on_processed)
        self._proc_worker.failed.connect(self._on_process_failed)
        self._proc_worker.cancelled.connect(self._on_process_cancelled)
        self._proc_worker.progress.connect(self._on_proc_progress)
        self._proc_worker.phase.connect(self._on_proc_phase)
        self._proc_worker.finished.connect(self._proc_worker.deleteLater)
        self._proc_worker.start()

    def _on_proc_progress(self, done: int, total: int):
        if self._proc_worker is None or self._proc_worker.is_cancelled():
            return          # keep the "stopping…" note; a tick may still be in flight
        pct = 100.0 * done / max(1, total)
        self.status_lbl.setText(
            f"processing whole video · extracting {done}/{total} frames "
            f"({pct:.0f}%) · {self._proc_ctx}"
            f"{self._eta(done, total, self._proc_t0)}")

    def _on_proc_phase(self, phase: str):
        if self._proc_worker is None or self._proc_worker.is_cancelled():
            return
        self.status_lbl.setText(
            f"processing whole video · {phase} · {self._proc_ctx}…")

    def _on_processed(self, res):
        # Read before the worker reference is dropped.
        short = getattr(self._proc_worker, "short", None)
        self._proc_worker = None
        self._set_busy(None)
        # Into the SAME accumulator the live pass fills, not a parallel result
        # object: one data structure with one coverage mask, so a commit and the
        # live passes around it compose instead of overwriting each other. The
        # commit's band power spans the whole clip and its edges are the clip's
        # own, so nothing is trimmed -- there is no arbitrary cut to protect
        # against here.
        self._sync_track(repaint=False)
        wrote = self._track_write(int(getattr(res, "window_start", 0)),
                                  res.band_power, trim=0)
        self._repaint_track()
        if not wrote:
            # The pass succeeded but its geometry does not match what the strip
            # is stamped with, so nothing was recorded. Said out loud: a commit
            # that silently produced no coverage would look identical to a clip
            # with no detections in it.
            self.status_lbl.setText(
                "whole-video pass finished but its block geometry does not "
                "match the current selection — nothing was recorded; "
                "restart the live pass and process again")
            return
        n = len(self._track.detected_intervals())
        if short is not None:
            # Never call this a whole-video pass. Past the cut point there is no
            # evidence either way, and a detection count presented without that
            # caveat reads as "this is what the video contains".
            covered, requested = short
            self.status_lbl.setText(
                f"PARTIAL pass · decode ended at frame {covered} of {requested} "
                f"· {n} detection{'s' if n != 1 else ''} in the part that was "
                f"read — the rest of the video was NOT examined")
            return
        self.status_lbl.setText(
            f"whole-video pass done · {n} detection{'s' if n != 1 else ''} — "
            f"click one (or step strongest) to verify in a window")

    def _on_process_failed(self, msg: str):
        self._proc_worker = None
        self._set_busy(None)
        self.status_lbl.setText(f"process failed: {msg}")

    def _on_process_cancelled(self):
        """Stopped mid-commit: the navigator keeps whatever earlier result it
        holds rather than being cleared by a pass that never finished."""
        self._proc_worker = None
        self._set_busy(None)
        self.status_lbl.setText(
            "whole-video pass stopped — settings are free to change")

    @staticmethod
    def _eta(done: int, total: int, t0: float) -> str:
        """A ' · ~Ns left' / ' · ~N.N min left' suffix from the rate so far. Empty
        until a frame has landed so the estimate is not division-by-zero noise."""
        if done <= 0 or total <= done:
            return ""
        elapsed = time.monotonic() - t0
        if elapsed <= 0:
            return ""
        remaining = (total - done) * elapsed / done
        if remaining < 90:
            return f" · ~{remaining:.0f}s left"
        return f" · ~{remaining / 60:.1f} min left"

    def _focus_frame(self, center: int):
        """A detection was chosen: park the window on it for verification.

        The window STARTS half a length before the event rather than being
        centred on it, because the playhead now sits at the middle of the served
        window (see follow_center): a pass starting here reaches the event with
        it under the cursor, which is what "focus this detection" means.

        Like a plain seek, this parks rather than starting a pass -- but it has
        to say that outright. "Next strongest" is a verb, and a button that
        moves a slider the user is not looking at and changes nothing else is
        indistinguishable from a broken one.
        """
        n = max(2, int(round(self.len_spin.value() * self.fps)))
        start = int(np.clip(center - n // 2, 0, max(0, self.frame_count - n)))
        self.navigator.set_cursor(center)
        self.start_slider.blockSignals(True)
        self.start_slider.setValue(start)
        self.start_slider.blockSignals(False)
        self._sync_window_label()
        if self._stream_worker is not None:
            self.restart_stream(f"moving to the detection at "
                                f"{center / self.fps:.2f} s…")
        else:
            self.status_lbl.setText(
                f"detection at {center / self.fps:.2f} s · press Play ▶ to "
                f"run from here")

    def toggle_playback(self):
        """Space handler the main window's focus-walk finds.

        Play IS the live pass now. There is no separate notion of playing back
        an already-extracted window: the surface has one forward-running pass,
        and Space starts and stops it exactly as the button does.
        """
        self._on_live_clicked()

    def hideEvent(self, e):
        # Switching tabs is the other way a tuning session ends without a
        # close, and the app can be quit from any tab. Cheap enough (one small
        # JSON write) to take unconditionally rather than track dirtiness.
        self._save_tuning()
        # Disarm before stopping. A knob or seek that armed a restart moments
        # ago is still pending, and _on_stream_cancelled consumes that flag
        # WITHOUT checking visibility -- so the stop below would unwind the
        # worker and immediately start another full-clip pass on a tab nobody
        # is looking at, which is the exact thing the stop exists to prevent.
        self._restart_stream_at = None
        # A live pass runs to the END OF THE CLIP: left running on a hidden tab
        # it holds the decoder and a full ring for minutes, and blocks the
        # whole-video commit when the user comes back. Stopped rather than
        # paused because the ring
        # is bounded anyway -- resuming would restart the island in most cases.
        self.stop_stream()
        super().hideEvent(e)

    def closeEvent(self, e):
        # Flush before anything is torn down: closing the tab (a new video, a
        # replicate edit that rebuilds the surface) is the most common way a
        # session ends, and an armed debounce would be dropped with the widget.
        self._save_tuning()
        self._restart_stream_at = None
        # A knob edited just before the close leaves a debounce armed; let it fire
        # and it restarts a pass on a surface that is on its way out.
        self._debounce.stop()
        self._block_debounce.stop()
        self._retune_debounce.stop()
        self._detect_timer.stop()
        # Before the workers: this one parks requests on the stream worker, and a
        # tick between the cancel and the wait would touch a thread on its way out.
        self._stream_timer.stop()
        for w in (self._proc_worker, self._sweep_worker,
                  self._render_worker, self._stream_worker):
            if w is not None:
                w.cancel()      # unwind at the next tick instead of waiting it out
                w.wait()
        self._proc_worker = self._sweep_worker = None
        self._render_worker = self._stream_worker = None
        if self._explorer is not None:
            self._explorer.close()
        super().closeEvent(e)
