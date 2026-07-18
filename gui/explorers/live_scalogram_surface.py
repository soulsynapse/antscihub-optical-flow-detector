"""The live tuning surface: preprocessing knobs feeding the scalogram directly.

This is the dissolved preprocessing tab. Instead of configuring a flow pass and
waiting for a cache, you pick a short window of a bare video and watch the
structure-tensor scalogram / detection stack respond as you drag downsample,
block size, and normalization. There is no flow solve and no cache write.

Because ``downsample`` and ``block_size`` change the block *geometry* (not just
pixels), the cleanest way to apply a knob is to re-extract the window into a
fresh ChannelData and rebuild the ScalogramExplorer from it -- the explorer's
constructor already derives every geometry-dependent structure consistently,
which is far safer than mutating grid/regions/cube-cache in place. The tuning
context you care about (selected channel, cursor, detection window, frequency
band) is captured off the old explorer and re-applied to the new one.

Every knob here is genuinely upstream and forces a re-extract -- accepted: the
window is short, so a pass is seconds. ``normalize`` is a per-frame pixel op
(z-score is ~invariant for tensor_speed, reshapes change/intensity); ``block_size``
is a geometry op whose expensive per-pixel tensor solve is actually block-size
independent (a future optimization, not needed now). See the branch plan.
"""
from __future__ import annotations

import time
from dataclasses import replace
from enum import Enum

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                             QPushButton, QSpinBox, QVBoxLayout, QWidget)

from core.channel_source import (LIVE_CHANNELS, live_channel_source,
                                 reduce_channel_data)
from core.config import FlowConfig, PipelineConfig, PreprocessConfig
from core.replicates import build_layout
from core.detection import detect_channel_region
from core.video import VideoSource
from core.wavelet import default_freqs
from gui.explorers.detection_timeline import DetectionNavigator
from gui.explorers.scalogram_explorer import ScalogramExplorer

# downsample spinbox sentinel: at/under this value means "auto" (derive from the
# default target width), matching the Preprocessing & Flow tab's convention.
_AUTO_DS = 0.05

# Idle labels for the two action buttons; each becomes "Stop" while its own pass
# runs (see _set_busy).
_EXTRACT_TEXT = "Extract"
_PROCESS_TEXT = "Process whole video ▶"


class _Busy(Enum):
    """Which pass owns the decoder. Named rather than bare strings because
    _set_busy silently disables both buttons for any value it does not
    recognise, which a mistyped literal would make unrecoverable."""
    EXTRACT = "extract"
    PROCESS = "process"


class _Cancelled(Exception):
    """Raised inside a worker's progress callback to unwind a cancelled pass."""


class _StreamWorker(QThread):
    """Base for the two streaming passes, giving both a uniform cancel path.

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
            raise _Cancelled
        self.progress.emit(done, total)

    def run(self):
        try:
            self._run()
        except _Cancelled:
            self.cancelled.emit()
        except Exception as e:                     # surface any extraction error
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _run(self):
        raise NotImplementedError


class _LiveExtractWorker(_StreamWorker):
    """Run a windowed structure-tensor pass off the GUI thread. A full-resolution
    window is a real streaming solve (seconds), so blocking here would freeze the
    knobs mid-drag."""

    def __init__(self, video_path, cfg, replicates, start, n, dims, parent=None):
        super().__init__(parent)
        self._args = (video_path, cfg, replicates, start, n, dims)

    def _run(self):
        video_path, cfg, reps, start, n, dims = self._args
        w, h, fps, fc = dims
        cd = live_channel_source(
            video_path, cfg, reps, start=start, n=n,
            width=w, height=h, fps=fps, frame_count=fc, progress=self._tick)
        self.done.emit(cd)


class _ProcessWorker(_StreamWorker):
    """The whole-video commit: stream the WHOLE clip's channels once, then run the
    tuned detector over the selected region. No flow, no cache -- just the
    detection track. This is the one expensive pass, paid after tuning."""
    phase = pyqtSignal(str)          # a coarse phase change (e.g. detection start)

    def __init__(self, video_path, cfg, replicates, dims, region_index, params,
                 parent=None):
        super().__init__(parent)
        self._args = (video_path, cfg, replicates, dims, region_index, params)

    def _run(self):
        video_path, cfg, reps, dims, region_index, params = self._args
        w, h, fps, fc = dims
        cd = live_channel_source(
            video_path, cfg, reps, start=0, n=None,
            width=w, height=h, fps=fps, frame_count=fc, progress=self._tick)
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


class LiveScalogramSurface(QWidget):
    def __init__(self, video_path: str, replicates: list[dict],
                 base_cfg: PipelineConfig | None = None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.replicates = list(replicates)
        cfg = base_cfg or PipelineConfig()

        with VideoSource(video_path) as src:
            info = src.info
        self._dims = (info.width, info.height, float(info.fps),
                      int(info.frame_count))
        self.fps = float(info.fps)
        self.frame_count = int(info.frame_count)

        self._explorer: ScalogramExplorer | None = None
        self._worker: _LiveExtractWorker | None = None
        self._proc_worker: _ProcessWorker | None = None
        self._pending_state: dict | None = None
        # Progress-readout timing/context, set when each pass starts.
        self._extract_t0 = 0.0
        self._extract_ctx = ""
        self._proc_t0 = 0.0
        self._proc_ctx = ""
        # Pixel-level (block_size=1) channel cache for the current window: a Block
        # change re-reduces this instead of re-extracting (the per-pixel tensor
        # solve is block independent). Keyed by everything upstream of the block:
        # window (start, n), downsample, normalize. None when the per-pixel
        # footprint would exceed the budget (then Block falls back to re-extract).
        self._pp: object | None = None
        self._pp_key: tuple | None = None
        self._pp_budget = 2 * 1024 ** 3
        self._pending_pp = False        # is the running extract building a cache?
        self._pending_key: tuple | None = None
        self._pending_cfg: PipelineConfig | None = None
        # Set when a knob change stops an in-flight extract: the replacement pass
        # starts from the cancelled handler, once the old thread has unwound.
        self._restart_extract = False

        # Coalesce rapid knob edits into a single re-extract on settle. Created
        # before the strip because the controls connect to it.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(500)
        self._debounce.timeout.connect(self.extract)

        # Block changes re-reduce the cached per-pixel channels (fast), so they get
        # their own shorter-settle handler distinct from the re-extract debounce.
        self._block_debounce = QTimer(self)
        self._block_debounce.setSingleShot(True)
        self._block_debounce.setInterval(250)
        self._block_debounce.timeout.connect(self._on_block_changed)

        root = QVBoxLayout(self)
        root.addWidget(self._build_strip(cfg))

        self._host = QWidget()
        self._host_lay = QVBoxLayout(self._host)
        self._host_lay.setContentsMargins(0, 0, 0, 0)
        self._placeholder = QLabel(
            "Set a window and press Extract to build the live scalogram.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color:#8ab; padding:40px;")
        self._host_lay.addWidget(self._placeholder)
        root.addWidget(self._host, 1)

        # Whole-clip detection navigator (hidden until a commit pass lands).
        self.navigator = DetectionNavigator()
        self.navigator.focus_requested.connect(self._focus_frame)
        self.navigator.setVisible(False)
        root.addWidget(self.navigator)

        # First extract once shown, so the window opens without a blocking pass.
        QTimer.singleShot(0, self.extract)

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
        self.start_slider.valueChanged.connect(self._on_window_changed)
        row.addWidget(self.start_slider)
        self.start_lbl = QLabel("0.00 s")
        self.start_lbl.setMinimumWidth(70)
        row.addWidget(self.start_lbl)

        row.addWidget(QLabel("Length"))
        self.len_spin = QDoubleSpinBox()
        max_len = min(60.0, self.frame_count / max(self.fps, 1e-6))
        self.len_spin.setRange(0.2, max(0.2, max_len))
        self.len_spin.setValue(min(10.0, max(0.2, max_len)))
        self.len_spin.setSuffix(" s")
        self.len_spin.valueChanged.connect(self._on_window_changed)
        row.addWidget(self.len_spin)

        row.addSpacing(12)
        row.addWidget(QLabel("Downsample"))
        self.ds_spin = QDoubleSpinBox()
        self.ds_spin.setRange(_AUTO_DS, 1.0)
        self.ds_spin.setSingleStep(0.05)
        self.ds_spin.setDecimals(3)
        # "auto" is derived from the source width and is fixed for this clip, so
        # resolve it once and show the factor it actually stands for.
        auto_scale = PreprocessConfig(downsample=None).resolve_downsample(
            self._dims[0])
        self.ds_spin.setSpecialValueText(f"auto ({auto_scale:.3f})")
        self.ds_spin.setValue(cfg.preprocess.downsample
                              if cfg.preprocess.downsample else _AUTO_DS)
        self.ds_spin.valueChanged.connect(self._debounce.start)
        row.addWidget(self.ds_spin)

        row.addWidget(QLabel("Block"))
        self.block_spin = QSpinBox()
        self.block_spin.setRange(1, 64)
        self.block_spin.setValue(int(cfg.flow.block_size))
        self.block_spin.setToolTip(
            "Pixels per block. Sets the scalogram grid (and cube memory); the "
            "per-pixel tensor solve is block-size independent, so a change here "
            "re-reduces the cached pixels instead of re-extracting.")
        self.block_spin.valueChanged.connect(self._block_debounce.start)
        row.addWidget(self.block_spin)

        row.addWidget(QLabel("Normalize"))
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["off", "zscore", "clahe"])
        self.norm_combo.setCurrentText(cfg.preprocess.normalize)
        self.norm_combo.setToolTip(
            "Upstream per-frame pixel op (re-extracts). z-score is ~invariant "
            "for tensor_speed; reshapes change/intensity. CLAHE has a known "
            "replicate-edge artifact.")
        self.norm_combo.currentTextChanged.connect(self._debounce.start)
        row.addWidget(self.norm_combo)

        # Two stacked, right-aligned status lines sit inline to the left of the
        # action buttons and take the row's slack: the extract / whole-video
        # compute on top, and the hosted explorer's graph-compute line just below
        # it. Darker orange + larger so they read against the light control strip.
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
        self.extract_btn = QPushButton(_EXTRACT_TEXT)
        self.extract_btn.clicked.connect(self._on_extract_clicked)
        row.addWidget(self.extract_btn)
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

    def _on_window_changed(self, *_):
        self._sync_window_label()
        self._debounce.start()

    def _sync_window_label(self):
        self.start_lbl.setText(f"{self.start_slider.value() / self.fps:.2f} s")

    # -- config assembly -----------------------------------------------------
    def _build_cfg(self) -> PipelineConfig:
        ds = self.ds_spin.value()
        return PipelineConfig(
            preprocess=PreprocessConfig(
                downsample=None if ds <= _AUTO_DS else float(ds),
                normalize=self.norm_combo.currentText(),
                denoise="off", registration="off", bg_subtract="off",
                mask_path=None),
            flow=FlowConfig(block_size=int(self.block_spin.value())),
        )

    def _window(self) -> tuple[int, int]:
        start = int(self.start_slider.value())
        n = max(2, int(round(self.len_spin.value() * self.fps)))
        n = min(n, self.frame_count - start)
        return start, n

    # -- button state --------------------------------------------------------
    def _set_busy(self, which: _Busy | None):
        """Reflect which pass is running in the two action buttons: the running
        one becomes Stop, the other is disabled (only one streaming pass at a
        time). ``None`` restores the idle labels."""
        extracting = which is _Busy.EXTRACT
        processing = which is _Busy.PROCESS
        self.extract_btn.setText("Stop" if extracting else _EXTRACT_TEXT)
        self.process_btn.setText("Stop" if processing else _PROCESS_TEXT)
        self.extract_btn.setEnabled(which is None or extracting)
        self.process_btn.setEnabled(which is None or processing)

    def _on_extract_clicked(self):
        if self._worker is not None:
            self._request_stop(self._worker, "extract")
        else:
            self.extract()

    def _on_process_clicked(self):
        if self._proc_worker is not None:
            self._request_stop(self._proc_worker, "whole-video pass")
        else:
            self.process_whole_video()

    def _request_stop(self, worker, what: str):
        """Ask a running pass to unwind. It stops at its next progress tick, so
        both buttons stay disabled until the worker's ``cancelled`` lands."""
        worker.cancel()
        self.extract_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.status_lbl.setText(f"stopping the {what}…")

    # -- extraction ----------------------------------------------------------
    def extract(self):
        if self._proc_worker is not None:
            return                          # the expensive commit owns the decoder
        if self._worker is not None:
            # A knob settled mid-extract: that pass is now stale, so supersede it
            # rather than dropping the edit. The restart runs once it unwinds.
            self._restart_extract = True
            self._request_stop(self._worker, "extract")
            return
        self._debounce.stop()
        start, n = self._window()
        if n < 2:
            self.status_lbl.setText("window too short at this start position")
            return
        cfg = self._build_cfg()
        if self._explorer is not None:
            self._pending_state = self._explorer.capture_view_state()
        # Extract at pixel level (block=1) when the per-pixel footprint fits, so a
        # later Block change is a cheap re-reduce; over budget, extract directly at
        # the block (no cache, Block falls back to re-extract).
        fits = self._perpixel_fits(cfg, n)
        self._pending_pp = fits
        self._pending_key = self._pp_signature(cfg, start, n) if fits else None
        self._pending_cfg = cfg
        extract_cfg = self._cfg_block1(cfg) if fits else cfg
        self._set_busy(_Busy.EXTRACT)
        self._extract_t0 = time.monotonic()
        self._extract_ctx = (f"block {cfg.flow.block_size}, "
                             f"norm {cfg.preprocess.normalize}")
        self.status_lbl.setText(
            f"extracting {n} frames from {start / self.fps:.2f} s "
            f"({self._extract_ctx})…")
        self._worker = _LiveExtractWorker(
            self.video_path, extract_cfg, self.replicates, start, n,
            self._dims, self)
        self._worker.done.connect(self._on_extracted)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_extract_cancelled)
        self._worker.progress.connect(self._on_extract_progress)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_extract_progress(self, done: int, total: int):
        if self._worker is None or self._worker.is_cancelled():
            return          # keep the "stopping…" note; a tick may still be in flight
        pct = 100.0 * done / max(1, total)
        self.status_lbl.setText(
            f"extracting… {done}/{total} frames ({pct:.0f}%) · "
            f"{self._extract_ctx}{self._eta(done, total, self._extract_t0)}")

    def _on_extracted(self, cd):
        self._worker = None
        self._set_busy(None)
        cfg = self._pending_cfg or self._build_cfg()
        if self._pending_pp:                         # cd is the block=1 pixel cache
            self._pp = cd
            self._pp_key = self._pending_key
            display = reduce_channel_data(cd, cfg, self.replicates)
        else:
            self._pp = None
            self._pp_key = None
            display = cd
        if self._restart_extract:
            # A knob change asked to supersede this pass, but it finished before
            # the cancel reached a tick, so `cancelled` never fired. Its channels
            # are still a valid cache for those settings -- keep them, skip the
            # now-stale display, and run the pass the user actually asked for.
            # Leaving the flag set here would also make the next Stop restart.
            self._restart_extract = False
            self.extract()
            return
        self._show_channel_data(display)

    def _show_channel_data(self, cd):
        approx = " · approximated" if cd.approximated else ""
        self.status_lbl.setText(
            f"window {cd.window_start / self.fps:.2f} s · {cd.n_frames} frames · "
            f"{cd.meta['grid'][0]}×{cd.meta['grid'][1]} blocks{approx}")
        self._swap_explorer(cd)

    # -- pixel-cache helpers -------------------------------------------------
    def _pp_signature(self, cfg: PipelineConfig, start: int, n: int) -> tuple:
        """Everything upstream of the block size: a change here invalidates the
        cached per-pixel channels, a Block change does not."""
        scale = cfg.preprocess.resolve_downsample(self._dims[0])
        return (start, n, round(float(scale), 6), cfg.preprocess.normalize)

    def _perpixel_fits(self, cfg: PipelineConfig, n: int) -> bool:
        w, h = self._dims[0], self._dims[1]
        scale = cfg.preprocess.resolve_downsample(w)
        ny, nx = build_layout(self.replicates, w, h, scale, 1).atlas_grid
        return n * ny * nx * len(LIVE_CHANNELS) * 4 <= self._pp_budget

    @staticmethod
    def _cfg_block1(cfg: PipelineConfig) -> PipelineConfig:
        return replace(cfg, flow=replace(cfg.flow, block_size=1))

    def _on_block_changed(self):
        """Block changed: re-reduce the cached pixel channels (instant) if they
        cover this window, else fall back to a full (block=1) extract."""
        if self._proc_worker is not None:
            return
        if self._worker is not None:
            # Mid-extract: even a cache hit would be overwritten by the pass in
            # flight, so supersede it with a full re-extract at the new block.
            self.extract()
            return
        self._block_debounce.stop()
        start, n = self._window()
        if n < 2:
            self.status_lbl.setText("window too short at this start position")
            return
        cfg = self._build_cfg()
        if self._pp is not None and self._pp_key == self._pp_signature(cfg, start, n):
            if self._explorer is not None:
                self._pending_state = self._explorer.capture_view_state()
            self.status_lbl.setText(
                f"re-reducing to block {cfg.flow.block_size} — no re-extract…")
            self.status_lbl.repaint()
            self._show_channel_data(
                reduce_channel_data(self._pp, cfg, self.replicates))
        else:
            self.extract()

    def _on_failed(self, msg: str):
        self._worker = None
        self._restart_extract = False
        self._set_busy(None)
        self.status_lbl.setText(f"extract failed: {msg}")

    def _on_extract_cancelled(self):
        """The stopped pass produced nothing, so the pixel cache and the pending
        bookkeeping it would have filled are left as they were."""
        self._worker = None
        self._pending_pp = False
        self._pending_key = None
        self._pending_cfg = None
        self._set_busy(None)
        if self._restart_extract:      # a knob change superseded it; run the new one
            self._restart_extract = False
            self.extract()
            return
        self.status_lbl.setText("extract stopped")

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
        if self._pending_state is not None:
            new.apply_view_state(self._pending_state)
            self._pending_state = None
        if old is not None:
            old.close()                             # releases source + event filter
            old.setParent(None)
            old.deleteLater()

    # -- whole-video commit --------------------------------------------------
    def process_whole_video(self):
        if self._worker is not None or self._proc_worker is not None:
            return                                  # a pass is already running
        if self._explorer is None:
            self.status_lbl.setText("extract a window and tune it first")
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
        self._proc_worker = None
        self._set_busy(None)
        self.navigator.set_result(res, self.fps)
        self.navigator.setVisible(True)
        n = len(res.detected_intervals())
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
        """A detection was chosen: load a window centered on it for verification."""
        n = max(2, int(round(self.len_spin.value() * self.fps)))
        start = int(np.clip(center - n // 2, 0, max(0, self.frame_count - n)))
        self.navigator.set_cursor(center)
        self.start_slider.blockSignals(True)
        self.start_slider.setValue(start)
        self.start_slider.blockSignals(False)
        self._sync_window_label()
        self.extract()

    def toggle_playback(self):
        """Space handler the main window's focus-walk finds; drives the hosted
        explorer so the embedded explorer needs no Space shortcut of its own."""
        if self._explorer is not None:
            self._explorer.toggle_playback()

    def closeEvent(self, e):
        self._restart_extract = False
        # A knob edited just before the close leaves a debounce armed; let it fire
        # and it re-enters extract() on a surface that is on its way out.
        self._debounce.stop()
        self._block_debounce.stop()
        for w in (self._worker, self._proc_worker):
            if w is not None:
                w.cancel()      # unwind at the next tick instead of waiting it out
                w.wait()
        self._worker = self._proc_worker = None
        if self._explorer is not None:
            self._explorer.close()
        super().closeEvent(e)
