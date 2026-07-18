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

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout,
                             QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from core.channel_source import live_channel_source
from core.config import FlowConfig, PipelineConfig, PreprocessConfig
from core.detection import detect_channel_region
from core.video import VideoSource
from core.wavelet import default_freqs
from gui.explorers.detection_timeline import DetectionNavigator
from gui.explorers.scalogram_explorer import ScalogramExplorer

# downsample spinbox sentinel: at/under this value means "auto" (derive from the
# default target width), matching the Preprocessing & Flow tab's convention.
_AUTO_DS = 0.05


class _LiveExtractWorker(QThread):
    """Run a windowed structure-tensor pass off the GUI thread. A full-resolution
    window is a real streaming solve (seconds), so blocking here would freeze the
    knobs mid-drag."""
    done = pyqtSignal(object)       # ChannelData
    failed = pyqtSignal(str)

    def __init__(self, video_path, cfg, replicates, start, n, dims, parent=None):
        super().__init__(parent)
        self._args = (video_path, cfg, replicates, start, n, dims)

    def run(self):
        video_path, cfg, reps, start, n, dims = self._args
        w, h, fps, fc = dims
        try:
            cd = live_channel_source(video_path, cfg, reps, start=start, n=n,
                                     width=w, height=h, fps=fps, frame_count=fc)
            self.done.emit(cd)
        except Exception as e:                     # surface any extraction error
            self.failed.emit(f"{type(e).__name__}: {e}")


class _ProcessWorker(QThread):
    """The whole-video commit: stream the WHOLE clip's channels once, then run the
    tuned detector over the selected region. No flow, no cache -- just the
    detection track. This is the one expensive pass, paid after tuning."""
    done = pyqtSignal(object)       # DetectionResult
    failed = pyqtSignal(str)

    def __init__(self, video_path, cfg, replicates, dims, region_index, params,
                 parent=None):
        super().__init__(parent)
        self._args = (video_path, cfg, replicates, dims, region_index, params)

    def run(self):
        video_path, cfg, reps, dims, region_index, params = self._args
        w, h, fps, fc = dims
        try:
            cd = live_channel_source(video_path, cfg, reps, start=0, n=None,
                                     width=w, height=h, fps=fps, frame_count=fc)
            res = detect_channel_region(
                cd, region_index, params["channel_attr"],
                freqs=default_freqs(fps), freq_band_hz=params["freq_band_hz"],
                value_band=params["value_band"], count_band=params["count_band"],
                detect_window=params["detect_window"], centered=params["centered"])
            self.done.emit(res)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


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

        # Coalesce rapid knob edits into a single re-extract on settle. Created
        # before the strip because the controls connect to it.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(500)
        self._debounce.timeout.connect(self.extract)

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
        box = QGroupBox("Live preprocessing  ·  no flow cache")
        outer = QVBoxLayout(box)
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
        self.ds_spin.setSpecialValueText("auto")
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
            "per-pixel tensor solve is block-size independent.")
        self.block_spin.valueChanged.connect(self._debounce.start)
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

        row.addStretch(1)
        self.extract_btn = QPushButton("Extract")
        self.extract_btn.clicked.connect(self.extract)
        row.addWidget(self.extract_btn)
        self.process_btn = QPushButton("Process whole video ▶")
        self.process_btn.setToolTip(
            "Run the tuned detector over the WHOLE clip (one streaming pass, no "
            "flow, no cache) and navigate the detections below.")
        self.process_btn.clicked.connect(self.process_whole_video)
        row.addWidget(self.process_btn)
        outer.addLayout(row)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(
            "color:#e0a94a; font-family:Consolas; font-size:11px;")
        outer.addWidget(self.status_lbl)

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

    # -- extraction ----------------------------------------------------------
    def extract(self):
        if self._worker is not None or self._proc_worker is not None:
            return                                  # a pass is already running
        self._debounce.stop()
        start, n = self._window()
        if n < 2:
            self.status_lbl.setText("window too short at this start position")
            return
        cfg = self._build_cfg()
        if self._explorer is not None:
            self._pending_state = self._explorer.capture_view_state()
        self.extract_btn.setEnabled(False)
        self.status_lbl.setText(
            f"extracting {n} frames from {start / self.fps:.2f} s "
            f"(block {cfg.flow.block_size}, norm {cfg.preprocess.normalize})…")
        self._worker = _LiveExtractWorker(
            self.video_path, cfg, self.replicates, start, n, self._dims, self)
        self._worker.done.connect(self._on_extracted)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_extracted(self, cd):
        self._worker = None
        self.extract_btn.setEnabled(True)
        approx = " · approximated" if cd.approximated else ""
        self.status_lbl.setText(
            f"window {cd.window_start / self.fps:.2f} s · {cd.n_frames} frames · "
            f"{cd.meta['grid'][0]}×{cd.meta['grid'][1]} blocks{approx}")
        self._swap_explorer(cd)

    def _on_failed(self, msg: str):
        self._worker = None
        self.extract_btn.setEnabled(True)
        self.status_lbl.setText(f"extract failed: {msg}")

    def _swap_explorer(self, cd):
        if self._placeholder is not None:
            self._placeholder.setParent(None)
            self._placeholder.deleteLater()
            self._placeholder = None
        old = self._explorer
        new = ScalogramExplorer.from_channel_data(
            cd, video_path=self.video_path, own_shortcuts=False, parent=self._host)
        self._host_lay.addWidget(new)
        self._explorer = new
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
        self.process_btn.setEnabled(False)
        self.extract_btn.setEnabled(False)
        flo, fhi = params["freq_band_hz"]
        self.status_lbl.setText(
            f"processing whole video ({self.frame_count} frames) · "
            f"{flo:.2f}–{fhi:.2f} Hz on {params['channel_attr']}… this is the "
            f"one expensive pass")
        self._proc_worker = _ProcessWorker(
            self.video_path, cfg, self.replicates, self._dims,
            params["region_index"], params, self)
        self._proc_worker.done.connect(self._on_processed)
        self._proc_worker.failed.connect(self._on_process_failed)
        self._proc_worker.finished.connect(self._proc_worker.deleteLater)
        self._proc_worker.start()

    def _on_processed(self, res):
        self._proc_worker = None
        self.process_btn.setEnabled(True)
        self.extract_btn.setEnabled(True)
        self.navigator.set_result(res, self.fps)
        self.navigator.setVisible(True)
        n = len(res.detected_intervals())
        self.status_lbl.setText(
            f"whole-video pass done · {n} detection{'s' if n != 1 else ''} — "
            f"click one (or step strongest) to verify in a window")

    def _on_process_failed(self, msg: str):
        self._proc_worker = None
        self.process_btn.setEnabled(True)
        self.extract_btn.setEnabled(True)
        self.status_lbl.setText(f"process failed: {msg}")

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
        for w in (self._worker, self._proc_worker):
            if w is not None:
                w.wait()
        self._worker = self._proc_worker = None
        if self._explorer is not None:
            self._explorer.close()
        super().closeEvent(e)
