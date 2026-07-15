"""Preprocessing & Flow: configuration and the expensive cache pass.

The design rule here is that the user should never be surprised by a two-hour
compute or a 20 GB file. Every control that costs time or disk shows its own cost
inline, recomputed from the actual loaded video's resolution and duration as soon
as anything changes.
"""
from __future__ import annotations

import json
import os
import shutil
import time

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QMessageBox, QProgressBar, QPushButton, QScrollArea,
                             QSpinBox, QVBoxLayout, QWidget)

from core import cache as cache_mod
from core.cache import estimate_cache_bytes, human_bytes
from core.config import (Band, FeatureConfig, FlowConfig, PipelineConfig,
                         PreprocessConfig)
from core.flow import backend_status
from core.pipeline import (Cancelled, _flow_atlas_geometry,
                           _flow_support_pixels, run_pipeline)
from core.replicates import build_layout, geometry_hash, validate_replicates
from gui.help import HelpButton, labelled
from gui.state import AppState

# Calibrated from the ROI-first FFmpeg input path on the reference clip. At
# 166,980 privately-supported solver pixels/frame, Farneback processed ~40.5
# fps; normalized to the former 1300x731 reference area that is ~7.1 fps.
# Used only for ETA; layout and backend still introduce run-to-run variation.
_REF_FPS_AT_1300PX = 7.1
_REF_WIDTH = 1300
_BACKEND_SPEED = {"farneback": 1.0, "dis": 2.25, "raft": 6.0}


def _number_slug(value: float) -> str:
    """Compact filesystem-safe number used in descriptive cache names."""
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p")


def _test_cache_suffix(cfg: PipelineConfig, duration_s: float) -> str:
    """Describe every cache-affecting setting exposed by this tab.

    The leading cache hash remains authoritative and includes the complete
    configuration. This suffix is for humans comparing test runs in the cache
    picker and filesystem.
    """
    pre = cfg.preprocess
    flow = cfg.flow
    feat = cfg.features
    bands = "-".join(
        f"{_number_slug(b.lo_hz)}to{_number_slug(b.hi_hz)}"
        for b in feat.bands)
    downsample = "auto" if pre.downsample is None else \
        _number_slug(pre.downsample)
    precision = {"float16": "f16", "float32": "f32"}.get(
        feat.dtype, feat.dtype)
    parts = [
        f"test{_number_slug(duration_s)}s",
        flow.backend,
        f"b{flow.block_size}",
        f"ds{downsample}",
        f"reg{pre.registration}",
        f"den{pre.denoise}",
        f"bg{pre.bg_subtract}",
        f"norm{pre.normalize}",
        f"band{bands}",
        f"win{_number_slug(feat.window_s)}",
        f"hop{_number_slug(feat.hop_s)}",
        precision,
        feat.compression,
    ]
    if pre.mask_path:
        parts.append("mask")
    if feat.cache_fb_error:
        parts.append("fberr")
    if feat.cache_texture:
        parts.append("texture")
    return "_" + "_".join(parts)


class PipelineWorker(QThread):
    progress = pyqtSignal(object)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, video_path: str, cfg: PipelineConfig, cache_root: str,
                 duration_s: float | None, replicates: list[dict],
                 suffix: str = ""):
        super().__init__()
        self.video_path = video_path
        self.cfg = cfg
        self.cache_root = cache_root
        self.duration_s = duration_s
        self.replicates = replicates
        self.suffix = suffix
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            key = run_pipeline(
                self.video_path, self.cfg, self.cache_root,
                duration_s=self.duration_s,
                progress=self.progress.emit,
                should_cancel=lambda: self._cancel,
                cache_key_suffix=self.suffix,
                replicates=self.replicates,
            )
            self.finished_ok.emit(key)
        except Cancelled:
            self.failed.emit("Cancelled.")
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class Tab1Flow(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.worker: PipelineWorker | None = None
        self._last_full_min = 0.0
        self._last_test_min = 0.0

        root = QHBoxLayout(self)
        left = QVBoxLayout()
        right = QVBoxLayout()
        root.addLayout(left, 3)
        root.addLayout(right, 2)

        # -- video info ------------------------------------------------------
        # Qt's Fusion palette is LIGHT. Styling these panels with the dark
        # background borrowed from the histogram widgets left near-black text on
        # a near-black box. Light panel, dark text.
        self.info_label = QLabel("No video loaded.")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            "background:#eceff1; color:#000; font-weight:bold; padding:8px; "
            "border:1px solid #b0b7bc;")
        left.addWidget(self.info_label)

        self.nyquist_label = QLabel("")
        self.nyquist_label.setWordWrap(True)
        self.nyquist_label.setStyleSheet("color:#000; padding:4px;")
        left.addWidget(self.nyquist_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form_lay = QVBoxLayout(inner)
        scroll.setWidget(inner)
        left.addWidget(scroll, 1)

        # -- preprocessing ---------------------------------------------------
        pre_box = QGroupBox("Preprocessing")
        pre = QFormLayout(pre_box)

        self.downsample = QDoubleSpinBox()
        self.downsample.setRange(0.05, 1.0)
        self.downsample.setSingleStep(0.05)
        self.downsample.setDecimals(3)
        self.downsample.setSpecialValueText("auto")
        self.downsample.setValue(0.05)  # == "auto"
        self.downsample.valueChanged.connect(self._refresh_estimates)
        pre.addRow(labelled(self.downsample, "downsample"), QLabel("Downsample"))

        self.registration = QComboBox()
        self.registration.addItems(["off", "phase", "orb"])
        pre.addRow(labelled(self.registration, "registration"),
                   QLabel("Registration"))

        self.denoise = QComboBox()
        self.denoise.addItems(["off", "median", "gaussian"])
        pre.addRow(labelled(self.denoise, "denoise"), QLabel("Temporal denoise"))

        self.bg = QComboBox()
        self.bg.addItems(["off", "median", "mog2"])
        pre.addRow(labelled(self.bg, "bg_subtract"), QLabel("Background subtract"))

        self.normalize = QComboBox()
        self.normalize.addItems(["off", "clahe", "zscore"])
        pre.addRow(labelled(self.normalize, "normalize"), QLabel("Normalization"))

        self._mask_path: str | None = None
        self.mask_btn = QPushButton("No within-box mask")
        self.mask_btn.clicked.connect(self._pick_mask)
        draw_mask_btn = QPushButton("Draw…")
        draw_mask_btn.clicked.connect(self._draw_mask)
        mask_w = QWidget()
        mask_row = QHBoxLayout(mask_w)
        mask_row.setContentsMargins(0, 0, 0, 0)
        mask_row.addWidget(self.mask_btn, 1)
        mask_row.addWidget(draw_mask_btn)
        pre.addRow(labelled(mask_w, "mask"), QLabel("Within-replicate mask"))

        form_lay.addWidget(pre_box)

        # -- flow ------------------------------------------------------------
        flow_box = QGroupBox("Optical flow")
        flow = QFormLayout(flow_box)

        self.backend = QComboBox()
        self._status = backend_status()
        for b in ("farneback", "dis", "raft"):
            self.backend.addItem(b)
        self.backend.currentTextChanged.connect(self._on_backend_changed)
        flow.addRow(labelled(self.backend, "flow_backend"), QLabel("Backend"))

        self.backend_note = QLabel("")
        self.backend_note.setWordWrap(True)
        self.backend_note.setStyleSheet("color:#333; font-size:11px;")
        flow.addRow(self.backend_note)

        self.block = QSpinBox()
        # 1 means per-pixel. It is allowed, but the estimate panel will show what
        # it costs (~173 GB on 5.3K footage at 0.25 downsample) -- the flow is
        # already computed per-pixel regardless, so this only changes how much of
        # it is kept, not how good it is.
        self.block.setRange(1, 64)
        self.block.setSingleStep(1)
        self.block.setValue(4)
        self.block.valueChanged.connect(self._refresh_estimates)
        self.block.setToolTip(
            "Pixels per stored block. Smaller = finer regions, and it costs DISK "
            "but not compute — optical flow runs per-pixel either way, this only "
            "sets how much of it is kept. 8 is a good step up from 16. Below 4 "
            "the cache stops fitting in RAM when opened.")
        flow.addRow(labelled(self.block, "block_size"), QLabel("Block size"))

        form_lay.addWidget(flow_box)

        # -- features --------------------------------------------------------
        feat_box = QGroupBox("Features")
        feat = QFormLayout(feat_box)

        self.band_lo = QDoubleSpinBox()
        self.band_lo.setRange(0.0, 10000.0)
        self.band_lo.setValue(12.0)
        self.band_lo.setSuffix(" Hz")
        self.band_lo.valueChanged.connect(self._refresh_estimates)
        self.band_hi = QDoubleSpinBox()
        self.band_hi.setRange(0.1, 10000.0)
        self.band_hi.setValue(25.0)
        self.band_hi.setSuffix(" Hz")
        self.band_hi.valueChanged.connect(self._refresh_estimates)
        band_row = QHBoxLayout()
        band_row.addWidget(self.band_lo)
        band_row.addWidget(QLabel("to"))
        band_row.addWidget(self.band_hi)
        band_w = QWidget()
        band_w.setLayout(band_row)
        feat.addRow(labelled(band_w, "bands"), QLabel("Band power"))

        self.window_s = QDoubleSpinBox()
        self.window_s.setRange(0.05, 30.0)
        self.window_s.setValue(1.0)
        self.window_s.setSuffix(" s")
        self.window_s.valueChanged.connect(self._refresh_estimates)
        feat.addRow(labelled(self.window_s, "window_s"), QLabel("FFT window length"))

        self.hop_s = QDoubleSpinBox()
        self.hop_s.setRange(0.01, 10.0)
        self.hop_s.setValue(0.25)
        self.hop_s.setSuffix(" s")
        self.hop_s.valueChanged.connect(self._refresh_estimates)
        feat.addRow(labelled(self.hop_s, "hop_s"), QLabel("FFT hop interval"))

        self.dtype = QComboBox()
        self.dtype.addItems(["float16", "float32"])
        self.dtype.currentTextChanged.connect(self._refresh_estimates)
        feat.addRow(labelled(self.dtype, "dtype"), QLabel("Precision"))

        self.compression = QComboBox()
        self.compression.addItems(["zstd", "lz4", "none"])
        self.compression.currentTextChanged.connect(self._refresh_estimates)
        feat.addRow(labelled(self.compression, "compression"), QLabel("Compression"))

        self.cache_fb_error = QCheckBox(
            "DIAGNOSTIC: cache forward/backward error (~2.7× measured runtime)")
        self.cache_fb_error.setToolTip(
            "Computes backward flow while both frames are available, evaluates "
            "the exact pixelwise F(x)+B(x+F(x)) residual, and stores its p90 per "
            "block as a continuous feature. No rejection threshold is baked in. "
            "It runs inside each replicate and is disabled by default because "
            "doubling dense-flow solves is not a production setting for long "
            "recordings.")
        self.cache_fb_error.toggled.connect(self._refresh_estimates)
        feat.addRow(self.cache_fb_error)

        self.cache_texture = QCheckBox(
            "Cache texture / cornerness evidence (+1 block plane)")
        self.cache_texture.setToolTip(
            "Caches the block mean of the structure tensor's smaller eigenvalue. "
            "Tab 3 also exposes a per-frame texture percentile so q25 remains "
            "tunable at analysis time.")
        self.cache_texture.toggled.connect(self._refresh_estimates)
        feat.addRow(self.cache_texture)

        form_lay.addWidget(feat_box)

        note = QLabel(
            "The cache stores flow (u, v, speed) plus band power. Everything else "
            "— coherence, divergence, curl, spectral flatness, direction "
            "oscillation, spatial median views, relative-noise views, and band "
            "power for any pass-band — is derived on demand. FB error and texture "
            "are the exceptions because they require the original frames.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#555; font-size:11px; font-style:italic;")
        form_lay.addWidget(note)
        form_lay.addStretch(1)

        # -- right: estimates + run -----------------------------------------
        est_box = QGroupBox("Estimated cost")
        est = QVBoxLayout(est_box)

        # The headline number is the cache size AS A MULTIPLE OF THE SOURCE VIDEO,
        # not an absolute figure. "54 GB" means nothing without a reference point;
        # "14x your original file" is immediately legible as a bad idea.
        self.size_banner = QLabel("")
        self.size_banner.setWordWrap(True)
        self.size_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        est.addWidget(self.size_banner)

        self.estimate_label = QLabel("Load a video to see estimates.")
        self.estimate_label.setWordWrap(True)
        # Black. The app runs on Qt's light Fusion palette, so the near-white this
        # used to be was invisible against the panel behind it.
        self.estimate_label.setStyleSheet(
            "font-family: Consolas; font-size: 11px; color:#000;")
        est.addWidget(self.estimate_label)
        right.addWidget(est_box)

        run_box = QGroupBox("Run")
        run = QVBoxLayout(run_box)

        self.test_seconds = QDoubleSpinBox()
        self.test_seconds.setRange(1.0, 600.0)
        self.test_seconds.setValue(10.0)
        self.test_seconds.setSuffix(" s")
        tr = QHBoxLayout()
        tr.addWidget(QLabel("Test duration"))
        tr.addWidget(self.test_seconds, 1)
        tr.addWidget(HelpButton("test_seconds"))
        trw = QWidget()
        trw.setLayout(tr)
        run.addWidget(trw)

        # Both buttons spell out exactly how many frames and how long, because
        # "Run test" next to "Run full pass" is otherwise one misclick away from
        # a two-hour job, and a 10 s test still takes ~2 minutes -- long enough
        # that a user reasonably wonders whether it ignored them.
        self.test_btn = QPushButton("Run TEST")
        self.test_btn.setStyleSheet("padding:6px; font-weight:bold;")
        self.test_btn.clicked.connect(lambda: self._run(test=True))
        run.addWidget(self.test_btn)

        self.full_btn = QPushButton("Run FULL pass")
        self.full_btn.setStyleSheet("padding:6px;")
        self.full_btn.clicked.connect(lambda: self._run(test=False))
        run.addWidget(self.full_btn)

        self.test_seconds.valueChanged.connect(self._refresh_estimates)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        run.addWidget(self.cancel_btn)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        run.addWidget(self.progress)

        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.progress_label.setStyleSheet("color:#000; font-size:11px;")
        run.addWidget(self.progress_label)

        right.addWidget(run_box)

        # -- existing caches -------------------------------------------------
        cache_box = QGroupBox("Existing caches")
        cl = QVBoxLayout(cache_box)
        self.cache_list = QComboBox()
        cl.addWidget(self.cache_list)
        open_btn = QPushButton("Open selected cache")
        open_btn.clicked.connect(self._open_selected)
        cl.addWidget(open_btn)

        crow = QHBoxLayout()
        del_btn = QPushButton("Delete selected")
        del_btn.clicked.connect(self._delete_selected_cache)
        folder_btn = QPushButton("Open cache folder")
        folder_btn.clicked.connect(self._open_cache_folder)
        crow.addWidget(del_btn)
        crow.addWidget(folder_btn)
        cl.addLayout(crow)

        clear_btn = QPushButton("Clear ALL caches")
        clear_btn.clicked.connect(self._clear_all_caches)
        cl.addWidget(clear_btn)

        self.cache_disk_label = QLabel("")
        self.cache_disk_label.setStyleSheet("color:#333; font-size:11px;")
        cl.addWidget(self.cache_disk_label)
        right.addWidget(cache_box)

        # -- settings io -----------------------------------------------------
        io_box = QGroupBox("Settings")
        io = QHBoxLayout(io_box)
        exp_btn = QPushButton("Export JSON")
        exp_btn.clicked.connect(self._export_settings)
        imp_btn = QPushButton("Import JSON")
        imp_btn.clicked.connect(self._import_settings)
        io.addWidget(exp_btn)
        io.addWidget(imp_btn)
        right.addWidget(io_box)
        right.addStretch(1)

        self.state.video_loaded.connect(self._on_video_loaded)
        self.state.rois_changed.connect(self._refresh_estimates)
        self._on_backend_changed(self.backend.currentText())
        self._refresh_cache_list()

    # -- config ------------------------------------------------------------

    def build_config(self) -> PipelineConfig:
        ds = self.downsample.value()
        return PipelineConfig(
            preprocess=PreprocessConfig(
                mask_path=self._mask_path,
                registration=self.registration.currentText(),
                denoise=self.denoise.currentText(),
                bg_subtract=self.bg.currentText(),
                downsample=None if ds <= 0.05 else float(ds),
                normalize=self.normalize.currentText(),
            ),
            flow=FlowConfig(
                backend=self.backend.currentText(),
                block_size=self.block.value(),
            ),
            features=FeatureConfig(
                bands=(Band(self.band_lo.value(), self.band_hi.value()),),
                window_s=self.window_s.value(),
                hop_s=self.hop_s.value(),
                cache_fb_error=self.cache_fb_error.isChecked(),
                cache_texture=self.cache_texture.isChecked(),
                dtype=self.dtype.currentText(),
                compression=self.compression.currentText(),
            ),
        )

    def _on_video_loaded(self):
        info = self.state.source.info
        self.info_label.setText(info.describe())
        band = self.state.cfg.features.bands[0]
        self.band_lo.blockSignals(True)
        self.band_hi.blockSignals(True)
        self.band_lo.setValue(band.lo_hz)
        self.band_hi.setValue(band.hi_hz)
        self.band_lo.blockSignals(False)
        self.band_hi.blockSignals(False)
        self.band_lo.setMaximum(info.nyquist_hz)
        self.band_hi.setMaximum(info.nyquist_hz)
        self._refresh_estimates()
        self._refresh_cache_list()

    def _on_backend_changed(self, name: str):
        msg = self._status.get(name, "")
        available = msg.startswith("Available")
        self.backend_note.setText(msg)
        self.backend_note.setStyleSheet(
            "color:#333; font-size:11px;" if available
            else "color:#b00020; font-weight:bold; font-size:11px;")
        self._refresh_estimates()

    def _pick_mask(self):
        # Clicking when a mask is set clears it; a set-and-forget mask you can't
        # remove is a trap.
        if self._mask_path:
            self._mask_path = None
            self.mask_btn.setText("No within-box mask")
            self._refresh_estimates()
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Within-replicate mask (white = keep)", "",
            "Images (*.png *.jpg *.bmp)")
        if path:
            self._mask_path = path
            self.mask_btn.setText(f"✕ {os.path.basename(path)}")

    def _draw_mask(self):
        if not self.state.has_video:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        from gui.mask_dialog import MaskDrawDialog
        frame = self.state.source.frame_at(self.state.current_frame)
        if frame is None:
            frame = self.state.source.frame_at(0)
        out = os.path.join(self.state.cache_root, "_masks",
                           f"{self.state.source.info.video_hash}_mask.png")
        dlg = MaskDrawDialog(frame, out, self)
        if dlg.exec():
            self._mask_path = out
            self.mask_btn.setText("✕ drawn mask")
            self._refresh_estimates()

    # -- estimates ---------------------------------------------------------

    def _refresh_estimates(self):
        if not self.state.has_video:
            return
        info = self.state.source.info
        cfg = self.build_config()
        reps = self.state.replicate_specs

        if not reps:
            self.size_banner.setText(
                "Define replicate boxes in the Replicates tab first")
            self.size_banner.setStyleSheet(
                "background:#fff176; color:#000; font-weight:bold; "
                "padding:6px; border:1px solid #444;")
            self.estimate_label.setText(
                "ROI-first processing requires at least one replicate box.\n"
                "Decode is shared, but preprocessing and flow run independently "
                "inside each box.")
            return

        try:
            layout = build_layout(
                reps, info.width, info.height,
                cfg.preprocess.resolve_downsample(info.width),
                cfg.flow.block_size)
        except ValueError as e:
            self.size_banner.setText("Replicate geometry needs attention")
            self.estimate_label.setText(str(e))
            return

        warnings = cfg.features.validate_bands(info.fps)
        self.nyquist_label.setText(
            "  ".join(warnings) if warnings else
            f"Band {cfg.features.bands[0].lo_hz:g}–"
            f"{cfg.features.bands[0].hi_hz:g} Hz is safely below the "
            f"{info.nyquist_hz:.1f} Hz Nyquist limit.")
        self.nyquist_label.setStyleSheet(
            "color:#000; background:#fff176; padding:4px; border:1px solid #c9b458;"
            if warnings else
            "color:#000; background:#c8e6c9; padding:4px; border:1px solid #91b894;")

        sizes = estimate_cache_bytes(cfg, info.width, info.height,
                                     info.frame_count, info.fps, reps)
        total = sum(sizes.values())

        scale = layout.scale
        ny, nx = layout.atlas_grid
        support = _flow_support_pixels(cfg)
        flow_shape, _ = _flow_atlas_geometry(layout, support)
        compute_pixels = flow_shape[0] * flow_shape[1]
        reference_pixels = _REF_WIDTH * 731
        rate = (_REF_FPS_AT_1300PX * reference_pixels / max(1, compute_pixels)
                * _BACKEND_SPEED.get(cfg.flow.backend, 1.0))
        if cfg.features.cache_fb_error:
            rate /= 2.7
        full_s = info.frame_count / max(0.01, rate)
        test_s = min(info.frame_count, self.test_seconds.value() * info.fps) \
            / max(0.01, rate)
        self._last_full_min = full_s / 60.0
        self._last_test_min = test_s / 60.0

        self._update_size_banner(total)

        lines = [
            f"replicates    {len(layout.tiles)} independent exact crops",
            f"decoder       {'FFmpeg ROI stream' if shutil.which('ffmpeg') else 'OpenCV full-frame FALLBACK'}",
            f"owned pixels  {layout.work_pixels_per_frame:,}/frame "
            f"(downsample {scale:.3f})",
            f"flow pixels   {compute_pixels:,}/frame incl. private support",
            f"packed grid   {ny} x {nx}  ({ny * nx} cells/frame)",
            f"frames        {info.frame_count}",
            "",
            "cache, uncompressed:",
        ]
        for name, b in sorted(sizes.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:<28} {human_bytes(b):>10}")
        lines += [
            f"  {'TOTAL':<28} {human_bytes(total):>10}",
            f"  (zstd typically takes ~25% off flow data)",
            "",
            f"full pass     ~{full_s / 60:.0f} min",
            f"test pass     ~{test_s:.0f} s",
        ]
        if cfg.features.cache_fb_error:
            lines += [
                "",
                "WARNING: FB error doubles per-replicate flow solves.",
                "Use it on sampled QC windows, not the full corpus.",
            ]
        self.estimate_label.setText("\n".join(lines))

        test_frames = int(min(info.frame_count,
                              self.test_seconds.value() * info.fps))
        self.test_btn.setText(
            f"Run TEST — {self.test_seconds.value():.0f} s "
            f"({test_frames} frames, ~{test_s / 60:.1f} min)")
        self.full_btn.setText(
            f"Run FULL pass — {info.duration_s:.0f} s "
            f"({info.frame_count} frames, ~{full_s / 60:.0f} min)")

    # Cache size relative to the source video, and how alarmed to be about it.
    # Under 1x is genuinely cheap. Past 10x you are writing a cache that dwarfs
    # the footage it came from, which for most settings means the block size is
    # far smaller than the behavior actually needs.
    _SIZE_BANDS = [
        (1.0, "#a5d6a7", "comfortably smaller than the video"),
        (3.0, "#fff176", "larger than the video"),
        (10.0, "#ffb74d", "several times the video"),
        (float("inf"), "#ef9a9a", "far larger than the video"),
    ]

    def _update_size_banner(self, total_bytes: int) -> None:
        video_bytes = os.path.getsize(self.state.source.info.path)
        ratio = total_bytes / max(1, video_bytes)

        for limit, colour, blurb in self._SIZE_BANDS:
            if ratio < limit:
                break

        # "0.0x" tells the user nothing. Keep two decimals until the ratio is
        # large enough that one is informative.
        ratio_txt = f"{ratio:.2f}" if ratio < 1 else f"{ratio:.1f}"
        self.size_banner.setText(
            f"cache ≈ {ratio_txt}× your original file\n"
            f"{human_bytes(total_bytes)} vs {human_bytes(video_bytes)} — {blurb}")
        self.size_banner.setStyleSheet(
            f"background:{colour}; color:#000; font-weight:bold; "
            f"font-size:12px; padding:6px; border:1px solid #444;")

    # -- run ---------------------------------------------------------------

    def _run(self, test: bool):
        if not self.state.has_video:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        cfg = self.build_config()
        info = self.state.source.info
        reps = self.state.replicate_specs
        try:
            validate_replicates(reps)
        except ValueError as e:
            QMessageBox.warning(self, "Replicate geometry required", str(e))
            self.state.request_tab.emit(0)
            return

        msg = self._status.get(cfg.flow.backend, "")
        if not msg.startswith("Available"):
            QMessageBox.critical(self, "Backend unavailable", msg)
            return

        warnings = cfg.features.validate_bands(info.fps)
        hard = [w for w in warnings if "exceeds the Nyquist" in w]
        if hard:
            QMessageBox.critical(self, "Band exceeds Nyquist", "\n\n".join(hard))
            return
        if warnings and not test:
            r = QMessageBox.question(
                self, "Band near Nyquist",
                "\n\n".join(warnings) + "\n\nRun the full pass anyway?")
            if r != QMessageBox.StandardButton.Yes:
                return

        duration = self.test_seconds.value() if test else None

        # Test caches retain a human-readable settings snapshot after the complete
        # config hash. Duration keeps its meaningful decimals, so 10 s and 10.5 s
        # cannot collide.
        suffix = _test_cache_suffix(cfg, duration) if test else ""
        key = cfg.cache_key(info.video_hash, geometry_hash(reps)) + suffix

        if cache_mod.cache_exists(self.state.cache_root, key) and \
                not cache_mod.cache_is_complete(self.state.cache_root, key):
            self.state.status.emit(
                f"Replacing incomplete cache {key} from a cancelled/failed run.")
        elif cache_mod.cache_exists(self.state.cache_root, key):
            meta = cache_mod.read_meta(self.state.cache_root, key) or {}
            size = human_bytes(sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(
                    cache_mod.cache_dir(self.state.cache_root, key)) for f in fs))
            kind = "test" if test else "full"
            r = QMessageBox.question(
                self, "This cache already exists",
                f"A {kind} cache with these EXACT settings already exists:\n\n"
                f"  {meta.get('duration_s', 0):.0f} s of video, "
                f"{meta.get('n_frames', 0)} frames\n"
                f"  {meta.get('grid', ['?', '?'])[0]}x"
                f"{meta.get('grid', ['?', '?'])[1]} blocks, {size} on disk\n\n"
                f"Load it instead of recomputing?\n\n"
                f"(Recomputing would take about "
                f"{(self._last_full_min if not test else self._last_test_min):.1f} "
                f"minutes and produce an identical result.)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
            if r == QMessageBox.StandardButton.Yes:
                self.state.open_cache(key)
                return
            # Falling through re-runs and overwrites it, which is what "No" means.

        self.state.cfg = cfg
        self._mode = "TEST" if test else "FULL"
        self._set_running(True)
        self._t0 = time.perf_counter()

        self.worker = PipelineWorker(
            info.path, cfg, self.state.cache_root,
            duration_s=duration,
            replicates=[{**r, "frac": tuple(r["frac"])} for r in reps],
            suffix=suffix,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _set_running(self, running: bool):
        self.test_btn.setEnabled(not running)
        self.full_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        if not running:
            self.progress.setValue(0)

    def _cancel(self):
        if self.worker:
            self.worker.cancel()
            self.progress_label.setText("Cancelling...")

    def _on_progress(self, p):
        self.progress.setValue(int(p.frac * 100))
        eta = "" if p.eta_s != p.eta_s else f"  eta {p.eta_s / 60:.1f} min"
        stage_no = {"flow": "1/2", "bandpower": "2/2"}.get(p.stage, "")
        # Name the mode and the stage explicitly. The bar runs 0->100% twice
        # (flow, then band-power), which otherwise looks like it restarted and
        # is doing far more work than you asked for.
        self.progress_label.setText(
            f"{getattr(self, '_mode', '')} mode · stage {stage_no} "
            f"[{p.stage}] {p.done}/{p.total}{eta}\n{p.message}")

    def _on_done(self, key: str):
        self._set_running(False)
        self.progress_label.setText(
            f"Done in {(time.perf_counter() - self._t0) / 60:.1f} min. "
            f"Cache opened; Behavior Classification is ready.")
        self._refresh_cache_list()
        self.state.open_cache(key)

    def _on_failed(self, msg: str):
        self._set_running(False)
        self.progress_label.setText(msg)
        if msg != "Cancelled.":
            QMessageBox.critical(self, "Pipeline failed", msg)

    # -- caches / settings -------------------------------------------------

    def _refresh_cache_list(self):
        self.cache_list.clear()
        total = 0
        for c in cache_mod.list_caches(self.state.cache_root):
            tag = " [test]" if c.get("test_mode") else ""
            if c.get("complete") is False:
                tag += " [incomplete]"
            n_reps = len(c.get("replicate_tiles", []))
            if n_reps:
                tag += f" [{n_reps} reps]"
            self.cache_list.addItem(
                f"{c['key']}{tag}  {os.path.basename(c.get('video_path', '?'))}  "
                f"{c.get('duration_s', 0):.0f}s  "
                f"{c.get('grid', ['?', '?'])[0]}x{c.get('grid', ['?', '?'])[1]}",
                c["key"])
        root = self.state.cache_root
        if os.path.isdir(root):
            for dp, _, fs in os.walk(root):
                for f in fs:
                    total += os.path.getsize(os.path.join(dp, f))
        self.cache_disk_label.setText(
            f"{self.cache_list.count()} cache(s), {human_bytes(total)} total\n"
            f"at {os.path.abspath(root)}")

    def _delete_selected_cache(self):
        key = self.cache_list.currentData()
        if not key:
            return
        if QMessageBox.question(
                self, "Delete cache", f"Delete cache '{key}' from disk?") \
                == QMessageBox.StandardButton.Yes:
            cache_mod.delete_cache(self.state.cache_root, key)
            self._refresh_cache_list()
            self.state.status.emit(f"Deleted cache {key}.")

    def _open_cache_folder(self):
        root = os.path.abspath(self.state.cache_root)
        os.makedirs(root, exist_ok=True)
        # Reveal in the OS file browser. QDesktopServices handles all platforms;
        # on Windows it opens Explorer at the folder.
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(root))

    def _clear_all_caches(self):
        caches = cache_mod.list_caches(self.state.cache_root)
        if not caches:
            QMessageBox.information(self, "No caches", "There are no caches to clear.")
            return
        r = QMessageBox.warning(
            self, "Clear ALL caches",
            f"Permanently delete all {len(caches)} cache(s)?\n\n"
            f"This includes full passes, which are expensive to recompute. "
            f"Videos and behavior definitions are NOT affected.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes:
            return
        # If the open cache is among them, close it first so its files are free.
        if self.state.cache is not None:
            self.state.cache.close()
            self.state.cache = None
            self.state.ctx = None
        for c in caches:
            try:
                cache_mod.delete_cache(self.state.cache_root, c["key"])
            except OSError:
                pass
        self._refresh_cache_list()
        self.state.status.emit("Cleared all caches.")

    def _open_selected(self):
        key = self.cache_list.currentData()
        if not key:
            return
        try:
            self.state.open_cache(key)
        except Exception as e:
            QMessageBox.critical(self, "Could not open cache", str(e))

    def _export_settings(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export settings", "flow_settings.json", "JSON (*.json)")
        if not path:
            return
        vh = self.state.source.info.video_hash if self.state.has_video else None
        with open(path, "w") as f:
            f.write(self.build_config().to_json(video_hash=vh))

    def _import_settings(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import settings", "", "JSON (*.json)")
        if not path:
            return
        with open(path) as f:
            d = json.load(f)

        # A config tuned on different footage may be silently wrong here --
        # band-power thresholds and block sizes are resolution-dependent.
        vh = d.get("_video_hash")
        if vh and self.state.has_video and vh != self.state.source.info.video_hash:
            r = QMessageBox.warning(
                self, "Config was tuned on different footage",
                "This settings file was saved against a different video.\n\n"
                "Resolution- and frame-rate-dependent settings (band edges, "
                "block size, downsample) may not transfer. Import anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                return

        cfg = PipelineConfig.from_dict(d)
        self._apply_config(cfg)

    def _apply_config(self, cfg: PipelineConfig):
        p, f, ft = cfg.preprocess, cfg.flow, cfg.features
        self.downsample.setValue(p.downsample if p.downsample else 0.05)
        self.registration.setCurrentText(p.registration)
        self.denoise.setCurrentText(p.denoise)
        self.bg.setCurrentText(p.bg_subtract)
        self.normalize.setCurrentText(p.normalize)
        self._mask_path = p.mask_path
        self.mask_btn.setText(os.path.basename(p.mask_path) if p.mask_path
                              else "No within-box mask")
        self.backend.setCurrentText(f.backend)
        self.block.setValue(f.block_size)
        if ft.bands:
            self.band_lo.setValue(ft.bands[0].lo_hz)
            self.band_hi.setValue(ft.bands[0].hi_hz)
        self.window_s.setValue(ft.window_s)
        self.hop_s.setValue(ft.hop_s)
        self.dtype.setCurrentText(ft.dtype)
        self.compression.setCurrentText(ft.compression)
        self.cache_fb_error.setChecked(ft.cache_fb_error)
        self.cache_texture.setChecked(ft.cache_texture)
        self._refresh_estimates()
