"""Embeddable structure-tensor explorer (tool 3).

The third of the three sibling explorers, and the "change" end of the spectrum:

    speed explorer     -- per-frame motion magnitude
    coherent-flow expl -- flow VECTOR integrated over a window (slow coherent drift)
    THIS               -- temporal intensity change integrated over a window

It surfaces the channels the flow cache does not store, precomputed once from the
video by :mod:`core.tensor_channels` (reading the cached block flow for the
appearance residual, so nothing here re-solves optical flow):

    amplitude variance   <(I - mean I)^2> over the window   -- "background
                         subtraction, integrated"; catches slow brightness change.
    change energy J_tt   <I_t^2>                             -- fast-weighted
                         flicker; the backlit-wing channel.
    appearance energy    windowed residual^2                -- change no motion
                          explains; the detection channel for wing/shape events.
    appearance fraction  residual^2 / change                -- per-event
                          classifier (is this change motion or appearance?),
                          kept as a spatial OVERLAY only: the question is
                          per-block-at-a-moment, which a density plot cannot
                          anchor. Gated to zero where no real change exists.
    texture              spatial min-eigen (from the cache when present).
    tensor speed          Lucas-Kanade speed solved from J; an audit of the
                          cached flow, surfaced as |diff| / disagreement plots
                          and overlay modes rather than raw speed traces.

The three detection channels are drawn as time x value density heatmaps of the
full per-block distribution (log value axis by default), never spatial means:
each replicate is mostly empty space, so a mean buries the handful of behaving
blocks that are the actual signal -- the same lesson the speed explorer learned.
Detection bands live on the heavy-tailed *energies*, never the bounded fraction:
a band on a ratio alone lets a barely-above-floor block at fraction 1.0 outrank
a wing beat with 100x the energy at fraction 0.7.

The integration-window slider is the point: the value-vs-W density shows the
selected channel's whole per-block distribution accumulating as W widens, so
the knee where integration stops paying off is visible directly.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from core.replicates import block_weight_plane
from core.tensor_channels import load_or_extract_channels

from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                             QGridLayout, QHBoxLayout, QLabel, QProgressDialog,
                             QPushButton, QScrollArea, QSlider, QVBoxLayout,
                             QWidget)

from gui.video_panel import FrameView
from gui.explorers.speed_explorer import (DensityPlot, MiniPlot, PixelBarPlot,
                                          _regions_from_meta,
                                          DISPLAY_MAX_W, LINE, LINE2)

VAR_C = QColor(120, 215, 255)      # amplitude variance
CHANGE_C = QColor(255, 170, 80)    # change energy
APPEAR_C = QColor(180, 140, 255)   # appearance
SWEEP_C = QColor(110, 230, 120)    # threshold sweep
VW_C = QColor(255, 150, 90)        # value-vs-window
SPEED_DIFF_C = QColor(255, 105, 135)
EPS = 1e-6

# Which precomputed channel each detection target derives from, and how the
# window turns the raw per-frame channel into the displayed quantity.  All
# three are energies (nonnegative, decades of range, heavy right tail), so the
# log-heatmap + tail-band instrument applies to each; the bounded appearance
# *fraction* is deliberately not a target (see module docstring).
DETECT_TARGETS = ("amplitude variance", "change energy", "appearance energy")

# Overlay-combo item for each detection target, so selecting a channel to
# detect on also shows that channel spatially. The user can still pick any
# other overlay afterwards; the sync happens only at selection time.
OVERLAY_BY_TARGET = {"amplitude variance": "Amplitude variance",
                     "change energy": "Change energy Jtt",
                     "appearance energy": "Appearance energy"}


class StructureTensorExplorer(QWidget):
    """Explore the temporal-change reads from the structure-tensor POC.

    The widget deliberately follows the same cache-backed, replicate-aware UI
    contract as :class:`gui.explorers.speed_explorer.SpeedExplorer` and
    :class:`gui.explorers.coherent_flow_explorer.CoherentFlowExplorer`.
    """

    def __init__(self, cache=None, video_path: str | None = None, *, state=None,
                 sidecar_path: str | None = None, parent=None):
        super().__init__(parent)
        if state is not None:
            cache = state.cache
        if cache is None:
            raise ValueError("StructureTensorExplorer requires an open cache")
        self.state = state
        self.cache = cache
        self.meta = cache.meta
        self.fps = float(self.meta["fps"])
        self.dt = 1.0 / max(self.fps, EPS)
        self.block = int(self.meta["block_size"])
        self.ny, self.nx = map(int, self.meta["grid"])
        self.regions = _regions_from_meta(self.meta, (self.ny, self.nx))
        self.packed = bool(self.meta.get("replicate_tiles"))
        self.active_region_index = -1 if len(self.regions) > 1 else 0

        # -- precompute the channels the cache lacks (slow; sidecar-cached) ----
        dlg = QProgressDialog("Extracting structure-tensor channels from video...", None,
                              0, int(self.meta["n_frames"]), self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setCancelButton(None)

        def prog(done, total):
            dlg.setMaximum(total)
            dlg.setValue(done)
            QApplication.processEvents()

        ch = load_or_extract_channels(cache, sidecar_path=sidecar_path, progress=prog)
        dlg.close()
        self.intensity = ch["intensity"]           # (T, ny, nx)
        self.change = ch["change"]
        self.appearance = ch["appearance"]
        self.texture = ch["texture"]
        self.tensor_speed = np.asarray(ch["tensor_speed"], np.float32)
        self.approximated = ch["meta"]["approximated"]
        self.T = self.intensity.shape[0]

        ctx = state.ctx if state is not None else None
        speed_src = ctx.speed if ctx is not None else cache.read("speed")
        self.cached_speed = np.asarray(speed_src, np.float32)
        if self.tensor_speed.shape != self.intensity.shape or \
                self.cached_speed.shape != self.intensity.shape:
            raise ValueError(
                "Tensor, cached-speed, and temporal channels must share the "
                "cache's (frames, grid-y, grid-x) shape")
        speed_ref = np.maximum(self.tensor_speed, self.cached_speed)
        finite_speed = speed_ref[np.isfinite(speed_ref)]
        self.speed_display_vmax = max(
            float(np.percentile(finite_speed, 99.0))
            if finite_speed.size else 0.0, 1e-4)
        # Symmetric relative error is otherwise unstable when both estimators are
        # essentially zero.  One percent of the clip p99 acts only as a quiet-
        # block denominator floor; it does not alter either speed channel.
        self.speed_floor = max(0.01 * self.speed_display_vmax, 1e-3)
        self.speed_absdiff = np.abs(self.tensor_speed - self.cached_speed)
        self.speed_disagreement = self.speed_absdiff / (
            self.tensor_speed + self.cached_speed + self.speed_floor)

        # Prefix sums (leading zero row) for O(1) windowed mean/var/energy.
        z = np.zeros((1, self.ny, self.nx), np.float32)
        self._ci = np.concatenate([z, np.cumsum(self.intensity, 0, dtype=np.float32)])
        self._ci2 = np.concatenate([z, np.cumsum(self.intensity ** 2, 0, dtype=np.float32)])
        self._cc = np.concatenate([z, np.cumsum(self.change, 0, dtype=np.float32)])
        self._ca = np.concatenate([z, np.cumsum(self.appearance, 0, dtype=np.float32)])

        self.source = None
        self._owns_source = False
        if state is None and video_path and os.path.exists(video_path):
            from core.video import VideoSource
            try:
                self.source = VideoSource(video_path)
                self._owns_source = True
            except Exception:
                self.source = None
        self.src_w = max(1, int(self.meta.get("src_width", 0)) or
                         int(self.meta.get("work_width", 1)))
        self.src_h = max(1, int(self.meta.get("src_height", 0)) or
                         int(self.meta.get("work_height", 1)))

        self.win_frames = max(2, min(self.T - 1, int(round(self.fps))))
        # Detection-sweep integration window (frames). The binary detection
        # gate reads the centered MEAN of "# blocks in band" over this window,
        # so a single-frame spike of N blocks dilutes to N/D and cannot fake a
        # sustained event. D = 1 reproduces the raw per-frame count.
        self.sweep_win = max(1, min(self.T - 1, int(round(self.fps))))
        # Centered vs trailing integration windows (see _window_bounds).
        self.centered = True
        self.detect = "appearance energy"
        self.frame = int(state.current_frame) if state is not None else 0
        self.playing = False
        self._overlay_peek_hidden = False

        self._rebuild_owned_prefixes()
        # Appearance fraction gates division by a real-change floor.  The floor
        # must exist before the windowed series (and hence the seeded detection
        # band) can be computed.  It is computed once over every replicate-owned
        # block -- never atlas separators/padding, whose junk values would
        # inflate the gate -- and over ALL replicates rather than the active
        # scope, so focusing a replicate cannot silently reinterpret the gate.
        owned_change = np.concatenate([
            self.change[:, y0:y1, x0:x1].reshape(self.T, -1)
            for (y0, x0, y1, x1) in (r["atlas_bbox"] for r in self.regions)],
            axis=1)
        positive_change = owned_change[owned_change > 0]
        self.change_floor = float(np.percentile(positive_change, 50)) \
            if positive_change.size else 0.0

        self._build_ui()
        self._event_filter_app = QApplication.instance()
        if self._event_filter_app is not None:
            self._event_filter_app.installEventFilter(self)
        self._space_shortcut = None
        if state is None:
            self._space_shortcut = QShortcut(
                QKeySequence(Qt.Key.Key_Space.value), self)
            self._space_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            self._space_shortcut.setAutoRepeat(False)
            self._space_shortcut.activated.connect(self.toggle_playback)
        if state is not None:
            state.frame_changed.connect(self._on_state_frame_changed)

        self._recompute_series()
        # Seed the selected channel's band from its now-populated series, and
        # expand that plot. Must follow series computation so the band is
        # absolute in the channel's own units.
        self._apply_selected_plot_ui()
        self._recompute_sweep()
        self._recompute_value_curve()
        self._apply_frame(self.frame)

    @classmethod
    def from_app_state(cls, state, parent=None) -> "StructureTensorExplorer":
        if not state.has_cache:
            raise ValueError("Open a feature cache before creating this explorer")
        return cls(state=state, parent=parent)

    # -- owned-block bookkeeping ----------------------------------------------

    def _active_regions(self) -> list[dict]:
        if self.active_region_index < 0:
            return self.regions
        return [self.regions[self.active_region_index]]

    def _rebuild_owned_prefixes(self):
        ci, ci2, cc, ca = [], [], [], []
        speed_absdiff, speed_disagreement = [], []
        for region in self._active_regions():
            y0, x0, y1, x1 = region["atlas_bbox"]
            ci.append(self._ci[:, y0:y1, x0:x1].reshape(self.T + 1, -1))
            ci2.append(self._ci2[:, y0:y1, x0:x1].reshape(self.T + 1, -1))
            cc.append(self._cc[:, y0:y1, x0:x1].reshape(self.T + 1, -1))
            ca.append(self._ca[:, y0:y1, x0:x1].reshape(self.T + 1, -1))
            speed_absdiff.append(
                self.speed_absdiff[:, y0:y1, x0:x1].reshape(self.T, -1))
            speed_disagreement.append(
                self.speed_disagreement[:, y0:y1, x0:x1].reshape(self.T, -1))
        cat = lambda parts: parts[0] if len(parts) == 1 else np.concatenate(parts, 1)
        self._ci_own, self._ci2_own = cat(ci), cat(ci2)
        self._cc_own, self._ca_own = cat(cc), cat(ca)
        self._speed_absdiff_own = cat(speed_absdiff)
        self._speed_disagreement_own = cat(speed_disagreement)
        self.n_blocks = self._ci_own.shape[1]

    def _window_bounds(self, W):
        """Prefix-sum bounds per frame; centered or trailing per the checkbox.

        Centered (default): frame t integrates [t - W//2, t + ceil(W/2) - 1],
        so an event's windowed mass peaks where the event IS, not W/2 frames
        later -- this is an offline explorer over a cached clip, so reading
        the future is free. Trailing ([t - W + 1, t]) is the causal view: what
        a live detector running these thresholds would actually see, W/2 lag
        included. Windows truncate at the clip edges; neff keeps the means
        honest there.
        """
        t = np.arange(self.T)
        if self.centered:
            lo = np.maximum(0, t - W // 2)
            hi = np.minimum(self.T, t + (W - W // 2))
        else:
            hi = t + 1
            lo = np.maximum(0, hi - W)
        return hi, lo, (hi - lo).astype(np.float32)

    def _win_slice(self, t: int, W: int) -> tuple[int, int]:
        """Scalar (lo, hi) of ``_window_bounds`` for one frame (overlay path)."""
        if self.centered:
            lo = max(0, t - W // 2)
            hi = min(self.T, t + (W - W // 2))
        else:
            hi = t + 1
            lo = max(0, hi - W)
        return lo, hi

    def _owned_windowed(self, W):
        """(T, B) windowed reads over the owned blocks."""
        hi, lo, neff = self._window_bounds(W)
        mean_i = (self._ci_own[hi] - self._ci_own[lo]) / neff[:, None]
        mean_i2 = (self._ci2_own[hi] - self._ci2_own[lo]) / neff[:, None]
        var = np.maximum(mean_i2 - mean_i * mean_i, 0.0)
        change = (self._cc_own[hi] - self._cc_own[lo]) / neff[:, None]
        appear = (self._ca_own[hi] - self._ca_own[lo]) / neff[:, None]
        return {"variance": var, "change": change, "appearance": appear}

    def _detect_field(self, w: dict) -> np.ndarray:
        if self.detect == "amplitude variance":
            return w["variance"]
        if self.detect == "change energy":
            return w["change"]
        return w["appearance"]

    def _selected_plot_key(self) -> str:
        return self.detect_plot_key[self.detect]

    def _selected_plot(self) -> "MiniPlot":
        return self.plots[self._selected_plot_key()]

    def _band(self) -> tuple[float, float]:
        """Current (lo, hi) detection band in the selected channel's Y units."""
        return self._selected_plot().band()

    def _apply_selected_plot_ui(self):
        """Expand + activate the band on the selected plot; reset the others."""
        for target, key in self.detect_plot_key.items():
            selected = (target == self.detect)
            pl = self.plots[key]
            pl.set_band_active(selected)
            pl.set_expanded(selected)

    # -- layout ---------------------------------------------------------------

    def _build_ui(self):
        self._sync_window_title()
        self.resize(1500, 900)
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        self.video_view = FrameView()
        self.video_view.setMinimumSize(720, 480)
        self.video_view.clicked.connect(self._on_video_clicked)
        self.video_view.back_requested.connect(self._clear_region_focus)
        self._sync_video_boxes()
        left.addWidget(self.video_view, 1)

        trow = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        trow.addWidget(self.play_btn)
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.setRange(0, self.T - 1)
        self.scrub.valueChanged.connect(self._on_scrub)
        trow.addWidget(self.scrub, 1)
        self.time_lbl = QLabel("0.00 s")
        self.time_lbl.setMinimumWidth(120)
        trow.addWidget(self.time_lbl)
        left.addLayout(trow)

        rrow = QHBoxLayout()
        rrow.addWidget(QLabel("Replicate:"))
        self.region_combo = QComboBox()
        if len(self.regions) > 1:
            self.region_combo.addItem(
                f"All {len(self.regions)} replicates (pooled diagnostic)", -1)
        for i, region in enumerate(self.regions):
            rid = region["id"]
            suffix = f" (#{rid})" if rid is not None else ""
            self.region_combo.addItem(f"{region['label']}{suffix}", i)
        self.region_combo.setCurrentIndex(0)
        self.region_combo.currentIndexChanged.connect(self._on_region_changed)
        rrow.addWidget(self.region_combo, 1)
        left.addLayout(rrow)

        orow = QHBoxLayout()
        orow.addWidget(QLabel("Overlay:"))
        self.overlay_mode = QComboBox()
        self.overlay_mode.addItems([
            "Amplitude variance", "Change energy Jtt", "Appearance energy",
            "Appearance fraction", "Texture", "Tensor speed",
            "Cached flow speed", "Relative speed disagreement", "Raw frame"])
        self.overlay_mode.setCurrentText("Appearance energy")
        self.overlay_mode.currentTextChanged.connect(lambda _: self._redraw_video())
        orow.addWidget(self.overlay_mode, 1)
        self.hi_chk = QCheckBox("Highlight blocks in detection band")
        self.hi_chk.setChecked(True)
        self.hi_chk.stateChanged.connect(lambda _: self._redraw_video())
        orow.addWidget(self.hi_chk)
        left.addLayout(orow)

        # Detection thresholds no longer live on a horizontal slider: they are
        # the min/max handles of the band on whichever plot is the selected
        # channel (same UI contract as the speed explorer).
        band_hint = QLabel(
            "Detection thresholds: drag the min/max handles on the selected "
            "plot (checked ✓) at right.")
        band_hint.setStyleSheet("color:#888;")
        band_hint.setWordWrap(True)
        left.addWidget(band_hint)

        self.log_chk = QCheckBox("Log value axis on channel heatmaps")
        self.log_chk.setToolTip(
            "Draw the amplitude-variance / change-energy / appearance-energy "
            "heatmaps' (and the vs-W density's) value axis as log1p(value): "
            "these channels span decades and a linear axis crushes the "
            "noise-floor bulk flat under a few hot blocks. Axis-only -- band "
            "thresholds are still set and compared in raw units.")
        self.log_chk.setChecked(True)
        self.log_chk.stateChanged.connect(self._on_log_toggle)
        left.addWidget(self.log_chk)

        self.centered_chk = QCheckBox("Centered integration windows")
        self.centered_chk.setToolTip(
            "Checked: frame t integrates [t-W/2, t+W/2], so windowed mass and "
            "detections peak ON the event (offline view; reads the future). "
            "Unchecked: trailing [t-W+1, t] -- what a causal live detector "
            "would see, with its W/2 lag. Applies to W, the detection window "
            "D, the overlay, and the vs-W density alike.")
        self.centered_chk.setChecked(self.centered)
        self.centered_chk.stateChanged.connect(self._on_centered_toggle)
        left.addWidget(self.centered_chk)

        self.win_slider, self.win_lbl = self._add_slider(
            left, 2, max(3, self.T - 1), self.win_frames, self._on_window)
        self.sweep_win_slider, self.sweep_win_lbl = self._add_slider(
            left, 1, max(2, self.T - 1), self.sweep_win, self._on_sweep_window)

        note = QLabel(
            "Window W integrates temporal change: amplitude variance = "
            "<(I-mean)^2> over W (background subtraction, integrated); change "
            "energy = <I_t^2> (fast flicker); appearance energy = residual "
            "change no motion explains (wing/shape detector). The appearance "
            "fraction (residual/change, gated where change is below floor) "
            "survives as an OVERLAY: it classifies a detected event as motion "
            "vs appearance, a per-block question no time plot can anchor.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8ab; padding-top:2px;")
        left.addWidget(note)

        speed_note = QLabel(
            "Tensor speed solves the Lucas-Kanade velocity directly from J. "
            "Compare it with the cached pyramidal flow speed: disagreement marks "
            "large-displacement, low-texture, or brightness-constancy failures. "
            "The two speed heatmaps share one color scale; relative disagreement "
            "uses a small quiet-speed floor.")
        speed_note.setWordWrap(True)
        speed_note.setStyleSheet("color:#8ab; padding-top:2px;")
        left.addWidget(speed_note)

        approx = "  |  preprocessing APPROXIMATED (stateful steps skipped)" \
            if self.approximated else ""
        info = QLabel(
            f"cache: {self.meta.get('backend', '?')} | fps {self.fps:.2f} | "
            f"block {self.block}px | {self.T} frames{approx}")
        info.setStyleSheet("color:#888;")
        left.addWidget(info)
        root.addLayout(left, 3)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        self.plot_col = QGridLayout(holder)
        self.plot_col.setSpacing(3)
        self.plot_col.setColumnMinimumWidth(0, 20)
        self.plot_col.setColumnStretch(1, 1)
        scroll.setWidget(holder)
        scroll.setMinimumWidth(440)
        root.addWidget(scroll, 2)

        self.plots: dict[str, MiniPlot] = {}
        self.detect_checks: dict[str, QCheckBox] = {}
        # detect target -> plot key, so the selected channel's band handles are
        # found on the right plot.
        self.detect_plot_key: dict[str, str] = {}
        self.detect_group = QButtonGroup(self)
        self.detect_group.setExclusive(True)
        self._plot_row = 0

        def section(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("color:#7fd7ff; font-weight:bold; padding-top:6px;")
            self.plot_col.addWidget(lbl, self._plot_row, 0, 1, 2)
            self._plot_row += 1

        def add(key, title, unit, color=LINE, seek=True,
                detect_target: str | None = None, cls=MiniPlot):
            pl = cls(title, unit, color)
            if seek:
                pl.seek_requested.connect(self._seek)
            self.plots[key] = pl
            self.plot_col.addWidget(pl, self._plot_row, 1)
            if detect_target is not None:
                self.detect_plot_key[detect_target] = key
                pl.band_changed.connect(self._on_band_changed)
                pl.band_committed.connect(self._on_band_committed)
                check = QCheckBox()
                check.setAccessibleName(f"Detect on {title}")
                check.setToolTip(
                    f"Use {title} as the detection channel and drag its band")
                check.setProperty("detect_target", detect_target)
                self.detect_group.addButton(check)
                self.detect_checks[detect_target] = check
                check.setChecked(detect_target == self.detect)
                self.plot_col.addWidget(
                    check, self._plot_row, 0, Qt.AlignmentFlag.AlignCenter)
            self._plot_row += 1

        section("Per-frame appearance channels (no integration)")
        add("intensity", "Mean block intensity", "", LINE)
        add("change_pf", "Per-frame change energy Jtt", "", CHANGE_C)
        add("appear_pf", "Per-frame appearance energy", "", APPEAR_C)

        section("Integrated over window W (per-block density heatmaps)")
        add("variance", "Per-block amplitude variance (density)", "", VAR_C,
            detect_target="amplitude variance", cls=DensityPlot)
        add("change_w", "Per-block change energy over W (density)", "", CHANGE_C,
            detect_target="change energy", cls=DensityPlot)
        add("appear_w", "Per-block appearance energy over W (density)", "",
            APPEAR_C, detect_target="appearance energy", cls=DensityPlot)

        section("Detection sweep on selected channel (drag band handles)")
        add("count", "# blocks in band", "blk", SWEEP_C, cls=PixelBarPlot)
        # The binary gate: centered mean of the in-band count over D, with its
        # own always-active band. The min handle is the sustained-evidence
        # threshold; the max handle can reject whole-replicate artifacts where
        # every block goes in-band at once (a lighting flash, not a behavior).
        add("count_w", "Windowed # blocks in band (mean over D)", "blk", SWEEP_C)
        add("detect", "Positive detection (windowed count in band)", "0/1",
            QColor(120, 255, 140))
        add("clump", "Largest connected clump in band", "blk", SWEEP_C)
        count_w = self.plots["count_w"]
        count_w.band_changed.connect(self._recompute_detect)
        count_w.band_committed.connect(self._recompute_detect)
        count_w.set_band_active(True)
        count_w.set_expanded(True)

        section("Value of integration @ cursor (x axis = window length)")
        # Per-block density of the SELECTED channel vs window length: the knee
        # where widening W stops sharpening the tail is the right W to run at.
        add("vw_sel", self._vw_title(), "", VW_C, seek=False, cls=DensityPlot)

        section("Flow audit: |tensor - cached| speed (no temporal integration)")
        add("speed_absdiff", "Mean |tensor - cached speed|", "px/s", SPEED_DIFF_C)
        add("speed_disagreement", "Mean relative speed disagreement", "", SPEED_DIFF_C)
        self.plot_col.setRowStretch(self._plot_row, 1)
        self.detect_group.buttonToggled.connect(self._on_detect_plot_toggled)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._win_debounce = QTimer(self)
        self._win_debounce.setSingleShot(True)
        self._win_debounce.setInterval(120)
        self._win_debounce.timeout.connect(self._apply_window_change)
        # Debounce for the cheap in-band counts so dragging a band handle stays
        # smooth; the expensive clump refreshes only on release.
        self._sweep_debounce = QTimer(self)
        self._sweep_debounce.setSingleShot(True)
        self._sweep_debounce.setInterval(120)
        self._sweep_debounce.timeout.connect(self._recompute_sweep_counts)
        self._on_log_toggle()          # plots exist now; apply the default axis
        self._sync_labels()

    def _add_slider(self, layout, lo, hi, val, handler):
        row = QHBoxLayout()
        lbl = QLabel()
        lbl.setMinimumWidth(230)
        row.addWidget(lbl)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        s.valueChanged.connect(handler)
        row.addWidget(s, 1)
        layout.addLayout(row)
        return s, lbl

    # -- series ---------------------------------------------------------------

    def _recompute_series(self):
        w = self._owned_windowed(self.win_frames)
        self.plots["speed_absdiff"].set_series(self._speed_absdiff_own.mean(1))
        self.plots["speed_disagreement"].set_series(
            self._speed_disagreement_own.mean(1))
        self.plots["intensity"].set_series(
            (self._ci_own[1:] - self._ci_own[:-1]).mean(1))
        self.plots["change_pf"].set_series(
            (self._cc_own[1:] - self._cc_own[:-1]).mean(1))
        self.plots["appear_pf"].set_series(
            (self._ca_own[1:] - self._ca_own[:-1]).mean(1))
        # The windowed channels show the full per-block distribution as density
        # heatmaps, not spatial means -- a replicate is mostly empty space, so
        # the mean buries the few behaving blocks that are the actual signal.
        self.plots["variance"].set_matrix(w["variance"])
        self.plots["change_w"].set_matrix(w["change"])
        self.plots["appear_w"].set_matrix(w["appearance"])
        self._w_cache = w
        # The overlay display scale is derived from this cache; a fresh cache
        # invalidates it so the next redraw re-percentiles the new data.
        self._scale_cache_key = None

    def _recompute_sweep_counts(self):
        lo, hi = self._band()
        field = self._detect_field(self._w_cache)
        mask = (field >= lo) & (field <= hi)
        count = mask.sum(1).astype(np.float32)
        self.plots["count"].set_series(count)
        self._recompute_windowed_count(count)

    def _recompute_windowed_count(self, count: np.ndarray | None = None):
        """Mean of the in-band count over the detection window D.

        Follows the ``_window_bounds`` convention (centered by default, so the
        gate fires ON the event rather than D/2 frames after it). The dilution
        property holds either way: a 1-frame spike of N blocks never reads
        more than N/D, so sustained evidence is still required to clear the
        min handle.
        """
        if count is None:
            count = self.plots["count"].y
        if count.size == 0:
            self.plots["count_w"].set_series(count)
            self._recompute_detect()
            return
        if self.sweep_win <= 1:
            windowed = count.astype(np.float32)
        else:
            # count is always the full-clip (T,) series, so the shared bounds
            # apply directly.
            c = np.concatenate([[0.0], np.cumsum(count, dtype=np.float64)])
            hi, lo, neff = self._window_bounds(self.sweep_win)
            windowed = ((c[hi] - c[lo]) / neff).astype(np.float32)
        self.plots["count_w"].set_series(windowed)
        self._recompute_detect()

    def _recompute_detect(self):
        """Binary detection: windowed in-band count inside its own band.

        Cheap (one comparison over a (T,) series), so it runs live while
        either the channel band or the count band is being dragged.
        """
        blo, bhi = self.plots["count_w"].band()
        y = self.plots["count_w"].y
        self.plots["detect"].set_series(
            ((y >= blo) & (y <= bhi)).astype(np.float32))

    def _recompute_sweep(self):
        self._recompute_sweep_counts()
        self._recompute_clump()

    def _recompute_clump(self):
        W = self.win_frames
        hi, lo, neff = self._window_bounds(W)
        band_lo, band_hi = self._band()
        weight_plane = block_weight_plane(self.meta)
        largest = np.zeros(self.T, np.float32)
        for region in self._active_regions():
            y0, x0, y1, x1 = region["atlas_bbox"]
            w_flat = weight_plane[y0:y1, x0:x1].reshape(-1)
            field = self._detect_field_full(y0, x0, y1, x1, hi, lo, neff)
            for t in range(self.T):
                m = ((field[t] >= band_lo) &
                     (field[t] <= band_hi)).astype(np.uint8)
                if not m.any():
                    continue
                n_lab, labels, _, _ = cv2.connectedComponentsWithStats(
                    m, connectivity=8)
                if n_lab > 1:
                    areas = np.bincount(labels.reshape(-1), weights=w_flat,
                                        minlength=n_lab)
                    largest[t] = max(largest[t], float(areas[1:].max()))
        self.plots["clump"].set_series(largest)

    def _detect_field_full(self, y0, x0, y1, x1, hi, lo, neff):
        """(T, th, tw) selected-channel field for one atlas tile."""
        n = neff[:, None, None]
        if self.detect == "change energy":
            return (self._cc[hi, y0:y1, x0:x1] - self._cc[lo, y0:y1, x0:x1]) / n
        if self.detect == "amplitude variance":
            mi = (self._ci[hi, y0:y1, x0:x1] - self._ci[lo, y0:y1, x0:x1]) / n
            mi2 = (self._ci2[hi, y0:y1, x0:x1] - self._ci2[lo, y0:y1, x0:x1]) / n
            return np.maximum(mi2 - mi * mi, 0.0)
        # appearance energy: windowed residual change no motion explains
        return (self._ca[hi, y0:y1, x0:x1] - self._ca[lo, y0:y1, x0:x1]) / n

    def _vw_title(self) -> str:
        return f"{self.detect.capitalize()} vs W (per-block density)"

    def _recompute_value_curve(self):
        """Selected channel's per-block distribution as a function of window
        length, anchored at the cursor frame -- the W-choosing instrument."""
        t0 = self.frame
        Ws = np.unique(np.round(
            np.geomspace(1, self.T, num=min(160, self.T))).astype(int))
        Ws = Ws[Ws >= 1]
        # Same convention as _window_bounds, per window length.
        if self.centered:
            los = np.maximum(0, t0 - Ws // 2)
            his = np.minimum(self.T, t0 + (Ws - Ws // 2))
        else:
            his = np.full_like(Ws, t0 + 1)
            los = np.maximum(0, t0 + 1 - Ws)
        neff = (his - los).astype(np.float32)[:, None]
        if self.detect == "amplitude variance":
            mi = (self._ci_own[his] - self._ci_own[los]) / neff
            mi2 = (self._ci2_own[his] - self._ci2_own[los]) / neff
            vals = np.maximum(mi2 - mi * mi, 0.0)
        elif self.detect == "change energy":
            vals = (self._cc_own[his] - self._cc_own[los]) / neff
        else:
            vals = (self._ca_own[his] - self._ca_own[los]) / neff
        self._vw_Ws = Ws
        vw = self.plots["vw_sel"]
        vw.title = self._vw_title()
        vw.set_matrix(vals)
        self._sync_value_cursor()

    def _sync_value_cursor(self):
        idx = int(np.searchsorted(self._vw_Ws, self.win_frames))
        idx = max(0, min(idx, self._vw_Ws.size - 1))
        self.plots["vw_sel"].set_cursor(idx)

    # -- handlers -------------------------------------------------------------

    def _sync_labels(self):
        self.win_lbl.setText(
            f"Integration window W: {self.win_frames} fr "
            f"({self.win_frames * self.dt:.2f} s)")
        self.sweep_win_lbl.setText(
            f"Detection window D (count mean): {self.sweep_win} fr "
            f"({self.sweep_win * self.dt:.2f} s)")

    def _on_sweep_window(self, v):
        self.sweep_win = max(1, int(v))
        self._sync_labels()
        # O(T) on an already-computed series: no debounce needed.
        self._recompute_windowed_count()

    def _on_window(self, v):
        self.win_frames = max(2, int(v))
        self._sync_labels()
        self._win_debounce.start()
        self._sync_value_cursor()

    def _apply_window_change(self):
        self._recompute_series()
        self._recompute_sweep()
        self._recompute_value_curve()
        self._redraw_video()

    def _on_log_toggle(self, _state=None):
        # Axis-only: never touches matrices or bands -- a threshold means the
        # same raw value whether the axis is log or linear. The fraction view
        # is bounded [0,1], where log1p is pointless; it stays linear.
        on = self.log_chk.isChecked()
        for key in ("variance", "change_w", "appear_w", "vw_sel"):
            self.plots[key].set_log_axis(on)

    def _on_centered_toggle(self, _state=None):
        # Same recompute path as a window-length change: every windowed read
        # (series, sweep, vs-W curve, overlay) depends on the convention.
        self.centered = self.centered_chk.isChecked()
        self._apply_window_change()

    def _on_band_changed(self):
        """A band handle is being dragged: cheap series debounced, overlay live."""
        self._sweep_debounce.start()
        self._redraw_video()          # highlight overlay follows immediately

    def _on_band_committed(self):
        """Handle released: flush the cheap series and refresh the clump."""
        self._sweep_debounce.stop()
        self._recompute_sweep()
        self._redraw_video()

    def _on_detect_plot_toggled(self, button, checked: bool):
        if not checked:
            return
        data = button.property("detect_target")
        self.detect = str(data) if data is not None else DETECT_TARGETS[-1]
        self._apply_selected_plot_ui()
        # The overlay follows the channel being detected on (signals blocked:
        # the redraw below covers it). Manual overlay picks still work -- this
        # resync happens only when the detection channel itself changes.
        self.overlay_mode.blockSignals(True)
        self.overlay_mode.setCurrentText(OVERLAY_BY_TARGET[self.detect])
        self.overlay_mode.blockSignals(False)
        self._recompute_sweep()
        self._recompute_value_curve()
        self._redraw_video()

    def _on_region_changed(self, _index):
        data = self.region_combo.currentData()
        self.active_region_index = int(data) if data is not None else 0
        focus = None if self.active_region_index < 0 else \
            self.regions[self.active_region_index]["frac"]
        self.video_view.set_focus_frac(focus)
        self._sync_video_boxes()
        self._rebuild_owned_prefixes()
        self._recompute_series()
        # Each scope lives on its own value scale, so a band frozen from the
        # previous scope would sit off this scope's plot entirely. Re-seed every
        # channel's band to the new scope's range; the active plot reseeds
        # immediately, the rest lazily.
        for key in self.detect_plot_key.values():
            self.plots[key].band_lo = self.plots[key].band_hi = None
        # The count band is in blocks and the scope's block count just changed;
        # it re-seeds wide open on the same principle as the channel bands.
        count_w = self.plots["count_w"]
        count_w.band_lo = count_w.band_hi = None
        count_w.set_band_active(True)
        self._apply_selected_plot_ui()
        self._recompute_sweep()
        self._recompute_value_curve()
        self._sync_labels()
        self._sync_window_title()
        self._redraw_video()

    def _on_video_clicked(self, point):
        width, height = self.video_view._src_size
        fx = point.x() / max(1, width)
        fy = point.y() / max(1, height)
        for i, region in enumerate(self.regions):
            x0, y0, x1, y1 = region["frac"]
            if x0 <= fx <= x1 and y0 <= fy <= y1:
                idx = self.region_combo.findData(i)
                if idx >= 0:
                    self.region_combo.setCurrentIndex(idx)
                return

    def _clear_region_focus(self):
        idx = self.region_combo.findData(-1)
        if idx >= 0:
            self.region_combo.setCurrentIndex(idx)
        else:
            self.video_view.set_focus_frac(None)

    def _sync_video_boxes(self):
        if not self.packed:
            self.video_view.set_boxes([])
            return
        self.video_view.set_boxes([
            (*region["frac"], region["label"], "#50dcff",
             i == self.active_region_index)
            for i, region in enumerate(self.regions)
        ])

    def _sync_window_title(self):
        scope = (f"all {len(self.regions)} replicates pooled"
                 if self.active_region_index < 0
                 else self.regions[self.active_region_index]["label"])
        self.setWindowTitle(
            f"Structure tensor explorer -- "
            f"{os.path.basename(self.meta.get('video_path', '?'))} -- {scope}")

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start(int(1000 / max(1.0, self.fps)))
        else:
            self.timer.stop()

    def toggle_playback(self):
        self._toggle_play()

    def eventFilter(self, watched, event):
        et = event.type()
        if et in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and \
                event.key() == Qt.Key.Key_Shift and not event.isAutoRepeat():
            focus = QApplication.focusWidget()
            ours = focus is self or (focus is not None and self.isAncestorOf(focus))
            if et == QEvent.Type.KeyPress and ours and not self._overlay_peek_hidden:
                self._overlay_peek_hidden = True
                self.video_view.set_overlays_hidden(True)
                self._redraw_video()
            elif et == QEvent.Type.KeyRelease and self._overlay_peek_hidden:
                self._overlay_peek_hidden = False
                self.video_view.set_overlays_hidden(False)
                self._redraw_video()
        elif et == QEvent.Type.ApplicationDeactivate and self._overlay_peek_hidden:
            self._overlay_peek_hidden = False
            self.video_view.set_overlays_hidden(False)
            self._redraw_video()
        return super().eventFilter(watched, event)

    def _tick(self):
        self._update_frame(0 if self.frame + 1 >= self.T else self.frame + 1)

    def _on_scrub(self, v):
        if v != self.frame:
            self._update_frame(v)

    def _seek(self, frame):
        self._update_frame(frame)

    def _update_frame(self, frame):
        frame = max(0, min(int(frame), self.T - 1))
        if self.state is not None and self.state.current_frame != frame:
            self.state.set_frame(frame)
            return
        self._apply_frame(frame)

    def _on_state_frame_changed(self, frame):
        self._apply_frame(frame)

    def _apply_frame(self, frame):
        self.frame = max(0, min(int(frame), self.T - 1))
        if self.scrub.value() != self.frame:
            self.scrub.blockSignals(True)
            self.scrub.setValue(self.frame)
            self.scrub.blockSignals(False)
        self.time_lbl.setText(f"{self.frame / self.fps:.2f} s  (#{self.frame})")
        for key, pl in self.plots.items():
            if not key.startswith("vw_"):
                pl.set_cursor(self.frame)
        self._recompute_value_curve()
        if self.isVisible():
            self._redraw_video()

    # -- video + overlay ------------------------------------------------------

    def _base_frame(self):
        focus = None if self.active_region_index < 0 else \
            self.regions[self.active_region_index]["frac"]
        self._render_frac = focus or (0.0, 0.0, 1.0, 1.0)
        if self.state is not None:
            bgr = self.state.display_frame(self.frame, focus_frac=focus)
        elif self.source is not None:
            bgr = self.source.frame_at(self.frame)
            if bgr is not None and focus is not None:
                h, w = bgr.shape[:2]
                x0, y0, x1, y1 = focus
                sx0 = max(0, min(w - 1, int(round(x0 * w))))
                sy0 = max(0, min(h - 1, int(round(y0 * h))))
                sx1 = max(sx0 + 1, min(w, int(round(x1 * w))))
                sy1 = max(sy0 + 1, min(h, int(round(y1 * h))))
                bgr = np.ascontiguousarray(bgr[sy0:sy1, sx0:sx1])
        else:
            bgr = None
        vx0, vy0, vx1, vy1 = self._render_frac
        view_w = max(1, int(round(self.src_w * (vx1 - vx0))))
        view_h = max(1, int(round(self.src_h * (vy1 - vy0))))
        if bgr is None:
            dw = min(view_w, DISPLAY_MAX_W)
            dh = max(1, int(round(view_h * dw / view_w)))
            return np.zeros((dh, dw, 3), np.uint8)
        h, w = bgr.shape[:2]
        if w > DISPLAY_MAX_W:
            scale = DISPLAY_MAX_W / w
            bgr = cv2.resize(bgr, (DISPLAY_MAX_W, max(1, int(round(h * scale)))),
                             interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(bgr)

    def _display_bbox(self, region, width, height):
        x0, y0, x1, y1 = region["frac"]
        vx0, vy0, vx1, vy1 = self._render_frac
        x0 = (x0 - vx0) / (vx1 - vx0); x1 = (x1 - vx0) / (vx1 - vx0)
        y0 = (y0 - vy0) / (vy1 - vy0); y1 = (y1 - vy0) / (vy1 - vy0)
        dx0 = max(0, min(width - 1, int(round(x0 * width))))
        dy0 = max(0, min(height - 1, int(round(y0 * height))))
        dx1 = max(dx0 + 1, min(width, int(round(x1 * width))))
        dy1 = max(dy0 + 1, min(height, int(round(y1 * height))))
        return dx0, dy0, dx1, dy1

    def _overlay_field(self, y0, x0, y1, x1):
        """(th, tw) field for the current overlay mode at the current frame/window."""
        lo, hi = self._win_slice(self.frame, self.win_frames)
        neff = max(1, hi - lo)
        mode = self.overlay_mode.currentText()
        if mode == "Texture":
            return self.texture[self.frame, y0:y1, x0:x1], None
        if mode == "Tensor speed":
            return self.tensor_speed[self.frame, y0:y1, x0:x1], None
        if mode == "Cached flow speed":
            return self.cached_speed[self.frame, y0:y1, x0:x1], None
        if mode == "Relative speed disagreement":
            return self.speed_disagreement[self.frame, y0:y1, x0:x1], (0.0, 1.0)
        if mode == "Appearance energy":
            return (self._ca[hi, y0:y1, x0:x1] -
                    self._ca[lo, y0:y1, x0:x1]) / neff, None
        change = (self._cc[hi, y0:y1, x0:x1] - self._cc[lo, y0:y1, x0:x1]) / neff
        if mode == "Change energy Jtt":
            return change, None
        if mode == "Amplitude variance":
            mi = (self._ci[hi, y0:y1, x0:x1] - self._ci[lo, y0:y1, x0:x1]) / neff
            mi2 = (self._ci2[hi, y0:y1, x0:x1] - self._ci2[lo, y0:y1, x0:x1]) / neff
            return np.maximum(mi2 - mi * mi, 0.0), None
        # appearance fraction
        appear = (self._ca[hi, y0:y1, x0:x1] - self._ca[lo, y0:y1, x0:x1]) / neff
        frac = np.where(change > self.change_floor, appear / (change + EPS), 0.0)
        return frac, (0.0, 1.0)

    def _redraw_video(self):
        base = self._base_frame()
        ch, cw = base.shape[:2]
        raw = self._overlay_peek_hidden or \
            self.overlay_mode.currentText() == "Raw frame"
        out = base.copy()
        active = (range(len(self.regions)) if self.active_region_index < 0
                  else [self.active_region_index])
        cmap = cv2.COLORMAP_VIRIDIS \
            if self.overlay_mode.currentText() in (
                "Appearance fraction", "Relative speed disagreement") \
            else cv2.COLORMAP_TURBO
        for ri in list(active):
            region = self.regions[ri]
            y0, x0, y1, x1 = region["atlas_bbox"]
            dx0, dy0, dx1, dy1 = self._display_bbox(region, cw, ch)
            rw, rh = dx1 - dx0, dy1 - dy0
            roi = out[dy0:dy1, dx0:dx1]
            field, fixed = self._overlay_field(y0, x0, y1, x1)
            if not raw:
                if fixed is not None:
                    lo, hi = fixed
                    norm = np.clip((field - lo) / max(hi - lo, EPS), 0, 1)
                else:
                    norm = np.clip(field / max(self._overlay_scale(), EPS), 0, 1)
                heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
                heat = cv2.resize(heat, (rw, rh), interpolation=cv2.INTER_NEAREST)
                out[dy0:dy1, dx0:dx1] = cv2.addWeighted(roi, 0.45, heat, 0.55, 0)

        if self.hi_chk.isChecked() and not self._overlay_peek_hidden:
            lo_i, hi_i = self._win_slice(self.frame, self.win_frames)
            neff = np.array([max(1, hi_i - lo_i)], np.float32)
            band_lo, band_hi = self._band()
            for ri in list(active):
                region = self.regions[ri]
                y0, x0, y1, x1 = region["atlas_bbox"]
                dx0, dy0, dx1, dy1 = self._display_bbox(region, cw, ch)
                fld = self._detect_field_full(
                    y0, x0, y1, x1, np.array([hi_i]), np.array([lo_i]), neff)[0]
                m = ((fld >= band_lo) & (fld <= band_hi)).astype(np.uint8)
                mm = cv2.resize(m, (dx1 - dx0, dy1 - dy0),
                                interpolation=cv2.INTER_NEAREST)
                roi = out[dy0:dy1, dx0:dx1]
                tint = np.zeros_like(roi)
                tint[..., 1] = 255
                blended = cv2.addWeighted(roi, 0.5, tint, 0.5, 0)
                np.copyto(roi, blended, where=(mm > 0)[:, :, None])
                contours, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(roi, contours, -1, (60, 255, 60), 1)

        self.video_view.set_frame(out, image_frac=self._render_frac,
                                  coordinate_size=(self.src_w, self.src_h))

    def _overlay_scale(self) -> float:
        """Percentile display scale for the current unbounded overlay channel.

        Cached per (mode, window) so scrubbing does not re-percentile the clip;
        the appearance-fraction overlay is bounded [0,1] and never reaches here.
        """
        mode = self.overlay_mode.currentText()
        if mode in ("Tensor speed", "Cached flow speed"):
            # A shared scale makes flipping between estimators a real visual
            # comparison rather than independently recolored heatmaps.
            return self.speed_display_vmax
        # Scope is part of the key: each replicate lives on its own value
        # scale, and a scale frozen from another scope would recolor this one.
        key = (mode, self.win_frames, self.active_region_index)
        if getattr(self, "_scale_cache_key", None) != key:
            if mode == "Texture":
                src = self.texture
            elif mode == "Tensor speed":
                src = self.tensor_speed
            elif mode == "Cached flow speed":
                src = self.cached_speed
            else:
                # Reuse the windowed cache _recompute_series just built:
                # recomputing _owned_windowed here doubled the full-clip
                # windowed pass (six (T, B) allocations) on every window or
                # scope change, purely to derive a display percentile.
                key_by_mode = {"Amplitude variance": "variance",
                               "Change energy Jtt": "change",
                               "Appearance energy": "appearance"}
                src = self._w_cache[key_by_mode[mode]]
            finite = src[np.isfinite(src)]
            self._scale_cache = max(
                float(np.percentile(finite, 99.0)) if finite.size else 1.0, 1e-4)
            self._scale_cache_key = key
        return self._scale_cache

    def showEvent(self, event):
        super().showEvent(event)
        self._redraw_video()

    def resizeEvent(self, _):
        self.video_view.update()

    def closeEvent(self, event):
        if self._event_filter_app is not None:
            self._event_filter_app.removeEventFilter(self)
            self._event_filter_app = None
        if self._owns_source and self.source is not None:
            self.source.release()
            self.source = None
        super().closeEvent(event)


# Compatibility for the early tool-3 draft and its original launcher.  New code
# should import StructureTensorExplorer from gui.explorers.structure_tensor_explorer.
TemporalVarianceExplorer = StructureTensorExplorer
