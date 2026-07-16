"""Embeddable per-block coherent-flow integration explorer.

Sibling to the speed explorer, and deliberately its opposite end of the temporal
spectrum. Where the speed explorer asks "how much motion is there this frame?",
this asks "if I integrate the flow VECTOR over a trailing window, what slow,
coherent drift survives -- and what does integrating longer buy me?"

The physics it makes tangible (see the project's motion-scale discussion):

    net displacement   D(t,W) = sum_{tau in window} (u,v) * dt        [px, vector]
    path length        L(t,W) = sum_{tau in window} |(u,v)| * dt      [px, scalar]
    straightness       C = |D| / L   in [0, 1]
    drift speed        |D| / (W*dt)                                   [px/s]

There are two DIFFERENT discriminators here and they do different jobs -- a
distinction the headless check (scripts scratchpad) makes concrete:

  * Straightness C is a SHAPE gate. It cleanly separates pure oscillation
    (a wingbeat returns to where it started: |D|->0, L keeps growing, C->0)
    from translation-dominated motion (C->1). But for a slow drift BURIED in
    high-amplitude jitter, C does NOT rise with W -- it asymptotes to
    v_drift / mean|v|, which is small when jitter >> drift, no matter how long
    you integrate, because L is dominated by the jitter. So C rejects
    oscillation; it does not by itself denoise.

  * The drift-speed estimate |D|/(W*dt) is the DENOISING readout. The vector
    integral cancels zero-mean jitter (its contribution to |D| grows only as
    sqrt(W)) while a true drift accumulates linearly (v0*W*dt), so the estimate
    converges to the true slow velocity as W grows -- SNR ~ sqrt(W). THIS is
    the "value of integration" the vs-window plots visualise; watch drift speed
    flatten toward the true velocity while per-frame speed stays swamped.

Practical consequence for detection: gate anti-oscillation on straightness and
anti-noise on the drift-speed estimate. A high straightness threshold alone
will hide real slow movers that live under heavy flow noise.

Like the speed explorer it recomputes NOTHING about optical flow: it reads the
cached per-block u/v/speed arrays the real pipeline produced and integrates
them. It requires vector flow (u, v); a speed-only cache cannot be explored here.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from core.replicates import block_weight_plane

from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QHBoxLayout,
                             QLabel, QPushButton, QScrollArea, QSlider,
                             QVBoxLayout, QWidget)

from gui.video_panel import FrameView
# Reuse the speed explorer's sparkline, region normalisation and palette so the
# two tools stay visually and behaviourally identical where they overlap.
from gui.speed_explorer import (MiniPlot, _regions_from_meta, DISPLAY_MAX_W,
                                LINE, LINE2, CURSOR)

DRIFT_C = QColor(150, 200, 255)     # integrated / drift series
COH_C = QColor(180, 140, 255)       # straightness / coherence
SWEEP_C = QColor(110, 230, 120)     # threshold-sweep series
VW_C = QColor(255, 150, 90)         # value-vs-window series
EPS = 1e-6


class CoherentFlowExplorer(QWidget):
    """Cache-backed explorer for temporal integration of the flow vector.

    Runs standalone (pass ``cache`` + ``video_path``) or embedded over the shared
    AppState (``from_app_state``), mirroring :class:`SpeedExplorer`.
    """

    def __init__(self, cache=None, video_path: str | None = None, *, state=None,
                 parent=None):
        super().__init__(parent)
        if state is not None:
            cache = state.cache
        if cache is None:
            raise ValueError("CoherentFlowExplorer requires an open feature cache")
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

        # Vector flow is mandatory here: integration is defined on (u, v), not on
        # the sign-free speed magnitude.
        feats = set(self.meta.get("features", []))
        if not {"u", "v"}.issubset(feats):
            raise ValueError(
                "This cache has no vector flow (u, v). Coherent-flow integration "
                "needs the flow direction, not just speed. Rebuild the cache with "
                "u/v caching enabled, or use the Speed Explorer instead.")

        ctx = state.ctx if state is not None else None
        u = np.asarray(ctx.u if ctx is not None else cache.read("u"), np.float32)
        v = np.asarray(ctx.v if ctx is not None else cache.read("v"), np.float32)
        speed_src = ctx.speed if ctx is not None else cache.read("speed")
        self.speed = np.asarray(speed_src, np.float32)      # (T, ny, nx), px/s
        self.T = self.speed.shape[0]

        # Prefix sums with a leading zero row so a trailing window [lo, t] is a
        # single subtraction and the clip's opening ramp (t < W) is handled
        # exactly instead of being special-cased. Units stay px/s summed; the
        # division by fps that turns them into pixels happens at read time.
        z = np.zeros((1, self.ny, self.nx), np.float32)
        self._cu = np.concatenate([z, np.cumsum(u, axis=0, dtype=np.float32)])
        self._cv = np.concatenate([z, np.cumsum(v, axis=0, dtype=np.float32)])
        self._cs = np.concatenate([z, np.cumsum(self.speed, axis=0,
                                                dtype=np.float32)])
        del u, v   # only the prefix sums are needed from here on.

        self.source = None
        self._owns_source = False
        if state is None and video_path and os.path.exists(video_path):
            from core.video import VideoSource
            try:
                self.source = VideoSource(video_path)
                self._owns_source = True
            except Exception:
                self.source = None
        self.src_w = int(self.meta.get("src_width", 0))
        self.src_h = int(self.meta.get("src_height", 0))
        dimension_source = state.source if state is not None else self.source
        if dimension_source is not None:
            self.src_w = self.src_w or int(dimension_source.info.width)
            self.src_h = self.src_h or int(dimension_source.info.height)
        self.src_w = max(1, self.src_w or int(self.meta.get("work_width", 1)))
        self.src_h = max(1, self.src_h or int(self.meta.get("work_height", 1)))

        # Integration window, in frames. Default ~1 s (or half the clip if short)
        # -- long enough that a slow drift separates from jitter, short enough to
        # still localise a behaviour in time.
        self.win_frames = max(2, min(self.T - 1, int(round(self.fps))))
        self.frame = int(state.current_frame) if state is not None else 0
        self.playing = False
        self._overlay_peek_hidden = False

        self._rebuild_owned_prefixes()
        # Display/threshold scales, computed once over the whole clip at the
        # default window so the sliders and heatmap keep a stable mapping as the
        # user scrubs and refocuses.
        self.disp_vmax = self._percentile_over_clip("disp", 99.9)
        self.drift_vmax = self._percentile_over_clip("drift", 99.9)
        # Straightness default is a modest anti-oscillation floor, NOT a tight
        # gate: see the module docstring -- a real slow mover under heavy flow
        # noise has low straightness, so most of the discrimination should come
        # from the drift-speed threshold instead.
        self.c_thresh = 0.35
        self.drift_thresh = self.drift_vmax * 0.3

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
        self._recompute_sweep()
        self._recompute_value_curve()
        self._apply_frame(self.frame)

    @classmethod
    def from_app_state(cls, state, parent=None) -> "CoherentFlowExplorer":
        if not state.has_cache:
            raise ValueError("Open a feature cache before creating this explorer")
        return cls(state=state, parent=parent)

    # -- owned-block bookkeeping (mirrors SpeedExplorer) ----------------------

    def _active_regions(self) -> list[dict]:
        if self.active_region_index < 0:
            return self.regions
        return [self.regions[self.active_region_index]]

    def _active_block_count(self) -> int:
        return sum((r["atlas_bbox"][2] - r["atlas_bbox"][0]) *
                   (r["atlas_bbox"][3] - r["atlas_bbox"][1])
                   for r in self._active_regions())

    def _rebuild_owned_prefixes(self) -> None:
        """Flatten the prefix sums to only the currently-owned blocks: (T+1, B).

        Every per-frame series and threshold sweep reads from these, so a scope
        change is the only place the owned set is resolved.
        """
        cu, cv, cs = [], [], []
        for region in self._active_regions():
            y0, x0, y1, x1 = region["atlas_bbox"]
            cu.append(self._cu[:, y0:y1, x0:x1].reshape(self.T + 1, -1))
            cv.append(self._cv[:, y0:y1, x0:x1].reshape(self.T + 1, -1))
            cs.append(self._cs[:, y0:y1, x0:x1].reshape(self.T + 1, -1))
        self._cu_own = cu[0] if len(cu) == 1 else np.concatenate(cu, axis=1)
        self._cv_own = cv[0] if len(cv) == 1 else np.concatenate(cv, axis=1)
        self._cs_own = cs[0] if len(cs) == 1 else np.concatenate(cs, axis=1)
        self.n_blocks = self._cu_own.shape[1]

    def _window_bounds(self, W: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-frame trailing-window [lo, hi) prefix indices and length."""
        hi = np.arange(self.T) + 1
        lo = np.maximum(0, hi - W)
        return hi, lo, (hi - lo).astype(np.float32)

    def _owned_integration(self, W: int) -> dict[str, np.ndarray]:
        """(T, B) integrated quantities over the owned blocks for window W."""
        hi, lo, neff = self._window_bounds(W)
        inv_fps = self.dt
        dx = (self._cu_own[hi] - self._cu_own[lo]) * inv_fps
        dy = (self._cv_own[hi] - self._cv_own[lo]) * inv_fps
        dmag = np.hypot(dx, dy)
        path = (self._cs_own[hi] - self._cs_own[lo]) * inv_fps
        straight = np.where(path > EPS, dmag / np.maximum(path, EPS), 0.0)
        drift = dmag * self.fps / neff[:, None]
        return {"dmag": dmag, "path": path, "straight": straight, "drift": drift}

    def _percentile_over_clip(self, which: str, pct: float) -> float:
        vals = self._owned_integration(self.win_frames)[
            "dmag" if which == "disp" else which]
        finite = vals[np.isfinite(vals)]
        return max(float(np.percentile(finite, pct)) if finite.size else 1.0, 1e-3)

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
            "Net displacement heatmap", "Straightness heatmap",
            "Drift vectors (integrated)", "Per-frame speed heatmap",
            "Raw frame"])
        self.overlay_mode.setCurrentText("Net displacement heatmap")
        self.overlay_mode.currentTextChanged.connect(lambda _: self._redraw_video())
        orow.addWidget(self.overlay_mode, 1)
        self.hi_chk = QCheckBox("Highlight coherent slow-mover blocks")
        self.hi_chk.setChecked(True)
        self.hi_chk.stateChanged.connect(lambda _: self._redraw_video())
        orow.addWidget(self.hi_chk)
        left.addLayout(orow)

        # -- the three knobs ---------------------------------------------------
        self.win_slider, self.win_lbl = self._add_slider(
            left, 2, max(3, self.T - 1), self.win_frames, self._on_window)
        self.c_slider, self.c_lbl = self._add_slider(
            left, 0, 1000, int(self.c_thresh * 1000), self._on_c_thresh)
        self.drift_slider, self.drift_lbl = self._add_slider(
            left, 0, 1000, 300, self._on_drift_thresh)

        note = QLabel(
            "Window W integrates the flow VECTOR over the last W frames. "
            "Straightness = |net displacement| / path length: ~1 is coherent "
            "drift, ~0 is jitter that cancels. Widen W to watch slow coherent "
            "motion climb out of the per-frame noise floor.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8ab; padding-top:2px;")
        left.addWidget(note)

        info = QLabel(
            f"cache: {self.meta.get('backend', '?')} | fps {self.fps:.2f} | "
            f"block {self.block}px | {self.T} frames | "
            f"{len(self.regions)} {'replicate tiles' if self.packed else 'region'}")
        info.setStyleSheet("color:#888;")
        left.addWidget(info)
        root.addLayout(left, 3)

        # -- right: scrollable readouts ---------------------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        self.plot_col = QVBoxLayout(holder)
        self.plot_col.setSpacing(3)
        scroll.setWidget(holder)
        scroll.setMinimumWidth(440)
        root.addWidget(scroll, 2)

        self.plots: dict[str, MiniPlot] = {}

        def section(text: str):
            lbl = QLabel(text)
            lbl.setStyleSheet("color:#7fd7ff; font-weight:bold; padding-top:6px;")
            self.plot_col.addWidget(lbl)

        def add(key, title, unit, color=LINE, seek=True):
            pl = MiniPlot(title, unit, color)
            if seek:
                pl.seek_requested.connect(self._seek)
            self.plots[key] = pl
            self.plot_col.addWidget(pl)

        section("Per-frame baseline (no integration)")
        add("raw_speed", "Mean per-frame speed", "px/s", LINE)

        section("Integrated over trailing window W (time axis)")
        add("disp", "Mean net displacement |D|", "px", DRIFT_C)
        add("path", "Mean path length L", "px", LINE2)
        add("straight", "Mean straightness |D|/L", "", COH_C)
        add("drift", "Mean drift speed |D|/(W dt)", "px/s", DRIFT_C)

        section("Coherent slow-mover sweep (straightness x drift, time axis)")
        add("count", "# coherent slow-mover blocks", "blk", SWEEP_C)
        add("frac", "Fraction of blocks", "", SWEEP_C)
        add("clump", "Largest connected coherent clump", "blk", SWEEP_C)
        add("cond_drift", "Mean drift OF coherent blocks", "px/s", SWEEP_C)

        section("Value of integration @ cursor frame (x axis = window length)")
        add("vw_straight", "Straightness vs W", "", VW_C, seek=False)
        add("vw_disp", "Net displacement vs W", "px", VW_C, seek=False)
        add("vw_drift", "Drift speed vs W (flattens to true v)", "px/s", VW_C,
            seek=False)
        self.plot_col.addStretch(1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        # Window and threshold recomputes touch every block over the whole clip,
        # so debounce them to keep dragging smooth.
        self._win_debounce = QTimer(self)
        self._win_debounce.setSingleShot(True)
        self._win_debounce.setInterval(120)
        self._win_debounce.timeout.connect(self._apply_window_change)
        self._sweep_debounce = QTimer(self)
        self._sweep_debounce.setSingleShot(True)
        self._sweep_debounce.setInterval(120)
        self._sweep_debounce.timeout.connect(self._recompute_sweep)
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

    # -- series computation ---------------------------------------------------

    def _recompute_series(self):
        integ = self._owned_integration(self.win_frames)
        self.plots["raw_speed"].set_series(
            (self._cs_own[1:] - self._cs_own[:-1]).mean(1))  # per-frame speed
        self.plots["disp"].set_series(integ["dmag"].mean(1))
        self.plots["path"].set_series(integ["path"].mean(1))
        self.plots["straight"].set_series(integ["straight"].mean(1))
        self.plots["drift"].set_series(integ["drift"].mean(1))
        self._integ_cache = integ

    def _recompute_sweep(self):
        integ = self._integ_cache
        mask = (integ["straight"] > self.c_thresh) & \
               (integ["drift"] > self.drift_thresh)
        count = mask.sum(1).astype(np.float32)
        self.plots["count"].set_series(count)
        self.plots["frac"].set_series(count / max(1, self.n_blocks))
        with np.errstate(invalid="ignore"):
            cond = np.where(count > 0,
                            np.where(mask, integ["drift"], 0.0).sum(1) /
                            np.maximum(count, 1), 0.0)
        self.plots["cond_drift"].set_series(cond.astype(np.float32))
        self._recompute_clump()

    def _recompute_clump(self):
        """Largest 8-connected clump of coherent slow-mover blocks, per frame.

        Computed on the full grid per active region (never bridging packed
        replicate tiles) and area-weighted like roi_detection, so a one-row edge
        sliver never counts as a whole block.
        """
        W = self.win_frames
        hi, lo, neff = self._window_bounds(W)
        weight_plane = block_weight_plane(self.meta)
        largest = np.zeros(self.T, np.float32)
        for region in self._active_regions():
            y0, x0, y1, x1 = region["atlas_bbox"]
            w_flat = weight_plane[y0:y1, x0:x1].reshape(-1)
            dx = (self._cu[hi, y0:y1, x0:x1] - self._cu[lo, y0:y1, x0:x1]) * self.dt
            dy = (self._cv[hi, y0:y1, x0:x1] - self._cv[lo, y0:y1, x0:x1]) * self.dt
            path = (self._cs[hi, y0:y1, x0:x1] - self._cs[lo, y0:y1, x0:x1]) * self.dt
            dmag = np.hypot(dx, dy)
            straight = np.where(path > EPS, dmag / np.maximum(path, EPS), 0.0)
            drift = dmag * self.fps / neff[:, None, None]
            coh = ((straight > self.c_thresh) &
                   (drift > self.drift_thresh)).astype(np.uint8)
            for t in range(self.T):
                m = coh[t]
                if not m.any():
                    continue
                n_lab, labels, _, _ = cv2.connectedComponentsWithStats(
                    m, connectivity=8)
                if n_lab > 1:
                    areas = np.bincount(labels.reshape(-1), weights=w_flat,
                                        minlength=n_lab)
                    largest[t] = max(largest[t], float(areas[1:].max()))
        self.plots["clump"].set_series(largest)

    def _recompute_value_curve(self):
        """Integrated quantities at the cursor frame as W sweeps 1..T.

        This is the "what does integrating longer buy me" plot: straightness and
        net displacement should climb as a real slow drift accumulates, while
        drift speed flattens toward the true velocity once the jitter averages
        out. x axis is window length, not time.
        """
        t0 = self.frame
        Ws = np.unique(np.round(
            np.geomspace(1, self.T, num=min(160, self.T))).astype(int))
        Ws = Ws[Ws >= 1]
        hi = t0 + 1
        los = np.maximum(0, hi - Ws)
        neff = (hi - los).astype(np.float32)
        dx = (self._cu_own[hi][None, :] - self._cu_own[los]) * self.dt
        dy = (self._cv_own[hi][None, :] - self._cv_own[los]) * self.dt
        dmag = np.hypot(dx, dy)
        path = (self._cs_own[hi][None, :] - self._cs_own[los]) * self.dt
        straight = np.where(path > EPS, dmag / np.maximum(path, EPS), 0.0)
        drift = dmag * self.fps / neff[:, None]
        self._vw_Ws = Ws
        self.plots["vw_straight"].set_series(straight.mean(1))
        self.plots["vw_disp"].set_series(dmag.mean(1))
        self.plots["vw_drift"].set_series(drift.mean(1))
        self._sync_value_cursor()

    def _sync_value_cursor(self):
        # The value plots' x axis is window length; point their cursor at the
        # current W, not the current frame.
        idx = int(np.searchsorted(self._vw_Ws, self.win_frames))
        idx = max(0, min(idx, self._vw_Ws.size - 1))
        for key in ("vw_straight", "vw_disp", "vw_drift"):
            self.plots[key].set_cursor(idx)

    # -- control handlers -----------------------------------------------------

    def _sync_labels(self):
        self.win_lbl.setText(
            f"Integration window W: {self.win_frames} fr "
            f"({self.win_frames * self.dt:.2f} s)")
        self.c_lbl.setText(f"Straightness threshold: {self.c_thresh:.2f}")
        self.drift_lbl.setText(f"Drift-speed threshold: {self.drift_thresh:.2f} px/s")

    def _on_window(self, v: int):
        self.win_frames = max(2, int(v))
        self._sync_labels()
        self._win_debounce.start()
        self._sync_value_cursor()

    def _apply_window_change(self):
        self._recompute_series()
        self._recompute_sweep()
        self._recompute_value_curve()
        self._redraw_video()

    def _on_c_thresh(self, v: int):
        self.c_thresh = v / 1000.0
        self._sync_labels()
        self._sweep_debounce.start()
        self._redraw_video()

    def _on_drift_thresh(self, v: int):
        self.drift_thresh = v / 1000.0 * self.drift_vmax
        self._sync_labels()
        self._sweep_debounce.start()
        self._redraw_video()

    def _on_region_changed(self, _index: int):
        data = self.region_combo.currentData()
        self.active_region_index = int(data) if data is not None else 0
        focus = None if self.active_region_index < 0 else \
            self.regions[self.active_region_index]["frac"]
        self.video_view.set_focus_frac(focus)
        self._sync_video_boxes()
        self._rebuild_owned_prefixes()
        self._recompute_series()
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
                combo_index = self.region_combo.findData(i)
                if combo_index >= 0:
                    self.region_combo.setCurrentIndex(combo_index)
                return

    def _clear_region_focus(self):
        pooled_index = self.region_combo.findData(-1)
        if pooled_index >= 0:
            self.region_combo.setCurrentIndex(pooled_index)
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
            f"Coherent flow integration -- "
            f"{os.path.basename(self.meta.get('video_path', '?'))} -- {scope} "
            f"({self.T} frames, {self.n_blocks} blocks)")

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

    def _on_scrub(self, v: int):
        if v != self.frame:
            self._update_frame(v)

    def _seek(self, frame: int):
        self._update_frame(frame)

    def _update_frame(self, frame: int):
        frame = max(0, min(int(frame), self.T - 1))
        if self.state is not None and self.state.current_frame != frame:
            self.state.set_frame(frame)
            return
        self._apply_frame(frame)

    def _on_state_frame_changed(self, frame: int):
        self._apply_frame(frame)

    def _apply_frame(self, frame: int):
        self.frame = max(0, min(int(frame), self.T - 1))
        if self.scrub.value() != self.frame:
            self.scrub.blockSignals(True)
            self.scrub.setValue(self.frame)
            self.scrub.blockSignals(False)
        self.time_lbl.setText(f"{self.frame / self.fps:.2f} s  (#{self.frame})")
        for key, pl in self.plots.items():
            if not key.startswith("vw_"):
                pl.set_cursor(self.frame)
        self._recompute_value_curve()   # value curve is anchored at the cursor
        if self.isVisible():
            self._redraw_video()

    # -- video + overlay ------------------------------------------------------

    def _base_frame(self) -> np.ndarray:
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

    def _frame_fields(self, y0, x0, y1, x1) -> dict[str, np.ndarray]:
        """Integrated fields for the current frame/window over one atlas tile."""
        W = self.win_frames
        hi = self.frame + 1
        lo = max(0, hi - W)
        neff = max(1, hi - lo)
        dx = (self._cu[hi, y0:y1, x0:x1] - self._cu[lo, y0:y1, x0:x1]) * self.dt
        dy = (self._cv[hi, y0:y1, x0:x1] - self._cv[lo, y0:y1, x0:x1]) * self.dt
        path = (self._cs[hi, y0:y1, x0:x1] - self._cs[lo, y0:y1, x0:x1]) * self.dt
        dmag = np.hypot(dx, dy)
        straight = np.where(path > EPS, dmag / np.maximum(path, EPS), 0.0)
        drift = dmag * self.fps / neff
        return {"dx": dx, "dy": dy, "dmag": dmag, "straight": straight,
                "drift": drift}

    def _redraw_video(self):
        base = self._base_frame()
        ch, cw = base.shape[:2]
        mode = "Raw frame" if self._overlay_peek_hidden else \
            self.overlay_mode.currentText()
        out = base.copy()
        active = (range(len(self.regions)) if self.active_region_index < 0
                  else [self.active_region_index])
        for ri in list(active):
            region = self.regions[ri]
            y0, x0, y1, x1 = region["atlas_bbox"]
            dx0, dy0, dx1, dy1 = self._display_bbox(region, cw, ch)
            rw, rh = dx1 - dx0, dy1 - dy0
            roi = out[dy0:dy1, dx0:dx1]
            f = self._frame_fields(y0, x0, y1, x1)

            if mode == "Net displacement heatmap":
                norm = np.clip(f["dmag"] / self.disp_vmax, 0, 1)
                heat = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                         cv2.COLORMAP_TURBO)
                heat = cv2.resize(heat, (rw, rh), interpolation=cv2.INTER_NEAREST)
                out[dy0:dy1, dx0:dx1] = cv2.addWeighted(roi, 0.45, heat, 0.55, 0)
            elif mode == "Straightness heatmap":
                norm = np.clip(f["straight"], 0, 1)
                heat = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                         cv2.COLORMAP_VIRIDIS)
                heat = cv2.resize(heat, (rw, rh), interpolation=cv2.INTER_NEAREST)
                out[dy0:dy1, dx0:dx1] = cv2.addWeighted(roi, 0.45, heat, 0.55, 0)
            elif mode == "Per-frame speed heatmap":
                sp = self.speed[self.frame, y0:y1, x0:x1]
                norm = np.clip(sp / max(self.drift_vmax, EPS), 0, 1)
                heat = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                         cv2.COLORMAP_TURBO)
                heat = cv2.resize(heat, (rw, rh), interpolation=cv2.INTER_NEAREST)
                out[dy0:dy1, dx0:dx1] = cv2.addWeighted(roi, 0.45, heat, 0.55, 0)
            elif mode == "Drift vectors (integrated)":
                gy, gx = f["dmag"].shape
                cell_w, cell_h = rw / gx, rh / gy
                vscale = min(cell_w, cell_h) / max(self.disp_vmax, EPS) * 1.5
                for by in range(0, gy, 2):
                    for bx in range(0, gx, 2):
                        if f["straight"][by, bx] < self.c_thresh:
                            continue
                        ax0 = int(dx0 + (bx + 0.5) * cell_w)
                        ay0 = int(dy0 + (by + 0.5) * cell_h)
                        ax1 = int(ax0 + f["dx"][by, bx] * vscale)
                        ay1 = int(ay0 + f["dy"][by, bx] * vscale)
                        cv2.arrowedLine(out, (ax0, ay0), (ax1, ay1),
                                        (255, 200, 150), 1, tipLength=0.35)

        if self.hi_chk.isChecked() and not self._overlay_peek_hidden:
            for ri in list(active):
                region = self.regions[ri]
                y0, x0, y1, x1 = region["atlas_bbox"]
                dx0, dy0, dx1, dy1 = self._display_bbox(region, cw, ch)
                f = self._frame_fields(y0, x0, y1, x1)
                m = ((f["straight"] > self.c_thresh) &
                     (f["drift"] > self.drift_thresh)).astype(np.uint8)
                mm = cv2.resize(m, (dx1 - dx0, dy1 - dy0),
                                interpolation=cv2.INTER_NEAREST)
                roi = out[dy0:dy1, dx0:dx1]
                tint = np.zeros_like(roi)
                tint[..., 0] = 255      # blue-ish tint for coherent drift
                tint[..., 1] = 180
                blended = cv2.addWeighted(roi, 0.5, tint, 0.5, 0)
                np.copyto(roi, blended, where=(mm > 0)[:, :, None])
                contours, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(roi, contours, -1, (255, 210, 120), 1)

        self.video_view.set_frame(out, image_frac=self._render_frac,
                                  coordinate_size=(self.src_w, self.src_h))

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
