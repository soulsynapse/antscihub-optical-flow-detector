"""Embeddable per-block speed explorer.

A throwaway diagnostic to answer one question the main app currently makes you
guess at: what does the speed signal actually look like, per block, over time --
and in particular, what does the "number of blocks above a threshold" signal (the
thing detection really thresholds) do as you move the threshold?

It does NOT recompute optical flow. It opens a feature cache you already built in
the main app (Preprocessing & Flow) and reads its cached `speed` / `u` / `v`
block arrays, so
every number here is exactly the number the real pipeline sees -- there is no
second, slightly-different flow implementation to mistrust.

Left panel  : the video with replicate-aware optical-flow overlay + transport.
Right panel : a long scrollable column of speed-only time readouts. The
              threshold slider drives the "above threshold" group live, so you can
              watch the block-count signal reshape itself as you sweep the cutoff.
              A separate detection minimum drops quieter blocks before the
              detection signal is averaged over a trailing time window.

Current caches process each replicate independently and pack their block grids
into a sparse storage atlas.  This explorer maps those tiles back to their real
source-frame boxes; atlas separators are never displayed or included in plots.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from core.replicates import block_weight_plane

from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QKeySequence, QPainter, QPen, QPolygonF,
                         QShortcut)
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                             QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                             QSlider, QVBoxLayout, QWidget)

from gui.video_panel import FrameView

# -- palette -----------------------------------------------------------------
BG = QColor(24, 24, 24)
PLOT_BG = QColor(12, 12, 12)
LINE = QColor(120, 215, 255)
LINE2 = QColor(255, 170, 80)
CURSOR = QColor(255, 210, 80)
TXT = QColor(210, 210, 210)
TXT_DIM = QColor(140, 140, 140)
DISPLAY_MAX_W = 1280


def _regions_from_meta(meta: dict, grid: tuple[int, int]) -> list[dict]:
    """Return validated cache regions, with a whole-frame legacy fallback.

    ``replicate_tiles`` describe sparse atlas storage, not image coordinates.
    Keeping this normalization in one place makes every plot and overlay use the
    same ownership boundary.
    """
    ny, nx = grid
    raw = meta.get("replicate_tiles", [])
    if not raw:
        src_w = int(meta.get("src_width", meta.get("work_width", nx)))
        src_h = int(meta.get("src_height", meta.get("work_height", ny)))
        return [{
            "id": None,
            "label": "Whole frame",
            "frac": (0.0, 0.0, 1.0, 1.0),
            "source_box": (0, 0, src_w, src_h),
            "grid": (ny, nx),
            "atlas_bbox": (0, 0, ny, nx),
        }]

    regions = []
    for i, tile in enumerate(raw):
        try:
            y0, x0, y1, x1 = map(int, tile["atlas_bbox"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Replicate tile {i} has no valid atlas_bbox") from exc
        if not (0 <= y0 < y1 <= ny and 0 <= x0 < x1 <= nx):
            raise ValueError(
                f"Replicate tile {tile.get('id', i)} has atlas_bbox "
                f"{(y0, x0, y1, x1)} outside cache grid {(ny, nx)}")
        tile_grid = tuple(map(int, tile.get("grid", (y1 - y0, x1 - x0))))
        if tile_grid != (y1 - y0, x1 - x0):
            raise ValueError(
                f"Replicate tile {tile.get('id', i)} grid {tile_grid} does not "
                f"match atlas_bbox {(y0, x0, y1, x1)}")

        frac = tuple(float(v) for v in tile.get("frac", (0, 0, 1, 1)))
        source_box = tuple(map(int, tile.get("source_box", (0, 0, 1, 1))))
        if len(frac) != 4 or len(source_box) != 4:
            raise ValueError(f"Replicate tile {tile.get('id', i)} has bad geometry")
        regions.append({
            **tile,
            "id": int(tile.get("id", i)),
            "label": str(tile.get("label", f"rep{tile.get('id', i)}")),
            "frac": frac,
            "source_box": source_box,
            "grid": tile_grid,
            "atlas_bbox": (y0, x0, y1, x1),
        })
    return regions


# -- one time-series readout -------------------------------------------------

class MiniPlot(QWidget):
    """A compact autoscaled sparkline for one (T,) series, with a frame cursor.

    Long series are decimated to the widget width by taking the MAX within each
    pixel column, not the mean: a behavior is a brief burst, and the mean per
    column would smear a 3-frame spike into the baseline -- exactly the signal we
    are here to see. The current-frame exact value is printed regardless, so the
    decimation never hides the number you are reading.
    """
    seek_requested = pyqtSignal(int)

    def __init__(self, title: str, unit: str = "", color: QColor = LINE):
        super().__init__()
        self.title = title
        self.unit = unit
        self.color = color
        self.y: np.ndarray = np.zeros(0, np.float32)
        self.cursor = 0
        self.setMinimumHeight(66)
        self.setMaximumHeight(66)

    def set_series(self, y: np.ndarray) -> None:
        self.y = np.asarray(y, np.float32)
        self.update()

    def set_cursor(self, frame: int) -> None:
        self.cursor = int(frame)
        self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(6, 16, max(1, self.width() - 12), max(1, self.height() - 22))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), BG)
        r = self._plot_rect()
        p.fillRect(r, PLOT_BG)

        n = self.y.size
        p.setFont(QFont("Consolas", 7))
        if n == 0:
            p.setPen(TXT_DIM)
            p.drawText(8, 12, self.title)
            p.end()
            return

        lo = float(np.nanmin(self.y))
        hi = float(np.nanmax(self.y))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0

        w = int(r.width())
        cols = max(1, min(n, w))
        edges = np.linspace(0, n, cols + 1).astype(int)
        env = np.empty(cols, np.float32)
        for i in range(cols):
            a, b = edges[i], max(edges[i] + 1, edges[i + 1])
            seg = self.y[a:b]
            env[i] = np.nanmax(seg) if seg.size else lo

        def y_of(val: float) -> float:
            return r.bottom() - (val - lo) / (hi - lo) * r.height()

        # Zero baseline, if zero is in range -- makes "how far above nothing" legible.
        if lo < 0 < hi:
            yb = y_of(0.0)
            p.setPen(QPen(QColor(70, 70, 70), 1, Qt.PenStyle.DashLine))
            p.drawLine(int(r.left()), int(yb), int(r.right()), int(yb))

        poly = QPolygonF()
        for i in range(cols):
            x = r.left() + (i + 0.5) / cols * r.width()
            poly.append(QPointF(x, y_of(env[i])))
        p.setPen(QPen(self.color, 1))
        p.drawPolyline(poly)

        # Cursor + current exact value.
        cx = r.left() + (self.cursor + 0.5) / n * r.width()
        p.setPen(QPen(CURSOR, 1))
        p.drawLine(int(cx), int(r.top()), int(cx), int(r.bottom()))

        cur = float(self.y[min(self.cursor, n - 1)])
        p.setPen(TXT)
        p.drawText(8, 12, self.title)
        val_txt = f"{cur:.4g} {self.unit}".strip()
        p.setPen(QColor(self.color))
        p.drawText(int(r.right()) - 7 * len(val_txt) - 2, 12, val_txt)

        # min/max axis ticks
        p.setPen(TXT_DIM)
        p.drawText(int(r.left()), int(r.bottom()) + 8, f"{lo:.3g}")
        p.end()

    def mousePressEvent(self, e):
        n = self.y.size
        if n == 0:
            return
        r = self._plot_rect()
        frac = np.clip((e.pos().x() - r.left()) / max(1, r.width()), 0, 1)
        self.seek_requested.emit(int(frac * (n - 1)))


# -- the main window ---------------------------------------------------------

class SpeedExplorer(QWidget):
    """Cache-backed explorer that can run alone or follow the shared AppState.

    The standalone launcher passes ``cache`` and ``video_path``.  An eventual
    main-app tab should use :meth:`from_app_state`; frame decoding and seeking then
    go through AppState, so this widget cannot fight the other tabs' decoder or
    establish a second current-frame value.
    """

    def __init__(self, cache=None, video_path: str | None = None, *, state=None,
                 parent=None):
        super().__init__(parent)
        if state is not None:
            cache = state.cache
        if cache is None:
            raise ValueError("SpeedExplorer requires an open feature cache")
        self.state = state
        self.cache = cache
        self.meta = cache.meta
        self.fps = float(self.meta["fps"])
        self.block = int(self.meta["block_size"])
        self.ny, self.nx = map(int, self.meta["grid"])
        self.regions = _regions_from_meta(self.meta, (self.ny, self.nx))
        self.packed = bool(self.meta.get("replicate_tiles"))
        # Default to one experimental unit.  "All" remains available as an
        # explicitly pooled diagnostic, but behavior detection is per replicate.
        self.active_region_index = -1 if len(self.regions) > 1 else 0
        self.n_blocks = self._active_block_count()
        self.src_w = int(self.meta.get("src_width", 0))
        self.src_h = int(self.meta.get("src_height", 0))

        # Read the per-block arrays once, as float32. This is the whole point:
        # these are the same arrays roi_detection thresholds.
        # In-app construction reuses FeatureContext's already-loaded float32
        # planes. Reading them from the cache again would double several GB of
        # memory on a full clip just because this tab was opened.
        ctx = state.ctx if state is not None else None
        speed_source = ctx.speed if ctx is not None else cache.read("speed")
        self.speed = np.asarray(speed_source, dtype=np.float32)  # (T, ny, nx)
        feats = set(self.meta.get("features", []))
        have_vectors = {"u", "v"}.issubset(feats)
        if have_vectors:
            self.u = np.asarray(ctx.u if ctx is not None else cache.read("u"),
                                dtype=np.float32)
            self.v = np.asarray(ctx.v if ctx is not None else cache.read("v"),
                                dtype=np.float32)
        else:
            self.u = self.v = None
        self.T = self.speed.shape[0]

        self.source = None
        self._owns_source = False
        if state is None and video_path and os.path.exists(video_path):
            from core.video import VideoSource
            try:
                self.source = VideoSource(video_path)
                self._owns_source = True
            except Exception:
                self.source = None
        dimension_source = state.source if state is not None else self.source
        if dimension_source is not None:
            self.src_w = self.src_w or int(dimension_source.info.width)
            self.src_h = self.src_h or int(dimension_source.info.height)
        self.src_w = max(1, self.src_w or int(self.meta.get("work_width", 1)))
        self.src_h = max(1, self.src_h or int(self.meta.get("work_height", 1)))

        # Threshold slider is in px/s; cap the range just past the bulk so the
        # slider's travel lands where the data actually is, not on a lone outlier.
        # This scale is intentionally computed once over the initial scope.  For
        # packed caches that is every owned replicate cell (never atlas padding),
        # which gives the slider one stable absolute px/s mapping.  Focusing a
        # replicate must not silently reinterpret or reset an analysis threshold.
        self.vmax = self._active_speed_scale()
        self.detection_min = 0.0
        self.win_frames = max(2, min(self.T - 1, int(round(self.fps))))
        self.detect_on = "per_frame"
        self.thr_vmax = self.vmax
        self.threshold = self.thr_vmax * 0.5

        self.frame = int(state.current_frame) if state is not None else 0
        self.playing = False
        self._overlay_peek_hidden = False

        self._build_ui()
        # Key events go to whichever child currently has focus. Filtering at the
        # application level keeps Shift a press-and-hold raw-video peek after the
        # user touches the video, slider, or overlay menu; other tabs are ignored.
        self._event_filter_app = QApplication.instance()
        if self._event_filter_app is not None:
            self._event_filter_app.installEventFilter(self)
        # Standalone mode has no MainWindow to own the application Space shortcut.
        # Embedded mode deliberately leaves this to MainWindow, avoiding two
        # competing shortcuts for the same key.
        self._space_shortcut = None
        if state is None:
            self._space_shortcut = QShortcut(
                QKeySequence(Qt.Key.Key_Space.value), self)
            self._space_shortcut.setContext(
                Qt.ShortcutContext.ApplicationShortcut)
            self._space_shortcut.setAutoRepeat(False)
            self._space_shortcut.activated.connect(self.toggle_playback)
        if state is not None:
            state.frame_changed.connect(self._on_state_frame_changed)
        self._compute_static_series()
        self._recompute_threshold_series()
        self._update_frame(self.frame)

    @classmethod
    def from_app_state(cls, state, parent=None) -> "SpeedExplorer":
        """Build an embeddable instance over the app's current video/cache."""
        if not state.has_cache:
            raise ValueError("Open a feature cache before creating Speed Explorer")
        return cls(state=state, parent=parent)

    # -- layout --------------------------------------------------------------

    def _build_ui(self):
        self._sync_window_title()
        self.resize(1500, 900)
        root = QHBoxLayout(self)

        # ---- left: video + controls ----
        left = QVBoxLayout()
        self.video_view = FrameView()
        self.video_view.setMinimumSize(720, 480)
        self.video_view.clicked.connect(self._on_video_clicked)
        self.video_view.back_requested.connect(self._clear_region_focus)
        self._sync_video_boxes()
        left.addWidget(self.video_view, 1)

        # transport
        trow = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        trow.addWidget(self.play_btn)
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.setRange(0, self.T - 1)
        self.scrub.valueChanged.connect(self._on_scrub)
        trow.addWidget(self.scrub, 1)
        self.time_lbl = QLabel("0.00 s")
        self.time_lbl.setMinimumWidth(90)
        trow.addWidget(self.time_lbl)
        left.addLayout(trow)

        # Replicate scope. Atlas-wide pooling is useful for broad diagnostics but
        # is deliberately labelled: the app's actual detection unit is one box.
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
        rrow.addWidget(QLabel("Focused view:"))
        self.focus_mode = QComboBox()
        self.focus_mode.addItem("Source detail", "source")
        self.focus_mode.addItem("Flow working resolution", "flow")
        self.focus_mode.setToolTip(
            "Source detail crops the original frame before scaling. Flow working "
            "resolution replays downsampling, grayscale conversion and contrast "
            "normalization at the exact cached flow-input dimensions. Stateful "
            "registration, temporal denoising/background and masks are omitted. "
            "Choose Raw frame below to inspect it without a flow overlay.")
        self.focus_mode.currentIndexChanged.connect(
            lambda: self._redraw_video())
        rrow.addWidget(self.focus_mode)
        left.addLayout(rrow)

        # overlay mode
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Overlay:"))
        self.overlay_mode = QComboBox()
        modes = ["Raw frame", "Speed heatmap"]
        if self.u is not None:
            modes += ["Flow direction (HSV)", "Flow vectors"]
        self.overlay_mode.addItems(modes)
        self.overlay_mode.setCurrentText("Speed heatmap")
        self.overlay_mode.currentTextChanged.connect(lambda _: self._redraw_video())
        orow.addWidget(self.overlay_mode, 1)
        self.hi_chk = QCheckBox("Highlight blocks > threshold")
        self.hi_chk.setChecked(True)
        self.hi_chk.stateChanged.connect(lambda _: self._redraw_video())
        orow.addWidget(self.hi_chk)
        left.addLayout(orow)

        # threshold slider
        thr_row = QHBoxLayout()
        self.thr_lbl = QLabel()
        self.thr_lbl.setMinimumWidth(190)
        thr_row.addWidget(self.thr_lbl)
        self.thr_slider = QSlider(Qt.Orientation.Horizontal)
        self.thr_slider.setRange(0, 1000)
        self.thr_slider.setValue(500)
        self.thr_slider.valueChanged.connect(self._on_threshold)
        self.thr_slider.sliderReleased.connect(self._recompute_clump)
        thr_row.addWidget(self.thr_slider, 1)
        left.addLayout(thr_row)

        # Detection minimum: values quieter than this floor are absent from all
        # detection-facing calculations. Raw distribution plots stay raw so the
        # user can still see the signal being rejected and choose the floor.
        min_row = QHBoxLayout()
        self.min_lbl = QLabel()
        self.min_lbl.setMinimumWidth(190)
        min_row.addWidget(self.min_lbl)
        self.min_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_slider.setRange(0, 1000)
        self.min_slider.setValue(0)
        self.min_slider.setToolTip(
            "Blocks below this speed are dropped from the windowed detection "
            "signal, threshold/clump detection, and video overlays. Raw "
            "distribution plots remain unchanged.")
        self.min_slider.valueChanged.connect(self._on_detection_min)
        self.min_slider.sliderReleased.connect(self._recompute_clump)
        min_row.addWidget(self.min_slider, 1)
        left.addLayout(min_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Temporal aggregation:"))
        self.temporal_mode_lbl = QLabel(
            "Trailing-window mean (same contract as structure tensor)")
        self.temporal_mode_lbl.setToolTip(
            "Each frame uses that frame and up to W-1 preceding frames. At the "
            "start of the clip, only the available history is averaged.")
        mode_row.addWidget(self.temporal_mode_lbl, 1)
        left.addLayout(mode_row)

        rw_row = QHBoxLayout()
        self.rw_lbl = QLabel()
        self.rw_lbl.setMinimumWidth(190)
        rw_row.addWidget(self.rw_lbl)
        self.rw_slider = QSlider(Qt.Orientation.Horizontal)
        self.rw_slider.setRange(2, max(3, self.T - 1))
        self.rw_slider.setValue(self.win_frames)
        self.rw_slider.valueChanged.connect(self._on_roll_win)
        self.rw_slider.sliderReleased.connect(self._on_roll_win_released)
        rw_row.addWidget(self.rw_slider, 1)
        left.addLayout(rw_row)

        info = QLabel(
            f"cache: {self.meta.get('backend', '?')} | fps {self.fps:.2f} | "
            f"block {self.block}px | downsample {self.meta.get('downsample', '?')} | "
            f"{len(self.regions)} {'replicate tiles' if self.packed else 'region'}")
        info.setStyleSheet("color:#888;")
        left.addWidget(info)

        root.addLayout(left, 3)

        # ---- right: scrollable readouts ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        self.plot_col = QGridLayout(holder)
        self.plot_col.setSpacing(3)
        self.plot_col.setColumnMinimumWidth(0, 20)
        self.plot_col.setColumnStretch(1, 1)
        scroll.setWidget(holder)
        scroll.setMinimumWidth(500)
        root.addWidget(scroll, 2)

        self.plots: dict[str, MiniPlot] = {}
        self.detect_checks: dict[str, QCheckBox] = {}
        self.detect_group = QButtonGroup(self)
        self.detect_group.setExclusive(True)
        self._plot_row = 0

        def section(text: str):
            lbl = QLabel(text)
            lbl.setStyleSheet("color:#7fd7ff; font-weight:bold; padding-top:6px;")
            self.plot_col.addWidget(lbl, self._plot_row, 0, 1, 2)
            self._plot_row += 1
            return lbl

        def add(key: str, title: str, unit: str, color=LINE,
                detect_target: str | None = None):
            pl = MiniPlot(title, unit, color)
            pl.seek_requested.connect(self._seek)
            self.plots[key] = pl
            self.plot_col.addWidget(pl, self._plot_row, 1)
            if detect_target is not None:
                check = QCheckBox()
                check.setAccessibleName(f"Detect on {title}")
                check.setToolTip(f"Use {title} as the per-block detection field")
                check.setProperty("detect_target", detect_target)
                self.detect_group.addButton(check)
                self.detect_checks[detect_target] = check
                check.setChecked(detect_target == self.detect_on)
                self.plot_col.addWidget(
                    check, self._plot_row, 0, Qt.AlignmentFlag.AlignCenter)
            self._plot_row += 1

        section("Raw selected-block speed distribution (per frame)")
        add("mean", "Mean speed", "px/s", detect_target="per_frame")
        add("median", "Median speed", "px/s", detect_target="scalar:median")
        add("p90", "90th pct speed", "px/s", detect_target="scalar:p90")
        add("p99", "99th pct speed", "px/s", detect_target="scalar:p99")
        add("max", "Max speed (single fastest block)", "px/s",
            detect_target="scalar:max")
        add("sstd", "Spatial std of speed", "px/s", detect_target="scalar:sstd")
        add("peak", "Max - median (peakedness)", "px/s",
            detect_target="scalar:peak")

        section("Trailing-window detection signal (minimum + window)")
        add("roll_mean", "Mean retained speed over trailing window", "px/s", LINE2,
            detect_target="windowed")
        add("roll_std", "Rolling std of retained-block speed", "px/s", LINE2,
            detect_target="scalar:roll_std")

        self.detection_section_lbl = section(
            "Detection sweep on selected channel (threshold slider)")
        add("count", "# blocks > threshold", "blk", QColor(110, 230, 120))
        add("frac", "Fraction of blocks > threshold", "", QColor(110, 230, 120))
        add("clump", "Largest connected clump > threshold", "blk",
            QColor(110, 230, 120))
        add("cond_mean", "Mean speed OF blocks > threshold", "px/s",
            QColor(110, 230, 120))
        add("energy", "Total speed summed over blocks > threshold", "px/s",
            QColor(110, 230, 120))
        self.plot_col.setRowStretch(self._plot_row, 1)
        self.detect_group.buttonToggled.connect(self._on_detect_plot_toggled)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        # Debounce for the cheap threshold series so dragging stays smooth on a
        # full-clip cache (the sum over ~100M elements is ~100 ms).
        self._thr_debounce = QTimer(self)
        self._thr_debounce.setSingleShot(True)
        self._thr_debounce.setInterval(120)
        self._thr_debounce.timeout.connect(self._recompute_threshold_series)

        self._min_debounce = QTimer(self)
        self._min_debounce.setSingleShot(True)
        self._min_debounce.setInterval(120)
        self._min_debounce.timeout.connect(self._recompute_temporal_series)

        self._sync_labels()

    # -- static (threshold-independent) series -------------------------------

    def _active_regions(self) -> list[dict]:
        if self.active_region_index < 0:
            return self.regions
        return [self.regions[self.active_region_index]]

    def _active_block_count(self) -> int:
        return sum((r["atlas_bbox"][2] - r["atlas_bbox"][0]) *
                   (r["atlas_bbox"][3] - r["atlas_bbox"][1])
                   for r in self._active_regions())

    def _active_block_values(self, arr: np.ndarray) -> np.ndarray:
        """(T, K) values from owned cells only; never atlas separators/padding."""
        parts = []
        for region in self._active_regions():
            y0, x0, y1, x1 = region["atlas_bbox"]
            parts.append(arr[:, y0:y1, x0:x1].reshape(arr.shape[0], -1))
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

    def _active_speed_scale(self) -> float:
        values = self._active_block_values(self.speed)
        finite = values[np.isfinite(values)]
        vmax = float(np.percentile(finite, 99.9)) if finite.size else 1.0
        return max(vmax, 1.0)

    def _scope_text(self) -> str:
        if self.active_region_index < 0:
            return f"all {len(self.regions)} replicates pooled"
        region = self.regions[self.active_region_index]
        rid = region["id"]
        return region["label"] + (f" (#{rid})" if rid is not None else "")

    def _sync_window_title(self) -> None:
        self.setWindowTitle(
            f"Speed explorer -- {os.path.basename(self.meta.get('video_path', '?'))} "
            f"-- {self._scope_text()} ({self.T} frames, {self.n_blocks} blocks)")

    def _compute_static_series(self):
        s = self._active_block_values(self.speed)
        self.plots["mean"].set_series(s.mean(1))
        self.plots["median"].set_series(np.median(s, 1))
        self.plots["p90"].set_series(np.percentile(s, 90, axis=1))
        self.plots["p99"].set_series(np.percentile(s, 99, axis=1))
        self.plots["max"].set_series(s.max(1))
        self.plots["sstd"].set_series(s.std(1))
        self.plots["peak"].set_series(s.max(1) - np.median(s, 1))
        self._recompute_temporal_series()

    def _detection_input_series(self) -> np.ndarray:
        """Spatial mean of the per-block field used as window input.

        Blocks below the detection minimum become zero before temporal averaging.
        Keeping the full block denominator makes this series exactly the spatial
        mean of the per-block windowed field that detection thresholds.
        """
        values = self._active_block_values(self.speed)
        return np.where(values >= self.detection_min, values, 0.0).mean(
            1, dtype=np.float64).astype(np.float32)

    def _window_bounds(self, W: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Structure-tensor-compatible trailing [lo, hi) window bounds."""
        hi = np.arange(self.T) + 1
        lo = np.maximum(0, hi - W)
        return hi, lo, (hi - lo).astype(np.float32)

    def _recompute_temporal_series(self):
        win = self._temporal_window_frames()
        x = self._detection_input_series()
        hi, lo, neff = self._window_bounds(win)
        cs = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
        cs2 = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
        rmean = (cs[hi] - cs[lo]) / neff
        rmsq = (cs2[hi] - cs2[lo]) / neff
        rstd = np.sqrt(np.maximum(rmsq - rmean ** 2, 0.0))
        self.plots["roll_mean"].set_series(rmean.astype(np.float32))
        self.plots["roll_std"].set_series(rstd.astype(np.float32))

    # -- threshold-dependent series ------------------------------------------

    def _temporal_window_frames(self) -> int:
        return max(1, int(self.win_frames))

    def _iter_windowed_fields(self, values: np.ndarray):
        """Yield the selected temporal field one frame at a time.

        Keeping only one block plane plus running sums avoids allocating another
        full ``(T, ny, nx)`` array beside the already-large speed cache.
        """
        win = self._temporal_window_frames()
        total = np.zeros(values.shape[1:], np.float64)

        def update(index: int, sign: int):
            sample = values[index]
            retained = sample >= self.detection_min
            if sign > 0:
                np.add(total, sample, out=total, where=retained)
            else:
                np.subtract(total, sample, out=total, where=retained)

        for frame in range(self.T):
            update(frame, 1)
            outgoing = frame - win
            if outgoing >= 0:
                update(outgoing, -1)
            field = np.zeros(values.shape[1:], np.float32)
            np.multiply(total, 1.0 / min(frame + 1, win), out=field,
                        casting="unsafe")
            yield field

    def _iter_detection_fields(self, values: np.ndarray):
        if self.detect_on == "windowed":
            yield from self._iter_windowed_fields(values)
            return
        for frame in values:
            yield np.where(frame >= self.detection_min, frame, 0.0)

    def _detection_field_at(self, frame: int, values: np.ndarray) -> np.ndarray:
        """Selected per-block detection field at one frame."""
        if self.detect_on != "windowed":
            current = values[frame]
            return np.where(current >= self.detection_min, current, 0.0)

        win = self._temporal_window_frames()
        lo = max(0, frame + 1 - win)
        indices = np.arange(lo, frame + 1)
        samples = values[indices]
        retained = samples >= self.detection_min
        total = np.where(retained, samples, 0.0).sum(0, dtype=np.float64)
        return (total / len(indices)).astype(np.float32)

    def _scalar_detection_key(self) -> str | None:
        prefix = "scalar:"
        return self.detect_on[len(prefix):] if self.detect_on.startswith(prefix) \
            else None

    def _scalar_detection_series(self) -> np.ndarray | None:
        key = self._scalar_detection_key()
        return self.plots[key].y if key in self.plots else None

    def _detection_scale(self) -> float:
        series = self._scalar_detection_series()
        if series is not None:
            finite = series[np.isfinite(series)]
            vmax = float(np.percentile(finite, 99.9)) if finite.size else 1.0
            return max(vmax, 1e-4)
        if self.detect_on == "windowed":
            return self.vmax
        return self.vmax

    def _detection_label(self) -> str:
        key = self._scalar_detection_key()
        if key in self.plots:
            return self.plots[key].title
        if self.detect_on != "windowed":
            return "per-frame speed"
        return "trailing-window mean speed"

    def _detection_unit(self) -> str:
        key = self._scalar_detection_key()
        if key in self.plots:
            return self.plots[key].unit
        return "px/s"

    def _sync_detection_plot_labels(self):
        label = self._detection_label()
        unit = self._detection_unit()
        scalar = self._scalar_detection_key() is not None
        self.detection_section_lbl.setText(
            "Scalar detection on selected plot (binary threshold)" if scalar
            else "Detection sweep on selected channel (threshold slider)")
        for key in ("frac", "clump", "cond_mean", "energy"):
            self.plots[key].setHidden(scalar)
        if scalar:
            self.plots["count"].title = f"{label} > threshold"
            self.plots["count"].unit = "0/1"
            self.plots["count"].update()
            return
        self.plots["count"].title = "# blocks > threshold"
        self.plots["count"].unit = "blk"
        self.plots["cond_mean"].title = \
            f"Mean {label} OF blocks > threshold"
        self.plots["cond_mean"].unit = unit
        self.plots["energy"].title = \
            f"Total {label} summed over blocks > threshold"
        self.plots["energy"].unit = unit
        self.plots["cond_mean"].update()
        self.plots["energy"].update()
        self.plots["count"].update()

    def _recompute_threshold_series(self):
        thr = self.threshold
        scalar_series = self._scalar_detection_series()
        if scalar_series is not None:
            detected = (scalar_series > thr).astype(np.float32)
            self.plots["count"].set_series(detected)
            self._sync_detection_plot_labels()
            return
        if self.detect_on == "per_frame":
            values = self._active_block_values(self.speed)
            mask = (values >= self.detection_min) & (values > thr)
            count = mask.sum(1).astype(np.float32)
            energy = np.where(mask, values, 0.0).sum(1, dtype=np.float64)
        else:
            count = np.zeros(self.T, np.float32)
            energy = np.zeros(self.T, np.float64)
            for region in self._active_regions():
                y0, x0, y1, x1 = region["atlas_bbox"]
                values = self.speed[:, y0:y1, x0:x1]
                for frame, field in enumerate(
                        self._iter_windowed_fields(values)):
                    mask = field > thr
                    count[frame] += float(mask.sum())
                    energy[frame] += float(field[mask].sum(dtype=np.float64))
        self.plots["count"].set_series(count)
        self.plots["frac"].set_series(count / max(1, self.n_blocks))

        self.plots["energy"].set_series(energy.astype(np.float32))
        with np.errstate(invalid="ignore", divide="ignore"):
            cond = np.where(count > 0, energy / np.maximum(count, 1), 0.0)
        self.plots["cond_mean"].set_series(cond.astype(np.float32))
        self._sync_detection_plot_labels()
        # Clump is the expensive one; keep whatever it last held during a drag and
        # refresh it on slider release.
        if "clump" not in self.plots or self.plots["clump"].y.size == 0:
            self._recompute_clump()

    def _recompute_clump(self):
        """Largest 8-connected component of the above-threshold mask, per frame.

        This is the signal the spatial min_blocks criterion gates on, so it is the
        most direct readout of 'is there a real moving CLUMP here, or just a few
        scattered noisy blocks that happen to exceed the cutoff'.
        """
        if self._scalar_detection_key() is not None:
            self.plots["clump"].set_series(np.zeros(self.T, np.float32))
            return
        thr = self.threshold
        largest = np.zeros(self.T, np.float32)
        # Valid-area weights so a one-pixel-tall edge sliver is not counted as a
        # full block -- mirrors the same discount in roi_detection.
        weight_plane = block_weight_plane(self.meta)
        # Components may never bridge two packed replicate tiles. This mirrors
        # roi_detection, which applies its spatial gate inside one replicate.
        for region in self._active_regions():
            y0, x0, y1, x1 = region["atlas_bbox"]
            w_flat = weight_plane[y0:y1, x0:x1].reshape(-1)
            values = self.speed[:, y0:y1, x0:x1]
            for t, field in enumerate(self._iter_detection_fields(values)):
                m = (field > thr).astype(np.uint8)
                if not m.any():
                    continue
                n_lab, labels, _, _ = cv2.connectedComponentsWithStats(
                    m, connectivity=8)
                if n_lab > 1:
                    areas = np.bincount(labels.reshape(-1), weights=w_flat,
                                        minlength=n_lab)
                    largest[t] = max(largest[t], float(areas[1:].max()))
        self.plots["clump"].set_series(largest)

    # -- control handlers ----------------------------------------------------

    def _sync_labels(self):
        self.thr_lbl.setText(
            f"Threshold ({self._detection_label()}): {self.threshold:.2f} "
            f"{self._detection_unit()}")
        self.min_lbl.setText(
            f"Detection minimum: {self.detection_min:.2f} px/s")
        self.rw_lbl.setText(
            f"Detection window W: {self.win_frames} fr "
            f"({self.win_frames / self.fps:.2f} s)")

    def _rescale_detection_threshold(self):
        self.thr_vmax = self._detection_scale()
        self.threshold = self.thr_slider.value() / 1000.0 * self.thr_vmax

    def _on_threshold(self, v: int):
        self.threshold = v / 1000.0 * self.thr_vmax
        self._sync_labels()
        self._thr_debounce.start()
        self._redraw_video()          # highlight overlay follows immediately

    def _on_roll_win(self, v: int):
        self.win_frames = max(2, int(v))
        self._recompute_temporal_series()
        if self.detect_on in ("windowed", "scalar:roll_std"):
            self._rescale_detection_threshold()
            self._thr_debounce.start()
        self._sync_labels()
        self._redraw_video()

    def _on_roll_win_released(self):
        if self.detect_on in ("windowed", "scalar:roll_std"):
            self._thr_debounce.stop()
            self._recompute_threshold_series()
            self._recompute_clump()

    def _on_detection_min(self, v: int):
        self.detection_min = v / 1000.0 * self.vmax
        self._sync_labels()
        self._min_debounce.start()
        self._thr_debounce.start()
        self._redraw_video()

    def _on_detect_plot_toggled(self, button, checked: bool):
        if not checked:
            return
        data = button.property("detect_target")
        self.detect_on = str(data) if data is not None else "per_frame"
        self._rescale_detection_threshold()
        self._sync_labels()
        self._recompute_threshold_series()
        self._recompute_clump()
        self._redraw_video()

    def _on_region_changed(self, _index: int):
        data = self.region_combo.currentData()
        self.active_region_index = int(data) if data is not None else 0
        focus = None if self.active_region_index < 0 else \
            self.regions[self.active_region_index]["frac"]
        self.video_view.set_focus_frac(focus)
        self._sync_video_boxes()
        self.n_blocks = self._active_block_count()
        # Scope changes select different cached cells; they do not alter the
        # absolute px/s cutoff or its slider mapping.
        self._compute_static_series()
        self._recompute_threshold_series()
        self._recompute_clump()
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

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start(int(1000 / max(1.0, self.fps)))
        else:
            self.timer.stop()

    def toggle_playback(self):
        """Public playback hook shared with VideoPanel/MainWindow dispatch."""
        self._toggle_play()

    def eventFilter(self, watched, event):
        event_type = event.type()
        if event_type in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and \
                event.key() == Qt.Key.Key_Shift and not event.isAutoRepeat():
            focus = QApplication.focusWidget()
            focus_is_ours = focus is self or (
                focus is not None and self.isAncestorOf(focus))
            if event_type == QEvent.Type.KeyPress and focus_is_ours and \
                    not self._overlay_peek_hidden:
                self._overlay_peek_hidden = True
                self.video_view.set_overlays_hidden(True)
                self._redraw_video()
            elif event_type == QEvent.Type.KeyRelease and \
                    self._overlay_peek_hidden:
                self._overlay_peek_hidden = False
                self.video_view.set_overlays_hidden(False)
                self._redraw_video()
        elif event_type == QEvent.Type.ApplicationDeactivate and \
                self._overlay_peek_hidden:
            # A Shift release delivered to another application must not leave the
            # explorer permanently stuck in its temporary raw-video state.
            self._overlay_peek_hidden = False
            self.video_view.set_overlays_hidden(False)
            self._redraw_video()
        return super().eventFilter(watched, event)

    def _tick(self):
        nxt = self.frame + 1
        if nxt >= self.T:
            nxt = 0
        self._update_frame(nxt)

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
        for pl in self.plots.values():
            pl.set_cursor(self.frame)
        if self.isVisible():
            self._redraw_video()

    # -- video + overlay -----------------------------------------------------

    def _base_frame(self) -> np.ndarray:
        """Source-detail or flow-resolution pixels for the current view."""
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

        view_x0, view_y0, view_x1, view_y1 = self._render_frac
        source_view_w = max(1, int(round(self.src_w * (view_x1 - view_x0))))
        source_view_h = max(1, int(round(self.src_h * (view_y1 - view_y0))))
        if bgr is None:
            if focus is not None and self.focus_mode.currentData() == "flow":
                region = self.regions[self.active_region_index]
                work_w = int(region.get(
                    "work_width", region["grid"][1] * self.block))
                work_h = int(region.get(
                    "work_height", region["grid"][0] * self.block))
                return np.zeros((max(1, work_h), max(1, work_w), 3), np.uint8)
            display_w = min(source_view_w, DISPLAY_MAX_W)
            display_h = max(1, int(round(source_view_h * display_w /
                                           source_view_w)))
            return np.zeros((display_h, display_w, 3), np.uint8)

        if focus is not None and self.focus_mode.currentData() == "flow":
            from core.config import PipelineConfig
            from core.preprocess import flow_input_preview
            region = self.regions[self.active_region_index]
            work_w = int(region.get("work_width",
                                    region["grid"][1] * self.block))
            work_h = int(region.get("work_height",
                                    region["grid"][0] * self.block))
            cfg = PipelineConfig.from_dict(
                self.meta.get("config", {})).preprocess
            return flow_input_preview(bgr, (work_w, work_h), cfg)

        h, w = bgr.shape[:2]
        if w > DISPLAY_MAX_W:
            scale = DISPLAY_MAX_W / w
            bgr = cv2.resize(bgr, (DISPLAY_MAX_W, max(1, int(round(h * scale)))),
                             interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(bgr)

    def _display_bbox(self, region: dict, width: int, height: int
                      ) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = region["frac"]
        vx0, vy0, vx1, vy1 = self._render_frac
        x0 = (x0 - vx0) / (vx1 - vx0)
        x1 = (x1 - vx0) / (vx1 - vx0)
        y0 = (y0 - vy0) / (vy1 - vy0)
        y1 = (y1 - vy0) / (vy1 - vy0)
        dx0 = max(0, min(width - 1, int(round(x0 * width))))
        dy0 = max(0, min(height - 1, int(round(y0 * height))))
        dx1 = max(dx0 + 1, min(width, int(round(x1 * width))))
        dy1 = max(dy0 + 1, min(height, int(round(y1 * height))))
        return dx0, dy0, dx1, dy1

    def _redraw_video(self):
        base = self._base_frame()
        ch, cw = base.shape[:2]
        sp = self.speed[self.frame]                       # (ny, nx)
        mode = "Raw frame" if self._overlay_peek_hidden else \
            self.overlay_mode.currentText()

        out = base.copy()
        active_indices = (range(len(self.regions)) if self.active_region_index < 0
                          else [self.active_region_index])
        active_indices = list(active_indices)
        for region_index in active_indices:
            region = self.regions[region_index]
            y0, x0, y1, x1 = region["atlas_bbox"]
            sub_speed = sp[y0:y1, x0:x1]
            dx0, dy0, dx1, dy1 = self._display_bbox(region, cw, ch)
            rw, rh = dx1 - dx0, dy1 - dy0
            roi = out[dy0:dy1, dx0:dx1]

            if mode == "Speed heatmap":
                norm = np.clip(sub_speed / self.vmax, 0, 1)
                heat = cv2.applyColorMap(
                    (norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                heat = cv2.resize(heat, (rw, rh), interpolation=cv2.INTER_NEAREST)
                blended = cv2.addWeighted(roi, 0.45, heat, 0.55, 0)
                retained = cv2.resize(
                    (sub_speed >= self.detection_min).astype(np.uint8),
                    (rw, rh), interpolation=cv2.INTER_NEAREST)
                np.copyto(roi, blended, where=(retained > 0)[:, :, None])
            elif mode == "Flow direction (HSV)" and self.u is not None:
                u = self.u[self.frame, y0:y1, x0:x1]
                v = self.v[self.frame, y0:y1, x0:x1]
                ang = (np.degrees(np.arctan2(v, u)) % 360) / 2.0
                mag = np.clip(sub_speed / self.vmax, 0, 1)
                hsv = np.zeros((*sub_speed.shape, 3), np.uint8)
                hsv[..., 0] = ang.astype(np.uint8)
                hsv[..., 1] = 255
                hsv[..., 2] = (mag * 255).astype(np.uint8)
                direction = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                direction = cv2.resize(
                    direction, (rw, rh), interpolation=cv2.INTER_NEAREST)
                blended = cv2.addWeighted(roi, 0.4, direction, 0.6, 0)
                retained = cv2.resize(
                    (sub_speed >= self.detection_min).astype(np.uint8),
                    (rw, rh), interpolation=cv2.INTER_NEAREST)
                np.copyto(roi, blended, where=(retained > 0)[:, :, None])
            elif mode == "Flow vectors" and self.u is not None:
                u = self.u[self.frame, y0:y1, x0:x1]
                v = self.v[self.frame, y0:y1, x0:x1]
                gy, gx = sub_speed.shape
                cell_w, cell_h = rw / gx, rh / gy
                vector_scale = min(cell_w, cell_h) / max(self.vmax, 1e-6) * 1.5
                for by in range(0, gy, 2):
                    for bx in range(0, gx, 2):
                        if sub_speed[by, bx] < self.detection_min:
                            continue
                        ax0 = int(dx0 + (bx + 0.5) * cell_w)
                        ay0 = int(dy0 + (by + 0.5) * cell_h)
                        ax1 = int(ax0 + u[by, bx] * vector_scale)
                        ay1 = int(ay0 + v[by, bx] * vector_scale)
                        cv2.arrowedLine(out, (ax0, ay0), (ax1, ay1),
                                        (80, 220, 255), 1, tipLength=0.35)

        if self.hi_chk.isChecked() and not self._overlay_peek_hidden:
            scalar_series = self._scalar_detection_series()
            scalar_detected = scalar_series is not None and \
                bool(scalar_series[self.frame] > self.threshold)
            for region_index in active_indices:
                region = self.regions[region_index]
                y0, x0, y1, x1 = region["atlas_bbox"]
                dx0, dy0, dx1, dy1 = self._display_bbox(region, cw, ch)
                if scalar_series is not None:
                    m = np.full((y1 - y0, x1 - x0), scalar_detected,
                                dtype=np.uint8)
                else:
                    values = self.speed[:, y0:y1, x0:x1]
                    field = self._detection_field_at(self.frame, values)
                    m = (field > self.threshold).astype(np.uint8)
                mm = cv2.resize(m, (dx1 - dx0, dy1 - dy0),
                                interpolation=cv2.INTER_NEAREST)
                roi = out[dy0:dy1, dx0:dx1]
                green = np.zeros_like(roi)
                green[..., 1] = 255
                blended = cv2.addWeighted(roi, 0.5, green, 0.5, 0)
                np.copyto(roi, blended, where=(mm > 0)[:, :, None])
                contours, _ = cv2.findContours(
                    mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(roi, contours, -1, (60, 255, 60), 1)

        self.video_view.set_frame(
            out, image_frac=self._render_frac,
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
