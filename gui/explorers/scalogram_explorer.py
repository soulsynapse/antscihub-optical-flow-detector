"""Scalogram explorer: the expanded-cache proposal in the explorer idiom.

Same contract as the speed / coherent-flow / structure-tensor explorers -- video
with a channel overlay on the left, custom-painted detection plots on the right,
per-replicate, cache-backed. It exists to make ONE comparison legible:

  * the structure-tensor explorer detects on a single fixed frequency band (the
    cache stores one band power); its detection knob is a value band on that
    channel.
  * this explorer shows the whole Morlet scalogram (frequency x time) of the same
    channel and puts a *frequency* band on it. You place the band where the
    rhythm actually is, then read the per-block power that band yields and its
    value band -- the detection channel the cache WOULD store.

The scalogram IS the "first integration" here (the structure-tensor explorer's
windowed mean over W): instead of a window length, the knob is the frequency band
on the scalogram. So the "detect on which channel" picker -- which structure
tensor puts as exclusive checkboxes beside its density plots -- lives on the RIGHT
here too, an exclusive checkbox group over the source channels feeding the
scalogram.

Two stacked bands, mirroring the two questions: the scalogram's frequency band
("which rhythm") and the density plot's value band ("which blocks are hot").
Nothing here is written to the cache; scalograms are derived on the fly from the
structure-tensor channels so the cost/benefit is visible before it is paid.
See docs/expanded_cache_plan.md.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import cv2
import numpy as np

from PyQt6.QtCore import QEvent, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QImage, QKeySequence, QPainter, QPen,
                         QShortcut)
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                             QHBoxLayout, QLabel, QProgressDialog, QPushButton,
                             QScrollArea, QSlider, QVBoxLayout, QWidget)

from core.tensor_channels import load_or_extract_channels
from core.wavelet import default_freqs, morlet_power
from gui.video_panel import FrameView
from gui.explorers.speed_explorer import (BG, CURSOR, DISPLAY_MAX_W, PLOT_BG,
                                          TXT, TXT_DIM, DensityPlot, MiniPlot,
                                          PixelBarPlot, _regions_from_meta)

EPS = 1e-6

# Channel -> (attribute, human label, overlay colormap). All are nonnegative
# energies except cached speed; the scalogram cares only about their temporal
# fluctuation, so the choice is about which signal's rhythm you want to see.
CHANNELS = {
    "change energy Jtt": ("change", cv2.COLORMAP_TURBO),
    "appearance energy": ("appearance", cv2.COLORMAP_TURBO),
    "tensor speed": ("tensor_speed", cv2.COLORMAP_TURBO),
    "intensity": ("intensity", cv2.COLORMAP_TURBO),
    "cached flow speed": ("speed", cv2.COLORMAP_TURBO),
}

# Warm scalogram ramp (0 -> plot bg, up to hot white), distinct from the cyan
# density ramp so the two heatmaps never read as the same instrument.
_SG_RAMP = np.array([[12, 12, 12], [70, 25, 30], [150, 55, 40],
                     [240, 140, 40], [255, 240, 210]], np.float64)


class ScalogramPlot(MiniPlot):
    """Frequency x time heatmap of one signal, with a draggable FREQUENCY band.

    Reuses MiniPlot's band machinery wholesale -- the two 1px handles, the shaded
    rejected strips, the pull-off-the-edge -> unbounded rule -- but rides them on
    a log-frequency Y axis instead of a value axis. The band it returns is a
    frequency range in Hz: exactly the "which band would the cache store" knob.
    """

    BASE_H = MiniPlot.EXPANDED_H

    def __init__(self, title: str, freqs: np.ndarray):
        super().__init__(title, unit="Hz")
        self.freqs = np.asarray(freqs, np.float64)
        self.matrix = np.zeros((0, 0), np.float32)     # (F, T) power
        self._img = None
        self._img_key = None
        self._ver = 0
        self.band_active = True
        self.setMinimumHeight(self.BASE_H)
        self.setMaximumHeight(self.BASE_H)

    # Log-frequency axis: the band handles and heatmap rows share it.
    def _fwd(self, v):
        return np.log10(np.maximum(v, 1e-6))

    def _inv(self, t):
        return np.power(10.0, t)

    def _data_range(self):
        return float(self.freqs[0]), float(self.freqs[-1])

    def set_scalogram(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, np.float32)   # (F, T)
        self.y = self.matrix.sum(axis=0) if self.matrix.size \
            else np.zeros(0, np.float32)               # cursor readout: total power
        self._img = None
        self._ver += 1
        self.update()

    def band_hz(self) -> tuple[float, float]:
        lo, hi = self.band()
        flo, fhi = float(self.freqs[0]), float(self.freqs[-1])
        return (flo if lo == float("-inf") else max(flo, lo),
                fhi if hi == float("inf") else min(fhi, hi))

    def _heatmap(self, w, h, lo, hi):
        if w <= 0 or h <= 0 or self.matrix.size == 0:
            return None
        key = (w, h, self._ver)
        if self._img is not None and self._img_key == key:
            return self._img
        F, T = self.matrix.shape
        # Rows: map each display row to a log-frequency, nearest scale. Top row is
        # the highest frequency, matching the band handles' orientation.
        tlo, thi = self._fwd(lo), self._fwd(hi)
        row_f = self._inv(thi - (np.arange(h) + 0.5) / h * (thi - tlo))
        fidx = np.clip(np.searchsorted(self.freqs, row_f), 0, F - 1)
        col = np.clip((np.arange(w) * T) // max(1, T), 0, T - 1)
        cells = np.log10(self.matrix[np.ix_(fidx, col)] + EPS)
        cmin, cmax = float(cells.min()), float(cells.max())
        norm = (cells - cmin) / max(1e-9, cmax - cmin)
        x = np.clip(norm, 0, 1) * (len(_SG_RAMP) - 1)
        i = np.clip(x.astype(int), 0, len(_SG_RAMP) - 2)
        f = (x - i)[..., None]
        rgb = np.ascontiguousarray(
            (_SG_RAMP[i] * (1 - f) + _SG_RAMP[i + 1] * f).astype(np.uint8))
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._img, self._img_key = img, key
        return img

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), BG)
        r = self._plot_rect()
        p.fillRect(r, PLOT_BG)
        p.setFont(QFont("Consolas", 7))
        if self.matrix.size == 0:
            p.setPen(TXT_DIM)
            p.drawText(8, 12, self.title)
            p.end()
            return
        lo, hi = self._data_range()
        img = self._heatmap(int(r.width()), int(r.height()), lo, hi)
        if img is not None:
            p.drawImage(r.topLeft(), img)
        if self.band_active:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self._paint_band(p, r, lo, hi)
        n = self.y.size
        cx = r.left() + (self.cursor + 0.5) / max(1, n) * r.width()
        p.setPen(QPen(CURSOR, 1))
        p.drawLine(int(cx), int(r.top()), int(cx), int(r.bottom()))
        flo, fhi = self.band_hz()
        p.setPen(TXT)
        p.drawText(8, 12, f"{self.title}   band {flo:.2f}-{fhi:.2f} Hz")
        p.setPen(TXT_DIM)
        p.drawText(int(r.left()), int(r.bottom()) + 8, f"{lo:.3g} Hz")
        p.drawText(int(r.left()), int(r.top()) + 4, f"{hi:.3g} Hz")
        p.end()


class _ScalogramWorker(QThread):
    """Builds one per-block Morlet cube off the GUI thread.

    The cube is a (F, T, B) float32 array that can run to several GB and tens of
    seconds for a full-length clip; computing it in ``run`` keeps the window
    responsive. ``blocks`` is a private contiguous copy, so the main thread is
    free to keep reading the channel arrays while this runs.
    """
    done = pyqtSignal(object, object)      # (key, cube) -- queued to GUI thread

    def __init__(self, key, blocks, fps, freqs, parent=None):
        super().__init__(parent)
        self._key = key
        self._blocks = blocks
        self._fps = fps
        self._freqs = freqs

    def run(self):
        cube = morlet_power(self._blocks, self._fps, self._freqs)  # (F, T, B)
        self.done.emit(self._key, cube)


class ScalogramExplorer(QWidget):
    # Total bytes of retained per-block cubes. ~6 GB keeps several test-clip
    # cubes (610 MB each) but only the current one for a multi-minute clip.
    _SG_CACHE_BUDGET = 6 * 1024 ** 3

    def __init__(self, cache=None, video_path: str | None = None, *,
                 state=None, sidecar_path: str | None = None, parent=None):
        super().__init__(parent)
        if state is not None:
            cache = state.cache
        if cache is None:
            raise ValueError("ScalogramExplorer requires an open cache")
        self.state = state
        self.cache = cache
        self.meta = cache.meta
        self.fps = float(self.meta["fps"])
        self.ny, self.nx = map(int, self.meta["grid"])
        self.regions = _regions_from_meta(self.meta, (self.ny, self.nx))
        self.packed = bool(self.meta.get("replicate_tiles"))
        # A pooled scalogram over EVERY replicate's blocks would be a multi-GB
        # (F, T, B) cube, so there is no "all replicates" processing mode here.
        # Instead index < 0 is a *selection* view: the full frame with every
        # replicate boxed and nothing computed, purely so a replicate can be
        # clicked to focus (and only then processed). With multiple replicates
        # that selection view is the default; a lone replicate is auto-selected.
        self.active_region_index = -1 if len(self.regions) > 1 else 0
        self.src_w = max(1, int(self.meta.get("src_width", 0)) or
                         int(self.meta.get("work_width", 1)))
        self.src_h = max(1, int(self.meta.get("src_height", 0)) or
                         int(self.meta.get("work_height", 1)))

        dlg = QProgressDialog("Extracting structure-tensor channels...", None,
                              0, int(self.meta["n_frames"]), self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)

        def prog(done, total):
            dlg.setMaximum(total)
            dlg.setValue(done)
            QApplication.processEvents()

        ch = load_or_extract_channels(cache, sidecar_path=sidecar_path, progress=prog)
        dlg.close()
        self._chan = {
            "change": np.asarray(ch["change"], np.float32),
            "appearance": np.asarray(ch["appearance"], np.float32),
            "tensor_speed": np.asarray(ch["tensor_speed"], np.float32),
            "intensity": np.asarray(ch["intensity"], np.float32),
            "speed": np.asarray(cache.read("speed"), np.float32),
        }
        self.T = self._chan["change"].shape[0]
        self.freqs = default_freqs(self.fps)
        self.channel = "change energy Jtt"
        self.frame = int(state.current_frame) if state is not None else 0
        self.playing = False
        self._overlay_peek_hidden = False
        self._render_frac = (0.0, 0.0, 1.0, 1.0)
        self._ov_scale = EPS       # replaced by a real percentile in _rebuild
        # Per-block scalogram of the active scope: (F, T, B). Built off the GUI
        # thread on region/channel change; band drags only re-sum it.
        self._block_sg = None
        # Per-region column bookkeeping into the cube's B axis, so a pooled scope
        # can map each block column back to the replicate (and grid cell) it came
        # from. Set by _apply_cube; a list of dicts {bbox, gy, gx, c0, n}.
        self._block_snap: list[dict] = []
        # Memoize built cubes by (region_index, channel) so switching back is
        # instant. Cubes are large, so retain them under a byte budget rather
        # than by count: a full-length clip may hold only the current one, a
        # short clip many -- degrades correctly instead of running out of RAM.
        self._sg_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._worker = None

        self.source = None
        self._owns_source = False
        if state is None and video_path and os.path.exists(video_path):
            from core.video import VideoSource
            try:
                self.source = VideoSource(video_path)
                self._owns_source = True
            except Exception:
                self.source = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._freq_debounce = QTimer(self)
        self._freq_debounce.setSingleShot(True)
        self._freq_debounce.setInterval(150)
        self._freq_debounce.timeout.connect(self._on_freq_band)
        self._build_ui()

        # Shift-to-peek (hide the overlay to read the raw frame) rides an
        # application-wide event filter, exactly as the structure-tensor
        # explorer does; Space toggles playback in the standalone case (when
        # embedded, the host window owns Space and drives us via frame_changed).
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

        self._sync_video_boxes()
        self._sync_window_title()
        self._rebuild_scalograms()
        # The per-block cube builds in the background; draw the video + channel
        # overlay now so the panel isn't blank until it lands (the cube-ready
        # path redraws again to add the in-band highlight).
        self._redraw_video()

    @classmethod
    def from_app_state(cls, state, parent=None) -> "ScalogramExplorer":
        if not state.has_cache:
            raise ValueError("Open a feature cache before creating this explorer")
        return cls(state=state, parent=parent)

    # -- scope / geometry ----------------------------------------------------
    def _regions_for_index(self, idx: int) -> list[dict]:
        """The regions a scope index covers: all of them when pooled (<0)."""
        return self.regions if idx < 0 else [self.regions[idx]]

    def _active_regions(self) -> list[dict]:
        return self._regions_for_index(self.active_region_index)

    def _chan_arr(self):
        return self._chan[CHANNELS[self.channel][0]]

    def _scope_blocks(self, arr: np.ndarray, idx: int) -> np.ndarray:
        """(T, B) block columns of ``arr`` over the regions of scope ``idx``,
        concatenated in region order -- the column order the cube's B axis and
        ``_block_snap`` both follow."""
        parts = [arr[:, y0:y1, x0:x1].reshape(self.T, -1)
                 for (y0, x0, y1, x1) in
                 (r["atlas_bbox"] for r in self._regions_for_index(idx))]
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

    # -- UI ------------------------------------------------------------------
    def _build_ui(self):
        self.resize(1600, 950)
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        self.video_view = FrameView()
        self.video_view.setMinimumSize(720, 480)
        self.video_view.clicked.connect(self._on_video_clicked)
        self.video_view.back_requested.connect(self._clear_region_focus)
        left.addWidget(self.video_view, 1)

        trow = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        trow.addWidget(self.play_btn)
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.setRange(0, self.T - 1)
        self.scrub.setValue(self.frame)
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
                "— all replicates (click one to select) —", -1)
        for i, r in enumerate(self.regions):
            rid = r["id"]
            suffix = f" (#{rid})" if rid is not None else ""
            self.region_combo.addItem(f"{r['label']}{suffix}", i)
        idx = self.region_combo.findData(self.active_region_index)
        self.region_combo.setCurrentIndex(max(0, idx))
        self.region_combo.currentIndexChanged.connect(self._on_region_changed)
        rrow.addWidget(self.region_combo, 1)
        left.addLayout(rrow)

        hrow = QHBoxLayout()
        self.hi_chk = QCheckBox("Highlight blocks in band")
        self.hi_chk.setChecked(True)
        self.hi_chk.stateChanged.connect(lambda _: self._redraw_video())
        hrow.addWidget(self.hi_chk)
        hrow.addStretch(1)
        left.addLayout(hrow)

        note = QLabel(
            "Drag the FREQUENCY band on the scalogram to pick which rhythm to "
            "detect on; the density plot below shows the per-block power that "
            "band yields, with its own VALUE band -- the detection channel a "
            "scalogram cache would store. Compare with the structure-tensor "
            "explorer, whose single fixed band cannot be moved in frequency. "
            "Click a replicate in the video to select it (nothing is computed "
            "until you do). Space toggles playback; hold Shift to peek at the "
            "raw frame.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8ab; padding-top:4px;")
        left.addWidget(note)

        info = QLabel(f"cache: {self.meta.get('backend','?')} | fps "
                      f"{self.fps:.2f} | {self.T} frames | scalogram "
                      f"{self.freqs[0]:.2f}-{self.freqs[-1]:.2f} Hz, "
                      f"{len(self.freqs)} scales")
        info.setStyleSheet("color:#888;")
        left.addWidget(info)
        self._busy_lbl = QLabel("")
        self._busy_lbl.setStyleSheet("color:#e0a94a;")
        left.addWidget(self._busy_lbl)
        root.addLayout(left, 3)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        col = QVBoxLayout(holder)
        col.setSpacing(4)

        # Detection-channel picker, on the RIGHT beside the plots (the structure
        # -tensor explorer's idiom): an exclusive checkbox group over the source
        # channels, selecting which signal the scalogram / density stack reads.
        sec = QLabel("Detect on channel")
        sec.setStyleSheet("color:#7fd7ff; font-weight:bold; padding-top:2px;")
        col.addWidget(sec)
        self.chan_group = QButtonGroup(self)
        self.chan_group.setExclusive(True)
        self.chan_checks: dict[str, QCheckBox] = {}
        for name in CHANNELS:
            cb = QCheckBox(name)
            cb.setChecked(name == self.channel)
            cb.setToolTip(f"Compute the scalogram on '{name}' and detect on it")
            self.chan_group.addButton(cb)
            self.chan_checks[name] = cb
            col.addWidget(cb)
        self.chan_group.buttonToggled.connect(self._on_channel_toggled)

        self.trace_plot = MiniPlot("selected channel (replicate mean)")
        self.trace_plot.seek_requested.connect(self._seek)
        col.addWidget(self.trace_plot)

        self.scalo_plot = ScalogramPlot("scalogram (drag frequency band)",
                                        self.freqs)
        self.scalo_plot.seek_requested.connect(self._seek)
        # band_changed fires on every mouse-move; the band-power re-sum is an
        # O(F*T*B) pass, so debounce it (same pattern as the speed explorer's
        # sweep debounce) and only recompute eagerly on release.
        self.scalo_plot.band_changed.connect(self._freq_debounce.start)
        self.scalo_plot.band_committed.connect(self._on_freq_band_committed)
        col.addWidget(self.scalo_plot)

        self.density_plot = DensityPlot("per-block band power (drag value band)")
        self.density_plot.set_log_axis(True)
        self.density_plot.set_band_active(True)
        self.density_plot.seek_requested.connect(self._seek)
        self.density_plot.band_changed.connect(self._redraw_video)
        self.density_plot.band_committed.connect(self._update_count)
        col.addWidget(self.density_plot)

        self.count_plot = PixelBarPlot("# blocks in band", unit="blocks",
                                       color=QColor(110, 230, 120))
        self.count_plot.seek_requested.connect(self._seek)
        col.addWidget(self.count_plot)

        col.addStretch(1)
        scroll.setWidget(holder)
        scroll.setMinimumWidth(460)
        root.addWidget(scroll, 2)

    # -- scalogram computation ----------------------------------------------
    def _rebuild_scalograms(self):
        """Refresh the cheap per-frame views synchronously, then serve the
        per-block cube from the memo cache or a background build.

        Everything here is <10 ms (the pooled-mean scalogram is one column), so
        it runs on the GUI thread for instant feedback. The heavy per-block cube
        is handed to :meth:`_request_cube`, which threads it and caches it."""
        if self.active_region_index < 0:
            # Selection view: nothing selected, so process nothing. Clear the
            # plots (a prior selection's data must not linger) and leave the
            # cube state empty; the video shows the full frame with boxes.
            self._block_sg = None
            self._block_snap = []
            self.trace_plot.set_series(np.zeros(0, np.float32))
            self.scalo_plot.set_scalogram(np.zeros((0, 0), np.float32))
            self.density_plot.set_matrix(np.zeros((0, 0), np.float32))
            self.count_plot.set_series(np.zeros(0, np.float32))
            self._set_busy(False)
            return
        arr = self._chan_arr()
        blocks = self._scope_blocks(arr, self.active_region_index)   # (T, B)
        pooled = blocks.mean(axis=1)
        self.trace_plot.set_series(pooled)
        self.scalo_plot.set_scalogram(morlet_power(pooled, self.fps, self.freqs))
        # Overlay color scale is a whole-clip percentile of this channel/scope;
        # freeze it here so scrubbing (which redraws the video every frame) never
        # re-percentiles the array in the hot path.
        self._ov_scale = max(float(np.percentile(blocks, 99)), EPS)

        key = (self.active_region_index, self.channel)
        cube = self._sg_cache.get(key)
        if cube is not None:                       # memo hit: apply immediately
            self._sg_cache.move_to_end(key)
            self._apply_cube(key, cube)
            self._set_busy(False)
            return

        # Cube not ready: drop the stale per-block state (a pending scrub must
        # not pair the new scope's blocks with the old scope's coords) and build.
        self._block_sg = None
        self._block_snap = []
        self._request_cube()

    def _request_cube(self):
        """Start a background build for the current view unless one is already
        in flight -- the running worker re-checks the current view when it ends,
        so a rapid region/channel switch coalesces onto the latest target."""
        if self.active_region_index < 0 or self._worker is not None:
            return
        self._launch_worker((self.active_region_index, self.channel))

    def _launch_worker(self, key):
        region_idx, channel = key
        arr = self._chan[CHANNELS[channel][0]]
        # A private contiguous copy: the worker reads it while the GUI thread
        # keeps using self._chan freely.
        blocks = np.ascontiguousarray(self._scope_blocks(arr, region_idx))
        self._set_busy(True)
        self._worker = _ScalogramWorker(key, blocks, self.fps, self.freqs, self)
        self._worker.done.connect(self._on_cube_ready)
        # Free the finished thread (and its private (T, B) blocks copy) instead
        # of letting it linger as a parented child until the window closes --
        # otherwise every channel/scope switch leaks one dead thread.
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_cube_ready(self, key, cube):
        self._sg_cache[key] = cube
        self._sg_cache.move_to_end(key)
        self._worker = None
        cur = (self.active_region_index, self.channel)
        self._evict_cache(protect=cur)
        if key == cur:                             # still the view being shown
            self._apply_cube(key, cube)
            self._set_busy(False)
        elif cur[0] < 0:                           # view moved to the selection
            self._set_busy(False)                  # (no-processing) scope
        elif self._sg_cache.get(cur) is not None:  # view moved to a cached cube
            self._sg_cache.move_to_end(cur)
            self._apply_cube(cur, self._sg_cache[cur])
            self._set_busy(False)
        else:                                      # view moved on: build that one
            self._launch_worker(cur)

    def _apply_cube(self, key, cube):
        """Bind ``cube`` as the active per-block scalogram and rebuild the
        per-region column map. Only ever called with a cube whose scope matches
        the current view, so the block coords line up with the cube's B axis."""
        region_idx, _ = key
        self._block_sg = cube
        snap: list[dict] = []
        c0 = 0
        for region in self._regions_for_index(region_idx):
            y0, x0, y1, x1 = region["atlas_bbox"]
            gy, gx = np.mgrid[y0:y1, x0:x1]
            n = (y1 - y0) * (x1 - x0)
            snap.append({"bbox": (y0, x0, y1, x1), "gy": gy.ravel(),
                         "gx": gx.ravel(), "c0": c0, "n": n, "region": region})
            c0 += n
        self._block_snap = snap
        self._on_freq_band()

    def _evict_cache(self, protect):
        """Drop oldest cubes until the retained total is under budget, but never
        the ``protect`` (current) key."""
        total = sum(c.nbytes for c in self._sg_cache.values())
        for k in list(self._sg_cache):             # oldest first
            if total <= self._SG_CACHE_BUDGET:
                break
            if k == protect:
                continue
            total -= self._sg_cache.pop(k).nbytes

    def _set_busy(self, busy):
        self._busy_lbl.setText("building scalogram…" if busy else "")

    def closeEvent(self, e):
        if self._event_filter_app is not None:
            self._event_filter_app.removeEventFilter(self)
            self._event_filter_app = None
        # Don't let Qt tear down a running QThread (crashes); wait it out.
        if self._worker is not None:
            self._worker.wait()
            self._worker = None
        if self._owns_source and self.source is not None:
            self.source.release()
            self.source = None
        super().closeEvent(e)

    def _on_freq_band(self):
        """Sum the per-block scalogram over the selected frequency band -> the
        per-block band-power detection channel."""
        if self._block_sg is None:
            return
        flo, fhi = self.scalo_plot.band_hz()
        # The band is a contiguous interval on the sorted frequency axis, so a
        # plain slice sums in place; a boolean mask would fancy-index a copy of
        # the whole (F, T, B) block first.
        i = int(np.searchsorted(self.freqs, flo, "left"))
        j = int(np.searchsorted(self.freqs, fhi, "right"))
        if j <= i:                              # empty band: snap to nearest scale
            i = int(np.argmin(np.abs(self.freqs - flo)))
            j = i + 1
        band_power = self._block_sg[i:j].sum(axis=0)           # (T, B)
        self.density_plot.set_matrix(band_power)
        self._update_count()

    def _on_freq_band_committed(self):
        self._freq_debounce.stop()
        self._on_freq_band()

    def _update_count(self, *_):
        if self._block_sg is None:
            return
        m = self.density_plot.matrix                            # (T, B)
        lo, hi = self.density_plot.band()
        inband = (m >= lo) & (m <= hi) & np.isfinite(m)
        self.count_plot.set_series(inband.sum(axis=1).astype(np.float32))
        self.count_plot.set_cursor(self.frame)
        self._redraw_video()

    # -- video overlay -------------------------------------------------------
    def _base_frame(self):
        focus = None if self.active_region_index < 0 else \
            self._active_regions()[0]["frac"]
        self._render_frac = focus or (0.0, 0.0, 1.0, 1.0)
        bgr = None
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
        vx0, vy0, vx1, vy1 = self._render_frac
        view_w = max(1, int(round(self.src_w * (vx1 - vx0))))
        view_h = max(1, int(round(self.src_h * (vy1 - vy0))))
        if bgr is None:
            dw = min(view_w, DISPLAY_MAX_W)
            dh = max(1, int(round(view_h * dw / max(1, view_w))))
            return np.zeros((dh, dw, 3), np.uint8)
        h, w = bgr.shape[:2]
        if w > DISPLAY_MAX_W:
            s = DISPLAY_MAX_W / w
            bgr = cv2.resize(bgr, (DISPLAY_MAX_W, max(1, int(round(h * s)))),
                             interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(bgr)

    def _display_bbox(self, region, width, height):
        """Where a replicate's full-frame frac lands in the rendered view."""
        x0, y0, x1, y1 = region["frac"]
        vx0, vy0, vx1, vy1 = self._render_frac
        x0 = (x0 - vx0) / (vx1 - vx0); x1 = (x1 - vx0) / (vx1 - vx0)
        y0 = (y0 - vy0) / (vy1 - vy0); y1 = (y1 - vy0) / (vy1 - vy0)
        dx0 = max(0, min(width - 1, int(round(x0 * width))))
        dy0 = max(0, min(height - 1, int(round(y0 * height))))
        dx1 = max(dx0 + 1, min(width, int(round(x1 * width))))
        dy1 = max(dy0 + 1, min(height, int(round(y1 * height))))
        return dx0, dy0, dx1, dy1

    def _redraw_video(self, *_):
        base = self._base_frame()
        ch_h, ch_w = base.shape[:2]
        out = base.copy()
        # In the selection view nothing is processed, so paint the bare frame:
        # FrameView draws the replicate boxes on top so a tile can be clicked.
        if self.active_region_index < 0:
            self.video_view.set_frame(out, image_frac=self._render_frac,
                                      coordinate_size=(self.src_w, self.src_h))
            return
        raw = self._overlay_peek_hidden
        arr = self._chan_arr()
        cmap = CHANNELS[self.channel][1]

        # Spatial overlay: the selected channel at the current frame, per active
        # replicate tile.
        for region in self._active_regions():
            y0, x0, y1, x1 = region["atlas_bbox"]
            dx0, dy0, dx1, dy1 = self._display_bbox(region, ch_w, ch_h)
            if raw:
                continue
            field = arr[self.frame, y0:y1, x0:x1]
            norm = np.clip(field / self._ov_scale, 0, 1)
            heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
            heat = cv2.resize(heat, (dx1 - dx0, dy1 - dy0),
                              interpolation=cv2.INTER_NEAREST)
            roi = out[dy0:dy1, dx0:dx1]
            out[dy0:dy1, dx0:dx1] = cv2.addWeighted(roi, 0.45, heat, 0.55, 0)

        # Highlight blocks passing the value band (the detection footprint).
        if self.hi_chk.isChecked() and not raw and self._block_sg is not None:
            m = self.density_plot.matrix
            lo, hi = self.density_plot.band()
            total = sum(s["n"] for s in self._block_snap)
            # Guard against a stale matrix mid-rebuild (region/channel switch).
            if m.size and self.frame < m.shape[0] and m.shape[1] == total:
                for s in self._block_snap:
                    y0, x0, y1, x1 = s["bbox"]
                    dx0, dy0, dx1, dy1 = self._display_bbox(
                        s["region"], ch_w, ch_h)
                    cols = m[self.frame, s["c0"]:s["c0"] + s["n"]]
                    passing = (cols >= lo) & (cols <= hi) & np.isfinite(cols)
                    grid = np.zeros((y1 - y0, x1 - x0), np.uint8)
                    grid[s["gy"] - y0, s["gx"] - x0] = passing.astype(np.uint8)
                    mm = cv2.resize(grid, (dx1 - dx0, dy1 - dy0),
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

    # -- events --------------------------------------------------------------
    def _seek(self, frame):
        self._update_frame(int(frame))

    def _on_scrub(self, v):
        if int(v) != self.frame:
            self._update_frame(int(v))

    def _advance(self):
        self._update_frame(0 if self.frame + 1 >= self.T else self.frame + 1)

    def _update_frame(self, frame):
        """Single entry point for a frame change. When embedded, route through
        the shared app state so sibling panels move together; otherwise apply
        locally."""
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
        for pl in (self.trace_plot, self.scalo_plot, self.density_plot,
                   self.count_plot):
            pl.set_cursor(self.frame)
        self.time_lbl.setText(f"{self.frame/self.fps:.2f} s  (#{self.frame})")
        if self.isVisible():
            self._redraw_video()

    def _on_region_changed(self, _index):
        data = self.region_combo.currentData()
        self.active_region_index = int(data) if data is not None else 0
        focus = None if self.active_region_index < 0 else \
            self._active_regions()[0]["frac"]
        self.video_view.set_focus_frac(focus)
        self._sync_video_boxes()
        self._sync_window_title()
        self._rebuild_scalograms()
        self._redraw_video()

    def _on_channel_toggled(self, button, checked: bool):
        # buttonToggled fires for both the newly-unchecked and newly-checked
        # box; act only on the one that turned on.
        if not checked:
            return
        self.channel = button.text()
        self._rebuild_scalograms()
        self._redraw_video()

    def _on_video_clicked(self, point):
        """Click a replicate tile in the video to focus it (packed atlases)."""
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
        if idx >= 0:                       # pooled scope exists: back to it
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
        scope = ("select a replicate"
                 if self.active_region_index < 0
                 else self._active_regions()[0]["label"])
        self.setWindowTitle(
            f"Scalogram explorer (expanded-cache proposal) -- "
            f"{os.path.basename(self.meta.get('video_path', '?'))} -- {scope}")

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self._timer.start(int(1000 / max(self.fps, 1)))
        else:
            self._timer.stop()

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

    def resizeEvent(self, _):
        self.video_view.update()

    def showEvent(self, e):
        super().showEvent(e)
        self._redraw_video()
