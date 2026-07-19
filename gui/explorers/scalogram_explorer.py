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
here too, an exclusive checkbox group beside a per-channel band-power heatmap.

To let the data be explored multiple ways (like the tensor explorer's stack of
heatmaps) there is one band-power density heatmap PER source channel, plus the
tensor's detection-sweep suite (windowed in-band count over D, a binary detection
gate, and the largest connected clump). The heavy part -- a per-block Morlet cube
runs hundreds of MB and scales with clip length -- is why the per-channel cubes
build LAZILY: only the channel you actually check (detect on) is built, so you
never pay for five cubes at once. A channel you have visited stays cached, so
you can flick between channels to compare their band-power distributions.

Nothing here is written to the cache; scalograms are derived on the fly from the
structure-tensor channels so the cost/benefit is visible before it is paid.
See docs/expanded_cache_plan.md.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from functools import partial

import cv2
import numpy as np

from PyQt6.QtCore import QEvent, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QImage, QKeySequence, QPainter, QPen,
                         QShortcut)
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                             QGridLayout, QHBoxLayout, QLabel, QProgressDialog,
                             QPushButton, QScrollArea, QSlider, QVBoxLayout,
                             QWidget)

from core.channel_source import cache_channel_source
from core.detection import (detect_gate, inband_count, largest_clump_per_frame,
                            window_bounds, windowed_mean)
from core.wavelet import band_indices, default_freqs, morlet_power
from gui.video_panel import FrameView
from gui.explorers.speed_explorer import (BG, CURSOR, DISPLAY_MAX_W, PLOT_BG,
                                          TXT, TXT_DIM, DensityPlot, MiniPlot,
                                          PixelBarPlot, _regions_from_meta)

EPS = 1e-6
SWEEP_C = QColor(110, 230, 120)      # in-band count / clump
DETECT_C = QColor(120, 255, 140)     # binary detection gate

# Channel -> (attribute, overlay colormap). All are nonnegative energies except
# cached speed; the scalogram cares only about their temporal fluctuation, so the
# choice is about which signal's rhythm you want to see.
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
        # Seed the band through set_band_active, as every other band plot does,
        # rather than a bare band_active = True (both handles start wide open).
        self.set_band_active(True)
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
                 state=None, sidecar_path: str | None = None,
                 channel_data=None, own_shortcuts: bool = True,
                 own_status: bool = True, parent=None):
        super().__init__(parent)
        if state is not None and cache is None and channel_data is None:
            cache = state.cache
        self.state = state
        self.cache = cache
        # When embedded in a host that owns Space (the live surface / main
        # window), skip our own Space shortcut so the two do not fight over it.
        self._own_shortcuts = own_shortcuts
        # When embedded, the host displays the status line (in its own strip): skip
        # rendering our internal copy and mirror the text to the host-supplied
        # relay label instead (see set_status_relay).
        self._own_status = own_status
        self._status_relay = None

        # Source of geometry + channels. A ChannelData decouples us from the
        # cache: it comes either from an open cache (all five channels) or a live
        # windowed pass over a bare video (four channels, no cached flow speed).
        if channel_data is None:
            if cache is None:
                raise ValueError(
                    "ScalogramExplorer requires a cache, state, or channel_data")
            dlg = QProgressDialog("Extracting structure-tensor channels...", None,
                                  0, int(cache.meta["n_frames"]), self)
            dlg.setWindowModality(Qt.WindowModality.WindowModal)
            dlg.setMinimumDuration(0)

            def prog(done, total):
                dlg.setMaximum(total)
                dlg.setValue(done)
                QApplication.processEvents()

            channel_data = cache_channel_source(
                cache, sidecar_path=sidecar_path, progress=prog)
            dlg.close()

        self._cd = channel_data
        self.meta = channel_data.meta
        # Absolute video frame the T axis starts at (0 for full-clip sources);
        # the video overlay adds it back to decode the right frame.
        self.window_start = int(channel_data.window_start)
        self.fps = float(self.meta["fps"])
        self.dt = 1.0 / max(self.fps, EPS)
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

        # Channels present in this source. tensor_speed/change/intensity/
        # appearance are always here; "speed" (cached flow) only from a cache.
        self._chan = {k: np.asarray(v, np.float32)
                      for k, v in channel_data.channels.items()}
        self._channels_bytes = sum(a.nbytes for a in self._chan.values())
        self.T = self._chan["change"].shape[0]
        self.freqs = default_freqs(self.fps)
        self.channel = "change energy Jtt"
        # One-line status ("what is it doing right now") shown persistently under
        # the video. Set before every heavy step so a GUI-thread stall is legible.
        self._phase = "starting…"
        self.frame = int(state.current_frame) if state is not None else 0
        self.playing = False
        self._overlay_peek_hidden = False
        self._render_frac = (0.0, 0.0, 1.0, 1.0)
        self._ov_scale = EPS       # replaced by a real percentile in _rebuild
        # Detection-window D (frames): the binary gate reads the centered mean of
        # "# blocks in band" over D, so a 1-frame spike of N blocks dilutes to
        # N/D and cannot fake a sustained event. Centered by default (offline, so
        # the gate fires ON the event rather than D/2 frames after it).
        self.sweep_win = max(1, min(self.T - 1, int(round(self.fps))))
        self.centered = True
        # Per-region column bookkeeping into a cube's B axis: maps each block
        # column back to its grid cell. Cube-independent (pure geometry), so it
        # is set the moment a replicate is selected -- before any cube lands.
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
        # Cheap in-band counts stay live while a band drags; the expensive clump
        # (per-frame connected components) refreshes only on release.
        self._sweep_debounce = QTimer(self)
        self._sweep_debounce.setSingleShot(True)
        self._sweep_debounce.setInterval(120)
        self._sweep_debounce.timeout.connect(self._recompute_counts)
        self._build_ui()

        # Shift-to-peek (hide the overlay to read the raw frame) rides an
        # application-wide event filter, exactly as the structure-tensor
        # explorer does; Space toggles playback in the standalone case (when
        # embedded, the host window owns Space and drives us via frame_changed).
        self._event_filter_app = QApplication.instance()
        if self._event_filter_app is not None:
            self._event_filter_app.installEventFilter(self)
        self._space_shortcut = None
        if state is None and self._own_shortcuts:
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

    @classmethod
    def from_channel_data(cls, channel_data, video_path: str | None = None,
                          own_shortcuts: bool = True, own_status: bool = True,
                          parent=None) -> "ScalogramExplorer":
        """Standalone explorer over a ChannelData (e.g. a live windowed source),
        decoding its own video for the overlay."""
        return cls(channel_data=channel_data,
                   video_path=video_path or channel_data.meta.get("video_path"),
                   own_shortcuts=own_shortcuts, own_status=own_status,
                   parent=parent)

    # -- view-state carry-over (survives a live re-extract rebuild) ----------
    def capture_view_state(self) -> dict:
        """The tuning context worth preserving when the live surface rebuilds this
        explorer with a fresh ChannelData: which channel, where the cursor and
        detection window are, and the frequency band you dragged.

        The two detection threshold bands travel too: ``detection_params`` reads
        the value band and the count band off live widgets, so omitting them here
        silently reverted both to defaults on every rebuild (T17). Endpoints are
        carried raw -- ``None`` means "never placed", which ``set_band_active``
        seeds lazily; collapsing it to +/-inf here would turn "unset" into an
        explicit unbounded threshold and defeat that seeding.
        """
        return {
            "channel": self.channel,
            "frame": self.frame,
            "sweep_win": self.sweep_win,
            "centered": self.centered,
            "freq_band": (self.scalo_plot.band_lo, self.scalo_plot.band_hi),
            # Per channel, not just the selected one: switching channels after a
            # rebuild should not find the others wiped either.
            "value_bands": {name: (dp.band_lo, dp.band_hi)
                            for name, dp in self.density_plots.items()},
            "count_band": (self.count_w_plot.band_lo, self.count_w_plot.band_hi),
        }

    def detection_params(self) -> dict:
        """The tuned detector settings, for the whole-video commit pass. Returns
        the channel attribute, selected region, and the three bands + window so
        the whole-clip pass reproduces exactly what this preview shows."""
        dp = self._selected_density()
        return {
            "channel_attr": CHANNELS[self.channel][0],
            "region_index": self.active_region_index,
            "freq_band_hz": self.scalo_plot.band_hz(),
            "value_band": dp.band() if dp is not None
            else (float("-inf"), float("inf")),
            "count_band": self.count_w_plot.band(),
            "detect_window": self.sweep_win,
            "centered": self.centered,
        }

    def apply_view_state(self, st: dict) -> None:
        """Restore a captured view state onto this (freshly built) explorer."""
        ch = st.get("channel")
        if ch in self.chan_checks and self._channel_available(ch):
            self.channel = ch
            self.chan_checks[ch].setChecked(True)   # fires _on_channel_toggled
        sw = st.get("sweep_win")
        if sw:
            self.sweep_win = max(1, min(max(2, self.T - 1), int(sw)))
            self.sweep_win_slider.setValue(self.sweep_win)
        if "centered" in st:
            self.centered = bool(st["centered"])
            self.centered_chk.setChecked(self.centered)
        fb = st.get("freq_band")
        if fb is not None:
            self.scalo_plot.band_lo, self.scalo_plot.band_hi = fb
            self.scalo_plot.set_band_active(True)
            self.scalo_plot.update()
        # Detection thresholds must land BEFORE the _on_freq_band_committed()
        # below, or the recompute at the end of this method runs against the
        # defaults these are here to replace.
        for name, band in (st.get("value_bands") or {}).items():
            dp = self.density_plots.get(name)
            if dp is not None:
                dp.band_lo, dp.band_hi = band
                dp.update()
        cb = st.get("count_band")
        if cb is not None:
            self.count_w_plot.band_lo, self.count_w_plot.band_hi = cb
            self.count_w_plot.set_band_active(True)
            self.count_w_plot.update()
        frame = st.get("frame")
        if frame is not None:
            self._apply_frame(int(frame))
        if self.active_region_index >= 0:
            self._on_freq_band_committed()

    # -- scope / geometry ----------------------------------------------------
    def _regions_for_index(self, idx: int) -> list[dict]:
        """The regions a scope index covers: all of them when pooled (<0)."""
        return self.regions if idx < 0 else [self.regions[idx]]

    def _active_regions(self) -> list[dict]:
        return self._regions_for_index(self.active_region_index)

    def _channel_available(self, name: str) -> bool:
        """Whether this source carries the channel behind display name ``name``.
        Live (cacheless) sources lack ``cached flow speed``."""
        return CHANNELS[name][0] in self._chan

    def _chan_arr(self):
        attr = CHANNELS[self.channel][0]
        if attr not in self._chan:                     # defensive: absent channel
            return np.zeros((self.T, self.ny, self.nx), np.float32)
        return self._chan[attr]

    def _scope_blocks(self, arr: np.ndarray, idx: int) -> np.ndarray:
        """(T, B) block columns of ``arr`` over the regions of scope ``idx``,
        concatenated in region order -- the column order the cube's B axis and
        ``_block_snap`` both follow."""
        parts = [arr[:, y0:y1, x0:x1].reshape(self.T, -1)
                 for (y0, x0, y1, x1) in
                 (r["atlas_bbox"] for r in self._regions_for_index(idx))]
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

    def _make_snap(self, idx: int) -> list[dict]:
        """Per-region block-column map for scope ``idx`` (pure geometry)."""
        snap: list[dict] = []
        c0 = 0
        for region in self._regions_for_index(idx):
            y0, x0, y1, x1 = region["atlas_bbox"]
            gy, gx = np.mgrid[y0:y1, x0:x1]
            n = (y1 - y0) * (x1 - x0)
            snap.append({"bbox": (y0, x0, y1, x1), "gy": gy.ravel(),
                         "gx": gx.ravel(), "c0": c0, "n": n, "region": region})
            c0 += n
        return snap

    def _all_plots(self):
        return [self.trace_plot, self.scalo_plot, *self.density_plots.values(),
                self.count_plot, self.count_w_plot, self.detect_plot,
                self.clump_plot]

    # -- UI ------------------------------------------------------------------
    def _build_ui(self):
        self.resize(1600, 980)
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
        self.centered_chk = QCheckBox("Centered detection window")
        self.centered_chk.setToolTip(
            "Checked: the detection window D is centered [t-D/2, t+D/2], so the "
            "gate fires ON the event (offline; reads the future). Unchecked: "
            "trailing [t-D+1, t] -- what a causal live detector would see.")
        self.centered_chk.setChecked(self.centered)
        self.centered_chk.stateChanged.connect(self._on_centered_toggle)
        hrow.addWidget(self.centered_chk)
        hrow.addStretch(1)
        left.addLayout(hrow)

        self.sweep_win_slider, self.sweep_win_lbl = self._add_slider(
            left, 1, max(2, self.T - 1), self.sweep_win, self._on_sweep_window)

        note = QLabel(
            "Drag the FREQUENCY band on the scalogram to pick which rhythm to "
            "detect on; each channel's density plot below shows the per-block "
            "power that band yields, with its own VALUE band. Check a channel to "
            "detect on it -- its per-block cube builds on demand (nothing else "
            "is precomputed). Click a replicate in the video to select it. "
            "Space toggles playback; hold Shift to peek at the raw frame.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8ab; padding-top:4px;")
        left.addWidget(note)

        info = QLabel(f"cache: {self.meta.get('backend','?')} | fps "
                      f"{self.fps:.2f} | {self.T} frames | scalogram "
                      f"{self.freqs[0]:.2f}-{self.freqs[-1]:.2f} Hz, "
                      f"{len(self.freqs)} scales")
        info.setStyleSheet("color:#888;")
        left.addWidget(info)
        # Persistent activity/memory line. Everything heavy on this explorer is
        # either off-thread (the cube worker) or a blocking O(F*T*B) / O(T)
        # GUI-thread pass; this line names the current step and reports retained
        # cube memory vs budget so an unresponsive stretch has a visible cause.
        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(False)
        self._status_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self._status_lbl.setStyleSheet(
            "color:#e0a94a; font-family:Consolas; font-size:11px;")
        if self._own_status:
            left.addWidget(self._status_lbl)
        root.addLayout(left, 3)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        col = QVBoxLayout(holder)
        col.setSpacing(4)

        def section(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "color:#7fd7ff; font-weight:bold; padding-top:6px;")
            lbl.setWordWrap(True)
            col.addWidget(lbl)

        section("Selected channel")
        self.trace_plot = MiniPlot("selected channel (replicate mean)")
        self.trace_plot.seek_requested.connect(self._seek)
        col.addWidget(self.trace_plot)

        self.scalo_plot = ScalogramPlot("scalogram (drag frequency band)",
                                        self.freqs)
        self.scalo_plot.seek_requested.connect(self._seek)
        # band_changed fires on every mouse-move; the band-power re-sum is an
        # O(F*T*B) pass per cached channel, so debounce it and recompute eagerly
        # only on release.
        self.scalo_plot.band_changed.connect(self._freq_debounce.start)
        self.scalo_plot.band_committed.connect(self._on_freq_band_committed)
        col.addWidget(self.scalo_plot)

        # Per-channel band-power heatmaps, each with an inline exclusive checkbox
        # (the structure-tensor idiom). Checking one selects it as the detection
        # channel AND triggers its lazy cube build; visited channels stay cached.
        section("Per-block band power by channel — check to detect (builds on "
                "demand)")
        grid_w = QWidget()
        grid = QGridLayout(grid_w)
        grid.setSpacing(3)
        grid.setColumnMinimumWidth(0, 20)
        grid.setColumnStretch(1, 1)
        self.chan_group = QButtonGroup(self)
        self.chan_group.setExclusive(True)
        self.chan_checks: dict[str, QCheckBox] = {}
        self.density_plots: dict[str, DensityPlot] = {}
        for r, name in enumerate(CHANNELS):
            cb = QCheckBox()
            cb.setProperty("channel", name)
            available = self._channel_available(name)
            cb.setEnabled(available)
            cb.setToolTip(
                f"Detect on {name} (builds its cube on first check)" if available
                else f"{name} needs a flow cache — unavailable on the live source")
            cb.setChecked(available and name == self.channel)
            self.chan_group.addButton(cb)
            self.chan_checks[name] = cb
            grid.addWidget(cb, r, 0, Qt.AlignmentFlag.AlignCenter)
            dp = DensityPlot(name)
            dp.set_log_axis(True)
            dp.seek_requested.connect(self._seek)
            dp.band_changed.connect(partial(self._on_density_band_changed, name))
            dp.band_committed.connect(
                partial(self._on_density_band_committed, name))
            self.density_plots[name] = dp
            grid.addWidget(dp, r, 1)
        col.addWidget(grid_w)
        self.chan_group.buttonToggled.connect(self._on_channel_toggled)

        section("Detection sweep (selected channel)")
        self.count_plot = PixelBarPlot("# blocks in band", unit="blocks",
                                       color=SWEEP_C)
        self.count_plot.seek_requested.connect(self._seek)
        col.addWidget(self.count_plot)
        self.count_w_plot = MiniPlot("windowed # blocks in band (mean over D)",
                                     "blocks", SWEEP_C)
        self.count_w_plot.seek_requested.connect(self._seek)
        self.count_w_plot.set_band_active(True)
        self.count_w_plot.set_expanded(True)
        self.count_w_plot.band_changed.connect(self._recompute_detect)
        self.count_w_plot.band_committed.connect(self._recompute_detect)
        col.addWidget(self.count_w_plot)
        self.detect_plot = MiniPlot("positive detection (windowed count in band)",
                                    "0/1", DETECT_C)
        self.detect_plot.seek_requested.connect(self._seek)
        col.addWidget(self.detect_plot)
        self.clump_plot = PixelBarPlot("largest connected clump in band",
                                       unit="blocks", color=SWEEP_C)
        self.clump_plot.seek_requested.connect(self._seek)
        col.addWidget(self.clump_plot)

        col.addStretch(1)
        scroll.setWidget(holder)
        scroll.setMinimumWidth(480)
        root.addWidget(scroll, 2)

        self._apply_selected_plot_ui()
        self._sync_labels()
        self._update_status()

    def _add_slider(self, layout, lo, hi, val, handler):
        row = QHBoxLayout()
        lbl = QLabel()
        lbl.setMinimumWidth(260)
        row.addWidget(lbl)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        s.valueChanged.connect(handler)
        row.addWidget(s, 1)
        layout.addLayout(row)
        return s, lbl

    def _apply_selected_plot_ui(self):
        """Expand + activate the value band on the selected channel's density;
        collapse and deactivate the rest (their matrices stay visible)."""
        for name, dp in self.density_plots.items():
            sel = (name == self.channel)
            dp.set_band_active(sel)
            dp.set_expanded(sel)

    def _sync_labels(self):
        self.sweep_win_lbl.setText(
            f"Detection window D (count mean): {self.sweep_win} fr "
            f"({self.sweep_win * self.dt:.2f} s)")

    # -- frequency-band -> per-channel band power ---------------------------
    def _band_indices(self) -> tuple[int, int]:
        """[i, j) slice on the sorted frequency axis for the current band."""
        flo, fhi = self.scalo_plot.band_hz()
        return band_indices(self.freqs, flo, fhi)

    def _refresh_densities(self):
        """Re-sum every CACHED channel cube over the current frequency band into
        its density heatmap; blank the channels whose cube isn't built (or was
        evicted) so the display never shows stale band power."""
        if self.active_region_index < 0:
            return
        i, j = self._band_indices()
        n_cached = sum(1 for name in self.density_plots
                       if self._sg_cache.get((self.active_region_index, name))
                       is not None)
        if n_cached:
            self._set_phase(f"summing band power ({n_cached} cube"
                            f"{'s' if n_cached > 1 else ''})…", paint=True)
        empty = np.zeros((0, 0), np.float32)
        for name, dp in self.density_plots.items():
            cube = self._sg_cache.get((self.active_region_index, name))
            dp.set_matrix(cube[i:j].sum(axis=0) if cube is not None else empty)

    # -- scalogram computation ----------------------------------------------
    def _rebuild_selected_views(self):
        """Cheap per-frame views of the selected channel (no cube needed): the
        replicate-mean trace and the pooled-mean scalogram + overlay scale."""
        arr = self._chan_arr()
        blocks = self._scope_blocks(arr, self.active_region_index)   # (T, B)
        pooled = blocks.mean(axis=1)
        self.trace_plot.set_series(pooled)
        self.trace_plot.set_cursor(self.frame)
        self._set_phase("transforming pooled scalogram…", paint=True)
        self.scalo_plot.set_scalogram(morlet_power(pooled, self.fps, self.freqs))
        self.scalo_plot.set_cursor(self.frame)
        # Overlay color scale is a whole-clip percentile of this channel/scope;
        # freeze it here so scrubbing (which redraws every frame) never
        # re-percentiles the array in the hot path.
        self._ov_scale = max(float(np.percentile(blocks, 99)), EPS)

    def _rebuild_scalograms(self):
        """React to a replicate or channel change: refresh the cheap views, then
        serve the selected channel's per-block cube from the memo cache or a
        background build. Other channels' densities fill in only if their cube is
        already cached (lazy)."""
        if self.active_region_index < 0:
            # Selection view: nothing selected, so process nothing. Clear the
            # plots (a prior selection's data must not linger); the video shows
            # the full frame with boxes.
            self._block_snap = []
            for pl in self._all_plots():
                if isinstance(pl, ScalogramPlot):
                    pl.set_scalogram(np.zeros((0, 0), np.float32))
                elif isinstance(pl, DensityPlot):
                    pl.set_matrix(np.zeros((0, 0), np.float32))
                else:
                    pl.set_series(np.zeros(0, np.float32))
            self._set_busy(False)
            return

        self._block_snap = self._make_snap(self.active_region_index)
        self._rebuild_selected_views()
        self._apply_selected_plot_ui()
        self._refresh_densities()
        self._recompute_sweep()
        # Build the selected channel's cube if it isn't cached yet.
        if self._sg_cache.get((self.active_region_index, self.channel)) is None:
            self._request_cube()
        else:
            self._sg_cache.move_to_end((self.active_region_index, self.channel))
            self._set_busy(False)

    def _request_cube(self):
        """Build the SELECTED channel's cube unless it is already cached or a
        build is in flight. Only the selected channel is ever built (lazy): the
        running worker re-checks the current selection when it ends, so a rapid
        channel/region switch coalesces onto the latest target."""
        if self.active_region_index < 0 or self._worker is not None:
            return
        key = (self.active_region_index, self.channel)
        if self._sg_cache.get(key) is not None:
            return
        self._launch_worker(key)

    def _launch_worker(self, key):
        region_idx, channel = key
        arr = self._chan[CHANNELS[channel][0]]
        # A private contiguous copy: the worker reads it while the GUI thread
        # keeps using self._chan freely.
        blocks = np.ascontiguousarray(self._scope_blocks(arr, region_idx))
        # Announce the build with the cube's projected footprint (F*T*B float32),
        # so the wait has a number attached before the worker even starts.
        b = blocks.shape[1]
        est = len(self.freqs) * self.T * b * 4
        self._set_phase(f"building scalogram · {channel} · "
                        f"{len(self.freqs)}×{self.T}×{b} (~{self._fmt_bytes(est)})…",
                        paint=True)
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
        region_idx, channel = key
        self._evict_cache(protect=(self.active_region_index, self.channel))
        if region_idx == self.active_region_index and region_idx >= 0:
            # Fold the new cube into its channel's density (and, if it is the
            # selected one, the sweep + overlay).
            self._refresh_densities()
            if channel == self.channel:
                self._recompute_sweep()
                self._redraw_video()
        # The selection may have moved while this built; build the new target.
        self._request_cube()
        if self._worker is None:
            self._set_busy(False)

    def _evict_cache(self, protect):
        """Drop oldest cubes until the retained total is under budget, but never
        the ``protect`` (current selected) key."""
        total = sum(c.nbytes for c in self._sg_cache.values())
        for k in list(self._sg_cache):             # oldest first
            if total <= self._SG_CACHE_BUDGET:
                break
            if k == protect:
                continue
            total -= self._sg_cache.pop(k).nbytes

    # -- status line ---------------------------------------------------------
    @staticmethod
    def _fmt_bytes(n) -> str:
        n = float(n)
        for unit, thresh in (("GB", 1024 ** 3), ("MB", 1024 ** 2),
                             ("KB", 1024.0)):
            if n >= thresh:
                return f"{n / thresh:.1f} {unit}"
        return f"{n:.0f} B"

    def _update_status(self) -> None:
        """Recompose the persistent line from live state (cheap: a few sums)."""
        if not hasattr(self, "_status_lbl"):
            return
        parts = [self._phase]
        total = sum(c.nbytes for c in self._sg_cache.values())
        parts.append(
            f"cubes {len(self._sg_cache)} · {self._fmt_bytes(total)}/"
            f"{self._fmt_bytes(self._SG_CACHE_BUDGET)}")
        cube = self._sg_cache.get((self.active_region_index, self.channel))
        if cube is not None:
            f, t, b = cube.shape
            parts.append(f"cube {f}×{t}×{b} ({self._fmt_bytes(cube.nbytes)})")
        parts.append(f"channels {self._fmt_bytes(self._channels_bytes)}")
        # The frequency band is deliberately absent: ScalogramPlot already draws
        # it in its own title, right above the handles you drag to set it.
        text = "   ·   ".join(parts)
        self._status_lbl.setText(text)
        if self._status_relay is not None:
            self._status_relay.setText(text)

    def set_status_relay(self, label) -> None:
        """Mirror the status line into a host-supplied QLabel (the live surface's
        top strip). The direct reference matters: ``_set_phase(paint=True)`` force-
        repaints this label so a 'computing…' phase shows *during* the blocking
        GUI-thread step, which a queued signal could not do. Pushes the current
        text immediately so the host is not blank until the next state change."""
        self._status_relay = label
        self._update_status()

    def _set_phase(self, phase: str, *, paint: bool = False) -> None:
        """Set the current-activity string and refresh the line. ``paint=True``
        forces an immediate synchronous repaint -- use it right before a blocking
        GUI-thread step so the label shows the step *while* it stalls."""
        self._phase = phase
        self._update_status()
        if paint:
            # Force a synchronous repaint of whichever label is actually visible so
            # the phase shows *while* the blocking step that follows stalls the
            # event loop. The relay (when hosted) is the one on screen.
            target = self._status_relay
            if target is None and hasattr(self, "_status_lbl"):
                target = self._status_lbl
            if target is not None:
                target.repaint()

    def _settle_phase(self):
        """Return the line to a resting state -- but never stomp a build that is
        still running, whose phase should stand until its cube lands."""
        if self._worker is not None:
            self._update_status()
            return
        self._set_phase("select a replicate" if self.active_region_index < 0
                        else "ready")

    def _set_busy(self, busy):
        # A finished/settled state: name what's next rather than blanking, so the
        # line always reports memory even at rest.
        if not busy:
            self._settle_phase()

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
        """Frequency band moved: re-sum cached channel cubes and refresh the
        cheap sweep counts (clump waits for release)."""
        if self.active_region_index < 0 or not self._block_snap:
            return
        self._refresh_densities()
        self._recompute_counts()
        self._redraw_video()

    def _on_freq_band_committed(self):
        self._freq_debounce.stop()
        if self.active_region_index < 0 or not self._block_snap:
            return
        self._refresh_densities()
        self._recompute_sweep()
        self._redraw_video()

    # -- detection sweep (selected channel) ---------------------------------
    def _selected_density(self) -> "DensityPlot | None":
        return self.density_plots.get(self.channel)

    def _window_bounds(self, W):
        """Prefix-sum bounds per frame for the detection window D (shared with the
        whole-video pass so the two agree)."""
        return window_bounds(self.T, W, self.centered)

    def _recompute_sweep(self):
        self._recompute_counts()
        self._recompute_clump()
        self._settle_phase()

    def _recompute_counts(self):
        """# blocks in the selected channel's value band per frame, and its
        windowed mean over D. Cheap: runs live while a band drags."""
        dp = self._selected_density()
        m = dp.matrix if dp is not None else None
        if m is None or m.size == 0:
            for pl in (self.count_plot, self.count_w_plot, self.detect_plot):
                pl.set_series(np.zeros(0, np.float32))
            return
        lo, hi = dp.band()
        count = inband_count(m, lo, hi)
        self.count_plot.set_series(count)
        self.count_plot.set_cursor(self.frame)
        self._recompute_windowed_count(count)
        self._settle_phase()

    def _recompute_windowed_count(self, count: np.ndarray | None = None):
        if count is None:
            count = self.count_plot.y
        if count.size == 0:
            self.count_w_plot.set_series(np.zeros(0, np.float32))
            self._recompute_detect()
            return
        windowed = windowed_mean(count, self.sweep_win, self.centered)
        self.count_w_plot.set_series(windowed)
        self.count_w_plot.set_cursor(self.frame)
        self._recompute_detect()

    def _recompute_detect(self):
        blo, bhi = self.count_w_plot.band()
        y = self.count_w_plot.y
        self.detect_plot.set_series(
            detect_gate(y, blo, bhi) if y.size else np.zeros(0, np.float32))
        self.detect_plot.set_cursor(self.frame)

    def _recompute_clump(self):
        """Largest connected in-band clump per frame for the selected channel
        (block-grid 8-connectivity). O(T) connected-components; refreshed only on
        band/frequency release, never mid-drag."""
        dp = self._selected_density()
        if dp is None or dp.matrix.size == 0 or not self._block_snap:
            self.clump_plot.set_series(np.zeros(0, np.float32))
            return
        m = dp.matrix                                          # (T, B)
        s = self._block_snap[0]                                # single region
        y0, x0, y1, x1 = s["bbox"]
        dy, dx = y1 - y0, x1 - x0
        gy, gx = s["gy"] - y0, s["gx"] - x0
        if m.shape[1] != gy.size:                              # stale mid-switch
            self.clump_plot.set_series(np.zeros(0, np.float32))
            return
        lo, hi = dp.band()
        self._set_phase(f"computing connected clumps ({m.shape[0]} frames)…",
                        paint=True)
        largest = largest_clump_per_frame(m, lo, hi, dy, dx, gy, gx)
        self.clump_plot.set_series(largest)
        self.clump_plot.set_cursor(self.frame)

    # -- video overlay -------------------------------------------------------
    def _base_frame(self):
        focus = None if self.active_region_index < 0 else \
            self._active_regions()[0]["frac"]
        self._render_frac = focus or (0.0, 0.0, 1.0, 1.0)
        bgr = None
        if self.state is not None:
            bgr = self.state.display_frame(self.frame, focus_frac=focus)
        elif self.source is not None:
            bgr = self.source.frame_at(self.window_start + self.frame)
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

        # Spatial overlay: the selected channel at the current frame.
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

        # Highlight blocks passing the selected channel's value band.
        dp = self._selected_density()
        if self.hi_chk.isChecked() and not raw and dp is not None \
                and self._block_snap:
            m = dp.matrix
            lo, hi = dp.band()
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
        for pl in self._all_plots():
            pl.set_cursor(self.frame)
        self.time_lbl.setText(f"{self.frame/self.fps:.2f} s  (#{self.frame})")
        if self.isVisible():
            self._redraw_video()

    def _on_region_changed(self, _index):
        data = self.region_combo.currentData()
        self.active_region_index = int(data) if data is not None else 0
        # Each replicate lives on its own value scale, so a value band frozen
        # from the previous replicate would sit off this one's plots. Re-seed
        # every channel's value band (and the count gate, whose block count just
        # changed) wide open; the selected plot re-seeds via _apply_selected_ui.
        for dp in self.density_plots.values():
            dp.band_lo = dp.band_hi = None
        self.count_w_plot.band_lo = self.count_w_plot.band_hi = None
        self.count_w_plot.set_band_active(True)
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
        self.channel = str(button.property("channel"))
        if self.active_region_index < 0:
            return
        self._rebuild_selected_views()
        self._apply_selected_plot_ui()
        self._refresh_densities()
        self._recompute_sweep()
        self._redraw_video()
        self._request_cube()      # build the newly-selected channel if needed

    def _on_density_band_changed(self, name: str):
        if name != self.channel:
            return
        self._sweep_debounce.start()   # cheap counts debounced
        self._redraw_video()           # highlight overlay follows immediately

    def _on_density_band_committed(self, name: str):
        if name != self.channel:
            return
        self._sweep_debounce.stop()
        self._recompute_sweep()
        self._redraw_video()

    def _on_sweep_window(self, v):
        self.sweep_win = max(1, int(v))
        self._sync_labels()
        self._recompute_windowed_count()

    def _on_centered_toggle(self, _state=None):
        self.centered = self.centered_chk.isChecked()
        self._recompute_windowed_count()

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
        if idx >= 0:                       # selection scope exists: back to it
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
