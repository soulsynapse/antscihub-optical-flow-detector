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

Two stacked bands, mirroring the two questions: the scalogram's frequency band
("which rhythm") and the density plot's value band ("which blocks are hot").
Nothing here is written to the cache; scalograms are derived on the fly from the
structure-tensor channels so the cost/benefit is visible before it is paid.
See docs/expanded_cache_plan.md.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QHBoxLayout,
                             QLabel, QProgressDialog, QPushButton, QScrollArea,
                             QSlider, QVBoxLayout, QWidget)

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


class ScalogramExplorer(QWidget):
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
        self.active_region_index = 0 if len(self.regions) == 1 else 0
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
        self._render_frac = (0.0, 0.0, 1.0, 1.0)
        # Per-block scalogram of the active replicate: (F, T, B). Rebuilt on
        # region/channel/frequency change; band drags only re-sum it.
        self._block_sg = None
        self._block_xy = None      # (gy, gx) atlas coords of each block column

        self.source = None
        if state is None and video_path and os.path.exists(video_path):
            from core.video import VideoSource
            try:
                self.source = VideoSource(video_path)
            except Exception:
                self.source = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._build_ui()
        self._rebuild_scalograms()

    # -- geometry ------------------------------------------------------------
    def _bbox(self):
        return self.regions[self.active_region_index]["atlas_bbox"]

    def _chan_arr(self):
        return self._chan[CHANNELS[self.channel][0]]

    # -- UI ------------------------------------------------------------------
    def _build_ui(self):
        self.setWindowTitle("Scalogram explorer (expanded-cache proposal)")
        self.resize(1600, 950)
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        self.video_view = FrameView()
        self.video_view.setMinimumSize(720, 480)
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
        for i, r in enumerate(self.regions):
            rid = r["id"]
            suffix = f" (#{rid})" if rid is not None else ""
            self.region_combo.addItem(f"{r['label']}{suffix}", i)
        self.region_combo.currentIndexChanged.connect(self._on_region)
        rrow.addWidget(self.region_combo, 1)
        left.addLayout(rrow)

        crow = QHBoxLayout()
        crow.addWidget(QLabel("Channel:"))
        self.chan_combo = QComboBox()
        self.chan_combo.addItems(list(CHANNELS))
        self.chan_combo.currentTextChanged.connect(self._on_channel)
        crow.addWidget(self.chan_combo, 1)
        self.hi_chk = QCheckBox("Highlight blocks in band")
        self.hi_chk.setChecked(True)
        self.hi_chk.stateChanged.connect(lambda _: self._redraw_video())
        crow.addWidget(self.hi_chk)
        left.addLayout(crow)

        note = QLabel(
            "Drag the FREQUENCY band on the scalogram to pick which rhythm to "
            "detect on; the density plot below shows the per-block power that "
            "band yields, with its own VALUE band -- the detection channel a "
            "scalogram cache would store. Compare with the structure-tensor "
            "explorer, whose single fixed band cannot be moved in frequency.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8ab; padding-top:4px;")
        left.addWidget(note)

        approx = ""
        info = QLabel(f"cache: {self.meta.get('backend','?')} | fps "
                      f"{self.fps:.2f} | {self.T} frames | scalogram "
                      f"{self.freqs[0]:.2f}-{self.freqs[-1]:.2f} Hz, "
                      f"{len(self.freqs)} scales{approx}")
        info.setStyleSheet("color:#888;")
        left.addWidget(info)
        root.addLayout(left, 3)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        col = QVBoxLayout(holder)
        col.setSpacing(4)

        self.trace_plot = MiniPlot("selected channel (replicate mean)")
        self.trace_plot.seek_requested.connect(self._seek)
        col.addWidget(self.trace_plot)

        self.scalo_plot = ScalogramPlot("scalogram (drag frequency band)",
                                        self.freqs)
        self.scalo_plot.seek_requested.connect(self._seek)
        self.scalo_plot.band_changed.connect(self._on_freq_band)
        self.scalo_plot.band_committed.connect(self._on_freq_band)
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
        """Recompute the replicate-mean scalogram (cheap) and the per-block
        scalogram (heavier; progress dialog) for the active replicate/channel."""
        y0, x0, y1, x1 = self._bbox()
        arr = self._chan_arr()
        pooled = arr[:, y0:y1, x0:x1].reshape(self.T, -1).mean(axis=1)
        self.trace_plot.set_series(pooled)
        self.scalo_plot.set_scalogram(morlet_power(pooled, self.fps, self.freqs))
        # Overlay color scale is a whole-clip percentile of this channel/region;
        # freeze it here so scrubbing (which redraws the video every frame) never
        # re-percentiles the array in the hot path.
        self._ov_scale = max(float(np.percentile(arr[:, y0:y1, x0:x1], 99)), EPS)

        blocks = arr[:, y0:y1, x0:x1].reshape(self.T, -1)     # (T, B)
        nb = blocks.shape[1]
        dlg = QProgressDialog(
            f"Morlet scalogram for {nb} blocks x {len(self.freqs)} scales...",
            None, 0, 0, self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        QApplication.processEvents()
        self._block_sg = morlet_power(blocks, self.fps, self.freqs)  # (F, T, B)
        dlg.close()
        gy, gx = np.mgrid[y0:y1, x0:x1]
        self._block_xy = (gy.ravel(), gx.ravel())
        self._on_freq_band()

    def _on_freq_band(self):
        """Sum the per-block scalogram over the selected frequency band -> the
        per-block band-power detection channel."""
        if self._block_sg is None:
            return
        flo, fhi = self.scalo_plot.band_hz()
        mask = (self.freqs >= flo) & (self.freqs <= fhi)
        if not mask.any():
            mask[np.argmin(np.abs(self.freqs - flo))] = True
        band_power = self._block_sg[mask].sum(axis=0)          # (T, B)
        self.density_plot.set_matrix(band_power)
        self._update_count()

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
        region = self.regions[self.active_region_index]
        focus = region["frac"]
        self._render_frac = focus
        bgr = None
        if self.state is not None:
            bgr = self.state.display_frame(self.frame, focus_frac=focus)
        elif self.source is not None:
            bgr = self.source.frame_at(self.frame)
            if bgr is not None:
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

    def _redraw_video(self, *_):
        base = self._base_frame()
        ch_h, ch_w = base.shape[:2]
        out = base.copy()
        y0, x0, y1, x1 = self._bbox()
        # Spatial overlay: the selected channel at the current frame.
        field = self._chan_arr()[self.frame, y0:y1, x0:x1]
        norm = np.clip(field / getattr(self, "_ov_scale", 1.0), 0, 1)
        heat = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                 CHANNELS[self.channel][1])
        heat = cv2.resize(heat, (ch_w, ch_h), interpolation=cv2.INTER_NEAREST)
        out = cv2.addWeighted(out, 0.45, heat, 0.55, 0)

        # Highlight blocks passing the value band (the detection footprint).
        if self.hi_chk.isChecked() and self._block_sg is not None:
            m = self.density_plot.matrix
            lo, hi = self.density_plot.band()
            if m.size and self.frame < m.shape[0]:
                passing = (m[self.frame] >= lo) & (m[self.frame] <= hi) & \
                    np.isfinite(m[self.frame])
                grid = np.zeros((y1 - y0, x1 - x0), np.uint8)
                gy, gx = self._block_xy
                grid[gy - y0, gx - x0] = passing.astype(np.uint8)
                mm = cv2.resize(grid, (ch_w, ch_h),
                                interpolation=cv2.INTER_NEAREST)
                tint = np.zeros_like(out)
                tint[..., 1] = 255
                blended = cv2.addWeighted(out, 0.5, tint, 0.5, 0)
                np.copyto(out, blended, where=(mm > 0)[:, :, None])
                contours, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, contours, -1, (60, 255, 60), 1)

        self.video_view.set_frame(out, image_frac=self._render_frac,
                                  coordinate_size=(self.src_w, self.src_h))

    # -- events --------------------------------------------------------------
    def _seek(self, frame):
        self.scrub.setValue(int(frame))

    def _on_scrub(self, v):
        self.frame = int(v)
        for pl in (self.trace_plot, self.scalo_plot, self.density_plot,
                   self.count_plot):
            pl.set_cursor(self.frame)
        self.time_lbl.setText(f"{self.frame/self.fps:.2f} s")
        self._redraw_video()

    def _on_region(self, i):
        self.active_region_index = int(i)
        self._rebuild_scalograms()
        self._redraw_video()

    def _on_channel(self, name):
        self.channel = name
        self._rebuild_scalograms()
        self._redraw_video()

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self._timer.start(int(1000 / max(self.fps, 1)))
        else:
            self._timer.stop()

    def _advance(self):
        self.scrub.setValue((self.frame + 1) % self.T)

    def showEvent(self, e):
        super().showEvent(e)
        self._redraw_video()
