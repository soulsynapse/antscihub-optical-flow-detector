"""Standalone per-block SPEED explorer.

A throwaway diagnostic to answer one question the main app currently makes you
guess at: what does the speed signal actually look like, per block, over time --
and in particular, what does the "number of blocks above a threshold" signal (the
thing detection really thresholds) do as you move the threshold?

It does NOT recompute optical flow. It opens a feature cache you already built in
the main app (Tab 1) and reads its cached `speed` / `u` / `v` block arrays, so
every number here is exactly the number the real pipeline sees -- there is no
second, slightly-different flow implementation to mistrust.

Run:
    .venv\\Scripts\\python.exe scripts/speed_explorer.py
    .venv\\Scripts\\python.exe scripts/speed_explorer.py --cache <key>
    .venv\\Scripts\\python.exe scripts/speed_explorer.py --cache-root ./.cache

With no --cache it lists the caches under --cache-root and lets you pick one.

Left panel  : the video with a toggleable optical-flow overlay + transport.
Right panel : a long scrollable column of speed-only time readouts. The
              threshold slider drives the "above threshold" group live, so you can
              watch the block-count signal reshape itself as you sweep the cutoff.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

# Import the project's core modules. This script lives in scripts/, so add the
# repo root to sys.path when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QHBoxLayout,
                             QInputDialog, QLabel, QPushButton, QScrollArea,
                             QSlider, QVBoxLayout, QWidget)

from core import cache as cache_mod

# -- palette -----------------------------------------------------------------
BG = QColor(24, 24, 24)
PLOT_BG = QColor(12, 12, 12)
LINE = QColor(120, 215, 255)
LINE2 = QColor(255, 170, 80)
CURSOR = QColor(255, 210, 80)
TXT = QColor(210, 210, 210)
TXT_DIM = QColor(140, 140, 140)


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
    def __init__(self, cache, video_path: str | None):
        super().__init__()
        self.cache = cache
        self.meta = cache.meta
        self.fps = float(self.meta["fps"])
        self.block = int(self.meta["block_size"])
        self.ny, self.nx = map(int, self.meta["grid"])
        self.n_blocks = self.ny * self.nx
        self.work_w = int(self.meta.get("work_width", self.nx * self.block))
        self.work_h = int(self.meta.get("work_height", self.ny * self.block))

        # Read the per-block arrays once, as float32. This is the whole point:
        # these are the same arrays roi_detection thresholds.
        self.speed = cache.read("speed").astype(np.float32)   # (T, ny, nx)
        feats = self.meta.get("features", [])
        self.u = cache.read("u").astype(np.float32) if "u" in feats else None
        self.v = cache.read("v").astype(np.float32) if "v" in feats else None
        self.T = self.speed.shape[0]

        self.source = None
        if video_path and os.path.exists(video_path):
            from core.video import VideoSource
            try:
                self.source = VideoSource(video_path)
            except Exception:
                self.source = None

        # Threshold slider is in px/s; cap the range just past the bulk so the
        # slider's travel lands where the data actually is, not on a lone outlier.
        self.vmax = float(np.percentile(self.speed, 99.9)) or 1.0
        self.vmax = max(self.vmax, 1.0)
        self.threshold = self.vmax * 0.5
        self.roll_win_s = 0.5

        self.frame = 0
        self.playing = False

        self._build_ui()
        self._compute_static_series()
        self._recompute_threshold_series()
        self._update_frame(0)

    # -- layout --------------------------------------------------------------

    def _build_ui(self):
        self.setWindowTitle(
            f"Speed explorer -- {os.path.basename(self.meta.get('video_path', '?'))} "
            f"({self.T} frames, {self.ny}x{self.nx} = {self.n_blocks} blocks)")
        self.resize(1500, 900)
        root = QHBoxLayout(self)

        # ---- left: video + controls ----
        left = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setMinimumSize(720, 480)
        self.video_label.setStyleSheet("background:#000;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.video_label, 1)

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

        # rolling window slider
        rw_row = QHBoxLayout()
        self.rw_lbl = QLabel()
        self.rw_lbl.setMinimumWidth(190)
        rw_row.addWidget(self.rw_lbl)
        self.rw_slider = QSlider(Qt.Orientation.Horizontal)
        self.rw_slider.setRange(0, 40)         # 0.0 .. 2.0 s in 0.05 steps
        self.rw_slider.setValue(10)
        self.rw_slider.valueChanged.connect(self._on_roll_win)
        rw_row.addWidget(self.rw_slider, 1)
        left.addLayout(rw_row)

        info = QLabel(
            f"cache: {self.meta.get('backend', '?')} | fps {self.fps:.2f} | "
            f"block {self.block}px | downsample {self.meta.get('downsample', '?')}")
        info.setStyleSheet("color:#888;")
        left.addWidget(info)

        root.addLayout(left, 3)

        # ---- right: scrollable readouts ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        self.plot_col = QVBoxLayout(holder)
        self.plot_col.setSpacing(3)
        scroll.setWidget(holder)
        scroll.setMinimumWidth(430)
        root.addWidget(scroll, 2)

        self.plots: dict[str, MiniPlot] = {}

        def section(text: str):
            lbl = QLabel(text)
            lbl.setStyleSheet("color:#7fd7ff; font-weight:bold; padding-top:6px;")
            self.plot_col.addWidget(lbl)

        def add(key: str, title: str, unit: str, color=LINE):
            pl = MiniPlot(title, unit, color)
            pl.seek_requested.connect(self._seek)
            self.plots[key] = pl
            self.plot_col.addWidget(pl)

        section("Whole-frame speed distribution (all blocks, per frame)")
        add("mean", "Mean speed", "px/s")
        add("median", "Median speed", "px/s")
        add("p90", "90th pct speed", "px/s")
        add("p99", "99th pct speed", "px/s")
        add("max", "Max speed (single fastest block)", "px/s")
        add("sstd", "Spatial std of speed", "px/s")
        add("peak", "Max - median (peakedness)", "px/s")

        section("Temporal smoothing of mean speed (rolling-window slider)")
        add("roll_mean", "Rolling mean of mean speed", "px/s", LINE2)
        add("roll_std", "Rolling std of mean speed", "px/s", LINE2)

        section("Threshold-gated -- what detection actually sees (slider)")
        add("count", "# blocks > threshold", "blk", QColor(110, 230, 120))
        add("frac", "Fraction of blocks > threshold", "", QColor(110, 230, 120))
        add("clump", "Largest connected clump > threshold", "blk",
            QColor(110, 230, 120))
        add("cond_mean", "Mean speed OF blocks > threshold", "px/s",
            QColor(110, 230, 120))
        add("energy", "Total speed summed over blocks > threshold", "px/s",
            QColor(110, 230, 120))
        self.plot_col.addStretch(1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        # Debounce for the cheap threshold series so dragging stays smooth on a
        # full-clip cache (the sum over ~100M elements is ~100 ms).
        self._thr_debounce = QTimer(self)
        self._thr_debounce.setSingleShot(True)
        self._thr_debounce.setInterval(120)
        self._thr_debounce.timeout.connect(self._recompute_threshold_series)

        self._sync_labels()

    # -- static (threshold-independent) series -------------------------------

    def _compute_static_series(self):
        s = self.speed.reshape(self.T, -1)
        self.plots["mean"].set_series(s.mean(1))
        self.plots["median"].set_series(np.median(s, 1))
        self.plots["p90"].set_series(np.percentile(s, 90, axis=1))
        self.plots["p99"].set_series(np.percentile(s, 99, axis=1))
        self.plots["max"].set_series(s.max(1))
        self.plots["sstd"].set_series(s.std(1))
        self.plots["peak"].set_series(s.max(1) - np.median(s, 1))
        self._mean_series = s.mean(1)
        self._recompute_rolling()

    def _recompute_rolling(self):
        win = max(1, int(round(self.roll_win_s * self.fps)))
        x = self._mean_series
        if win <= 1:
            self.plots["roll_mean"].set_series(x)
            self.plots["roll_std"].set_series(np.zeros_like(x))
            return
        pad = win // 2
        xp = np.pad(x, (pad, win - 1 - pad), mode="edge")
        cs = np.concatenate([[0.0], np.cumsum(xp, dtype=np.float64)])
        rmean = (cs[win:] - cs[:-win]) / win
        cs2 = np.concatenate([[0.0], np.cumsum(xp.astype(np.float64) ** 2)])
        rmsq = (cs2[win:] - cs2[:-win]) / win
        rstd = np.sqrt(np.maximum(rmsq - rmean ** 2, 0.0))
        self.plots["roll_mean"].set_series(rmean.astype(np.float32))
        self.plots["roll_std"].set_series(rstd.astype(np.float32))

    # -- threshold-dependent series ------------------------------------------

    def _recompute_threshold_series(self):
        thr = self.threshold
        mask = self.speed > thr                      # (T, ny, nx)
        flat = mask.reshape(self.T, -1)
        count = flat.sum(1).astype(np.float32)
        self.plots["count"].set_series(count)
        self.plots["frac"].set_series(count / max(1, self.n_blocks))

        masked = np.where(mask, self.speed, 0.0)
        energy = masked.reshape(self.T, -1).sum(1)
        self.plots["energy"].set_series(energy.astype(np.float32))
        with np.errstate(invalid="ignore", divide="ignore"):
            cond = np.where(count > 0, energy / np.maximum(count, 1), 0.0)
        self.plots["cond_mean"].set_series(cond.astype(np.float32))
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
        thr = self.threshold
        largest = np.zeros(self.T, np.float32)
        for t in range(self.T):
            m = (self.speed[t] > thr).astype(np.uint8)
            if not m.any():
                continue
            n_lab, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if n_lab > 1:
                largest[t] = float(stats[1:, cv2.CC_STAT_AREA].max())
        self.plots["clump"].set_series(largest)

    # -- control handlers ----------------------------------------------------

    def _sync_labels(self):
        self.thr_lbl.setText(f"Speed threshold: {self.threshold:.2f} px/s")
        self.rw_lbl.setText(f"Rolling window: {self.roll_win_s:.2f} s")

    def _on_threshold(self, v: int):
        self.threshold = v / 1000.0 * self.vmax
        self._sync_labels()
        self._thr_debounce.start()
        self._redraw_video()          # highlight overlay follows immediately

    def _on_roll_win(self, v: int):
        self.roll_win_s = v * 0.05
        self._sync_labels()
        self._recompute_rolling()

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start(int(1000 / max(1.0, self.fps)))
        else:
            self.timer.stop()

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
        self.frame = max(0, min(int(frame), self.T - 1))
        if self.scrub.value() != self.frame:
            self.scrub.blockSignals(True)
            self.scrub.setValue(self.frame)
            self.scrub.blockSignals(False)
        self.time_lbl.setText(f"{self.frame / self.fps:.2f} s  (#{self.frame})")
        for pl in self.plots.values():
            pl.set_cursor(self.frame)
        self._redraw_video()

    # -- video + overlay -----------------------------------------------------

    def _base_frame(self) -> np.ndarray:
        """Working-resolution BGR frame cropped to the block grid, or a black
        canvas if there is no video."""
        ch, cw = self.ny * self.block, self.nx * self.block
        if self.source is None:
            return np.zeros((ch, cw, 3), np.uint8)
        bgr = self.source.frame_at(self.frame)
        if bgr is None:
            return np.zeros((ch, cw, 3), np.uint8)
        bgr = cv2.resize(bgr, (self.work_w, self.work_h),
                         interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(bgr[:ch, :cw])

    def _redraw_video(self):
        base = self._base_frame()
        ch, cw = base.shape[:2]
        sp = self.speed[self.frame]                       # (ny, nx)
        mode = self.overlay_mode.currentText()

        out = base.copy()
        if mode == "Speed heatmap":
            norm = np.clip(sp / self.vmax, 0, 1)
            heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            heat = cv2.resize(heat, (cw, ch), interpolation=cv2.INTER_NEAREST)
            out = cv2.addWeighted(base, 0.45, heat, 0.55, 0)
        elif mode == "Flow direction (HSV)" and self.u is not None:
            u, v = self.u[self.frame], self.v[self.frame]
            ang = (np.degrees(np.arctan2(v, u)) % 360) / 2.0
            mag = np.clip(sp / self.vmax, 0, 1)
            hsv = np.zeros((self.ny, self.nx, 3), np.uint8)
            hsv[..., 0] = ang.astype(np.uint8)
            hsv[..., 1] = 255
            hsv[..., 2] = (mag * 255).astype(np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            rgb = cv2.resize(rgb, (cw, ch), interpolation=cv2.INTER_NEAREST)
            out = cv2.addWeighted(base, 0.4, rgb, 0.6, 0)
        elif mode == "Flow vectors" and self.u is not None:
            u, v = self.u[self.frame], self.v[self.frame]
            step = 2
            for by in range(0, self.ny, step):
                for bx in range(0, self.nx, step):
                    x0 = int((bx + 0.5) * self.block)
                    y0 = int((by + 0.5) * self.block)
                    scale = self.block / max(self.vmax, 1e-6) * 1.5
                    x1 = int(x0 + u[by, bx] * scale)
                    y1 = int(y0 + v[by, bx] * scale)
                    cv2.arrowedLine(out, (x0, y0), (x1, y1),
                                    (80, 220, 255), 1, tipLength=0.35)

        if self.hi_chk.isChecked():
            m = (sp > self.threshold).astype(np.uint8)
            mm = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_NEAREST)
            green = np.zeros_like(out)
            green[..., 1] = 255
            sel = mm.astype(bool)
            out[sel] = cv2.addWeighted(out, 0.5, green, 0.5, 0)[sel]
            contours, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (60, 255, 60), 1)

        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def resizeEvent(self, _):
        self._redraw_video()


# -- cache picking + entry point ---------------------------------------------

def _pick_cache(cache_root: str) -> str | None:
    caches = cache_mod.list_caches(cache_root)
    if not caches:
        print(f"No caches under {cache_root}. Build one in the main app (Tab 1) "
              f"first.")
        return None
    caches.sort(key=lambda c: c.get("duration_s", 0))
    labels = []
    for c in caches:
        vid = os.path.basename(c.get("video_path", "?"))
        tag = "test" if c.get("test_mode") else "full"
        labels.append(f"{c['key']}  |  {vid}  |  {c.get('n_frames', '?')} fr  "
                      f"|  {c['grid'][0]}x{c['grid'][1]}  |  {tag}")
    app = QApplication.instance() or QApplication(sys.argv)
    choice, ok = QInputDialog.getItem(
        None, "Open a feature cache", "Cache to explore:", labels, 0, False)
    if not ok:
        return None
    return caches[labels.index(choice)]["key"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", help="cache key under --cache-root")
    ap.add_argument("--cache-root", default=os.path.join(".", ".cache"))
    ap.add_argument("--video", help="override the video path stored in the cache")
    args = ap.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)

    key = args.cache or _pick_cache(args.cache_root)
    if not key:
        return
    cache = cache_mod.open_cache(args.cache_root, key)
    video_path = args.video or cache.meta.get("video_path")

    win = SpeedExplorer(cache, video_path)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
