"""The ROI inspection panel, shared by Tabs 2 and 3.

Shows, for the selected ROI: per-feature time series (mean over the ROI's
blocks), the PSD of speed, and -- in Tab 3 -- the binary behavior traces plus a
readout of which constraints are met right now.

Custom-painted rather than matplotlib: these redraw on every scrub, and a
matplotlib canvas cannot keep up with that without a lot of blitting machinery.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygon
from PyQt6.QtWidgets import (QLabel, QSizePolicy, QVBoxLayout, QWidget)

BG = QColor(24, 24, 24)
GRID = QColor(55, 55, 55)
TRACE = QColor(120, 215, 255)
CURSOR = QColor(255, 210, 80)
NYQ = QColor(230, 100, 100, 160)


class TimeSeriesPlot(QWidget):
    """One feature's time series with a playhead. Click to seek."""

    seek_requested = pyqtSignal(float)   # seconds

    def __init__(self, title: str = "", height: int = 74):
        super().__init__()
        self.title = title
        self.x = np.zeros(0, np.float32)
        self.y = np.zeros(0, np.float32)
        self.cursor_s = 0.0
        self.bands: list[tuple[float, float, str]] = []   # shaded (t0,t1,color)
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_series(self, x_s: np.ndarray, y: np.ndarray, title: str | None = None):
        self.x = np.asarray(x_s, np.float32)
        self.y = np.asarray(y, np.float32)
        if title is not None:
            self.title = title
        self.update()

    def set_bands(self, bands):
        self.bands = bands
        self.update()

    def set_cursor(self, t_s: float):
        self.cursor_s = t_s
        self.update()

    def _rect(self) -> QRect:
        return self.rect().adjusted(4, 14, -4, -4)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), BG)
        r = self._rect()
        if self.x.size < 2:
            p.setPen(QColor(110, 110, 110))
            p.setFont(QFont("Segoe UI", 7))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no data")
            p.end()
            return

        t0, t1 = float(self.x[0]), float(self.x[-1])
        span = max(1e-6, t1 - t0)
        lo, hi = float(np.nanmin(self.y)), float(np.nanmax(self.y))
        if hi <= lo:
            hi = lo + 1.0

        for b0, b1, col in self.bands:
            x0 = r.left() + (b0 - t0) / span * r.width()
            x1 = r.left() + (b1 - t0) / span * r.width()
            c = QColor(col)
            c.setAlpha(70)
            p.fillRect(QRect(int(x0), r.top(), max(1, int(x1 - x0)), r.height()), c)

        p.setPen(QPen(GRID, 1))
        p.drawRect(r)

        # Decimate to at most one point per pixel column. A 30k-point polyline
        # redrawn on every scrub is what makes naive Qt plots stutter.
        n = self.x.size
        step = max(1, n // max(1, r.width()))
        pts = []
        for i in range(0, n, step):
            px = r.left() + (self.x[i] - t0) / span * r.width()
            py = r.bottom() - (self.y[i] - lo) / (hi - lo) * r.height()
            pts.append(QPoint(int(px), int(py)))
        p.setPen(QPen(TRACE, 1))
        p.drawPolyline(QPolygon(pts))

        cx = r.left() + (self.cursor_s - t0) / span * r.width()
        p.setPen(QPen(CURSOR, 1))
        p.drawLine(int(cx), r.top(), int(cx), r.bottom())

        p.setFont(QFont("Consolas", 7))
        p.setPen(QColor(190, 190, 190))
        p.drawText(6, 11, f"{self.title}   [{lo:.3g} .. {hi:.3g}]")
        p.end()

    def mousePressEvent(self, e):
        if self.x.size < 2:
            return
        r = self._rect()
        frac = np.clip((e.pos().x() - r.left()) / max(1, r.width()), 0, 1)
        t0, t1 = float(self.x[0]), float(self.x[-1])
        self.seek_requested.emit(t0 + frac * (t1 - t0))


class RangePlot(QWidget):
    """A feature's time series with two draggable horizontal lines (lo, hi) that
    set a behavior constraint directly on the data.

    This replaces the histogram as the threshold editor. You see the selected
    replicate's actual signal, and you drag the dashed lines to bracket the part
    you want -- the segments where the signal sits inside the band are highlighted
    on the time axis, so you get immediate feedback on what this one constraint
    catches. Drag near the top line to move hi, near the bottom to move lo, or in
    the middle band to slide both.
    """

    range_changed = pyqtSignal(str, float, float)
    seek_requested = pyqtSignal(float)
    remove_requested = pyqtSignal(str)

    HIT = 6

    def __init__(self, feature: str, title: str, color: str = "#69d7ff",
                 height: int = 96):
        super().__init__()
        self.feature = feature
        self.title = title
        self.accent = QColor(color)
        self.x = np.zeros(0, np.float32)
        self.y = np.zeros(0, np.float32)              # median, for hit-testing
        self.block_vals = np.zeros((0, 0), np.float32)  # (T, k) per-block
        self._env = None                              # (p10, p50, p90, max)
        self.lo = 0.0
        self.hi = 1.0
        self.cursor_s = 0.0
        self.active = True
        self.behavior_bands: list[tuple[float, float, str]] = []
        self._y_lo = 0.0
        self._y_hi = 1.0
        self._fixed_yrange: tuple[float, float] | None = None
        self._drag: str | None = None
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip("Drag the dashed lines to set this feature's range. "
                        "Right-click to remove this constraint.")

    def set_series(self, x_s, y):
        """Single-series compatibility: treat as one block."""
        self.set_blocks(x_s, np.asarray(y, np.float32)[:, None])

    def set_blocks(self, x_s, block_vals):
        """Per-block values (T, k). The plot shows the distribution of blocks --
        a shaded p10-p90 envelope, the median, and a faint max line -- so the
        range you drag brackets the block values detection actually thresholds,
        not the box mean."""
        self.x = np.asarray(x_s, np.float32)
        self.block_vals = np.asarray(block_vals, np.float32)
        if self.block_vals.ndim == 1:
            self.block_vals = self.block_vals[:, None]
        if self.block_vals.size:
            p10, p50, p90 = np.percentile(self.block_vals, [10, 50, 90], axis=1)
            mx = self.block_vals.max(axis=1)
            self._env = (p10.astype(np.float32), p50.astype(np.float32),
                         p90.astype(np.float32), mx.astype(np.float32))
            self.y = p50.astype(np.float32)
        else:
            self._env = None
            self.y = np.zeros(self.x.size, np.float32)
        if self._fixed_yrange is None and self._env is not None:
            lo = float(np.nanmin(self._env[0]))
            hi = float(np.nanmax(self._env[3]))
            self._y_lo, self._y_hi = lo, (hi if hi > lo else lo + 1.0)
        self.update()

    def set_yrange(self, lo: float, hi: float):
        """Pin the y-axis (shared across replicates)."""
        self._fixed_yrange = (float(lo), float(hi))
        self._y_lo, self._y_hi = float(lo), (float(hi) if hi > lo else lo + 1.0)
        self.update()

    def set_range(self, lo, hi):
        self.lo, self.hi = float(lo), float(hi)
        self.update()

    def set_cursor(self, t_s):
        self.cursor_s = t_s
        self.update()

    def set_active(self, active: bool):
        """Dim when the measure is not part of the AND (shown for reference)."""
        self.active = active
        self.update()

    def set_behavior_bands(self, bands):
        self.behavior_bands = bands
        self.update()

    def _rect(self) -> QRect:
        return self.rect().adjusted(4, 14, -46, -4)

    def _y_to_px(self, val):
        r = self._rect()
        span = self._y_hi - self._y_lo or 1.0
        v = np.clip(val, self._y_lo, self._y_hi) if np.isfinite(val) else self._y_hi
        return r.bottom() - (v - self._y_lo) / span * r.height()

    def _px_to_y(self, py):
        r = self._rect()
        frac = np.clip((r.bottom() - py) / max(1, r.height()), 0, 1)
        return float(self._y_lo + frac * (self._y_hi - self._y_lo))

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), BG if self.active else QColor(18, 18, 18))
        if not self.active:
            p.setOpacity(0.6)
        r = self._rect()
        if self.x.size < 2:
            p.setPen(QColor(110, 110, 110))
            p.setFont(QFont("Segoe UI", 7))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "select a replicate")
            p.end()
            return

        t0, t1 = float(self.x[0]), float(self.x[-1])
        span = max(1e-6, t1 - t0)

        # Behavior-active shading (all constraints met), behind everything.
        for b0, b1, col in self.behavior_bands:
            x0 = r.left() + (b0 - t0) / span * r.width()
            x1 = r.left() + (b1 - t0) / span * r.width()
            c = QColor(col)
            c.setAlpha(60)
            p.fillRect(QRect(int(x0), r.top(), max(1, int(x1 - x0)), r.height()), c)

        lo_px, hi_px = self._y_to_px(self.lo), self._y_to_px(self.hi)
        # Shade the keep band across the plot width.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self.accent.red(), self.accent.green(),
                          self.accent.blue(), 28))
        p.drawRect(QRect(r.left(), int(min(lo_px, hi_px)),
                         r.width(), int(abs(hi_px - lo_px))))

        p.setPen(QPen(GRID, 1))
        p.drawRect(r)

        n = self.x.size
        step = max(1, n // max(1, r.width()))
        idxs = list(range(0, n, step))
        xpix = [int(r.left() + (self.x[i] - t0) / span * r.width()) for i in idxs]

        if self._env is not None:
            p10, p50, p90, mx = self._env
            # p10-p90 envelope (the bulk of the blocks), shaded.
            top = [QPoint(xpix[j], int(self._y_to_px(p90[i])))
                   for j, i in enumerate(idxs)]
            bot = [QPoint(xpix[j], int(self._y_to_px(p10[i])))
                   for j, i in enumerate(idxs)]
            poly = QPolygon(top + bot[::-1])
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(120, 215, 255, 45))
            p.drawPolygon(poly)
            # Faint MAX line -- the peak block, which is what a "detect high"
            # threshold actually keys on.
            p.setPen(QPen(QColor(120, 215, 255, 90), 1))
            p.drawPolyline(QPolygon(
                [QPoint(xpix[j], int(self._y_to_px(mx[i])))
                 for j, i in enumerate(idxs)]))
            # Median, bright.
            p.setPen(QPen(TRACE, 1))
            p.drawPolyline(QPolygon(
                [QPoint(xpix[j], int(self._y_to_px(p50[i])))
                 for j, i in enumerate(idxs)]))

            # Bottom strip: fraction of blocks IN the current [lo,hi], per column.
            # This is exactly what feeds the ethogram strength, so the plot and
            # the detection agree by construction.
            base = r.bottom() - 1
            bv = self.block_vals
            for j, i in enumerate(idxs):
                col = bv[i]
                frac = float(((col >= self.lo) & (col <= self.hi)).mean()) \
                    if col.size else 0.0
                if frac <= 0.01:
                    continue
                cc = QColor(self.accent)
                cc.setAlpha(int(40 + 160 * frac))
                p.setPen(QPen(cc, 2))
                p.drawLine(xpix[j], base, xpix[j], base - int(3 + 7 * frac))

        # The draggable lo/hi lines.
        for py, tag in ((hi_px, "hi"), (lo_px, "lo")):
            p.setPen(QPen(self.accent, 1, Qt.PenStyle.DashLine))
            p.drawLine(r.left(), int(py), r.right(), int(py))
            p.setPen(QColor(230, 230, 230))
            p.setFont(QFont("Consolas", 7))
            val = self.hi if tag == "hi" else self.lo
            txt = "inf" if not np.isfinite(val) else f"{val:.3g}"
            p.drawText(r.right() + 2, int(py) + 3, txt)

        # Playhead.
        cx = r.left() + (self.cursor_s - t0) / span * r.width()
        p.setPen(QPen(CURSOR, 1))
        p.drawLine(int(cx), r.top(), int(cx), r.bottom())

        p.setFont(QFont("Consolas", 7))
        p.setPen(QColor(200, 200, 200))
        p.drawText(6, 11, f"{self.title}   [{self._y_lo:.3g} .. {self._y_hi:.3g}]")
        p.end()

    def _hit(self, py) -> str | None:
        lo_px, hi_px = self._y_to_px(self.lo), self._y_to_px(self.hi)
        if abs(py - hi_px) <= self.HIT:
            return "hi"
        if abs(py - lo_px) <= self.HIT:
            return "lo"
        if min(lo_px, hi_px) < py < max(lo_px, hi_px):
            return "band"
        return None

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            self.remove_requested.emit(self.feature)
            return
        if self.x.size < 2:
            return
        self._drag = self._hit(e.pos().y())
        if self._drag is None:
            # Click outside the band = seek.
            r = self._rect()
            frac = np.clip((e.pos().x() - r.left()) / max(1, r.width()), 0, 1)
            t0, t1 = float(self.x[0]), float(self.x[-1])
            self.seek_requested.emit(t0 + frac * (t1 - t0))
        else:
            self._band_anchor = (self._px_to_y(e.pos().y()), self.lo, self.hi)

    def mouseMoveEvent(self, e):
        if self._drag is None:
            hit = self._hit(e.pos().y())
            self.setCursor(Qt.CursorShape.SizeVerCursor if hit in ("lo", "hi", "band")
                           else Qt.CursorShape.ArrowCursor)
            return
        val = self._px_to_y(e.pos().y())
        if self._drag == "hi":
            self.hi = max(val, self.lo)
        elif self._drag == "lo":
            self.lo = min(val, self.hi)
        else:
            y0, lo0, hi0 = self._band_anchor
            d = val - y0
            self.lo, self.hi = lo0 + d, hi0 + d
        self.update()
        self.range_changed.emit(self.feature, self.lo, self.hi)

    def mouseReleaseEvent(self, _):
        self._drag = None


class PSDPlot(QWidget):
    """Power spectral density, log-y, with the Nyquist limit drawn explicitly.

    Nyquist is on the plot because it is the single most common way to get a
    frequency-domain result wrong: everything above it is folded back down and
    appears as a plausible peak somewhere below it.
    """

    def __init__(self):
        super().__init__()
        self.freqs = np.zeros(0)
        self.psd = np.zeros(0)
        self.nyquist = 0.0
        self.band: tuple[float, float] | None = None
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_psd(self, freqs, psd, nyquist: float,
                band: tuple[float, float] | None = None):
        self.freqs = np.asarray(freqs)
        self.psd = np.asarray(psd)
        self.nyquist = nyquist
        self.band = band
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), BG)
        r = self.rect().adjusted(4, 14, -4, -16)
        if self.freqs.size < 2:
            p.setPen(QColor(110, 110, 110))
            p.setFont(QFont("Segoe UI", 7))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "select an ROI")
            p.end()
            return

        f_max = float(self.freqs[-1])
        y = np.log10(self.psd + 1e-12)
        lo, hi = float(y.min()), float(y.max())
        if hi <= lo:
            hi = lo + 1

        if self.band:
            x0 = r.left() + self.band[0] / f_max * r.width()
            x1 = r.left() + self.band[1] / f_max * r.width()
            c = QColor(120, 215, 255, 45)
            p.fillRect(QRect(int(x0), r.top(), max(1, int(x1 - x0)), r.height()), c)

        p.setPen(QPen(GRID, 1))
        p.drawRect(r)

        pts = []
        for i in range(self.freqs.size):
            px = r.left() + self.freqs[i] / f_max * r.width()
            py = r.bottom() - (y[i] - lo) / (hi - lo) * r.height()
            pts.append(QPoint(int(px), int(py)))
        p.setPen(QPen(TRACE, 1))
        p.drawPolyline(QPolygon(pts))

        if self.nyquist > 0:
            nx = r.left() + min(self.nyquist, f_max) / f_max * r.width()
            p.setPen(QPen(NYQ, 1, Qt.PenStyle.DashLine))
            p.drawLine(int(nx), r.top(), int(nx), r.bottom())
            p.setFont(QFont("Consolas", 6))
            p.setPen(NYQ)
            p.drawText(int(nx) - 46, r.top() + 9, "Nyquist")

        # Peak, excluding DC.
        if self.freqs.size > 2:
            k = int(np.argmax(self.psd[1:]) + 1)
            px = r.left() + self.freqs[k] / f_max * r.width()
            p.setPen(QPen(CURSOR, 1))
            p.drawLine(int(px), r.top(), int(px), r.bottom())
            p.setFont(QFont("Consolas", 7))
            p.drawText(int(px) + 3, r.top() + 10, f"{self.freqs[k]:.1f} Hz")

        p.setFont(QFont("Consolas", 7))
        p.setPen(QColor(190, 190, 190))
        p.drawText(6, 11, "PSD of speed (log power)")
        p.setPen(QColor(140, 140, 140))
        p.drawText(r.left(), r.bottom() + 12, "0 Hz")
        txt = f"{f_max:.0f} Hz"
        p.drawText(r.right() - 7 * len(txt), r.bottom() + 12, txt)
        p.end()


class ConstraintList(QLabel):
    """Which leaves of the behavior spec are met at the current instant."""

    def __init__(self):
        super().__init__()
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setStyleSheet("font-family: Consolas; font-size: 11px;")
        self.setText("<i>no behavior selected</i>")

    def set_status(self, rows: list[tuple[str, bool, float]], behavior: str):
        if not rows:
            self.setText("<i>no constraints</i>")
            return
        html = [f"<b>{behavior}</b> at this instant:<br>"]
        for desc, met, val in rows:
            colour = "#7ab87a" if met else "#e06c6c"
            tick = "PASS" if met else "FAIL"
            html.append(
                f"<span style='color:{colour}'>{tick}</span> {desc} "
                f"<span style='color:#999'>(now {val:.4g})</span><br>")
        self.setText("".join(html))
