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
from PyQt6.QtWidgets import (QSizePolicy, QWidget)

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


