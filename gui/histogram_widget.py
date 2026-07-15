"""The Lightroom-style range histogram. The core interaction of the whole tool.

Each widget shows one feature's distribution with a draggable "keep this range"
band. Two densities are drawn on top of each other:

  * dim grey  -- the unconditional distribution of this feature.
  * bright    -- the CROSS-FILTERED distribution: only the data that survives
                 every OTHER feature's range.

The gap between the two is the entire point. If the bright curve collapses to a
narrow mode well away from the bulk of the grey one, the other filters have
already isolated something and this feature will separate it cleanly. If bright
just tracks grey scaled down, this feature is telling you nothing that the others
have not already said, and you are only shrinking your sample.

Conventions (borrowed from the reference color detector so the two tools feel the
same): dark plot, dashed white boundary lines, dimmed regions outside the
selection, drag a boundary to move it, drag inside the band to slide the whole
range.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRect, Qt, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QFont, QMouseEvent, QPainter, QPen,
                         QPolygon, QWheelEvent)
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QSizePolicy, QWidget

BG = QColor(30, 30, 30)
PLOT_BG = QColor(0, 0, 0)
TOTAL_FILL = QColor(105, 105, 105, 110)
FILTERED_FILL = QColor(120, 215, 255, 200)
BOUND_PEN = QColor(255, 255, 255, 170)
DIM = QColor(0, 0, 0, 130)


class RangeHistogram(QWidget):
    """One feature, one histogram, one draggable range."""

    range_changed = pyqtSignal(str, float, float)
    enabled_changed = pyqtSignal(str, bool)
    remove_requested = pyqtSignal(str)

    MARGIN_L = 8
    MARGIN_R = 8
    MARGIN_T = 18
    MARGIN_B = 16
    HIT = 6

    def __init__(self, name: str, label: str, units: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.label = label
        self.units = units

        self.edges = np.linspace(0, 1, 129, dtype=np.float32)
        self.total = np.zeros(128, dtype=np.float32)
        self.filtered = np.zeros(128, dtype=np.float32)

        self.lo = 0.0
        self.hi = 1.0
        self.is_enabled = True

        self._drag: str | None = None
        self._drag_anchor: tuple[float, float, float] | None = None

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(96)
        self.setToolTip(label)

    # -- data ----------------------------------------------------------------

    def set_data(self, edges: np.ndarray, total: np.ndarray,
                 filtered: np.ndarray) -> None:
        self.edges = edges
        self.total = total.astype(np.float32)
        self.filtered = filtered.astype(np.float32)
        self.update()

    def set_range(self, lo: float, hi: float) -> None:
        self.lo, self.hi = float(lo), float(hi)
        self.update()

    def reset_range_to_full(self) -> None:
        self.set_range(float(self.edges[0]), float(self.edges[-1]))
        self.range_changed.emit(self.name, self.lo, self.hi)

    def set_enabled_filter(self, on: bool) -> None:
        self.is_enabled = on
        self.update()

    # -- geometry ------------------------------------------------------------

    def _plot(self) -> QRect:
        return QRect(self.MARGIN_L, self.MARGIN_T,
                     max(1, self.width() - self.MARGIN_L - self.MARGIN_R),
                     max(1, self.height() - self.MARGIN_T - self.MARGIN_B))

    def _x_of(self, val: float) -> float:
        r = self._plot()
        lo, hi = float(self.edges[0]), float(self.edges[-1])
        if hi <= lo:
            return r.left()
        return r.left() + (val - lo) / (hi - lo) * r.width()

    def _val_of(self, px: float) -> float:
        r = self._plot()
        lo, hi = float(self.edges[0]), float(self.edges[-1])
        frac = np.clip((px - r.left()) / max(1, r.width()), 0, 1)
        return float(lo + frac * (hi - lo))

    # -- painting ------------------------------------------------------------

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), BG)
        r = self._plot()
        p.fillRect(r, PLOT_BG)

        n = len(self.total)
        if n == 0 or r.width() <= 0:
            p.end()
            return

        # Shared vertical scale so the bright curve is visibly a SUBSET of the
        # dim one. Normalizing them independently would make a filter that keeps
        # 0.1% of the data look identical to one that keeps everything -- which
        # is precisely the thing the user needs to be able to tell apart.
        peak = float(self.total.max()) or 1.0

        # A log-ish scale, because flow feature histograms have one huge mode at
        # near-zero motion and the interesting populations are three or four
        # orders of magnitude down. On a linear axis they are invisible.
        def to_poly(counts: np.ndarray) -> QPolygon:
            h = np.log1p(counts) / np.log1p(peak)
            pts = [QPoint(r.left(), r.bottom())]
            for i in range(n):
                x = int(r.left() + (i + 0.5) / n * r.width())
                y = int(r.bottom() - h[i] * r.height())
                pts.append(QPoint(x, y))
            pts.append(QPoint(r.right(), r.bottom()))
            return QPolygon(pts)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(TOTAL_FILL))
        p.drawPolygon(to_poly(self.total))

        if self.filtered.max() > 0:
            p.setBrush(QBrush(FILTERED_FILL))
            p.drawPolygon(to_poly(self.filtered))

        lo_px, hi_px = int(self._x_of(self.lo)), int(self._x_of(self.hi))

        if self.is_enabled:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(DIM))
            p.drawRect(QRect(r.left(), r.top(), max(0, lo_px - r.left()),
                             r.height()))
            p.drawRect(QRect(hi_px, r.top(), max(0, r.right() - hi_px),
                             r.height()))

            pen = QPen(BOUND_PEN, 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(lo_px, r.top(), lo_px, r.bottom())
            p.drawLine(hi_px, r.top(), hi_px, r.bottom())

        p.setPen(QColor(120, 120, 120))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)

        p.setFont(QFont("Segoe UI", 7))
        title = self.label + (f"  ({self.units})" if self.units else "")
        p.setPen(QColor(210, 210, 210) if self.is_enabled else QColor(110, 110, 110))
        p.drawText(r.left(), r.top() - 5, title)

        pct = 0.0
        if self.total.sum() > 0:
            pct = 100.0 * self.filtered.sum() / self.total.sum()
        p.setPen(QColor(150, 150, 150))
        p.drawText(r.right() - 42, r.top() - 5, f"{pct:5.1f}%")

        p.setFont(QFont("Consolas", 7))
        p.setPen(QColor(190, 190, 190))
        p.drawText(r.left(), r.bottom() + 12, f"{self.lo:.4g}")
        txt = f"{self.hi:.4g}"
        p.drawText(r.right() - 7 * len(txt), r.bottom() + 12, txt)
        p.end()

    # -- interaction ---------------------------------------------------------

    def _hit(self, x: int) -> str | None:
        lo_px, hi_px = self._x_of(self.lo), self._x_of(self.hi)
        if abs(x - lo_px) <= self.HIT:
            return "lo"
        if abs(x - hi_px) <= self.HIT:
            return "hi"
        if lo_px < x < hi_px:
            return "band"
        return None

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.MouseButton.RightButton:
            self.reset_range_to_full()
            return
        if e.button() == Qt.MouseButton.MiddleButton:
            self.remove_requested.emit(self.name)
            return
        self._drag = self._hit(e.pos().x())
        if self._drag == "band":
            self._drag_anchor = (self._val_of(e.pos().x()), self.lo, self.hi)

    def mouseMoveEvent(self, e: QMouseEvent):
        if self._drag is None:
            h = self._hit(e.pos().x())
            self.setCursor(
                Qt.CursorShape.SizeHorCursor if h in ("lo", "hi")
                else Qt.CursorShape.OpenHandCursor if h == "band"
                else Qt.CursorShape.ArrowCursor)
            return

        v = self._val_of(e.pos().x())
        if self._drag == "lo":
            self.lo = min(v, self.hi)
        elif self._drag == "hi":
            self.hi = max(v, self.lo)
        elif self._drag == "band" and self._drag_anchor:
            v0, lo0, hi0 = self._drag_anchor
            d = v - v0
            lo_lim, hi_lim = float(self.edges[0]), float(self.edges[-1])
            width = hi0 - lo0
            new_lo = np.clip(lo0 + d, lo_lim, hi_lim - width)
            self.lo = float(new_lo)
            self.hi = float(new_lo + width)
        self.update()
        self.range_changed.emit(self.name, self.lo, self.hi)

    def mouseReleaseEvent(self, _):
        self._drag = None
        self._drag_anchor = None

    def wheelEvent(self, e: QWheelEvent):
        """Scroll widens/narrows the band around its center -- much faster than
        dragging both edges when you are hunting for a threshold."""
        center = 0.5 * (self.lo + self.hi)
        half = 0.5 * (self.hi - self.lo)
        factor = 0.9 if e.angleDelta().y() > 0 else 1.1
        half = max(1e-9, half * factor)
        lo_lim, hi_lim = float(self.edges[0]), float(self.edges[-1])
        self.lo = max(lo_lim, center - half)
        self.hi = min(hi_lim, center + half)
        self.update()
        self.range_changed.emit(self.name, self.lo, self.hi)

    def mouseDoubleClickEvent(self, _):
        self.is_enabled = not self.is_enabled
        self.enabled_changed.emit(self.name, self.is_enabled)
        self.update()
