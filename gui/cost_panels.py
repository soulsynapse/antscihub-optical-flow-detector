"""Reusable pieces of the lever-decision dialogs (downsample, block size).

``downsample`` and ``block_size`` are different levers buying different things,
but the *decision* has the same shape for both: read explanatory prose, see a
frontier of cost against what you give up, find the point past which you give up
resolution for nothing, and read a concrete projection for your own corpus. So
the machinery lives here and each dialog supplies only what is lever-specific.

Deliberately absent: any single number summarizing "how much worse". Everything
these widgets render is measured wall clock, arithmetic storage, or a named
setting. The withdrawn ``sig_corr`` reading in todo.md is the cautionary case --
an aggregate that looked authoritative and did not mean what it appeared to.
"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget

_BG = QColor("#1b1f24")
_AXIS = QColor("#5b6672")
_TEXT = QColor("#c8d2dc")
_CURVE = QColor("#4ac6ff")
_KNEE = QColor("#ffb347")
_SELECTED = QColor("#7ee787")
_FLAT = QColor("#3a4450")     # the region past the knee, shaded as "buys nothing"


class LeverPreamble(QLabel):
    """The explanatory prose a lever dialog must show on open.

    Batch M requires this in the window itself, as real prose, not a tooltip:
    without the *first* half users refuse the lever on principle and quietly make
    large projects infeasible; without the *second* they read the tool as
    endorsing a cheaper default and accept silent degradation. Neither half works
    alone, which is why this is one widget taking both and not an optional
    string.
    """

    def __init__(self, why_it_is_offered: str, why_it_is_not_assumed: str,
                 parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setText(f"<p style='margin:0 0 8px 0;'>{why_it_is_offered}</p>"
                     f"<p style='margin:0;'>{why_it_is_not_assumed}</p>")
        self.setStyleSheet(
            "color:#c8d2dc; background:#232a31; border:1px solid #33404d;"
            "border-radius:4px; padding:10px; font-size:12px;")


class FrontierPlot(QWidget):
    """Projected cost against a lever setting, with the knee marked.

    The knee is the single most valuable thing on screen and is invisible
    everywhere else in the tool: cost has a floor this lever cannot go below, so
    past some setting the user gives up resolution and buys nothing. Nobody can
    guess where that is -- measured, 1300 -> 650 px was a 4x pixel cut for ~13%
    wall time. Marking it turns "how brave am I" into "here is the frontier, pick
    a point on it".

    Generic over the lever: ``values`` are settings on the x axis and ``costs``
    the projected cost of each, so Batch N's block-size dialog reuses this with
    storage on y. Click or drag to pick a setting.
    """
    picked = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self._values: list[float] = []
        self._costs: list[float] = []
        self._labels: list[str] = []
        self._knee: float | None = None
        self._selected: float | None = None
        self._y_label = ""

    def set_curve(self, values, costs, *, labels=None, knee=None, selected=None,
                  y_label=""):
        self._values = list(values)
        self._costs = list(costs)
        self._labels = list(labels) if labels else [f"{v:g}" for v in self._values]
        self._knee = knee
        self._selected = selected
        self._y_label = y_label
        self.update()

    # -- geometry ------------------------------------------------------------
    def _plot_rect(self):
        return 54, 10, max(1, self.width() - 66), max(1, self.height() - 44)

    def _x_of(self, v: float) -> float:
        x, _, w, _ = self._plot_rect()
        lo, hi = min(self._values), max(self._values)
        if hi == lo:
            return x + w / 2
        return x + w * (v - lo) / (hi - lo)

    def _y_of(self, c: float) -> float:
        _, y, _, h = self._plot_rect()
        hi = max(self._costs) if self._costs else 1.0
        hi = hi if hi > 0 else 1.0
        return y + h * (1.0 - c / hi)      # baseline at zero, so ratios read true

    def _value_at_x(self, px: float) -> float:
        x, _, w, _ = self._plot_rect()
        lo, hi = min(self._values), max(self._values)
        frac = min(1.0, max(0.0, (px - x) / w))
        target = lo + frac * (hi - lo)
        return min(self._values, key=lambda v: abs(v - target))

    # -- interaction ---------------------------------------------------------
    def mousePressEvent(self, ev):
        self._pick(ev)

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.MouseButton.LeftButton:
            self._pick(ev)

    def _pick(self, ev):
        if not self._values:
            return
        v = self._value_at_x(ev.position().x())
        if v != self._selected:
            self._selected = v
            self.update()
            self.picked.emit(v)

    # -- paint ---------------------------------------------------------------
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), _BG)
        if len(self._values) < 2:
            p.setPen(_AXIS)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "measuring…")
            return

        x0, y0, w, h = self._plot_rect()
        small = QFont(self.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1.5))
        p.setFont(small)

        # Shade the region past the knee: this is the "you pay resolution and get
        # almost nothing" zone, and it reads faster as an area than as a line.
        if self._knee is not None:
            kx = self._x_of(self._knee)
            lo_x = min(kx, self._x_of(min(self._values)))
            p.fillRect(int(lo_x), y0, int(abs(kx - lo_x)), h, _FLAT)

        p.setPen(QPen(_AXIS, 1))
        p.drawLine(x0, y0, x0, y0 + h)
        p.drawLine(x0, y0 + h, x0 + w, y0 + h)

        # y ticks: zero and the max, enough to read the ratio off the curve.
        hi = max(self._costs)
        p.setPen(_TEXT)
        for frac in (0.0, 0.5, 1.0):
            yy = y0 + h * (1.0 - frac)
            p.drawText(0, int(yy) - 7, x0 - 6, 14,
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                       self._fmt_y(hi * frac))

        path = QPainterPath()
        for i, (v, c) in enumerate(zip(self._values, self._costs)):
            pt = QPointF(self._x_of(v), self._y_of(c))
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        p.setPen(QPen(_CURVE, 2))
        p.drawPath(path)

        for v, c, lab in zip(self._values, self._costs, self._labels):
            cx, cy = self._x_of(v), self._y_of(c)
            sel = self._selected is not None and abs(v - self._selected) < 1e-9
            p.setPen(QPen(_SELECTED if sel else _CURVE, 1))
            p.setBrush(_SELECTED if sel else _CURVE)
            p.drawEllipse(QPointF(cx, cy), 5 if sel else 3.5, 5 if sel else 3.5)
            p.setPen(_SELECTED if sel else _TEXT)
            p.drawText(int(cx) - 24, y0 + h + 4, 48, 16,
                       int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
                       lab)

        if self._knee is not None:
            kx = self._x_of(self._knee)
            p.setPen(QPen(_KNEE, 1, Qt.PenStyle.DashLine))
            p.drawLine(int(kx), y0, int(kx), y0 + h)
            p.setPen(_KNEE)
            # Flip the annotation to the left of the line when the knee sits far
            # enough right that the text would run off the plot -- a knee near
            # 1.0 is common on decode-heavy footage, and clipping the "save
            # almost no time" line loses the entire point of the marker.
            tw, th = 150, 46
            tx = (kx + 4) if (kx + 4 + tw) <= (x0 + w) else (kx - 4 - tw)
            align = (Qt.AlignmentFlag.AlignLeft if tx > kx
                     else Qt.AlignmentFlag.AlignRight)
            p.drawText(int(tx), y0 + 2, tw, th,
                       int(align | Qt.AlignmentFlag.AlignTop),
                       f"knee {self._knee:.2f}\nbelow: lose detail,\n"
                       f"save almost no time")

        p.setPen(_TEXT)
        p.drawText(x0 + 4, 0, w, 14,
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
                   self._y_label)

    def _fmt_y(self, v: float) -> str:
        if v >= 100:
            return f"{v:.0f}"
        if v >= 10:
            return f"{v:.0f}"
        return f"{v:.1f}"
