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
from PyQt6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)

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


class EvidencePanel(QWidget):
    """Run the tuned detector at each candidate setting and show what it found.

    This is the sensitivity half of a lever dialog and the reason the dialog is
    defensible at all. A window that shows only a frontier answers "can I afford
    full resolution", never "can I afford to lose it", and a user reading an
    asymmetric window downsamples for the reason the tool put in front of them.
    Batch K's rule -- that a coarser setting must be *demonstrated* to still
    resolve the behaviour -- is executable here and nowhere else in the tool.

    Every cell is a count on a named clip: events, flagged frames, and the
    kept/missed/added split against the reference row. No summary score, and the
    scope caption is not optional -- an event count that does not say which clip
    and which behaviour it came from will be read as a general guarantee, which
    is the exact claim the panel exists to avoid making.
    """
    run_requested = pyqtSignal()
    cancel_requested = pyqtSignal()

    _HEAD = ("setting", "events", "frames", "vs reference", "grid", "pass")

    def __init__(self, scope_note: str, run_text: str = "Run the detector at each scale",
                 parent=None):
        super().__init__(parent)
        self._rows: dict[str, list[QLabel]] = {}
        self._running = False
        self._void: str | None = None       # why the comparison carries no info
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)

        cap = QLabel(scope_note)
        cap.setWordWrap(True)
        cap.setTextFormat(Qt.TextFormat.RichText)
        cap.setStyleSheet(
            "color:#e6c07b; background:#2b2620; border:1px solid #4a3f2c;"
            "border-radius:4px; padding:8px; font-size:12px;")
        lay.addWidget(cap)

        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(2)
        for c, h in enumerate(self._HEAD):
            lab = QLabel(h)
            lab.setStyleSheet("color:#8fa3b5; font-size:11px; font-weight:700;")
            self.grid.addWidget(lab, 0, c)
        lay.addLayout(self.grid)

        row = QHBoxLayout()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#8fa3b5; font-size:11px;")
        self.run_btn = QPushButton(run_text)
        self.run_btn.clicked.connect(self._on_click)
        row.addWidget(self.status, 1)
        row.addWidget(self.run_btn)
        lay.addLayout(row)

    # -- state ---------------------------------------------------------------
    def _on_click(self):
        (self.cancel_requested if self._running else self.run_requested).emit()

    def set_available(self, ok: bool, reason: str = ""):
        """Disable the run with a stated reason. Silence would read as a broken
        button; the reasons here (no window extracted, no region selected) are
        all things the user can fix in one action -- and once fixed the reason
        has to go, or the panel contradicts its own enabled button."""
        self.run_btn.setEnabled(bool(ok))
        if not ok:
            self.status.setText(reason)
        elif not self._rows:
            self.status.setText("")

    def begin(self, labels: list[str], note: str = ""):
        """Lay out one pending row per setting before any pass has run, so the
        cost of the sweep is visible up front rather than arriving as a surprise
        stall."""
        self._running = True
        self._void = None
        self.run_btn.setText("Stop")
        self.run_btn.setEnabled(True)
        self._clear()
        for lab in labels:
            self._set_row(lab, [lab, "…", "", "", "", "waiting"], dim=True)
        self.status.setText(note)

    def void_comparison(self, reason: str | None):
        """Mark the reference as unable to support a comparison at all.

        Set from the reference row (see ``core.evidence.reference_caveat``). Once
        set, no row prints a kept/missed/added split: a detector that fires on
        every frame agrees with itself perfectly at every scale, and printing
        that as "kept 96 · missed 0" would be a passing grade awarded for a
        vacuous test -- the single most misleading thing this panel could do.
        """
        self._void = reason
        if reason:
            self.status.setText(self._void_text())

    def _void_text(self) -> str:
        return f"⚠ These rows are not evidence: {self._void}."

    def add_row(self, label: str, ev, ref=None):
        """One completed pass. ``ref`` is the reference row's evidence (the
        first, full-resolution pass); ``None`` marks this row as the reference."""
        kept_txt, grid_txt = "reference", f"{ev.grid[0]}×{ev.grid[1]}"
        warn = False
        if self._void:
            kept_txt = "no comparison — see below"
            warn = True
        elif ref is not None:
            kept, missed, added = ev.agreement_with(ref)
            kept_txt = f"kept {kept} · missed {missed} · added {added}"
            if tuple(ev.grid) != tuple(ref.grid):
                # The count band counts blocks. A different grid counts a
                # different number of them, so the two rows are thresholded on
                # different quantities -- comparable only if the block tracks
                # the scale, which is what auto does and a pinned block does not.
                warn = True
                grid_txt += " ⚠ differs"
        self._set_row(label, [
            label,
            str(ev.n_events),
            str(ev.n_detected_frames),
            kept_txt,
            f"{grid_txt} · block {ev.block}",
            f"{ev.wall:.1f} s",
        ], warn=warn)

    def fail_row(self, label: str, msg: str):
        self._set_row(label, [label, "—", "", msg, "", ""], warn=True)

    def finish(self, note: str):
        """End the sweep, however it ended.

        Any row still pending is retired to "not run". A stopped sweep is the
        common case and leaving its remaining rows reading "waiting" would claim
        passes are still coming when nothing is running -- worse here than
        cosmetically, because a half-populated table that looks live invites
        reading the scales that DID run as the whole comparison.
        """
        self._running = False
        self.run_btn.setText("Run again")
        self.run_btn.setEnabled(True)
        # A void comparison outranks the closing note: the note says the sweep
        # completed, which is true and, on its own, exactly the wrong thing to
        # leave on screen when the rows mean nothing.
        self.status.setText(self._void_text() if self._void else note)
        for cells in self._rows.values():
            if cells[-1].text() == "waiting":
                cells[1].setText("—")
                cells[-1].setText("not run")
                for c in cells:
                    c.setStyleSheet(
                        "font-family:Consolas; font-size:12px; color:#6b7885;")

    # -- rows ----------------------------------------------------------------
    def _clear(self):
        for cells in self._rows.values():
            for c in cells:
                c.setParent(None)
                c.deleteLater()
        self._rows.clear()

    def _set_row(self, label: str, cells: list[str], *, dim=False, warn=False):
        if label not in self._rows:
            made = [QLabel() for _ in self._HEAD]
            r = len(self._rows) + 1
            for c, lab in enumerate(made):
                self.grid.addWidget(lab, r, c)
            self._rows[label] = made
        colour = "#6b7885" if dim else ("#e6a23c" if warn else "#c8d2dc")
        for lab, text in zip(self._rows[label], cells):
            lab.setText(text)
            lab.setStyleSheet(
                f"font-family:Consolas; font-size:12px; color:{colour};")
