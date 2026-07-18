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
from PyQt6.QtGui import (QColor, QFont, QImage, QPainter, QPainterPath, QPen,
                         QPixmap)
from PyQt6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)

_BG = QColor("#1b1f24")
_AXIS = QColor("#5b6672")
_TEXT = QColor("#c8d2dc")
_CURVE = QColor("#4ac6ff")
_KNEE = QColor("#ffb347")
_SELECTED = QColor("#7ee787")
_FLAT = QColor("#3a4450")     # the region past the knee, shaded as "buys nothing"
_STORAGE = QColor("#c792ea")  # the companion storage series, on its own axis
_RISE = QColor("#ff6b6b")     # where storage starts going back UP


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
        self._second: list[float] = []
        self._second_label = ""
        self._second_fmt = None
        self._rise: float | None = None

    def set_curve(self, values, costs, *, labels=None, knee=None, selected=None,
                  y_label="", second=None, second_label="", second_fmt=None,
                  rise_below=None):
        """``second`` is an optional companion series on its own right-hand axis.

        Two axes rather than one because the series are in different units
        (hours against bytes) and the interesting fact is the SHAPE difference:
        under the tracked block the storage line is dead flat while the time
        curve falls, which is the whole two-lever argument in one picture.
        Normalizing them onto a shared axis would destroy exactly that reading.
        """
        self._values = list(values)
        self._costs = list(costs)
        self._labels = list(labels) if labels else [f"{v:g}" for v in self._values]
        self._knee = knee
        self._selected = selected
        self._y_label = y_label
        self._second = list(second) if second else []
        self._second_label = second_label
        self._second_fmt = second_fmt
        self._rise = rise_below
        self.update()

    # -- geometry ------------------------------------------------------------
    def _plot_rect(self):
        # Right margin widens when a second series needs its own axis labels.
        right = 76 if self._second else 12
        return 54, 22, max(1, self.width() - 54 - right), max(1, self.height() - 56)

    def _y2_of(self, v: float) -> float:
        """Second series, on its own axis. Zero-based like the primary, so a flat
        line reads as flat rather than as noise amplified to fill the panel."""
        _, y, _, h = self._plot_rect()
        hi = max(self._second) if self._second else 1.0
        hi = hi if hi > 0 else 1.0
        return y + h * (1.0 - v / hi)

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

        # The storage series is drawn first, under the time curve: it is context
        # for the primary decision, and on a tracked block it is a flat line that
        # would otherwise sit on top of the curve it is meant to be compared to.
        if self._second:
            p.setPen(QPen(_STORAGE, 1))
            for frac in (0.0, 0.5, 1.0):
                yy = y0 + h * (1.0 - frac)
                hi2 = max(self._second)
                txt = (self._second_fmt(hi2 * frac) if self._second_fmt
                       else f"{hi2 * frac:.0f}")
                p.drawText(x0 + w + 6, int(yy) - 7, 70, 14,
                           int(Qt.AlignmentFlag.AlignLeft
                               | Qt.AlignmentFlag.AlignVCenter), txt)
            spath = QPainterPath()
            for i, (v, c) in enumerate(zip(self._values, self._second)):
                pt = QPointF(self._x_of(v), self._y2_of(c))
                spath.moveTo(pt) if i == 0 else spath.lineTo(pt)
            p.setPen(QPen(_STORAGE, 2, Qt.PenStyle.DashLine))
            p.drawPath(spath)

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

        # Where storage stops falling and starts rising again: past here the
        # user pays MORE cache for less resolution, which is a hard reason to
        # stop and is invisible without the marker.
        if self._rise is not None:
            rx = self._x_of(self._rise)
            p.setPen(QPen(_RISE, 1, Qt.PenStyle.DotLine))
            p.drawLine(int(rx), y0, int(rx), y0 + h)
            p.setPen(_RISE)
            # Flip to the right of the line when it sits near the left edge --
            # the rising tail is by definition at the SMALLEST scale, i.e. hard
            # against the y axis, so the leftward default clipped this label to
            # "orage rises" in the common case.
            tw = 96
            if rx - 4 - tw >= x0:
                tx, align = int(rx) - 4 - tw, Qt.AlignmentFlag.AlignRight
            else:
                tx, align = int(rx) + 4, Qt.AlignmentFlag.AlignLeft
            # The arrow always points LEFT, because the rising region is always
            # the smaller scales and smaller is left. The placement flips; the
            # direction it means must not.
            p.drawText(tx, y0 + h - 16, tw, 14,
                       int(align | Qt.AlignmentFlag.AlignTop),
                       "◂ storage rises")

        p.setPen(_TEXT)
        p.drawText(x0 + 4, 0, w, 16,
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
                   self._y_label)
        if self._second_label:
            p.setPen(_STORAGE)
            p.drawText(x0 + 4, 0, w, 16,
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop),
                       self._second_label)

    def _fmt_y(self, v: float) -> str:
        if v >= 100:
            return f"{v:.0f}"
        if v >= 10:
            return f"{v:.0f}"
        return f"{v:.1f}"


def _gray_pixmap(arr, side: int) -> QPixmap:
    """A uint8 (H, W) array as a square-ish pixmap at ``side`` px, NEAREST.

    ``FastTransformation`` is nearest-neighbour and is the load-bearing choice:
    the panel exists to show what the working resolution costs, and smooth
    scaling would paint back an interpolated approximation of exactly the detail
    the lever removed. The tile is upscaled to a COMMON size rather than drawn at
    its own, so a coarser scale reads as blockier and not merely as smaller.
    """
    h, w = arr.shape[:2]
    buf = arr.tobytes()                     # keep the copy alive past QImage
    img = QImage(buf, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(img).scaled(
        side, side, Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation)


class RenderStrip(QWidget):
    """What one replicate looks like to the pipeline at each candidate setting.

    The companion to the frontier: that panel prices the lever, this one shows
    what is being bought with. Without it the user is asked to trade away
    resolution with no view of the resolution.

    Two rows, and the lower one carries the argument. The top row is the working
    grey image the per-pixel stages receive; the bottom is the frame-difference
    field they are built out of, rendered on a display range SHARED across every
    tile (see ``core/scale_render.py``). Sharing the range is what makes the
    amplitude loss visible -- auto-contrasting each tile would make every setting
    look equally vivid and quietly delete the only thing worth seeing.

    Generic over the lever: it renders whatever :class:`ScaleRender`-shaped rows
    it is handed, so Batch N's block-size dialog reuses it for grid granularity.
    """
    # Sized so the full candidate set fits one row beside the legend column: the
    # strip then lines up with the frontier's x axis and reads as its picture,
    # rather than as a separate shorter list the user has to match up by hand.
    _TILE = 96

    def __init__(self, note: str, parent=None):
        super().__init__(parent)
        self._tiles: dict[float, list[QLabel]] = {}
        self._selected: float | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)

        cap = QLabel(note)
        cap.setWordWrap(True)
        cap.setTextFormat(Qt.TextFormat.RichText)
        cap.setStyleSheet(
            "color:#c8d2dc; background:#232a31; border:1px solid #33404d;"
            "border-radius:4px; padding:8px; font-size:12px;")
        lay.addWidget(cap)

        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(2)
        # Column 0 is the row legend; tiles start at column 1. Rows match
        # set_renders: 0 label, 1 grey, 2 working size, 3 difference.
        for row, text in ((1, "what the solve sees"), (3, "frame difference")):
            lab = QLabel(text)
            lab.setStyleSheet("color:#8fa3b5; font-size:10px;")
            self.grid.addWidget(lab, row, 0)
        lay.addLayout(self.grid)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#8fa3b5; font-size:11px;")
        lay.addWidget(self.status)

    def set_status(self, text: str):
        self.status.setText(text)

    def set_renders(self, renders, selected: float | None = None):
        self._clear()
        self._selected = selected
        for col, r in enumerate(renders, start=1):
            head = QLabel(r.label)
            head.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            gray = QLabel()
            gray.setPixmap(_gray_pixmap(r.gray, self._TILE))
            gray.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # Fixed, not minimum: a QLabel holding a pixmap will happily be
            # squeezed by a layout under vertical pressure and CLIP the pixmap
            # rather than scale it, which silently turns every tile into a
            # letterbox strip and destroys the comparison the panel exists for.
            gray.setFixedSize(self._TILE, self._TILE)
            size = QLabel(r.size_label)
            size.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            size.setStyleSheet("color:#8fa3b5; font-size:10px;")
            chg = QLabel()
            chg.setPixmap(_gray_pixmap(r.change, self._TILE))
            chg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chg.setFixedSize(self._TILE, self._TILE)

            sel = selected is not None and abs(r.scale - selected) < 1e-9
            head.setStyleSheet(
                f"font-family:Consolas; font-size:12px; font-weight:700;"
                f"color:{'#7ee787' if sel else '#c8d2dc'};")
            for row, wdg in ((0, head), (1, gray), (2, size), (3, chg)):
                self.grid.addWidget(wdg, row, col)
            self._tiles[r.scale] = [head, gray, size, chg]

    def _clear(self):
        for cells in self._tiles.values():
            for c in cells:
                c.setParent(None)
                c.deleteLater()
        self._tiles.clear()


class SweepPanel(QWidget):
    """Time a real pass at each candidate setting, and price its storage.

    Replaces the empirical *detection* panel that briefly sat here. That panel
    ran the tuned detector at each scale and reported events kept and lost, which
    sounds like the sensitivity evidence Batch K asks for and is not: the value
    and count bands are absolute, and downsampling averages pixels before
    differencing, so per-block band power falls with scale and a fixed threshold
    catches less regardless of whether the behaviour is still resolved. Measured,
    the loss was monotone with ZERO added frames at any scale -- the signature of
    threshold drift, not of lost structure. Deciding whether a behaviour is
    detectable is what the live surface and the whole-video pass are for; this
    window prices the lever, and says so.

    What is left is honest and useful on its own: measured wall time per scale,
    projected against the user's corpus, alongside the storage that setting
    actually costs. Both are things no other view shows.
    """
    run_requested = pyqtSignal()
    cancel_requested = pyqtSignal()

    _HEAD = ("setting", "pass", "per frame", "projected", "storage", "grid")

    def __init__(self, note: str, run_text: str = "Time a pass at each scale",
                 parent=None):
        super().__init__(parent)
        self._rows: dict[str, list[QLabel]] = {}
        self._running = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)

        cap = QLabel(note)
        cap.setWordWrap(True)
        cap.setTextFormat(Qt.TextFormat.RichText)
        cap.setStyleSheet(
            "color:#c8d2dc; background:#232a31; border:1px solid #33404d;"
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
        button -- and once the reason is fixed it has to go, or the panel
        contradicts its own enabled button."""
        self.run_btn.setEnabled(bool(ok))
        if not ok:
            self.status.setText(reason)
        elif not self._rows:
            self.status.setText("")

    def begin(self, labels: list[str], note: str = ""):
        """Lay out one pending row per setting before any pass has run, so the
        cost of the sweep is visible up front rather than arriving as a stall."""
        self._running = True
        self.run_btn.setText("Stop")
        self.run_btn.setEnabled(True)
        self._clear()
        for lab in labels:
            self._set_row(lab, [lab, "…", "", "", "", "waiting"], dim=True)
        self.status.setText(note)

    def add_row(self, label: str, sp, projected: str, storage: str,
                ref=None):
        """One completed pass. ``ref`` is the reference (full-resolution) pass,
        used only to express this row's speedup; ``None`` marks the reference."""
        speed = ""
        if ref is not None and sp.wall > 0 and ref.wall > 0:
            speed = f"  ({ref.wall / sp.wall:.1f}× faster)"
        self._set_row(label, [
            label,
            f"{sp.wall:.1f} s{speed}",
            f"{sp.seconds_per_frame * 1000:.0f} ms",
            projected,
            storage,
            f"{sp.grid[0]}×{sp.grid[1]} · block {sp.block}",
        ], warn=not sp.usable)

    def fail_row(self, label: str, msg: str):
        self._set_row(label, [label, "—", "", msg, "", ""], warn=True)

    def finish(self, note: str):
        """End the sweep, however it ended.

        Any row still pending is retired to "not run": a stopped sweep is the
        common case, and rows left reading "waiting" claim passes are still
        coming when nothing is running.
        """
        self._running = False
        self.run_btn.setText("Run again")
        self.run_btn.setEnabled(True)
        self.status.setText(note)
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
