"""The timeline raster strip: behavior state per ROI across the whole clip.

One row per ROI, time on x, a coloured block wherever a behavior is active. This
is the ethogram, and it is the thing a behaviorist will actually read the result
off of -- it makes bout structure, rhythmicity and co-occurrence visible at a
glance in a way that no table does.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

BG = QColor(24, 24, 24)
ROW_BG = QColor(38, 38, 38)
CURSOR = QColor(255, 210, 80)
LABEL = QColor(180, 180, 180)


class TimelineStrip(QWidget):
    """rows: [(roi_id, {behavior_name: (trace, color)})]"""

    seek_requested = pyqtSignal(float)
    roi_clicked = pyqtSignal(int)
    marks_changed = pyqtSignal()

    LABEL_W = 64
    ROW_H = 16
    GAP = 2

    def __init__(self):
        super().__init__()
        self.rows: list[tuple[int, dict]] = []
        self.duration_s = 1.0
        self.cursor_s = 0.0
        self.selected_roi: int | None = None
        # Manual ground-truth markup: {roi_id: [(t0, t1, label), ...]}. Middle-drag
        # on a row paints a span you believe a behavior occurs in; the span is
        # TAGGED with the label you are currently marking (set via set_active_label,
        # driven by the behavior picker), so you can lay down "Flying" spans, switch,
        # and lay down "Still" spans on the same rows in different colours. These are
        # the labels a threshold is judged against, so they persist (see Tab 3).
        self.marks: dict[int, list[tuple[float, float, str]]] = {}
        self.label_colors: dict[str, str] = {}
        self.active_label: str = ""
        self.active_color: str = "#ffffff"
        self._mark_drag: tuple[int, float, float] | None = None
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_active_label(self, label: str, color: str) -> None:
        """Which label new middle-drags are tagged with, and its colour."""
        self.active_label = label or ""
        self.active_color = color or "#ffffff"
        if self.active_label:
            self.label_colors[self.active_label] = self.active_color

    def clear_marks(self, label: str | None = None):
        """Clear all marks, or only those of one label when given."""
        if label is None:
            self.marks = {}
        else:
            self.marks = {rid: [s for s in spans if s[2] != label]
                          for rid, spans in self.marks.items()}
            self.marks = {rid: spans for rid, spans in self.marks.items() if spans}
        self.update()
        self.marks_changed.emit()

    def marks_to_dict(self) -> dict:
        """Ground-truth spans keyed by replicate id (as strings, for JSON), each
        span carrying its behavior label; plus the label->colour map."""
        return {
            "spans": {str(rid): [[float(t0), float(t1), lbl]
                                 for (t0, t1, lbl) in spans]
                      for rid, spans in self.marks.items() if spans},
            "colors": dict(self.label_colors),
        }

    def set_marks_from_dict(self, d: dict) -> None:
        d = d or {}
        # Back-compat: the old format was {roi_id: [[t0, t1], ...]} with no labels.
        spans = d.get("spans", d)
        self.label_colors = dict(d.get("colors", {}))
        out: dict[int, list[tuple[float, float, str]]] = {}
        for rid, sp in spans.items():
            try:
                rid_i = int(rid)
            except (TypeError, ValueError):
                continue
            lst = []
            for s in sp:
                t0, t1 = float(s[0]), float(s[1])
                lbl = s[2] if len(s) > 2 else ""
                lst.append((t0, t1, lbl))
            if lst:
                out[rid_i] = lst
        self.marks = out
        self.update()

    def set_rows(self, rows, duration_s: float):
        self.rows = rows
        self.duration_s = max(1e-6, duration_s)
        self.setMinimumHeight(
            max(60, len(rows) * (self.ROW_H + self.GAP) + 22))
        self.update()

    def set_cursor(self, t_s: float):
        self.cursor_s = t_s
        self.update()

    def _row_rect(self, i: int) -> QRect:
        return QRect(self.LABEL_W, 16 + i * (self.ROW_H + self.GAP),
                     max(1, self.width() - self.LABEL_W - 6), self.ROW_H)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), BG)
        p.setFont(QFont("Consolas", 7))

        if not self.rows:
            p.setPen(QColor(110, 110, 110))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "extract ROIs and define a behavior to see the ethogram")
            p.end()
            return

        for i, row in enumerate(self.rows):
            # Rows may be (roi_id, behaviors) or (roi_id, label, behaviors).
            if len(row) == 3:
                roi_id, label, behaviors = row
            else:
                roi_id, behaviors = row
                label = f"#{roi_id}"
            r = self._row_rect(i)
            p.fillRect(r, ROW_BG)

            if roi_id == self.selected_roi:
                # NoBrush is essential: the previous row's behavior draw leaves a
                # solid (behavior-coloured) brush set, and without resetting it
                # this "outline" rect fills the whole selected row with that
                # colour -- which read as "the selected replicate is detected
                # everywhere" and hid its real, sparse detection.
                p.setPen(QPen(QColor(255, 210, 80, 160), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(r.adjusted(-1, -1, 0, 0))

            p.setPen(LABEL)
            p.drawText(3, r.center().y() + 4, label[:8])

            for name, payload in behaviors.items():
                # payload is (trace, colour) or (trace, strength, colour).
                if len(payload) == 3:
                    trace, strength, colour = payload
                else:
                    trace, colour = payload
                    strength = None
                if trace is None or trace.size == 0:
                    continue
                c = QColor(colour)
                n = trace.size
                p.setPen(Qt.PenStyle.NoPen)

                # Graded strength: a faint per-pixel-column intensity showing HOW
                # MUCH of the box is passing, even where it does not (yet) clear
                # the detection thresholds. Binned to pixel columns so it is ~800
                # rects, not 30k. Detection runs are drawn solid on top.
                if strength is not None and strength.size:
                    cols = max(1, r.width())
                    edges = np.linspace(0, n, cols + 1).astype(int)
                    for cx in range(cols):
                        a, b = edges[cx], max(edges[cx] + 1, edges[cx + 1])
                        s = float(strength[a:b].max())
                        if s <= 0.01:
                            continue
                        cc = QColor(c)
                        cc.setAlpha(int(30 + 150 * min(1.0, s)))
                        p.setBrush(cc)
                        p.drawRect(QRect(r.left() + cx, r.top() + r.height() // 2,
                                         1, r.height() // 2))

                p.setBrush(c)
                # Draw detection runs, not per-frame rectangles.
                d = np.diff(trace.astype(np.int8))
                starts = np.flatnonzero(d == 1) + 1
                ends = np.flatnonzero(d == -1) + 1
                if trace[0]:
                    starts = np.r_[0, starts]
                if trace[-1]:
                    ends = np.r_[ends, n]
                for s, e in zip(starts, ends):
                    x0 = r.left() + s / n * r.width()
                    x1 = r.left() + e / n * r.width()
                    p.drawRect(QRect(int(x0), r.top(),
                                     max(1, int(x1 - x0)), r.height() // 2))

        # Manual ground-truth spans, drawn on top as a dashed outline in the
        # LABEL's colour so different behaviors are distinguishable and never read
        # as detection (which fills the lower half solid; these outline the row).
        def _mark_x(t):
            return self.LABEL_W + (t / self.duration_s) * \
                max(1, self.width() - self.LABEL_W - 6)

        for i, row in enumerate(self.rows):
            rid = row[0]
            spans = list(self.marks.get(rid, []))
            if self._mark_drag and self._mark_drag[0] == rid:
                spans.append((self._mark_drag[1], self._mark_drag[2],
                              self.active_label))
            if not spans:
                continue
            rr = self._row_rect(i)
            for t0, t1, lbl in spans:
                col = QColor(self.label_colors.get(lbl, self.active_color))
                a, b = _mark_x(min(t0, t1)), _mark_x(max(t0, t1))
                rect = QRect(int(a), rr.top(), max(2, int(b - a)), rr.height())
                fill = QColor(col)
                fill.setAlpha(45)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(fill)
                p.drawRect(rect)
                outline = QColor(col)
                outline.setAlpha(220)
                p.setPen(QPen(outline, 1, Qt.PenStyle.DashLine))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(rect)

        cx = self.LABEL_W + (self.cursor_s / self.duration_s) * \
            max(1, self.width() - self.LABEL_W - 6)
        p.setPen(QPen(CURSOR, 1))
        p.drawLine(int(cx), 12, int(cx), self.height() - 6)

        p.setPen(QColor(140, 140, 140))
        p.drawText(self.LABEL_W, 10, "0 s")
        txt = f"{self.duration_s:.0f} s"
        p.drawText(self.width() - 6 - 7 * len(txt), 10, txt)
        p.end()

    def _time_at(self, x: int) -> float:
        frac = (x - self.LABEL_W) / max(1, self.width() - self.LABEL_W - 6)
        return float(np.clip(frac, 0, 1)) * self.duration_s

    def _row_at(self, pos) -> int | None:
        for i, row in enumerate(self.rows):
            rr = self._row_rect(i)
            if rr.top() <= pos.y() <= rr.bottom():
                return row[0]
        return None

    def mousePressEvent(self, e):
        if not self.rows:
            return
        # Middle-drag paints a ground-truth markup span on that replicate's row.
        if e.button() == Qt.MouseButton.MiddleButton:
            rid = self._row_at(e.pos())
            if rid is not None:
                t = self._time_at(e.pos().x())
                self._mark_drag = (rid, t, t)
                self.update()
            return

        x = e.pos().x()
        if x > self.LABEL_W:
            self.seek_requested.emit(self._time_at(x))
        rid = self._row_at(e.pos())
        if rid is not None:
            self.roi_clicked.emit(rid)

    def mouseMoveEvent(self, e):
        if self._mark_drag is not None:
            rid, t0, _ = self._mark_drag
            self._mark_drag = (rid, t0, self._time_at(e.pos().x()))
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton and self._mark_drag is not None:
            rid, t0, t1 = self._mark_drag
            self._mark_drag = None
            if abs(t1 - t0) > 1e-3:
                if self.active_label:
                    self.label_colors[self.active_label] = self.active_color
                self.marks.setdefault(rid, []).append(
                    (min(t0, t1), max(t0, t1), self.active_label))
                self.marks_changed.emit()
            self.update()
