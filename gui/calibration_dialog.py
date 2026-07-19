"""Draw a line on the frame to calibrate a replicate.

Two lines, drawn on the full-resolution source frame:

* **Ruler** — across something of known physical size (a scale bar, a ruler, an
  arena feature you have measured). Fixes ``pixels_per_mm``.
* **Animal** — along the body of the animal. With the ruler, fixes
  ``body_length_mm``.

Both feed fields that already exist on a replicate (``core/roi.py``), so exports,
``speed_body_lengths_s`` and the downsample dialog's "what you keep" readout all
pick them up without learning anything new.

Why the animal line alone is enough for the resolution question
----------------------------------------------------------------
The downsample dialog reports working pixels per body length, which works out to
``body_line_px * scale`` — the ruler cancels exactly (see ``core/calibration.py``).
So this window shows that number the moment the animal line is drawn, before any
ruler exists. The ruler is what makes the result *portable*: expressed in mm it
survives a different camera, a different crop, and an export. A user who only
wants to know whether 0.5 still resolves their animal never needs to find one.

Why it magnifies, and why it says how well you drew the line
-------------------------------------------------------------
An ant is ~50 source pixels in a 5312 px frame. Fitted to a widget that is a
five-fold reduction, its whole body is ten display pixels and a hand-drawn line
across it is worth nothing. So the view pans and zooms to at least 1:1, and the
readout carries the placement error propagated from the zoom the line was drawn
at (``core/calibration.py:line_error_px``). That error is a statement about the
measurement, not about the footage — it tells the user to zoom in or draw along
something longer, which is a thing they can act on.

The frame is passed in rather than decoded here, so this dialog owns no video and
can be opened from the replicate tab (which has a frame in hand) and from the
downsample dialog (which asks its owner for one).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QButtonGroup, QDialog, QDoubleSpinBox, QFrame,
                             QHBoxLayout, QLabel, QPushButton, QRadioButton,
                             QScrollArea, QVBoxLayout, QWidget)

from core.calibration import (Calibration, line_error_px, line_length_px,
                              pixels_per_mm_from_line, relative_error)

# Offered zooms, in display pixels per source pixel. "Fit" is computed from the
# viewport. 1:1 and up are the ones that matter -- see the module docstring.
_ZOOMS = (0.5, 1.0, 2.0, 4.0, 8.0)

_RULER_COLOR = "#ffd24a"
_ANIMAL_COLOR = "#7ee787"

_INTRO = (
    "<b>Draw a line to calibrate this replicate.</b> Pick <b>Animal</b> and drag "
    "along the body: that alone gives working pixels per body length, which is "
    "what decides whether a working scale still resolves your animal. Pick "
    "<b>Ruler</b> and drag across something whose size in millimetres you know "
    "to express the result in physical units, which is what exports and "
    "cross-clip comparisons need.<br>"
    "<b>Zoom in before you draw.</b> A line is placed to about one display pixel "
    "per end, so a short line measured at a small zoom carries a large error — "
    "the readout below states it."
)


class _LineView(QLabel):
    """A frame at a fixed zoom, inside a scroll area, with one line drawn on it.

    Coordinates in and out are SOURCE pixels. The widget is sized to
    ``zoom * source``, but only the exposed region is ever painted, so an 8x view
    of a 5312x2988 frame costs no more memory than a 1:1 one -- scaling the whole
    pixmap would allocate gigabytes at the magnifications this tool needs.
    """
    line_drawn = pyqtSignal(object, object)   # (x0,y0), (x1,y1) in source px

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setStyleSheet("background-color:#111;")
        self._pix: QPixmap | None = None
        self._src = (1, 1)
        self._zoom = 1.0
        self._box: tuple[int, int, int, int] | None = None   # source px
        self._lines: list[tuple] = []      # (p0, p1, hex, label) in source px
        self._drag: tuple[float, float] | None = None
        self._cursor_src: tuple[float, float] | None = None

    # -- content -------------------------------------------------------------
    def set_frame(self, frame_bgr: np.ndarray):
        import cv2
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._pix = QPixmap.fromImage(img.copy())
        self._src = (w, h)
        self._apply_size()

    def set_box(self, box: tuple[int, int, int, int] | None):
        self._box = box
        self.update()

    def set_lines(self, lines: list[tuple]):
        self._lines = list(lines)
        self.update()

    # -- zoom ----------------------------------------------------------------
    @property
    def zoom(self) -> float:
        return self._zoom

    def set_zoom(self, zoom: float):
        self._zoom = max(0.01, float(zoom))
        self._apply_size()

    def fit_zoom(self, viewport) -> float:
        w, h = self._src
        return min(viewport.width() / max(1, w), viewport.height() / max(1, h))

    def _apply_size(self):
        w, h = self._src
        self.setFixedSize(max(1, int(round(w * self._zoom))),
                          max(1, int(round(h * self._zoom))))
        self.update()

    # -- painting ------------------------------------------------------------
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#111"))
        if self._pix is None:
            p.end()
            return
        # Only the exposed strip, mapped back into the unscaled pixmap. Smooth
        # transform is deliberately OFF above 1:1: an interpolated blow-up would
        # invent edges to align a line against that the source does not have.
        target = QRectF(ev.rect())
        source = QRectF(target.x() / self._zoom, target.y() / self._zoom,
                        target.width() / self._zoom,
                        target.height() / self._zoom)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,
                        self._zoom < 1.0)
        p.drawPixmap(target, self._pix, source)

        if self._box is not None:
            x0, y0, x1, y1 = self._box
            p.setPen(QPen(QColor("#4ac6ff"), 1, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(*self._to_view(x0, y0),
                              (x1 - x0) * self._zoom, (y1 - y0) * self._zoom))

        for p0, p1, hexcol, label in self._lines:
            self._draw_line(p, p0, p1, hexcol, label)
        p.end()

    def _to_view(self, x: float, y: float) -> tuple[float, float]:
        return x * self._zoom, y * self._zoom

    def _draw_line(self, p: QPainter, p0, p1, hexcol: str, label: str):
        a = QPointF(*self._to_view(*p0))
        b = QPointF(*self._to_view(*p1))
        col = QColor(hexcol)
        p.setPen(QPen(col, 2))
        p.drawLine(a, b)
        # End caps: the endpoints are the measurement, so they get to be visible
        # at any zoom rather than disappearing under the line's own width.
        for pt in (a, b):
            p.drawEllipse(pt, 3.5, 3.5)
        if label:
            p.setPen(QPen(col, 1))
            p.drawText(QPointF((a.x() + b.x()) / 2 + 6,
                               (a.y() + b.y()) / 2 - 6), label)

    # -- interaction ---------------------------------------------------------
    def _src_of(self, pos: QPoint) -> tuple[float, float]:
        w, h = self._src
        return (min(w, max(0.0, pos.x() / self._zoom)),
                min(h, max(0.0, pos.y() / self._zoom)))

    def mousePressEvent(self, e):
        if self._pix is None or e.button() != Qt.MouseButton.LeftButton:
            return
        self._drag = self._src_of(e.pos())

    def mouseMoveEvent(self, e):
        self._cursor_src = self._src_of(e.pos())
        if self._drag is not None:
            # Preview through the same signal as the committed line, so what the
            # readout shows while dragging is computed exactly as the final
            # number is -- no second code path to disagree with the first.
            self.line_drawn.emit(self._drag, self._cursor_src)

    def mouseReleaseEvent(self, e):
        if self._drag is None:
            return
        end = self._src_of(e.pos())
        start, self._drag = self._drag, None
        self.line_drawn.emit(start, end)


class CalibrationDialog(QDialog):
    """Calibrate one replicate by drawing on ``frame_bgr`` (a SOURCE frame).

    ``box`` is the replicate's ``source_box`` (x0, y0, x1, y1) if known; it is
    drawn as a guide and the view opens centred on it. The whole frame is shown
    rather than only the box, because fiducials often sit outside the replicate
    -- a scale bar at the edge of the arena is the ordinary case.

    Read :attr:`calibration` after ``exec()`` returns accepted.
    """

    def __init__(self, frame_bgr: np.ndarray, *,
                 box: tuple[int, int, int, int] | None = None,
                 label: str = "",
                 pixels_per_mm: float | None = None,
                 body_length_mm: float | None = None,
                 body_length_px: float | None = None,
                 scale: float = 1.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Calibrate {label}".strip() or "Calibrate replicate")
        self.resize(1080, 820)
        self._scale = float(scale)
        self._box = box
        self.calibration: Calibration | None = None

        # Each line is (p0, p1, zoom it was drawn at); zoom is kept per line
        # because the error depends on it and the user may zoom between them.
        self._ruler: tuple | None = None
        self._animal: tuple | None = None
        # One-shot: the opening zoom/centre is applied on first show only, so a
        # hide/show does not throw away where the user had panned to.
        self._zoomed = False
        # Pre-existing calibration seeds the ruler side, so reopening to redraw
        # only the animal does not silently discard a px/mm set by hand.
        self._seed_ppm = pixels_per_mm if (pixels_per_mm or 0) > 0 else None
        self._seed_blm = body_length_mm if (body_length_mm or 0) > 0 else None
        # The animal length in source pixels, which is what an earlier
        # fiducial-free session stored. Seeding it is what lets "measure the
        # animal now, find a ruler later" actually complete: without it,
        # reopening reads "nothing measured yet" and drawing a ruler can never
        # derive a body_length_mm from the measurement already on the replicate.
        self._seed_blpx = body_length_px if (body_length_px or 0) > 0 else None

        root = QVBoxLayout(self)
        intro = QLabel(_INTRO)
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#c8d2dc; font-size:12px;")
        root.addWidget(intro)

        root.addLayout(self._build_toolbar())

        self.view = _LineView()
        self.view.set_frame(frame_bgr)
        self.view.set_box(box)
        self.view.line_drawn.connect(self._on_line)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.view)
        root.addWidget(self.scroll, 1)

        root.addWidget(self._build_readout())
        root.addLayout(self._build_buttons())
        self._refresh()

    # -- construction --------------------------------------------------------
    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.animal_btn = QRadioButton("Animal (body length)")
        self.ruler_btn = QRadioButton("Ruler (known size)")
        # Animal first and default: it is the measurement that answers the
        # resolution question on its own, and making the ruler the entry point
        # would imply the tool is unusable without one.
        self.animal_btn.setChecked(True)
        for b, col in ((self.animal_btn, _ANIMAL_COLOR),
                       (self.ruler_btn, _RULER_COLOR)):
            b.setStyleSheet(f"color:{col}; font-weight:600;")
            self.mode_group.addButton(b)
            row.addWidget(b)

        self.known_mm = QDoubleSpinBox()
        self.known_mm.setRange(0.0, 1_000_000.0)
        self.known_mm.setDecimals(3)
        self.known_mm.setSuffix(" mm")
        self.known_mm.setSpecialValueText("length?")
        self.known_mm.setToolTip(
            "The true length of whatever the ruler line is drawn across.")
        self.known_mm.valueChanged.connect(self._refresh)
        row.addWidget(QLabel("ruler is"))
        row.addWidget(self.known_mm)
        row.addStretch(1)

        row.addWidget(QLabel("zoom"))
        fit = QPushButton("Fit")
        fit.clicked.connect(self._zoom_fit)
        row.addWidget(fit)
        for z in _ZOOMS:
            b = QPushButton(f"{z:g}x")
            b.setFixedWidth(44)
            b.clicked.connect(lambda _=False, zz=z: self._set_zoom(zz))
            row.addWidget(b)
        if self._box is not None:
            go = QPushButton("Go to replicate")
            go.clicked.connect(self._center_on_box)
            row.addWidget(go)
        return row

    def _build_readout(self) -> QWidget:
        box = QFrame()
        box.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(box)
        self.line_lbl = QLabel("")
        self.result_lbl = QLabel("")
        self.warn_lbl = QLabel("")
        self.warn_lbl.setWordWrap(True)
        self.line_lbl.setStyleSheet("font-family:Consolas; font-size:12px;")
        self.result_lbl.setStyleSheet(
            "font-family:Consolas; font-size:13px; font-weight:700;")
        self.warn_lbl.setStyleSheet("color:#ffb454; font-size:11px;")
        for w in (self.line_lbl, self.result_lbl, self.warn_lbl):
            lay.addWidget(w)
        return box

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        clear = QPushButton("Clear lines")
        clear.clicked.connect(self._clear)
        row.addWidget(clear)
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.apply_btn = QPushButton("Apply calibration")
        self.apply_btn.setDefault(True)
        self.apply_btn.clicked.connect(self._apply)
        row.addWidget(cancel)
        row.addWidget(self.apply_btn)
        return row

    # -- zoom ----------------------------------------------------------------
    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._zoomed:
            self._zoomed = True
            # Open where the work is: on the replicate, at 1:1, since that is the
            # smallest zoom at which an animal line means anything. Fit only when
            # there is no box to aim at.
            if self._box is not None:
                self._set_zoom(1.0)
                self._center_on_box()
            else:
                self._zoom_fit()

    def _set_zoom(self, z: float):
        # Hold the viewport centre across the zoom change, in source pixels --
        # otherwise zooming in on a magnified frame jumps to the top-left corner
        # and the user loses the animal they were aiming at.
        centre = self._viewport_centre_src()
        self.view.set_zoom(z)
        self._centre_on_src(*centre)
        self._refresh()

    def _zoom_fit(self):
        self._set_zoom(self.view.fit_zoom(self.scroll.viewport()))

    def _viewport_centre_src(self) -> tuple[float, float]:
        h, v = self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()
        vp = self.scroll.viewport()
        z = self.view.zoom
        return ((h.value() + vp.width() / 2) / z,
                (v.value() + vp.height() / 2) / z)

    def _centre_on_src(self, cx: float, cy: float):
        z = self.view.zoom
        vp = self.scroll.viewport()
        self.scroll.horizontalScrollBar().setValue(
            int(round(cx * z - vp.width() / 2)))
        self.scroll.verticalScrollBar().setValue(
            int(round(cy * z - vp.height() / 2)))

    def _center_on_box(self):
        if self._box is None:
            return
        x0, y0, x1, y1 = self._box
        self._centre_on_src((x0 + x1) / 2, (y0 + y1) / 2)

    # -- measurement ---------------------------------------------------------
    def _on_line(self, p0, p1):
        entry = (p0, p1, self.view.zoom)
        if self.ruler_btn.isChecked():
            self._ruler = entry
        else:
            self._animal = entry
        self._refresh()

    def _clear(self):
        self._ruler = None
        self._animal = None
        self._seed_ppm = None
        self._seed_blm = None
        self._seed_blpx = None
        self._refresh()

    def _ppm(self) -> tuple[float | None, float | None]:
        """(pixels_per_mm, its fractional error) from the ruler, else the seed."""
        if self._ruler is not None and self.known_mm.value() > 0:
            p0, p1, z = self._ruler
            length = line_length_px(p0, p1)
            if length > 0:
                err = relative_error(length, line_error_px(z))
                return pixels_per_mm_from_line(length,
                                               self.known_mm.value()), err
        if self._seed_ppm is not None:
            # Set by hand or by an earlier session: no line, so no line error.
            return self._seed_ppm, None
        return None, None

    def _build_calibration(self) -> Calibration:
        ppm, ppm_err = self._ppm()
        body_px = body_err = None
        if self._animal is not None:
            p0, p1, z = self._animal
            body_px = line_length_px(p0, p1)
            body_err = relative_error(body_px, line_error_px(z))
        elif self._seed_blpx is not None:
            # Measured in an earlier session and stored in pixels. Preferred over
            # the mm seed because it is what was actually measured -- deriving it
            # back from mm would fold in whatever ruler was used then, and a
            # ruler re-drawn now would then move a length nobody re-measured.
            body_px = self._seed_blpx
        elif self._seed_blm is not None and ppm is not None:
            # Only mm stored (typed by hand, or calibrated before body_length_px
            # existed): carry it through in pixels so a re-drawn ruler updates
            # the mm consistently.
            body_px = self._seed_blm * ppm
        fid_px = line_length_px(*self._ruler[:2]) if self._ruler else None
        return Calibration(
            pixels_per_mm=ppm, body_length_px=body_px,
            fiducial_px=fid_px,
            fiducial_mm=self.known_mm.value() or None,
            fiducial_rel_err=ppm_err, body_rel_err=body_err)

    # -- readout -------------------------------------------------------------
    def _refresh(self):
        lines = []
        if self._ruler is not None:
            p0, p1, z = self._ruler
            lines.append((p0, p1, _RULER_COLOR,
                          f"{line_length_px(p0, p1):.1f} px"))
        if self._animal is not None:
            p0, p1, z = self._animal
            lines.append((p0, p1, _ANIMAL_COLOR,
                          f"{line_length_px(p0, p1):.1f} px"))
        self.view.set_lines(lines)

        cal = self._build_calibration()
        self.line_lbl.setText(self._line_text(cal))
        self.result_lbl.setText(self._result_text(cal))
        self.warn_lbl.setText(self._warn_text(cal))
        # Applicable as soon as anything was measured. A partial calibration is
        # legitimate: the animal line alone is the resolution answer, and
        # as_replicate_fields() writes only what it actually determined.
        self.apply_btn.setEnabled(bool(cal.as_replicate_fields()) or
                                  cal.body_length_px is not None)

    def _line_text(self, cal: Calibration) -> str:
        bits = []
        if cal.fiducial_px:
            e = cal.fiducial_rel_err
            bits.append(f"ruler {cal.fiducial_px:.1f} source px"
                        + (f" ±{e * 100:.1f}%" if e else ""))
        if cal.body_length_px:
            e = cal.body_rel_err
            bits.append(f"animal {cal.body_length_px:.1f} source px"
                        + (f" ±{e * 100:.1f}%" if e else ""))
        bits.append(f"view {self.view.zoom:g}x")
        return "   ·   ".join(bits)

    def _result_text(self, cal: Calibration) -> str:
        bits = []
        px_bl = cal.working_px_per_body_length(self._scale)
        if px_bl is not None:
            bits.append(f"{px_bl:.1f} working px per body length "
                        f"at scale {self._scale:.2f}")
        if cal.pixels_per_mm is not None:
            bits.append(f"{cal.pixels_per_mm:.3f} px/mm")
        blm = cal.body_length_mm
        if blm is not None:
            e = cal.body_length_mm_rel_err
            bits.append(f"body {blm:.2f} mm"
                        + (f" ±{e * 100:.1f}%" if e else ""))
        if not bits:
            return "Nothing measured yet — drag a line across the animal."
        return "   ·   ".join(bits)

    def _warn_text(self, cal: Calibration) -> str:
        out = []
        # 10% is where the propagated error starts to swamp the decision the
        # number feeds: at that point 0.5 and 0.35 are indistinguishable in
        # working px per body length, which is precisely the comparison the
        # downsample dialog asks the user to make.
        for name, err in (("ruler", cal.fiducial_rel_err),
                          ("animal", cal.body_rel_err)):
            if err is not None and err > 0.10:
                out.append(
                    f"The {name} line carries ±{err * 100:.0f}%: it is short "
                    f"relative to the zoom it was drawn at. Zoom in and redraw, "
                    f"or draw along something longer.")
        if self._ruler is not None and self.known_mm.value() <= 0:
            out.append("Enter the ruler's true length in mm to use it.")
        if cal.pixels_per_mm is None and cal.body_length_px is not None:
            out.append(
                "No ruler, so the body length cannot be stored in millimetres — "
                "the working-px-per-body-length figure above is still exact, but "
                "it will not transfer to another clip or reach exports.")
        return "  ".join(out)

    def _apply(self):
        self.calibration = self._build_calibration()
        self.accept()
