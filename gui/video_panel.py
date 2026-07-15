"""Video display with playback controls, overlay, and click-to-inspect.

Playback conventions and hotkeys follow the reference color detector so the two
tools feel like siblings: Space toggles play, arrows step a frame, shift+arrows
step a second, Home/End jump to the ends, and the scrub bar is always live.

Where the reference samples a colour on click, this samples an ROI: clicking maps
the pixel back to a block and asks the state which ROI (if any) owns it.
"""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor, QImage, QPainter, QPen, QColor, QPixmap
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                             QSlider, QStyle, QStyleOptionSlider, QVBoxLayout,
                             QWidget)

from gui.state import AppState

# Every per-frame display operation runs at THIS width, not the source width.
#
# The source here is 5312x2988. Copying that frame, upscaling the block mask to
# it, alpha-blending a tint over it and converting BGR->RGB costs ~250 ms per
# frame -- and it is all thrown away, because the widget it lands in is about
# 700 px wide. Doing the same work at 1280 px is ~15x less pixel traffic and
# takes playback from roughly 1.5 fps to real time. Nothing is lost: the block
# grid is 45x81, so the overlay has nowhere near enough detail to justify being
# composited at 5K.
DISPLAY_MAX_W = 1280


class FrameView(QLabel):
    """Displays a frame scaled to fit, and maps clicks back to frame coords.

    Also supports a rectangle-drawing mode: when draw_enabled is on, dragging
    rubber-bands a box and emits box_drawn with its corners as fractions of the
    frame (0-1). Fractions rather than pixels so the consumer is independent of
    both the display scale and the source resolution.
    """
    clicked = pyqtSignal(QPoint)
    box_drawn = pyqtSignal(float, float, float, float)   # x0,y0,x1,y1 fractions
    stamp_at = pyqtSignal(float, float)                  # click point, fractions

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 320)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self._pix: QPixmap | None = None
        self._draw_rect = QRect()
        self._src_size = (1, 1)

        self.draw_enabled = False
        # Persistent boxes to render: [(x0,y0,x1,y1 fractions, label, hex, selected)]
        self.boxes: list[tuple] = []
        self._rubber: tuple | None = None    # in-progress drag, display coords
        self._drag_start: QPoint | None = None

    def set_boxes(self, boxes: list[tuple]) -> None:
        self.boxes = boxes
        self.update()

    def _frac_of(self, pos: QPoint) -> tuple[float, float]:
        r = self._draw_rect
        fx = (pos.x() - r.x()) / max(1, r.width())
        fy = (pos.y() - r.y()) / max(1, r.height())
        return float(np.clip(fx, 0, 1)), float(np.clip(fy, 0, 1))

    def _rect_for_frac(self, x0, y0, x1, y1) -> QRect:
        r = self._draw_rect
        return QRect(
            int(r.x() + x0 * r.width()), int(r.y() + y0 * r.height()),
            int((x1 - x0) * r.width()), int((y1 - y0) * r.height()))

    def set_frame(self, img: np.ndarray):
        h, w = img.shape[:2]
        self._src_size = (w, h)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._pix = QPixmap.fromImage(qimg.copy())
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1a1a1a"))
        if self._pix is None:
            p.setPen(QColor("#777"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Open a video (File > Open Video)")
            p.end()
            return

        area = self.rect().adjusted(1, 1, -1, -1)
        scaled = self._pix.size().scaled(area.size(),
                                         Qt.AspectRatioMode.KeepAspectRatio)
        self._draw_rect = QRect(
            area.x() + (area.width() - scaled.width()) // 2,
            area.y() + (area.height() - scaled.height()) // 2,
            scaled.width(), scaled.height())
        p.drawPixmap(self._draw_rect, self._pix)

        # Persistent boxes.
        for x0, y0, x1, y1, label, hexcol, selected in self.boxes:
            rect = self._rect_for_frac(x0, y0, x1, y1)
            col = QColor(hexcol)
            p.setPen(QPen(col, 3 if selected else 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(rect)
            if selected:
                fill = QColor(col)
                fill.setAlpha(40)
                p.fillRect(rect, fill)
            if label:
                p.setPen(QPen(col, 1))
                p.drawText(rect.x() + 3, rect.y() + 14, label)

        # Rubber-band in progress.
        if self._rubber is not None:
            p.setPen(QPen(QColor("#ffd24a"), 2, Qt.PenStyle.DashLine))
            p.setBrush(QColor(255, 210, 74, 40))
            p.drawRect(self._rubber)

        p.setPen(QPen(QColor("#333"), 1))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        p.end()

    def mousePressEvent(self, e):
        if self._pix is None or not self._draw_rect.contains(e.pos()):
            return
        if self.draw_enabled and e.button() == Qt.MouseButton.LeftButton:
            self._drag_start = e.pos()
            self._rubber = QRect(e.pos(), e.pos())
            return
        fx = (e.pos().x() - self._draw_rect.x()) / self._draw_rect.width()
        fy = (e.pos().y() - self._draw_rect.y()) / self._draw_rect.height()
        self.clicked.emit(QPoint(int(fx * self._src_size[0]),
                                 int(fy * self._src_size[1])))

    def mouseMoveEvent(self, e):
        if self._drag_start is not None:
            self._rubber = QRect(self._drag_start, e.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, e):
        if self._drag_start is None:
            return
        rect = QRect(self._drag_start, e.pos()).normalized()
        self._drag_start = None
        self._rubber = None
        self.update()
        # A tiny drag is a click. In stamp mode that means "drop a fixed-size box
        # here"; otherwise ignore it.
        if rect.width() < 5 or rect.height() < 5:
            fx, fy = self._frac_of(e.pos())
            self.stamp_at.emit(fx, fy)
            return
        x0, y0 = self._frac_of(rect.topLeft())
        x1, y1 = self._frac_of(rect.bottomRight())
        self.box_drawn.emit(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


class JumpSlider(QSlider):
    """A slider that jumps to where you click, instead of paging toward it.

    Qt's default is to advance by one pageStep per click on the groove, which on
    a 30,600-frame video means a click near the end nudges you forward by a few
    seconds. For a scrub bar that behaviour is simply wrong: clicking a position
    means "go there".
    """

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            style = self.style()
            handle = style.subControlRect(
                QStyle.ComplexControl.CC_Slider, opt,
                QStyle.SubControl.SC_SliderHandle, self)
            if not handle.contains(e.pos()):
                groove = style.subControlRect(
                    QStyle.ComplexControl.CC_Slider, opt,
                    QStyle.SubControl.SC_SliderGroove, self)
                span = groove.width() - handle.width()
                pos = e.pos().x() - groove.x() - handle.width() // 2
                val = QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(), int(pos), int(max(1, span)))
                self.setValue(val)
                # Fall through to the base handler so the handle -- now under the
                # cursor -- immediately picks up a drag if the user keeps holding.
        super().mousePressEvent(e)


class VideoPanel(QWidget):
    """Frame view + transport. Overlays are supplied by whichever tab owns it."""

    block_clicked = pyqtSignal(int, int)   # block y, x

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._overlay: np.ndarray | None = None       # (ny, nx) bool
        self._overlay_color = (60, 200, 255)
        self._roi_boxes: list[tuple[int, tuple, str]] = []
        self._cache_frame: np.ndarray | None = None
        self._cache_idx = -1
        self._pending = False
        self._tint_cache: tuple | None = None

        self.view = FrameView()
        self.view.clicked.connect(self._on_click)

        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedWidth(70)
        # No focus, or a focused Play button swallows the Space shortcut (Qt maps
        # Space to "activate the focused button"), and Space stops working the
        # moment you have clicked anything.
        self.play_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.play_btn.clicked.connect(self.toggle_playback)

        self.scrubber = JumpSlider(Qt.Orientation.Horizontal)
        self.scrubber.setEnabled(False)
        self.scrubber.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Dragging the scrub bar emits valueChanged for EVERY intermediate value
        # the handle passes over -- dozens per second. Wiring that straight to
        # set_frame asks the decoder for every frame between where you started and
        # where you let go, and each one is a ~400 ms random seek on this footage.
        # The result is a huge backlog that looks like the app is decoding the
        # whole range you dragged across, because it is.
        #
        # Instead, coalesce: remember the latest requested frame, and only decode
        # once the handle has been still for a moment. The time readout still
        # updates instantly, so the drag stays responsive.
        self._pending_frame = 0
        self._scrub_timer = QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(60)
        self._scrub_timer.timeout.connect(
            lambda: self.state.set_frame(self._pending_frame))
        self.scrubber.valueChanged.connect(self._on_scrub)

        self.time_label = QLabel("00:00.00 / 00:00.00")
        self.time_label.setStyleSheet("font-family: Consolas; color: #ccc;")
        self.time_label.setFixedWidth(170)

        bar = QHBoxLayout()
        bar.addWidget(self.play_btn)
        bar.addWidget(self.scrubber, 1)
        bar.addWidget(self.time_label)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.view, 1)
        lay.addLayout(bar)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)

        self.state.video_loaded.connect(self._on_video_loaded)
        self.state.cache_opened.connect(self._on_video_loaded)
        self.state.frame_changed.connect(self._on_frame_changed)

    # -- overlays ------------------------------------------------------------

    def set_overlay(self, mask: np.ndarray | None,
                    color: tuple[int, int, int] = (60, 200, 255)) -> None:
        self._overlay = mask
        self._overlay_color = color
        self.refresh()

    def set_roi_boxes(self, boxes: list[tuple[int, tuple, str]]) -> None:
        """[(roi_id, (y0,x0,y1,x1) in blocks, hex_color)]"""
        self._roi_boxes = boxes
        self.refresh()

    def set_draw_mode(self, on: bool) -> None:
        self.view.draw_enabled = on
        self.view.setCursor(QCursor(
            Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor))

    def set_frac_boxes(self, boxes: list[tuple]) -> None:
        """Boxes in frame fractions, drawn directly on the view (not the block
        grid): [(x0,y0,x1,y1, label, hex, selected)]."""
        self.view.set_boxes(boxes)

    # -- playback ------------------------------------------------------------

    def _on_scrub(self, value: int):
        """Slider moved: update the readout now, decode after it settles."""
        self._pending_frame = int(value)
        self._update_time(frame=self._pending_frame)
        self._scrub_timer.start()

    def toggle_playback(self):
        if not self.state.has_video:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
            return

        # Play from wherever we are. If we are sitting at the end, loop back to
        # the start rather than refusing to do anything -- pressing play on a
        # finished video obviously means "play it again".
        if self.state.current_frame >= self._n_frames() - 1:
            self.state.set_frame(0)
        self.timer.start(max(1, int(1000 / self.state.fps)))
        self.play_btn.setText("Pause")

    def _advance(self):
        n = self._n_frames()
        if self.state.current_frame >= n - 1:
            self.timer.stop()
            self.play_btn.setText("Play")
            return
        self.state.set_frame(self.state.current_frame + 1)

    def step(self, delta: int):
        self.state.set_frame(self.state.current_frame + delta)

    def _n_frames(self) -> int:
        if self.state.cache is not None:
            return self.state.cache.n_frames
        return self.state.source.info.frame_count if self.state.source else 1

    # -- rendering -----------------------------------------------------------

    def _on_video_loaded(self):
        n = self._n_frames()
        self.scrubber.setEnabled(True)
        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, max(0, n - 1))
        self.scrubber.setValue(0)
        self.scrubber.blockSignals(False)
        self._cache_idx = -1
        self.refresh()

    def _on_frame_changed(self, idx: int):
        if self.scrubber.value() != idx:
            self.scrubber.blockSignals(True)
            self.scrubber.setValue(idx)
            self.scrubber.blockSignals(False)
        self.refresh()

    def refresh(self):
        """Request a repaint, coalescing multiple requests in the same event-loop
        turn into one.

        A single frame step used to trigger three full repaints: this panel's own
        frame_changed handler, the OTHER tab's panel (which is hidden but still
        connected), and then again when the owning tab pushed a new overlay or ROI
        box set. Each repaint is a full decode-scale-blend-QPixmap cycle. Debounce
        them, and skip entirely when the panel is not on screen.
        """
        if not self.state.has_video or self._pending:
            return
        self._pending = True
        QTimer.singleShot(0, self._do_refresh)

    def _do_refresh(self):
        self._pending = False
        if not self.state.has_video or not self.isVisible():
            return
        idx = self.state.current_frame
        # Decoding is centralised in AppState so the two VideoPanels (Tab 2 and
        # Tab 3) share one decode per frame instead of fighting each other for
        # the decoder position. See AppState.display_frame().
        frame = self.state.display_frame(idx)
        if frame is None:
            return
        self._cache_frame = frame
        self._cache_idx = idx

        img = frame.copy()
        h, w = img.shape[:2]

        if self._overlay is not None and self._overlay.any():
            m = cv2.resize(self._overlay.astype(np.uint8) * 255, (w, h),
                           interpolation=cv2.INTER_NEAREST)
            # Blend with addWeighted + copyTo rather than boolean fancy indexing.
            # `img[sel] = 0.45*img[sel] + 0.55*tint[sel]` builds several temporary
            # arrays the size of the selection and runs the arithmetic in float64;
            # this stays in uint8 inside OpenCV and is roughly an order of
            # magnitude faster. The tint plane is constant, so cache it.
            if self._tint_cache is None or self._tint_cache[0] != (h, w, self._overlay_color):
                tint = np.empty_like(img)
                tint[:, :] = self._overlay_color[::-1]   # to BGR
                self._tint_cache = ((h, w, self._overlay_color), tint)
            tint = self._tint_cache[1]
            blended = cv2.addWeighted(img, 0.45, tint, 0.55, 0.0)
            np.copyto(img, blended, where=(m > 0)[:, :, None])

        if self._roi_boxes and self.state.cache is not None:
            ny, nx = self.state.cache.grid
            sy, sx = h / ny, w / nx
            for roi_id, (y0, x0, y1, x1), col in self._roi_boxes:
                bgr = tuple(int(col.lstrip("#")[i:i + 2], 16)
                            for i in (4, 2, 0))
                thick = 4 if roi_id == self.state.selected_roi else 2
                cv2.rectangle(img, (int(x0 * sx), int(y0 * sy)),
                              (int(x1 * sx), int(y1 * sy)), bgr, thick)
                cv2.putText(img, f"#{roi_id}", (int(x0 * sx), int(y0 * sy) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)

        self.view.set_frame(img)
        self._update_time()

    def _update_time(self, frame: int | None = None):
        n = self._n_frames()
        fps = self.state.fps
        f = self.state.current_frame if frame is None else frame
        cur, tot = f / fps, n / fps
        self.time_label.setText(
            f"{int(cur // 60):02d}:{cur % 60:05.2f} / "
            f"{int(tot // 60):02d}:{tot % 60:05.2f}")
        self.time_label.setToolTip(f"frame {f} of {n}")

    def _on_click(self, pt: QPoint):
        if self.state.cache is None or self._cache_frame is None:
            return
        ny, nx = self.state.cache.grid
        # FrameView reports coordinates in the image it was handed, which is the
        # DOWNSCALED frame -- not the source. Dividing by the source dimensions
        # here would place every click near the top-left corner.
        h, w = self._cache_frame.shape[:2]
        by = int(pt.y() / h * ny)
        bx = int(pt.x() / w * nx)
        if 0 <= by < ny and 0 <= bx < nx:
            self.block_clicked.emit(by, bx)
