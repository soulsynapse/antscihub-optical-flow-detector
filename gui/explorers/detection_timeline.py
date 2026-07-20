"""The whole-clip detection strip: the data becomes the navigation.

This is the primary position control for the clip. It is always present, it fills
in as detection runs -- live or committed -- and clicking or dragging anywhere on
it seeks there, whether or not a detection has been computed at that point. The
older behaviour (appear after a commit pass, click only to jump between
detections) made it a report; it is a seeker that happens to be drawn out of the
detector's own output.

**What it must never do is let three different claims look alike.** The strip
paints, per column:

* **unexamined** -- nobody has computed this stretch. Drawn as bare trough with
  no baseline rule, visibly *empty* rather than dark.
* **examined, quiet** -- computed, and the detector says nothing is there. Drawn
  with the baseline rule lit, so "we looked" is visible independently of whether
  anything was found.
* **examined under other settings** -- computed, but the channel, frequency band
  or geometry has moved since. Drawn desaturated.

The first two collapsing into each other is the standing failure of this codebase
(FINDINGS.md section 10, traps 7 and 8): a strip that paints unfilled regions the
same colour as a computed zero turns "nobody looked here" into "nothing happened
here", which is a false negative wearing the costume of a result. The baseline
rule exists for exactly that and is not decoration.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                             QWidget)

from gui.explorers.speed_explorer import BG, CURSOR, PLOT_BG, TXT, TXT_DIM

# Examined-and-current. The clump ramp runs cool-to-hot; the gate band and the
# "we looked" baseline are the same green so they read as one statement.
_GATE_C = np.array([90, 230, 110], np.float64)
_CLUMP_TOP = np.array([255, 240, 210], np.float64)
_CLUMP_BOT = np.array([70, 25, 30], np.float64)
# Examined under settings no longer in force: the same shapes, drained of hue.
# Deliberately not merely dimmed -- a dim red still reads as a weak detection,
# whereas a gray one reads as a detection that is not being claimed.
_STALE_TOP = np.array([150, 150, 158], np.float64)
_STALE_BOT = np.array([48, 48, 54], np.float64)
_STALE_GATE = np.array([110, 118, 112], np.float64)
# Unexamined trough. Darker than PLOT_BG so "empty" is a different surface from
# "flat", not a different shade of the same one.
_UNSEEN = np.array([10, 10, 12], np.float64)
_SEEN_BG = np.array([26, 26, 32], np.float64)
# Rows reserved at the bottom of the strip for the gate band and, under it, the
# coverage baseline rule.
_GATE_ROWS = 5
_BASE_ROWS = 2


class _Strip(QWidget):
    """Painted clump profile over the whole clip, with coverage and staleness,
    scrubbable. Downsampled to the widget width, so a 30k-frame clip paints
    cheaply and the image is rebuilt only when the data or the width changes."""
    # Continuous while dragging: cheap consumers only (move the cursor, show the
    # frame). Kept separate from `seek_committed` so a drag across the clip does
    # not fire one re-extract per pixel.
    scrubbed = pyqtSignal(int)
    seek_committed = pyqtSignal(int)        # release: the expensive action

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(84)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.T = 0
        self.cursor = 0
        self._clump = np.zeros(0, np.float32)
        self._gate = np.zeros(0, np.float32)
        self._current = np.zeros(0, bool)
        self._covered = np.zeros(0, bool)
        self._img = None
        self._img_w = -1
        self._dragging = False

    # -- data ----------------------------------------------------------------
    def set_track(self, track) -> None:
        """Take the whole-video track. Read by value rather than held, so a
        worker mutating the track cannot repaint half-updated arrays."""
        self._clump = np.asarray(track.clump, np.float32).copy()
        self._gate = np.asarray(track.gate, np.float32).copy()
        self._current = np.asarray(track.current, bool).copy()
        self._covered = np.asarray(track.covered, bool).copy()
        self.T = int(track.n_frames)
        self._img = None
        self.update()

    def set_span(self, n_frames: int) -> None:
        """Establish the clip length before anything has been computed, so the
        strip is a usable seeker from the moment the tab opens rather than an
        empty box that says to press a button."""
        if int(n_frames) == self.T:
            return
        self.T = int(n_frames)
        self._clump = np.zeros(self.T, np.float32)
        self._gate = np.zeros(self.T, np.float32)
        self._current = np.zeros(self.T, bool)
        self._covered = np.zeros(self.T, bool)
        self._img = None
        self.update()

    def set_cursor(self, frame: int) -> None:
        f = int(np.clip(frame, 0, max(0, self.T - 1)))
        if f == self.cursor:
            return
        self.cursor = f
        self.update()          # cursor only: the cached image is not rebuilt

    # -- painting ------------------------------------------------------------
    def _build_image(self, w: int, h: int):
        if self.T == 0 or w <= 0 or h <= 0:
            return None
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[:] = _UNSEEN
        # One source column per screen column. reduceat over the segment bounds
        # rather than a Python loop: at 30k frames this runs on every live tick.
        bounds = np.linspace(0, self.T, w + 1).astype(int)
        lo = np.minimum(bounds[:-1], self.T - 1)
        seg_c = np.maximum.reduceat(self._clump, lo)
        seg_g = np.maximum.reduceat(self._gate, lo)
        # A column is only claimed as covered/current if EVERY frame in it is:
        # `min` rather than `max`, so a column straddling the frontier reads as
        # partly-unexamined rather than as examined. At 30k frames over ~1200
        # columns each column is ~25 frames, which is enough for the difference
        # to matter at the edge of a live pass.
        seg_seen = np.minimum.reduceat(self._covered.view(np.int8), lo) > 0
        seg_cur = np.minimum.reduceat(self._current.view(np.int8), lo) > 0

        top = h - _GATE_ROWS - _BASE_ROWS
        rgb[:top][:, seg_seen] = _SEEN_BG
        cmax = max(1e-6, float(self._clump[self._covered].max())
                   if self._covered.any() else 1e-6)
        frac = np.clip(seg_c / cmax, 0.0, 1.0)
        bars = (frac * max(0, top - 2)).astype(int)
        for x in np.flatnonzero(seg_seen):
            n = int(bars[x])
            if n <= 0:
                continue
            f = float(frac[x])
            hot, cold = ((_CLUMP_TOP, _CLUMP_BOT) if seg_cur[x]
                         else (_STALE_TOP, _STALE_BOT))
            rgb[top - n:top, x] = (cold * (1 - f) + hot * f).astype(np.uint8)
        # Gate band, then the coverage rule beneath it. The rule is what makes
        # examined-and-quiet distinguishable from unexamined at a glance.
        fired = seg_seen & (seg_g > 0.5)
        for x in np.flatnonzero(fired):
            rgb[top:top + _GATE_ROWS, x] = (_GATE_C if seg_cur[x]
                                            else _STALE_GATE)
        base = h - _BASE_ROWS
        for x in np.flatnonzero(seg_seen):
            rgb[base:h, x] = (_GATE_C * 0.45 if seg_cur[x]
                              else _STALE_GATE * 0.6)
        return QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w,
                      QImage.Format.Format_RGB888).copy()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), BG)
        r = self.rect().adjusted(0, 0, -1, -1)
        p.fillRect(r, PLOT_BG)
        w, h = r.width(), r.height()
        if self.T == 0:
            p.setPen(TXT_DIM)
            p.setFont(QFont("Consolas", 8))
            p.drawText(8, 16, "no clip loaded")
            p.end()
            return
        if self._img is None or self._img_w != w:
            self._img = self._build_image(w, h)
            self._img_w = w
        if self._img is not None:
            p.drawImage(0, 0, self._img)
        if not self._covered.any():
            p.setPen(TXT_DIM)
            p.setFont(QFont("Consolas", 8))
            p.drawText(8, 16, "nothing examined yet — drag to seek, "
                              "press Live ▶ or Process whole video")
        cx = int((self.cursor + 0.5) / max(1, self.T) * w)
        p.setPen(QPen(CURSOR, 1))
        p.drawLine(cx, 0, cx, h)
        p.end()

    # -- scrubbing -----------------------------------------------------------
    def _frame_at(self, x: float) -> int:
        w = max(1, self.width() - 1)
        return int(np.clip(x / w * self.T, 0, max(0, self.T - 1)))

    def mousePressEvent(self, e):
        if self.T == 0 or e.button() != Qt.MouseButton.LeftButton:
            return
        self._dragging = True
        f = self._frame_at(e.position().x())
        self.set_cursor(f)
        self.scrubbed.emit(f)

    def mouseMoveEvent(self, e):
        if not self._dragging:
            return
        f = self._frame_at(e.position().x())
        self.set_cursor(f)
        self.scrubbed.emit(f)

    def mouseReleaseEvent(self, e):
        if not self._dragging or e.button() != Qt.MouseButton.LeftButton:
            return
        self._dragging = False
        self.seek_committed.emit(self._frame_at(e.position().x()))


class DetectionNavigator(QWidget):
    """The strip plus its readout and strongest-first stepping.

    Always visible: it is the clip's position control, so hiding it until a
    commit pass lands would remove the seeker from a tab that has one.
    """
    scrubbed = pyqtSignal(int)              # cheap: cursor moved
    seek_committed = pyqtSignal(int)        # expensive: load this position
    focus_requested = pyqtSignal(int)       # a detection was chosen

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fps = 30.0
        self._intervals: list[tuple[int, int]] = []
        self._by_strength: list[int] = []
        self._cur = -1

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.summary = QLabel("Drag anywhere to seek. Nothing examined yet.")
        self.summary.setStyleSheet("color:#cbd; font-size:11px;")
        lay.addWidget(self.summary)
        self.strip = _Strip()
        self.strip.scrubbed.connect(self.scrubbed)
        self.strip.seek_committed.connect(self._on_committed)
        lay.addWidget(self.strip)

        row = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev strongest")
        self.next_btn = QPushButton("Next strongest ▶")
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn.clicked.connect(lambda: self._step(+1))
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        row.addWidget(self.prev_btn)
        row.addWidget(self.next_btn)
        row.addStretch(1)
        self.legend = QLabel()
        self.legend.setStyleSheet("color:#8a8a96; font-size:10px;")
        self.legend.setText("▁ lit = examined · gray = other settings · "
                            "dark = not examined")
        row.addWidget(self.legend)
        lay.addLayout(row)

    def set_span(self, n_frames: int, fps: float) -> None:
        self.fps = float(fps)
        self.strip.set_span(n_frames)

    def set_track(self, track) -> None:
        """Repaint from the whole-video track and re-derive the navigation."""
        self.fps = float(track.fps) or self.fps
        self.strip.set_track(track)
        self._intervals = track.detected_intervals(current_only=True)
        clump = np.asarray(track.clump, np.float32)
        strengths = [float(clump[s:e].max()) if e > s else 0.0
                     for (s, e) in self._intervals]
        self._by_strength = list(np.argsort(strengths)[::-1]) if strengths else []
        self._cur = -1
        n = len(self._intervals)
        cov = track.coverage_fraction()
        det_s = float(np.asarray(track.gate)[track.current].sum()) / max(self.fps, 1e-6)
        stale = int(track.stale.sum())
        stale_note = (f" · {stale / max(self.fps, 1e-6):.0f} s under other "
                      f"settings" if stale else "")
        if cov <= 0.0:
            self.summary.setText(
                f"Drag anywhere to seek. Nothing examined yet{stale_note}.")
        else:
            self.summary.setText(
                f"{cov * 100:.0f}% of the clip examined · "
                f"{n} detection{'s' if n != 1 else ''} · "
                f"{det_s:.1f} s detected{stale_note}")
        self.prev_btn.setEnabled(n > 0)
        self.next_btn.setEnabled(n > 0)

    def set_cursor(self, frame: int) -> None:
        self.strip.set_cursor(frame)

    def _on_committed(self, frame: int):
        # A release inside a detection focuses it (the window is centred on the
        # event, which is what you want to verify); anywhere else is a plain
        # seek to the frame under the pointer.
        for iv in self._intervals:
            if iv[0] <= frame < iv[1]:
                self.focus_requested.emit(int((iv[0] + iv[1]) // 2))
                return
        self.seek_committed.emit(int(frame))

    def _step(self, delta: int):
        if not self._by_strength:
            return
        self._cur = (self._cur + delta) % len(self._by_strength)
        s, e = self._intervals[self._by_strength[self._cur]]
        self.focus_requested.emit(int((s + e) // 2))
