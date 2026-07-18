"""Whole-clip detection navigator: the data becomes the navigation.

After the whole-video commit pass, this strip shows every detection over the full
clip -- the positive gate plus the largest-clump strength -- and lets you click a
detection (or step to the strongest ones) to load that ~10 s window back into the
live view and verify it against the footage. You never scrub blind through the
whole clip; the detector points you at what to check.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from gui.explorers.speed_explorer import BG, CURSOR, PLOT_BG, TXT, TXT_DIM

_GATE_C = QColor(90, 230, 110)
_CLUMP_TOP = np.array([255, 240, 210], np.float64)
_CLUMP_BOT = np.array([70, 25, 30], np.float64)


class _Strip(QWidget):
    """Painted clump-strength profile with the positive gate marked, clickable to
    a frame. Downsamples to the widget width, so a 30k-frame clip paints cheaply."""
    clicked = pyqtSignal(int)      # frame index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(72)
        self.gate = np.zeros(0, np.float32)
        self.clump = np.zeros(0, np.float32)
        self.T = 0
        self.cursor = 0
        self._img = None
        self._img_w = -1
        self.setMouseTracking(False)

    def set_result(self, gate: np.ndarray, clump: np.ndarray) -> None:
        self.gate = np.asarray(gate, np.float32)
        self.clump = np.asarray(clump, np.float32)
        self.T = int(self.gate.size)
        self._img = None
        self.update()

    def set_cursor(self, frame: int) -> None:
        self.cursor = int(frame)
        self.update()

    def _build_image(self, w: int, h: int):
        if self.T == 0 or w <= 0 or h <= 0:
            return None
        bounds = np.linspace(0, self.T, w + 1).astype(int)
        cmax = max(1e-6, float(self.clump.max()))
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[:] = (18, 18, 22)
        for x in range(w):
            a, b = bounds[x], max(bounds[x] + 1, bounds[x + 1])
            seg_c = float(self.clump[a:b].max()) if b > a else 0.0
            seg_g = float(self.gate[a:b].max()) if b > a else 0.0
            frac = min(1.0, seg_c / cmax)
            bar = int(frac * (h - 6))
            if bar > 0:
                col = (_CLUMP_BOT * (1 - frac) + _CLUMP_TOP * frac).astype(np.uint8)
                rgb[h - 6 - bar:h - 6, x] = col
            if seg_g > 0.5:                          # gate band along the bottom
                rgb[h - 5:h, x] = (90, 230, 110)
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
            p.drawText(8, 16, "no whole-video result yet — press Process whole video")
            p.end()
            return
        if self._img is None or self._img_w != w:
            self._img = self._build_image(w, h)
            self._img_w = w
        if self._img is not None:
            p.drawImage(0, 0, self._img)
        cx = int((self.cursor + 0.5) / max(1, self.T) * w)
        p.setPen(QPen(CURSOR, 1))
        p.drawLine(cx, 0, cx, h)
        p.end()

    def mousePressEvent(self, e):
        if self.T == 0:
            return
        w = max(1, self.width() - 1)
        frame = int(np.clip(e.position().x() / w * self.T, 0, self.T - 1))
        self.clicked.emit(frame)


class DetectionNavigator(QWidget):
    """Strip + summary + strongest-first stepping. Emits the center frame of a
    detection to focus; the surface loads a window around it."""
    focus_requested = pyqtSignal(int)      # absolute center frame

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fps = 30.0
        self._intervals: list[tuple[int, int]] = []     # (start, end) absolute
        self._by_strength: list[int] = []               # indices, strongest first
        self._cur = -1

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.summary = QLabel("Process the whole video to navigate detections.")
        self.summary.setStyleSheet("color:#cbd; font-size:11px;")
        lay.addWidget(self.summary)
        self.strip = _Strip()
        self.strip.clicked.connect(self._on_click)
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
        lay.addLayout(row)

    def set_result(self, res, fps: float) -> None:
        self.fps = float(fps)
        self.strip.set_result(res.gate, res.clump)
        self._intervals = res.detected_intervals()
        clump = np.asarray(res.clump, np.float32)
        ws = int(getattr(res, "window_start", 0))
        # Strength = peak clump within each interval; step strongest-first.
        strengths = []
        for (s, e) in self._intervals:
            a, b = s - ws, e - ws
            strengths.append(float(clump[a:b].max()) if b > a else 0.0)
        self._by_strength = list(np.argsort(strengths)[::-1]) if strengths else []
        self._cur = -1
        n = len(self._intervals)
        total_frames = int(np.asarray(res.gate).sum())
        peak = max(strengths) if strengths else 0.0
        self.summary.setText(
            f"{n} detection{'s' if n != 1 else ''} · "
            f"{total_frames / self.fps:.1f} s detected total · "
            f"peak clump {peak:.0f} blocks")
        self.prev_btn.setEnabled(n > 0)
        self.next_btn.setEnabled(n > 0)

    def set_cursor(self, frame: int) -> None:
        self.strip.set_cursor(frame)

    def _center_of(self, interval) -> int:
        s, e = interval
        return int((s + e) // 2)

    def _on_click(self, frame: int):
        # Focus the interval containing the click, else just the clicked frame.
        for iv in self._intervals:
            if iv[0] <= frame < iv[1]:
                self.focus_requested.emit(self._center_of(iv))
                return
        self.focus_requested.emit(int(frame))

    def _step(self, delta: int):
        if not self._by_strength:
            return
        self._cur = (self._cur + delta) % len(self._by_strength)
        iv = self._intervals[self._by_strength[self._cur]]
        self.focus_requested.emit(self._center_of(iv))
