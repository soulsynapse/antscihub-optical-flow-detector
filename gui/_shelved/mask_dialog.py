"""Draw an optional within-replicate mask directly on a video frame.

The user draws one or more rectangles marking the regions to KEEP; everything
outside them is suppressed after the replicate rectangles are decoded. Replicate
boxes already provide the processing boundary; this tool is only for excluding a
fixed nuisance inside one or more boxes.

Writes a full-resolution white-on-black PNG (white = keep) that the preprocessor
loads exactly like an externally-authored mask, so nothing downstream needs to
know the mask was drawn here.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout)

from gui.video_panel import FrameView


class MaskDrawDialog(QDialog):
    def __init__(self, frame_bgr: np.ndarray, out_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw within-replicate mask — keep-regions")
        self.resize(1100, 780)
        self.out_path = out_path
        self.src_h, self.src_w = frame_bgr.shape[:2]
        self.boxes: list[tuple] = []       # (x0,y0,x1,y1) fractions

        lay = QVBoxLayout(self)
        info = QLabel(
            "Drag rectangles over the pixels to KEEP inside your replicate "
            "boxes. Everything else is suppressed, but cache size and processing "
            "area do not change. Keep mask edges away from animal movement.")
        info.setWordWrap(True)
        lay.addWidget(info)

        self.view = FrameView()
        self.view.draw_enabled = True
        self.view.set_frame(frame_bgr)
        self.view.box_drawn.connect(self._on_box)
        lay.addWidget(self.view, 1)

        row = QHBoxLayout()
        undo = QPushButton("Undo last")
        undo.clicked.connect(self._undo)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear)
        row.addWidget(undo)
        row.addWidget(clear)
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        ok = QPushButton("Save mask")
        ok.setDefault(True)
        ok.clicked.connect(self._save)
        row.addWidget(cancel)
        row.addWidget(ok)
        lay.addLayout(row)

    def _redraw(self):
        self.view.set_boxes([
            (*b, "keep", "#4ac6ff", False) for b in self.boxes])

    def _on_box(self, x0, y0, x1, y1):
        self.boxes.append((x0, y0, x1, y1))
        self._redraw()

    def _undo(self):
        if self.boxes:
            self.boxes.pop()
            self._redraw()

    def _clear(self):
        self.boxes = []
        self._redraw()

    def _save(self):
        if not self.boxes:
            self.reject()
            return
        mask = np.zeros((self.src_h, self.src_w), np.uint8)
        for x0, y0, x1, y1 in self.boxes:
            cv2.rectangle(
                mask,
                (int(x0 * self.src_w), int(y0 * self.src_h)),
                (int(x1 * self.src_w), int(y1 * self.src_h)),
                255, thickness=-1)
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        cv2.imwrite(self.out_path, mask)
        self.accept()
