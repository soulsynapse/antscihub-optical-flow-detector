"""Main window: three tabs over one shared AppState.

Hotkeys follow the reference color detector so the two tools feel the same:
Space plays/pauses, arrows step a frame, shift+arrows step a second, Home/End
jump to the ends, Ctrl+1/2/3 switch tabs.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (QFileDialog, QLabel, QMainWindow, QMessageBox,
                             QTabWidget)

from gui.state import AppState
from gui.tab1_flow import Tab1Flow
from gui.tab2_replicates import Tab2Replicates
from gui.tab3_behavior import Tab3Behavior


class MainWindow(QMainWindow):
    def __init__(self, project_dir: str = "."):
        super().__init__()
        self.setWindowTitle("Optical Flow Behavior Detector")
        self.setMinimumSize(1400, 860)
        self.resize(1750, 980)

        self.state = AppState(project_dir)

        self.tabs = QTabWidget()
        self.tab1 = Tab1Flow(self.state)
        self.tab2 = Tab2Replicates(self.state)
        self.tab3 = Tab3Behavior(self.state)
        self.tabs.addTab(self.tab1, "1 · Preprocessing && Flow")
        self.tabs.addTab(self.tab2, "2 · Replicates")
        self.tabs.addTab(self.tab3, "3 · Behavior Classification")
        self.setCentralWidget(self.tabs)

        # Tabs 2 and 3 are meaningless without a cache; leaving them clickable
        # just invites confusing empty panels.
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)

        self.status = QLabel("Open a video to begin.")
        self.statusBar().addWidget(self.status)

        self.state.status.connect(self.status.setText)
        self.state.cache_opened.connect(self._on_cache_opened)
        self.state.request_tab.connect(self.tabs.setCurrentIndex)

        self._menu()
        self._shortcuts()

    def _menu(self):
        f = self.menuBar().addMenu("File")
        a = f.addAction("Open Video…")
        a.setShortcut(QKeySequence.StandardKey.Open)
        a.triggered.connect(self._open_video)
        f.addSeparator()
        q = f.addAction("Quit")
        q.triggered.connect(self.close)

        h = self.menuBar().addMenu("Help")
        h.addAction("About").triggered.connect(self._about)

    def _shortcuts(self):
        def sc(seq, fn):
            s = QShortcut(QKeySequence(seq), self)
            s.setContext(Qt.ShortcutContext.ApplicationShortcut)
            s.activated.connect(fn)
            return s

        self._space = QShortcut(QKeySequence(Qt.Key.Key_Space.value), self)
        self._space.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._space.setAutoRepeat(False)
        self._space.activated.connect(self._toggle_play)

        sc(Qt.Key.Key_Right.value, lambda: self.state.set_frame(
            self.state.current_frame + 1))
        sc(Qt.Key.Key_Left.value, lambda: self.state.set_frame(
            self.state.current_frame - 1))
        sc("Shift+Right", lambda: self.state.set_frame(
            self.state.current_frame + int(self.state.fps)))
        sc("Shift+Left", lambda: self.state.set_frame(
            self.state.current_frame - int(self.state.fps)))
        sc(Qt.Key.Key_Home.value, lambda: self.state.set_frame(0))
        sc(Qt.Key.Key_End.value, lambda: self.state.set_frame(10 ** 9))
        sc("Ctrl+1", lambda: self.tabs.setCurrentIndex(0))
        sc("Ctrl+2", lambda: self.tabs.setCurrentIndex(1))
        sc("Ctrl+3", lambda: self.tabs.setCurrentIndex(2))

    def _toggle_play(self):
        panel = {1: self.tab2.video, 2: self.tab3.video}.get(
            self.tabs.currentIndex())
        if panel:
            panel.toggle_playback()

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.MP4 *.avi *.mov *.mkv *.wmv);;All Files (*)")
        if not path:
            return
        try:
            self.state.load_video(path)
        except Exception as e:
            QMessageBox.critical(self, "Could not open video", str(e))

    def _on_cache_opened(self):
        self.tabs.setTabEnabled(1, True)
        self.tabs.setTabEnabled(2, True)

    def _about(self):
        QMessageBox.about(
            self, "Optical Flow Behavior Detector",
            "<h3>Optical Flow Behavior Detector</h3>"
            "<p>Domain-general detection of animal behaviors from dense optical "
            "flow, by histogram range selection over flow-derived features.</p>"
            "<p><b>Time is always in seconds and frequency in Hz.</b> Frame "
            "indices appear only in tooltips.</p>"
            "<p><b>Nyquist:</b> you cannot measure any frequency above half the "
            "frame rate. Content above it aliases down and imitates real signal. "
            "The tool warns you, but it cannot fix it — that needs a faster "
            "camera.</p>")

    def closeEvent(self, e):
        if self.state.cache is not None:
            self.state.cache.close()
        if self.state.source is not None:
            self.state.source.release()
        super().closeEvent(e)
