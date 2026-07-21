"""Main window: two tabs over one shared AppState.

Hotkeys follow the reference color detector so the two tools feel the same:
Space plays/pauses, arrows step a frame, shift+arrows step a second, Home/End
jump to the ends, Ctrl+1/2 switch tabs.

The flow-cache commit and Behavior Classification tabs were retired to
gui/_shelved/ -- the tensor path does not read a flow cache, so both had lost
the artefact they were built around. See gui/_shelved/README.md.
"""
from __future__ import annotations


from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QFileDialog, QLabel, QMainWindow,
                             QMessageBox, QTabWidget)

from gui.state import AppState
from gui.tab2_replicates import Tab2Replicates
from gui.tab_live_preprocess import TabLivePreprocess


class MainWindow(QMainWindow):
    def __init__(self, project_dir: str = "."):
        super().__init__()
        self.setWindowTitle("Optical Flow Behavior Detector")
        self.setMinimumSize(1400, 860)
        self.resize(1750, 980)

        self.state = AppState(project_dir)

        self.tabs = QTabWidget()
        self.tab2 = Tab2Replicates(self.state)
        self.tab_live = TabLivePreprocess(self.state)
        # Tensor path primary: live preprocessing is the main surface, and it
        # needs no cache. Tab indices: 0 Replicates, 1 Live preprocessing.
        self.tabs.addTab(self.tab2, "1 · Replicates")
        self.tabs.addTab(self.tab_live, "2 · Preprocessing (live)")
        self.setCentralWidget(self.tabs)

        self.status = QLabel("Open a video to begin.")
        self.statusBar().addWidget(self.status)

        self.state.status.connect(self.status.setText)
        self.state.video_loaded.connect(self._on_video_loaded)
        self.state.request_tab.connect(self.tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._menu()
        self._shortcuts()

    def _on_tab_changed(self, index: int):
        # Leaving the Replicates tab latches its boxes as processed. This is the
        # only route to the surfaces that consume the layout, so it strictly
        # precedes any pass -- latching on the pass as well would be redundant.
        # Moving a latched box is still allowed; it just has to be acknowledged
        # (Tab2Replicates._confirm_move), because the measurements behind it are
        # discarded rather than converted.
        if index != 0:
            self.tab2.mark_replicates_processed()

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

    def _toggle_play(self):
        # Prefer the video the user actually clicked. Both VideoPanel and the
        # embeddable SpeedExplorer expose toggle_playback(), so this remains valid
        # when Speed Explorer becomes a preprocessing sub-tab.
        # A widget can carry toggle_playback() for its own button and still
        # decline Space -- the scalogram explorer embedded in the live surface
        # does exactly that, because there Space means the live pass, and the
        # explorer sits between the focus and the surface that owns it.
        widget = QApplication.focusWidget()
        while widget is not None:
            toggle = getattr(widget, "toggle_playback", None)
            if callable(toggle) and getattr(widget, "space_toggles_playback",
                                            True):
                toggle()
                return
            widget = widget.parentWidget()

        # Focus somewhere with no playback of its own (a knob, the tab bar):
        # fall back to whatever the current tab plays.
        if self.tabs.currentIndex() == 0:
            self.tab2.video.toggle_playback()
        else:
            self.tab_live.toggle_playback()

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

    def _on_video_loaded(self):
        # ROI-first workflow: draw/import ownership boxes before processing.
        # Both tabs open with the video -- neither needs a cache now that live
        # preprocessing runs off the raw frames.
        self.tabs.setTabEnabled(0, True)
        self.tabs.setTabEnabled(1, True)
        self.tabs.setCurrentIndex(0)

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
