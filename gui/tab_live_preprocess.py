"""Preprocessing as a live scalogram surface.

This is the dissolved preprocessing tab: instead of configuring a flow pass and
waiting for a cache, you tune downsample / block size / normalization on a short
window of the raw video and watch the scalogram and detection stack respond, with
no flow solve. The old "Preprocessing & Flow" tab is now the optional commit step
(it still runs the flow pass and writes the cache the Behavior tab consumes).

The heavy LiveScalogramSurface is built lazily -- only when the tab is shown and a
video plus at least one replicate box exist, and rebuilt only when the video or
replicate geometry actually changes (a windowed extraction is a real pass, so it
must not fire on every box edit made on another tab).
"""
from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from core.replicates import geometry_hash
from gui.explorers.live_scalogram_surface import LiveScalogramSurface


class TabLivePreprocess(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self._lay = QVBoxLayout(self)
        self._info = QLabel(
            "Load a video and define at least one replicate box in the "
            "Replicates tab to tune preprocessing live here — no flow cache "
            "required.")
        self._info.setWordWrap(True)
        self._info.setStyleSheet("color:#8ab; padding:24px;")
        self._lay.addWidget(self._info)

        self._surface: LiveScalogramSurface | None = None
        self._sig: tuple | None = None

        state.video_loaded.connect(self._drop)        # new clip invalidates all
        state.rois_changed.connect(self._maybe_refresh)

    def _current_sig(self) -> tuple | None:
        if not self.state.has_video or not self.state.replicate_specs:
            return None
        return (self.state.source.info.path,
                geometry_hash(self.state.replicate_specs))

    def _drop(self):
        if self._surface is not None:
            self._surface.close()
            self._surface.setParent(None)
            self._surface.deleteLater()
            self._surface = None
        self._sig = None
        self._info.setVisible(True)

    def _maybe_refresh(self):
        # Boxes are usually edited on another tab; only (re)build when visible.
        if self.isVisible():
            self._ensure_surface()

    def _ensure_surface(self):
        sig = self._current_sig()
        if sig is None:
            self._drop()
            return
        if sig == self._sig and self._surface is not None:
            # Same geometry, so the surface stands -- but its replicate dicts are
            # references into an AppState list that is REPLACED with fresh copies
            # on every edit, so a calibration or baseline set on another tab has
            # not reached it. This runs on every show, which is exactly when the
            # user could have been editing elsewhere.
            self._surface.refresh_replicate_metadata(self.state.replicate_specs)
            return
        self._drop()
        self._info.setVisible(False)
        self._surface = LiveScalogramSurface(
            self.state.source.info.path, self.state.replicate_specs,
            base_cfg=self.state.cfg, parent=self,
            frame_provider=lambda: self.state.current_frame)
        # The surface works from AppState's copies of the replicate dicts, so a
        # calibration measured in its downsample window has to be relayed here
        # to reach the replicate tab's list and its per-video sidecar.
        self._surface.calibration_changed.connect(self.state.apply_calibration)
        self._lay.addWidget(self._surface, 1)
        self._sig = sig

    def showEvent(self, e):
        super().showEvent(e)
        self._ensure_surface()

    def toggle_playback(self):
        if self._surface is not None:
            self._surface.toggle_playback()
