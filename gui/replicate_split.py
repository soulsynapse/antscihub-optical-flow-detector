"""Background worker for cutting one source into its replicate clips.

The actual split and all of its validity rules live in ``core.pretranscode``.
This module only adapts that blocking operation to Qt signals so the Replicates
tab stays responsive and can cancel without leaving partial clips behind.
"""
from __future__ import annotations

import copy
import threading

from PyQt6.QtCore import QThread, pyqtSignal

from core.pretranscode import (DEFAULT_CLIP_LAYOUT, PretranscodeCancelled,
                               build_pretranscode)


class ReplicateSplitWorker(QThread):
    progress = pyqtSignal(int)          # thousandths, so the final hash moves
    complete = pyqtSignal(object)       # Manifest
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, video_path: str, replicates: list[dict], out_dir: str,
                 *, quality: str, overwrite: bool, parent=None):
        super().__init__(parent)
        self.video_path = str(video_path)
        # Geometry must mean what it meant when Start was pressed even if a
        # caller outside the GUI mutates its list while the worker is running.
        self.replicates = copy.deepcopy(replicates)
        self.out_dir = str(out_dir)
        self.quality = str(quality)
        self.overwrite = bool(overwrite)
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            manifest = build_pretranscode(
                self.video_path, self.replicates, self.out_dir,
                quality=self.quality, clip_layout=DEFAULT_CLIP_LAYOUT,
                overwrite=self.overwrite,
                progress=lambda f: self.progress.emit(
                    max(0, min(1000, int(round(float(f) * 1000))))),
                should_cancel=self._cancel.is_set)
        except PretranscodeCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
        else:
            self.complete.emit(manifest)
