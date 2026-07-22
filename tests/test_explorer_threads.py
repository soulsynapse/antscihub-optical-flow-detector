"""Cube-thread lifetime in ScalogramExplorer, and the teardown crash it caused.

The failure these pin does not raise, fail a test, or print a traceback: Qt
destroys a running QThread and the process takes an access violation, which
surfaces as a suite that reports "840 passed" and then dies at interpreter
shutdown with a faulthandler dump. It was intermittent for months and was
repeatedly written off as environmental.

Two distinct holes let a cube thread outlive the widget that owns it, and both
come from the same seam: ``_ScalogramWorker.run`` ends with ``done.emit(...)``,
so ``_on_cube_ready`` runs -- on the GUI thread, by queued connection -- while
the thread that produced the cube is still unwinding.

1. ``_on_cube_ready`` clears ``self._worker`` and may immediately launch the
   next cube into it. A close landing in that window waited on the NEW thread
   and let Qt destroy the old one, still running, as a child QObject.
2. ``close()`` is not deletion. The widget keeps receiving queued signals until
   ``deleteLater`` gets an event-loop turn, so a ``done`` arriving after the
   close re-entered ``_request_cube`` and started a thread *after* every thread
   had been joined.

Asserted through the real ``closeEvent``, because the bug is entirely in the
ordering of that method against a queued signal -- a test that called the join
directly would pass in both the fixed and the broken tree.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication

from gui.explorers.scalogram_explorer import ScalogramExplorer
from tests.test_view_state import _channel_data


class _BlockingThread(QThread):
    """Runs until released, so a test can hold a thread in the running state
    across a close instead of racing a real cube build."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._release = False

    def run(self):
        while not self._release:
            self.msleep(1)

    def release(self):
        self._release = True


class CubeThreadLifetimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _explorer(self):
        ex = ScalogramExplorer.from_channel_data(
            _channel_data(), video_path=None, own_shortcuts=False,
            own_status=False)
        self.addCleanup(self._destroy, ex)
        return ex

    def _destroy(self, ex):
        ex.close()
        ex.deleteLater()
        self.app.processEvents()

    def test_close_joins_a_thread_that_is_no_longer_self_worker(self):
        """Hole 1. The thread most likely to still be alive at close is exactly
        the one `_on_cube_ready` has already dropped from `self._worker`."""
        ex = self._explorer()
        t = _BlockingThread(ex)
        ex._threads.append(t)
        t.start()
        while not t.isRunning():
            self.app.processEvents()
        # Precisely the state _on_cube_ready leaves behind: a running thread that
        # `self._worker` no longer names.
        ex._worker = None

        t.release()
        ex.close()

        self.assertFalse(t.isRunning(),
                         "closeEvent must join every started thread, not just "
                         "the one self._worker happens to point at")
        self.assertEqual(ex._threads, [])

    def test_a_closed_explorer_refuses_to_start_another_cube(self):
        """Hole 2. `done` is queued, so it can land after the close and re-enter
        `_request_cube`, spawning a thread past every join."""
        ex = self._explorer()
        ex.close()
        before = list(ex._threads)

        ex._request_cube()

        self.assertEqual(ex._threads, before,
                         "a closed explorer started a cube thread")
        self.assertIsNone(ex._worker)

    def test_a_finished_thread_stops_being_tracked(self):
        """Otherwise the join list grows for the widget's whole life and close
        walks a list of dead wrappers."""
        ex = self._explorer()
        t = _BlockingThread(ex)
        ex._threads.append(t)
        t.finished.connect(ex._retire_thread)
        t.start()
        while not t.isRunning():
            self.app.processEvents()
        t.release()
        t.wait(5000)
        # finished is delivered on the GUI thread.
        for _ in range(50):
            self.app.processEvents()
            if not ex._threads:
                break

        self.assertEqual(ex._threads, [])


if __name__ == "__main__":
    unittest.main()
