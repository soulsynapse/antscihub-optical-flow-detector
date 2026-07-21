"""The playback rate readout: footage-seconds advanced per wall-second.

The meter exists because the band-power lag (FINDINGS.md §20) was diagnosed by
eye -- "it looks about a third speed" -- and a number that can be read off the
panel while dragging a knob is worth more than a benchmark run afterwards.

Driven here with a fake clock rather than real sleeps: the assertions are about
arithmetic (two clocks compared over one window), and real timing would make
them flaky for no added coverage.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PyQt6.QtWidgets import QApplication, QLabel

from gui.explorers import scalogram_explorer as sg

_APP = QApplication.instance() or QApplication([])


class _Meter:
    """Just the rate-meter slice of ScalogramExplorer, on a controlled clock."""

    def __init__(self, fps=30.0):
        self.fps = fps
        self._advance_ticks = 0
        self._rate_wall0 = 0.0
        self._rate_ticks0 = 0
        self.rate_lbl = QLabel("")
        self.reports: list[str] = []
        self._start_rate_window()

    _sync_rate_label = sg.ScalogramExplorer._sync_rate_label
    _start_rate_window = sg.ScalogramExplorer._start_rate_window
    _set_rate_text = sg.ScalogramExplorer._set_rate_text

    def tick(self, n=1):
        self._advance_ticks += n
        before = self.rate_lbl.text()
        self._sync_rate_label()
        if self.rate_lbl.text() != before:
            self.reports.append(self.rate_lbl.text())


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class PlaybackRateTest(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self._real = sg.time.monotonic
        sg.time.monotonic = self.clock
        self.addCleanup(lambda: setattr(sg.time, "monotonic", self._real))
        self.m = _Meter(fps=30.0)

    def _play(self, n, dt):
        """Advance n frames, dt wall-seconds apart."""
        for _ in range(n):
            self.clock.now += dt
            self.m.tick()

    # Each case plays ~1.1 wall seconds rather than exactly 1.0. The window
    # closes on the first tick PAST a full second, so sitting exactly on the
    # boundary makes the assertion depend on whether n*dt accumulates to just
    # under or just over 1.0 in floating point -- which is a property of the
    # test's arithmetic, not of the meter.

    def test_no_reading_before_a_full_wall_second(self):
        """The window closes on wall time, not on frame count."""
        self._play(29, 1 / 30)                     # 0.97 s
        self.assertEqual(self.m.rate_lbl.text(), "",
                         "reported a rate before a full second elapsed")

    def test_keeping_up_reads_one_times(self):
        self._play(33, 1 / 30)
        self.assertEqual(self.m.rate_lbl.text(), "1.00x realtime")
        self.assertIn("#6ee678", self.m.rate_lbl.styleSheet(),
                      "keeping up was not shown as green")

    def test_a_third_realtime_reads_a_third(self):
        """The symptom that started this: 30 fps footage advancing 10 frames
        per wall second."""
        self._play(11, 0.1)
        self.assertEqual(self.m.rate_lbl.text(), "0.33x realtime")
        self.assertIn("#e06a5a", self.m.rate_lbl.styleSheet(),
                      "a third realtime was not flagged red")

    def test_the_window_resets_so_a_slow_second_does_not_taint_the_next(self):
        """Each reading covers its own second only. A cumulative average would
        keep reporting the lag after the fix that removed it.

        Asserted on the LAST reading, not the first healthy one: the window
        that straddles the speed change legitimately reports a blend of the two
        rates, because that is what happened during that second. Only once a
        window falls entirely inside the healthy stretch does it read 1.00x --
        and that recovery is the property under test."""
        self._play(11, 0.1)                        # a slow second
        self.assertEqual(self.m.reports[0], "0.33x realtime")
        self._play(70, 1 / 30)                     # then sustained healthy play
        self.assertEqual(self.m.reports[-1], "1.00x realtime",
                         f"never recovered; readings were {self.m.reports}")

    def test_a_loop_back_to_frame_zero_does_not_corrupt_the_reading(self):
        """Frames are counted with a cumulative tick counter precisely so the
        wrap at the end of the window is not read as a negative jump."""
        self.m.frame = 0
        self._play(33, 1 / 30)                     # ticks never decrease
        self.assertEqual(self.m.rate_lbl.text(), "1.00x realtime")

    def test_a_pause_does_not_charge_wall_time_against_zero_frames(self):
        """_toggle_play restarts the window on resume. Without that, the first
        second back reads ~0.00x and looks like a catastrophic regression."""
        self.clock.now += 45.0                     # a long pause
        self.m._start_rate_window()                # what resuming does
        self._play(33, 1 / 30)
        self.assertEqual(self.m.rate_lbl.text(), "1.00x realtime")

    def test_without_the_resume_reset_the_first_reading_back_is_bogus(self):
        """The negative case for the test above: it must be the reset doing the
        work, not the meter recovering on its own.

        It DOES recover on its own -- the stale window closes on the first tick
        back and the next one is clean -- so the cost of omitting the reset is
        exactly one garbage reading, not a persistent wrong number. Worth
        pinning at that precision: a test asserting a lasting collapse would
        overstate the bug and fail for the wrong reason."""
        self.clock.now += 45.0                     # paused, window left open
        self._play(70, 1 / 30)
        self.assertTrue(self.m.reports[0].startswith("0.0"),
                        f"expected one bogus reading, got {self.m.reports[0]}")
        self.assertEqual(self.m.reports[-1], "1.00x realtime",
                         "never recovered after the stale window closed")


if __name__ == "__main__":
    unittest.main()
