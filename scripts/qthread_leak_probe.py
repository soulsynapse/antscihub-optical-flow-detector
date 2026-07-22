"""Find QThreads still running when the test that started them ends.

This is the tool that found the teardown crash (`FINDINGS.md` §27) after four
sessions of blaming the machine. Keep it: the crash itself would not reproduce
on demand (~1 run in 35), but its PRECONDITION -- a QThread outliving its test --
is deterministic, and checking that took one run.

Qt destroying a running QThread is an access violation rather than an exception,
so this class of bug surfaces as a suite that reports every test passing and
then dies at interpreter shutdown with a faulthandler dump and no failed test.
There is nothing to attribute it to, which is exactly why it went unowned.

Usage, from the repo root::

    PYTHONPATH=scripts .venv/Scripts/python.exe -m pytest -q -p qthread_leak_probe -s

Read the report as follows. An entry with **no** ``(deleted)`` marker is a
thread that was genuinely still running when its test finished -- that is the
crash precondition and it is what to fix. An entry marked ``(deleted)`` is a
thread that had already finished and been freed, and is only visible because
this probe holds a stale Python wrapper; those are benign.

The baseline after the §27 fixes is 28 events, all of them ``(deleted)``. Any
entry WITHOUT the marker is a regression.
"""
from __future__ import annotations

from PyQt6.QtCore import QThread

_LIVE: list = []
_orig_start = QThread.start


def _start(self, *a, **kw):
    _LIVE.append(self)
    return _orig_start(self, *a, **kw)


# Wrapped at import, so a worker started before the first test is caught too.
QThread.start = _start

LEAKS: list[tuple[str, str]] = []


def pytest_runtest_teardown(item):
    still = []
    for t in list(_LIVE):
        try:
            running = t.isRunning()
        except RuntimeError:
            # The C++ side is gone. Reported rather than dropped silently,
            # because "already deleted" and "still running" are the two halves
            # of the same question and only one of them is benign.
            still.append(type(t).__name__ + "(deleted)")
            _LIVE.remove(t)
            continue
        if running:
            still.append(type(t).__name__)
        else:
            _LIVE.remove(t)
    if still:
        LEAKS.append((item.nodeid, ",".join(still)))


def pytest_sessionfinish(session, exitstatus):
    print("\n=== QThread leak report ===")
    for nodeid, names in LEAKS:
        print(f"LEAK {nodeid} -> {names}")
    running = [n for _, n in LEAKS if "(deleted)" not in n]
    print(f"TOTAL LEAK EVENTS: {len(LEAKS)}  "
          f"(STILL RUNNING, i.e. crash-capable: {len(running)})")
