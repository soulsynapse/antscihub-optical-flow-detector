"""Timer accumulates spans, formats them, and stays a no-op when disabled."""
import logging
import time

import pytest

from core.timing import LOGGER_NAME, Timer, timing_enabled


@pytest.fixture
def lines():
    """Capture the timing logger directly.

    ``caplog`` cannot see these: core.timing sets ``propagate = False`` so a later
    root-logger configuration in the app cannot double-print every timing line.
    """
    captured: list[str] = []

    class _Sink(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    log = logging.getLogger(LOGGER_NAME)
    h = _Sink()
    log.addHandler(h)
    try:
        yield captured
    finally:
        log.removeHandler(h)


def test_span_accumulates_across_entries():
    tm = Timer("pass", enabled=True)
    for _ in range(3):
        with tm.span("work"):
            time.sleep(0.005)
    assert tm.counts["work"] == 3
    assert tm.totals["work"] >= 0.012


def test_span_records_even_when_body_raises():
    tm = Timer("pass", enabled=True)
    try:
        with tm.span("work"):
            raise ValueError("boom")
    except ValueError:
        pass
    assert tm.counts["work"] == 1          # __exit__ must not swallow the error


def test_disabled_timer_records_nothing(lines):
    tm = Timer("pass", enabled=False)
    with tm.span("work"):
        pass
    tm.add("other", 1.0)
    assert tm.totals == {} and tm.counts == {}
    tm.log()
    assert lines == []


def test_format_ranks_slowest_first_and_carries_extras():
    tm = Timer("extract", enabled=True)
    tm.add("slow", 2.0)
    tm.add("fast", 0.5)
    line = tm.format(frames=100)
    assert line.startswith("extract ")
    assert "frames=100" in line
    assert line.index("slow") < line.index("fast")


def test_log_respects_min_seconds(lines):
    tm = Timer("cheap", enabled=True)
    tm.log(min_seconds=10.0)
    assert lines == []
    tm.log(min_seconds=0.0)
    assert len(lines) == 1 and lines[0].startswith("cheap ")


def test_env_flag_disables(monkeypatch):
    monkeypatch.setenv("OFD_TIMING", "0")
    assert not timing_enabled()
    assert not Timer("pass").enabled
    monkeypatch.setenv("OFD_TIMING", "1")
    assert timing_enabled()
