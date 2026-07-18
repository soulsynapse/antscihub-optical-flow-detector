"""``prefetch`` preserves order, forwards errors, and always joins its thread.

The shutdown cases are the ones that matter in the app: the extraction pass hands
``prefetch`` a generator that holds an open VideoCapture, and releases that
capture immediately after ``close()``. If close could return with the producer
still running -- a cancelled pass leaves via an exception, with the queue full and
nobody draining it -- the decode thread would read from a freed handle.
"""
import threading
import time

import pytest

from core.video import prefetch


def _live_prefetch_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate()
            if t.name == "ofd-prefetch" and t.is_alive()]


def test_yields_every_item_in_order():
    assert list(prefetch(iter(range(50)))) == list(range(50))


def test_empty_source():
    assert list(prefetch(iter(()))) == []


def test_runs_ahead_of_the_consumer():
    """The producer must not be lock-step with the consumer, or it hides nothing."""
    seen: list[int] = []

    def source():
        for i in range(10):
            seen.append(i)
            yield i

    gen = prefetch(source(), depth=3)
    assert next(gen) == 0
    # Give the producer a moment to fill the queue behind that first item.
    deadline = time.monotonic() + 2.0
    while len(seen) < 4 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(seen) >= 4, f"producer did not run ahead: {seen}"
    gen.close()


def test_source_exception_is_reraised_on_the_consumer():
    def source():
        yield 1
        raise ValueError("boom")

    gen = prefetch(source())
    assert next(gen) == 1
    with pytest.raises(ValueError, match="boom"):
        next(gen)


def test_close_joins_the_producer_thread():
    """Early close with the queue full: the producer is parked in put() and must
    still notice the stop flag and exit before close() returns."""
    started = threading.Event()

    def source():
        started.set()
        while True:
            yield 1

    before = len(_live_prefetch_threads())
    gen = prefetch(source(), depth=2)
    next(gen)
    started.wait(2.0)
    gen.close()
    assert len(_live_prefetch_threads()) == before


def test_close_closes_the_source_generator():
    """The source's own finally must run -- that is where the caller releases
    decoder state, and it is only reached if the producer closes it."""
    closed = threading.Event()

    def source():
        try:
            while True:
                yield 1
        finally:
            closed.set()

    gen = prefetch(source(), depth=2)
    next(gen)
    gen.close()
    assert closed.is_set()


def test_exhausted_source_leaves_no_thread():
    before = len(_live_prefetch_threads())
    assert sum(prefetch(iter(range(100)))) == 4950
    deadline = time.monotonic() + 2.0
    while len(_live_prefetch_threads()) > before and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(_live_prefetch_threads()) == before
