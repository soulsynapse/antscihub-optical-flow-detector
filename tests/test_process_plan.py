"""Which spans a whole-video pass covers, and in what order.

The property under test throughout is the one the feature exists for: a pass you
stop early must leave you with a sample of the CLIP, not a sample of its
beginning. Everything else here is bookkeeping around that.
"""
import unittest

import numpy as np

from core.process_plan import (BISECT, CONTINUOUS, FROM_HERE, GAPS, UNIFORM,
                               Segment, coverage_note, plan_segments)

FPS = 30.0
N = 30_000          # ~16.7 min, roughly the clips this is aimed at


def _plan(strategy, **kw):
    kw.setdefault("n_frames", N)
    kw.setdefault("fps", FPS)
    return plan_segments(strategy, **kw)


def _covered(segments):
    m = np.zeros(N, bool)
    for s in segments:
        m[s.start:s.stop] = True
    return m


class Shapes(unittest.TestCase):
    def test_continuous_is_the_whole_clip_in_one_span(self):
        self.assertEqual(_plan(CONTINUOUS), [Segment(0, N)])

    def test_from_here_starts_at_the_cursor(self):
        self.assertEqual(_plan(FROM_HERE, cursor=1234), [Segment(1234, N)])

    def test_from_here_clamps_a_cursor_past_the_end(self):
        segs = _plan(FROM_HERE, cursor=N + 500)
        self.assertEqual(len(segs), 1)
        self.assertGreaterEqual(segs[0].n, 2)

    def test_segments_never_overlap(self):
        for strategy in (CONTINUOUS, FROM_HERE, BISECT, UNIFORM):
            with self.subTest(strategy=strategy):
                segs = sorted(_plan(strategy, budget=0.3),
                              key=lambda s: s.start)
                for a, b in zip(segs, segs[1:]):
                    self.assertLessEqual(a.stop, b.start)

    def test_short_clip_yields_nothing_rather_than_a_degenerate_span(self):
        self.assertEqual(plan_segments(CONTINUOUS, n_frames=1, fps=FPS), [])

    def test_unknown_strategy_raises_rather_than_defaulting(self):
        with self.assertRaises(ValueError):
            _plan("scan_backwards")


class BudgetAndCoverage(unittest.TestCase):
    def test_budget_buys_roughly_that_fraction_of_the_clip(self):
        for frac in (0.05, 0.1, 0.25, 0.5):
            with self.subTest(frac=frac):
                total = sum(s.n for s in _plan(BISECT, budget=frac))
                self.assertAlmostEqual(total / N, frac, delta=0.03)

    def test_full_budget_covers_every_frame(self):
        """A full-budget sample is a full pass; the chunking must be exhaustive
        and not leave slivers between spans."""
        segs = _plan(BISECT, budget=1.0)
        self.assertTrue(_covered(segs).all())

    def test_a_tiny_budget_still_processes_something(self):
        """A plan that runs, reports success and examines nothing is the worst
        possible outcome -- it is indistinguishable from a quiet clip."""
        segs = _plan(BISECT, budget=0.0001)
        self.assertGreaterEqual(len(segs), 1)
        self.assertGreaterEqual(sum(s.n for s in segs), 2)

    def test_uniform_and_bisect_cover_the_same_footage(self):
        """They differ only in ORDER. If they differed in contents, the choice
        between them would be a coverage decision in disguise."""
        a = sorted((s.start, s.stop) for s in _plan(BISECT, budget=1.0))
        b = sorted((s.start, s.stop) for s in _plan(UNIFORM, budget=1.0))
        self.assertEqual(a, b)


class CoverageAwareBudget(unittest.TestCase):
    """A budget on a partly-examined clip must spend itself on what is LEFT.

    The bug this guards: the plan used to pick its budgeted chunks over the whole
    clip and only afterwards subtract coverage, so re-running a sampling plan on
    a clip that was already partly done cancelled most of the budget against work
    behind you -- and could plan nothing at all while most of the clip sat
    unexamined.
    """

    def _mask(self, *spans):
        m = np.zeros(N, bool)
        for a, b in spans:
            m[a:b] = True
        return m

    def test_plan_lands_only_on_uncovered_frames(self):
        covered = self._mask((0, N // 2))          # first half already examined
        segs = _plan(BISECT, budget=0.25, covered=covered)
        for s in segs:
            self.assertFalse(covered[s.start:s.stop].any(),
                             f"{s} overlaps examined footage")

    def test_partly_done_clip_still_plans_a_full_budget_of_new_work(self):
        """The regression: some coverage must not shrink the budget."""
        fresh = sum(s.n for s in _plan(BISECT, budget=0.2))
        covered = self._mask((0, N // 3))
        partial = sum(s.n for s in _plan(BISECT, budget=0.2, covered=covered))
        # A fifth of the clip's worth of NEW footage, give or take a chunk --
        # not the sliver an after-the-fact intersection would have left.
        self.assertAlmostEqual(partial / N, fresh / N, delta=0.05)

    def test_a_covered_clip_that_still_has_gaps_is_not_blocked(self):
        """Coverage everywhere but a late window: the plan must find that window
        rather than report nothing to do."""
        covered = self._mask((0, N - 4000))
        segs = _plan(BISECT, budget=0.1, covered=covered)
        self.assertTrue(segs)
        self.assertTrue(all(s.start >= N - 4000 for s in segs))

    def test_fully_covered_plans_nothing(self):
        segs = _plan(BISECT, budget=0.5, covered=self._mask((0, N)))
        self.assertEqual(segs, [])

    def test_covered_none_is_the_unchanged_plan(self):
        a = _plan(BISECT, budget=0.3)
        b = _plan(BISECT, budget=0.3, covered=None)
        self.assertEqual(a, b)


class BisectOrdering(unittest.TestCase):
    """The reason this strategy exists: every prefix is spread over the clip."""

    def test_uniform_bunches_at_the_front_and_bisect_does_not(self):
        """The contrast that motivates the whole feature."""
        n = 8
        uni = _plan(UNIFORM, budget=1.0)[:n]
        bis = _plan(BISECT, budget=1.0)[:n]
        # Front-to-back ordering: the first eight chunks are all in the first
        # part of the clip.
        self.assertLess(max(s.stop for s in uni), N // 2)
        # Bisecting order: the same count already reaches both ends.
        self.assertGreater(max(s.stop for s in bis), 0.8 * N)
        self.assertLess(min(s.start for s in bis), 0.2 * N)

    def test_every_prefix_is_evenly_spread(self):
        """Low discrepancy, stated as the thing that actually matters: for any
        stopping point, the examined fraction of the FIRST half of the clip is
        close to the examined fraction of the second."""
        segs = _plan(BISECT, budget=1.0)
        for k in (4, 8, 16, 32, 64):
            if k > len(segs):
                continue
            with self.subTest(k=k):
                m = _covered(segs[:k])
                first, second = m[:N // 2].mean(), m[N // 2:].mean()
                self.assertLess(abs(first - second), 0.2,
                                f"prefix of {k} is lopsided: "
                                f"{first:.2f} vs {second:.2f}")

    def test_order_is_a_permutation_of_the_chunks(self):
        """However the rounding falls, no chunk is dropped or repeated."""
        segs = _plan(BISECT, budget=1.0)
        starts = [s.start for s in segs]
        self.assertEqual(len(starts), len(set(starts)))
        self.assertEqual(sorted(starts),
                         sorted(s.start for s in _plan(UNIFORM, budget=1.0)))

    def test_first_chunks_are_the_start_then_the_middle(self):
        segs = _plan(BISECT, budget=1.0, chunk_s=60.0)
        self.assertEqual(segs[0].start, 0)
        mid = segs[1].start / N
        self.assertAlmostEqual(mid, 0.5, delta=0.06)


class Gaps(unittest.TestCase):
    def test_gaps_passes_through_what_the_track_reports(self):
        segs = _plan(GAPS, gaps=[(0, 100), (5000, 9000)])
        self.assertEqual(segs, [Segment(0, 100), Segment(5000, 9000)])

    def test_gaps_drops_spans_too_short_to_be_a_time_series(self):
        """A one-frame gap would be decoded, transformed and recorded as
        examined while meaning nothing."""
        segs = _plan(GAPS, gaps=[(10, 11), (500, 900)])
        self.assertEqual(segs, [Segment(500, 900)])

    def test_no_gaps_is_no_work(self):
        self.assertEqual(_plan(GAPS, gaps=[]), [])


class CoverageNote(unittest.TestCase):
    def test_sampling_plans_say_what_is_left_unexamined(self):
        """The standing rule: 'nobody looked here' must never be presentable as
        'nothing happened here'."""
        segs = _plan(BISECT, budget=0.1)
        note = coverage_note(BISECT, segs, N, FPS)
        self.assertIn("NOT be examined", note)
        # The unexamined share, not a hard-coded percentage: the chunking rounds
        # to whole chunks, so a 10% budget on a clip that is not a whole number
        # of chunks long is legitimately 9% or 11%.
        left = 100 - round(100.0 * sum(s.n for s in segs) / N)
        self.assertIn(f"{left}%", note)
        self.assertGreater(left, 80)

    def test_continuous_makes_no_such_claim(self):
        note = coverage_note(CONTINUOUS, _plan(CONTINUOUS), N, FPS)
        self.assertNotIn("NOT", note)
        self.assertIn("100%", note)

    def test_empty_plan_says_so(self):
        self.assertEqual(coverage_note(GAPS, [], N, FPS), "nothing to process")


if __name__ == "__main__":
    unittest.main()
