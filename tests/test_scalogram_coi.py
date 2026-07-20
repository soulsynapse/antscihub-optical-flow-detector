"""The scalogram's time axis, and the cone of influence drawn on it.

Both halves are here because they are one requirement: the COI is a wedge
anchored to the ENDS of the record, so it can only be right if the time axis
maps the record onto the widget correctly. The axis half is a regression test --
the original gather divided by T instead of the pixel count, which cancels to
arange(w) and silently showed only the first w frames.
"""
import unittest

import numpy as np
from PyQt6.QtWidgets import QApplication

from core.wavelet import W0, coi_edge_samples, coi_efolding_s, morlet_scales
from gui.explorers.scalogram_explorer import ScalogramPlot

app = QApplication.instance() or QApplication([])

_BG = 12.0          # _SG_RAMP[0], the plot background the wedge fades toward


def _pixels(plot, w, h, freqs):
    plot.resize(w + 80, plot.maximumHeight())
    plot.grab()
    img = plot._heatmap(w, h, float(freqs[0]), float(freqs[-1]))
    buf = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8)
    return buf.reshape(h, img.bytesPerLine() // 3, 3)[:, :w].astype(float)


class CoiConstantTest(unittest.TestCase):
    def test_efolding_is_sqrt2_times_the_scale(self):
        """Torrence & Compo table 1 for Morlet: tau = sqrt(2) * s."""
        f = np.geomspace(0.25, 30.0, 17)
        np.testing.assert_allclose(coi_efolding_s(f),
                                   np.sqrt(2.0) * morlet_scales(f), rtol=1e-12)

    def test_efolding_is_1_369_over_f_at_w0_6(self):
        """The number the renderer's wedge width comes out to, pinned so a
        change to W0 that silently moves it cannot pass unnoticed. This is NOT
        the 1.46/f the plan quoted -- that figure is ~7% too large."""
        self.assertAlmostEqual(W0, 6.0, places=9)
        f = np.geomspace(0.5, 20.0, 9)
        np.testing.assert_allclose(coi_efolding_s(f) * f, 1.36897755,
                                   rtol=1e-6)
        self.assertAlmostEqual(float(coi_efolding_s(np.array([0.5]))[0]),
                               2.7379551, places=6)

    def test_edge_samples_scales_with_rate(self):
        f = np.array([1.0, 4.0])
        np.testing.assert_allclose(coi_edge_samples(f, 30.0),
                                   coi_efolding_s(f) * 30.0, rtol=1e-12)


class TimeAxisTest(unittest.TestCase):
    """The regression the COI could not have been built on top of."""

    def test_late_energy_renders_at_the_right_edge(self):
        F, T, w, h = 24, 1500, 300, 120
        freqs = np.geomspace(0.5, 20.0, F)
        m = np.zeros((F, T), np.float32)
        m[:, -50:] = 1.0                      # energy ONLY in the last 50 frames
        sp = ScalogramPlot("s", freqs, 24.0)
        sp.set_scalogram(m)
        lum = _pixels(sp, w, h, freqs).mean(2).mean(0)
        self.assertGreater(lum.argmax(), 0.9 * w,
                           "late energy must render near the right edge; "
                           "gathering arange(w) puts it off-screen entirely")
        self.assertLess(lum[:w // 2].mean(), lum[-10:].mean(),
                        "the lit half is on the wrong side of the plot")

    def test_the_whole_record_maps_onto_the_columns(self):
        """T >> w must RESAMPLE, not truncate, and must cover every frame."""
        F = 8
        freqs = np.geomspace(0.5, 20.0, F)
        for T, w in ((1500, 300), (11308, 600), (400, 900)):
            sp = ScalogramPlot("s", freqs, 24.0)
            sp.set_scalogram(np.zeros((F, T), np.float32))
            _, d = sp._columns(w, T, np.zeros(4, int))
            self.assertEqual(len(d), w)
            self.assertEqual(int(np.min(d)), 0,
                             f"T={T} w={w}: no column touches an end")
            self.assertGreater(int(np.max(d)), 0)

    def test_a_brief_burst_between_sample_points_still_renders(self):
        """The aliasing a point-gather loses. At T=11308 into 600 columns a
        one-frame event sits between sampled indices ~94% of the time; a
        per-column max keeps it, and signal silently absent from the plot is
        exactly what this project treats as the dangerous failure."""
        F, T, w, h = 8, 11308, 600, 60
        freqs = np.geomspace(0.5, 20.0, F)
        for spike in (5000, 5001, 5002, 11307):
            m = np.zeros((F, T), np.float32)
            m[:, spike] = 1.0
            sp = ScalogramPlot("s", freqs, 24.0)
            sp.set_scalogram(m)
            vals, _ = sp._columns(w, T, np.arange(F))
            self.assertGreater(float(vals.max()), 0.5,
                               f"a burst at frame {spike} vanished entirely")
            hit = int(np.argmax(vals.max(0)))
            self.assertAlmostEqual(hit / w, spike / T, delta=0.02,
                                   msg=f"burst at {spike} drawn at the wrong x")


class CoiRenderTest(unittest.TestCase):
    """Read the wedge back off rendered pixels.

    fps<=0 disables the fade, so rendering the same matrix twice and dividing
    recovers alpha directly instead of asserting on absolute colours.
    """
    F, T, w, h = 24, 1500, 300, 120
    fps = 24.0

    def setUp(self):
        self.freqs = np.geomspace(0.5, 20.0, self.F)
        rng = np.random.default_rng(0)
        # Contrast matters: a CONSTANT matrix collapses the colour ramp onto
        # its zero entry, which is the background, and the fade then has
        # nothing to act on -- a fade test written on uniform power is vacuous.
        self.m = (1.0 + rng.random((self.F, self.T))).astype(np.float32)

    def _alpha(self):
        def render(fps):
            sp = ScalogramPlot("s", self.freqs, fps)
            sp.set_scalogram(self.m)
            return _pixels(sp, self.w, self.h, self.freqs)
        on, off = render(self.fps), render(0.0)
        num, den = (on - _BG).mean(2), (off - _BG).mean(2)
        return np.where(den > 5, num / np.maximum(den, 1e-9), np.nan)

    def test_edges_fade_and_the_centre_does_not(self):
        a = self._alpha()
        self.assertAlmostEqual(float(np.nanmean(a[:, self.w // 2])), 1.0,
                               places=2)
        self.assertAlmostEqual(float(np.nanmean(a[:, 0])), 0.0, places=2)

    def test_both_ends_fade_not_only_the_newest(self):
        """A live trailing window's oldest edge has real frames before it, but
        the transform did not see them -- it is a zero-padding edge all the
        same.

        Asserted on the LOW-frequency rows, not on a mean over the plot: the
        axis is log-frequency, so most rows sit where the cone is under a
        column wide and alpha is legitimately 1. A row-mean would mostly
        measure how many high frequencies the bank happens to hold.
        """
        a = self._alpha()
        low = slice(self.h - 6, self.h - 1)
        self.assertLess(float(np.nanmean(a[low, 1])), 0.3, "oldest edge")
        self.assertLess(float(np.nanmean(a[low, -2])), 0.3, "newest edge")

    def test_the_wedge_is_frequency_shaped(self):
        """The whole point: the cone is ~1/f, so low frequencies fade far into
        the record while high ones are trustworthy almost to the edge."""
        a = self._alpha()
        hi = float(np.nanmean(a[1:4, 5]))            # top rows = high f
        lo = float(np.nanmean(a[self.h - 4:self.h - 1, 5]))   # bottom = low f
        self.assertGreater(hi, 0.95, "high frequencies should be usable by col 5")
        self.assertLess(lo, 0.75, "low frequencies must still be faded at col 5")
        self.assertLess(lo, hi)

    def test_alpha_tracks_the_predicted_cone_width(self):
        """alpha is the linear ramp d/edge, so it is predictable rather than
        merely monotone."""
        a = self._alpha()
        row = self.h - 3
        rf = 10 ** (np.log10(self.freqs[-1]) - (row + 0.5) / self.h *
                    (np.log10(self.freqs[-1]) - np.log10(self.freqs[0])))
        edge_cols = coi_edge_samples(np.array([rf]), self.fps)[0] / self.T * self.w
        for c in (3, 5, 8):
            self.assertAlmostEqual(float(np.nanmean(a[row - 1:row + 2, c])),
                                   min(1.0, c / edge_cols), delta=0.15,
                                   msg=f"column {c} is off the predicted ramp")

    def test_a_window_too_short_for_a_frequency_is_suppressed_throughout(self):
        """2 s cannot resolve 0.5 Hz: the cone half-width (65.7 samples) exceeds
        the whole 48-sample record, so EVERY column of that row is inside the
        cone and none of it reaches full brightness.

        Suppressed rather than blanked, deliberately. The ramp is continuous
        because the e-folding time is a scale and not a cutoff, so a record
        wholly inside the cone dims toward background as it shortens instead of
        disappearing at a threshold. What must not happen is a confident-looking
        0.5 Hz band.
        """
        T = int(2 * self.fps)
        m = (1.0 + np.random.default_rng(1).random((self.F, T))).astype(np.float32)

        def render(fps):
            sp = ScalogramPlot("s", self.freqs, fps)
            sp.set_scalogram(m)
            return _pixels(sp, self.w, self.h, self.freqs)

        num = (render(self.fps) - _BG).mean(2)
        den = (render(0.0) - _BG).mean(2)
        a = np.where(den > 5, num / np.maximum(den, 1e-9), np.nan)
        low = np.nanmax(a[self.h - 4:self.h - 1])
        self.assertLess(low, 0.45,
                        "0.5 Hz over 2 s must never reach full brightness")
        self.assertGreater(float(np.nanmean(a[1:4, self.w // 2])), 0.9,
                           "high frequencies over the same 2 s are fine and "
                           "must NOT be suppressed with them")


if __name__ == "__main__":
    unittest.main()
