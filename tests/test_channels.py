"""The derived-channel registry (core.channels) and the velocity-gradient family.

The velocity gradient decomposes the block flow's spatial gradient into three
invariant parts; these tests pin the math on flows whose divergence, shear and
vorticity are known in closed form, and check that a spatial derivative never
crosses a replicate seam.
"""
from __future__ import annotations

import unittest

import numpy as np

from core import channels as ch
from core.channel_source import ChannelData, with_derived_channels


def _flow(u: np.ndarray, v: np.ndarray) -> dict:
    """A one-frame base-field dict from 2-D (ny, nx) u, v planes."""
    return {"u": u[None].astype(np.float32), "v": v[None].astype(np.float32)}


class VelocityGradientMathTest(unittest.TestCase):
    """Known analytic flows -> known div/shear/vorticity (mean over the grid,
    away from the one-sided edges np.gradient handles differently)."""

    def setUp(self):
        ny, nx = 8, 9
        self.yy, self.xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
        self.meta = {"grid": [ny, nx]}       # single whole-grid region

    def _means(self, u, v):
        r = ch.evaluate(_flow(u, v), self.meta, ch.VELOCITY_GRADIENT)
        return (float(r["vel_divergence"].mean()),
                float(r["vel_shear"].mean()),
                float(r["vel_vorticity"].mean()))

    def test_pure_expansion(self):
        # u=x, v=y  ->  du/dx=dv/dy=1  ->  div=2, shear=0, vort=0
        div, shear, vort = self._means(self.xx, self.yy)
        self.assertAlmostEqual(div, 2.0, places=5)
        self.assertAlmostEqual(shear, 0.0, places=5)
        self.assertAlmostEqual(vort, 0.0, places=5)

    def test_pure_rotation(self):
        # u=-y, v=x  ->  dv/dx=1, du/dy=-1  ->  div=0, shear=0, vort=2
        div, shear, vort = self._means(-self.yy, self.xx)
        self.assertAlmostEqual(div, 0.0, places=5)
        self.assertAlmostEqual(shear, 0.0, places=5)
        self.assertAlmostEqual(vort, 2.0, places=5)

    def test_pure_strain(self):
        # u=x, v=-y  ->  du/dx=1, dv/dy=-1  ->  div=0, shear=2, vort=0
        div, shear, vort = self._means(self.xx, -self.yy)
        self.assertAlmostEqual(div, 0.0, places=5)
        self.assertAlmostEqual(shear, 2.0, places=5)
        self.assertAlmostEqual(vort, 0.0, places=5)

    def test_shear_is_nonnegative_but_div_vort_signed(self):
        # A contraction has NEGATIVE divergence; shear stays >= 0 by construction.
        r = ch.evaluate(_flow(-self.xx, -self.yy), self.meta,
                        ch.VELOCITY_GRADIENT)
        self.assertLess(float(r["vel_divergence"].mean()), 0.0)
        self.assertTrue((r["vel_shear"] >= 0).all())


class RegionIsolationTest(unittest.TestCase):
    """A spatial derivative must stay inside a replicate's atlas box, or a jump
    at the packing seam between two replicates reads as a huge fake gradient."""

    def test_gradient_does_not_cross_the_seam(self):
        ny, nx = 4, 8                         # two side-by-side 4x4 tiles
        u = np.zeros((ny, nx), np.float32)
        u[:, 4:] = 100.0                      # a cliff at the col 3|4 seam
        v = np.zeros((ny, nx), np.float32)

        tiled = {"grid": [ny, nx], "replicate_tiles": [
            {"atlas_bbox": (0, 0, ny, 4)}, {"atlas_bbox": (0, 4, ny, nx)}]}
        r = ch.evaluate(_flow(u, v), tiled, ("vel_divergence",))
        # Each region holds a constant u, so per-region du/dx is 0 throughout --
        # the seam is invisible.
        self.assertLess(float(np.abs(r["vel_divergence"]).max()), 1e-4)

        # Whole-grid (no tiles) DOES see the cliff: np.gradient spreads the jump
        # across the boundary columns. This is exactly what the region loop avoids.
        whole = {"grid": [ny, nx]}
        r2 = ch.evaluate(_flow(u, v), whole, ("vel_divergence",))
        self.assertGreater(float(np.abs(r2["vel_divergence"]).max()), 10.0)


class RegistryTest(unittest.TestCase):
    def test_needs_for_reports_declared_base_fields(self):
        self.assertEqual(ch.needs_for(ch.VELOCITY_GRADIENT), {"u", "v"})

    def test_evaluate_rejects_unknown_channel(self):
        with self.assertRaises(KeyError):
            ch.evaluate({}, {"grid": [2, 2]}, ["not_a_channel"])

    def test_evaluate_rejects_missing_base_field(self):
        # vel_shear needs both u and v; only u supplied.
        with self.assertRaises(KeyError):
            ch.evaluate({"u": np.zeros((1, 2, 2), np.float32)},
                        {"grid": [2, 2]}, ["vel_shear"])

    def test_filters_are_shape_preserving_over_ny_nx(self):
        # The temporal filters moved here must accept a (T, ny, nx) field and
        # return the same shape (they filter along axis 0, block-independent).
        x = np.random.default_rng(0).standard_normal((64, 3, 5)).astype(np.float32)
        b = ch.butter_band_energy(x, 30.0, (2.0, 8.0))
        m = ch.morlet_band(x, 30.0, (2.0, 8.0))
        self.assertEqual(b.shape, x.shape)
        self.assertEqual(m.shape, x.shape)
        self.assertTrue(np.isfinite(b).all())


class WithDerivedChannelsTest(unittest.TestCase):
    def _cd(self, channels):
        return ChannelData(meta={"grid": [4, 5], "n_frames": 1},
                           channels=channels)

    def test_derives_when_base_fields_present(self):
        ny, nx = 4, 5
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
        cd = self._cd({"change": xx[None], "u": xx[None], "v": (-yy)[None]})
        out = with_derived_channels(cd, ch.VELOCITY_GRADIENT)
        for name in ch.VELOCITY_GRADIENT:
            self.assertIn(name, out.available)
        self.assertEqual(out.meta["channels_computed"],
                         sorted(out.channels))

    def test_noop_without_base_fields(self):
        cd = self._cd({"change": np.zeros((1, 4, 5), np.float32)})
        out = with_derived_channels(cd, ch.VELOCITY_GRADIENT)
        self.assertIs(out, cd)                # unchanged, safe to call always


if __name__ == "__main__":
    unittest.main()
