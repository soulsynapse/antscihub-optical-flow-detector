"""Native gray16 preprocessing fast path.

The ROI decoder carries sub-8-bit precision in a uint16 atlas.  A bare z-score
does not need that plane converted to 0..255 first because positive scale cancels
from the normalization.  These tests pin both the numerical bound and the strict
conditions under which that shortcut is allowed.
"""
import numpy as np
import pytest

from core.config import PreprocessConfig
from core.preprocess import Preprocessor


def test_native_gray16_matches_the_established_scaled_zscore():
    rng = np.random.default_rng(17)
    native = rng.integers(0, 65536, (73, 79), dtype=np.uint16)
    pre = Preprocessor(PreprocessConfig(downsample=1.0, normalize="zscore"),
                       native.shape[1], native.shape[0])

    established = pre.apply(native.astype(np.float32) * (255.0 / 65535.0))
    fast = pre.apply_native_gray16(native)

    assert fast.dtype == np.float32
    np.testing.assert_allclose(fast, established, rtol=0.0, atol=1.6e-5)


def test_native_gray16_constant_plane_stays_float32_zero():
    native = np.full((6, 8), 32768, np.uint16)
    pre = Preprocessor(PreprocessConfig(downsample=1.0, normalize="zscore"),
                       8, 6)

    fast = pre.apply_native_gray16(native)

    assert fast.dtype == np.float32
    np.testing.assert_array_equal(fast, np.zeros((6, 8), np.float32))


@pytest.mark.parametrize("override", [
    {"normalize": "off"},
    {"normalize": "clahe"},
    {"registration": "phase"},
    {"denoise": "mean"},
    {"bg_subtract": "mog2"},
])
def test_native_gray16_is_restricted_to_the_scale_invariant_pipeline(override):
    cfg = PreprocessConfig(downsample=1.0, **override)
    pre = Preprocessor(cfg, 8, 6)

    assert not pre.accepts_native_gray16
    with pytest.raises(ValueError, match="bare z-score"):
        pre.apply_native_gray16(np.zeros((6, 8), np.uint16))


def test_native_gray16_rejects_wrong_dtype_and_geometry():
    pre = Preprocessor(PreprocessConfig(downsample=1.0, normalize="zscore"),
                       8, 6)
    with pytest.raises(ValueError, match="uint16 HxW"):
        pre.apply_native_gray16(np.zeros((6, 8), np.float32))
    with pytest.raises(ValueError, match="expected 8x6"):
        pre.apply_native_gray16(np.zeros((5, 8), np.uint16))
