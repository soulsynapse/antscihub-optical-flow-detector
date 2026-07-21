"""The render strip's fields, and the two properties that make it honest.

The panel's whole job is to show what a scale costs in resolution. Two things
would silently destroy that and neither is visible by looking at the picture:

* normalising each tile against its own range, which auto-contrasts the
  amplitude loss away and makes every scale look equally vivid;
* returning tiles at their own size, so a coarse scale reads as "smaller
  picture" rather than "coarser picture" once the widget lays them out.

Both are pinned here. The dimming direction is pinned too, because it is the
mechanism behind the finding that retired the empirical detection panel:
downsampling averages pixels *before* they are differenced, so per-block band
power falls with scale and a fixed absolute threshold catches less whether or
not the behaviour is still resolved.
"""
import numpy as np
import pytest

from core.config import PreprocessConfig
from core.scale_render import (fit_box_to, render_box_at_scales,
                               _preprocess_cfg)


class _FakeSource:
    """Stands in for ``core.video.VideoSource`` over a synthetic clip."""
    frames: list = []
    width = 64
    height = 64

    def __init__(self, path):
        self.info = type("I", (), {"frame_count": len(self.frames),
                                   "width": self.width,
                                   "height": self.height})()

    def frame_at(self, i):
        return self.frames[i] if 0 <= i < len(self.frames) else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _clip(n=4, size=64):
    """A finely textured field shifting one pixel per frame.

    Texture, not a flat blob: a hard-edged shape has no high spatial frequencies
    to lose, so area-averaging preserves its edge contrast almost exactly and the
    difference field does NOT dim -- an earlier version of this fixture used one
    and made the dimming tests fail for a reason that had nothing to do with the
    code. Fine texture is what downsampling actually destroys, and it is what
    real footage has.
    """
    rng = np.random.default_rng(0)
    field = rng.integers(40, 220, (size + n, size + n), dtype=np.uint8)
    return [np.repeat(field[t:t + size, t:t + size, None], 3, axis=2)
            for t in range(n)]


@pytest.fixture
def fake_video(monkeypatch):
    _FakeSource.frames = _clip()
    monkeypatch.setattr("core.scale_render.VideoSource", _FakeSource)
    return "fake.mp4"


SCALES = (1.0, 0.5, 0.25)


def test_working_size_tracks_the_scale(fake_video):
    """Tiles come back at WORKING resolution, not padded to a common size.

    The widget upscales them with NEAREST to a common display size; if this
    returned equal-sized arrays there would be nothing left to upscale and the
    strip would show seven identical pictures.
    """
    rs = render_box_at_scales(fake_video, (0, 0, 64, 64), 0, SCALES)
    assert [(r.width, r.height) for r in rs] == [(64, 64), (32, 32), (16, 16)]
    for r in rs:
        assert r.gray.shape == (r.height, r.width)
        assert r.change.shape == (r.height, r.width)
        assert r.gray.dtype == np.uint8 and r.change.dtype == np.uint8


def test_change_field_dims_with_scale(fake_video):
    """The mechanism the panel exists to show, in the direction it must run.

    Averaging before differencing costs contrast, so the difference field is
    weaker at coarser scales. Asserted on the MEAN and not the peak: a hard
    high-contrast edge keeps its peak under area-averaging whenever the edge
    happens to align with the coarser grid, so the peak is not the quantity that
    moves. Measured on real footage the mean fell 12.97 -> 5.59 across
    1.00 -> 0.10, while p99 stayed within a factor of two.
    """
    rs = render_box_at_scales(fake_video, (0, 0, 64, 64), 0, SCALES)
    means = [float(r.change.mean()) for r in rs]
    assert means[0] > means[1] > means[2]


def test_display_range_is_shared_not_per_tile(fake_video):
    """A tile must render differently depending on what it is shown NEXT TO.

    That is the direct signature of a shared normalisation, and it is what a
    per-tile one would destroy: auto-contrasting each tile drives them all to
    the same range and the amplitude loss vanishes -- the same failure as the
    withdrawn ``sig_corr`` reading, where a flattering normalisation hid a real
    difference. Rendering 0.25 beside 1.0 must therefore NOT match rendering it
    alone, where it is its own reference.
    """
    beside = render_box_at_scales(fake_video, (0, 0, 64, 64), 0, (1.0, 0.25))[1]
    alone = render_box_at_scales(fake_video, (0, 0, 64, 64), 0, (0.25,))[0]
    assert beside.scale == alone.scale == 0.25
    assert beside.change.mean() < alone.change.mean()
    # The reference tile is the one allowed to define the top of the range.
    assert alone.change.max() == 255


def test_reference_is_the_largest_scale_asked_for(fake_video):
    """Not hardcoded to 1.0: a caller comparing a narrower range would otherwise
    normalise against a scale that was never rendered, putting the whole strip
    on an invisible axis."""
    rs = render_box_at_scales(fake_video, (0, 0, 64, 64), 0, (0.5, 0.25))
    assert rs[0].scale == 0.5 and rs[0].change.max() == 255


def test_static_pair_does_not_amplify_noise(fake_video):
    """Nothing moving anywhere must render as black, not as convincing motion.

    Normalising against a ~zero reference would stretch quantisation noise to
    full contrast and show a difference field for a scene where nothing
    happened.
    """
    _FakeSource.frames = [_clip()[0], _clip()[0].copy()]
    rs = render_box_at_scales(fake_video, (0, 0, 64, 64), 0, SCALES)
    assert all(r.change.max() == 0 for r in rs)


def test_last_frame_falls_back_instead_of_failing(fake_video):
    """A window ending on the final frame is ordinary. The grey half must still
    render rather than the whole panel erroring out."""
    rs = render_box_at_scales(fake_video, (0, 0, 64, 64), 99, SCALES)
    assert len(rs) == len(SCALES)
    assert rs[0].gray.max() > 0


def test_empty_box_is_an_error_not_a_blank_tile(fake_video):
    with pytest.raises(ValueError):
        render_box_at_scales(fake_video, (10, 10, 10, 20), 0, SCALES)


def test_no_scales_renders_nothing(fake_video):
    assert render_box_at_scales(fake_video, (0, 0, 64, 64), 0, ()) == []


def test_fitted_stages_are_forced_off():
    """Registration and background subtraction are stateful across frames and a
    two-frame render cannot reproduce that state, so they are forced off exactly
    as the tensor path forces them off. ``normalize`` is per-frame and is
    carried through, because it changes what the solve sees."""
    base = PreprocessConfig(downsample=0.5, normalize="clahe",
                            registration="phase", bg_subtract="median",
                            denoise="median")
    cfg = _preprocess_cfg(base, 0.25)
    assert cfg.downsample == 0.25
    assert cfg.normalize == "clahe"
    assert (cfg.registration, cfg.bg_subtract, cfg.denoise) == ("off",) * 3


class TestFitBox:
    def test_leaves_a_small_box_alone(self):
        assert fit_box_to((10, 20, 310, 330), 420) == (10, 20, 310, 330)

    def test_crops_a_large_box_about_its_centre(self):
        x0, y0, x1, y1 = fit_box_to((0, 0, 5312, 2988), 420)
        assert max(x1 - x0, y1 - y0) == 420
        # Centred, so the strip shows the middle of the frame rather than a
        # corner -- an uncropped fallback box is usually the whole arena.
        assert abs((x0 + x1) / 2 - 2656) <= 1
        assert abs((y0 + y1) / 2 - 1494) <= 1

    def test_stays_inside_the_original_box(self):
        src = (2157, 975, 2456, 1285)
        x0, y0, x1, y1 = fit_box_to(src, 100)
        assert x0 >= src[0] and y0 >= src[1] and x1 <= src[2] and y1 <= src[3]

    def test_preserves_aspect(self):
        x0, y0, x1, y1 = fit_box_to((0, 0, 800, 400), 200)
        assert (x1 - x0, y1 - y0) == (200, 100)
