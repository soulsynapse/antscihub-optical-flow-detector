"""Turn lines drawn on a frame into the calibration fields the tool already has.

Two measurements, both drawn as a line on a source frame:

* against a **known fiducial** -- a ruler, a scale bar, an arena feature of known
  size -- giving ``pixels_per_mm``;
* along the **animal**, giving its length, which becomes ``body_length_mm`` once
  the fiducial has fixed the scale.

Both write into fields that already exist (``core/roi.py``: ``pixels_per_mm``,
``body_length_mm``), so nothing downstream learns a new concept. Exports and
``speed_body_lengths_s`` read the same two numbers.

Everything here is in SOURCE pixels
------------------------------------
``core/roi.py`` fixes the convention: calibration is stored in source pixels per
mm and the working scale is applied at use time (``working_px_per_mm =
pixels_per_mm * downsample``). A line drawn on a *cropped* frame is still in
source pixels -- a crop is a translation -- but a line drawn on a *resized*
display is not, so the caller must map display coordinates back before calling
anything here. That mapping is the one place this can silently go wrong, which is
why this module takes lengths in source pixels and never sees a widget.

The fiducial cancels out of the downsample dialog's readout
------------------------------------------------------------
The dialog reports working px per body length, which is
``pixels_per_mm * body_length_mm * scale``. Substituting the two measurements::

    pixels_per_mm    = L_fid  / mm_fid
    body_length_mm   = L_body / pixels_per_mm
    -> px_per_bl     = L_body * scale

So **the ruler is not needed to answer the resolution question at all** -- the
animal line alone gives it, exactly. The fiducial is needed only to *store* the
result in millimetres, i.e. to make it portable to exports, to other clips, and
to anything reasoning in physical units. :func:`working_px_per_body_length_from_line`
exists so the dialog can show the number the moment the animal line is drawn,
rather than making the user find a ruler before it will tell them anything.

Uncertainty is reported, and it is not a quality score
-------------------------------------------------------
A line is placed by hand at some display zoom, so its endpoints carry roughly one
display pixel of error each, which is ``1/zoom`` source pixels. Short lines are
therefore badly conditioned: a 20 px animal measured at 1:1 is +/-7%, and that
error propagates into every physical number derived from it. :func:`line_error_px`
and :func:`relative_error` make it visible so a user can respond by zooming in or
drawing along something longer.

This is a *propagated measurement error*, not a judgement about the data -- the
distinction the rest of this batch turns on. It says how well the line was drawn,
and claims nothing about whether the scale still resolves the behaviour.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# One display pixel of placement error per endpoint, combined over two endpoints
# in quadrature: sqrt(2). Endpoint error is dominated by hand and cursor, not by
# the display grid, so this is a floor rather than a full error budget -- it is
# labelled as such wherever it is shown.
_ENDPOINT_ERR_DISPLAY_PX = 1.0


def line_length_px(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    """Euclidean length of a line, in whatever units its endpoints are in.

    Assumes square pixels. Anamorphic footage would need the sample aspect ratio
    applied first; no source the tool has handled has one, and a wrong assumption
    here would be invisible, so it is stated rather than silently accommodated.
    """
    return math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))


def line_error_px(zoom: float) -> float:
    """Placement error of a drawn line, in source pixels, at a display ``zoom``.

    ``zoom`` is display pixels per source pixel: 1.0 is 1:1, 4.0 is magnified 4x
    (so the error shrinks), 0.25 is a 5312 px frame fitted into ~1300 px of
    widget (so one display pixel is 4 source pixels and the error grows).
    """
    if zoom <= 0:
        raise ValueError("zoom must be positive")
    return math.sqrt(2.0) * _ENDPOINT_ERR_DISPLAY_PX / float(zoom)


def relative_error(length_px: float, err_px: float) -> float:
    """Fractional error of a length. 1.0 (i.e. 100%) for a degenerate line."""
    if length_px <= 0:
        return 1.0
    return min(1.0, float(err_px) / float(length_px))


def pixels_per_mm_from_line(length_px: float, known_mm: float) -> float:
    """Source pixels per mm from a line drawn across something of known size."""
    if length_px <= 0:
        raise ValueError("The fiducial line has zero length.")
    if known_mm <= 0:
        raise ValueError("The fiducial's known length must be positive.")
    return float(length_px) / float(known_mm)


def mm_from_line(length_px: float, pixels_per_mm: float) -> float:
    """Physical length of a drawn line, given a fixed scale."""
    if pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm must be positive.")
    return float(length_px) / float(pixels_per_mm)


def working_px_per_body_length_from_line(body_length_px: float,
                                         scale: float) -> float:
    """Working pixels per body length, straight from the animal line.

    No fiducial: see the module docstring -- it cancels exactly. This is the
    number the downsample dialog's "what you keep" readout wants, and it is
    available from one line.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    return max(0.0, float(body_length_px)) * float(scale)


@dataclass(frozen=True)
class Calibration:
    """A completed calibration, ready to write onto a replicate dict.

    ``pixels_per_mm`` is None when only the animal was measured -- a legitimate
    partial state, since that alone answers the resolution question. Callers
    persisting to ``body_length_mm`` need both; callers previewing resolution
    need only :attr:`body_length_px`.
    """
    pixels_per_mm: float | None
    body_length_px: float | None
    fiducial_px: float | None = None
    fiducial_mm: float | None = None
    # Fractional error on each measured line, propagated from the zoom it was
    # drawn at. None where the corresponding line was not drawn.
    fiducial_rel_err: float | None = None
    body_rel_err: float | None = None

    @property
    def body_length_mm(self) -> float | None:
        if self.pixels_per_mm is None or self.body_length_px is None:
            return None
        return mm_from_line(self.body_length_px, self.pixels_per_mm)

    @property
    def body_length_mm_rel_err(self) -> float | None:
        """Errors on the two independent lines, combined in quadrature."""
        if self.fiducial_rel_err is None or self.body_rel_err is None:
            return None
        return math.hypot(self.fiducial_rel_err, self.body_rel_err)

    def working_px_per_body_length(self, scale: float) -> float | None:
        if self.body_length_px is None:
            return None
        return working_px_per_body_length_from_line(self.body_length_px, scale)

    def as_replicate_fields(self) -> dict:
        """The subset of replicate-dict fields this calibration determines.

        Only keys it actually measured, so a partial calibration merged onto a
        replicate never clears a field the user set by hand elsewhere.

        ``body_length_px`` is written alongside the two established fields, and
        it is the one addition to the replicate schema this module makes. It is
        here because the animal-only calibration is a legitimate and useful state
        that the existing pair cannot express: without a fiducial there is no
        ``body_length_mm``, so the resolution answer -- which needs no fiducial
        (see the module docstring) -- would exist only while the window was open.
        Storing the pixel length lets it survive, in the same source-pixel units
        the box geometry already uses. Everything downstream keeps reading the mm
        fields; nothing is required to know about this one.
        """
        out: dict = {}
        if self.pixels_per_mm is not None:
            out["pixels_per_mm"] = float(self.pixels_per_mm)
        blm = self.body_length_mm
        if blm is not None:
            out["body_length_mm"] = float(blm)
        if self.body_length_px is not None:
            out["body_length_px"] = float(self.body_length_px)
        return out
