"""Embeddable per-block speed explorer.

A throwaway diagnostic to answer one question the main app currently makes you
guess at: what does the speed signal actually look like, per block, over time --
and what does the spatial detector (an in-band clump that must bridge nearby
blocks) do as you move its threshold and gap? The raw
per-block distribution is shown as a density heatmap rather than a spatial mean,
because the mean buries the sparse fast blocks that are the actual signal.

It does NOT recompute optical flow. It opens a feature cache you already built in
the main app (Preprocessing & Flow) and reads its cached `speed` / `u` / `v`
block arrays, so
every number here is exactly the number the real pipeline sees -- there is no
second, slightly-different flow implementation to mistrust.

Left panel  : the video with replicate-aware optical-flow overlay + transport.
Right panel : a long scrollable column of speed-only readouts, following the
              same double-integration instrument as the structure-tensor
              explorer. The per-frame per-block distribution is shown as a
              density heatmap (context only); the *windowed* distribution --
              mean speed over the integration window W, O(1) via prefix sums
              -- is the detection channel and carries the detection *band*:
              a draggable min/max pair (in that plot's own Y units). Detection
              is ``band_lo <= value <= band_hi`` -- the same range primitive
              the real pipeline thresholds -- and the in-band group reshapes
              live as you drag either handle. A handle parked at the plot's
              edge means *unbounded* on that side (readout shows an infinity
              sign): individual block values run far past the plotted density
              curve's own maximum, so an unbounded side must not silently
              reject them.

              The band feeds two gates. Temporal: the in-band count is
              averaged over a detection window D (its own band, min handle =
              sustained-evidence threshold, max handle = whole-replicate
              artifact rejection) and a frame is a positive detection when
              that windowed count sits inside the count band -- a 1-frame
              spike of N blocks dilutes to N/D and cannot fake a sustained
              event. Spatial (diagnostic): in-band blocks within a Chebyshev
              block *gap* are single-link clustered into clumps and the
              largest clump's weighted size is plotted. Both windows follow
              one centered/trailing convention (see the checkbox), and a
              value-vs-W density at the cursor shows where more integration
              stops paying off.

Current caches process each replicate independently and pack their block grids
into a sparse storage atlas.  This explorer maps those tiles back to their real
source-frame boxes; atlas separators are never displayed or included in plots.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QImage, QKeySequence, QPainter, QPen,
                         QPolygonF, QShortcut)
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                             QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                             QSlider, QVBoxLayout, QWidget)

from gui.video_panel import FrameView

# -- palette -----------------------------------------------------------------
BG = QColor(24, 24, 24)
PLOT_BG = QColor(12, 12, 12)
LINE = QColor(120, 215, 255)
LINE2 = QColor(255, 170, 80)
CURSOR = QColor(255, 210, 80)
TXT = QColor(210, 210, 210)
TXT_DIM = QColor(140, 140, 140)
BAND = QColor(255, 95, 95)          # detection min/max lines drawn on the plot
DETECT = QColor(60, 220, 110)       # positive-detection spans shaded behind it
# "Nothing has covered this column yet." A cool blue-grey, deliberately off both
# heatmap ramps, and carried by a hatch rather than a flat fill -- see
# DensityPlot._hatch_unexamined for why a tone could not work.
_UNEXAMINED_BG = np.array([22, 24, 30], np.float64)
_UNEXAMINED_FG = np.array([44, 48, 60], np.float64)
DISPLAY_MAX_W = 1280


def _regions_from_meta(meta: dict, grid: tuple[int, int]) -> list[dict]:
    """Return validated cache regions, with a whole-frame legacy fallback.

    ``replicate_tiles`` describe sparse atlas storage, not image coordinates.
    Keeping this normalization in one place makes every plot and overlay use the
    same ownership boundary.
    """
    ny, nx = grid
    raw = meta.get("replicate_tiles", [])
    if not raw:
        src_w = int(meta.get("src_width", meta.get("work_width", nx)))
        src_h = int(meta.get("src_height", meta.get("work_height", ny)))
        return [{
            "id": None,
            "label": "Whole frame",
            "frac": (0.0, 0.0, 1.0, 1.0),
            "source_box": (0, 0, src_w, src_h),
            "grid": (ny, nx),
            "atlas_bbox": (0, 0, ny, nx),
        }]

    regions = []
    for i, tile in enumerate(raw):
        try:
            y0, x0, y1, x1 = map(int, tile["atlas_bbox"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Replicate tile {i} has no valid atlas_bbox") from exc
        if not (0 <= y0 < y1 <= ny and 0 <= x0 < x1 <= nx):
            raise ValueError(
                f"Replicate tile {tile.get('id', i)} has atlas_bbox "
                f"{(y0, x0, y1, x1)} outside cache grid {(ny, nx)}")
        tile_grid = tuple(map(int, tile.get("grid", (y1 - y0, x1 - x0))))
        if tile_grid != (y1 - y0, x1 - x0):
            raise ValueError(
                f"Replicate tile {tile.get('id', i)} grid {tile_grid} does not "
                f"match atlas_bbox {(y0, x0, y1, x1)}")

        frac = tuple(float(v) for v in tile.get("frac", (0, 0, 1, 1)))
        source_box = tuple(map(int, tile.get("source_box", (0, 0, 1, 1))))
        if len(frac) != 4 or len(source_box) != 4:
            raise ValueError(f"Replicate tile {tile.get('id', i)} has bad geometry")
        regions.append({
            **tile,
            "id": int(tile.get("id", i)),
            "label": str(tile.get("label", f"rep{tile.get('id', i)}")),
            "frac": frac,
            "source_box": source_box,
            "grid": tile_grid,
            "atlas_bbox": (y0, x0, y1, x1),
        })
    return regions


# -- one time-series readout -------------------------------------------------

class MiniPlot(QWidget):
    """A compact autoscaled sparkline for one (T,) series, with a frame cursor.

    Long series are decimated to the widget width by taking the MAX within each
    pixel column, not the mean: a behavior is a brief burst, and the mean per
    column would smear a 3-frame spike into the baseline -- exactly the signal we
    are here to see. The current-frame exact value is printed regardless, so the
    decimation never hides the number you are reading.

    When it is the selected detection channel the plot grows to double height and
    grows a detection *band*: two draggable 1px lines (a minimum and a maximum, in
    the plot's own Y units) whose rejected strips (outside the band) are shaded.
    The lines extend a few pixels past the plot's right edge into a handle
    margin so they read as pull handles, mirroring the sketch. Detection is
    ``lo <= value <= hi`` -- a simple inclusive band on the plotted value.

    A handle dragged to (or past) the plot's edge sets that side to +/-inf --
    "no limit on this side" -- and the readout shows an infinity sign. This is
    load-bearing, not cosmetic: in per-block detection modes the band is applied
    to individual block values, which exceed the plotted spatial-mean series'
    own maximum, so a hi clamped to the plot's data max would silently reject
    exactly the fastest blocks. Bands therefore also *seed* unbounded.
    """
    seek_requested = pyqtSignal(int)
    band_changed = pyqtSignal()       # emitted continuously while dragging a line
    band_committed = pyqtSignal()     # emitted once on release (expensive recompute)
    collapse_toggled = pyqtSignal(bool)   # user clicked the [+]/[-] header marker

    BASE_H = 66
    EXPANDED_H = 132
    # A collapsed plot is exactly its own header: the title line and nothing
    # else. Sized to a checkbox so a row of collapsed per-channel heatmaps lines
    # up with the checkbox column beside it (T13) instead of leaving a black
    # slab where an unpopulated heatmap would have been.
    COLLAPSED_H = 18
    HANDLE_MARGIN = 20                 # right-edge room for the band pull handles
    GRAB_PX = 6                        # vertical pick tolerance for a band line
    TOGGLE_W = 14                      # hit width of the [+]/[-] marker

    def __init__(self, title: str, unit: str = "", color: QColor = LINE):
        super().__init__()
        self.title = title
        self.unit = unit
        self.color = color
        self.y: np.ndarray = np.zeros(0, np.float32)
        self.cursor = 0
        # Series version + memoised derivations of it. Everything paintEvent
        # draws except the cursor is a function of (series, widget size), but it
        # was all being rebuilt per paint -- i.e. per cursor move, so once per
        # frame during playback. The envelope in particular is a Python loop
        # calling np.nanmax once per pixel column: 1.7 ms/paint at T=1200,
        # against 0.11 ms for the DensityPlot beside it, spent re-deriving a
        # constant. Same shape as the DensityPlot._data_range memo below it.
        # Bumped by set_series (and by DensityPlot.set_matrix, which writes y
        # directly); the caches key on it, so staleness is not representable.
        self._ver = 0
        self._range_memo: tuple[float, float] | None = None
        self._env_memo: tuple[tuple, np.ndarray] | None = None
        self._poly_memo: tuple[tuple, "QPolygonF"] | None = None
        # T15: frames the detector calls positive, painted as green spans behind
        # the series. A picture of a quantity computed elsewhere -- this widget
        # never derives it, so a collapsed plot cannot disarm the detector.
        self._detect_mask: np.ndarray | None = None
        # Sticky value axis. Off by default, so every existing caller keeps the
        # per-series autoscale it had. On, the axis only ever WIDENS: the bounds
        # are a running union over every series this plot has been shown, and
        # they are what paintEvent, the band handles and the drag arithmetic all
        # read. This exists because a live pass hands the plot a new trailing
        # window ~10x a second, and a per-window autoscale makes the axis --
        # and therefore the pixel position of a band the user placed against it
        # -- move continuously under them while the underlying threshold has not
        # changed. Reset is explicit (see reset_sticky_range), because the only
        # things that invalidate the accumulated bounds are a change of what is
        # being measured, not a change in the data.
        self._sticky = False
        self._sticky_range: tuple[float, float] | None = None
        # Widen the drawn value range to always cover the band endpoints. For a
        # per-window autoscaling threshold plot: when the window's own data max
        # falls below the threshold, a plain 0..data_max axis stops reaching the
        # band and the handle vanishes off the top -- so the one control you came
        # to read leaves the plot exactly when the window goes quiet. Including
        # the band keeps it anchored at its true value on every window.
        self._include_band_in_range = False
        self.band_active = False
        self.band_lo: float | None = None
        self.band_hi: float | None = None
        self._drag: str | None = None
        self._expanded = False
        self._collapsible = False
        # Two separate states, deliberately. ``_user_collapsed`` is what the
        # user asked for via [+]; ``_auto_collapsed`` is a data-driven collapse
        # (an unpopulated heatmap). Folding them into one flag would let a
        # refresh that finds an empty matrix overwrite the user's intent, so
        # that a plot the user expanded stays expanded once its data arrives.
        self._user_collapsed = False
        self._auto_collapsed = False
        # T13/T27: a plot with no series has nothing but a black slab to draw,
        # so it collapses itself to header height until its data arrives.
        self._auto_collapse_empty = False
        self.setMinimumHeight(self.BASE_H)
        self.setMaximumHeight(self.BASE_H)

    def set_series(self, y: np.ndarray) -> None:
        self.y = np.asarray(y, np.float32)
        self._bump_series_version()
        # A mask belongs to the series it was computed from, so a new series
        # invalidates it. Batch D's rule again: EMPTY rather than stale. A mask
        # held across a replicate switch would shade the new clip with the
        # previous one's detections -- indistinguishable from a real result,
        # and wrong in the direction that invents events rather than losing
        # them. Every caller recomputes the gate immediately after setting the
        # series, so the blank is momentary.
        self._detect_mask = None
        self._refresh_auto_collapse()
        self._repaint_unless_collapsed()

    def set_detect_mask(self, mask: np.ndarray | None) -> None:
        """Per-frame positive-detection flags to shade behind the series."""
        self._detect_mask = (None if mask is None
                             else np.asarray(mask, bool).ravel())
        self._repaint_unless_collapsed()

    def set_cursor(self, frame: int) -> None:
        self.cursor = int(frame)
        self._repaint_unless_collapsed()

    def set_expanded(self, on: bool) -> None:
        self._expanded = bool(on)
        self._apply_height()

    # -- collapse ------------------------------------------------------------

    def set_collapsible(self, on: bool) -> None:
        """Grow a clickable [+]/[-] marker in the header. Off by default, so
        explorers that have not opted in keep their previous geometry exactly."""
        self._collapsible = bool(on)
        self.update()

    def is_collapsed(self) -> bool:
        return self._user_collapsed or self._auto_collapsed

    def set_collapsed(self, on: bool) -> None:
        """The user's explicit collapse state."""
        self._user_collapsed = bool(on)
        self._sync_collapse()

    def set_auto_collapsed(self, on: bool) -> None:
        """Data-driven collapse (nothing to draw), independent of user intent."""
        self._auto_collapsed = bool(on)
        self._sync_collapse()

    def set_auto_collapse_empty(self, on: bool) -> None:
        """Collapse this plot whenever it has nothing to draw."""
        self._auto_collapse_empty = bool(on)
        self._refresh_auto_collapse()

    def _is_empty(self) -> bool:
        """What "nothing to draw" means for this plot. A line plot draws its
        series; :class:`DensityPlot` draws its matrix and overrides this."""
        return self.y.size == 0

    def _refresh_auto_collapse(self) -> None:
        self.set_auto_collapsed(self._auto_collapse_empty and self._is_empty())

    def _sync_collapse(self) -> None:
        # Dropping the rendered image is the memory half of T14; skipping the
        # paint is the latency half, and it is the larger one -- a density
        # repaint is an O(T*K) bincount over the whole matrix, per resize, per
        # cursor move.
        if self.is_collapsed():
            self._release_render_cache()
        self._apply_height()

    def _release_render_cache(self) -> None:
        """Drop any cached rendered array. No-op for a line plot, which holds
        none; the heatmap subclasses override it."""

    def _apply_height(self) -> None:
        if self.is_collapsed():
            h = self.COLLAPSED_H
        else:
            h = self.EXPANDED_H if self._expanded else self.BASE_H
        self.setMinimumHeight(h)
        self.setMaximumHeight(h)
        self.update()

    def _repaint_unless_collapsed(self) -> None:
        if not self.is_collapsed():
            self.update()

    def _toggle_rect(self) -> QRectF:
        return QRectF(4, 2, self.TOGGLE_W, 14)

    def _has_marker(self) -> bool:
        """Whether the header carries a marker at all.

        A collapsible plot always shows one. A NON-collapsible plot shows one
        only while auto-collapsed: it has no [+] to offer, but it has still
        shrunk to a header, and a pane that silently loses its body with no
        mark is the same unexplained control the [.] form exists to prevent."""
        return self._collapsible or self._auto_collapsed

    def _title_x(self) -> int:
        return 8 + self.TOGGLE_W if self._has_marker() else 8

    def _paint_header(self, p: QPainter) -> None:
        """The [+]/[-] marker. Every paintEvent draws it, collapsed or not --
        it is the only way back from a collapsed plot.

        An auto-collapsed plot draws [.] instead: expanding it would show an
        empty pane, so the toggle genuinely does nothing and must not advertise
        otherwise. Its data arrives by another route (checking its channel),
        and a [+] that silently no-ops would read as a broken control."""
        if not self._has_marker():
            return
        p.setFont(QFont("Consolas", 7))
        p.setPen(TXT_DIM)
        if self._auto_collapsed:
            mark = "[.]"
        else:
            mark = "[+]" if self.is_collapsed() else "[-]"
        p.drawText(6, 12, mark)

    def _paint_collapsed(self, p: QPainter) -> bool:
        """Draw the header-only form and report that painting is done."""
        if not self.is_collapsed():
            return False
        p.fillRect(self.rect(), BG)
        self._paint_header(p)
        p.setFont(QFont("Consolas", 7))
        p.setPen(TXT_DIM)
        p.drawText(self._title_x(), 12, self.title)
        p.end()
        return True

    def set_band_active(self, on: bool) -> None:
        """Show/hide the draggable band. Lazily seeds it wide open (+/-inf).

        Seeding once (never on every ``set_series``) keeps the band an absolute
        value: focusing a replicate or changing the window must not silently
        reinterpret a threshold the user has placed. Seeding unbounded (rather
        than to the plot's data range) means "no threshold set" really accepts
        everything -- including per-block values above the plotted series' max.
        """
        self.band_active = bool(on)
        if on and (self.band_lo is None or self.band_hi is None):
            self.band_lo, self.band_hi = float("-inf"), float("inf")
        self.update()

    def _bump_series_version(self) -> None:
        """Invalidate every memo derived from the series. The single writer for
        all of them, so a new derivation cannot forget to be invalidated."""
        self._ver += 1
        self._range_memo = None
        self._env_memo = None
        self._poly_memo = None

    # -- value axis ----------------------------------------------------------

    def set_sticky_range(self, on: bool) -> None:
        """Accumulate the value axis instead of autoscaling per series."""
        if bool(on) == self._sticky:
            return
        self._sticky = bool(on)
        self._sticky_range = None
        self._bump_series_version()          # the drawn geometry depends on it
        self._repaint_unless_collapsed()

    def reset_sticky_range(self) -> None:
        """Forget the accumulated bounds and re-seed from the current series.

        Called when the plot changes what it is MEASURING -- another replicate,
        another channel -- not when it merely receives new data. Widening across
        a scope change would carry one replicate's outliers onto another's axis,
        which is the same category of error as showing stale data as current.
        """
        self._sticky_range = None
        self._bump_series_version()
        self._repaint_unless_collapsed()

    def set_include_band_in_range(self, on: bool) -> None:
        """Keep the drawn axis wide enough to show the band endpoints. See the
        ``_include_band_in_range`` note; used by the windowed-count threshold plot
        so its handle stays visible as the live window autoscales beneath it."""
        if bool(on) == self._include_band_in_range:
            return
        self._include_band_in_range = bool(on)
        self._bump_series_version()          # the drawn geometry depends on it
        self._repaint_unless_collapsed()

    def _widen_for_band(self, lo: float, hi: float) -> tuple[float, float]:
        """Union the range with the finite band endpoints. Endpoints are ``None``
        (unset) or +/-inf (a half-open threshold) when a side is unbounded, and
        neither belongs on the axis -- only a placed, finite handle does."""
        if not self._include_band_in_range:
            return lo, hi
        for edge in (self.band_lo, self.band_hi):
            if edge is not None and np.isfinite(edge):
                lo, hi = min(lo, float(edge)), max(hi, float(edge))
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def _data_range(self) -> tuple[float, float]:
        """The drawn value bounds: the raw per-series range, or its running
        union when sticky, widened to include the band when asked. Subclasses
        override :meth:`_raw_data_range`, so the sticky layer applies to every
        plot kind without being reimplemented."""
        lo, hi = self._raw_data_range()
        if not self._sticky:
            return self._widen_for_band(lo, hi)
        if self._is_empty():
            # An empty plot's raw range is the 0..1 placeholder, not a
            # measurement. Folding it in would permanently pull the accumulated
            # floor to 0 the first time a window arrives short.
            base = self._sticky_range if self._sticky_range else (lo, hi)
            return self._widen_for_band(*base)
        if self._sticky_range is not None:
            slo, shi = self._sticky_range
            lo, hi = min(lo, slo), max(hi, shi)
        self._sticky_range = (lo, hi)
        return self._widen_for_band(lo, hi)

    def _raw_data_range(self) -> tuple[float, float]:
        if self._range_memo is not None:
            return self._range_memo
        n = self.y.size
        if n == 0:
            return 0.0, 1.0                  # not memoised: nothing to reuse
        lo = float(np.nanmin(self.y))
        hi = float(np.nanmax(self.y))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        self._range_memo = (lo, hi)
        return self._range_memo

    def _envelope(self, cols: int, lo: float) -> np.ndarray:
        """Per-pixel-column MAX of the series (never the mean -- see the class
        docstring: a 3-frame burst must not be smeared into the baseline).

        ``np.fmax.reduceat`` replaces a Python loop over the columns. reduceat
        on a run of equal indices returns that single element, which is exactly
        the old loop's ``max(edges[i] + 1, edges[i + 1])`` clamp for a column
        narrower than one sample. fmax, not maximum: fmax ignores NaN, matching
        the nanmax the loop called.
        """
        key = (cols, self._ver)
        if self._env_memo is not None and self._env_memo[0] == key:
            return self._env_memo[1]
        n = self.y.size
        edges = np.linspace(0, n, cols + 1).astype(int)
        starts = np.minimum(edges[:-1], n - 1)
        env = np.fmax.reduceat(self.y, starts).astype(np.float32)
        # An all-NaN column has no max; the loop fell back to the axis minimum.
        env = np.where(np.isnan(env), lo, env)
        self._env_memo = (key, env)
        return env

    def band(self) -> tuple[float, float]:
        # An unset endpoint is unbounded on ITS OWN side only -- never force the
        # whole band open just because one handle hasn't been placed. (Forcing
        # both open silently swallowed a lone finite handle, so a band left
        # unseeded ignored every drag until both endpoints were assigned.)
        lo = float("-inf") if self.band_lo is None else self.band_lo
        hi = float("inf") if self.band_hi is None else self.band_hi
        return lo, hi

    def _plot_rect(self) -> QRectF:
        rm = self.HANDLE_MARGIN if self.band_active else 6
        return QRectF(6, 16, max(1, self.width() - 6 - rm),
                      max(1, self.height() - 22))

    def _fwd(self, v):
        """Value -> axis-position space. Identity here; :class:`DensityPlot`
        overrides this for its optional log value axis. Never touches stored
        band values -- only where they get drawn/picked."""
        return v

    def _inv(self, t):
        """Inverse of :meth:`_fwd`."""
        return t

    def _y_of(self, val: float, r: QRectF, lo: float, hi: float) -> float:
        tval, tlo, thi = self._fwd(val), self._fwd(lo), self._fwd(hi)
        return r.bottom() - (tval - tlo) / max(1e-12, thi - tlo) * r.height()

    def _val_of(self, y: float, r: QRectF, lo: float, hi: float) -> float:
        tlo, thi = self._fwd(lo), self._fwd(hi)
        tval = tlo + (r.bottom() - y) / max(1.0, r.height()) * (thi - tlo)
        return self._inv(tval)

    def paintEvent(self, _):
        p = QPainter(self)
        if self._paint_collapsed(p):
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), BG)
        r = self._plot_rect()
        p.fillRect(r, PLOT_BG)
        self._paint_header(p)

        n = self.y.size
        p.setFont(QFont("Consolas", 7))
        if n == 0:
            p.setPen(TXT_DIM)
            p.drawText(self._title_x(), 12, self.title)
            p.end()
            return

        lo, hi = self._data_range()
        self._paint_detect_spans(p, r, n)

        w = int(r.width())
        cols = max(1, min(n, w))
        env = self._envelope(cols, lo)

        def y_of(val: float) -> float:
            return self._y_of(val, r, lo, hi)

        # Zero baseline, if zero is in range -- makes "how far above nothing" legible.
        if lo < 0 < hi:
            yb = y_of(0.0)
            p.setPen(QPen(QColor(70, 70, 70), 1, Qt.PenStyle.DashLine))
            p.drawLine(int(r.left()), int(yb), int(r.right()), int(yb))

        # The polyline is cached too, not just the envelope behind it: building
        # it is another per-column Python loop, and between two frames of
        # playback only the cursor moves, so the whole curve is identical.
        # Keyed on the plot rect as well as the series, since the geometry is
        # what maps values to pixels.
        pkey = (cols, self._ver, r.left(), r.top(), r.width(), r.height(),
                lo, hi)
        if self._poly_memo is not None and self._poly_memo[0] == pkey:
            poly = self._poly_memo[1]
        else:
            xs = r.left() + (np.arange(cols) + 0.5) / cols * r.width()
            ys = self._y_of(env.astype(np.float64), r, lo, hi)
            poly = QPolygonF([QPointF(float(x), float(y))
                              for x, y in zip(xs, ys)])
            self._poly_memo = (pkey, poly)
        p.setPen(QPen(self.color, 1))
        p.drawPolyline(poly)

        if self.band_active:
            self._paint_band(p, r, lo, hi)

        # Cursor + current exact value.
        cx = r.left() + (self.cursor + 0.5) / n * r.width()
        p.setPen(QPen(CURSOR, 1))
        p.drawLine(int(cx), int(r.top()), int(cx), int(r.bottom()))

        cur = float(self.y[min(self.cursor, n - 1)])
        p.setPen(TXT)
        p.drawText(self._title_x(), 12, self.title)
        val_txt = f"{cur:.4g} {self.unit}".strip()
        p.setPen(QColor(self.color))
        p.drawText(int(r.right()) - 7 * len(val_txt) - 2, 12, val_txt)

        # min/max axis ticks
        p.setPen(TXT_DIM)
        p.drawText(int(r.left()), int(r.bottom()) + 8, f"{lo:.3g}")
        p.end()

    def _paint_detect_spans(self, p: QPainter, r: QRectF, n: int) -> None:
        """Shade the frames the detector called positive, full plot height.

        Runs are drawn in FRAME coordinates and floored to one pixel wide. The
        series itself is enveloped down to the widget's pixel columns, but a
        detection must not be: a single positive frame in a 14k-frame clip is
        0.03 px, and rounding it away would silently draw "nothing detected"
        over a real detection -- the failure shape this whole panel exists to
        avoid.
        """
        m = self._detect_mask
        if m is None or m.size == 0 or n <= 0 or not m.any():
            return
        m = m[:n] if m.size >= n else np.pad(m, (0, n - m.size))
        # Run starts/ends over the padded edges, so a run touching either end is
        # closed rather than dropped.
        d = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        fill = QColor(DETECT)
        fill.setAlpha(48)
        for a, b in zip(starts, ends):
            x0 = r.left() + a / n * r.width()
            x1 = r.left() + b / n * r.width()
            p.fillRect(QRectF(x0, r.top(), max(1.0, x1 - x0), r.height()), fill)

    def _paint_band(self, p: QPainter, r: QRectF, lo: float, hi: float) -> None:
        """Shade the rejected strips outside [band_lo, band_hi] and draw handles."""
        blo, bhi = self.band()
        ylo = self._y_of(min(max(blo, lo), hi), r, lo, hi)
        yhi = self._y_of(min(max(bhi, lo), hi), r, lo, hi)
        top, bot = min(ylo, yhi), max(ylo, yhi)

        shade = QColor(BAND)
        shade.setAlpha(30)
        if top > r.top():
            p.fillRect(QRectF(r.left(), r.top(), r.width(), top - r.top()), shade)
        if r.bottom() > bot:
            p.fillRect(QRectF(r.left(), bot, r.width(), r.bottom() - bot), shade)

        line_right = self.width() - 3          # a few px past the plot's right edge
        p.setPen(QPen(BAND, 1))
        for y in (yhi, ylo):
            p.drawLine(int(r.left()), int(y), int(line_right), int(y))
        p.setBrush(BAND)
        p.setPen(Qt.PenStyle.NoPen)
        for y in (yhi, ylo):
            p.drawEllipse(QPointF(line_right - 3, y), 3.0, 3.0)

        # Numeric readouts riding just inside each handle line. An unbounded
        # side is explicit (infinity sign), never a silent clamp to the plot max.
        def fmt(v: float) -> str:
            return ("∞" if v == float("inf")
                    else "-∞" if v == float("-inf") else f"{v:.3g}")
        p.setPen(BAND)
        p.setFont(QFont("Consolas", 7))
        p.drawText(int(r.right()) - 44, int(yhi) - 2, fmt(bhi))
        p.drawText(int(r.right()) - 44, int(ylo) + 9, fmt(blo))

    # -- band interaction ----------------------------------------------------

    def _line_ys(self) -> tuple[float, float, QRectF]:
        r = self._plot_rect()
        lo, hi = self._data_range()
        blo, bhi = self.band()
        ylo = self._y_of(min(max(blo, lo), hi), r, lo, hi)
        yhi = self._y_of(min(max(bhi, lo), hi), r, lo, hi)
        return ylo, yhi, r

    def _apply_drag(self, y: float) -> None:
        r = self._plot_rect()
        lo, hi = self._data_range()
        if self._drag == "hi":
            if y <= r.top():               # pulled off the top: no upper limit
                self.band_hi = float("inf")
            else:
                val = float(np.clip(self._val_of(y, r, lo, hi), lo, hi))
                self.band_hi = max(
                    val, self.band_lo if self.band_lo is not None else lo)
        else:
            if y >= r.bottom():            # pulled off the bottom: no lower limit
                self.band_lo = float("-inf")
            else:
                val = float(np.clip(self._val_of(y, r, lo, hi), lo, hi))
                self.band_lo = min(
                    val, self.band_hi if self.band_hi is not None else hi)
        self.update()
        self.band_changed.emit()

    def mousePressEvent(self, e):
        # The marker is checked before anything else and swallows the click:
        # collapsed, it is the only live target on the widget, and expanded it
        # sits inside the plot rect where a seek would otherwise fire.
        if self._collapsible and self._toggle_rect().contains(QPointF(e.pos())):
            self.set_collapsed(not self._user_collapsed)
            self.collapse_toggled.emit(not self.is_collapsed())
            return
        if self.is_collapsed():
            return
        if self.band_active and self.y.size:
            ylo, yhi, _ = self._line_ys()
            yv = e.pos().y()
            grab_lo = abs(yv - ylo) <= self.GRAB_PX
            grab_hi = abs(yv - yhi) <= self.GRAB_PX
            if grab_lo or grab_hi:
                if grab_lo and grab_hi:            # overlapping: pick by side
                    self._drag = "lo" if yv >= (ylo + yhi) / 2 else "hi"
                else:
                    self._drag = "lo" if grab_lo else "hi"
                self._apply_drag(yv)
                return
        n = self.y.size
        if n == 0:
            return
        r = self._plot_rect()
        frac = np.clip((e.pos().x() - r.left()) / max(1, r.width()), 0, 1)
        self.seek_requested.emit(int(frac * (n - 1)))

    def mouseMoveEvent(self, e):
        if self._drag:
            self._apply_drag(e.pos().y())

    def mouseReleaseEvent(self, e):
        if self._drag:
            self._drag = None
            self.band_committed.emit()


class DensityPlot(MiniPlot):
    """A time x value density heatmap of the whole per-block distribution.

    Where :class:`MiniPlot` collapses each frame to one summary line, this draws
    the entire per-block distribution: each column is a frame, each row a value
    bin, and brightness is how many blocks land in that (frame, value) cell.
    Counts are log-scaled so the handful of genuinely fast blocks -- the signal
    a spatial *mean* buries under the low-speed bulk -- stay visible instead of
    vanishing into the baseline.

    It is still a detection channel: the band handles ride the value axis exactly
    as on a line plot, so the shaded strips show precisely which slice of the
    distribution the detector rejects. The heatmap is a binned image (cost scales
    with pixels, not blocks x frames), so a full-clip cache never lags.
    """
    # 0 -> plot background, ramping to bright cyan-white at the densest bin.
    _RAMP = np.array([[12, 12, 12], [20, 60, 90], [30, 120, 170],
                      [90, 210, 255], [230, 250, 255]], np.float64)

    def __init__(self, title: str, unit: str = "", color: QColor = LINE):
        super().__init__(title, unit, color)
        self.matrix = np.zeros((0, 0), np.float32)
        # The matrix covers axis frames [axis_off, axis_off + T) of an axis
        # axis_total long. They differ once this heatmap's source lags the
        # per-frame plots stacked beside it -- see set_matrix.
        self.axis_off = 0
        self.axis_total = 0
        self._img: QImage | None = None
        self._img_key: tuple | None = None
        self._ver = 0
        # Memoised _data_range. The min/max is a property of the MATRIX, but it
        # was being recomputed from paintEvent, _line_ys and _apply_drag -- i.e.
        # several full T x B scans per mouse-move, and one per plot per frame
        # during playback, all returning the same number. At 1800 frames x 8000
        # blocks that measured 22.5 ms of a 22.8 ms repaint: effectively the
        # entire per-frame cost of this widget, spent re-deriving a constant.
        # Invalidated in set_matrix, which is the only writer.
        self._range: tuple[float, float] | None = None
        # Value-axis-only log1p toggle: spreads out the low-speed bulk that a
        # linear axis crushes against the bottom under a handful of fast-block
        # outliers. Never changes what's stored in the matrix or the band --
        # only where a given raw value gets drawn/picked on the Y axis.
        self._log_axis = False

    def _is_empty(self) -> bool:
        # A heatmap draws its matrix, not the per-frame max it derives, so
        # emptiness is the matrix's. The two can disagree: set_matrix leaves a
        # zero-length y for an empty matrix, but a subclass could carry a series
        # with nothing to shade behind it.
        return self.matrix.size == 0

    def _release_render_cache(self) -> None:
        self._img = None
        self._img_key = None

    def set_log_axis(self, on: bool) -> None:
        self._log_axis = bool(on)
        self._img = None
        self.update()

    def _fwd(self, v):
        return np.log1p(v) if self._log_axis else v

    def _inv(self, t):
        return np.expm1(t) if self._log_axis else t

    def set_matrix(self, m: np.ndarray, axis_off: int = 0,
                   axis_total: int | None = None) -> None:
        """Set the (T, K) matrix, optionally placing it on a LONGER axis.

        ``axis_off``/``axis_total`` serve the continuous-plots split, where the
        per-frame plots refresh at 10 Hz while this heatmap's per-block cube
        takes seconds to transform (measured: 6 s at T=30k, B=377). This matrix
        is then over an older, shorter span than the plots stacked above it --
        and because they share a widget width and are read down a column, a
        matrix drawn edge-to-edge over its OWN length would put its frame 7800
        directly above their frame 8300. That is an x-axis misregistration, not
        merely stale data: it invites reading a burst against the wrong value.
        A staleness label does not fix it.

        So the matrix is drawn at its true position on the shared axis and the
        uncovered tail is hatched as UNEXAMINED, reusing the coverage vocabulary
        `_Strip` established. Registration is exact everywhere and the lag reads
        as a visibly short frontier instead of a badge to be believed.

        Defaults reproduce the previous behaviour exactly, so every non-live
        caller is unaffected.
        """
        self.matrix = np.asarray(m, np.float32)          # (T, K)
        T = self.matrix.shape[0] if self.matrix.size else 0
        self.axis_off = max(0, int(axis_off))
        self.axis_total = max(int(axis_total) if axis_total is not None else T,
                              self.axis_off + T)
        # A per-frame max feeds the cursor readout: for a distribution the
        # single most telling number is the fastest block right now, not a mean.
        # NaN cells mean "this block is not part of the frame's distribution"
        # (e.g. the tensor explorer's gated appearance-fraction blocks); an
        # all-NaN frame reads 0 rather than leaking NaN into the readout.
        #
        # y spans the whole AXIS, not the matrix. It positions the cursor
        # (paintEvent divides by y.size), so leaving it at matrix length would
        # put the cursor at the wrong frame the moment the two differ. Uncovered
        # frames are NaN rather than 0: nothing computed a value there, and 0 is
        # the examined-and-quiet reading.
        if self.matrix.size:
            y = np.full(self.axis_total, np.nan, np.float32)
            seg = np.zeros(T, np.float32)
            has = np.isfinite(self.matrix).any(1)
            if has.any():
                seg[has] = np.nanmax(self.matrix[has], 1)
            y[self.axis_off:self.axis_off + T] = seg
            self.y = y
        else:
            self.y = np.zeros(0, np.float32)
        self._img = None
        self._range = None
        # set_matrix writes self.y directly rather than going through
        # set_series, so it has to invalidate the base class's memos itself.
        self._bump_series_version()
        self._refresh_auto_collapse()
        self._repaint_unless_collapsed()

    def _raw_data_range(self) -> tuple[float, float]:
        if self._range is not None:
            return self._range
        m = self.matrix
        # nanmin/nanmax scan in place. The old m[np.isfinite(m)] allocated a
        # full-size boolean mask AND a full copy of the values -- ~100 MB of
        # churn per call on a long clip -- before reducing them to two floats.
        # +/-inf is handled below rather than by pre-filtering, which keeps the
        # single pass; NaN-only and all-inf both fall back to the 0..1 default.
        if m.size:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slice
                lo = float(np.nanmin(m))
                hi = float(np.nanmax(m))
        else:
            lo = hi = float("nan")
        if not (np.isfinite(lo) and np.isfinite(hi)):
            # An infinite endpoint means the fast path cannot answer: +/-inf is
            # not a plottable bound, and the original contract is the range of
            # the FINITE cells. Pay for the mask only in this rare case.
            finite = m[np.isfinite(m)] if m.size else m
            if finite.size == 0:
                lo, hi = 0.0, 1.0
            else:
                lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1.0
        self._range = (lo, hi)
        return self._range

    def _hatch_unexamined(self, rgb, w):
        """Hatch the columns lying OUTSIDE the span this matrix covers.

        It has to be a hatch, not a tone. ``_RAMP[0]`` is (12, 12, 12), i.e.
        exactly ``PLOT_BG`` -- so "a bin with zero blocks in it" and "no data has
        covered this column yet" would render as identical dark pixels. That is
        the §10 traps 7/8 failure: nobody looked, wearing the appearance of
        nothing happened. A hatch is not producible by any colormap value, so it
        reads as absence rather than as an empty bin.

        The mask comes from the covered SPAN, not from which columns happened to
        receive a frame. Deriving it from the frames -- ``np.unique(col)`` -- was
        the first cut and is wrong whenever the covered region holds fewer frames
        than it spans pixels: 64 frames over 300 columns leave 236 covered
        columns frameless, and they were painted as unexamined, shredding real
        data into stripes. That is this method's own failure inverted, and it is
        the normal case for any live window shorter than the plot is wide.
        """
        T = self.matrix.shape[0]
        total = max(int(self.axis_total), T)
        if total <= T:
            return rgb                       # matrix spans the axis: nothing to mark
        h = rgb.shape[0]
        # Columns whose axis range intersects the covered frames
        # [axis_off, axis_off + T). Column c spans axis positions
        # [c*total/w, (c+1)*total/w), so the right edge needs a CEILING: flooring
        # it drops the last partial column, leaving a few columns of real data
        # hatched at the frontier -- the same lie as the bug above, just narrower.
        c_lo = (self.axis_off * w) // total
        c_hi = -(-((self.axis_off + T) * w) // total)      # ceil, exclusive
        gap = np.ones(w, bool)
        gap[max(0, c_lo):min(w, c_hi)] = False
        if not gap.any():
            return rgb
        # Broadcast rather than mgrid: two 1-D aranges give the same stripe mask
        # without materializing two full (h, w) index arrays per rebuild.
        stripe = ((np.arange(w)[None, :] + np.arange(h)[:, None]) % 8) < 2
        out = np.where(gap[None, :, None], _UNEXAMINED_BG, rgb)
        return np.where((gap[None, :] & stripe)[..., None], _UNEXAMINED_FG, out)

    def _density_image(self, w: int, h: int, lo: float, hi: float):
        if w <= 0 or h <= 0 or self.matrix.size == 0:
            return None
        key = (w, h, float(lo), float(hi), self._ver, self._log_axis,
               self.axis_off, self.axis_total)
        if self._img is not None and self._img_key == key:
            return self._img
        T, K = self.matrix.shape
        total = max(int(self.axis_total), T)
        # Frames are placed by their AXIS index, so the heatmap lands under the
        # matching column of every plot stacked with it.
        col = np.clip(((np.arange(T) + self.axis_off) * w) // max(1, total),
                      0, w - 1)
        col_idx = np.repeat(col, K).astype(np.int64)
        vals = self.matrix.ravel()
        ok = np.isfinite(vals)
        if not ok.all():                     # NaN cells are masked-out blocks
            vals, col_idx = vals[ok], col_idx[ok]
        tlo, thi = self._fwd(lo), self._fwd(hi)
        frac = (self._fwd(vals) - tlo) / max(1e-9, thi - tlo)
        # Row 0 is the top of the plot, so invert: the fastest blocks sit high.
        row = np.clip(((1.0 - frac) * (h - 1)).astype(np.int64), 0, h - 1)
        counts = np.bincount(row * w + col_idx,
                             minlength=w * h).reshape(h, w).astype(np.float64)
        peak = counts.max()
        norm = np.log1p(counts) / np.log1p(peak) if peak > 0 else counts
        x = np.clip(norm, 0, 1) * (len(self._RAMP) - 1)
        i = np.clip(x.astype(int), 0, len(self._RAMP) - 2)
        f = (x - i)[..., None]
        rgb = (self._RAMP[i] * (1 - f) + self._RAMP[i + 1] * f)
        rgb = np.ascontiguousarray(self._hatch_unexamined(rgb, w)
                                   .astype(np.uint8))
        img = QImage(rgb.data, w, h, 3 * w,
                     QImage.Format.Format_RGB888).copy()
        self._img, self._img_key = img, key
        return img

    def paintEvent(self, _):
        p = QPainter(self)
        if self._paint_collapsed(p):
            return
        p.fillRect(self.rect(), BG)
        r = self._plot_rect()
        p.fillRect(r, PLOT_BG)
        self._paint_header(p)
        p.setFont(QFont("Consolas", 7))
        if self.matrix.size == 0:
            p.setPen(TXT_DIM)
            p.drawText(self._title_x(), 12, self.title)
            p.end()
            return

        lo, hi = self._data_range()
        img = self._density_image(int(r.width()), int(r.height()), lo, hi)
        if img is not None:
            p.drawImage(r.topLeft(), img)

        if self.band_active:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self._paint_band(p, r, lo, hi)

        n = self.y.size
        cx = r.left() + (self.cursor + 0.5) / max(1, n) * r.width()
        p.setPen(QPen(CURSOR, 1))
        p.drawLine(int(cx), int(r.top()), int(cx), int(r.bottom()))

        cur = float(self.y[min(self.cursor, n - 1)]) if n else 0.0
        p.setPen(TXT)
        p.drawText(self._title_x(), 12,
                   self.title + (" [log axis]" if self._log_axis else ""))
        # NaN means the cursor sits past this matrix's span while the per-frame
        # plots have run further ahead. Say so: "max nan" reads as a broken
        # number, and any finite stand-in would be a value nothing computed.
        val_txt = ("unexamined here" if not np.isfinite(cur)
                   else f"max {cur:.4g} {self.unit}".strip())
        p.setPen(TXT_DIM if not np.isfinite(cur) else QColor(self.color))
        p.drawText(int(r.right()) - 7 * len(val_txt) - 2, 12, val_txt)
        p.setPen(TXT_DIM)
        p.drawText(int(r.left()), int(r.bottom()) + 8, f"{lo:.3g}")
        p.end()


class PixelBarPlot(MiniPlot):
    """A bar chart rendered the same pixelated way as :class:`DensityPlot`.

    :class:`MiniPlot` draws its envelope as an antialiased polyline. That reads
    fine for a smooth line, but the per-block speed heatmap right above this
    plot is a raster of hard square cells, and a smooth line under a blocky
    heatmap looks like two different instruments. This plot instead rasterizes
    into a ``QImage`` -- one flat-colored bar per pixel column, no antialiasing
    -- so "# blocks in band" reads as the same pixel-grid instrument as the
    density plot it sits under, just turned into bars instead of a value
    histogram.

    Columns use the same frame -> pixel mapping as
    :meth:`DensityPlot._density_image` (``(frame * w) // T``), so a burst in
    the heatmap and the bar it produces land in the same x column.
    """

    BASE_H = MiniPlot.BASE_H * 2

    def _raw_data_range(self) -> tuple[float, float]:
        if self._range_memo is not None:
            return self._range_memo
        n = self.y.size
        if n == 0:
            return 0.0, 1.0
        hi = float(np.nanmax(self.y))
        if not np.isfinite(hi) or hi <= 0:
            hi = 1.0
        self._range_memo = (0.0, hi)
        return self._range_memo

    def _release_render_cache(self) -> None:
        self._bar_memo = None

    def _bar_image(self, w: int, h: int, hi: float):
        if w <= 0 or h <= 0 or self.y.size == 0:
            return None
        # Cached like DensityPlot's heatmap: the bars are a function of the
        # series and the widget size, but this was rasterising the whole image
        # -- including a per-column Python loop -- on every cursor move.
        key = (w, h, float(hi), self._ver)
        memo = getattr(self, "_bar_memo", None)
        if memo is not None and memo[0] == key:
            return memo[1]
        T = self.y.size
        col = np.clip((np.arange(T) * w) // max(1, T), 0, w - 1)
        heights = np.zeros(w, np.float32)
        np.maximum.at(heights, col, self.y)          # envelope max per column
        frac = np.clip(heights / max(1e-9, hi), 0, 1)
        bar_px = np.round(frac * h).astype(np.int64)

        rgb = np.empty((h, w, 3), np.uint8)
        rgb[:, :] = (PLOT_BG.red(), PLOT_BG.green(), PLOT_BG.blue())
        bar_color = (self.color.red(), self.color.green(), self.color.blue())
        for x in range(w):
            bp = int(bar_px[x])
            if bp > 0:
                rgb[h - bp:h, x] = bar_color
        rgb = np.ascontiguousarray(rgb)
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._bar_memo = (key, img)
        return img

    def paintEvent(self, _):
        p = QPainter(self)
        if self._paint_collapsed(p):
            return
        p.fillRect(self.rect(), BG)
        r = self._plot_rect()
        p.fillRect(r, PLOT_BG)
        self._paint_header(p)
        p.setFont(QFont("Consolas", 7))
        if self.y.size == 0:
            p.setPen(TXT_DIM)
            p.drawText(self._title_x(), 12, self.title)
            p.end()
            return

        lo, hi = self._data_range()
        img = self._bar_image(int(r.width()), int(r.height()), hi)
        if img is not None:
            p.drawImage(r.topLeft(), img)

        n = self.y.size
        cx = r.left() + (self.cursor + 0.5) / n * r.width()
        p.setPen(QPen(CURSOR, 1))
        p.drawLine(int(cx), int(r.top()), int(cx), int(r.bottom()))

        cur = float(self.y[min(self.cursor, n - 1)])
        p.setPen(TXT)
        p.drawText(self._title_x(), 12, self.title)
        val_txt = f"{cur:.4g} {self.unit}".strip()
        p.setPen(QColor(self.color))
        p.drawText(int(r.right()) - 7 * len(val_txt) - 2, 12, val_txt)
        p.setPen(TXT_DIM)
        p.drawText(int(r.left()), int(r.bottom()) + 8, f"{hi:.3g}")
        p.end()


