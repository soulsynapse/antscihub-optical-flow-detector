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
Right panel : a long scrollable column of speed-only time readouts. The
              per-block speed distribution plot carries the detection *band*:
              a draggable min/max pair (in that plot's own Y units). Detection
              is ``band_lo <= value <= band_hi`` -- the same range primitive
              the real pipeline thresholds -- and the "above band" group
              reshapes live as you drag either handle. A handle parked at the
              plot's edge means *unbounded* on that side (readout shows an
              infinity sign): individual block values run far past the
              plotted density curve's own maximum, so an unbounded side must
              not silently reject them.

              The band feeds a *spatial* gate mirroring the intended
              downstream: in-band blocks within a Chebyshev block *gap* are
              single-link clustered into clumps, and a frame is a positive
              detection when the largest clump is non-empty. The gap is a
              horizontal slider; the video highlights the qualifying clump
              bright and the gated-out in-band blocks dim, so you can see
              exactly which blocks the gate keeps.

Current caches process each replicate independently and pack their block grids
into a sparse storage atlas.  This explorer maps those tiles back to their real
source-frame boxes; atlas separators are never displayed or included in plots.
"""
from __future__ import annotations

import csv
import os

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from core.replicates import block_weight_plane

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
CURSOR = QColor(255, 210, 80)
TXT = QColor(210, 210, 210)
TXT_DIM = QColor(140, 140, 140)
BAND = QColor(255, 95, 95)          # detection min/max lines drawn on the plot
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


def _cluster_inband(mask: np.ndarray, weight: np.ndarray, gap: int):
    """Single-link cluster the in-band blocks, bridging gaps up to ``gap``.

    Two in-band blocks join the same clump when their Chebyshev block distance
    is ``<= gap`` (``gap == 1`` reduces exactly to 8-connectivity). This is
    single-link (DBSCAN eps semantics): a chain of blocks each within ``gap`` of
    the next forms one clump even when the endpoints are far apart, so a real
    object split by a couple of noisy sub-threshold blocks still reads as one.

    Returns ``(labels, sizes)`` where ``labels`` is a ``mask``-shaped int array
    (0 = not in band, ``k >= 1`` = clump id) and ``sizes`` maps clump id to its
    summed valid-area ``weight``.  Only the original in-band blocks contribute to
    a clump's size -- the bridged gaps are used to *group*, never to *count*, so
    the size stays a truthful block tally rather than an inflated dilated area.

    Dilation would have been cheaper but wrong on both fronts: it counts the
    phantom bridging blocks, and two isolated blocks merge at Chebyshev ``2*gap``
    (overlapping balls) rather than the intended ``gap``.

    ``gap > 1`` used to pair up every in-band block against every other in
    pure Python (O(K^2), called once per frame) -- fine at startup's default
    K, but with a wide-open band and thousands of in-band blocks per frame
    across a full clip, that stalls or hangs the UI the moment the gap slider
    moves past 1. A KD-tree under Chebyshev distance (``p=inf``) finds the
    same edge set -- pairs within ``gap`` -- in O(K log K), then
    ``connected_components`` unions them in C.
    """
    if not mask.any():
        return np.zeros(mask.shape, np.int32), {}

    if gap <= 1:
        # gap == 1 is plain 8-connectivity: let OpenCV label it in C.
        n_lab, labels, _, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        areas = np.bincount(labels.reshape(-1),
                            weights=weight.reshape(-1), minlength=n_lab)
        sizes = {cid: float(areas[cid]) for cid in range(1, n_lab)}
        return labels.astype(np.int32), sizes

    labels = np.zeros(mask.shape, np.int32)
    coords = np.argwhere(mask)
    k = len(coords)

    if k == 1:
        r, c = coords[0]
        labels[r, c] = 1
        return labels, {1: float(weight[r, c])}

    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=gap, p=np.inf, output_type="ndarray")
    graph = csr_matrix((np.ones(len(pairs), np.uint8), (pairs[:, 0], pairs[:, 1])),
                       shape=(k, k))
    _, comp = connected_components(graph, directed=False)

    sizes: dict[int, float] = {}
    for idx, (r, c) in enumerate(coords):
        cid = int(comp[idx]) + 1
        labels[r, c] = cid
        sizes[cid] = sizes.get(cid, 0.0) + float(weight[r, c])
    return labels, sizes


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
    ``lo <= value <= hi`` -- the same
    band primitive the real pipeline uses (:class:`core.behavior.RangeLeaf`).

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

    BASE_H = 66
    EXPANDED_H = 132
    HANDLE_MARGIN = 20                 # right-edge room for the band pull handles
    GRAB_PX = 6                        # vertical pick tolerance for a band line

    def __init__(self, title: str, unit: str = "", color: QColor = LINE):
        super().__init__()
        self.title = title
        self.unit = unit
        self.color = color
        self.y: np.ndarray = np.zeros(0, np.float32)
        self.cursor = 0
        self.band_active = False
        self.band_lo: float | None = None
        self.band_hi: float | None = None
        self._drag: str | None = None
        self.setMinimumHeight(self.BASE_H)
        self.setMaximumHeight(self.BASE_H)

    def set_series(self, y: np.ndarray) -> None:
        self.y = np.asarray(y, np.float32)
        self.update()

    def set_cursor(self, frame: int) -> None:
        self.cursor = int(frame)
        self.update()

    def set_expanded(self, on: bool) -> None:
        h = self.EXPANDED_H if on else self.BASE_H
        self.setMinimumHeight(h)
        self.setMaximumHeight(h)
        self.update()

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

    def _data_range(self) -> tuple[float, float]:
        n = self.y.size
        if n == 0:
            return 0.0, 1.0
        lo = float(np.nanmin(self.y))
        hi = float(np.nanmax(self.y))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def band(self) -> tuple[float, float]:
        if self.band_lo is None or self.band_hi is None:
            return float("-inf"), float("inf")
        return self.band_lo, self.band_hi

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
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), BG)
        r = self._plot_rect()
        p.fillRect(r, PLOT_BG)

        n = self.y.size
        p.setFont(QFont("Consolas", 7))
        if n == 0:
            p.setPen(TXT_DIM)
            p.drawText(8, 12, self.title)
            p.end()
            return

        lo, hi = self._data_range()

        w = int(r.width())
        cols = max(1, min(n, w))
        edges = np.linspace(0, n, cols + 1).astype(int)
        env = np.empty(cols, np.float32)
        for i in range(cols):
            a, b = edges[i], max(edges[i] + 1, edges[i + 1])
            seg = self.y[a:b]
            env[i] = np.nanmax(seg) if seg.size else lo

        def y_of(val: float) -> float:
            return self._y_of(val, r, lo, hi)

        # Zero baseline, if zero is in range -- makes "how far above nothing" legible.
        if lo < 0 < hi:
            yb = y_of(0.0)
            p.setPen(QPen(QColor(70, 70, 70), 1, Qt.PenStyle.DashLine))
            p.drawLine(int(r.left()), int(yb), int(r.right()), int(yb))

        poly = QPolygonF()
        for i in range(cols):
            x = r.left() + (i + 0.5) / cols * r.width()
            poly.append(QPointF(x, y_of(env[i])))
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
        p.drawText(8, 12, self.title)
        val_txt = f"{cur:.4g} {self.unit}".strip()
        p.setPen(QColor(self.color))
        p.drawText(int(r.right()) - 7 * len(val_txt) - 2, 12, val_txt)

        # min/max axis ticks
        p.setPen(TXT_DIM)
        p.drawText(int(r.left()), int(r.bottom()) + 8, f"{lo:.3g}")
        p.end()

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
        self._img: QImage | None = None
        self._img_key: tuple | None = None
        self._ver = 0
        # Value-axis-only log1p toggle: spreads out the low-speed bulk that a
        # linear axis crushes against the bottom under a handful of fast-block
        # outliers. Never changes what's stored in the matrix or the band --
        # only where a given raw value gets drawn/picked on the Y axis.
        self._log_axis = False

    def set_log_axis(self, on: bool) -> None:
        self._log_axis = bool(on)
        self._img = None
        self.update()

    def _fwd(self, v):
        return np.log1p(v) if self._log_axis else v

    def _inv(self, t):
        return np.expm1(t) if self._log_axis else t

    def set_matrix(self, m: np.ndarray) -> None:
        self.matrix = np.asarray(m, np.float32)          # (T, K)
        # A per-frame max feeds the cursor readout: for a distribution the
        # single most telling number is the fastest block right now, not a mean.
        self.y = (self.matrix.max(1) if self.matrix.size
                  else np.zeros(0, np.float32))
        self._img = None
        self._ver += 1
        self.update()

    def _data_range(self) -> tuple[float, float]:
        m = self.matrix
        if m.size == 0:
            return 0.0, 1.0
        lo = float(np.nanmin(m))
        hi = float(np.nanmax(m))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def _density_image(self, w: int, h: int, lo: float, hi: float):
        if w <= 0 or h <= 0 or self.matrix.size == 0:
            return None
        key = (w, h, float(lo), float(hi), self._ver, self._log_axis)
        if self._img is not None and self._img_key == key:
            return self._img
        T, K = self.matrix.shape
        col = np.clip((np.arange(T) * w) // max(1, T), 0, w - 1)
        col_idx = np.repeat(col, K).astype(np.int64)
        tlo, thi = self._fwd(lo), self._fwd(hi)
        frac = (self._fwd(self.matrix.ravel()) - tlo) / max(1e-9, thi - tlo)
        # Row 0 is the top of the plot, so invert: the fastest blocks sit high.
        row = np.clip(((1.0 - frac) * (h - 1)).astype(np.int64), 0, h - 1)
        counts = np.bincount(row * w + col_idx,
                             minlength=w * h).reshape(h, w).astype(np.float64)
        peak = counts.max()
        norm = np.log1p(counts) / np.log1p(peak) if peak > 0 else counts
        x = np.clip(norm, 0, 1) * (len(self._RAMP) - 1)
        i = np.clip(x.astype(int), 0, len(self._RAMP) - 2)
        f = (x - i)[..., None]
        rgb = (self._RAMP[i] * (1 - f) + self._RAMP[i + 1] * f).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)
        img = QImage(rgb.data, w, h, 3 * w,
                     QImage.Format.Format_RGB888).copy()
        self._img, self._img_key = img, key
        return img

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), BG)
        r = self._plot_rect()
        p.fillRect(r, PLOT_BG)
        p.setFont(QFont("Consolas", 7))
        if self.matrix.size == 0:
            p.setPen(TXT_DIM)
            p.drawText(8, 12, self.title)
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
        p.drawText(8, 12, self.title + (" [log axis]" if self._log_axis else ""))
        val_txt = f"max {cur:.4g} {self.unit}".strip()
        p.setPen(QColor(self.color))
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

    def _data_range(self) -> tuple[float, float]:
        n = self.y.size
        if n == 0:
            return 0.0, 1.0
        hi = float(np.nanmax(self.y))
        if not np.isfinite(hi) or hi <= 0:
            hi = 1.0
        return 0.0, hi

    def _bar_image(self, w: int, h: int, hi: float):
        if w <= 0 or h <= 0 or self.y.size == 0:
            return None
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
        return img

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), BG)
        r = self._plot_rect()
        p.fillRect(r, PLOT_BG)
        p.setFont(QFont("Consolas", 7))
        if self.y.size == 0:
            p.setPen(TXT_DIM)
            p.drawText(8, 12, self.title)
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
        p.drawText(8, 12, self.title)
        val_txt = f"{cur:.4g} {self.unit}".strip()
        p.setPen(QColor(self.color))
        p.drawText(int(r.right()) - 7 * len(val_txt) - 2, 12, val_txt)
        p.setPen(TXT_DIM)
        p.drawText(int(r.left()), int(r.bottom()) + 8, f"{hi:.3g}")
        p.end()


# -- the main window ---------------------------------------------------------

class SpeedExplorer(QWidget):
    """Cache-backed explorer that can run alone or follow the shared AppState.

    The standalone launcher passes ``cache`` and ``video_path``.  An eventual
    main-app tab should use :meth:`from_app_state`; frame decoding and seeking then
    go through AppState, so this widget cannot fight the other tabs' decoder or
    establish a second current-frame value.
    """

    def __init__(self, cache=None, video_path: str | None = None, *, state=None,
                 parent=None):
        super().__init__(parent)
        if state is not None:
            cache = state.cache
        if cache is None:
            raise ValueError("SpeedExplorer requires an open feature cache")
        self.state = state
        self.cache = cache
        self.meta = cache.meta
        self.fps = float(self.meta["fps"])
        self.block = int(self.meta["block_size"])
        self.ny, self.nx = map(int, self.meta["grid"])
        self.regions = _regions_from_meta(self.meta, (self.ny, self.nx))
        self.packed = bool(self.meta.get("replicate_tiles"))
        # Default to one experimental unit.  "All" remains available as an
        # explicitly pooled diagnostic, but behavior detection is per replicate.
        self.active_region_index = -1 if len(self.regions) > 1 else 0
        # Each replicate (and the pooled overview) has its own value scale, so a
        # band set on one scope is meaningless on another's axis. Cache bands per
        # (scope, channel) instead of a single live value: revisiting a scope
        # restores exactly what was left there, and a scope visited for the first
        # time still seeds wide-open.
        self._band_cache: dict[tuple[int, str], tuple[float, float]] = {}
        self.n_blocks = self._active_block_count()
        self.src_w = int(self.meta.get("src_width", 0))
        self.src_h = int(self.meta.get("src_height", 0))

        # Read the per-block arrays once, as float32. This is the whole point:
        # these are the same arrays roi_detection thresholds.
        # In-app construction reuses FeatureContext's already-loaded float32
        # planes. Reading them from the cache again would double several GB of
        # memory on a full clip just because this tab was opened.
        ctx = state.ctx if state is not None else None
        speed_source = ctx.speed if ctx is not None else cache.read("speed")
        self.speed = np.asarray(speed_source, dtype=np.float32)  # (T, ny, nx)
        feats = set(self.meta.get("features", []))
        have_vectors = {"u", "v"}.issubset(feats)
        if have_vectors:
            self.u = np.asarray(ctx.u if ctx is not None else cache.read("u"),
                                dtype=np.float32)
            self.v = np.asarray(ctx.v if ctx is not None else cache.read("v"),
                                dtype=np.float32)
        else:
            self.u = self.v = None
        self.T = self.speed.shape[0]

        self.source = None
        self._owns_source = False
        if state is None and video_path and os.path.exists(video_path):
            from core.video import VideoSource
            try:
                self.source = VideoSource(video_path)
                self._owns_source = True
            except Exception:
                self.source = None
        dimension_source = state.source if state is not None else self.source
        if dimension_source is not None:
            self.src_w = self.src_w or int(dimension_source.info.width)
            self.src_h = self.src_h or int(dimension_source.info.height)
        self.src_w = max(1, self.src_w or int(self.meta.get("work_width", 1)))
        self.src_h = max(1, self.src_h or int(self.meta.get("work_height", 1)))

        # Threshold slider is in px/s; cap the range just past the bulk so the
        # slider's travel lands where the data actually is, not on a lone outlier.
        # This scale is intentionally computed once over the initial scope.  For
        # packed caches that is every owned replicate cell (never atlas padding),
        # which gives the slider one stable absolute px/s mapping.  Focusing a
        # replicate must not silently reinterpret or reset an analysis threshold.
        self.vmax = self._active_speed_scale()
        # Spatial gate mirroring the intended downstream: bridge in-band blocks
        # up to a Chebyshev block gap into clumps. A frame is a positive
        # detection when the largest clump is non-empty -- there is no
        # separate min-size gate/slider; it was more trouble than it was worth.
        self.clump_gap = 1

        self.frame = int(state.current_frame) if state is not None else 0
        self.playing = False
        self._overlay_peek_hidden = False

        self._build_ui()
        # Key events go to whichever child currently has focus. Filtering at the
        # application level keeps Shift a press-and-hold raw-video peek after the
        # user touches the video, slider, or overlay menu; other tabs are ignored.
        self._event_filter_app = QApplication.instance()
        if self._event_filter_app is not None:
            self._event_filter_app.installEventFilter(self)
        # Standalone mode has no MainWindow to own the application Space shortcut.
        # Embedded mode deliberately leaves this to MainWindow, avoiding two
        # competing shortcuts for the same key.
        self._space_shortcut = None
        if state is None:
            self._space_shortcut = QShortcut(
                QKeySequence(Qt.Key.Key_Space.value), self)
            self._space_shortcut.setContext(
                Qt.ShortcutContext.ApplicationShortcut)
            self._space_shortcut.setAutoRepeat(False)
            self._space_shortcut.activated.connect(self.toggle_playback)
        if state is not None:
            state.frame_changed.connect(self._on_state_frame_changed)
        self._compute_static_series()
        # Seed each channel's band from its now-populated series, and expand the
        # selected plot. Must follow series computation so the band is absolute.
        self._apply_selected_plot_ui()
        self._recompute_threshold_series()
        self._update_frame(self.frame)

    @classmethod
    def from_app_state(cls, state, parent=None) -> "SpeedExplorer":
        """Build an embeddable instance over the app's current video/cache."""
        if not state.has_cache:
            raise ValueError("Open a feature cache before creating Speed Explorer")
        return cls(state=state, parent=parent)

    # -- layout --------------------------------------------------------------

    def _build_ui(self):
        self._sync_window_title()
        self.resize(1500, 900)
        root = QHBoxLayout(self)

        # ---- left: video + controls ----
        left = QVBoxLayout()
        self.video_view = FrameView()
        self.video_view.setMinimumSize(720, 480)
        self.video_view.clicked.connect(self._on_video_clicked)
        self.video_view.back_requested.connect(self._clear_region_focus)
        self._sync_video_boxes()
        left.addWidget(self.video_view, 1)

        # transport
        trow = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        trow.addWidget(self.play_btn)
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.setRange(0, self.T - 1)
        self.scrub.valueChanged.connect(self._on_scrub)
        trow.addWidget(self.scrub, 1)
        self.time_lbl = QLabel("0.00 s")
        self.time_lbl.setMinimumWidth(90)
        trow.addWidget(self.time_lbl)
        left.addLayout(trow)

        # Replicate scope. Atlas-wide pooling is useful for broad diagnostics but
        # is deliberately labelled: the app's actual detection unit is one box.
        rrow = QHBoxLayout()
        rrow.addWidget(QLabel("Replicate:"))
        self.region_combo = QComboBox()
        if len(self.regions) > 1:
            self.region_combo.addItem(
                f"All {len(self.regions)} replicates (pooled diagnostic)", -1)
        for i, region in enumerate(self.regions):
            rid = region["id"]
            suffix = f" (#{rid})" if rid is not None else ""
            self.region_combo.addItem(f"{region['label']}{suffix}", i)
        self.region_combo.setCurrentIndex(0)
        self.region_combo.currentIndexChanged.connect(self._on_region_changed)
        rrow.addWidget(self.region_combo, 1)
        rrow.addWidget(QLabel("Focused view:"))
        self.focus_mode = QComboBox()
        self.focus_mode.addItem("Source detail", "source")
        self.focus_mode.addItem("Flow working resolution", "flow")
        self.focus_mode.setToolTip(
            "Source detail crops the original frame before scaling. Flow working "
            "resolution replays downsampling, grayscale conversion and contrast "
            "normalization at the exact cached flow-input dimensions. Stateful "
            "registration, temporal denoising/background and masks are omitted. "
            "Choose Raw frame below to inspect it without a flow overlay.")
        self.focus_mode.currentIndexChanged.connect(
            lambda: self._redraw_video())
        rrow.addWidget(self.focus_mode)
        left.addLayout(rrow)

        # overlay mode
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Overlay:"))
        self.overlay_mode = QComboBox()
        modes = ["Raw frame", "Speed heatmap"]
        if self.u is not None:
            modes += ["Flow direction (HSV)", "Flow vectors"]
        self.overlay_mode.addItems(modes)
        self.overlay_mode.setCurrentText("Speed heatmap")
        self.overlay_mode.currentTextChanged.connect(lambda _: self._redraw_video())
        orow.addWidget(self.overlay_mode, 1)
        self.hi_chk = QCheckBox("Highlight detected clumps (band, gap-bridged)")
        self.hi_chk.setChecked(True)
        self.hi_chk.stateChanged.connect(lambda _: self._redraw_video())
        orow.addWidget(self.hi_chk)
        left.addLayout(orow)

        # Detection thresholds no longer live on horizontal sliders: they are the
        # min/max handles of the band on the per-block speed plot.
        band_hint = QLabel(
            "Detection thresholds: drag the min/max handles on the per-block "
            "speed plot at right.")
        band_hint.setStyleSheet("color:#888;")
        band_hint.setWordWrap(True)
        left.addWidget(band_hint)

        info = QLabel(
            f"cache: {self.meta.get('backend', '?')} | fps {self.fps:.2f} | "
            f"block {self.block}px | downsample {self.meta.get('downsample', '?')} | "
            f"{len(self.regions)} {'replicate tiles' if self.packed else 'region'}")
        info.setStyleSheet("color:#888;")
        left.addWidget(info)

        root.addLayout(left, 1)

        # ---- right: scrollable readouts. The gap for future toggles lives
        # *inside* the scrolled grid (its own column), not beside the
        # QScrollArea -- otherwise the scrollbar would float away from the
        # program's right edge instead of hugging it.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        self.plot_col = QGridLayout(holder)
        self.plot_col.setSpacing(3)
        self.plot_col.setColumnMinimumWidth(0, 20)
        self.plot_col.setColumnStretch(1, 1)
        self.plot_col.setColumnMinimumWidth(2, 160)
        self.plot_col.setColumnStretch(2, 0)
        scroll.setWidget(holder)
        scroll.setMinimumWidth(500)
        root.addWidget(scroll, 1)

        self.plots: dict[str, MiniPlot] = {}
        self._plot_row = 0

        def section(text: str):
            lbl = QLabel(text)
            lbl.setStyleSheet("color:#7fd7ff; font-weight:bold; padding-top:6px;")
            self.plot_col.addWidget(lbl, self._plot_row, 0, 1, 2)
            self._plot_row += 1
            return lbl

        def add(key: str, title: str, unit: str, color=LINE, cls=MiniPlot):
            pl = cls(title, unit, color)
            pl.seek_requested.connect(self._seek)
            self.plots[key] = pl
            self.plot_col.addWidget(pl, self._plot_row, 1)
            self._plot_row += 1
            return pl

        section("Raw selected-block speed distribution (per frame)")
        dist = add("dist", "Per-block speed (density)", "px/s", cls=DensityPlot)
        dist.band_changed.connect(self._on_band_changed)
        dist.band_committed.connect(self._on_band_committed)
        add("count", "# blocks in band", "blk", QColor(110, 230, 120),
            cls=PixelBarPlot)
        add("sstd", "Spatial std of speed", "px/s")
        add("peak", "Max - median (peakedness)", "px/s")

        section("Detection sweep (drag band handles on per-block speed plot)")
        add("frac", "Fraction of blocks in band", "", QColor(110, 230, 120))
        add("clump", "Largest clump in band (gap-bridged)", "blk",
            QColor(110, 230, 120))
        add("detect", "Positive detection (non-empty clump)", "0/1",
            QColor(120, 255, 140))
        add("cond_mean", "Mean speed OF blocks in band", "px/s",
            QColor(110, 230, 120))
        add("energy", "Total speed summed over blocks in band", "px/s",
            QColor(110, 230, 120))
        self.plot_col.setRowStretch(self._plot_row, 1)

        # Reserved side strip (column 2): a home for toggles that apply to the
        # whole readout column rather than one plot. Placed after every row is
        # added so its -1 row-span actually covers them all.
        side_widget = QWidget()
        self.side_panel = QVBoxLayout(side_widget)
        self.side_panel.setContentsMargins(6, 6, 6, 6)
        self.log_chk = QCheckBox("Log-transform per-block speed")
        self.log_chk.setToolTip(
            "Draw the per-block speed distribution's value axis as log1p(speed) "
            "instead of linear px/s, so the low-speed bulk isn't crushed flat by "
            "a few fast-block outliers. Axis-only: band thresholds are still set "
            "and compared in raw px/s regardless of this toggle.")
        self.log_chk.stateChanged.connect(self._on_log_toggle)
        self.log_chk.setChecked(True)
        self.side_panel.addWidget(self.log_chk)
        self.save_btn = QPushButton("Save plots (CSV)")
        self.save_btn.setToolTip(
            "Export every readout plot's per-frame series to one CSV -- a "
            "'frame' column plus one column per plot, in the same order they "
            "appear in the scroll column above.")
        self.save_btn.clicked.connect(self._export_plots_csv)
        self.side_panel.addWidget(self.save_btn)
        self.side_panel.addStretch(1)
        self.plot_col.addWidget(side_widget, 0, 2, -1, 1,
                                 Qt.AlignmentFlag.AlignTop)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        # Debounce for the cheap band series so dragging a handle stays smooth on
        # a full-clip cache (the sum over ~100M elements is ~100 ms).
        self._thr_debounce = QTimer(self)
        self._thr_debounce.setSingleShot(True)
        self._thr_debounce.setInterval(120)
        self._thr_debounce.timeout.connect(self._recompute_threshold_series)

    # -- static (threshold-independent) series -------------------------------

    def _active_regions(self) -> list[dict]:
        if self.active_region_index < 0:
            return self.regions
        return [self.regions[self.active_region_index]]

    def _active_block_count(self) -> int:
        return sum((r["atlas_bbox"][2] - r["atlas_bbox"][0]) *
                   (r["atlas_bbox"][3] - r["atlas_bbox"][1])
                   for r in self._active_regions())

    def _active_block_values(self, arr: np.ndarray) -> np.ndarray:
        """(T, K) values from owned cells only; never atlas separators/padding."""
        parts = []
        for region in self._active_regions():
            y0, x0, y1, x1 = region["atlas_bbox"]
            parts.append(arr[:, y0:y1, x0:x1].reshape(arr.shape[0], -1))
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

    def _active_speed_scale(self) -> float:
        values = self._active_block_values(self.speed)
        finite = values[np.isfinite(values)]
        vmax = float(np.percentile(finite, 99.9)) if finite.size else 1.0
        return max(vmax, 1.0)

    def _scope_text(self) -> str:
        if self.active_region_index < 0:
            return f"all {len(self.regions)} replicates pooled"
        region = self.regions[self.active_region_index]
        rid = region["id"]
        return region["label"] + (f" (#{rid})" if rid is not None else "")

    def _sync_window_title(self) -> None:
        self.setWindowTitle(
            f"Speed explorer -- {os.path.basename(self.meta.get('video_path', '?'))} "
            f"-- {self._scope_text()} ({self.T} frames, {self.n_blocks} blocks)")

    def _compute_static_series(self):
        s = self._active_block_values(self.speed)
        # The per-frame channel shows the full per-block distribution as a
        # density heatmap, not a spatial mean -- the mean buries the sparse fast
        # blocks that are the actual signal.
        self.plots["dist"].set_matrix(s)
        self.plots["sstd"].set_series(s.std(1))
        self.plots["peak"].set_series(s.max(1) - np.median(s, 1))

    def _on_log_toggle(self, _state=None):
        # Axis-only: repaints how raw values map to pixel rows. Never touches
        # the matrix data or the band -- a threshold means the same speed
        # whether the checkbox is on or off.
        self.plots["dist"].set_log_axis(self.log_chk.isChecked())

    def _export_plots_csv(self):
        """Dump every readout plot's per-frame series to one CSV.

        Each plot (however it's computed -- density max, spatial std, a
        threshold sweep) reduces to one (T,) series in ``plot.y``, so a
        single frame-indexed table covers all of them regardless of type.
        """
        stem = os.path.splitext(os.path.basename(
            self.meta.get("video_path", "speed")))[0]
        default = f"{stem}_speed_plots.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export plots to CSV", default, "CSV (*.csv)")
        if not path:
            return
        keys = list(self.plots.keys())
        cols = [self.plots[k].y for k in keys]
        n = max((c.size for c in cols), default=0)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame"] + keys)
            for i in range(n):
                w.writerow([i] + [float(c[i]) if i < c.size else ""
                                   for c in cols])

    # -- threshold-dependent series ------------------------------------------

    def _band(self) -> tuple[float, float]:
        """Current (lo, hi) detection band, in per-block speed's Y units."""
        return self.plots["dist"].band()

    def _apply_selected_plot_ui(self):
        """The per-block speed plot is the sole detection channel: always on."""
        self.plots["dist"].set_band_active(True)
        self.plots["dist"].set_expanded(True)

    def _recompute_threshold_series(self):
        lo, hi = self._band()
        values = self._active_block_values(self.speed)
        mask = (values >= lo) & (values <= hi)
        count = mask.sum(1).astype(np.float32)
        energy = np.where(mask, values, 0.0).sum(1, dtype=np.float64)
        self.plots["count"].set_series(count)
        self.plots["frac"].set_series(count / max(1, self.n_blocks))

        self.plots["energy"].set_series(energy.astype(np.float32))
        with np.errstate(invalid="ignore", divide="ignore"):
            cond = np.where(count > 0, energy / np.maximum(count, 1), 0.0)
        self.plots["cond_mean"].set_series(cond.astype(np.float32))
        # Clump is the expensive one; keep whatever it last held during a drag and
        # refresh it on band release.
        if "clump" not in self.plots or self.plots["clump"].y.size == 0:
            self._recompute_clump()

    def _recompute_clump(self):
        """Largest gap-bridged in-band clump per frame (weighted block size).

        Blocks join one clump when their Chebyshev block distance is within
        ``clump_gap`` (``gap == 1`` is plain 8-connectivity); only the in-band
        blocks' valid-area weights are summed, never the bridged gaps, so the
        size stays the truthful block tally the downstream min-clump gate reads.
        This is the most direct readout of 'is there a real moving CLUMP here,
        or just a few scattered noisy blocks that happen to fall in the band'.
        """
        lo, hi = self._band()
        largest = np.zeros(self.T, np.float32)
        # Valid-area weights so a one-pixel-tall edge sliver is not counted as a
        # full block -- mirrors the same discount in roi_detection.
        weight_plane = block_weight_plane(self.meta)
        # Clumps may never bridge two packed replicate tiles. This mirrors
        # roi_detection, which applies its spatial gate inside one replicate.
        for region in self._active_regions():
            y0, x0, y1, x1 = region["atlas_bbox"]
            w = weight_plane[y0:y1, x0:x1]
            values = self.speed[:, y0:y1, x0:x1]
            for t, field in enumerate(values):
                m = (field >= lo) & (field <= hi)
                if not m.any():
                    continue
                _, sizes = _cluster_inband(m, w, self.clump_gap)
                if sizes:
                    largest[t] = max(largest[t], max(sizes.values()))
        self.plots["clump"].set_series(largest)
        self._recompute_detect()

    def _recompute_detect(self):
        """Positive-detection channel: is the largest clump non-empty?

        Cheap: it only re-compares the already-computed largest-clump series, so
        this never re-clusters -- only the gap slider or the band does.
        """
        detected = (self.plots["clump"].y > 0).astype(np.float32)
        self.plots["detect"].set_series(detected)

    # -- control handlers ----------------------------------------------------

    def _on_band_changed(self):
        """A band handle is being dragged: cheap series debounced, overlay live."""
        self._thr_debounce.start()
        self._redraw_video()          # highlight overlay follows immediately

    def _on_band_committed(self):
        """Handle released: flush the cheap series and refresh the clump."""
        self._thr_debounce.stop()
        self._recompute_threshold_series()
        self._recompute_clump()
        self._redraw_video()

    def _on_region_changed(self, _index: int):
        old_index = self.active_region_index
        # Stash the outgoing scope's bands so re-visiting it later (rep 25 ->
        # rep 26 -> rep 25) restores exactly what was left there, rather than
        # losing it to a blanket reseed.
        for key in ("dist",):
            pl = self.plots[key]
            if pl.band_lo is not None and pl.band_hi is not None:
                self._band_cache[old_index, key] = (pl.band_lo, pl.band_hi)

        data = self.region_combo.currentData()
        self.active_region_index = int(data) if data is not None else 0
        focus = None if self.active_region_index < 0 else \
            self.regions[self.active_region_index]["frac"]
        self.video_view.set_focus_frac(focus)
        self._sync_video_boxes()
        self.n_blocks = self._active_block_count()
        self._compute_static_series()
        # Each replicate lives on its own value scale, so a band frozen from a
        # *different* scope would sit off this replicate's plot entirely -- but
        # a band this exact scope held before is still meaningful. Restore it
        # from the cache if we have one; otherwise the plot's band stays
        # unseeded (None) and lazily seeds wide-open when it becomes active.
        for key in ("dist",):
            pl = self.plots[key]
            cached = self._band_cache.get((self.active_region_index, key))
            pl.band_lo, pl.band_hi = cached if cached is not None else (None, None)
        self._apply_selected_plot_ui()
        self._recompute_threshold_series()
        self._recompute_clump()
        self._sync_window_title()
        self._redraw_video()

    def _on_video_clicked(self, point):
        width, height = self.video_view._src_size
        fx = point.x() / max(1, width)
        fy = point.y() / max(1, height)
        for i, region in enumerate(self.regions):
            x0, y0, x1, y1 = region["frac"]
            if x0 <= fx <= x1 and y0 <= fy <= y1:
                combo_index = self.region_combo.findData(i)
                if combo_index >= 0:
                    self.region_combo.setCurrentIndex(combo_index)
                return

    def _clear_region_focus(self):
        pooled_index = self.region_combo.findData(-1)
        if pooled_index >= 0:
            self.region_combo.setCurrentIndex(pooled_index)
        else:
            self.video_view.set_focus_frac(None)

    def _sync_video_boxes(self):
        if not self.packed:
            self.video_view.set_boxes([])
            return
        self.video_view.set_boxes([
            (*region["frac"], region["label"], "#50dcff",
             i == self.active_region_index)
            for i, region in enumerate(self.regions)
        ])

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start(int(1000 / max(1.0, self.fps)))
        else:
            self.timer.stop()

    def toggle_playback(self):
        """Public playback hook shared with VideoPanel/MainWindow dispatch."""
        self._toggle_play()

    def eventFilter(self, watched, event):
        event_type = event.type()
        if event_type in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and \
                event.key() == Qt.Key.Key_Shift and not event.isAutoRepeat():
            focus = QApplication.focusWidget()
            focus_is_ours = focus is self or (
                focus is not None and self.isAncestorOf(focus))
            if event_type == QEvent.Type.KeyPress and focus_is_ours and \
                    not self._overlay_peek_hidden:
                self._overlay_peek_hidden = True
                self.video_view.set_overlays_hidden(True)
                self._redraw_video()
            elif event_type == QEvent.Type.KeyRelease and \
                    self._overlay_peek_hidden:
                self._overlay_peek_hidden = False
                self.video_view.set_overlays_hidden(False)
                self._redraw_video()
        elif event_type == QEvent.Type.ApplicationDeactivate and \
                self._overlay_peek_hidden:
            # A Shift release delivered to another application must not leave the
            # explorer permanently stuck in its temporary raw-video state.
            self._overlay_peek_hidden = False
            self.video_view.set_overlays_hidden(False)
            self._redraw_video()
        return super().eventFilter(watched, event)

    def _tick(self):
        nxt = self.frame + 1
        if nxt >= self.T:
            nxt = 0
        self._update_frame(nxt)

    def _on_scrub(self, v: int):
        if v != self.frame:
            self._update_frame(v)

    def _seek(self, frame: int):
        self._update_frame(frame)

    def _update_frame(self, frame: int):
        frame = max(0, min(int(frame), self.T - 1))
        if self.state is not None and self.state.current_frame != frame:
            self.state.set_frame(frame)
            return
        self._apply_frame(frame)

    def _on_state_frame_changed(self, frame: int):
        self._apply_frame(frame)

    def _apply_frame(self, frame: int):
        self.frame = max(0, min(int(frame), self.T - 1))
        if self.scrub.value() != self.frame:
            self.scrub.blockSignals(True)
            self.scrub.setValue(self.frame)
            self.scrub.blockSignals(False)
        self.time_lbl.setText(f"{self.frame / self.fps:.2f} s  (#{self.frame})")
        for pl in self.plots.values():
            pl.set_cursor(self.frame)
        if self.isVisible():
            self._redraw_video()

    # -- video + overlay -----------------------------------------------------

    def _base_frame(self) -> np.ndarray:
        """Source-detail or flow-resolution pixels for the current view."""
        focus = None if self.active_region_index < 0 else \
            self.regions[self.active_region_index]["frac"]
        self._render_frac = focus or (0.0, 0.0, 1.0, 1.0)

        if self.state is not None:
            bgr = self.state.display_frame(self.frame, focus_frac=focus)
        elif self.source is not None:
            bgr = self.source.frame_at(self.frame)
            if bgr is not None and focus is not None:
                h, w = bgr.shape[:2]
                x0, y0, x1, y1 = focus
                sx0 = max(0, min(w - 1, int(round(x0 * w))))
                sy0 = max(0, min(h - 1, int(round(y0 * h))))
                sx1 = max(sx0 + 1, min(w, int(round(x1 * w))))
                sy1 = max(sy0 + 1, min(h, int(round(y1 * h))))
                bgr = np.ascontiguousarray(bgr[sy0:sy1, sx0:sx1])
        else:
            bgr = None

        view_x0, view_y0, view_x1, view_y1 = self._render_frac
        source_view_w = max(1, int(round(self.src_w * (view_x1 - view_x0))))
        source_view_h = max(1, int(round(self.src_h * (view_y1 - view_y0))))
        if bgr is None:
            if focus is not None and self.focus_mode.currentData() == "flow":
                region = self.regions[self.active_region_index]
                work_w = int(region.get(
                    "work_width", region["grid"][1] * self.block))
                work_h = int(region.get(
                    "work_height", region["grid"][0] * self.block))
                return np.zeros((max(1, work_h), max(1, work_w), 3), np.uint8)
            display_w = min(source_view_w, DISPLAY_MAX_W)
            display_h = max(1, int(round(source_view_h * display_w /
                                           source_view_w)))
            return np.zeros((display_h, display_w, 3), np.uint8)

        if focus is not None and self.focus_mode.currentData() == "flow":
            from core.config import PipelineConfig
            from core.preprocess import flow_input_preview
            region = self.regions[self.active_region_index]
            work_w = int(region.get("work_width",
                                    region["grid"][1] * self.block))
            work_h = int(region.get("work_height",
                                    region["grid"][0] * self.block))
            cfg = PipelineConfig.from_dict(
                self.meta.get("config", {})).preprocess
            return flow_input_preview(bgr, (work_w, work_h), cfg)

        h, w = bgr.shape[:2]
        if w > DISPLAY_MAX_W:
            scale = DISPLAY_MAX_W / w
            bgr = cv2.resize(bgr, (DISPLAY_MAX_W, max(1, int(round(h * scale)))),
                             interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(bgr)

    def _display_bbox(self, region: dict, width: int, height: int
                      ) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = region["frac"]
        vx0, vy0, vx1, vy1 = self._render_frac
        x0 = (x0 - vx0) / (vx1 - vx0)
        x1 = (x1 - vx0) / (vx1 - vx0)
        y0 = (y0 - vy0) / (vy1 - vy0)
        y1 = (y1 - vy0) / (vy1 - vy0)
        dx0 = max(0, min(width - 1, int(round(x0 * width))))
        dy0 = max(0, min(height - 1, int(round(y0 * height))))
        dx1 = max(dx0 + 1, min(width, int(round(x1 * width))))
        dy1 = max(dy0 + 1, min(height, int(round(y1 * height))))
        return dx0, dy0, dx1, dy1

    def _redraw_video(self):
        base = self._base_frame()
        ch, cw = base.shape[:2]
        sp = self.speed[self.frame]                       # (ny, nx)
        mode = "Raw frame" if self._overlay_peek_hidden else \
            self.overlay_mode.currentText()

        out = base.copy()
        active_indices = (range(len(self.regions)) if self.active_region_index < 0
                          else [self.active_region_index])
        active_indices = list(active_indices)
        for region_index in active_indices:
            region = self.regions[region_index]
            y0, x0, y1, x1 = region["atlas_bbox"]
            sub_speed = sp[y0:y1, x0:x1]
            dx0, dy0, dx1, dy1 = self._display_bbox(region, cw, ch)
            rw, rh = dx1 - dx0, dy1 - dy0
            roi = out[dy0:dy1, dx0:dx1]

            if mode == "Speed heatmap":
                norm = np.clip(sub_speed / self.vmax, 0, 1)
                heat = cv2.applyColorMap(
                    (norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                heat = cv2.resize(heat, (rw, rh), interpolation=cv2.INTER_NEAREST)
                blended = cv2.addWeighted(roi, 0.45, heat, 0.55, 0)
                np.copyto(roi, blended)
            elif mode == "Flow direction (HSV)" and self.u is not None:
                u = self.u[self.frame, y0:y1, x0:x1]
                v = self.v[self.frame, y0:y1, x0:x1]
                ang = (np.degrees(np.arctan2(v, u)) % 360) / 2.0
                mag = np.clip(sub_speed / self.vmax, 0, 1)
                hsv = np.zeros((*sub_speed.shape, 3), np.uint8)
                hsv[..., 0] = ang.astype(np.uint8)
                hsv[..., 1] = 255
                hsv[..., 2] = (mag * 255).astype(np.uint8)
                direction = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                direction = cv2.resize(
                    direction, (rw, rh), interpolation=cv2.INTER_NEAREST)
                blended = cv2.addWeighted(roi, 0.4, direction, 0.6, 0)
                np.copyto(roi, blended)
            elif mode == "Flow vectors" and self.u is not None:
                u = self.u[self.frame, y0:y1, x0:x1]
                v = self.v[self.frame, y0:y1, x0:x1]
                gy, gx = sub_speed.shape
                cell_w, cell_h = rw / gx, rh / gy
                vector_scale = min(cell_w, cell_h) / max(self.vmax, 1e-6) * 1.5
                for by in range(0, gy, 2):
                    for bx in range(0, gx, 2):
                        ax0 = int(dx0 + (bx + 0.5) * cell_w)
                        ay0 = int(dy0 + (by + 0.5) * cell_h)
                        ax1 = int(ax0 + u[by, bx] * vector_scale)
                        ay1 = int(ay0 + v[by, bx] * vector_scale)
                        cv2.arrowedLine(out, (ax0, ay0), (ax1, ay1),
                                        (80, 220, 255), 1, tipLength=0.35)

        if self.hi_chk.isChecked() and not self._overlay_peek_hidden:
            lo, hi = self._band()
            # Bright green marks blocks in a non-empty gap-bridged clump --
            # there is no min-size gate to drop any of them dim anymore.
            weight_plane = block_weight_plane(self.meta)
            for region_index in active_indices:
                region = self.regions[region_index]
                y0, x0, y1, x1 = region["atlas_bbox"]
                dx0, dy0, dx1, dy1 = self._display_bbox(region, cw, ch)
                field = self.speed[self.frame, y0:y1, x0:x1]
                inband = (field >= lo) & (field <= hi)
                labels, sizes = _cluster_inband(
                    inband, weight_plane[y0:y1, x0:x1], self.clump_gap)
                qual = [cid for cid, s in sizes.items() if s > 0]
                hot = (np.isin(labels, qual) if qual
                       else np.zeros_like(labels, bool)).astype(np.uint8)
                roi = out[dy0:dy1, dx0:dx1]
                green = np.zeros_like(roi)
                green[..., 1] = 255
                mm = cv2.resize(hot, (dx1 - dx0, dy1 - dy0),
                                interpolation=cv2.INTER_NEAREST)
                blended = cv2.addWeighted(roi, 0.5, green, 0.5, 0)
                np.copyto(roi, blended, where=(mm > 0)[:, :, None])
                contours, _ = cv2.findContours(
                    mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(roi, contours, -1, (60, 255, 60), 1)

        self.video_view.set_frame(
            out, image_frac=self._render_frac,
            coordinate_size=(self.src_w, self.src_h))

    def showEvent(self, event):
        super().showEvent(event)
        self._redraw_video()

    def resizeEvent(self, _):
        self.video_view.update()

    def closeEvent(self, event):
        if self._event_filter_app is not None:
            self._event_filter_app.removeEventFilter(self)
            self._event_filter_app = None
        if self._owns_source and self.source is not None:
            self.source.release()
            self.source = None
        super().closeEvent(event)
