"""Replicates tab: manual experimental-unit bounding boxes.

The histogram-driven automatic ROI discovery is shelved in
gui/_shelved_tab2_roi_auto.py -- it worked, but on footage where the behavior is
everywhere and the camera moves, drawing the replicate regions by hand is faster
and less error-prone than tuning a filter to segment them. Each box becomes a
replicate: a spatial region present for the whole clip, which Tab 3 then
classifies exactly as it would an auto-discovered ROI.

Draw a box with the mouse (in Draw mode). Click a box to select it; rename or
delete from the list. Boxes save to and load from JSON so a plate layout can be
reused across clips.
"""
from __future__ import annotations

import json
import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QGroupBox, QHBoxLayout, QInputDialog, QLabel,
                             QListWidget, QListWidgetItem, QMessageBox,
                             QPushButton, QVBoxLayout, QWidget)

from core.roi import packed_rect_roi, rect_roi, roi_psd
from gui.inspector import PSDPlot, TimeSeriesPlot
from gui.state import AppState
from gui.video_panel import VideoPanel

# A fixed palette so successive replicates are visually distinct.
_PALETTE = ["#ff5a5a", "#4ac6ff", "#ffd24a", "#6ee06e", "#c78bff", "#ff9d3a",
            "#00d0b0", "#ff6ec7"]


class Tab2Replicates(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        # Each replicate: {"id", "label", "frac": (x0,y0,x1,y1), "color"}.
        self.replicates: list[dict] = []
        self._next_id = 1

        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)

        # -- left: video (half the window) -----------------------------------
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(2, 2, 2, 2)

        self.video = VideoPanel(state)
        self.video.view.box_drawn.connect(self._on_box_drawn)
        self.video.view.stamp_at.connect(self._on_stamp_at)
        self.video.view.clicked.connect(self._on_view_clicked)
        ll.addWidget(self.video, 1)

        # Stamp size in frame fractions, set from the last box you draw. When
        # stamp mode is on, clicking drops an identically-sized box -- which is
        # what keeps replicates the same size, so a fixed "min blocks" and the
        # max-aggregation mean the same thing in every box.
        self.stamp_size: tuple[float, float] | None = None

        bar = QHBoxLayout()
        self.draw_btn = QPushButton("✏ Draw boxes: OFF")
        self.draw_btn.setCheckable(True)
        self.draw_btn.setStyleSheet("padding:6px; font-weight:bold;")
        self.draw_btn.toggled.connect(self._toggle_draw)
        bar.addWidget(self.draw_btn, 1)
        self.stamp_chk = QCheckBox("Fixed-size stamp")
        self.stamp_chk.setToolTip(
            "Draw one box to set the size, then click to drop identically-sized "
            "boxes. Same-size replicates make block counts comparable.")
        self.stamp_chk.setChecked(True)
        bar.addWidget(self.stamp_chk)
        self.draw_btn.setChecked(True)
        ll.addLayout(bar)

        self.hint = QLabel(
            "Drag the first replicate box to set the stamp size (e.g. one per "
            "tube), then CLICK to place more boxes of the same size. Drag again "
            "at any time to change the stamp size.")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#333; font-size:11px;")
        ll.addWidget(self.hint)

        lay.addWidget(left, 1)   # stretch 1 of 2 == half the window

        # -- right: list + inspector -----------------------------------------
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)

        box = QGroupBox("Replicate boxes")
        bl = QVBoxLayout(box)
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_select)
        bl.addWidget(self.list, 1)

        row = QHBoxLayout()
        for text, fn in (("Rename", self._rename), ("Delete", self._delete),
                         ("Clear all", self._clear)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        bl.addLayout(row)

        row2 = QHBoxLayout()
        save = QPushButton("Save boxes…")
        save.clicked.connect(self._save)
        load = QPushButton("Load boxes…")
        load.clicked.connect(self._load)
        row2.addWidget(save)
        row2.addWidget(load)
        bl.addLayout(row2)

        to3 = QPushButton("Classify these in Tab 3 →")
        to3.setStyleSheet("padding:6px; font-weight:bold;")
        to3.clicked.connect(self._go_classify)
        bl.addWidget(to3)
        rl.addWidget(box)

        std_box = QGroupBox("Selected replicate standardization")
        sf = QFormLayout(std_box)
        self.use_baseline = QCheckBox("Use an explicitly quiescent interval")
        self.use_baseline.setToolTip(
            "The 99th percentile of speed inside this box during the interval "
            "becomes its noise reference. Pick a period with no real behavior.")
        sf.addRow(self.use_baseline)
        self.baseline_start = QDoubleSpinBox()
        self.baseline_start.setRange(0.0, 1_000_000.0)
        self.baseline_start.setDecimals(3)
        self.baseline_start.setSuffix(" s")
        sf.addRow("baseline start", self.baseline_start)
        self.baseline_end = QDoubleSpinBox()
        self.baseline_end.setRange(0.0, 1_000_000.0)
        self.baseline_end.setDecimals(3)
        self.baseline_end.setSuffix(" s")
        sf.addRow("baseline end", self.baseline_end)

        self.pixels_per_mm = QDoubleSpinBox()
        self.pixels_per_mm.setRange(0.0, 1_000_000.0)
        self.pixels_per_mm.setDecimals(4)
        self.pixels_per_mm.setSpecialValueText("unset")
        self.pixels_per_mm.setToolTip(
            "Calibration in SOURCE-video pixels per millimetre. Cache "
            "downsampling is accounted for automatically; raw cached values are "
            "never converted in place.")
        sf.addRow("source pixels / mm", self.pixels_per_mm)
        self.body_length_mm = QDoubleSpinBox()
        self.body_length_mm.setRange(0.0, 100_000.0)
        self.body_length_mm.setDecimals(4)
        self.body_length_mm.setSpecialValueText("unset")
        sf.addRow("body length (mm)", self.body_length_mm)
        for w in (self.use_baseline, self.baseline_start, self.baseline_end,
                  self.pixels_per_mm, self.body_length_mm):
            # Rebuilding every ROI trace on each intermediate spinbox keystroke
            # is ruinous on a full clip; commit numeric metadata when editing
            # finishes. The checkbox is already a discrete action.
            signal = w.toggled if isinstance(w, QCheckBox) else w.editingFinished
            signal.connect(self._standardization_changed)
        rl.addWidget(std_box)
        self._sync_standardization_controls(None)

        insp = QGroupBox("Replicate inspector")
        il = QVBoxLayout(insp)
        self.ts_speed = TimeSeriesPlot("speed")
        self.ts_band = TimeSeriesPlot("band power")
        self.psd = PSDPlot()
        for w in (self.ts_speed, self.ts_band):
            w.seek_requested.connect(
                lambda t: self.state.set_frame(int(t * self.state.fps)))
            il.addWidget(w)
        il.addWidget(self.psd)
        rl.addWidget(insp, 1)

        lay.addWidget(right, 1)

        state.video_loaded.connect(self._on_video_loaded)
        state.cache_opened.connect(self._on_cache_opened)
        state.frame_changed.connect(self._on_frame_changed)

    # -- per-video persistence ----------------------------------------------

    def _on_video_loaded(self):
        # A new clip: forget the previous clip's boxes and load THIS video's own
        # sidecar (if any). This is what stops boxes ghosting across videos.
        self._load_sidecar()

    def _load_sidecar(self):
        path = self.state.video_sidecar("rois")
        self.replicates = []
        self._next_id = 1
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                self.replicates = [{**r, "frac": tuple(r["frac"])}
                                   for r in data.get("replicates", [])]
                self._next_id = max(
                    [r["id"] for r in self.replicates], default=0) + 1
            except (OSError, ValueError) as e:
                self.state.status.emit(f"Could not read boxes: {e}")
        self._rebuild_rois()
        self._refresh_list()
        self._redraw_boxes()

    def _autosave(self):
        """Persist the current boxes to this video's sidecar. Called on every
        edit so the boxes are always attached to the clip -- there is no separate
        'save' step to forget."""
        path = self.state.video_sidecar("rois")
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump({"replicates": self.replicates}, f, indent=2)
        except OSError as e:
            self.state.status.emit(f"Could not save boxes: {e}")

    # -- cache ---------------------------------------------------------------

    def _on_cache_opened(self):
        # Boxes are geometry (fractions), so they survive re-caching the same
        # video at different settings -- only rebuild the ROIs onto the new grid.
        self._rebuild_rois()
        self._refresh_list()
        self._redraw_boxes()

    # -- drawing -------------------------------------------------------------

    def _toggle_draw(self, on: bool):
        self.video.set_draw_mode(on)
        self.draw_btn.setText(f"✏ Draw boxes: {'ON' if on else 'OFF'}")

    def _add_box(self, x0, y0, x1, y1):
        rep = {
            "id": self._next_id,
            "label": f"rep{self._next_id}",
            "frac": (x0, y0, x1, y1),
            "color": _PALETTE[(self._next_id - 1) % len(_PALETTE)],
        }
        self._next_id += 1
        self.replicates.append(rep)
        self._rebuild_rois()
        self._refresh_list()
        self._redraw_boxes()
        self._autosave()
        self.list.setCurrentRow(self.list.count() - 1)

    def _on_box_drawn(self, x0: float, y0: float, x1: float, y1: float):
        # Every drawn box updates the stamp size, so you can redraw to resize.
        self.stamp_size = (x1 - x0, y1 - y0)
        self.stamp_chk.setText(
            f"Fixed-size stamp  ({self.stamp_size[0]*100:.0f}%×"
            f"{self.stamp_size[1]*100:.0f}%)")
        self._add_box(x0, y0, x1, y1)

    def _on_stamp_at(self, cx: float, cy: float):
        """A click in draw mode: drop a stamp-sized box centred on the click."""
        if not self.stamp_chk.isChecked() or self.stamp_size is None:
            return
        w, h = self.stamp_size
        x0 = min(max(cx - w / 2, 0.0), 1.0 - w)
        y0 = min(max(cy - h / 2, 0.0), 1.0 - h)
        self._add_box(x0, y0, x0 + w, y0 + h)

    def _redraw_boxes(self):
        sel = self.state.selected_roi
        self.video.set_frac_boxes([
            (*r["frac"], r["label"], r["color"], r["id"] == sel)
            for r in self.replicates])

    def _rebuild_rois(self):
        """Materialize replicate boxes into rectangular ROIs on the current grid
        and hand them to shared state, so Tab 3 sees them as ROIs."""
        self.state.set_replicate_specs(self.replicates)
        if not self.state.has_cache:
            self.state.rois = []
            self.state.invalidate_series()
            self.state.rois_changed.emit()
            return
        grid = self.state.cache.grid
        n = self.state.cache.n_frames
        packed = self.state.cache.meta.get("processing_scope") == "replicate"
        tiles = {int(t["id"]): t
                 for t in self.state.cache.meta.get("replicate_tiles", [])}
        rois = []
        for r in self.replicates:
            baseline = r.get("baseline_s")
            common = dict(
                label=r["label"],
                baseline_start_s=float(baseline[0]) if baseline else None,
                baseline_end_s=float(baseline[1]) if baseline else None,
                pixels_per_mm=r.get("pixels_per_mm"),
                body_length_mm=r.get("body_length_mm"),
            )
            if packed:
                tile = tiles.get(int(r["id"]))
                if tile is None:
                    continue
                rois.append(packed_rect_roi(
                    r["id"], tuple(r["frac"]),
                    tuple(tile["atlas_bbox"]), grid, n, **common))
            else:
                rois.append(rect_roi(
                    r["id"], r["frac"], grid, n, **common))
        self.state.rois = rois
        self.state.invalidate_series()
        self.state.rois_changed.emit()

    # -- list ----------------------------------------------------------------

    def _refresh_list(self):
        self.list.blockSignals(True)
        self.list.clear()
        for r in self.replicates:
            it = QListWidgetItem(f"  {r['label']}")
            it.setData(Qt.ItemDataRole.UserRole, r["id"])
            # A colour SWATCH icon plus default (theme) text -- not coloured text,
            # which was invisible for light box colours on the light list. The
            # swatch still ties the row to its box; the label stays readable.
            pm = QPixmap(14, 14)
            pm.fill(QColor(r["color"]))
            it.setIcon(QIcon(pm))
            self.list.addItem(it)
        self.list.blockSignals(False)
        if self.list.currentItem() is None:
            self._sync_standardization_controls(None)

    def _selected_rep(self) -> dict | None:
        it = self.list.currentItem()
        if it is None:
            return None
        rid = it.data(Qt.ItemDataRole.UserRole)
        return next((r for r in self.replicates if r["id"] == rid), None)

    def _on_select(self, *_):
        rep = self._selected_rep()
        self.state.selected_roi = rep["id"] if rep else None
        self._sync_standardization_controls(rep)
        self._redraw_boxes()
        self._refresh_inspector()

    def _sync_standardization_controls(self, rep: dict | None):
        widgets = (self.use_baseline, self.baseline_start, self.baseline_end,
                   self.pixels_per_mm, self.body_length_mm)
        for w in widgets:
            w.blockSignals(True)
        try:
            enabled = rep is not None
            for w in widgets:
                w.setEnabled(enabled)
            baseline = rep.get("baseline_s") if rep else None
            self.use_baseline.setChecked(bool(baseline))
            self.baseline_start.setValue(float(baseline[0]) if baseline else 0.0)
            self.baseline_end.setValue(float(baseline[1]) if baseline else 0.0)
            self.pixels_per_mm.setValue(
                float(rep.get("pixels_per_mm") or 0.0) if rep else 0.0)
            self.body_length_mm.setValue(
                float(rep.get("body_length_mm") or 0.0) if rep else 0.0)
        finally:
            for w in widgets:
                w.blockSignals(False)

    def _standardization_changed(self, *_):
        rep = self._selected_rep()
        if rep is None:
            return
        rep["baseline_s"] = ([self.baseline_start.value(),
                              self.baseline_end.value()]
                             if self.use_baseline.isChecked() else None)
        rep["pixels_per_mm"] = self.pixels_per_mm.value() or None
        rep["body_length_mm"] = self.body_length_mm.value() or None
        self._rebuild_rois()
        self._autosave()

    def _on_view_clicked(self, pt):
        """Click a point on the video -> select the box under it."""
        if self.state.source is None:
            return
        # pt is in source-frame pixels; convert to fractions.
        if self.video._cache_frame is None:
            return
        w, h = self.video.view._src_size
        fx, fy = pt.x() / w, pt.y() / h
        for i, r in enumerate(self.replicates):
            x0, y0, x1, y1 = r["frac"]
            if x0 <= fx <= x1 and y0 <= fy <= y1:
                self.list.setCurrentRow(i)
                return

    def _rename(self):
        rep = self._selected_rep()
        if not rep:
            return
        name, ok = QInputDialog.getText(self, "Rename replicate", "Label:",
                                        text=rep["label"])
        if ok and name:
            rep["label"] = name
            self._rebuild_rois()
            self._refresh_list()
            self._redraw_boxes()
            self._autosave()

    def _delete(self):
        rep = self._selected_rep()
        if not rep:
            return
        self.replicates = [r for r in self.replicates if r["id"] != rep["id"]]
        self._rebuild_rois()
        self._refresh_list()
        self._redraw_boxes()
        self._autosave()

    def _clear(self):
        if not self.replicates:
            return
        if QMessageBox.question(self, "Clear all", "Delete all replicate boxes?") \
                == QMessageBox.StandardButton.Yes:
            self.replicates = []
            self._rebuild_rois()
            self._refresh_list()
            self._redraw_boxes()
            self._autosave()

    # -- inspector -----------------------------------------------------------

    def _on_frame_changed(self, idx: int):
        if not self.isVisible():
            return
        t = idx / self.state.fps
        self.ts_speed.set_cursor(t)
        self.ts_band.set_cursor(t)

    def _refresh_inspector(self):
        rep = self._selected_rep()
        if rep is None or not self.state.has_cache:
            return
        roi = self.state.roi_by_id(rep["id"])
        if roi is None:
            return
        ctx = self.state.ctx
        t = ctx.times_s()
        self.ts_speed.set_series(t, self.state.roi_series(roi, "speed"),
                                 f"{rep['label']} — speed (px/s)")
        if self.state.band_features:
            self.ts_band.set_series(
                t, self.state.roi_series(roi, self.state.band_features[0]),
                self.state.band_features[0])
        freqs, psd = roi_psd(self.state.cache, ctx, roi, "speed")
        band = None
        if self.state.cfg.features.bands:
            b = self.state.cfg.features.bands[0]
            band = (b.lo_hz, b.hi_hz)
        self.psd.set_psd(freqs, psd, nyquist=self.state.fps / 2, band=band)

    # -- io ------------------------------------------------------------------

    def _save(self):
        # Boxes already auto-save to this video's sidecar; this button EXPORTS a
        # copy (e.g. a plate layout) to reuse on another clip. Default it next to
        # the video so the common case is one click.
        if not self.replicates:
            QMessageBox.information(self, "Nothing to save", "Draw some boxes first.")
            return
        default = self.state.video_sidecar("rois") or "replicates.json"
        path, _ = QFileDialog.getSaveFileName(self, "Export replicate boxes",
                                              default, "JSON (*.json)")
        if not path:
            return
        with open(path, "w") as f:
            json.dump({"replicates": self.replicates}, f, indent=2)
        self.state.status.emit(f"Exported {len(self.replicates)} boxes.")

    def _load(self):
        # IMPORT a layout from another clip. It then belongs to THIS video, so we
        # immediately auto-save it into this video's sidecar.
        start = os.path.dirname(self.state.video_sidecar("rois") or "") or ""
        path, _ = QFileDialog.getOpenFileName(self, "Import replicate boxes", start,
                                              "JSON (*.json)")
        if not path:
            return
        with open(path) as f:
            data = json.load(f)
        self.replicates = [
            {**r, "frac": tuple(r["frac"])} for r in data.get("replicates", [])]
        self._next_id = max([r["id"] for r in self.replicates], default=0) + 1
        self._rebuild_rois()
        self._refresh_list()
        self._redraw_boxes()
        self._autosave()
        self.state.status.emit(f"Loaded {len(self.replicates)} boxes.")

    def _go_classify(self):
        if not self.replicates:
            QMessageBox.information(self, "No replicates",
                                    "Draw at least one box first.")
            return
        self._rebuild_rois()
        if not self.state.has_cache:
            QMessageBox.information(
                self, "Processing required",
                "These boxes do not yet have a matching per-replicate cache. "
                "Run Test or Full processing in Preprocessing & Flow first.")
            self.state.request_tab.emit(1)
            return
        self.state.request_tab.emit(2)
