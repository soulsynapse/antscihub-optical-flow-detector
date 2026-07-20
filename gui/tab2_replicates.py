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
from PyQt6.QtGui import QColor, QIcon, QKeyEvent, QPixmap
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
        # T12: the boxes on this tab are grabbable. A press inside one selects
        # it; releasing in place zooms to it; dragging repositions it.
        self.video.view.box_drag_enabled = True
        self.video.view.box_grabbed.connect(self._on_box_grabbed)
        self.video.view.box_clicked.connect(self._on_box_clicked)
        self.video.view.box_moved.connect(self._on_box_moved)
        # Right-click is "back" everywhere in this app (four explorers and tab3
        # wire it to un-zoom). T12 asked for right-click-to-delete here; that was
        # refused deliberately, because T12 also introduces the zoom that makes
        # this tab NEED an un-zoom gesture, and a tab where the same button means
        # "go back" in one place and "destroy a replicate" in another is a
        # misfire away from silent data loss. Deletion is the button and the
        # Delete key (see keyPressEvent), neither of which can be hit by
        # reaching for "go back".
        self.video.view.back_requested.connect(self.video.clear_focus)
        ll.addWidget(self.video, 1)

        bar = QHBoxLayout()
        self.draw_btn = QPushButton("✏ Draw boxes: OFF")
        self.draw_btn.setCheckable(True)
        self.draw_btn.setStyleSheet("padding:6px; font-weight:bold;")
        self.draw_btn.toggled.connect(self._toggle_draw)
        bar.addWidget(self.draw_btn, 1)
        self.stamp_chk = QCheckBox("Fixed-size stamp")
        self.stamp_chk.setToolTip(
            "Click empty space to drop a box the size of the SELECTED replicate. "
            "Same-size replicates make block counts comparable.")
        self.stamp_chk.setChecked(True)
        bar.addWidget(self.stamp_chk)
        self.draw_btn.setChecked(True)
        ll.addLayout(bar)

        self.hint = QLabel(
            "Drag out the first replicate box (e.g. one per tube), then CLICK "
            "empty space to place more of the same size. Click a box to select "
            "and zoom to it, drag it to reposition, press Delete to remove it, "
            "right-click to zoom out. "
            "The stamp is always the SELECTED box's size.")
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
        # itemClicked, not currentItemChanged: only a USER picking a row should
        # move the frame. _add_box selects the box it just placed, and zooming
        # on that would fight the place-several-in-a-row workflow T11 exists to
        # support -- every click would fly the view somewhere new.
        self.list.itemClicked.connect(lambda _: self._zoom_to_selected())
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
        self.calibrate_btn = QPushButton("Calibrate by drawing…")
        self.calibrate_btn.setToolTip(
            "Measure both fields on the frame instead of typing them: drag "
            "along the animal for its length, and across a ruler or scale bar "
            "of known size to fix pixels/mm.")
        self.calibrate_btn.clicked.connect(self._open_calibration)
        sf.addRow(self.calibrate_btn)
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
        state.calibration_changed.connect(self._on_calibration_changed)

    # -- per-video persistence ----------------------------------------------

    def _on_video_loaded(self):
        # A new clip: forget the previous clip's boxes and load THIS video's own
        # sidecar (if any). This is what stops boxes ghosting across videos.
        self._load_sidecar()

    def _load_sidecar(self):
        path = self.state.video_sidecar("rois")
        self.replicates = []
        self._next_id = 1
        # A zoom is a rectangle of the PREVIOUS video's frame; carrying it into
        # a new clip frames an arbitrary region of it.
        self.video.clear_focus()
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

    # -- T11: the stamp is the SELECTED replicate's size ----------------------
    #
    # It used to be a stored `stamp_size`, written by the last box DRAWN. That
    # made the stamp and the selection two independent pieces of state saying
    # what "the current box" is, and they diverged the moment you selected an
    # older box: the highlight said one thing, the next click placed another.
    # Deriving it removes the second copy rather than syncing it, so the label,
    # the highlight and what a click actually places cannot disagree.

    @property
    def stamp_size(self) -> tuple[float, float] | None:
        rep = self._selected_rep()
        if rep is None:
            return None
        x0, y0, x1, y1 = rep["frac"]
        return (x1 - x0, y1 - y0)

    def _sync_stamp_label(self):
        size = self.stamp_size
        self.stamp_chk.setText(
            "Fixed-size stamp" if size is None else
            f"Fixed-size stamp  ({size[0]*100:.0f}%×{size[1]*100:.0f}%)")

    def _on_box_drawn(self, x0: float, y0: float, x1: float, y1: float):
        # Drawing selects the new box (_add_box), which by the property above
        # makes it the stamp -- so "drag again to change the stamp size" still
        # holds, without a second variable to keep in step.
        self._add_box(x0, y0, x1, y1)

    def _on_stamp_at(self, cx: float, cy: float):
        """A click on empty space in draw mode: drop a box the size of the
        selected replicate, centred on the click."""
        size = self.stamp_size
        if not self.stamp_chk.isChecked() or size is None:
            return
        w, h = size
        x0 = min(max(cx - w / 2, 0.0), 1.0 - w)
        y0 = min(max(cy - h / 2, 0.0), 1.0 - h)
        self._add_box(x0, y0, x0 + w, y0 + h)

    # -- T12: select, zoom, reposition ---------------------------------------

    def _select_index(self, idx: int):
        if 0 <= idx < self.list.count():
            self.list.setCurrentRow(idx)

    def _zoom_to_selected(self):
        """Frame the selected replicate.

        The box's own frac, verbatim -- the same zoom the four explorers apply
        on a region change (`set_focus_frac(regions[i]["frac"])`). Deliberately
        no margin: this view and an explorer's are the same replicate at the
        same magnification, so a reader can compare them directly, and a tab
        that framed it slightly wider would make the two disagree about how big
        the animal is on screen.

        Dragging still works at full zoom -- the press lands inside the box, and
        _delta_to extrapolates past the widget edge (Qt grabs the mouse), so the
        box follows a cursor that leaves the view.
        """
        rep = self._selected_rep()
        if rep is None:
            return
        x0, y0, x1, y1 = rep["frac"]
        if x1 <= x0 or y1 <= y0:
            return
        self.video.set_focus_frac(rep["frac"])

    def _on_box_grabbed(self, idx: int):
        # Press: select only. The zoom waits for release, or it would move the
        # frame out from under a drag that is just starting.
        self._select_index(idx)

    def _on_box_clicked(self, idx: int):
        self._select_index(idx)
        self._zoom_to_selected()

    def _on_box_moved(self, idx: int, x0: float, y0: float,
                      x1: float, y1: float):
        if not (0 <= idx < len(self.replicates)):
            return
        self.replicates[idx]["frac"] = (x0, y0, x1, y1)
        # A moved box is new geometry, so the ROIs and every series computed
        # against them are stale -- same path as drawing one.
        self._rebuild_rois()
        self._redraw_boxes()
        self._autosave()
        # No _sync_stamp_label: the release clamps the ORIGIN and keeps the
        # size, so a reposition cannot change what the stamp reports.

    def _redraw_boxes(self):
        # This 1:1, in-order mapping IS the contract behind the index carried by
        # box_grabbed / box_clicked / box_moved -- FrameView reports a position
        # in the list published here, and _on_box_moved indexes self.replicates
        # with it. Filtering or reordering this comprehension (to hide boxes
        # outside a zoom, say) would silently reposition the WRONG replicate.
        # Note this tab matches by POSITION where _source_box deliberately
        # matches by id: there, build_layout re-sorts and position is a trap.
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
        # blockSignals above suppresses _on_select, so the label is refreshed
        # here too -- otherwise deleting the selected box leaves the stamp
        # advertising a size that no longer exists.
        self._sync_stamp_label()

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
        # T11: selection IS the stamp, so the label follows it here rather than
        # at the one place a box happens to be drawn.
        self._sync_stamp_label()
        self._redraw_boxes()
        self._refresh_inspector()

    def _sync_standardization_controls(self, rep: dict | None):
        widgets = (self.use_baseline, self.baseline_start, self.baseline_end,
                   self.pixels_per_mm, self.body_length_mm,
                   self.calibrate_btn)
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

    # -- calibration ---------------------------------------------------------

    def _source_box(self, rep: dict):
        """The replicate's box in source pixels, as extraction resolves it.

        Through ``build_layout`` rather than off ``frac`` directly, so the guide
        rectangle in the calibration window is the box the pass actually crops --
        rounding and clamping included. Matched by ``replicate_id``, never by
        position: ``build_layout`` sorts by id and this list is in draw order.
        """
        if self.state.source is None:
            return None
        from core.replicates import build_layout
        info = self.state.source.info
        layout = build_layout(self.replicates, info.width, info.height,
                              scale=1.0, block_size=1)
        tile = next((t for t in layout.tiles
                     if int(t.replicate_id) == int(rep["id"])), None)
        return tile.source_box if tile is not None else None

    def _open_calibration(self):
        rep = self._selected_rep()
        if rep is None or self.state.source is None:
            return
        frame = self.state.source.frame_at(self.state.current_frame)
        if frame is None:
            self.state.status.emit(
                "Could not decode the current frame to calibrate against.")
            return
        from gui.calibration_dialog import CalibrationDialog
        dlg = CalibrationDialog(
            frame, box=self._source_box(rep), label=str(rep.get("label", "")),
            pixels_per_mm=rep.get("pixels_per_mm"),
            body_length_mm=rep.get("body_length_mm"),
            body_length_px=rep.get("body_length_px"),
            scale=1.0, parent=self)
        try:
            if dlg.exec() and dlg.calibration is not None:
                self._merge_calibration(rep["id"],
                                        dlg.calibration.as_replicate_fields())
        finally:
            dlg.deleteLater()

    def _on_calibration_changed(self, replicate_id: int, fields: dict):
        """A calibration measured on another tab. Persisting is this tab's job:
        it owns the list and the per-video sidecar."""
        self._merge_calibration(replicate_id, fields)

    def _merge_calibration(self, replicate_id: int, fields: dict):
        rep = next((r for r in self.replicates
                    if int(r["id"]) == int(replicate_id)), None)
        if rep is None or not fields:
            return
        rep.update(fields)
        # Re-read the spin boxes off the dict rather than setting them from
        # `fields`, so a partial calibration (animal only, no ruler) leaves the
        # untouched field showing what it already held.
        if rep is self._selected_rep():
            self._sync_standardization_controls(rep)
        self._rebuild_rois()
        self._autosave()

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

    # _on_view_clicked is gone: FrameView._box_at now owns the hit test for both
    # tabs, and `clicked` no longer fires for a press inside a box. Keeping the
    # old scan would have left TWO hit tests with different overlap rules (this
    # one took the first match, _box_at prefers the selected box), which is the
    # kind of pair that agrees until boxes overlap and then disagrees silently.

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
        # Deleting what you are zoomed into otherwise strands the view on empty
        # frame, showing a region that no longer belongs to anything.
        self.video.clear_focus()
        self._rebuild_rois()
        self._refresh_list()
        self._redraw_boxes()
        self._autosave()

    def keyPressEvent(self, ev: QKeyEvent):
        """Delete/Backspace removes the selected replicate.

        Only reached when no text field holds focus: a QLineEdit or spin box
        consumes Delete for its own editing, and Qt stops propagating there, so
        deleting a digit in "source pixels / mm" can never destroy the box. The
        list and the frame view both ignore the key, which is exactly where the
        gesture is meant to work.

        Backspace too, because "in a replicate" here usually means zoomed into
        one with the frame view focused, and the Mac habit is Backspace.
        """
        if ev.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) \
                and self._selected_rep() is not None:
            self._delete()
            ev.accept()
            return
        super().keyPressEvent(ev)

    def _clear(self):
        if not self.replicates:
            return
        if QMessageBox.question(self, "Clear all", "Delete all replicate boxes?") \
                == QMessageBox.StandardButton.Yes:
            self.replicates = []
            self.video.clear_focus()
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
