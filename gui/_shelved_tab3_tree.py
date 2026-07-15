"""Tab 3: behavior classification.

The behavior spec is edited as an explicit AND/OR tree. Selecting a range leaf
shows that feature's histogram with the leaf's range on it -- but the histogram
here is over the ROI TIME-SERIES distribution, not the pixel distribution of
Tab 2. That distinction is not cosmetic: in Tab 2 a band-power histogram counts
every block in every frame, most of which are empty background, so the mode sits
at zero. Here it counts one value per ROI per window, so the mode sits wherever
your ROIs actually are. Putting a threshold on the wrong one of those two
distributions is the single easiest way to get a nonsense classifier.
"""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QColorDialog, QComboBox, QDoubleSpinBox,
                             QFileDialog, QGroupBox, QHBoxLayout, QInputDialog,
                             QLabel, QLineEdit, QMessageBox, QPushButton,
                             QSplitter, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from core.behavior import (Behavior, LogicNode, RangeLeaf, TemporalCriteria,
                           default_wingbeat, trace_to_bouts)
from core.export import (export_bouts, export_hdf5, export_roi_timeseries,
                         export_summary)
from core.features import REGISTRY
from core.filters import DEFAULT_BINS
from core.roi import roi_psd, roi_time_series
from gui.histogram_widget import RangeHistogram
from gui.inspector import ConstraintList, PSDPlot, TimeSeriesPlot
from gui.state import AppState
from gui.timeline import TimelineStrip
from gui.video_panel import VideoPanel


class Tab3Behavior(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.current: Behavior | None = None
        self._series_cache: dict[str, np.ndarray] = {}

        split = QSplitter(Qt.Orientation.Horizontal)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(split)

        # -- left: video + timeline -----------------------------------------
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(2, 2, 2, 2)
        self.video = VideoPanel(state)
        self.video.block_clicked.connect(self._on_block_clicked)
        ll.addWidget(self.video, 1)

        ll.addWidget(QLabel("Ethogram — behavior state per ROI"))
        self.timeline = TimelineStrip()
        self.timeline.seek_requested.connect(
            lambda t: self.state.set_frame(int(t * self.state.fps)))
        self.timeline.roi_clicked.connect(self._select_roi)
        ll.addWidget(self.timeline)
        split.addWidget(left)

        # -- middle: library + tree editor -----------------------------------
        mid = QWidget()
        ml = QVBoxLayout(mid)

        lib_box = QGroupBox("Behavior library")
        lb = QVBoxLayout(lib_box)
        self.lib_picker = QComboBox()
        self.lib_picker.currentTextChanged.connect(self._load_behavior)
        lb.addWidget(self.lib_picker)

        row = QHBoxLayout()
        for text, fn in (("New", self._new), ("Wingbeat example", self._add_example),
                         ("Save", self._save), ("Delete", self._delete)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lb.addLayout(row)

        row2 = QHBoxLayout()
        imp = QPushButton("Import…")
        imp.clicked.connect(self._import)
        exp = QPushButton("Export…")
        exp.clicked.connect(self._export_behavior)
        row2.addWidget(imp)
        row2.addWidget(exp)
        lb.addLayout(row2)
        ml.addWidget(lib_box)

        edit_box = QGroupBox("Specification tree")
        el = QVBoxLayout(edit_box)

        name_row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("behavior name")
        self.color_btn = QPushButton("Colour")
        self.color_btn.clicked.connect(self._pick_color)
        name_row.addWidget(self.name_edit, 1)
        name_row.addWidget(self.color_btn)
        el.addLayout(name_row)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["node", "constraint"])
        self.tree.currentItemChanged.connect(self._on_node_selected)
        el.addWidget(self.tree, 1)

        btn_row = QHBoxLayout()
        for text, fn in (("+ AND", lambda: self._add_node("and")),
                         ("+ OR", lambda: self._add_node("or")),
                         ("+ Range", self._add_leaf),
                         ("Remove", self._remove_node)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            btn_row.addWidget(b)
        el.addLayout(btn_row)

        self.leaf_hist: RangeHistogram | None = None
        self.leaf_hist_holder = QVBoxLayout()
        el.addLayout(self.leaf_hist_holder)

        ml.addWidget(edit_box, 1)

        crit_box = QGroupBox("Temporal criteria")
        cl = QVBoxLayout(crit_box)
        self.min_dur = QDoubleSpinBox()
        self.min_dur.setRange(0.0, 60.0)
        self.min_dur.setValue(0.3)
        self.min_dur.setPrefix("min duration  ")
        self.min_dur.setSuffix(" s")
        self.max_gap = QDoubleSpinBox()
        self.max_gap.setRange(0.0, 10.0)
        self.max_gap.setValue(0.15)
        self.max_gap.setPrefix("bridge gaps up to  ")
        self.max_gap.setSuffix(" s")
        self.smooth = QDoubleSpinBox()
        self.smooth.setRange(0.0, 5.0)
        self.smooth.setValue(0.0)
        self.smooth.setPrefix("smooth  ")
        self.smooth.setSuffix(" s")
        for w in (self.min_dur, self.max_gap, self.smooth):
            w.valueChanged.connect(self._on_criteria_changed)
            cl.addWidget(w)
        ml.addWidget(crit_box)

        split.addWidget(mid)

        # -- right: inspector + export ---------------------------------------
        right = QWidget()
        rl = QVBoxLayout(right)

        self.roi_label = QLabel("No ROI selected.")
        self.roi_label.setStyleSheet("color:#bbb;")
        rl.addWidget(self.roi_label)

        self.constraints = ConstraintList()
        rl.addWidget(self.constraints)

        self.plots: list[TimeSeriesPlot] = []
        for _ in range(3):
            p = TimeSeriesPlot("")
            p.seek_requested.connect(
                lambda t: self.state.set_frame(int(t * self.state.fps)))
            self.plots.append(p)
            rl.addWidget(p)

        self.psd = PSDPlot()
        rl.addWidget(self.psd)

        self.bout_label = QLabel("")
        self.bout_label.setStyleSheet(
            "font-family: Consolas; font-size:11px; color:#bbb;")
        self.bout_label.setWordWrap(True)
        rl.addWidget(self.bout_label)

        exp_box = QGroupBox("Export")
        xl = QVBoxLayout(exp_box)
        for text, fn in (("Per-ROI time series + traces (CSV)", self._export_ts),
                         ("Bouts (CSV)", self._export_bouts),
                         ("Summary: ROI x behavior (CSV)", self._export_summary),
                         ("Everything (HDF5)", self._export_h5)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            xl.addWidget(b)
        rl.addWidget(exp_box)
        rl.addStretch(1)

        split.addWidget(right)
        # Video + ethogram get half the window.
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 1)
        split.setSizes([900, 450, 450])

        state.cache_opened.connect(self._refresh_library)
        state.rois_changed.connect(self._recompute)
        state.frame_changed.connect(self._on_frame_changed)
        self._refresh_library()

    # -- library -------------------------------------------------------------

    def _refresh_library(self):
        self.state.reload_behaviors()
        self.lib_picker.blockSignals(True)
        self.lib_picker.clear()
        self.lib_picker.addItems(self.state.library.list())
        self.lib_picker.blockSignals(False)

    def _new(self):
        name, ok = QInputDialog.getText(self, "New behavior", "Name:")
        if not ok or not name:
            return
        self.current = Behavior(name=name, spec=LogicNode(op="and"))
        self._sync_editor()

    def _add_example(self):
        if not self.state.has_cache or not self.state.band_features:
            QMessageBox.information(
                self, "Need a cache",
                "Open a cache with at least one band-power feature first.")
            return
        b = default_wingbeat(self.state.fps, self.state.band_features[0])
        self.state.library.save(b)
        self._refresh_library()
        self.lib_picker.setCurrentText(b.name)
        QMessageBox.information(
            self, "Wingbeat example added",
            "The band-power threshold in this example is a starting point only. "
            "It is in (px/s)^2/Hz and scales with resolution and frame rate, so "
            "you MUST retune it on the histogram for your footage — select the "
            "band-power leaf in the tree and drag its range onto the mode your "
            "ROIs actually occupy.")

    def _load_behavior(self, name: str):
        if not name:
            return
        try:
            self.current = self.state.library.load(name)
        except Exception as e:
            QMessageBox.critical(self, "Could not load", str(e))
            return
        self._sync_editor()
        self._recompute()

    def _save(self):
        if not self.current:
            return
        self.current.name = self.name_edit.text() or self.current.name
        self.current.criteria = self._criteria()
        self.state.library.save(self.current)
        self._refresh_library()
        self.lib_picker.setCurrentText(self.current.name)
        self.state.status.emit(f"Saved behavior '{self.current.name}'.")

    def _delete(self):
        name = self.lib_picker.currentText()
        if not name:
            return
        if QMessageBox.question(self, "Delete", f"Delete '{name}'?") == \
                QMessageBox.StandardButton.Yes:
            self.state.library.delete(name)
            self.current = None
            self._refresh_library()
            self._recompute()

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import behavior", "",
                                              "JSON (*.json)")
        if not path:
            return
        b = Behavior.load(path)
        self.state.library.save(b)
        self._refresh_library()
        self.lib_picker.setCurrentText(b.name)

    def _export_behavior(self):
        if not self.current:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export behavior", f"{self.current.name}.json", "JSON (*.json)")
        if path:
            self.current.save(path)

    def _pick_color(self):
        if not self.current:
            return
        c = QColorDialog.getColor()
        if c.isValid():
            self.current.color = c.name()
            self.color_btn.setStyleSheet(f"background:{c.name()};")
            self._recompute()

    # -- tree editor ---------------------------------------------------------

    def _sync_editor(self):
        self.tree.clear()
        if not self.current:
            return
        self.name_edit.setText(self.current.name)
        self.color_btn.setStyleSheet(f"background:{self.current.color};")
        self.min_dur.blockSignals(True)
        self.max_gap.blockSignals(True)
        self.smooth.blockSignals(True)
        self.min_dur.setValue(self.current.criteria.min_duration_s)
        self.max_gap.setValue(self.current.criteria.max_gap_s)
        self.smooth.setValue(self.current.criteria.smooth_s)
        self.min_dur.blockSignals(False)
        self.max_gap.blockSignals(False)
        self.smooth.blockSignals(False)

        root = self._build_item(self.current.spec)
        self.tree.addTopLevelItem(root)
        self.tree.expandAll()

    def _build_item(self, node) -> QTreeWidgetItem:
        if isinstance(node, RangeLeaf):
            spec = REGISTRY.get(node.feature)
            it = QTreeWidgetItem([spec.label if spec else node.feature,
                                  f"[{node.lo:g}, {node.hi:g}]"])
        else:
            it = QTreeWidgetItem([node.op.upper(), ""])
            for c in node.children:
                it.addChild(self._build_item(c))
        it.setData(0, Qt.ItemDataRole.UserRole, node)
        return it

    def _selected_node(self):
        it = self.tree.currentItem()
        return it.data(0, Qt.ItemDataRole.UserRole) if it else None

    def _add_node(self, op: str):
        if not self.current:
            return
        target = self._selected_node()
        parent = target if isinstance(target, LogicNode) else self.current.spec
        parent.children.append(LogicNode(op=op))
        self._sync_editor()

    _CUSTOM_BAND = "Band power (custom band)…"

    def _add_leaf(self):
        if not self.current or not self.state.has_cache:
            return
        feats = self.state.available_features()
        labels = [REGISTRY[f].label if f in REGISTRY else f for f in feats]
        # Band power for an arbitrary pass-band is computed on demand from the
        # cached speed series -- it does not have to have been cached in Tab 1.
        labels = [self._CUSTOM_BAND] + labels
        label, ok = QInputDialog.getItem(self, "Add range constraint", "Feature:",
                                         labels, 0, False)
        if not ok:
            return

        if label == self._CUSTOM_BAND:
            nyq = self.state.fps / 2
            lo, ok1 = QInputDialog.getDouble(
                self, "Custom band", f"Low edge (Hz), Nyquist = {nyq:.1f}:",
                12.0, 0.0, nyq, 2)
            if not ok1:
                return
            hi, ok2 = QInputDialog.getDouble(
                self, "Custom band", f"High edge (Hz), Nyquist = {nyq:.1f}:",
                min(24.0, nyq), lo, nyq, 2)
            if not ok2:
                return
            name = f"bandpower_{lo:g}-{hi:g}Hz"
        else:
            name = feats[labels.index(label) - 1]  # -1 for the prepended entry

        col = self._roi_distribution(name)
        lo = float(np.percentile(col, 50)) if col.size else 0.0
        hi = float(np.percentile(col, 100)) if col.size else 1.0

        target = self._selected_node()
        parent = target if isinstance(target, LogicNode) else self.current.spec
        parent.children.append(RangeLeaf(feature=name, lo=lo, hi=hi))
        self._sync_editor()
        self._recompute()

    def _remove_node(self):
        node = self._selected_node()
        if not self.current or node is None or node is self.current.spec:
            return

        def prune(parent) -> bool:
            for c in list(parent.children):
                if c is node:
                    parent.children.remove(c)
                    return True
                if isinstance(c, LogicNode) and prune(c):
                    return True
            return False

        prune(self.current.spec)
        self._sync_editor()
        self._recompute()

    def _on_node_selected(self, item, _prev):
        # Tear down any previous leaf histogram.
        if self.leaf_hist is not None:
            self.leaf_hist_holder.removeWidget(self.leaf_hist)
            self.leaf_hist.deleteLater()
            self.leaf_hist = None

        node = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        if not isinstance(node, RangeLeaf) or not self.state.has_cache:
            return

        spec = REGISTRY.get(node.feature)
        h = RangeHistogram(node.feature, spec.label if spec else node.feature,
                           spec.units if spec else "")
        col = self._roi_distribution(node.feature)
        if col.size == 0:
            return

        lo = float(np.percentile(col, 0.5))
        hi = float(np.percentile(col, 99.5))
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, DEFAULT_BINS + 1, dtype=np.float32)
        total, _ = np.histogram(col, bins=edges)

        # The "filtered" curve here is the distribution restricted to samples
        # where every OTHER leaf of this behavior passes -- the same
        # cross-filtering idea as Tab 2, but over ROI time-series values.
        mask = self._other_leaves_mask(node)
        filt, _ = np.histogram(col[mask] if mask is not None else col, bins=edges)

        h.set_data(edges, total, filt)
        h.set_range(node.lo, min(node.hi, float(edges[-1]))
                    if np.isfinite(node.hi) else float(edges[-1]))
        h.range_changed.connect(
            lambda _n, lo_, hi_, nd=node: self._on_leaf_range(nd, lo_, hi_))
        self.leaf_hist_holder.addWidget(h)
        self.leaf_hist = h

    def _on_leaf_range(self, node: RangeLeaf, lo: float, hi: float):
        node.lo, node.hi = lo, hi
        it = self.tree.currentItem()
        if it:
            it.setText(1, f"[{lo:g}, {hi:g}]")
        self._recompute()

    def _criteria(self) -> TemporalCriteria:
        return TemporalCriteria(min_duration_s=self.min_dur.value(),
                                max_gap_s=self.max_gap.value(),
                                smooth_s=self.smooth.value())

    def _on_criteria_changed(self):
        if self.current:
            self.current.criteria = self._criteria()
            self._recompute()

    # -- ROI-space distributions --------------------------------------------

    def _roi_distribution(self, feature: str) -> np.ndarray:
        """Every ROI's time series for a feature, concatenated.

        This -- not the pixel distribution -- is the population a behavior
        threshold has to separate, because a behavior is evaluated on ROI time
        series. If no ROIs exist yet we fall back to the pixel sample so the
        histogram is at least drawn, but the axis will not be representative.
        """
        if not self.state.has_cache:
            return np.zeros(0, np.float32)
        if not self.state.rois:
            if self.state.sampler:
                return self.state.sampler.column(feature)
            return np.zeros(0, np.float32)

        key = feature
        if key in self._series_cache:
            return self._series_cache[key]
        parts = [self.state.roi_series(r, feature) for r in self.state.rois]
        col = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        self._series_cache[key] = col
        return col

    def _other_leaves_mask(self, exclude: RangeLeaf):
        if not self.state.rois or not self.current:
            return None
        leaves: list[RangeLeaf] = []

        def walk(n):
            if isinstance(n, RangeLeaf):
                leaves.append(n)
            else:
                for c in n.children:
                    walk(c)

        walk(self.current.spec)
        others = [l for l in leaves if l is not exclude]
        if not others:
            return None

        n_total = self._roi_distribution(exclude.feature).size
        m = np.ones(n_total, bool)
        for l in others:
            col = self._roi_distribution(l.feature)
            if col.size != n_total:
                continue
            m &= (col >= l.lo) & (col <= l.hi)
        return m

    # -- evaluation ----------------------------------------------------------

    def _recompute(self):
        self._series_cache.clear()
        if not self.state.has_cache:
            return
        if self.current:
            self.current.criteria = self._criteria()
            # Keep the edited behavior in the state's list so the timeline and
            # exports see the un-saved version the user is looking at.
            self.state.behaviors = [
                b for b in self.state.behaviors if b.name != self.current.name
            ] + [self.current]
        self.state.recompute_traces()
        self._refresh_timeline()
        self._refresh_inspector()

    def _refresh_timeline(self):
        rows = []
        for r in self.state.rois:
            beh = {}
            for b in self.state.behaviors:
                tr = self.state.traces.get((r.roi_id, b.name))
                if tr is not None:
                    beh[b.name] = (tr, b.color)
            # Show the replicate's label (its note), not just a numeric id.
            rows.append((r.roi_id, r.note or f"#{r.roi_id}", beh))
        self.timeline.selected_roi = self.state.selected_roi
        self.timeline.set_rows(
            rows, self.state.cache.n_frames / self.state.fps
            if self.state.has_cache else 1.0)

        # Colour each ROI box by whether it matches the current behavior NOW.
        if self.current and self.state.has_cache:
            f = self.state.current_frame
            boxes = []
            for r in self.state.rois:
                tr = self.state.traces.get((r.roi_id, self.current.name))
                on = tr is not None and f < tr.size and bool(tr[f])
                boxes.append((r.roi_id, r.bbox,
                              self.current.color if on else "#555555"))
            self.video.set_roi_boxes(boxes)

    def _select_roi(self, roi_id: int):
        self.state.selected_roi = roi_id
        self._refresh_inspector()
        self._refresh_timeline()

    def _on_block_clicked(self, by: int, bx: int):
        for r in self.state.rois:
            if r.mask is not None and r.mask[by, bx]:
                self._select_roi(r.roi_id)
                return

    def _on_frame_changed(self, idx: int):
        if not self.isVisible():
            return
        t = idx / self.state.fps
        self.timeline.set_cursor(t)
        for p in self.plots:
            p.set_cursor(t)
        self._refresh_constraints()
        self._refresh_timeline()

    def _refresh_inspector(self):
        roi = self.state.roi_by_id(self.state.selected_roi) \
            if self.state.selected_roi is not None else None
        if roi is None or not self.state.has_cache:
            self.roi_label.setText("No ROI selected.")
            return

        fps = self.state.fps
        self.roi_label.setText(
            f"ROI #{roi.roi_id} — {roi.duration_s(fps):.2f} s, "
            f"{int(roi.mask.sum())} blocks, bbox {roi.bbox}")

        cache, ctx = self.state.cache, self.state.ctx
        t = ctx.times_s()

        feats = ["speed", "coherence"]
        if self.state.band_features:
            feats.append(self.state.band_features[0])
        for p, f in zip(self.plots, feats):
            p.set_series(t, self.state.roi_series(roi, f), f)

        # Shade the plots wherever the current behavior is active, so you can see
        # the detection sitting on top of the signal that produced it.
        if self.current:
            tr = self.state.traces.get((roi.roi_id, self.current.name))
            if tr is not None:
                bands = [(b.start_s, b.end_s, self.current.color)
                         for b in trace_to_bouts(tr, fps)]
                for p in self.plots:
                    p.set_bands(bands)
                total = sum(b.duration_s for b in trace_to_bouts(tr, fps))
                bouts = trace_to_bouts(tr, fps)
                self.bout_label.setText(
                    f"{self.current.name}: {len(bouts)} bouts, "
                    f"{total:.2f} s total, "
                    f"mean {np.mean([b.duration_s for b in bouts]):.2f} s"
                    if bouts else f"{self.current.name}: no bouts")

        freqs, psd = roi_psd(cache, ctx, roi, "speed")
        band = None
        if self.state.cfg.features.bands:
            b = self.state.cfg.features.bands[0]
            band = (b.lo_hz, b.hi_hz)
        self.psd.set_psd(freqs, psd, nyquist=fps / 2, band=band)
        self._refresh_constraints()

    def _refresh_constraints(self):
        roi = self.state.roi_by_id(self.state.selected_roi) \
            if self.state.selected_roi is not None else None
        if roi is None or not self.current or not self.state.has_cache:
            return
        feats = sorted(self.current.features())
        try:
            series = {f: self.state.roi_series(roi, f) for f in feats}
        except KeyError:
            return
        rows = self.current.constraint_status(series, self.state.current_frame)
        self.constraints.set_status(rows, self.current.name)

    # -- export --------------------------------------------------------------

    def _check_ready(self) -> bool:
        if not self.state.has_cache or not self.state.rois:
            QMessageBox.information(self, "Nothing to export",
                                    "Extract ROIs in Tab 2 first.")
            return False
        return True

    def _features_for_export(self) -> list[str]:
        base = ["speed", "net_speed", "coherence", "angle", "rolling_std_speed"]
        return base + self.state.band_features

    def _export_ts(self):
        if not self._check_ready():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export time series", "roi_timeseries.csv", "CSV (*.csv)")
        if not path:
            return
        export_roi_timeseries(path, self.state.cache, self.state.ctx,
                              self.state.rois, self._features_for_export(),
                              self.state.behaviors, self.state.traces)
        self.state.status.emit(f"Wrote {os.path.basename(path)}")

    def _export_bouts(self):
        if not self._check_ready():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export bouts", "bouts.csv",
                                              "CSV (*.csv)")
        if path:
            export_bouts(path, self.state.rois, self.state.behaviors,
                         self.state.traces, self.state.fps)
            self.state.status.emit(f"Wrote {os.path.basename(path)}")

    def _export_summary(self):
        if not self._check_ready():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export summary",
                                              "summary.csv", "CSV (*.csv)")
        if path:
            export_summary(path, self.state.rois, self.state.behaviors,
                           self.state.traces, self.state.fps)
            self.state.status.emit(f"Wrote {os.path.basename(path)}")

    def _export_h5(self):
        if not self._check_ready():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export HDF5",
                                              "detections.h5", "HDF5 (*.h5)")
        if path:
            export_hdf5(path, self.state.cache, self.state.ctx, self.state.rois,
                        self._features_for_export(), self.state.behaviors,
                        self.state.traces)
            self.state.status.emit(f"Wrote {os.path.basename(path)}")
