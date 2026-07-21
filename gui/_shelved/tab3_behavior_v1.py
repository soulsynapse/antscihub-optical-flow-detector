"""Tab 3: Behavior classification, additive and visual.

A behavior is a flat AND of feature range constraints -- that is the common case,
and if you need an OR you make a second behavior and let it catch the other case.
You build it by adding features and dragging the range lines directly on the
selected replicate's own time series (no separate histogram): you see the signal,
you bracket the part you want, and two things update live:

  * the ethogram, one row per replicate, showing when the behavior is active;
  * the VIDEO, where every block that currently satisfies all the constraints is
    painted in the behavior's colour -- so you literally watch the detector fire
    on the footage as you tune it.

Those two views cannot disagree: a replicate's band power is the MAX over its
blocks, and the overlay thresholds those same per-block values, so a box's row is
active exactly when one of its blocks is lit.
"""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QColorDialog, QComboBox, QDoubleSpinBox,
                             QFileDialog, QGroupBox, QHBoxLayout, QInputDialog,
                             QLabel, QLineEdit, QMessageBox, QPushButton,
                             QCheckBox, QScrollArea, QSpinBox, QSplitter,
                             QVBoxLayout, QWidget)

from core.behavior import (Behavior, LogicNode, RangeLeaf, SpatialCriteria,
                           TemporalCriteria, default_wingbeat, trace_to_bouts)
from core.export import (export_bouts, export_hdf5, export_roi_timeseries,
                         export_summary)
from core.features import REGISTRY
from core.roi import behavior_block_mask, roi_psd
from gui.inspector import ConstraintList, PSDPlot, RangePlot
from gui.state import AppState
from gui._shelved.timeline import TimelineStrip
from gui.video_panel import VideoPanel


def _hex_to_bgr_tuple(hexcol: str) -> tuple[int, int, int]:
    h = hexcol.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


class Tab3Behavior(QWidget):
    _CUSTOM_BAND = "Band power (custom band)…"

    # Every standard measure shown in the all-measures panel, most useful for
    # behavior discrimination first. Band powers (cache + custom) are appended.
    MEASURES = [
        "speed", "rolling_mean_speed", "rel_speed", "rolling_std_speed",
        "net_speed", "coherence", "spectral_flatness", "direction_oscillation",
        "dominant_freq", "divergence", "curl",
    ]

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.current: Behavior | None = None
        self.plots: dict[str, RangePlot] = {}   # feature -> RangePlot

        split = QSplitter(Qt.Orientation.Horizontal)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(split)

        # -- left: video + ethogram -----------------------------------------
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(2, 2, 2, 2)
        self.video = VideoPanel(state)
        self.video.block_clicked.connect(self._on_block_clicked)
        ll.addWidget(self.video, 1)

        etho_row = QHBoxLayout()
        etho_row.addWidget(QLabel(
            "Ethogram — click a row to inspect; middle-drag a row to mark ground "
            "truth for the label below:"), 1)
        etho_row.addWidget(QLabel("Mark as:"))
        self.mark_label_picker = QComboBox()
        self.mark_label_picker.setEditable(True)
        self.mark_label_picker.setMinimumWidth(120)
        self.mark_label_picker.setToolTip(
            "Which behavior your middle-drag spans are labelled as. Pick a saved "
            "behavior, or type a new label (e.g. 'Still'). Each label draws in its "
            "own colour, so you can lay down Flying spans, switch, and lay down "
            "Still spans on the same rows.")
        self.mark_label_picker.currentTextChanged.connect(self._on_mark_label)
        etho_row.addWidget(self.mark_label_picker)
        clear_marks_btn = QPushButton("Clear label")
        clear_marks_btn.setToolTip(
            "Remove ground-truth spans for the 'Mark as' label. Hold Shift to clear "
            "ALL labels.")
        clear_marks_btn.clicked.connect(self._clear_marks_clicked)
        etho_row.addWidget(clear_marks_btn)
        ll.addLayout(etho_row)
        self.timeline = TimelineStrip()
        self.timeline.seek_requested.connect(
            lambda t: self.state.set_frame(int(t * self.state.fps)))
        self.timeline.roi_clicked.connect(self._select_roi)
        self.timeline.marks_changed.connect(self._save_marks)
        ll.addWidget(self.timeline)
        split.addWidget(left)

        # -- middle: library + criteria + add constraint --------------------
        mid = QWidget()
        ml = QVBoxLayout(mid)

        lib_box = QGroupBox("Behavior")
        lb = QVBoxLayout(lib_box)
        self.lib_picker = QComboBox()
        self.lib_picker.currentTextChanged.connect(self._load_behavior)
        lb.addWidget(self.lib_picker)

        name_row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("behavior name")
        self.color_btn = QPushButton("Colour")
        self.color_btn.clicked.connect(self._pick_color)
        name_row.addWidget(self.name_edit, 1)
        name_row.addWidget(self.color_btn)
        lb.addLayout(name_row)

        row = QHBoxLayout()
        for text, fn in (("New", self._new),
                         ("Wingbeat example", self._add_example),
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

        add_box = QGroupBox("Constraints  (behavior = AND of the ENABLED ones)")
        al = QVBoxLayout(add_box)
        self.add_btn = QPushButton("+ Add custom band")
        self.add_btn.clicked.connect(self._add_constraint)
        al.addWidget(self.add_btn)
        hint = QLabel("Every measure is shown as a plot on the right. Tick a "
                      "measure to make it part of the behavior; drag its dashed "
                      "lines to set the range. Unticked measures still show their "
                      "signal so you can see which ones separate the behavior. "
                      "For an OR, make a second behavior.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#333; font-size:11px;")
        al.addWidget(hint)
        ml.addWidget(add_box)

        spat_box = QGroupBox("Spatial criteria  (how much of the box must pass)")
        sl = QVBoxLayout(spat_box)
        self.min_blocks = QSpinBox()
        self.min_blocks.setRange(1, 100000)
        self.min_blocks.setValue(1)
        self.min_blocks.setPrefix("min blocks in a clump  ")
        self.min_blocks.setToolTip(
            "Fewest passing blocks (in one merged clump) for a frame to count. "
            "1 = any block. Raise it to require a real region — e.g. an ant "
            "crossing is a small clump of moving blocks, not one stray block.")
        self.min_fraction = QDoubleSpinBox()
        self.min_fraction.setRange(0.0, 1.0)
        self.min_fraction.setSingleStep(0.05)
        self.min_fraction.setValue(0.0)
        self.min_fraction.setPrefix("min fraction of box  ")
        self.min_fraction.setToolTip(
            "Alternatively, require this fraction of the box's blocks to pass "
            "(0 = ignore). Good when the behavior fills a known share of the "
            "region rather than a fixed block count.")
        self.merge_dist = QSpinBox()
        self.merge_dist.setRange(0, 50)
        self.merge_dist.setValue(0)
        self.merge_dist.setPrefix("merge distance  ")
        self.merge_dist.setSuffix(" blocks")
        self.merge_dist.setToolTip(
            "Blocks within this distance are treated as one clump before the "
            "min-blocks test (like the colour detector's merge distance). "
            "0 = only touching blocks count together.")
        for w in (self.min_blocks, self.min_fraction, self.merge_dist):
            w.valueChanged.connect(self._on_spatial_changed)
            sl.addWidget(w)
        ml.addWidget(spat_box)

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

        self.roi_label = QLabel("No replicate selected.")
        self.roi_label.setStyleSheet("color:#000; font-weight:bold;")
        ml.addWidget(self.roi_label)
        self.constraints = ConstraintList()
        ml.addWidget(self.constraints)
        ml.addStretch(1)
        split.addWidget(mid)

        # -- right: the per-constraint range plots + PSD --------------------
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.addWidget(QLabel("Drag the range lines on each plot for the selected "
                            "replicate:"))

        self.plot_scroll = QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_holder = QWidget()
        self.plot_lay = QVBoxLayout(self.plot_holder)
        self.plot_lay.setSpacing(8)
        self.plot_lay.addStretch(1)
        self.plot_scroll.setWidget(self.plot_holder)
        rl.addWidget(self.plot_scroll, 1)

        self.psd = PSDPlot()
        rl.addWidget(self.psd)

        self.bout_label = QLabel("")
        self.bout_label.setStyleSheet(
            "font-family: Consolas; font-size:11px; color:#000;")
        self.bout_label.setWordWrap(True)
        rl.addWidget(self.bout_label)

        exp_box = QGroupBox("Export")
        xl = QVBoxLayout(exp_box)
        for text, fn in (("Per-ROI time series + traces (CSV)", self._export_ts),
                         ("Bouts (CSV)", self._export_bouts),
                         ("Summary: replicate x behavior (CSV)", self._export_summary),
                         ("Everything (HDF5)", self._export_h5)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            xl.addWidget(b)
        rl.addWidget(exp_box)

        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 1)
        split.setSizes([900, 430, 470])

        state.cache_opened.connect(self._refresh_library)
        state.rois_changed.connect(self._on_rois_changed)
        state.frame_changed.connect(self._on_frame_changed)
        self._refresh_library()
        self._load_marks()
        self._refresh_mark_labels()

    # -- ground-truth marks (persisted) --------------------------------------

    # A fallback palette for labels that are not a saved behavior (so a typed
    # "Still" still gets a stable, distinct colour).
    _MARK_PALETTE = ["#ff4488", "#4ac6ff", "#ffd24a", "#6ee06e", "#c78bff",
                     "#ff9d3a", "#00d0b0", "#ff6d6d"]

    def _marks_path(self) -> str:
        return os.path.join(self.state.project_dir, "marks.json")

    def _save_marks(self):
        import json
        try:
            with open(self._marks_path(), "w") as f:
                json.dump({"marks": self.timeline.marks_to_dict()}, f, indent=2)
        except OSError as e:
            self.state.status.emit(f"Could not save marks: {e}")

    def _load_marks(self):
        import json
        try:
            with open(self._marks_path()) as f:
                self.timeline.set_marks_from_dict(json.load(f).get("marks", {}))
        except (OSError, ValueError):
            pass

    def _label_color(self, label: str) -> str:
        """A stable colour for a mark label: the behavior's own colour if it is a
        saved/current behavior, else a palette colour keyed off the name."""
        if self.current and label == self.current.name:
            return self.current.color
        if label in self.timeline.label_colors:
            return self.timeline.label_colors[label]
        try:
            if label in self.state.library.list():
                return self.state.library.load(label).color
        except Exception:
            pass
        return self._MARK_PALETTE[abs(hash(label)) % len(self._MARK_PALETTE)]

    def _refresh_mark_labels(self):
        """Populate the 'Mark as' picker with saved behaviors (plus the current
        one), keeping whatever the user has selected/typed."""
        cur = self.mark_label_picker.currentText()
        names = list(self.state.library.list())
        if self.current and self.current.name not in names:
            names.insert(0, self.current.name)
        self.mark_label_picker.blockSignals(True)
        self.mark_label_picker.clear()
        self.mark_label_picker.addItems(names)
        if cur:
            self.mark_label_picker.setCurrentText(cur)
        elif self.current:
            self.mark_label_picker.setCurrentText(self.current.name)
        self.mark_label_picker.blockSignals(False)
        self._on_mark_label(self.mark_label_picker.currentText())

    def _on_mark_label(self, label: str):
        self.timeline.set_active_label(label, self._label_color(label))

    def _clear_marks_clicked(self):
        from PyQt6.QtWidgets import QApplication
        mods = QApplication.keyboardModifiers()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            self.timeline.clear_marks()                       # all labels
        else:
            self.timeline.clear_marks(self.mark_label_picker.currentText())

    # -- library -------------------------------------------------------------

    def _refresh_library(self):
        # Populate ONLY the dropdown from the library. Do NOT load every saved
        # behavior into the active set: the ethogram and overlay show the
        # behavior you are currently editing, and dumping the whole library in
        # made same-coloured behaviors stack into one solid bar that hid the real
        # detection. The library is for save/load; `state.behaviors` holds just
        # what is being worked on.
        self.lib_picker.blockSignals(True)
        self.lib_picker.clear()
        self.lib_picker.addItems(self.state.library.list())
        self.lib_picker.blockSignals(False)
        if hasattr(self, "mark_label_picker"):
            self._refresh_mark_labels()

    def _new(self):
        name, ok = QInputDialog.getText(self, "New behavior", "Name:")
        if not ok or not name:
            return
        self.current = Behavior(name=name, spec=LogicNode(op="and"))
        self._sync_from_behavior()
        self._recompute()

    def _add_example(self):
        if not self.state.has_cache:
            QMessageBox.information(self, "Need a cache", "Open a cache first.")
            return
        band = self.state.band_features[0] if self.state.band_features \
            else self._suggest_band_name()
        b = default_wingbeat(self.state.fps, band)
        self.state.library.save(b)
        self._refresh_library()
        self.lib_picker.setCurrentText(b.name)
        QMessageBox.information(
            self, "Wingbeat example added",
            "Starting point only. Select a replicate that IS wingbeating, then "
            "drag the band-power line down until only its bouts stay lit — you "
            "will see the detection appear on the video and in the ethogram as "
            "you drag.")

    def _suggest_band_name(self) -> str:
        b = self.state.cfg.features.suggest_band(self.state.fps)
        return f"bandpower_{b.lo_hz:g}-{b.hi_hz:g}Hz"

    def _load_behavior(self, name: str):
        if not name:
            return
        try:
            self.current = self.state.library.load(name)
        except Exception as e:
            QMessageBox.critical(self, "Could not load", str(e))
            return
        self._sync_from_behavior()
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
        if name and QMessageBox.question(self, "Delete", f"Delete '{name}'?") \
                == QMessageBox.StandardButton.Yes:
            self.state.library.delete(name)
            self.current = None
            self._refresh_library()
            self._rebuild_plots()
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
            self.color_btn.setStyleSheet(f"background:{c.name()}; color:#000;")
            for pl in self.plots.values():
                pl.accent = c
            self._recompute()

    # -- leaves (constraints) ------------------------------------------------

    def _leaves(self) -> list[RangeLeaf]:
        if not self.current:
            return []
        return [c for c in self.current.spec.children
                if isinstance(c, RangeLeaf)]

    def _add_constraint(self):
        if not self.current:
            QMessageBox.information(self, "No behavior",
                                    "Make or load a behavior first (New).")
            return
        if not self.state.has_cache:
            return
        feats = self.state.available_features()
        labels = [self._CUSTOM_BAND] + [
            REGISTRY[f].label if f in REGISTRY else f for f in feats]
        label, ok = QInputDialog.getItem(self, "Add constraint", "Feature:",
                                         labels, 0, False)
        if not ok:
            return
        if label == self._CUSTOM_BAND:
            name = self._ask_custom_band()
            if name is None:
                return
        else:
            name = feats[labels.index(label) - 1]

        if any(l.feature == name for l in self._leaves()):
            self._set_enabled(name, True)
            self._recompute()
            return

        # Grabbable default: a third of the way up the (shared) axis, greater-than.
        # Off the axis, not a data percentile, so it never crushes against the top
        # for a feature whose values pile up near one edge (coherence, etc.).
        lo_ax, hi_ax = self._pooled_yrange(name)
        lo = lo_ax + 0.33 * (hi_ax - lo_ax)
        self.current.spec.children.append(
            RangeLeaf(feature=name, lo=lo, hi=float("inf"), enabled=True))
        self._rebuild_plots()
        self._recompute()

    def _ask_custom_band(self) -> str | None:
        nyq = self.state.fps / 2
        lo, ok1 = QInputDialog.getDouble(
            self, "Custom band", f"Low edge (Hz), Nyquist = {nyq:.1f}:",
            12.0, 0.0, nyq, 2)
        if not ok1:
            return None
        hi, ok2 = QInputDialog.getDouble(
            self, "Custom band", f"High edge (Hz), Nyquist = {nyq:.1f}:",
            min(24.0, nyq), lo, nyq, 2)
        if not ok2:
            return None
        return f"bandpower_{lo:g}-{hi:g}Hz"

    def _remove_constraint(self, feature: str):
        """Right-click removes a custom band entirely; a standard measure is just
        disabled (it stays in the panel so you can re-enable it)."""
        if not self.current:
            return
        from core.features import _parse_band
        if _parse_band(feature) and feature not in self.state.band_features:
            self.current.spec.children = [
                c for c in self.current.spec.children
                if not (isinstance(c, RangeLeaf) and c.feature == feature)]
            self._rebuild_plots()
        else:
            self._set_enabled(feature, False)
        self._recompute()

    def _on_leaf_range(self, feature: str, lo: float, hi: float):
        for l in self._leaves():
            if l.feature == feature:
                l.lo, l.hi = lo, hi
        self._recompute()

    def _set_enabled(self, feature: str, on: bool):
        for l in self._leaves():
            if l.feature == feature:
                l.enabled = on
        if feature in self.plots:
            self.plots[feature].set_active(on)
        if feature in self.plot_checks:
            cb = self.plot_checks[feature]
            cb.blockSignals(True)
            cb.setChecked(on)
            cb.blockSignals(False)

    def _on_measure_toggled(self, feature: str, checked: bool):
        self._set_enabled(feature, checked)
        self._recompute()

    # -- plots (the all-measures panel) --------------------------------------

    def _measure_order(self) -> list[str]:
        """Every measure to show, most useful for behavior first, then whatever
        band powers this cache holds, then any custom bands already on the
        behavior."""
        base = list(self.MEASURES)
        for b in self.state.band_features:
            if b not in base:
                base.append(b)
        for l in self._leaves():
            if l.feature not in base:
                base.append(l.feature)
        return [m for m in base if m in self.state.available_features()
                or m in [l.feature for l in self._leaves()]]

    def _ensure_measure_leaves(self):
        """Make sure the behavior has a leaf for every shown measure, so toggling
        one on/off just flips its `enabled`. Missing ones are added DISABLED with
        a grabbable default range (a third of the way up, greater-than)."""
        if not self.current:
            return
        have = {l.feature for l in self._leaves()}
        for feature in self._measure_order():
            if feature in have:
                continue
            lo, hi = self._pooled_yrange(feature)
            default_lo = lo + 0.33 * (hi - lo)
            self.current.spec.children.append(
                RangeLeaf(feature=feature, lo=default_lo, hi=float("inf"),
                          enabled=False))

    def _rebuild_plots(self):
        for w in getattr(self, "plot_rows", {}).values():
            self.plot_lay.removeWidget(w)
            w.deleteLater()
        self.plots = {}
        self.plot_rows: dict[str, QWidget] = {}
        self.plot_checks: dict[str, QCheckBox] = {}
        if not self.current:
            return
        self._ensure_measure_leaves()
        color = self.current.color
        leaf_by_feature = {l.feature: l for l in self._leaves()}
        for feature in self._measure_order():
            leaf = leaf_by_feature.get(feature)
            if leaf is None:
                continue
            spec = REGISTRY.get(feature)
            title = spec.label if spec else feature

            row = QWidget()
            rlay = QVBoxLayout(row)
            rlay.setContentsMargins(0, 0, 0, 0)
            rlay.setSpacing(1)
            cb = QCheckBox(title)
            cb.setChecked(leaf.enabled)
            cb.setStyleSheet("font-weight:bold;")
            cb.toggled.connect(
                lambda on, f=feature: self._on_measure_toggled(f, on))
            rlay.addWidget(cb)

            pl = RangePlot(feature, title, color=color)
            pl.set_range(leaf.lo, leaf.hi)
            pl.set_active(leaf.enabled)
            pl.range_changed.connect(self._on_leaf_range)
            pl.remove_requested.connect(self._remove_constraint)
            pl.seek_requested.connect(
                lambda t: self.state.set_frame(int(t * self.state.fps)))
            rlay.addWidget(pl)

            self.plot_lay.insertWidget(self.plot_lay.count() - 1, row)
            self.plots[feature] = pl
            self.plot_rows[feature] = row
            self.plot_checks[feature] = cb
        self._refresh_plots()

    def _refresh_plots(self):
        roi = self._selected_roi()
        if roi is None or not self.state.has_cache:
            return
        t = self.state.ctx.times_s()
        bands = []
        if self.current:
            tr = self.state.traces.get((roi.roi_id, self.current.name))
            if tr is not None:
                bands = [(b.start_s, b.end_s, self.current.color)
                         for b in trace_to_bouts(tr, self.state.fps)]
        leaf_by_feature = {l.feature: l for l in self._leaves()}
        for feature, pl in self.plots.items():
            lo, hi = self._pooled_yrange(feature)
            pl.set_yrange(lo, hi)
            try:
                pl.set_blocks(t, self.state.roi_blocks(roi, feature))
            except KeyError:
                continue
            leaf = leaf_by_feature.get(feature)
            if leaf is not None:
                pl.set_range(leaf.lo, leaf.hi)
                pl.set_active(leaf.enabled)
            pl.set_behavior_bands(bands)
            pl.set_cursor(self.state.current_frame / self.state.fps)

    def _pooled_yrange(self, feature: str) -> tuple[float, float]:
        """A y-axis shared by all replicates for this feature: pooled minimum to
        a high percentile (robust to the heavy tail of band power). Cached per
        feature so switching replicates does not recompute it."""
        cache = getattr(self, "_yrange_cache", None)
        if cache is None:
            cache = self._yrange_cache = {}
        if feature in cache:
            return cache[feature]
        try:
            # Pool the PER-BLOCK values across replicates, so the axis spans the
            # block distribution the range is set against, not the box means.
            pooled = np.concatenate(
                [self.state.roi_blocks(r, feature).ravel()
                 for r in self.state.rois])
        except KeyError:
            return (0.0, 1.0)
        if pooled.size == 0:
            return (0.0, 1.0)
        lo = float(np.min(pooled))
        hi = float(np.percentile(pooled, 99.5))
        # Keep any finite constraint edge visible even if it sits above p99.5.
        for l in self._leaves():
            if l.feature == feature and np.isfinite(l.hi):
                hi = max(hi, float(l.hi))
        if hi <= lo:
            hi = lo + 1.0
        cache[feature] = (lo, hi)
        return cache[feature]

    # -- behavior <-> UI -----------------------------------------------------

    def _sync_from_behavior(self):
        if not self.current:
            return
        self.name_edit.setText(self.current.name)
        self.color_btn.setStyleSheet(
            f"background:{self.current.color}; color:#000;")
        for w, v in ((self.min_dur, self.current.criteria.min_duration_s),
                     (self.max_gap, self.current.criteria.max_gap_s),
                     (self.smooth, self.current.criteria.smooth_s),
                     (self.min_blocks, self.current.spatial.min_blocks),
                     (self.min_fraction, self.current.spatial.min_fraction),
                     (self.merge_dist, self.current.spatial.merge_distance)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        # Default new ground-truth marks to the behavior you are now editing, so
        # middle-drag "just works" for the obvious case (mark Flying while on
        # Flying). The user can still switch the 'Mark as' label to something else.
        if hasattr(self, "mark_label_picker"):
            self.mark_label_picker.setCurrentText(self.current.name)
        self._rebuild_plots()

    def _criteria(self) -> TemporalCriteria:
        return TemporalCriteria(min_duration_s=self.min_dur.value(),
                                max_gap_s=self.max_gap.value(),
                                smooth_s=self.smooth.value())

    def _spatial(self) -> SpatialCriteria:
        return SpatialCriteria(min_blocks=self.min_blocks.value(),
                               min_fraction=self.min_fraction.value(),
                               merge_distance=self.merge_dist.value())

    def _on_criteria_changed(self):
        if self.current:
            self.current.criteria = self._criteria()
            self._recompute()

    def _on_spatial_changed(self):
        if self.current:
            self.current.spatial = self._spatial()
            self._recompute()

    # -- selection -----------------------------------------------------------

    def _selected_roi(self):
        if self.state.selected_roi is not None:
            r = self.state.roi_by_id(self.state.selected_roi)
            if r is not None:
                return r
        return self.state.rois[0] if self.state.rois else None

    def _selected_series(self, feature: str) -> np.ndarray:
        roi = self._selected_roi()
        if roi is None:
            return np.zeros(0, np.float32)
        try:
            return self.state.roi_series(roi, feature)
        except KeyError:
            return np.zeros(0, np.float32)

    def _select_roi(self, roi_id: int):
        self.state.selected_roi = roi_id
        self._refresh_plots()
        self._refresh_inspector()
        self._refresh_timeline()

    def _on_block_clicked(self, by: int, bx: int):
        for r in self.state.rois:
            if r.mask is not None and r.mask[by, bx]:
                self._select_roi(r.roi_id)
                return

    def _on_rois_changed(self):
        # Replicates changed in Tab 2. Their series (and the pooled y-range) are
        # now stale.
        self._yrange_cache = {}
        if self.state.selected_roi is None and self.state.rois:
            self.state.selected_roi = self.state.rois[0].roi_id
        self._recompute()

    def _on_frame_changed(self, idx: int):
        if not self.isVisible():
            return
        t = idx / self.state.fps
        self.timeline.set_cursor(t)
        for pl in self.plots.values():
            pl.set_cursor(t)
        self.psd  # no-op
        self._update_overlay()
        self._refresh_constraints()

    # -- evaluation ----------------------------------------------------------

    def _recompute(self):
        if not self.state.has_cache:
            return
        # The active set IS the current behavior -- nothing else is evaluated or
        # drawn. This is what keeps the ethogram showing one clean signal.
        if self.current:
            self.current.criteria = self._criteria()
            self.current.spatial = self._spatial()
            self.state.behaviors = [self.current]
        else:
            self.state.behaviors = []
        self.state.recompute_traces()
        self._refresh_plots()
        self._refresh_timeline()
        self._refresh_inspector()
        self._update_overlay()

    def _refresh_timeline(self):
        rows = []
        for r in self.state.rois:
            beh = {}
            for b in self.state.behaviors:
                tr = self.state.traces.get((r.roi_id, b.name))
                if tr is not None:
                    strength = self.state.strengths.get((r.roi_id, b.name))
                    beh[b.name] = (tr, strength, b.color)
            rows.append((r.roi_id, r.note or f"#{r.roi_id}", beh))
        self.timeline.selected_roi = self.state.selected_roi
        self.timeline.set_rows(
            rows, self.state.cache.n_frames / self.state.fps
            if self.state.has_cache else 1.0)

    def _update_overlay(self):
        """Paint every block satisfying the whole behavior at this frame, in the
        behavior colour, plus thin outlines of the replicate boxes."""
        if not self.isVisible() or not self.state.has_cache:
            return
        if self.current and self._leaves():
            mask = behavior_block_mask(self.state.cache, self.state.ctx,
                                       self.current, self.state.current_frame)
            self.video.set_overlay(mask, _hex_to_bgr_tuple(self.current.color))
        else:
            self.video.set_overlay(None)
        # Replicate outlines so you can see which box is which.
        self.video.set_roi_boxes([
            (r.roi_id, r.bbox,
             "#ffcc33" if r.roi_id == self.state.selected_roi else "#8899aa")
            for r in self.state.rois])

    def _refresh_inspector(self):
        roi = self._selected_roi()
        if roi is None or not self.state.has_cache:
            self.roi_label.setText("No replicate selected.")
            return
        fps = self.state.fps
        self.roi_label.setText(
            f"{roi.note or ('#' + str(roi.roi_id))} — "
            f"{int(roi.mask.sum())} blocks")
        freqs, psd = roi_psd(self.state.cache, self.state.ctx, roi, "speed")
        band = None
        if self.current:
            for l in self._leaves():
                b = None
                from core.features import _parse_band
                b = _parse_band(l.feature)
                if b:
                    band = b
                    break
        self.psd.set_psd(freqs, psd, nyquist=fps / 2, band=band)

        if self.current:
            tr = self.state.traces.get((roi.roi_id, self.current.name))
            if tr is not None:
                bouts = trace_to_bouts(tr, fps)
                total = sum(b.duration_s for b in bouts)
                self.bout_label.setText(
                    f"{self.current.name}: {len(bouts)} bouts, {total:.2f} s total"
                    + (f", mean {np.mean([b.duration_s for b in bouts]):.2f} s"
                       if bouts else " — nothing detected yet"))
        self._refresh_constraints()

    def _refresh_constraints(self):
        roi = self._selected_roi()
        if roi is None or not self.current or not self.state.has_cache:
            return
        feats = sorted(self.current.features())
        try:
            # Report the box-MAX (the peak block), not the box mean. Detection and
            # the plot's threshold both key on per-block values -- a "detect high"
            # range fires when the peak block clears it -- so the mean would show a
            # much smaller number than the line the user set, reading FAIL while the
            # box is genuinely detected. Max is the value the range is set against.
            def _box_max(f):
                b = self.state.roi_blocks(roi, f)
                return b.max(axis=1) if b.shape[1] else np.zeros(b.shape[0], np.float32)
            series = {f: _box_max(f) for f in feats}
        except KeyError:
            return
        rows = self.current.constraint_status(series, self.state.current_frame)
        self.constraints.set_status(rows, self.current.name)

    # -- export --------------------------------------------------------------

    def _check_ready(self) -> bool:
        if not self.state.has_cache or not self.state.rois:
            QMessageBox.information(self, "Nothing to export",
                                    "Draw replicate boxes in Tab 2 first.")
            return False
        return True

    def _features_for_export(self) -> list[str]:
        base = ["speed", "net_speed", "coherence", "rolling_std_speed"]
        if self.current:
            base += [l.feature for l in self._leaves()]
        base += self.state.band_features
        seen, out = set(), []
        for f in base:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out

    def _export_ts(self):
        if not self._check_ready():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export time series", "roi_timeseries.csv", "CSV (*.csv)")
        if path:
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
