"""Tab 2: ROI discovery.

Video with a live matched-block overlay on the left; the cross-filtered histogram
stack in the middle; the ROI list and inspector on the right.

The interaction loop this is built around: drag a range -> the overlay repaints
immediately -> every OTHER histogram redraws to show only what survives -> you
can see at once whether the intersection is isolating something or just eating
your sample.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox,
                             QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QMessageBox, QProgressDialog, QPushButton,
                             QScrollArea, QSpinBox, QSplitter, QVBoxLayout,
                             QWidget)

from core.features import REGISTRY
from core.filters import all_frame_masks, frame_mask
from core.roi import (ROIParams, blocks_to_pixels, extract_rois, roi_psd,
                      roi_time_series, save_rois)
from gui._shelved.histogram_widget import RangeHistogram
from gui.inspector import PSDPlot, TimeSeriesPlot
from gui.state import AppState
from gui.video_panel import VideoPanel

# Opened by default: the three that between them separate "moving" from
# "oscillating in place", which is the discovery question this tab exists to
# answer.
DEFAULT_HISTOGRAMS = ["speed", "coherence", "net_speed"]


class Tab2ROI(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.hists: dict[str, RangeHistogram] = {}

        split = QSplitter(Qt.Orientation.Horizontal)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(split)

        # -- left: video -----------------------------------------------------
        self.video = VideoPanel(state)
        self.video.block_clicked.connect(self._on_block_clicked)
        split.addWidget(self.video)

        # -- middle: histograms ---------------------------------------------
        mid = QWidget()
        mid_lay = QVBoxLayout(mid)
        mid_lay.setContentsMargins(4, 4, 4, 4)

        add_row = QHBoxLayout()
        self.feature_picker = QComboBox()
        add_btn = QPushButton("Add histogram")
        add_btn.clicked.connect(self._add_selected_feature)
        add_row.addWidget(self.feature_picker, 1)
        add_row.addWidget(add_btn)
        mid_lay.addLayout(add_row)

        hint = QLabel(
            "Drag an edge to move it · drag inside to slide the band · scroll to "
            "widen/narrow · right-click resets · double-click disables · "
            "middle-click removes")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:10px;")
        mid_lay.addWidget(hint)

        self.hist_scroll = QScrollArea()
        self.hist_scroll.setWidgetResizable(True)
        self.hist_container = QWidget()
        self.hist_lay = QVBoxLayout(self.hist_container)
        self.hist_lay.setSpacing(10)
        self.hist_lay.addStretch(1)
        self.hist_scroll.setWidget(self.hist_container)
        mid_lay.addWidget(self.hist_scroll, 1)

        self.match_label = QLabel("")
        self.match_label.setStyleSheet(
            "font-family: Consolas; color:#bbb; font-size:11px;")
        mid_lay.addWidget(self.match_label)

        split.addWidget(mid)

        # -- right: ROI extraction + list + inspector ------------------------
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(4, 4, 4, 4)

        params_box = QGroupBox("ROI criteria")
        pl = QVBoxLayout(params_box)

        self.min_area = QSpinBox()
        self.min_area.setRange(1, 10000)
        self.min_area.setValue(2)
        self.min_area.setPrefix("min area  ")
        self.min_area.setSuffix(" blocks")
        self.min_area.setToolTip(
            "Smallest connected region to keep, in blocks. A wingbeating "
            "grasshopper occupies only 1-4 blocks at the default settings, so "
            "raising this will discard exactly the behavior you are hunting.")

        self.min_dur = QDoubleSpinBox()
        self.min_dur.setRange(0.0, 60.0)
        self.min_dur.setValue(0.5)
        self.min_dur.setPrefix("min duration  ")
        self.min_dur.setSuffix(" s")

        self.max_gap = QDoubleSpinBox()
        self.max_gap.setRange(0.0, 10.0)
        self.max_gap.setValue(0.2)
        self.max_gap.setPrefix("bridge gaps up to  ")
        self.max_gap.setSuffix(" s")

        self.open_r = QSpinBox()
        self.open_r.setRange(0, 5)
        self.open_r.setValue(0)
        self.open_r.setPrefix("morph open r  ")
        self.open_r.setToolTip(
            "Off by default, on purpose. Opening with radius r erodes with a "
            "(2r+1)² element, so it DELETES any region smaller than that — "
            "radius 1 wipes out everything under 3×3 blocks, which is most "
            "small-extent behaviors. Turn it on only if the overlay is speckled "
            "with isolated blocks you know are spurious.")

        self.close_r = QSpinBox()
        self.close_r.setRange(0, 5)
        self.close_r.setValue(1)
        self.close_r.setPrefix("morph close r  ")
        self.close_r.setToolTip(
            "Fills small holes inside a region. Safe — it does not shrink "
            "regions the way opening does.")

        for w in (self.min_area, self.min_dur, self.max_gap, self.open_r,
                  self.close_r):
            pl.addWidget(w)

        self.extract_btn = QPushButton("Extract ROIs from current filter")
        self.extract_btn.clicked.connect(self._extract)
        pl.addWidget(self.extract_btn)
        right_lay.addWidget(params_box)

        self.roi_list = QListWidget()
        self.roi_list.currentItemChanged.connect(self._on_roi_selected)
        right_lay.addWidget(QLabel("ROIs"))
        right_lay.addWidget(self.roi_list, 1)

        io_row = QHBoxLayout()
        save_btn = QPushButton("Export ROIs")
        save_btn.clicked.connect(self._export)
        expand_btn = QPushButton("Can't isolate it? Expand cache →")
        expand_btn.setToolTip(
            "Go back to Tab 1 to add cached features and re-run.")
        expand_btn.clicked.connect(lambda: self.state.request_tab.emit(0))
        io_row.addWidget(save_btn)
        io_row.addWidget(expand_btn)
        right_lay.addLayout(io_row)

        insp = QGroupBox("ROI inspector")
        il = QVBoxLayout(insp)
        self.ts_speed = TimeSeriesPlot("speed")
        self.ts_coh = TimeSeriesPlot("coherence")
        self.ts_band = TimeSeriesPlot("band power")
        self.psd = PSDPlot()
        for w in (self.ts_speed, self.ts_coh, self.ts_band):
            w.seek_requested.connect(
                lambda t: self.state.set_frame(int(t * self.state.fps)))
            il.addWidget(w)
        il.addWidget(self.psd)
        right_lay.addWidget(insp)

        split.addWidget(right)
        # Video gets half the window; the histogram stack and the ROI panel
        # split the rest. Qt distributes by these ratios, not by pixels.
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 1)
        split.setSizes([900, 450, 450])

        state.cache_opened.connect(self._on_cache_opened)
        state.frame_changed.connect(self._on_frame_changed)
        state.filter_changed.connect(self._refresh_histograms)

    # -- setup ---------------------------------------------------------------

    def _on_cache_opened(self):
        self.feature_picker.clear()
        for name in self.state.available_features():
            spec = REGISTRY.get(name)
            label = spec.label if spec else name
            self.feature_picker.addItem(label, name)

        for name in list(self.hists):
            self._remove_histogram(name)
        self.state.filter.ranges.clear()

        for name in DEFAULT_HISTOGRAMS:
            if name in self.state.available_features():
                self._add_histogram(name)
        for name in self.state.band_features:
            self._add_histogram(name)

        self.roi_list.clear()
        self._refresh_histograms()
        self._update_overlay()

    def _add_selected_feature(self):
        name = self.feature_picker.currentData()
        if name and name not in self.hists:
            self._add_histogram(name)
            self._refresh_histograms()

    def _add_histogram(self, name: str):
        if not self.state.sampler or name in self.hists:
            return
        spec = REGISTRY.get(name)
        h = RangeHistogram(name, spec.label if spec else name,
                           spec.units if spec else "")
        if spec and spec.help:
            h.setToolTip(spec.help)

        edges = self.state.sampler.edges(name)
        h.set_range(float(edges[0]), float(edges[-1]))
        self.state.filter.set_range(name, float(edges[0]), float(edges[-1]))

        h.range_changed.connect(self._on_range_changed)
        h.enabled_changed.connect(self._on_enabled_changed)
        h.remove_requested.connect(self._remove_histogram)

        self.hist_lay.insertWidget(self.hist_lay.count() - 1, h)
        self.hists[name] = h

    def _remove_histogram(self, name: str):
        h = self.hists.pop(name, None)
        if h is None:
            return
        self.hist_lay.removeWidget(h)
        h.deleteLater()
        self.state.filter.ranges.pop(name, None)
        self._refresh_histograms()

    # -- filter --------------------------------------------------------------

    def _on_range_changed(self, name: str, lo: float, hi: float):
        self.state.filter.set_range(name, lo, hi)
        self._refresh_histograms()
        self._update_overlay()

    def _on_enabled_changed(self, name: str, on: bool):
        if name in self.state.filter.ranges:
            self.state.filter.ranges[name].enabled = on
        self._refresh_histograms()
        self._update_overlay()

    def _refresh_histograms(self):
        s = self.state.sampler
        if not s:
            return
        for name, h in self.hists.items():
            edges, total, filtered = s.histogram(name, self.state.filter)
            h.set_data(edges, total, filtered)
        frac = s.match_fraction(self.state.filter)
        self.match_label.setText(
            f"{frac * 100:.4f}% of block-frames match all active ranges "
            f"({int(frac * s.n):,} of {s.n:,} sampled)")

    def _update_overlay(self):
        if not self.state.has_cache:
            return
        m = frame_mask(self.state.cache, self.state.ctx, self.state.filter,
                       self.state.current_frame)
        self.video.set_overlay(m)

    def _on_frame_changed(self, idx: int):
        # A hidden tab still receives frame_changed. Repainting an overlay nobody
        # is looking at is pure cost on every scrub step, and both tabs doing it
        # doubles the per-frame budget.
        if not self.isVisible():
            return
        self._update_overlay()
        t = idx / self.state.fps
        for w in (self.ts_speed, self.ts_coh, self.ts_band):
            w.set_cursor(t)

    # -- ROI extraction ------------------------------------------------------

    def _extract(self):
        if not self.state.has_cache:
            return
        params = ROIParams(
            min_area_blocks=self.min_area.value(),
            open_radius=self.open_r.value(),
            close_radius=self.close_r.value(),
            min_duration_s=self.min_dur.value(),
            max_gap_s=self.max_gap.value(),
        )
        self.state.roi_params = params

        dlg = QProgressDialog("Extracting ROIs...", "Cancel", 0, 100, self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setValue(0)

        masks = all_frame_masks(self.state.cache, self.state.ctx,
                                self.state.filter)

        def prog(done, total):
            dlg.setValue(int(done / max(1, total) * 100))

        rois = extract_rois(masks, self.state.fps, params,
                            previous=self.state.rois or None, progress=prog)
        dlg.setValue(100)

        self.state.rois = rois
        self.state.invalidate_series()
        self.state.rois_changed.emit()
        self._refresh_roi_list()
        self.state.status.emit(f"{len(rois)} ROIs extracted.")

    def _refresh_roi_list(self):
        self.roi_list.clear()
        fps = self.state.fps
        for r in self.state.rois:
            item = QListWidgetItem(
                f"#{r.roi_id:3d}   {r.duration_s(fps):6.2f} s   "
                f"{int(r.mask.sum()):4d} blocks   @{r.first_s(fps):.1f}s")
            item.setData(Qt.ItemDataRole.UserRole, r.roi_id)
            self.roi_list.addItem(item)
        self.video.set_roi_boxes(
            [(r.roi_id, r.bbox, "#57d2ff") for r in self.state.rois])

    def _on_roi_selected(self, item):
        if item is None:
            return
        roi_id = item.data(Qt.ItemDataRole.UserRole)
        self.state.selected_roi = roi_id
        roi = self.state.roi_by_id(roi_id)
        if roi is None:
            return

        cache, ctx = self.state.cache, self.state.ctx
        t = ctx.times_s()
        self.ts_speed.set_series(t, roi_time_series(cache, ctx, roi, "speed"),
                                 "speed (px/s)")
        self.ts_coh.set_series(t, roi_time_series(cache, ctx, roi, "coherence"),
                               "coherence")
        bands = self.state.band_features
        if bands:
            self.ts_band.set_series(
                t, roi_time_series(cache, ctx, roi, bands[0]), bands[0])

        freqs, psd = roi_psd(cache, ctx, roi, "speed")
        band = None
        if self.state.cfg.features.bands:
            b = self.state.cfg.features.bands[0]
            band = (b.lo_hz, b.hi_hz)
        self.psd.set_psd(freqs, psd, nyquist=self.state.fps / 2, band=band)

        self.video.set_roi_boxes(
            [(r.roi_id, r.bbox, "#ffcc33" if r.roi_id == roi_id else "#57d2ff")
             for r in self.state.rois])

    def _on_block_clicked(self, by: int, bx: int):
        """Click the video to select whichever ROI owns that block."""
        for r in self.state.rois:
            if r.mask is not None and r.mask[by, bx]:
                for i in range(self.roi_list.count()):
                    it = self.roi_list.item(i)
                    if it.data(Qt.ItemDataRole.UserRole) == r.roi_id:
                        self.roi_list.setCurrentItem(it)
                        return
        self.state.status.emit(f"No ROI at block ({by}, {bx}).")

    def _export(self):
        if not self.state.rois:
            QMessageBox.information(self, "Nothing to export", "Extract ROIs first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export ROIs", "rois.json", "JSON (*.json)")
        if not path:
            return
        save_rois(path, self.state.rois, self.state.fps,
                  self.state.cache.block_size)
        self.state.status.emit(
            f"Exported {len(self.state.rois)} ROIs to {os.path.basename(path)} "
            f"(+ mask PNGs).")
