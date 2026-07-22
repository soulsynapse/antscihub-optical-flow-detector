"""Replicates tab: manual experimental-unit bounding boxes.

The histogram-driven automatic ROI discovery was retired -- it worked, but on
footage where the behavior is everywhere and the camera moves, drawing the
replicate regions by hand is faster
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

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QKeyEvent, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QGroupBox, QHBoxLayout, QInputDialog,
                             QLabel, QListWidget, QListWidgetItem, QMessageBox,
                             QProgressBar, QPushButton, QVBoxLayout, QWidget)

from core.pretranscode import (DEFAULT_QUALITY, QUALITY_PRESETS,
                               PretranscodeError, manifest_path_for,
                               read_manifest, verify_manifest)

from core.replicate_home import (describe_home, ensure_home, list_generations,
                                 replicate_dir, restore_generation,
                                 retire_current, sync_homes)
from gui.state import AppState
from gui.replicate_split import ReplicateSplitWorker
from gui.video_panel import VideoPanel

# A fixed palette so successive replicates are visually distinct.
_PALETTE = ["#ff5a5a", "#4ac6ff", "#ffd24a", "#6ee06e", "#c78bff", "#ff9d3a",
            "#00d0b0", "#ff6ec7"]

# The exact words the user has to click to move a box that downstream work has
# already consumed. Spelled out rather than "OK" because that is the whole point
# of the gate: the cost is not the drag, it is that everything measured against
# the old rectangle silently stops describing the new one. Same hazard class as
# T17 and the count-band re-denomination -- state that quietly stops meaning what
# it meant -- and like those it cannot be converted.
#
# "retire", not "lose" and not "keep": the results are neither destroyed nor
# carried forward. They move into old_NNN/ with the rectangle they were measured
# against, stay drawn where they were, and can be restored. Promising loss would
# talk a user out of a recoverable action; promising they still apply would
# attribute one animal's labelled behaviour to whatever the box now covers.
_MOVE_ANYWAY = "Move it and retire the existing results"


class Tab2Replicates(QWidget):
    split_running_changed = pyqtSignal(bool)

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        # Each replicate: {"id", "label", "frac": (x0,y0,x1,y1), "color",
        # "processed"}. `processed` means downstream work has consumed this box,
        # so moving it is gated (_confirm_move). It is always present on a box
        # this session created, but readers must still use .get -- sidecars
        # written before the field existed do not carry it.
        self.replicates: list[dict] = []
        self._next_id = 1
        # Retired geometries of the CURRENT boxes, read from disk by
        # _refresh_retired. Drawn as dashed ghosts and never interactive; disk is
        # the record, this is only what the last listing found.
        self._retired: list[dict] = []

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
            "The stamp is always the SELECTED box's size. "
            "A 🔒 box has been processed — moving it asks first, and retires its "
            "existing results rather than re-aiming them: they stay drawn as a "
            "dashed OLD box and come back from 'Older geometries…'. Each box "
            "owns a folder beside the video that keeps its results.")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#333; font-size:11px;")
        ll.addWidget(self.hint)

        lay.addWidget(left, 1)   # stretch 1 of 2 == half the window

        # -- right: list + inspector -----------------------------------------
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)

        box = self.box_group = QGroupBox("Replicate boxes")
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

        hist = QPushButton("Older geometries…")
        hist.setToolTip(
            "Moving a box retires its results rather than deleting them. This "
            "lists the selected replicate's retired rectangles and restores one "
            "— moving the box back and bringing its detections with it.")
        hist.clicked.connect(self._show_generations)
        bl.addWidget(hist)

        row2 = QHBoxLayout()
        save = QPushButton("Save boxes…")
        save.clicked.connect(self._save)
        load = QPushButton("Load boxes…")
        load.clicked.connect(self._load)
        row2.addWidget(save)
        row2.addWidget(load)
        bl.addLayout(row2)

        self.to_live = QPushButton("Tune these in Preprocessing →")
        self.to_live.setStyleSheet("padding:6px; font-weight:bold;")
        self.to_live.clicked.connect(self._go_preprocess)
        bl.addWidget(self.to_live)
        rl.addWidget(box)

        split_box = QGroupBox("Replicate clips (optional)")
        split_lay = QVBoxLayout(split_box)
        split_help = QLabel(
            "Split the source once into one full-resolution crop per replicate. "
            "This can make later decoding cheaper; it does not resize pixels or "
            "change the boxes. Clip use remains opt-in in Preprocessing.")
        split_help.setWordWrap(True)
        split_lay.addWidget(split_help)

        split_row = QHBoxLayout()
        self.split_quality = QComboBox()
        for key, label in (("high", "High · CRF 12"),
                           ("standard", "Standard · CRF 18"),
                           ("lossless", "Lossless · FFV1")):
            if key in QUALITY_PRESETS:
                self.split_quality.addItem(label, key)
        default_idx = self.split_quality.findData(DEFAULT_QUALITY)
        self.split_quality.setCurrentIndex(max(0, default_idx))
        self.split_quality.setToolTip(
            "Quality changes storage and pixel provenance, not crop geometry. "
            "High is the project default. Lossless preserves the encoded 8-bit "
            "crop but commonly uses more total disk than the source.")
        split_row.addWidget(self.split_quality, 1)
        self.split_btn = QPushButton("Create replicate clips…")
        self.split_btn.clicked.connect(self._split_or_cancel)
        split_row.addWidget(self.split_btn)
        split_lay.addLayout(split_row)

        self.split_progress = QProgressBar()
        self.split_progress.setRange(0, 1000)
        self.split_progress.setTextVisible(True)
        self.split_progress.setFormat("%p%")
        self.split_progress.hide()
        split_lay.addWidget(self.split_progress)
        self.split_status = QLabel("Open a video and draw boxes to create clips.")
        self.split_status.setWordWrap(True)
        split_lay.addWidget(self.split_status)
        rl.addWidget(split_box)
        self._split_worker: ReplicateSplitWorker | None = None

        std_box = self.std_box = QGroupBox("Selected replicate standardization")
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
        rl.addStretch(1)

        lay.addWidget(right, 1)

        state.video_loaded.connect(self._on_video_loaded)
        state.calibration_changed.connect(self._on_calibration_changed)
        self._refresh_split_status()

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
                self._next_id = self._resume_next_id(data)
            except (OSError, ValueError) as e:
                self.state.status.emit(f"Could not read boxes: {e}")
        # _rebuild_rois refreshes the retired listing, so the ghosts _redraw_boxes
        # paints are this clip's. A new clip's homes may already hold generations
        # from an earlier session -- that persistence is the whole reason the
        # history lives on disk rather than in an undo stack.
        self._rebuild_rois()
        self._refresh_list()
        self._redraw_boxes()
        # Boxes restored from a sidecar written before homes existed have none.
        # Created on load rather than lazily on first write, so the directory a
        # user is told about in a warning is one they can already see.
        sync_homes(self._video_path(), self.replicates)

    def _resume_next_id(self, data: dict) -> int:
        """The next id to hand out, never reusing one that has been retired.

        ``max(existing ids) + 1`` alone -- what this used to be -- silently
        recycles: delete the highest box, reopen the clip, and the next box drawn
        reclaims that id. With per-replicate homes (``core.replicate_home``) that
        is no longer merely confusing, it is wrong: the new box inherits the
        deleted animal's directory, and with it a detection track and a set of
        marks measured on different pixels.

        So the counter is PERSISTED and only ever moves forward. The max() is
        still taken, as a floor, because a sidecar written before this field
        existed has no counter and an imported layout can carry ids above it.
        """
        floor = max([int(r["id"]) for r in self.replicates], default=0) + 1
        try:
            stored = int(data.get("next_id", 0))
        except (TypeError, ValueError):
            stored = 0
        return max(floor, stored)

    def _autosave(self):
        """Persist the current boxes to this video's sidecar. Called on every
        edit so the boxes are always attached to the clip -- there is no separate
        'save' step to forget."""
        path = self.state.video_sidecar("rois")
        if not path:
            return
        try:
            with open(path, "w") as f:
                # next_id rides along with the boxes because it is the one piece
                # of state that must outlive them -- see _resume_next_id. Readers
                # that predate it (core.batch.load_replicates takes
                # data["replicates"] and ignores the rest) are unaffected.
                json.dump({"replicates": self.replicates,
                           "next_id": int(self._next_id)}, f, indent=2)
        except OSError as e:
            self.state.status.emit(f"Could not save boxes: {e}")

    def _video_path(self) -> str:
        return self.state.source.info.path if self.state.source else ""

    # -- optional per-replicate clip split ----------------------------------

    def _split_state(self):
        """Return ``(kind, manifest/detail)`` for this video and box geometry."""
        video = self._video_path()
        if not video:
            return "none", ""
        out_dir = os.path.dirname(os.path.abspath(video))
        path = manifest_path_for(video, out_dir)
        if not os.path.exists(path):
            return "none", ""
        try:
            manifest = read_manifest(path)
            verify_manifest(manifest, out_dir, self.replicates)
        except (PretranscodeError, OSError, ValueError) as e:
            return "stale", str(e)
        return "ready", manifest

    def _refresh_split_status(self):
        if self._split_worker is not None:
            return
        video = self._video_path()
        if not video:
            self.split_btn.setEnabled(False)
            self.split_btn.setText("Create replicate clips…")
            self.split_status.setStyleSheet("")
            self.split_status.setText("Open a video and draw boxes to create clips.")
            return
        if not self.replicates:
            self.split_btn.setEnabled(False)
            self.split_btn.setText("Create replicate clips…")
            self.split_status.setStyleSheet("")
            self.split_status.setText("Draw at least one replicate box first.")
            return

        self.split_btn.setEnabled(True)
        kind, value = self._split_state()
        if kind == "ready":
            manifest = value
            size = sum(int(c.size_bytes or 0) for c in manifest.clips)
            self.split_btn.setText("Rebuild replicate clips…")
            self.split_status.setStyleSheet("color:#287a3c;")
            self.split_status.setText(
                f"Ready: {len(manifest.clips)} {manifest.quality} clip(s), "
                f"{manifest.frame_count} frames, {size / 1e6:.0f} MB total. "
                "Enable 'Use ROI clips' in Preprocessing to use them.")
        elif kind == "stale":
            self.split_btn.setText("Rebuild stale clips…")
            self.split_status.setStyleSheet("color:#a45b00;")
            self.split_status.setText(f"Existing clips cannot be used: {value}")
        else:
            self.split_btn.setText("Create replicate clips…")
            self.split_status.setStyleSheet("")
            self.split_status.setText(
                "No replicate clips yet. Detection can still use the source video.")

    def _split_or_cancel(self):
        if self._split_worker is not None:
            if self._split_worker.isRunning():
                self._split_worker.cancel()
                self.split_btn.setEnabled(False)
                self.split_btn.setText("Canceling…")
                self.split_status.setText(
                    "Canceling; partial clips will be removed safely…")
            return

        video = self._video_path()
        if not video or not self.replicates:
            self._refresh_split_status()
            return
        kind, value = self._split_state()
        overwrite = kind in ("ready", "stale")
        if overwrite:
            if kind == "ready":
                question = (
                    "A verified split already exists for these boxes. Replace "
                    "all of its replicate clips?")
            else:
                question = (
                    "The existing split cannot be used with the current source "
                    f"and boxes:\n\n{value}\n\nReplace its replicate clips?")
            if QMessageBox.question(
                    self, "Rebuild replicate clips", question,
                    QMessageBox.StandardButton.Yes |
                    QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != \
                    QMessageBox.StandardButton.Yes:
                return

        out_dir = os.path.dirname(os.path.abspath(video))
        quality = str(self.split_quality.currentData())
        worker = ReplicateSplitWorker(
            video, self.replicates, out_dir, quality=quality,
            overwrite=overwrite, parent=self)
        self._split_worker = worker
        worker.progress.connect(self.split_progress.setValue)
        worker.complete.connect(self._split_complete)
        worker.failed.connect(self._split_failed)
        worker.cancelled.connect(self._split_cancelled)
        worker.finished.connect(self._split_thread_finished)

        # Freeze every geometry writer while the worker's snapshot is becoming
        # files. The manifest would detect a concurrent move as stale later,
        # but letting a minutes-long cut finish unusable is avoidable.
        self.video.setEnabled(False)
        self.box_group.setEnabled(False)
        self.std_box.setEnabled(False)
        self.split_quality.setEnabled(False)
        self.split_progress.setValue(0)
        self.split_progress.show()
        self.split_btn.setEnabled(True)
        self.split_btn.setText("Cancel splitting")
        self.split_status.setStyleSheet("")
        self.split_status.setText(
            f"Splitting {len(self.replicates)} replicate(s) at {quality} quality…")
        self.split_running_changed.emit(True)
        worker.start()

    def _split_complete(self, manifest):
        size = sum(int(c.size_bytes or 0) for c in manifest.clips)
        self.split_progress.setValue(1000)
        self.split_status.setStyleSheet("color:#287a3c;")
        self.split_status.setText(
            f"Created {len(manifest.clips)} clips, {manifest.frame_count} frames, "
            f"{size / 1e6:.0f} MB total. Enable 'Use ROI clips' in "
            "Preprocessing when you want to use them.")
        self.state.status.emit(
            f"Replicate split complete: {len(manifest.clips)} clips ({size / 1e6:.0f} MB).")

    def _split_failed(self, message: str):
        self.split_status.setStyleSheet("color:#a00000;")
        self.split_status.setText(f"Split failed: {message}")
        self.state.status.emit(f"Replicate split failed: {message}")

    def _split_cancelled(self):
        self.split_status.setStyleSheet("")
        self.split_status.setText("Split canceled; partial clips were removed.")
        self.state.status.emit("Replicate split canceled; partial clips removed.")

    def _split_thread_finished(self):
        worker = self._split_worker
        self._split_worker = None
        if worker is not None:
            worker.deleteLater()
        self.video.setEnabled(True)
        self.box_group.setEnabled(True)
        self.std_box.setEnabled(True)
        self.split_quality.setEnabled(True)
        self.split_progress.hide()
        self.split_running_changed.emit(False)
        self._refresh_split_status()

    def shutdown(self):
        """Stop the ffmpeg/hash worker before Qt destroys its QThread wrapper."""
        worker = self._split_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait()

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
            "processed": False,
        }
        self._next_id += 1
        self.replicates.append(rep)
        # The home is this replicate's identity on disk and exists from the
        # moment the box does, cut or not -- so the transcode stays an optional
        # speed-up rather than the thing that decides the layout.
        ensure_home(self._video_path(), rep["id"])
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
        rep = self.replicates[idx]
        if rep.get("processed") and not self._confirm_move(rep, (x0, y0)):
            # Declined: snap the box back to where it still is. self.replicates
            # was never written, so redrawing from it IS the revert -- there is
            # no saved-position copy to restore and therefore none to get wrong.
            self._redraw_boxes()
            return
        # Retire BEFORE the box's frac is overwritten: the rectangle the retired
        # files were measured against is the one it is leaving, and this is the
        # last moment anything holds it.
        self._retire(rep)
        rep["frac"] = (x0, y0, x1, y1)
        # Moving it makes it fresh again: whatever had been processed against
        # the old rectangle is exactly what the user just agreed to invalidate,
        # so leaving the flag set would warn a second time about results they
        # have already been told no longer apply.
        rep["processed"] = False
        # A moved box is new geometry, so the ROIs and every series computed
        # against them are stale -- same path as drawing one.
        self._rebuild_rois()
        self._sync_lock_badges()
        self._redraw_boxes()
        self._autosave()
        # No _sync_stamp_label: the release clamps the ORIGIN and keeps the
        # size, so a reposition cannot change what the stamp reports.

    def _confirm_move(self, rep: dict, to_origin: tuple[float, float]) -> bool:
        """Gate a drag of an already-processed replicate behind an explicit
        acknowledgement, naming what is at stake.

        Deliberately NOT a hard freeze. Freezing would force a delete-and-redraw
        for what is often a small correction, and redrawing loses the label, the
        colour and the standardization settings along with the geometry -- so the
        safe-looking option would cost the user more state than the dangerous one.
        With per-replicate homes the asymmetry is sharper still: a move keeps the
        id and therefore the whole directory, while a redraw takes a fresh id and
        starts from nothing. The cheap gesture is also the recoverable one.

        **It enumerates rather than threatens, and it deletes nothing -- but it
        no longer says the results survive the move.** That was the earlier
        ruling and it was reversed: a moved box can land on a different animal,
        or on nothing, so there is no reason to believe the marks under it were
        ever centred on the replicate the box now names. Carrying them forward as
        current is not staleness (which ``TrackStamp`` and ``load_track`` already
        catch and gray out) but **misattribution**, the failure class the
        per-region track split exists to prevent. So the fileset is RETIRED into
        ``old_NNN/`` with the rectangle it was measured against: still there,
        still drawn where it was, restorable, and no longer presented as
        describing the box's new position.

        ``to_origin`` is stated in the text because the box on screen is still
        drawn where it STARTED: FrameView clears the drag ghost before emitting
        box_moved, so by the time this dialog opens there is nothing showing the
        destination. Approving a destructive move with no indication of where it
        lands is exactly the blind confirmation this gate exists to avoid.
        """
        x0, y0 = rep["frac"][:2]
        nx, ny = to_origin
        held = describe_home(self._video_path(), rep["id"])
        inventory = ("<br><br>Its folder <b>%s</b> holds:<ul>%s</ul>"
                     % (os.path.basename(replicate_dir(self._video_path(),
                                                       rep["id"])),
                        "".join(f"<li>{h}</li>" for h in held))) if held else ""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Move a processed replicate?")
        box.setText(f"<b>{rep['label']}</b> has already been processed.")
        box.setInformativeText(
            "Moving it changes the region every existing measurement was "
            "computed over. Those results describe the old rectangle and "
            "cannot be re-aimed at the new one — the box may land on a "
            "different animal, or on none."
            f"{inventory}"
            "<br>Nothing is deleted. That work is <b>retired</b>: it moves into "
            f"a dated folder alongside the rectangle it was measured against, "
            f"stays drawn on this tab as a dashed <b>OLD {rep['label']}</b> box, "
            "and can be brought back — with its detections — from "
            "<b>Older geometries…</b>.<br><br>"
            f"Top-left would move from <b>{x0:.1%}, {y0:.1%}</b> to "
            f"<b>{nx:.1%}, {ny:.1%}</b> of the frame."
            "<br><br>The box keeps its name, colour, settings and folder.")
        go = box.addButton(_MOVE_ANYWAY, QMessageBox.ButtonRole.DestructiveRole)
        keep = box.addButton("Leave it where it is",
                             QMessageBox.ButtonRole.RejectRole)
        # Default AND escape route both land on the harmless option, so a
        # reflexive Enter or Esc on a dialog the user did not expect cannot
        # destroy anything.
        box.setDefaultButton(keep)
        box.setEscapeButton(keep)
        box.exec()
        return box.clickedButton() is go

    def mark_replicates_processed(self) -> None:
        """Latch every current box as processed.

        Called when the user leaves this tab, which is the only route to the
        surfaces that consume the layout -- so it strictly precedes any pass and
        a second trigger on the pass itself would be redundant. Boxes drawn after
        this are fresh again until the next time the user leaves.
        """
        fresh = [r for r in self.replicates if not r.get("processed")]
        if not fresh:
            return
        for r in fresh:
            r["processed"] = True
        self._sync_lock_badges()
        self._autosave()

    # -- retired geometries (slice 6) ----------------------------------------

    def _retire(self, rep: dict) -> None:
        """Move a replicate's current fileset into a fresh ``old_NNN/``.

        Called on every move, not only a gated one: an unprocessed box's home is
        empty and ``retire_current`` returns None for it, so the guard is the
        filesystem's rather than a second reading of ``processed`` that could
        disagree with it.
        """
        try:
            gen = retire_current(self._video_path(), rep["id"], rep["frac"],
                                 label=rep.get("label", ""))
        except OSError as e:
            # Reported, never raised: this runs inside a mouse-release handler,
            # where an exception is a crash. A failed retire leaves the files at
            # the root, describing a rectangle the box has left -- which is the
            # misattribution the retire prevents, so it must be said out loud.
            self.state.status.emit(
                f"Could not retire {rep.get('label', '')}'s existing results: "
                f"{e}. They still sit in the folder and describe the OLD box.")
            return
        if gen is not None:
            self.state.status.emit(
                f"Retired {rep.get('label', '')}'s results as generation {gen}; "
                "restore from 'Older geometries…'.")
        # Announced whether or not anything moved on disk. A live surface may
        # hold this replicate's track in memory and flushes it on close (that
        # flush is deliberate -- it is how an accumulated pass survives a
        # rebuild), so without this the retire is undone one tab switch later:
        # old-rectangle band power written straight back into the home root the
        # retire just emptied. Emitted even when gen is None because an
        # in-memory track can outlive a home whose files never landed.
        self.state.replicate_retired.emit(int(rep["id"]))
        # No _refresh_retired here: every caller runs _rebuild_rois immediately
        # after (the box's frac has just changed), and that is where the listing
        # is refreshed.

    def _refresh_retired(self) -> None:
        """Re-read every current box's retired generations from disk.

        Disk is the only record -- the box sidecar has no generation concept, by
        ruling -- so this is a listing rather than a cache invalidation. Only
        CURRENT boxes are scanned: a deleted box keeps its home (ruling 2) but is
        no longer part of the layout, and drawing its history would put
        rectangles on screen that no row in the list explains.
        """
        self._retired = []
        for r in self.replicates:
            for g in list_generations(self._video_path(), r["id"]):
                if g["frac"] is not None:
                    self._retired.append({**g, "rep": r})

    def _show_generations(self):
        """List the selected replicate's retired geometries and offer a restore."""
        rep = self._selected_rep()
        if rep is None:
            QMessageBox.information(self, "Older geometries",
                                    "Select a replicate first.")
            return
        gens = list_generations(self._video_path(), rep["id"])
        if not gens:
            QMessageBox.information(
                self, "Older geometries",
                f"<b>{rep['label']}</b> has never been moved, so it has no "
                "older geometries.")
            return
        # Keyed by the leading "gen N", which is unique per home and is what the
        # restore actually needs -- rather than matching the whole display string
        # back to its entry, which two generations retired in the same second
        # from the same rectangle would tie.
        choices: dict[str, int | None] = {}
        for g in gens:
            if g["frac"] is None:
                # Listed but not offered: restoring files without the rectangle
                # they were measured against is exactly the misattribution the
                # retirement prevented.
                choices[f"gen {g['gen']} — rectangle unknown, cannot restore"] = None
                continue
            x0, y0 = g["frac"][:2]
            held = ", ".join(g["held"]) or "nothing"
            when = f" · retired {g['retired_at']}" if g["retired_at"] else ""
            choices[f"gen {g['gen']} — at {x0:.1%}, {y0:.1%}{when} — {held}"] = \
                g["gen"]
        choice, ok = QInputDialog.getItem(
            self, "Older geometries",
            f"{rep['label']} — restoring moves the box back to that rectangle "
            "and brings its results with it.\nWhat is there now is retired in "
            "turn, so this is reversible.",
            list(choices), 0, False)
        if not ok:
            return
        gen = choices.get(choice)
        if gen is None:
            return
        self._restore(rep, gen)

    def _restore(self, rep: dict, gen: int) -> None:
        try:
            frac = restore_generation(self._video_path(), rep["id"], gen,
                                      rep["frac"], label=rep.get("label", ""))
        except (OSError, ValueError, FileNotFoundError) as e:
            QMessageBox.warning(self, "Could not restore", str(e))
            return
        rep["frac"] = tuple(frac)
        # A restore is a swap, so it retires the current fileset on the way past
        # -- the same announcement the move path makes, and for the same reason:
        # a live surface holding this replicate's track in memory would flush it
        # over the results just restored underneath it.
        self.state.replicate_retired.emit(int(rep["id"]))
        # Restored work was measured against the rectangle now in force, so the
        # box is processed again -- clearing the flag would let the next nudge
        # move a box holding real results without asking.
        rep["processed"] = True
        self._rebuild_rois()          # refreshes the retired listing too
        self._sync_lock_badges()
        self._redraw_boxes()
        self._zoom_to_selected()
        self._autosave()
        self.state.status.emit(
            f"Restored {rep.get('label', '')} generation {gen} and moved the "
            "box back to its rectangle.")

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
        # Retired rectangles go to a SEPARATE view list, never into the one
        # above: that list's indices are the drag contract. This is also why the
        # ghosts are drawn on THIS tab only -- the explorers route a pass into
        # whatever region is active, and a retired rectangle offered there could
        # aim a fresh pass at superseded geometry.
        self.video.set_ghost_frac_boxes([
            (*g["frac"], f"OLD {g['rep']['label']} · gen {g['gen']}",
             g["rep"]["color"])
            for g in self._retired])

    def _rebuild_rois(self):
        """Publish replicate geometry to shared state so the live detection path
        and the other tabs pick up the current boxes."""
        self.state.set_replicate_specs(self.replicates)
        # Here rather than at each call site: this is the one funnel every
        # mutation of the box list already passes through (add, move, delete,
        # clear, load, import), so the ghosts cannot be left describing boxes
        # that no longer exist. Deliberately NOT in _redraw_boxes, which runs on
        # every selection change and would put a directory listing per replicate
        # in a paint-adjacent path.
        self._refresh_retired()
        self.state.rois_changed.emit()
        self._refresh_split_status()

    # -- list ----------------------------------------------------------------

    @staticmethod
    def _row_text(r: dict) -> str:
        # The lock marks a box whose drag is gated. Shown on the row rather than
        # only in the dialog, so the state is legible BEFORE the user commits to
        # a gesture -- a warning that only appears once the drag is finished
        # teaches nothing about which boxes are safe to nudge.
        return f"  {r['label']}{'  🔒' if r.get('processed') else ''}"

    @staticmethod
    def _row_tip(r: dict) -> str:
        if not r.get("processed"):
            return ""
        return ("Processed. Moving this box makes the measurements computed "
                "over it stale; its folder is kept either way.")

    def _sync_lock_badges(self):
        """Update the 🔒 markers in place.

        Deliberately NOT _refresh_list: that clears the list, and the selection
        with it -- and ``stamp_size`` is derived from the selection (T11), so a
        rebuild here would silently empty the stamp on every move. Editing the
        rows relies on the same 1:1 ordering _redraw_boxes documents.
        """
        for i, r in enumerate(self.replicates):
            it = self.list.item(i)
            if it is not None:
                it.setText(self._row_text(r))
                it.setToolTip(self._row_tip(r))

    def _refresh_list(self):
        self.list.blockSignals(True)
        self.list.clear()
        for r in self.replicates:
            it = QListWidgetItem(self._row_text(r))
            it.setToolTip(self._row_tip(r))
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
        name = (name or "").strip()
        if not ok or not name or name == rep["label"]:
            return
        if not self._confirm_duplicate_label(rep, name):
            return
        # A rename is exactly this: one string. No directory moves with it,
        # because homes are named by id (core.replicate_home) -- which is the
        # same call core.replicates._canonical_geometry already made when it
        # excluded labels from the geometry hash. Had the folder taken the
        # label, this line would have to relocate a track, a marks file, a
        # tuning file and possibly a multi-gigabyte clip, and be atomic about it.
        rep["label"] = name
        self._rebuild_rois()
        self._refresh_list()
        self._redraw_boxes()
        self._autosave()

    def _confirm_duplicate_label(self, rep: dict, name: str) -> bool:
        """Warn about a label already in use, but do not forbid it.

        Blocking was tempting while the design still named folders after labels,
        where a duplicate really was a collision. It is not one now: two boxes
        called "control" own ``..._rep02`` and ``..._rep05``, and every store,
        export and provenance record keys on the id. What is left is a
        legibility problem -- a plot with two series called "control" -- and
        legibility problems get a warning, not a veto. Refusing here would also
        make a perfectly reasonable layout ("control" in four tubes, told apart
        by position) impossible to express.
        """
        clash = [r for r in self.replicates
                 if r is not rep and str(r.get("label", "")) == name]
        if not clash:
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Label already used")
        box.setText(f"Another replicate is already called <b>{name}</b>.")
        box.setInformativeText(
            "That is allowed — every box keeps its own folder and its own "
            "results regardless of what it is called — but the two will be "
            "indistinguishable by name in exports and plots.")
        go = box.addButton("Use it anyway", QMessageBox.ButtonRole.AcceptRole)
        back = box.addButton("Pick another", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(back)
        box.setEscapeButton(back)
        box.exec()
        return box.clickedButton() is go

    def _confirm_discard(self, reps: list[dict], title: str, what: str) -> bool:
        """Ask before dropping boxes whose homes hold real work.

        Silent when there is nothing to lose -- most deletes are a misdrawn box
        seconds old, and a dialog on those would train the user to dismiss the
        one that matters. The gate is the CONTENTS of the homes, not the
        ``processed`` flag, because that flag is about this session's state while
        the directory is about every session's.

        Nothing on disk is removed either way. The box list stops referring to
        the directory; the directory keeps its track, marks and clip. That is
        what makes ``next_id`` monotonic a safety property rather than a detail:
        no future box can be handed the retired id, so the orphan can never be
        silently adopted, and the user can delete it by hand once they are sure.
        """
        held = [(r, describe_home(self._video_path(), r["id"])) for r in reps]
        held = [(r, h) for (r, h) in held if h]
        if not held:
            return True
        lines = "".join(
            "<li><b>%s</b> (%s): %s</li>"
            % (r["label"],
               os.path.basename(replicate_dir(self._video_path(), r["id"])),
               ", ".join(h))
            for r, h in held)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(what)
        box.setInformativeText(
            f"These have results on disk:<ul>{lines}</ul>"
            "Their folders are <b>not</b> deleted — removing the box only stops "
            "this video referring to them, and the ids are never reissued, so "
            "nothing else can pick them up. Delete the folders by hand when you "
            "are sure.<br><br>Redrawing a box gives it a NEW folder; it does not "
            "recover this one.")
        go = box.addButton("Remove the box", QMessageBox.ButtonRole.DestructiveRole)
        keep = box.addButton("Keep it", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(keep)
        box.setEscapeButton(keep)
        box.exec()
        return box.clickedButton() is go

    def _delete(self):
        rep = self._selected_rep()
        if not rep:
            return
        if not self._confirm_discard(
                [rep], "Remove a replicate with saved results?",
                f"<b>{rep['label']}</b> has work stored beside the video."):
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
        # The contents gate comes FIRST and is the only prompt when it fires:
        # asking twice for one gesture is how a user learns to click through
        # both. With empty homes this falls back to the plain question, which is
        # all a layout of freshly drawn boxes warrants.
        if any(describe_home(self._video_path(), r["id"])
               for r in self.replicates):
            ok = self._confirm_discard(
                self.replicates, "Clear all replicates?",
                "Some of these boxes have work stored beside the video.")
        else:
            ok = QMessageBox.question(
                self, "Clear all", "Delete all replicate boxes?") \
                == QMessageBox.StandardButton.Yes
        if ok:
            self.replicates = []
            self.video.clear_focus()
            self._rebuild_rois()
            self._refresh_list()
            self._redraw_boxes()
            self._autosave()

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
        # `processed` is scoped to the clip it was measured on, so an IMPORTED
        # layout arrives fresh however locked it was in its source video --
        # otherwise a plate layout reused on a new clip would warn about
        # measurements that clip never had, which is the same ghosting-across-
        # videos failure _load_sidecar exists to prevent. (_load_sidecar itself
        # keeps the flag: that IS this video's own layout.)
        #
        # RENUMBERED onto fresh ids for the same reason, one level deeper. The
        # `processed: False` above says "these carry no measurements"; keeping
        # the source clip's ids would contradict it, because an imported id 1
        # would land in THIS video's rep01 home and adopt whatever that
        # replicate had already measured. Renumbering from the monotonic counter
        # makes the freshness structural instead of a flag that a later reader
        # has to remember to honour.
        self.replicates = []
        for r in data.get("replicates", []):
            self.replicates.append({**r, "id": self._next_id,
                                    "frac": tuple(r["frac"]),
                                    "processed": False})
            self._next_id += 1
        sync_homes(self._video_path(), self.replicates)
        self._rebuild_rois()
        self._refresh_list()
        self._redraw_boxes()
        self._autosave()
        self.state.status.emit(f"Loaded {len(self.replicates)} boxes.")

    def _go_preprocess(self):
        # Live preprocessing reads raw frames, so boxes are the only
        # precondition -- there is no cache to build first.
        if not self.replicates:
            QMessageBox.information(self, "No replicates",
                                    "Draw at least one box first.")
            return
        self._rebuild_rois()
        self.state.request_tab.emit(1)
