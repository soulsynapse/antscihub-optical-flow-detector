"""The downsampling decision tool: what the scale lever gains and costs.

Batch K made downsampling opt-in and off by default. That is right -- a pipeline
that silently downsamples has already decided which behaviours are detectable --
but it is only half a change. A bare "downsample?" knob with no visible upside
produces exactly one behaviour: *"I don't want my data to be worse, so I won't
touch it."* That is an avoidance, not a decision, and it trades silent
degradation for silently infeasible projects. Many projects genuinely need full
resolution; many do not, and for those the difference is whether the project
happens at all. This window exists to make that choice strategic and legible.

What it shows, and what it refuses to show
------------------------------------------
* The **frontier**: projected wall time for the user's own corpus at each
  candidate scale, with the **knee** marked -- the point past which resolution is
  given up for almost no time saved. The knee comes from a measured cost model
  (``core/cost_model.py``), not a hardcoded guess.
* **What you keep**: working pixels per body length where the replicate is
  calibrated, working pixels across the box where it is not.
* Storage alongside time, so the two levers stay visibly distinct: scale is the
  *compute* lever and carries the "may decide what is detectable" warning; block
  size is the *storage* lever and does not. They are not fused into one "quality"
  slider, because a storage-limited user reaching for a fused slider would pay a
  sensitivity cost they had no need to pay.
* **No quality score.** Nothing here summarizes "how much worse" as one number.
  Every figure is measured wall clock, arithmetic storage, or a named setting.

* The **sweep**: a real extraction pass timed at each candidate scale, at the
  block a production run would use, with the storage that setting costs beside
  it. This is also what repairs the frontier -- the live surface tunes at
  ``block_size=1`` and a model fitted there overstates a batch run by ~2.6x
  (see ``model_block`` below), while the sweep's passes resolve the block the way
  a batch run would.

What this window does NOT try to answer
---------------------------------------
Whether a coarser scale still *detects* your behaviour. An empirical detection
panel briefly lived here -- the tuned detector run at each scale, reporting events
kept and lost -- and it was removed as misleading rather than merely unhelpful.
The value and count bands are absolute thresholds, and downsampling averages
pixels before differencing, so per-block band power falls with scale and a fixed
threshold catches less whether or not the behaviour is still resolved. Measured
on real footage the loss was monotone with ZERO added frames at any scale, which
is the signature of threshold drift, not of lost structure. A table that looks
measured and does not mean what it appears to is exactly the withdrawn
``sig_corr`` failure the plan warns against.

Detectability is decided in the live surface and the whole-video pass, which is
what they are for. This window prices the lever and states the consequence in
prose; it does not pretend to have measured a sensitivity it cannot separate from
a threshold artifact.

What it shows *instead*, and why that is not the same thing
-----------------------------------------------------------
Refusing to score sensitivity leaves an obvious hole: the window would be asking
the user to trade away resolution without ever showing them the resolution. The
**render strip** fills it with the one thing that is neither a sensitivity
measurement nor an invented score -- the actual image the pipeline receives at
each scale, beside the frame-difference field the detector's band power is built
from, on a display range shared across every tile.

It shows the *mechanism* rather than a verdict. The difference field visibly
dims as the scale falls, because averaging happens before differencing, and that
is precisely why the removed detection panel measured a monotone loss: the
threshold drifted, the structure did not necessarily go. Saying so with two rows
of pictures is honest in a way the events-kept table was not, because a picture
does not claim to have counted anything.

The **Calibrate…** button beside the replicate selector opens
``gui/calibration_dialog.py``, where a line drawn along the animal fixes the
"what you keep" readout in organism-relative units. Note what that costs: the
ruler is optional for *this* window, because working px per body length reduces
to ``body_line_px * scale`` and the fiducial cancels exactly (see
``core/calibration.py``). It is needed only to store the result in millimetres.

The panels here are built as reusable components (``gui/cost_panels.py``) so
Batch N's block-size sibling shares them rather than diverging.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QDoubleSpinBox, QFormLayout, QFrame,
                             QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from core.config import FlowConfig
from core.replicates import build_layout, in_tile_order
from core.cost_model import (CostModel, atlas_cells, boxes_from_tiles,
                             format_bytes, format_duration,
                             storage_bytes_per_hour,
                             working_px_per_body_length)
from core.scale_sweep import storage_rises_below

# The candidate scales offered, matching the approved layout. Not a continuous
# slider: the decision is which point on the frontier to take, and a handful of
# named points is easier to compare than a value that slides under the cursor.
CANDIDATE_SCALES = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1)

_WHY_OFFERED = (
    "<b>Downsampling is offered because it is often what makes a project "
    "computationally feasible at all.</b> It shortens every per-pixel stage — "
    "preprocessing, the structure-tensor products and blur, the flow solve, "
    "appearance and texture — so a corpus that would take months at full "
    "resolution can come down to days. For many behaviours a coarser working "
    "scale resolves the signal perfectly well, and refusing the lever on "
    "principle can mean the study simply does not get run."
)

_WHY_NOT_ASSUMED = (
    "<b>It is deliberately NOT assumed on your behalf.</b> A pipeline that "
    "downsamples by default has already decided which behaviours are "
    "detectable — the tool would be defining the data collected rather than the "
    "other way around. Whether a coarser scale still resolves <i>your</i> "
    "behaviour and <i>your</i> species is a scientific result about the "
    "organism, not a default constant, and it has to be demonstrated rather "
    "than assumed. Downsampling loses small and fast structures first. Use the "
    "frontier below to find the cheapest scale that is still defensible, not "
    "the cheapest scale."
)

_SWEEP_NOTE = (
    "<b>Each row is a real extraction pass over the loaded window, timed on this "
    "machine at the block a production run would use.</b> That is what makes the "
    "curve above a measurement rather than an estimate — and the live tuning "
    "surface cannot supply it, because it extracts at block 1 so that a Block "
    "change can re-reduce instead of re-extract, which costs ~2.6× a production "
    "pass. Storage is arithmetic over the real layout, not a measurement, so it "
    "is exact. <b>These rows say what a scale COSTS, not whether it still "
    "resolves your behaviour</b> — that is decided in the tuning surface and the "
    "whole-video pass."
)

_RENDER_NOTE = (
    "<b>The selected replicate as the pipeline sees it at each scale.</b> Top "
    "row: the working image the structure-tensor solve receives. Bottom: the "
    "frame difference its <i>change</i> channel is built from. Tiles share one "
    "display size and one brightness range, and are drawn without smoothing, so "
    "a coarser scale reads as blockier rather than smaller.<br>"
    "<b>The difference field dims as you downsample</b> — pixels are averaged "
    "<i>before</i> they are differenced. That is why an absolute threshold "
    "catches less at a coarser scale and must be re-tuned rather than carried "
    "across; it is a fact about the threshold, <b>not</b> evidence the behaviour "
    "has stopped being resolved. Judge the top row for whether the structure you "
    "care about survives."
)


class _Readout(QFrame):
    """One `AT 0.50 → 4.2 px per body length · 62 d · 0.9 TB` line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.head = QLabel("")
        self.head.setMinimumWidth(84)
        self.body = QLabel("")
        # head's stylesheet is set per-update by set() (it carries the emphasis
        # colour), so only body is styled once here.
        self.body.setStyleSheet("font-family:Consolas; font-size:12px;")
        lay.addWidget(self.head)
        lay.addWidget(self.body)
        lay.addStretch(1)

    def set(self, head: str, body: str, emphasis: str = "#c8d2dc"):
        self.head.setText(head)
        self.body.setText(body)
        self.head.setStyleSheet(
            f"font-family:Consolas; font-size:12px; font-weight:700; color:{emphasis};")


class DownsampleDialog(QDialog):
    """Pick a working scale against a measured cost model.

    ``model`` is built from whatever passes have actually run; pass ``None`` and
    the window still opens (the frontier then reports it has nothing measured
    rather than inventing a curve).

    The sweep is NOT run here. This dialog owns no video and no decoder -- it
    emits :attr:`sweep_requested` with the scales to time and the owner
    (``gui/explorers/live_scalogram_surface.py``) drives the passes off the GUI
    thread and feeds rows back through :meth:`add_sweep_row`. That keeps the
    dialog free of threading and keeps every panel in it reusable by Batch N's
    block-size sibling.
    """
    sweep_requested = pyqtSignal(object)        # list[float] of scales to time
    sweep_cancelled = pyqtSignal()
    render_requested = pyqtSignal(int, object)  # replicate index, scales
    calibrate_requested = pyqtSignal(int)       # replicate index

    def __init__(self, replicates: list[dict], src_width: int, src_height: int,
                 fps: float, current_scale: float, model: CostModel | None,
                 flow: FlowConfig | None = None, n_channels: int = 5,
                 corpus_hours: float = 100.0, model_block: int | None = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Downsampling — what you gain and what you lose")
        self.resize(980, 880)
        # Tile order, because the render strip and the box geometry come back in
        # it while the readouts index this list. See core.replicates.in_tile_order.
        self._reps = in_tile_order(replicates)
        self._src = (int(src_width), int(src_height))
        self._fps = float(fps)
        self._flow = flow or FlowConfig()
        self._n_channels = int(n_channels)
        self._model = model
        # The working block the timed passes actually ran at. The live surface
        # extracts at block=1 whenever the per-pixel cache fits, and block=1 does
        # no reduction at all -- measured, block_reduce is 62% of such a pass
        # against ~15% at block 64. A model fitted there therefore describes a
        # much costlier pass than a production run, so the projection has to say
        # which regime it came from rather than quietly overstating the corpus.
        self._model_block = model_block
        self._current = float(current_scale)
        self._selected = float(current_scale)
        self.chosen_scale: float | None = None      # set only by "Use this scale"
        # Geometry is fixed for the dialog's lifetime, so both derivations of it
        # are memoized rather than recomputed per keystroke.
        self._boxes_cache: list[tuple[int, int]] | None = None
        self._cells_cache: dict[tuple[float, int], int] = {}

        # The reference (full-resolution) pass, held so later rows can express
        # their speedup against it as they land, and every completed pass so the
        # table can be re-rendered against the final model when the sweep ends.
        self._sweep_ref = None
        self._sweep_rows: list = []

        # Rendered tiles, held so a selection change can re-highlight without
        # re-decoding: a pick costs a seek otherwise, and the user picks often.
        self._renders: list = []
        self._render_asked = False

        # Everything above the buttons scrolls. This window is mostly prose and
        # measured panels, and it grew past any fixed height that fits a laptop
        # screen -- unscrolled, Qt resolves the overflow by squeezing whichever
        # widget yields, which silently truncated the render note mid-sentence
        # and clipped the image tiles into letterbox strips. The buttons stay
        # OUTSIDE the scroll so the decision the window exists for is never
        # somewhere the user has to scroll to find.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        root = QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # Imported lazily-ish at module scope; kept here so the prose reads in
        # order with the layout it heads.
        from gui.cost_panels import (FrontierPlot, LeverPreamble, RenderStrip,
                                     SweepPanel)
        root.addWidget(LeverPreamble(_WHY_OFFERED, _WHY_NOT_ASSUMED))

        root.addLayout(self._build_inputs())

        self.plot = FrontierPlot()
        self.plot.picked.connect(self._on_pick)
        # No stretch factor: inside a scroll area the body sizes to its hint, so
        # a stretch here would not grow the plot, it would only let the layout
        # steal height back off it under pressure.
        self.plot.setMinimumHeight(215)
        root.addWidget(self.plot)

        root.addWidget(self._build_readouts())

        self.render = RenderStrip(_RENDER_NOTE)
        root.addWidget(self.render)

        self.sweep = SweepPanel(_SWEEP_NOTE)
        self.sweep.run_requested.connect(self._on_run_sweep)
        self.sweep.cancel_requested.connect(self.sweep_cancelled.emit)
        root.addWidget(self.sweep)
        root.addStretch(1)

        buttons = self._build_buttons()
        buttons.setContentsMargins(9, 6, 9, 9)
        outer.addLayout(buttons)
        self._refresh()

    # -- construction --------------------------------------------------------
    def _build_inputs(self) -> QHBoxLayout:
        row = QHBoxLayout()
        form = QFormLayout()
        self.corpus_spin = QDoubleSpinBox()
        self.corpus_spin.setRange(0.1, 1_000_000.0)
        self.corpus_spin.setDecimals(1)
        self.corpus_spin.setValue(100.0)
        self.corpus_spin.setSuffix(" h")
        self.corpus_spin.setToolTip(
            "Total footage this project has to process. The frontier is "
            "projected for this much video on THIS machine, single process.")
        self.corpus_spin.valueChanged.connect(self._refresh)
        form.addRow("Corpus", self.corpus_spin)

        self.rep_spin = QSpinBox()
        self.rep_spin.setRange(0, max(0, len(self._reps) - 1))
        self.rep_spin.setToolTip(
            "Which replicate the 'what you keep' readout describes. Time and "
            "storage always cover every replicate.")
        self.rep_spin.valueChanged.connect(self._refresh)
        # Separate from _refresh: the render costs a decode, so it is requested
        # on the replicate changing and NOT on the corpus-hours spin, which also
        # runs _refresh and has nothing to do with what the box looks like.
        self.rep_spin.valueChanged.connect(self._on_rep_changed)
        rep_row = QHBoxLayout()
        rep_row.addWidget(self.rep_spin)
        self.calib_btn = QPushButton("Calibrate…")
        self.calib_btn.setToolTip(
            "Draw a line along the animal to get working pixels per body "
            "length exactly — no ruler needed for that number. Add a ruler line "
            "to store the result in millimetres.")
        self.calib_btn.clicked.connect(
            lambda: self.calibrate_requested.emit(self.rep_spin.value()))
        rep_row.addWidget(self.calib_btn)
        form.addRow("Replicate", rep_row)
        row.addLayout(form)
        row.addStretch(1)

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setMaximumWidth(520)
        self.note.setStyleSheet("color:#8fa3b5; font-size:11px;")
        row.addWidget(self.note)
        return row

    def _build_readouts(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 6, 0, 6)
        self.row_sel = _Readout()
        self.row_full = _Readout()
        lay.addWidget(self.row_sel)
        lay.addWidget(self.row_full)
        return box

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch(1)
        self.use_btn = QPushButton("Use this scale")
        self.use_btn.clicked.connect(self._accept_selected)
        keep = QPushButton("Keep 1.00 (no downsampling)")
        keep.setDefault(True)
        keep.clicked.connect(self._keep_full)
        row.addWidget(self.use_btn)
        row.addWidget(keep)
        return row

    # -- model ---------------------------------------------------------------
    def _boxes(self):
        """Source-pixel replicate boxes. Cached: the geometry cannot change
        while the dialog is open, and build_layout sha1-hashes the canonical
        geometry on every call."""
        if self._boxes_cache is None:
            if not self._reps:
                self._boxes_cache = [(self._src[0], self._src[1])]
            else:
                layout = build_layout(self._reps, self._src[0], self._src[1],
                                      scale=1.0, block_size=1)
                self._boxes_cache = boxes_from_tiles(layout.tiles)
        return self._boxes_cache

    def _hours(self, scale: float) -> float | None:
        """None when no trustworthy model exists -- including the provisional
        one, whose projections are wrong by several-fold at low scales. The
        readout rows then say so instead of printing a confident wrong duration."""
        if self._model is None or self._model.provisional:
            return None
        return self._model.hours_for_corpus(scale, self.corpus_spin.value(), self._fps)

    def _cells(self, scale: float) -> int:
        """Allocated atlas cells, memoized per (scale, block) -- the readout rows
        ask for the same scale several times per refresh and each miss is a
        layout rebuild plus a geometry hash."""
        key = (round(scale, 6), self._flow.resolve_block_size(scale))
        if key not in self._cells_cache:
            self._cells_cache[key] = atlas_cells(
                self._reps, self._src[0], self._src[1], scale, key[1])
        return self._cells_cache[key]

    def _storage(self, scale: float) -> float:
        return storage_bytes_per_hour(self._cells(scale), self._fps,
                                      self._n_channels) * self.corpus_spin.value()

    def _kept(self, scale: float) -> str:
        """What the animal is resolved by at this scale, organism-relative where
        the replicate is calibrated and geometric where it is not."""
        rep = self._reps[self.rep_spin.value()] if self._reps else {}
        px_bl = working_px_per_body_length(rep.get("pixels_per_mm"),
                                           rep.get("body_length_mm"), scale,
                                           rep.get("body_length_px"))
        if px_bl is not None:
            return f"{px_bl:.1f} px per body length"
        boxes = self._boxes()
        i = min(self.rep_spin.value(), len(boxes) - 1)
        w, h = boxes[i]
        return (f"{int(round(w * scale))}x{int(round(h * scale))} working px "
                f"(uncalibrated)")

    # -- refresh -------------------------------------------------------------
    def _refresh(self):
        scales = list(CANDIDATE_SCALES)
        # Ordered by cause, not by symptom: _hours() already returns None for a
        # provisional model, so testing "any hour missing" first would swallow
        # the provisional case and show the wrong explanation for it.
        if self._model is None:
            self.plot.set_curve([], [])
            self.note.setText(
                "No pass has been timed yet, so there is no cost model and no "
                "frontier. Run an extract, then reopen this window.")
        elif self._model.provisional:
            # Deliberately no curve. One pass cannot see the decode floor behind
            # the prefetch thread, and the frontier it implies under-prices heavy
            # downsampling by ~5.7x -- i.e. it would argue for exactly the choice
            # this window exists to stop being made carelessly. A plot that wrong
            # is worse than no plot, because it looks measured.
            self.plot.set_curve([], [])
            self.note.setText(
                "Only one working scale has been timed, which is not enough to "
                "separate the fixed decode cost from the per-pixel cost — and "
                "decode runs on its own thread, so a single pass barely sees it. "
                "The frontier is withheld rather than shown wrong: a one-pass "
                "estimate under-prices aggressive downsampling several-fold. "
                "Run the sweep below and the frontier appears: it times a pass "
                "per scale, at the block a production run would use. (Extracting "
                "once more at a different Downsample value also works.)")
        else:
            hours = [self._hours(s) for s in scales]
            knee = self._model.knee_scale(min_scale=min(scales))
            store = [self._storage(s) for s in scales]
            self.plot.set_curve(
                scales, hours, labels=[f"{s:g}" for s in scales], knee=knee,
                selected=self._selected,
                y_label=f"projected wall time for {self.corpus_spin.value():g} h "
                        f"of footage, single process",
                second=store, second_label="cache storage (dashed)",
                second_fmt=format_bytes,
                rise_below=storage_rises_below(scales, store))
            self.note.setText(self._note_text(knee))
        self._refresh_rows()

    def _note_text(self, knee) -> str:
        m = self._model
        parts = []
        if knee is None:
            parts.append(
                "No knee: decode already dominates at full resolution here, so "
                "downsampling buys very little time on this footage. Take the "
                "resolution.")
        else:
            parts.append(
                f"The knee is at {knee:.2f} — where the per-pixel work has "
                f"shrunk to equal the decode floor this lever cannot go below. "
                f"Above it, 1% less scale buys more than 1% less time; below it, "
                f"you give up more resolution than you gain.")
        parts.append(f"Fitted from {m.n_samples} measured passes on this machine "
                     f"and this footage; it is not a portable constant.")
        # The two-lever claim, stated where it is visible rather than asserted in
        # docs. Phrased as "does not fall" rather than "is flat" because it is
        # not flat: the block and the cell count round independently, so the
        # curve jitters (measured, up to 19% on a single full-frame replicate,
        # and non-monotone). What holds is the decision-relevant part -- reaching
        # for Downsample to save disk is reaching for the wrong knob.
        scales = list(CANDIDATE_SCALES)
        store = [self._storage(s) for s in scales]
        at_full = store[scales.index(1.0)] if 1.0 in scales else max(store)
        if at_full > 0 and min(store) >= 0.9 * at_full:
            worst = max(store)
            rises = "; at some scales it is higher" if worst > at_full else ""
            parts.append(
                f"Storage (dashed) does not fall as you downsample{rises}. The "
                f"block tracks the scale, so the grid stays fixed in source "
                f"pixels and the cache is the same size — block size is the "
                f"storage lever, this one is purely compute.")
        # The projected block tracks each scale, so it differs row to row; the
        # caveat is about the regime the MEASUREMENT came from, and naming a
        # single projected block here would contradict the per-row labels.
        if self._model_block is not None and self._model_block <= 1:
            parts.append(
                "Those passes ran at block 1 — the live per-pixel cache, which "
                "does no block reduction and is substantially slower than a "
                "production pass. The times below therefore OVERSTATE what a "
                "batch run would cost: read them as an upper bound and compare "
                "scales against each other rather than trusting the absolute "
                "figure. Running the sweep below replaces them — those passes "
                "resolve the block as a production run does.")
        return "  ".join(parts)

    def _refresh_rows(self):
        for row, scale, emph in ((self.row_sel, self._selected, "#7ee787"),
                                 (self.row_full, 1.0, "#c8d2dc")):
            hrs = self._hours(scale)
            bits = [self._kept(scale)]
            bits.append(format_duration(hrs) if hrs is not None else "time unknown")
            bits.append(format_bytes(self._storage(scale)))
            bits.append(f"block {self._flow.resolve_block_size(scale)} px · "
                        f"{self._cells(scale)} cells")
            row.set(f"AT {scale:.2f} →", "  ·  ".join(bits), emph)
        self.use_btn.setEnabled(abs(self._selected - 1.0) > 1e-9)
        self.use_btn.setText(f"Use {self._selected:.2f}")

    # -- the render strip ----------------------------------------------------
    def showEvent(self, ev):
        """Request the renders once the window is up.

        Not in ``__init__``: the owner connects :attr:`render_requested` after
        construction and before ``exec()``, so an emit from the constructor
        would go nowhere. Same reason the sweep is owner-driven -- this dialog
        holds no video and no decoder.
        """
        super().showEvent(ev)
        if not self._render_asked:
            self._render_asked = True
            self._request_render()

    def _request_render(self):
        if not self._reps:
            self.render.set_status(
                "No replicates are defined, so there is no box to render.")
            return
        self.render.set_status("rendering…")
        self.render_requested.emit(self.rep_spin.value(),
                                   list(CANDIDATE_SCALES))

    def set_renders(self, renders, note: str = ""):
        self._renders = list(renders)
        self.render.set_renders(self._renders, self._selected)
        self.render.set_status(note)

    def render_failed(self, msg: str):
        self.render.set_renders([])
        self._renders = []
        self.render.set_status(f"Could not render this replicate: {msg}")

    def apply_calibration(self, rep_index: int, fields: dict):
        """Merge a calibration the owner just measured, and re-read the rows.

        Only the readouts change: calibration is organism geometry and touches
        neither the cost model nor the storage arithmetic, so the frontier and
        the sweep rows are left alone.
        """
        if not fields or not self._reps:
            return
        i = min(max(0, int(rep_index)), len(self._reps) - 1)
        self._reps[i].update(fields)
        self._refresh_rows()

    # -- the timing sweep ----------------------------------------------------
    def set_sweep_available(self, ok: bool, reason: str = ""):
        """Called by the owner when it knows whether a sweep can run at all."""
        self.sweep.set_available(ok, reason)

    def _on_run_sweep(self):
        from core.scale_sweep import sweep_scales
        scales = sweep_scales(self._current, CANDIDATE_SCALES)
        self._sweep_ref = None
        self._sweep_rows = []
        self.sweep.begin(
            [f"{s:.2f}" for s in scales],
            f"{len(scales)} passes over the loaded window, full resolution "
            f"first so a sweep you stop early is still readable. Each runs at "
            f"the block a production run would use.")
        self.sweep_requested.emit(scales)

    def add_sweep_row(self, sp):
        """One completed pass. The first becomes the reference the rest express
        their speedup against."""
        self._sweep_rows.append(sp)
        if self._sweep_ref is None:
            self._sweep_ref = sp
        self._render_sweep_row(sp)

    def _render_sweep_row(self, sp):
        hrs = self._hours(sp.scale)
        self.sweep.add_row(
            f"{sp.scale:.2f}", sp,
            format_duration(hrs) if hrs is not None else "—",
            format_bytes(self._storage(sp.scale)),
            None if sp is self._sweep_ref else self._sweep_ref)

    def sweep_failed(self, scale: float, msg: str):
        self.sweep.fail_row(f"{float(scale):.2f}", msg)

    def sweep_finished(self, note: str):
        # Re-render every row against the FINAL model. The first row lands when
        # only one scale has been timed, so the model is still provisional and
        # its projection reads "—" -- and it would stay that way for the rest of
        # the session, on the one row (full resolution) a user most wants the
        # number for.
        for sp in self._sweep_rows:
            self._render_sweep_row(sp)
        self.sweep.finish(note)

    def set_model(self, model: CostModel | None, model_block: int | None):
        """Replace the cost model mid-dialog. The sweep produces samples at the
        production block, so the frontier can go from withheld to drawn (and
        from an upper bound to a real projection) without reopening the window."""
        self._model = model
        self._model_block = model_block
        self._refresh()

    # -- actions -------------------------------------------------------------
    def _on_rep_changed(self, *_):
        if self._render_asked:
            self._request_render()

    def _on_pick(self, scale: float):
        self._selected = float(scale)
        # Rows only: the curve, the knee and the note do not depend on which
        # point is selected, and the plot tracks its own selection marker.
        self._refresh_rows()
        if self._renders:
            self.render.set_renders(self._renders, self._selected)

    def _accept_selected(self):
        self.chosen_scale = self._selected
        self.accept()

    def _keep_full(self):
        self.chosen_scale = 1.0
        self.accept()
