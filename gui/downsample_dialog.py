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

* The **empirical panel**: the tuned detector run at each candidate scale over
  the loaded window, reporting the events it found and which of the reference
  run's frames survived. This is where Batch K's "demonstrated per behaviour,
  never assumed" becomes something a user executes in a minute rather than an
  instruction they cannot act on. Without it the window argues feasibility much
  harder than it argues sensitivity, and an asymmetric window biases toward
  downsampling -- so it is not an optional extra, it is what makes the rest of
  this window safe to show.

Running the sweep repairs the frontier as a side effect, which is deliberate: the
live surface tunes at ``block_size=1`` and a model fitted there overstates a
production run's cost (see ``model_block`` below), while the evidence passes
resolve the block the way a batch run would. One sweep therefore buys both a
sensitivity answer and a cost model in the right regime.

Not yet built (see todo.md Batch M): the render-at-each-scale image panel and the
draw-a-line calibration sub-tool (reading ``pixels_per_mm`` / ``body_length_mm``
off the replicate dicts works today). The panels below are built as reusable
components (``gui/cost_panels.py``) so Batch N's block-size sibling shares them
rather than diverging.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QDoubleSpinBox, QFormLayout, QFrame,
                             QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QSpinBox, QVBoxLayout, QWidget)

from core.config import FlowConfig
from core.replicates import build_layout
from core.cost_model import (CostModel, atlas_cells, boxes_from_tiles,
                             format_bytes, format_duration,
                             storage_bytes_per_hour,
                             working_px_per_body_length)

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

_EVIDENCE_SCOPE = (
    "<b>Evidence on THIS clip, THIS window and THIS behaviour — not a general "
    "sensitivity guarantee.</b> Each row re-extracts the loaded window at that "
    "scale and runs the detector you just tuned, unchanged. A scale that keeps "
    "your events here has been shown to keep them <i>here</i>; it says nothing "
    "about a different species, a different behaviour, a quieter animal, or a "
    "clip shot at another distance. Note also what this cannot separate: the "
    "value and count bands are absolute thresholds, and downsampling averages "
    "pixels before differencing, so a lost event may mean the motion is no "
    "longer resolved <i>or</i> that your threshold no longer sits where you put "
    "it. Treat a row that loses frames as a reason to re-tune and re-check at "
    "that scale, not as a settled result."
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

    The empirical sweep is NOT run here. This dialog owns no video, no decoder
    and no detector settings -- it emits :attr:`evidence_requested` with the
    scales to run and the owner (``gui/explorers/live_scalogram_surface.py``)
    drives the passes off the GUI thread and feeds rows back through
    :meth:`add_evidence`. That keeps the dialog free of threading and keeps every
    panel in it reusable by Batch N's block-size sibling.
    """
    evidence_requested = pyqtSignal(object)     # list[float] of scales to run
    evidence_cancelled = pyqtSignal()

    def __init__(self, replicates: list[dict], src_width: int, src_height: int,
                 fps: float, current_scale: float, model: CostModel | None,
                 flow: FlowConfig | None = None, n_channels: int = 5,
                 corpus_hours: float = 100.0, model_block: int | None = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Downsampling — what you gain and what you lose")
        self.resize(980, 880)
        self._reps = list(replicates)
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

        # The reference row every other evidence row is read against, held so
        # rows arriving one at a time can each be compared as they land.
        self._evidence_ref = None

        root = QVBoxLayout(self)
        # Imported lazily-ish at module scope; kept here so the prose reads in
        # order with the layout it heads.
        from gui.cost_panels import EvidencePanel, FrontierPlot, LeverPreamble
        root.addWidget(LeverPreamble(_WHY_OFFERED, _WHY_NOT_ASSUMED))

        root.addLayout(self._build_inputs())

        self.plot = FrontierPlot()
        self.plot.picked.connect(self._on_pick)
        root.addWidget(self.plot, 1)

        root.addWidget(self._build_readouts())

        self.evidence = EvidencePanel(_EVIDENCE_SCOPE)
        self.evidence.run_requested.connect(self._on_run_evidence)
        self.evidence.cancel_requested.connect(self.evidence_cancelled.emit)
        root.addWidget(self.evidence)

        root.addLayout(self._build_buttons())
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
        form.addRow("Replicate", self.rep_spin)
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
                                           rep.get("body_length_mm"), scale)
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
            self.plot.set_curve(
                scales, hours, labels=[f"{s:g}" for s in scales], knee=knee,
                selected=self._selected,
                y_label=f"projected wall time for {self.corpus_spin.value():g} h "
                        f"of footage, single process")
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

    # -- empirical sweep -----------------------------------------------------
    def set_evidence_available(self, ok: bool, reason: str = ""):
        """Called by the owner when it knows whether a sweep can run at all
        (a window must be extracted and a replicate selected first)."""
        self.evidence.set_available(ok, reason)

    def _on_run_evidence(self):
        from core.evidence import sweep_scales
        scales = sweep_scales(self._current, CANDIDATE_SCALES)
        self._evidence_ref = None
        self.evidence.begin(
            [f"{s:.2f}" for s in scales],
            f"{len(scales)} passes over the loaded window, the reference "
            f"({scales[0]:.2f}) first so a sweep you stop early is still "
            f"readable. Each pass runs at the block a production run would use.")
        self.evidence_requested.emit(scales)

    def add_evidence(self, ev):
        """One completed pass. The first row becomes the reference; the rest are
        reported against it. Also refreshes the frontier's caveat, since these
        passes are measured in the production block regime the live surface's
        own samples are not."""
        from core.evidence import reference_caveat
        # The check runs on the reference BEFORE it is drawn, so a vacuous sweep
        # is marked from its very first row rather than after the user has read
        # four rows of apparent agreement.
        if self._evidence_ref is None:
            self.evidence.void_comparison(reference_caveat(ev))
        self.evidence.add_row(f"{ev.scale:.2f}", ev, self._evidence_ref)
        if self._evidence_ref is None:
            self._evidence_ref = ev

    def evidence_failed(self, scale: float, msg: str):
        self.evidence.fail_row(f"{float(scale):.2f}", msg)

    def evidence_finished(self, note: str):
        self.evidence.finish(note)

    def set_model(self, model: CostModel | None, model_block: int | None):
        """Replace the cost model mid-dialog. The sweep produces samples at the
        production block, so the frontier can go from withheld to drawn (and
        from an upper bound to a real projection) without reopening the window."""
        self._model = model
        self._model_block = model_block
        self._refresh()

    # -- actions -------------------------------------------------------------
    def _on_pick(self, scale: float):
        self._selected = float(scale)
        # Rows only: the curve, the knee and the note do not depend on which
        # point is selected, and the plot tracks its own selection marker.
        self._refresh_rows()

    def _accept_selected(self):
        self.chosen_scale = self._selected
        self.accept()

    def _keep_full(self):
        self.chosen_scale = 1.0
        self.accept()
