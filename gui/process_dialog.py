"""The gear beside "Process whole video": HOW the remainder gets processed.

The button used to mean exactly one thing -- decode frame 0 to the end. On a
five-minute clip that is the only sensible thing it could mean. On the multi-hour
footage this project actually works with it is a commitment measured in hours,
and the failure it produces when interrupted is not "less data" but *biased*
data: you end up having examined the front of the video and nothing else, and the
detections you have describe the first stretch rather than the clip.

So this dialog exposes the schedule as a choice, and its job is to make each
option's consequence legible BEFORE it is paid for -- the same contract
``gui/downsample_dialog`` keeps for the working scale. Every strategy shows what
it will decode, how much of the clip that is, and, for the sampling ones, says
outright how much will be left unexamined. A projected wall time is shown when
the surface has measured a rate this session; when it has not, the line says so
rather than inventing one.

The strategies themselves, and the arithmetic behind that summary, live in
``core/process_plan`` -- this file is the widget over them and holds no policy.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFormLayout, QLabel,
                             QRadioButton, QVBoxLayout)

from core.process_plan import (BUDGETED, DEFAULT_BUDGET, DEFAULT_CHUNK_S,
                               DEFAULT_STRATEGY, STRATEGIES, coverage_note,
                               plan_segments)


class ProcessSettingsDialog(QDialog):
    """Pick the strategy and its two knobs. Returns them via :attr:`settings`.

    Deliberately modal and deliberately without a Run button: it configures the
    Process action, it does not launch it. Launching from here would give the
    tool two ways to start the same expensive pass, and the one on the strip is
    the one the user reaches for.
    """

    def __init__(self, *, n_frames: int, fps: float, cursor: int = 0,
                 gaps=None, settings: dict | None = None,
                 rate_fps: float | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("How to process the video")
        self.setMinimumWidth(620)
        self._n_frames = int(n_frames)
        self._fps = float(fps)
        self._cursor = int(cursor)
        self._gaps = list(gaps or [])
        # Measured on this machine and this footage, or None. Never defaulted to
        # a constant: a made-up throughput number on a decision this expensive is
        # worse than no number, because it reads as a measurement.
        self._rate = rate_fps

        cur = dict(settings or {})
        strategy = cur.get("strategy", DEFAULT_STRATEGY)

        root = QVBoxLayout(self)
        head = QLabel(
            f"This clip is {self._n_frames} frames "
            f"({self._n_frames / max(self._fps, 1e-6) / 60.0:.1f} min). "
            f"Choose what the whole-video pass should cover.")
        head.setWordWrap(True)
        root.addWidget(head)

        self._radios: dict[str, QRadioButton] = {}
        for sid, label, blurb in STRATEGIES:
            rb = QRadioButton(label)
            rb.setChecked(sid == strategy)
            rb.toggled.connect(self._refresh)
            self._radios[sid] = rb
            root.addWidget(rb)
            note = QLabel(blurb)
            note.setWordWrap(True)
            note.setStyleSheet("color:#8ab; margin-left:22px; margin-bottom:6px;")
            root.addWidget(note)

        knobs = QFormLayout()
        self.chunk_spin = QDoubleSpinBox()
        self.chunk_spin.setRange(1.0, 3600.0)
        self.chunk_spin.setSuffix(" s")
        self.chunk_spin.setValue(float(cur.get("chunk_s", DEFAULT_CHUNK_S)))
        self.chunk_spin.setToolTip(
            "How long each sampled span is. Shorter spans spread the sample "
            "more finely but pay the seek and the cone-of-influence trim more "
            "often; longer spans are cheaper per frame examined but land in "
            "fewer places.")
        self.chunk_spin.valueChanged.connect(self._refresh)
        knobs.addRow("Chunk length", self.chunk_spin)

        self.budget_spin = QDoubleSpinBox()
        self.budget_spin.setRange(1.0, 100.0)
        self.budget_spin.setSuffix(" % of the clip")
        self.budget_spin.setValue(100.0 * float(cur.get("budget",
                                                        DEFAULT_BUDGET)))
        self.budget_spin.setToolTip(
            "How much footage to process, as a share of the clip. At 10% the "
            "pass decodes a tenth of the video — but spread across all of it, "
            "not the first tenth.")
        self.budget_spin.valueChanged.connect(self._refresh)
        knobs.addRow("Budget", self.budget_spin)
        root.addLayout(knobs)

        self.skip_covered_chk = QCheckBox(
            "Skip spans already examined under these settings")
        self.skip_covered_chk.setChecked(bool(cur.get("skip_covered", True)))
        self.skip_covered_chk.setToolTip(
            "Drop any part of the plan the strip already shows as covered and "
            "current, so a re-run continues instead of repeating itself. Stale "
            "coverage — computed under a channel, band or geometry no longer in "
            "force — is never skipped; it has to be recomputed to mean "
            "anything.")
        self.skip_covered_chk.toggled.connect(self._refresh)
        root.addWidget(self.skip_covered_chk)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(
            "background:#1b2430; color:#cfe; padding:10px; "
            "border:1px solid #2b3a4a; font-family:Consolas;")
        root.addWidget(self.summary)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._refresh()

    # -- state ---------------------------------------------------------------
    @property
    def strategy(self) -> str:
        for sid, rb in self._radios.items():
            if rb.isChecked():
                return sid
        return DEFAULT_STRATEGY

    @property
    def settings(self) -> dict:
        return {"strategy": self.strategy,
                "chunk_s": float(self.chunk_spin.value()),
                "budget": float(self.budget_spin.value()) / 100.0,
                "skip_covered": bool(self.skip_covered_chk.isChecked())}

    # -- live summary --------------------------------------------------------
    def _refresh(self, *_):
        sid = self.strategy
        budgeted = sid in BUDGETED
        # Greyed rather than hidden: a knob that vanishes reads as a bug, and
        # the user needs to see that it exists and does not apply here.
        self.chunk_spin.setEnabled(budgeted)
        self.budget_spin.setEnabled(budgeted)
        s = self.settings
        segments = plan_segments(
            sid, n_frames=self._n_frames, fps=self._fps, cursor=self._cursor,
            chunk_s=s["chunk_s"], budget=s["budget"], gaps=self._gaps)
        note = coverage_note(sid, segments, self._n_frames, self._fps)
        lines = [f"This plan decodes {note}."]
        if not segments and sid == "gaps":
            lines = ["Nothing to do: every frame is already covered under the "
                     "current settings."]
        total = sum(seg.n for seg in segments)
        if self._rate and total:
            mins = total / self._rate / 60.0
            # "at least": the number is a floor, and deliberately labelled as one.
            # It counts only the frames inside the spans, not the cone-of-
            # influence padding each span is decoded with (2*coi extra per span,
            # which a many-chunk sample multiplies), and it assumes the commit
            # runs at the live pass's measured rate -- which holds only if that
            # pass computed the same one channel the commit does. Quoting it as a
            # point estimate would make both gaps read as precision.
            lines.append(
                f"At the {self._rate:.0f} fps this machine last measured, that "
                f"is at least about {mins:.0f} min — more if the timed pass was "
                f"computing extra channels, or across many small chunks (each "
                f"is decoded slightly wider than it is).")
        elif total:
            lines.append(
                "No pass has been timed this session, so there is no honest "
                "time estimate to give — run Play briefly to measure one.")
        self.summary.setText("\n".join(lines))
