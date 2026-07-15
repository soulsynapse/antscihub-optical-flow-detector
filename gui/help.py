"""The (?) buttons.

Every parameter in Tab 1 gets one. The handoff specifies what each must say:
what the parameter does, the effect of raising or lowering it, why the default is
the default, and -- the part that actually matters -- how to tell from the
DOWNSTREAM histograms whether you have set it well. A help text that only explains
the knob is useless; the user cannot see the knob's effect, they can only see what
Tab 2 shows them afterwards.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QLabel, QPushButton, QVBoxLayout, QWidget,
                             QHBoxLayout)


class HelpDialog(QDialog):
    def __init__(self, title: str, body: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Help: {title}")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        lbl = QLabel(body.strip())
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(lbl)
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(btn)
        lay.addLayout(row)


class HelpButton(QPushButton):
    def __init__(self, key: str, parent=None):
        super().__init__("?", parent)
        self.key = key
        self.setFixedSize(18, 18)
        self.setStyleSheet(
            "QPushButton { border: 1px solid #555; border-radius: 9px; "
            "color: #aaa; font-weight: bold; font-size: 10px; }"
            "QPushButton:hover { color: #fff; border-color: #999; }")
        self.setToolTip("What is this?")
        self.clicked.connect(self._show)

    def _show(self):
        title, body = HELP.get(self.key, (self.key, "No help written yet."))
        HelpDialog(title, body, self).exec()


def labelled(widget: QWidget, help_key: str) -> QWidget:
    """Wrap a widget with a (?) button on its right."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    lay.addWidget(widget, 1)
    lay.addWidget(HelpButton(help_key))
    return w


def _h(what: str, raise_lower: str, default: str, downstream: str) -> str:
    return (f"<p><b>What it does.</b> {what}</p>"
            f"<p><b>Raising / lowering it.</b> {raise_lower}</p>"
            f"<p><b>Why this default.</b> {default}</p>"
            f"<p><b>How to tell it is set well.</b> {downstream}</p>")


HELP: dict[str, tuple[str, str]] = {

    "downsample": ("Downsample factor", _h(
        "Shrinks each frame before optical flow is computed. This is the single "
        "biggest lever on both compute time and cache size: cost scales with the "
        "square of this number.",
        "Lower = faster and smaller, but small or slow-moving things stop "
        "producing measurable flow. Higher = finer motion is resolved, at "
        "quadratic cost. Going from 0.25 to 0.5 makes the pass roughly 4x slower "
        "and the cache 4x bigger.",
        "The default targets a working width of about 1300 px, so it adapts to "
        "the footage: a 5.3K GoPro frame gets ~0.25, a 1080p frame gets ~0.5. A "
        "fixed factor would be wrong for one or the other.",
        "In Tab 2, look at the <i>speed</i> histogram. If your animal's motion "
        "sits indistinguishably on top of the near-zero noise mode, you have "
        "downsampled too far and the motion is being averaged away. If the "
        "histogram has a clear, separable shoulder above the noise, you are fine."),
    ),

    "block_size": ("Block size", _h(
        "Flow is computed per pixel, then averaged down to a grid of NxN pixel "
        "blocks. Everything downstream -- histograms, ROIs, band power -- lives "
        "on that block grid.",
        "Smaller blocks resolve smaller animals and finer structure, and cost "
        "proportionally more disk. Larger blocks average more pixels together, "
        "which suppresses flow noise but will dilute a small behavior: if a "
        "wing occupies a quarter of a block, three quarters of that block's "
        "signal is background.",
        "16x16 at the default downsample means each block covers about 64x64 "
        "source pixels, which is a reasonable fraction of a grasshopper.",
        "Click a block you know contains the behavior and look at its time "
        "series in the inspector. If the oscillation is visible but small "
        "relative to its own baseline, the block is too big and is diluting it."),
    ),

    "registration": ("Frame registration", _h(
        "Aligns each frame to a reference before computing flow, removing "
        "apparent motion caused by the camera itself. Phase correlation removes "
        "translation only and is fast; ORB + homography also handles rotation "
        "and scale but is slower.",
        "Off is fine for a locked-down tripod. On costs perhaps 20-30% more "
        "compute.",
        "Off, because much footage does not need it and it is not free.",
        "<b>This is the one that will silently ruin your results.</b> Camera "
        "shake is broadband and lands in the same frequency band as many "
        "behaviors. If your band-power overlay in Tab 2 lights up the whole "
        "frame -- including the substrate, the walls, things you know are not "
        "moving -- that is camera motion, and you need this on. A correctly "
        "registered clip has a band-power histogram whose bulk sits near zero."),
    ),

    "denoise": ("Temporal denoising", _h(
        "Averages or medians each pixel across a few neighbouring frames before "
        "computing flow, suppressing sensor noise.",
        "A larger window removes more noise but low-pass filters the video in "
        "time -- which directly attenuates exactly the high-frequency behaviors "
        "you may be trying to detect.",
        "Off. On well-lit footage it costs more signal than it buys, and it is "
        "actively dangerous for high-frequency behaviors.",
        "Only turn this on if the <i>speed</i> histogram shows a broad noise "
        "floor even in regions you know are empty. If you turn it on and your "
        "band-power peak shrinks, turn it back off -- you are filtering away the "
        "signal."),
    ),

    "bg_subtract": ("Background subtraction", _h(
        "Removes a static background estimate (temporal median, or an adaptive "
        "MOG2 model) so that only things that change are fed to the flow "
        "computation.",
        "Helps when the background is textured enough to generate spurious flow. "
        "Hurts when the animal is stationary for long stretches, because a "
        "median background will absorb it and it will vanish.",
        "Off, because dense optical flow already ignores static texture -- it "
        "measures change. This mostly matters for lighting flicker.",
        "If the flow field is noisy over obviously static regions (shelving, "
        "substrate), try the temporal median. Watch that your animal does not "
        "disappear from the overlay when it holds still."),
    ),

    "normalize": ("Contrast / illumination normalization", _h(
        "CLAHE equalizes local contrast; z-score normalizes each frame's mean and "
        "standard deviation. Both counteract lighting that changes over time or "
        "across the frame.",
        "Helps when illumination flickers (mains hum on some lights produces a "
        "real, and quite periodic, brightness oscillation). Also amplifies noise "
        "in dark regions.",
        "Off. Most controlled lab footage does not need it.",
        "Suspect this if band power shows a spatially uniform oscillation across "
        "the whole frame at a suspiciously round frequency -- 50 or 60 Hz mains, "
        "or a harmonic of it. Note that at 60 fps you cannot even see 60 Hz "
        "flicker; you would see it aliased down to near-DC."),
    ),

    "flow_backend": ("Optical flow backend", _h(
        "Farneback (CPU) is a polynomial-expansion method: fast, but it "
        "over-smooths and will under-resolve small structures. DIS (CPU) is "
        "faster and noticeably better on subtle motion. RAFT (GPU) is a learned "
        "method and is the best available on fine motion, but needs CUDA.",
        "Farneback is adequate for coarse, high-amplitude behaviors such as "
        "wingbeat or walking. Move to DIS or RAFT for antennal movement, "
        "grooming, or anything where the moving part is only a few pixels.",
        "Farneback, because it is always available and is sufficient for the "
        "coarse-scale case.",
        "If the behavior is visible to your eye in the video but produces no "
        "distinguishable mode in the <i>speed</i> histogram, the backend is "
        "failing to resolve it. Try DIS before you reach for a smaller block "
        "size -- it is cheaper."),
    ),

    "bands": ("Band-power frequency bands", _h(
        "For each block, the power in the speed signal's spectrum between two "
        "frequencies, on a sliding window. This is the primary feature for any "
        "periodic behavior.",
        "A narrow band around the true frequency gives the best contrast against "
        "background motion. A band that is too wide admits broadband noise and "
        "washes out the signal.",
        "One band, proposed from the video's frame rate. The classic grasshopper "
        "wingbeat is around 20 Hz.",
        "<b>Nyquist is the hard constraint here.</b> You cannot measure any "
        "frequency above fps/2, and content above it does not simply vanish -- it "
        "aliases down and masquerades as a real signal inside your band. At "
        "59.94 fps that ceiling is 29.97 Hz, so a band reaching 30 Hz is "
        "meaningless at its top edge. The tool warns above 80% of Nyquist. If "
        "your behavior is genuinely faster than that, no setting here will save "
        "you -- you need a higher frame rate."),
    ),

    "window_s": ("Band-power window length", _h(
        "The length of the FFT window used for band power. Frequency resolution "
        "is 1/window: a 1 s window resolves 1 Hz.",
        "A longer window resolves frequency more finely but smears the behavior "
        "in time -- you will not be able to tell a 0.2 s bout from its "
        "neighbours. A shorter window localizes bouts in time but blurs "
        "frequency. This trade is not negotiable; it is the uncertainty "
        "principle.",
        "1.0 s, giving 1 Hz resolution -- fine enough to isolate a 20 Hz "
        "wingbeat from background, short enough to localize a bout of a few "
        "hundred ms.",
        "If your behavior comes in bouts shorter than the window, band power "
        "will never reach its true value, because each window is mostly "
        "background. Shorten the window until the bouts stand out in the ROI "
        "time series."),
    ),

    "hop_s": ("Band-power hop", _h(
        "How far the FFT window advances between successive band-power values. "
        "This sets the time resolution of the band-power track and its size on "
        "disk.",
        "A smaller hop gives a smoother, more finely time-resolved track at "
        "proportionally more disk. It does NOT give you more information -- "
        "neighbouring windows overlap and are correlated.",
        "0.25 s, i.e. 75% overlap at the default window.",
        "If behavior onsets in the timeline strip look quantized or late, "
        "shorten the hop."),
    ),

    "dtype": ("Storage precision", _h(
        "float16 halves the cache size relative to float32.",
        "float16 carries about 3 decimal digits, which is far more than optical "
        "flow accuracy warrants -- the flow estimate itself is nowhere near that "
        "precise. float32 doubles the disk cost to store noise.",
        "float16. The precision loss is genuinely negligible next to the error in "
        "the flow estimate.",
        "You will not see a difference in any histogram. Only raise this if you "
        "have a specific reason to believe quantization is biting, which for flow "
        "data it will not."),
    ),

    "compression": ("Compression", _h(
        "Blosc/zstd compression on each chunk of the cache.",
        "Higher levels give slightly smaller files and slower writes. Turning "
        "compression off makes writes much faster and reads somewhat faster.",
        "zstd level 5. Note the honest number: flow data in float16 only "
        "compresses about 1.3x, because the mantissa bits are close to random. "
        "Do not expect compression to rescue an over-large cache -- fix that with "
        "downsampling or block size.",
        "Irrelevant to correctness. If your disk is fast and space is free, "
        "turning compression off is a legitimate speedup."),
    ),

    "test_seconds": ("Test mode duration", _h(
        "Processes only the first N seconds into a scratch cache, then drops you "
        "into Tab 2 so you can see whether the settings are working before "
        "committing to the full pass.",
        "Longer gives a more representative sample, particularly for band power, "
        "which needs several windows to be meaningful.",
        "10 s. Long enough for ~40 band-power windows at the defaults.",
        "Make sure the clip's first N seconds actually contain the behavior. If "
        "they do not, the histograms will show you nothing and you will conclude, "
        "wrongly, that the settings are bad."),
    ),

    "mask": ("Spatial ROI mask", _h(
        "Restricts flow computation to a region of the frame. Load a white-on-"
        "black PNG (white = keep), or click Draw to rubber-band the keep-regions "
        "directly on the video.",
        "A tighter mask means less to compute and a smaller cache, and it keeps "
        "irrelevant motion (a doorway, a fan, glare) out of your histograms. Too "
        "tight and you clip the behavior.",
        "None. Most clips do not need one.",
        "Use this when part of the frame generates motion you never care about. "
        "On the reference footage, masking to just the tubes and dropping the "
        "shelving and background would remove a lot of the camera-motion signal "
        "that otherwise contaminates band power."),
    ),

    "cache_expand": ("Expanding the cache", _h(
        "Adds extra features to the on-disk cache, at the cost of size and "
        "compute.",
        "Each option shows its own estimated size below.",
        "All off. <b>Start lean.</b> Most of what looks like it needs caching "
        "does not: angle, coherence, net flow, divergence, curl, rolling stats "
        "and dominant frequency are all derived on demand from the three arrays "
        "that are always cached, at zero disk cost. They are already available in "
        "Tabs 2 and 3 without enabling anything here.",
        "Only expand if a filter you believe should work does not. The honest "
        "reason to enable these is speed, not availability -- caching coherence "
        "saves recomputing it per session, but it was never unavailable."),
    ),
}
