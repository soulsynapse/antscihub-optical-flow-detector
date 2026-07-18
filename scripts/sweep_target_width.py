"""Price DEFAULT_TARGET_WIDTH against detection output (todo.md Batch K/I).

The question this answers: how much working resolution does detection actually
need? `DEFAULT_TARGET_WIDTH` (1300) is what forces replicate-aware auto
downsample to cost ~16x pixels, and it is also the number a ROI pre-transcode
(Batch I) would have to bake into its clips. Nobody has ever measured whether
1300 is generous, tight, or wrong.

THE CONTROLLED VARIABLE. Lowering the target width does two separable things:
it makes each block average fewer working pixels, and -- if block_size is held
fixed in *working* pixels -- it also makes the block grid coarser, so blocks
cover more source area. Those are different questions. Detection reads
block-level band power over *time*, so the spatial resolution it needs is set by
localization (the block grid), not by target width per se.

So this sweep holds the SOURCE-REFERRED block size constant: block_size scales
with the target width (32 @ 2600, 16 @ 1300, 8 @ 650, 4 @ 325). Every rung then
produces the same block grid covering the same source area, and the only thing
that changes is how many working pixels get averaged into each block. That is
the variable we want to price, and it has a useful side effect: block counts are
comparable across rungs, so a count threshold means the same thing everywhere.

MATCHING THE OPERATING POINT. Band power is not scale-invariant -- averaging
more pixels into a block smooths it, so absolute thresholds drift with scale for
reasons that have nothing to do with sensitivity. Holding a raw threshold fixed
would measure that drift, not detection agreement. Instead each rung takes its
value band at the same *quantile* of its own band-power distribution, and the
count gate at the same quantile of its own windowed series. That holds the
detector's operating point fixed and asks the question that matters: are the
same time intervals flagged?

Usage:
    .venv\\Scripts\\python.exe scripts/sweep_target_width.py \\
        Videos/Stabilized/GX010047c2_02_17_26.MP4 --frames 479
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.channel_source import live_channel_source
from core.config import FlowConfig, PipelineConfig, PreprocessConfig
from core.detection import (detect_gate, inband_count, region_blocks_and_grid,
                            windowed_mean)
from core.video import VideoSource
from core.wavelet import band_indices, default_freqs, morlet_band_power

# (target_width, block_size). Block size tracks the width so every rung's blocks
# cover the same SOURCE area -- see the module docstring. 1300/16 is today's
# default and is the reference every other rung is scored against.
LADDER = ((2600, 32), (1300, 16), (650, 8), (325, 4))

# This sweep has been run; its result is recorded under "DEFAULT_TARGET_WIDTH
# sweep" in todo.md and is what retired the constant this used to import. The
# reference rung is kept as a literal so the script stays runnable as a
# historical comparison -- it is no longer a default anywhere in the tool.
REFERENCE_WIDTH = 1300

# Operating point, as quantiles of each rung's own distribution.
VALUE_Q = 0.90        # value band = [q90 of band power, +inf)
COUNT_Q = 0.85        # gate fires above q85 of the windowed in-band count


def load_replicates(video_path: str) -> list[dict]:
    """Replicate boxes from the video's sidecar .rois.json."""
    base = os.path.splitext(video_path)[0]
    for cand in (base + ".rois.json", base + ".json"):
        if os.path.exists(cand):
            with open(cand) as fh:
                return json.load(fh)["replicates"]
    raise SystemExit(f"No .rois.json sidecar next to {video_path}")


def run_rung(video_path: str, replicates: list[dict], info, target_w: int,
             block: int, n_frames: int, freq_band: tuple[float, float]) -> dict:
    """Extract + detect at one rung. Returns per-region windowed series."""
    scale = min(1.0, target_w / info.width)
    cfg = PipelineConfig(
        preprocess=PreprocessConfig(downsample=scale),
        flow=FlowConfig(block_size=block),
    )
    t0 = time.perf_counter()
    data = live_channel_source(video_path, cfg, replicates, start=0, n=n_frames,
                               width=info.width, height=info.height,
                               fps=info.fps, frame_count=info.frame_count)
    extract_s = time.perf_counter() - t0

    fps = float(data.meta["fps"])
    freqs = default_freqs(fps)
    i, j = band_indices(freqs, *freq_band)
    # Detection window D: one second, matching the explorer's default.
    win = max(1, min(data.channels["tensor_speed"].shape[0] - 1, int(round(fps))))

    t0 = time.perf_counter()
    regions = []
    arr = np.asarray(data.channels["tensor_speed"], np.float32)
    for r in range(len(data.meta["replicate_tiles"])):
        blocks, _ = region_blocks_and_grid(data.meta, arr, r)
        bp = morlet_band_power(blocks, fps, freqs, i, j)
        # Operating point matched by quantile, not by absolute threshold.
        finite = bp[np.isfinite(bp)]
        lo = float(np.quantile(finite, VALUE_Q)) if finite.size else 0.0
        count = inband_count(bp, lo, np.inf)
        windowed = windowed_mean(count, win, True)
        thr = float(np.quantile(windowed, COUNT_Q)) if windowed.size else 0.0
        regions.append({
            "windowed": windowed,
            "gate": detect_gate(windowed, thr, np.inf),
            # Threshold-free: region-mean band power per frame. The gate metrics
            # below depend on where the quantiles land, so a disagreement there
            # can be thresholding rather than signal. This series has no
            # threshold in it at all, so it isolates "did the underlying
            # temporal signal change" from "did the operating point move".
            "signal": np.nanmean(bp, axis=1).astype(np.float32),
            "n_blocks": int(blocks.shape[1]),
        })
    detect_s = time.perf_counter() - t0

    return {
        "target_w": target_w, "block": block, "scale": scale,
        "extract_s": extract_s, "detect_s": detect_s,
        "grid": list(data.meta["grid"]), "regions": regions,
    }


def score(rung: dict, ref: dict) -> dict:
    """Agreement of a rung's per-region detection against the reference rung."""
    corrs, sigcorrs, agrees, ious = [], [], [], []
    for reg, rref in zip(rung["regions"], ref["regions"]):
        a, b = reg["windowed"], rref["windowed"]
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
        if n > 1 and a.std() > 0 and b.std() > 0:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
        sa, sb = reg["signal"][:n], rref["signal"][:n]
        ok = np.isfinite(sa) & np.isfinite(sb)
        if ok.sum() > 1 and sa[ok].std() > 0 and sb[ok].std() > 0:
            sigcorrs.append(float(np.corrcoef(sa[ok], sb[ok])[0, 1]))
        ga = reg["gate"][:n] > 0.5
        gb = rref["gate"][:n] > 0.5
        agrees.append(float((ga == gb).mean()))
        union = (ga | gb).sum()
        ious.append(float((ga & gb).sum() / union) if union else 1.0)
    return {
        "corr": float(np.mean(corrs)) if corrs else float("nan"),
        "sig_corr": float(np.mean(sigcorrs)) if sigcorrs else float("nan"),
        "gate_agree": float(np.mean(agrees)),
        "gate_iou": float(np.mean(ious)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=479,
                    help="frames to extract from the clip start")
    ap.add_argument("--band", type=float, nargs=2, default=(6.0, 10.0),
                    metavar=("LO", "HI"), help="frequency band in Hz")
    ap.add_argument("--ladder", default=None,
                    help="comma-separated width:block rungs, e.g. "
                         "'5200:64,2600:32,1300:16'. Block size must track the "
                         "width or the block grid stops being the control.")
    ap.add_argument("--reference", type=int, default=REFERENCE_WIDTH,
                    help="which rung to score the others against")
    args = ap.parse_args()

    ladder = LADDER
    if args.ladder:
        ladder = tuple(tuple(int(v) for v in rung.split(":"))
                       for rung in args.ladder.split(","))
    if args.reference not in [w for w, _ in ladder]:
        raise SystemExit(f"--reference {args.reference} is not in the ladder")

    replicates = load_replicates(args.video)
    with VideoSource(args.video) as src:
        info = src.info
    print(info.describe())
    print(f"{len(replicates)} replicates, band {args.band[0]:g}-{args.band[1]:g} Hz, "
          f"{args.frames} frames\n")

    rungs = {}
    for target_w, block in ladder:
        r = run_rung(args.video, replicates, info, target_w, block,
                     args.frames, tuple(args.band))
        rungs[target_w] = r
        print(f"  w={target_w:>5} block={block:>2} scale={r['scale']:.4f} "
              f"grid={r['grid']}  extract {r['extract_s']:6.2f}s  "
              f"detect {r['detect_s']:5.2f}s")

    ref = rungs[args.reference]
    print(f"\nAgreement vs w={args.reference} (matched operating point, "
          f"tensor_speed):")
    print(f"  {'width':>6} {'rel.px':>7} {'rel.time':>9} {'sig_corr':>9} "
          f"{'corr':>7} {'gate_agr':>9} {'gate_IoU':>9}")
    for target_w, _ in ladder:
        r = rungs[target_w]
        s = score(r, ref)
        relpx = (r["scale"] / ref["scale"]) ** 2
        reltime = r["extract_s"] / ref["extract_s"]
        print(f"  {target_w:>6} {relpx:>7.2f} {reltime:>9.2f} "
              f"{s['sig_corr']:>9.3f} {s['corr']:>7.3f} "
              f"{s['gate_agree']:>9.3f} {s['gate_iou']:>9.3f}")


if __name__ == "__main__":
    main()
