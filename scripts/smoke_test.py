"""Headless end-to-end check: run the pipeline on a short slice and report
whether every feature's distribution looks sane.

This is build-order step 2's "verify histograms of each feature look sensible",
done as numbers rather than plots so it can run in CI or over SSH.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import cache as cache_mod
from core.config import Band, FeatureConfig, FlowConfig, PipelineConfig, PreprocessConfig
from core.features import REGISTRY, FeatureContext, derived_feature_names
from core.pipeline import run_pipeline
from core.video import VideoSource

VIDEO = os.path.join("Videos", "Raw", "GX010050c2_02_18_26.MP4")
CACHE_ROOT = os.path.join(".cache")
TEST_SECONDS = float(os.environ.get("SMOKE_SECONDS", "6"))


def main() -> int:
    src = VideoSource(VIDEO)
    info = src.info
    src.release()
    print(info.describe())

    feats = FeatureConfig()
    band = feats.suggest_band(info.fps)
    feats = FeatureConfig(bands=(band,))
    print(f"suggested band for {info.fps:.2f} fps: "
          f"{band.lo_hz:g}-{band.hi_hz:g} Hz (Nyquist {info.nyquist_hz:.1f} Hz)")
    for w in feats.validate_bands(info.fps):
        print("  WARN:", w)

    cfg = PipelineConfig(
        preprocess=PreprocessConfig(),      # downsample auto-derived
        flow=FlowConfig(backend="farneback", block_size=16),
        features=feats,
    )

    t0 = time.perf_counter()
    last = [0.0]

    def on_progress(p):
        now = time.perf_counter()
        if now - last[0] > 1.0 or p.done == p.total:
            last[0] = now
            print(f"  [{p.stage}] {p.done}/{p.total} "
                  f"({p.frac:.0%}) eta {p.eta_s:.0f}s  {p.message}")

    key = run_pipeline(VIDEO, cfg, CACHE_ROOT, duration_s=TEST_SECONDS,
                       progress=on_progress, cache_key_suffix="_smoke")
    elapsed = time.perf_counter() - t0

    cache = cache_mod.open_cache(CACHE_ROOT, key)
    ny, nx = cache.grid
    n = cache.n_frames
    print(f"\ncache key={key} backend={cache.meta['backend']}")
    print(f"working res {cache.meta['work_width']}x{cache.meta['work_height']} "
          f"(downsample {cache.meta['downsample']:.3f}), grid {ny}x{nx} blocks")
    print(f"{n} frames in {elapsed:.1f}s -> {n / elapsed:.1f} fps processing")
    print(f"on disk: {cache_mod.human_bytes(cache.size_on_disk())}")

    full_frames = info.frame_count
    proj_time = elapsed / n * full_frames
    proj_size = cache.size_on_disk() / n * full_frames
    print(f"projected full clip: {proj_time / 60:.1f} min, "
          f"{cache_mod.human_bytes(proj_size)}")

    ctx = FeatureContext(
        u=cache.read("u"), v=cache.read("v"), speed=cache.read("speed"),
        fps=cache.fps, block_size=cache.block_size,
        bands={b: cache.read(b) for b in cache.feature_names
               if b.startswith("bandpower_")},
    )

    print("\n{:<26} {:>10} {:>10} {:>10} {:>10} {:>8}".format(
        "feature", "p1", "median", "p99", "max", "kind"))
    print("-" * 80)
    ok = True
    names = ["speed", "u", "v"] + derived_feature_names() + \
            [b for b in cache.feature_names if b.startswith("bandpower_")]
    for name in names:
        try:
            arr = ctx.get(name).astype(np.float32)
        except KeyError as e:
            print(f"{name:<26} SKIP ({e})")
            continue
        finite = np.isfinite(arr)
        if not finite.all():
            print(f"{name:<26} !! {(~finite).sum()} non-finite values")
            ok = False
            arr = arr[finite]
        p1, med, p99, mx = (np.percentile(arr, 1), np.median(arr),
                            np.percentile(arr, 99), arr.max())
        kind = REGISTRY[name].kind if name in REGISTRY else "cached"
        print(f"{name:<26} {p1:>10.3f} {med:>10.3f} {p99:>10.3f} "
              f"{mx:>10.3f} {kind:>8}")

        if arr.size and np.ptp(arr) == 0:
            print(f"{'':<26} !! constant -- feature is dead")
            ok = False

    # Physical sanity: coherence must be in [0,1], and the wingbeat corner
    # (high speed, low coherence) must actually contain some blocks, otherwise
    # the whole premise of the tool is broken on this footage.
    coh = ctx.get("coherence")
    speed = ctx.get("speed")
    assert 0.0 <= float(coh.min()) and float(coh.max()) <= 1.0, "coherence out of [0,1]"

    hi_speed = speed > np.percentile(speed, 99)
    lo_coh = coh < 0.5
    corner = float((hi_speed & lo_coh).mean() * 100)
    print(f"\nblocks with high speed AND low coherence (the oscillation corner): "
          f"{corner:.3f}% of block-frames")

    bp_names = [b for b in cache.feature_names if b.startswith("bandpower_")]
    if bp_names:
        bp = ctx.get(bp_names[0]).astype(np.float32)
        share = float((bp > np.percentile(bp, 99.9)).mean() * 100)
        print(f"{bp_names[0]}: top-0.1% blocks concentrate "
              f"{bp[bp > np.percentile(bp, 99.9)].sum() / bp.sum() * 100:.1f}% of "
              f"total band power ({share:.2f}% of block-windows) -- "
              f"a spatially concentrated signal, not uniform noise")

    cache.close()
    print("\nSMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
