"""Benchmark: what does a per-block temporal-variance ("flicker") feature cost,
relative to the optical flow we already run?

The claim under test is that flicker energy is essentially free next to the
Farneback/DIS solve. We measure three per-frame costs on a representative
working-resolution tile:

  1. flow            -- one Farneback forward solve (config defaults). The thing
                        we already pay, once per frame, in stage 1.
  2. flicker         -- streaming sliding-window per-pixel temporal variance,
                        then block-reduced with the SAME reducer the pipeline
                        already uses for texture/fb-error. The proposed feature.
  3. texture_eigen   -- cv2.cornerMinEigenVal + block reduce. An OPTIONAL feature
                        the pipeline already computes per frame, included purely
                        as a "you already pay something in this ballpark" anchor.

Run:  python scripts/bench_flicker_cost.py [work_w] [work_h] [window_frames]

Defaults mirror config.py: block_size=16, Farneback winsize=15/levels=3/iters=3.
A single replicate tile at the default ~1300px working width is well under the
full frame; 720x720 is a generous stand-in for one box.
"""
from __future__ import annotations

import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, ".")
from core.flow import reduce_scalar_to_blocks  # noqa: E402

# -- config defaults, copied from core/config.py so this runs standalone -------
BLOCK = 16
FB = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5,
          poly_sigma=1.2, flags=0)
FPS = 60
CLIP_S = 10.0


def farneback(prev, curr):
    return cv2.calcOpticalFlowFarneback(
        prev, curr, None, FB["pyr_scale"], FB["levels"], FB["winsize"],
        FB["iterations"], FB["poly_n"], FB["poly_sigma"], FB["flags"])


class FlickerAccumulator:
    """Streaming per-pixel temporal variance over a sliding window, block-reduced.

    Keeps running sum (S1) and sum of squares (S2) over the last `win` frames.
    Per frame: two elementwise adds, one subtract of the evicted frame, one
    variance eval, one block reduce. No FFT, no per-pixel history scan.
    """
    def __init__(self, win: int, block: int):
        self.win = win
        self.block = block
        self.buf: deque[np.ndarray] = deque(maxlen=win)
        self.s1 = None
        self.s2 = None

    def update(self, gray: np.ndarray) -> np.ndarray | None:
        g = gray.astype(np.float32, copy=False)
        sq = g * g
        if self.s1 is None:
            self.s1 = np.zeros_like(g)
            self.s2 = np.zeros_like(g)
        if len(self.buf) == self.win:
            old = self.buf[0]
            self.s1 -= old
            self.s2 -= old * old
        self.buf.append(g)
        self.s1 += g
        self.s2 += sq
        n = len(self.buf)
        mean = self.s1 / n
        var = np.maximum(self.s2 / n - mean * mean, 0.0)  # clamp fp noise
        return reduce_scalar_to_blocks(var, self.block, "mean",
                                       include_partial=True)


def texture_eigen(gray: np.ndarray) -> np.ndarray:
    g8 = np.clip(gray, 0, 255).astype(np.uint8)
    eig = cv2.cornerMinEigenVal(g8, blockSize=3, ksize=3)
    return reduce_scalar_to_blocks(eig, BLOCK, "mean", include_partial=True)


def bench(fn, *args, iters=60, warmup=5):
    for _ in range(warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter() - t0) / iters


def main():
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 720
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 720
    win = int(sys.argv[3]) if len(sys.argv) > 3 else 30  # 0.5 s @ 60 fps

    rng = np.random.default_rng(0)
    # Textured frames so Farneback actually iterates rather than trivially bailing.
    base = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    prev = base.astype(np.float32)
    curr = np.roll(base, 2, axis=1).astype(np.float32) + rng.normal(0, 5, (h, w))
    prev_u8 = prev.astype(np.uint8)
    curr_u8 = curr.astype(np.uint8)

    ny, nx = h // BLOCK, w // BLOCK

    t_flow = bench(farneback, prev_u8, curr_u8)

    acc = FlickerAccumulator(win, BLOCK)
    for _ in range(win):  # prime the window so we time steady-state cost
        acc.update(rng.integers(0, 255, size=(h, w)).astype(np.float32))
    t_flick = bench(lambda: acc.update(
        rng.integers(0, 255, size=(h, w)).astype(np.float32)))

    t_tex = bench(texture_eigen, curr)

    # Cheap path: stage 1 only caches mean block intensity (one reduce, no
    # squaring, no window). Flicker energy is then derived post-hoc in stage 2.
    t_intensity = bench(
        lambda: reduce_scalar_to_blocks(curr, BLOCK, "mean", include_partial=True))

    n_frames = int(CLIP_S * FPS)
    print(f"\nworking tile: {w}x{h}  ->  block grid {ny}x{nx} = {ny*nx} blocks"
          f"   (block={BLOCK}, flicker window={win} frames)")
    print("-" * 68)
    print(f"{'per-frame':<22}{'ms':>10}{'x flow':>12}")
    print(f"{'flow (Farneback fwd)':<22}{t_flow*1e3:>10.3f}{1.0:>12.2f}")
    print(f"{'flicker (variance)':<22}{t_flick*1e3:>10.3f}"
          f"{t_flick/t_flow:>12.4f}")
    print(f"{'texture_eigen (exists)':<22}{t_tex*1e3:>10.3f}"
          f"{t_tex/t_flow:>12.4f}")
    print(f"{'intensity (cheap path)':<22}{t_intensity*1e3:>10.3f}"
          f"{t_intensity/t_flow:>12.4f}")
    print("-" * 68)
    print(f"stage-1 flow pass, {CLIP_S:g}s @ {FPS}fps = {n_frames} frames:")
    print(f"  flow only        : {t_flow*n_frames:8.2f} s")
    print(f"  + flicker        : +{t_flick*n_frames:7.2f} s"
          f"  ({100*t_flick/t_flow:.2f}% overhead)")

    # -- storage: one float per block per frame, same shape as speed/u/v -------
    for dt, name in ((np.float16, "float16"), (np.float32, "float32")):
        per = ny * nx * np.dtype(dt).itemsize
        print(f"\nstorage as per-frame per-block array ({name}, uncompressed):")
        print(f"  {per*n_frames/1e6:6.2f} MB / {CLIP_S:g}s"
              f"   ({per*FPS*60/1e6:.1f} MB/min, {per/1e3:.1f} KB/frame)")
    # Alternative: store only stage-2 windowed energy (one value per hop @ 4/s).
    hop_per_s = 4
    per32 = ny * nx * 4
    print(f"\nalt: stage-2 windowed energy only ({hop_per_s}/s, float32):")
    print(f"  {per32*hop_per_s*CLIP_S/1e6:6.2f} MB / {CLIP_S:g}s"
          f"   ({per32*hop_per_s*60/1e6:.1f} MB/min)")


if __name__ == "__main__":
    main()
