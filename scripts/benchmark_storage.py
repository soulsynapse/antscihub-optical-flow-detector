"""HDF5 vs Zarr, on data shaped exactly like a real full-clip cache.

The handoff calls for benchmarking both and committing to one. The three numbers
that matter, in priority order:

  1. Random-access read latency for scrubbing. This is the interactive path --
     the user drags the scrub bar and every feature overlay must repaint. If this
     is slow, the tool feels broken no matter what else is fast.
  2. ROI time-series read. Tab 3 pulls one block's full time column, which cuts
     across every time chunk. This is the access pattern the time-major chunking
     is worst at, so it is the one that could disqualify the layout.
  3. Compressed size and write throughput. These matter, but a one-off cost paid
     during a pass the user already knows is expensive.

Synthetic data is generated with realistic statistics (heavy-tailed speeds,
spatially correlated flow) because compression ratio depends entirely on the data
distribution -- benchmarking on random noise would understate both backends by
the same large factor and tell us nothing.
"""
from __future__ import annotations

import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import cache as cache_mod
from core.cache import human_bytes

BENCH_ROOT = os.path.join(".cache", "_bench")

# Shaped like the real thing: 8.5 min at 59.94 fps, 45x81 block grid.
N_FRAMES = 30600
NY, NX = 45, 81
DTYPE = "float16"
N_SCRUB_READS = 200
N_ROI_READS = 20


def make_realistic_flow(n: int, ny: int, nx: int, seed: int = 0):
    """Flow-like arrays: spatially smooth, heavy-tailed, mostly near zero.

    Real flow fields compress well precisely because they are smooth and sparse.
    Reproducing that here is what makes the size numbers meaningful.
    """
    rng = np.random.default_rng(seed)
    import cv2

    u = np.zeros((n, ny, nx), np.float32)
    v = np.zeros((n, ny, nx), np.float32)
    speed = np.zeros((n, ny, nx), np.float32)

    # A few moving "animals" against a still background.
    n_targets = 6
    pos = rng.uniform([0, 0], [ny, nx], size=(n_targets, 2))
    vel = rng.normal(0, 0.15, size=(n_targets, 2))
    wing_hz = rng.uniform(15, 24, size=n_targets)

    yy, xx = np.mgrid[0:ny, 0:nx]
    for t in range(n):
        fu = rng.normal(0, 0.3, (ny, nx)).astype(np.float32)
        fv = rng.normal(0, 0.3, (ny, nx)).astype(np.float32)
        fs = np.abs(rng.normal(0, 0.4, (ny, nx))).astype(np.float32)

        pos += vel
        pos[:, 0] %= ny
        pos[:, 1] %= nx
        for k in range(n_targets):
            d2 = (yy - pos[k, 0]) ** 2 + (xx - pos[k, 1]) ** 2
            blob = np.exp(-d2 / (2 * 2.5 ** 2)).astype(np.float32)
            osc = np.sin(2 * np.pi * wing_hz[k] * t / 59.94)
            fu += blob * (vel[k, 1] * 20 + osc * 12)
            fv += blob * (vel[k, 0] * 20 + osc * 4)
            fs += blob * (abs(vel[k]).sum() * 20 + abs(osc) * 30)

        u[t] = cv2.GaussianBlur(fu, (0, 0), 1.0)
        v[t] = cv2.GaussianBlur(fv, (0, 0), 1.0)
        speed[t] = cv2.GaussianBlur(fs, (0, 0), 1.0)
    return u, v, speed


def bench(backend: str, u, v, speed, compression: str = "zstd") -> dict:
    key = f"bench_{backend}_{compression}"
    meta = {
        "fps": 59.94, "n_frames": N_FRAMES, "block_size": 16,
        "grid": [NY, NX], "dtype": DTYPE, "compression": compression,
        "compression_level": 5, "features": ["u", "v", "speed"],
        "bands": [], "band_hop_s": 0.25, "band_window_s": 1.0,
    }

    t0 = time.perf_counter()
    cache = cache_mod.create_cache(BENCH_ROOT, key, meta, backend=backend)
    for name, arr in (("u", u), ("v", v), ("speed", speed)):
        cache.create_array(name, (N_FRAMES, NY, NX), DTYPE)
        for c0 in range(0, N_FRAMES, cache_mod.DEFAULT_CHUNK_FRAMES):
            c1 = min(N_FRAMES, c0 + cache_mod.DEFAULT_CHUNK_FRAMES)
            cache.write(name, c0, arr[c0:c1].astype(DTYPE))
    cache.close()
    write_s = time.perf_counter() - t0

    size = 0
    for dirpath, _, files in os.walk(cache_mod.cache_dir(BENCH_ROOT, key)):
        for f in files:
            size += os.path.getsize(os.path.join(dirpath, f))

    cache = cache_mod.open_cache(BENCH_ROOT, key)
    rng = np.random.default_rng(1)

    # 1. Scrub: random single-frame reads across all three features, which is
    #    exactly what repainting the overlay at a new scrub position costs.
    idxs = rng.integers(0, N_FRAMES, N_SCRUB_READS)
    t0 = time.perf_counter()
    for i in idxs:
        for name in ("u", "v", "speed"):
            cache.read_frame(name, int(i))
    scrub_ms = (time.perf_counter() - t0) / N_SCRUB_READS * 1000

    # 1b. Sequential scrub: stepping frame by frame, which is what dragging the
    #     scrub bar or playing back actually does. This is the number that has to
    #     fit inside a 16.7 ms frame budget, and it is the one the chunk cache is
    #     designed to serve -- consecutive frames share a chunk.
    cache.invalidate_chunk_cache()
    start = int(rng.integers(0, N_FRAMES - 300))
    t0 = time.perf_counter()
    for i in range(start, start + 300):
        for name in ("u", "v", "speed"):
            cache.read_frame(name, i)
    seq_ms = (time.perf_counter() - t0) / 300 * 1000

    # 2. ROI time series: one block's full time column. Worst case for
    #    time-major chunking.
    t0 = time.perf_counter()
    for _ in range(N_ROI_READS):
        r = int(rng.integers(0, NY))
        c = int(rng.integers(0, NX))
        _ = np.asarray(cache._dataset("speed")[:, r, c])
    roi_ms = (time.perf_counter() - t0) / N_ROI_READS * 1000

    # 3. Full sequential read (histogram build over the whole clip).
    t0 = time.perf_counter()
    _ = cache.read("speed")
    full_s = time.perf_counter() - t0

    cache.close()
    return {
        "backend": backend, "compression": compression,
        "write_s": write_s,
        "write_mbps": (N_FRAMES * NY * NX * 2 * 3 / 1e6) / write_s,
        "size": size,
        "scrub_ms": scrub_ms,
        "seq_ms": seq_ms,
        "roi_ms": roi_ms,
        "full_s": full_s,
    }


def main() -> int:
    raw = N_FRAMES * NY * NX * 2 * 3
    print(f"synthetic cache: 3 features x {N_FRAMES} frames x {NY}x{NX} blocks "
          f"{DTYPE}\nuncompressed = {human_bytes(raw)}\n")
    print("generating realistic flow data...")
    u, v, speed = make_realistic_flow(N_FRAMES, NY, NX)

    if os.path.exists(BENCH_ROOT):
        shutil.rmtree(BENCH_ROOT, ignore_errors=True)

    results = []
    for backend in ("zarr", "hdf5"):
        for comp in ("zstd", "none"):
            print(f"benchmarking {backend} / {comp} ...")
            try:
                results.append(bench(backend, u, v, speed, comp))
            except Exception as e:
                print(f"  FAILED: {e}")

    print(f"\n{'backend':<8} {'comp':<6} {'size':>10} {'ratio':>7} "
          f"{'write':>10} {'scrub':>10} {'ROI ts':>10} {'full read':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['backend']:<8} {r['compression']:<6} "
              f"{human_bytes(r['size']):>10} "
              f"{raw / r['size']:>6.1f}x "
              f"{r['write_mbps']:>7.0f} MB/s "
              f"{r['scrub_ms']:>8.2f} ms "
              f"{r['seq_ms']:>8.3f} ms "
              f"{r['roi_ms']:>7.2f} ms "
              f"{r['full_s']:>7.2f} s")

    print("\nscrub  = one random frame, all 3 features (the interactive path)")
    print("ROI ts = one block's full time column across the whole clip")

    z = next((r for r in results if r["backend"] == "zarr"
              and r["compression"] == "zstd"), None)
    h = next((r for r in results if r["backend"] == "hdf5"
              and r["compression"] == "zstd"), None)
    if z and h:
        print(f"\nzstd head to head (zarr vs hdf5):")
        print(f"  rnd scrub     : {z['scrub_ms']:.2f} vs {h['scrub_ms']:.2f} ms "
              f"({h['scrub_ms'] / z['scrub_ms']:.2f}x)")
        print(f"  seq scrub     : {z['seq_ms']:.3f} vs {h['seq_ms']:.3f} ms "
              f"({h['seq_ms'] / z['seq_ms']:.2f}x)")
        print(f"  ROI time series: {z['roi_ms']:.2f} vs {h['roi_ms']:.2f} ms "
              f"({h['roi_ms'] / z['roi_ms']:.2f}x)")
        print(f"  size          : {human_bytes(z['size'])} vs "
              f"{human_bytes(h['size'])}")
        print(f"  write         : {z['write_mbps']:.0f} vs "
              f"{h['write_mbps']:.0f} MB/s")

    shutil.rmtree(BENCH_ROOT, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
