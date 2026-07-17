"""POC: does a per-channel Morlet scalogram layer buy new, cache-worthy signal?

Runs on an existing feature cache (defaults to the stabilized hopper test cache).
It does NOT modify the cache or the pipeline. It:

  1. loads the structure-tensor channels (change/appearance/tensor_speed/...),
  2. computes an FFT-based Morlet scalogram (no pywt; scipy.signal.cwt is gone),
     per replicate and per-block on one tile,
  3. measures build time and storage, extrapolates to multi-hour footage,
  4. renders a figure where the scalogram exposes rhythmic structure that the
     single cached band power collapses,
  5. prints the storage math that decides the cache-appropriate form
     (full-rate / hop-subsampled / multiband).

Run:
    .venv\\Scripts\\python.exe scripts/poc_expanded_cache.py
    .venv\\Scripts\\python.exe scripts/poc_expanded_cache.py --cache <key> --channel change
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import cache as cache_mod
from core.tensor_channels import (CHANNEL_VERSION, CHANNELS, _tiles_from_meta,
                                  load_or_extract_channels)
from core.wavelet import morlet_power


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def main() -> int:
    default_key = ("8a6f4228668ab353_test10s_farneback_b1_dsauto_regoff_"
                   "denoff_bgoff_normzscore_band12to24_win1_hop0p25_f16_zstd")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=default_key)
    ap.add_argument("--cache-root", default=os.path.join(".", ".cache"))
    ap.add_argument("--channel", default="change", choices=list(CHANNELS))
    ap.add_argument("--fmin", type=float, default=0.5)
    ap.add_argument("--fmax", type=float, default=25.0)
    ap.add_argument("--nscales", type=int, default=24)
    ap.add_argument("--out", default=None, help="figure path (png)")
    args = ap.parse_args()

    cache = cache_mod.open_cache(args.cache_root, args.cache)
    meta = cache.meta
    fps = float(meta["fps"])
    ny, nx = map(int, meta["grid"])
    n_frames = int(meta["n_frames"])
    tiles = _tiles_from_meta(meta)
    fmax = min(args.fmax, 0.45 * fps)
    freqs = np.geomspace(args.fmin, fmax, args.nscales)

    print(f"cache      : {args.cache}")
    print(f"video      : {os.path.basename(meta.get('video_path','?'))}")
    print(f"grid       : {ny} x {nx} = {ny*nx:,} blocks, block_size="
          f"{meta.get('block_size')}")
    print(f"frames     : {n_frames} @ {fps:.2f} fps "
          f"({n_frames/fps:.1f}s), {len(tiles)} replicate tiles")
    print(f"channel    : {args.channel}")
    print(f"scalogram  : {args.nscales} Morlet scales, "
          f"{freqs[0]:.2f}-{freqs[-1]:.2f} Hz\n")

    # -- 1. tensor channels (currently a second decode into an .npz sidecar) --
    sidecar = os.path.join(args.cache_root, args.cache,
                           f"tensor_channels_v{CHANNEL_VERSION}.npz")
    had_sidecar = os.path.exists(sidecar)
    t0 = time.perf_counter()
    ch = load_or_extract_channels(cache, sidecar_path=sidecar)
    t_channels = time.perf_counter() - t0
    print(f"[1] tensor channels {'(sidecar reuse)' if had_sidecar else '(extracted)'}"
          f": {t_channels:.2f}s")

    data = ch[args.channel].astype(np.float32)          # (T, ny, nx)
    T = data.shape[0]

    # -- 2a. per-replicate pooled scalogram (the cheap, always-affordable form) -
    pooled = np.zeros((T, len(tiles)), np.float32)
    tile_ids = []
    for j, tl in enumerate(tiles):
        ay0, ax0, ay1, ax1 = tl["atlas_bbox"]
        sub = data[:, ay0:ay1, ax0:ax1].reshape(T, -1)
        pooled[:, j] = sub.mean(axis=1)
        tile_ids.append(tl["id"])
    t0 = time.perf_counter()
    pooled_sg = morlet_power(pooled, fps, freqs)        # (F,T,ntiles)
    t_pooled = time.perf_counter() - t0
    # most active replicate = largest temporal variance of the channel
    active = int(np.argmax(pooled.var(axis=0)))
    print(f"[2a] per-replicate scalogram ({len(tiles)} tiles): {t_pooled:.3f}s")
    print(f"     most active replicate: index {active} (tile id {tile_ids[active]})")

    # -- 2b. per-block scalogram on the most active tile (timing + storage rate) -
    tl = tiles[active]
    ay0, ax0, ay1, ax1 = tl["atlas_bbox"]
    tile_blocks = data[:, ay0:ay1, ax0:ax1].reshape(T, -1)
    nb = tile_blocks.shape[1]
    t0 = time.perf_counter()
    tile_sg = morlet_power(tile_blocks, fps, freqs)     # (F,T,nb)
    t_tile = time.perf_counter() - t0
    rate = (nb * T) / max(t_tile, 1e-9)                 # block-frames / s
    print(f"[2b] per-block scalogram on active tile: {nb} blocks x {T} frames "
          f"in {t_tile:.3f}s  ({rate:,.0f} block-frames/s)\n")

    # -- 3. storage math for the full grid, and multi-hour extrapolation -------
    bytes_per = 4  # float32 scalogram
    full_rate_per_frame = ny * nx * args.nscales * bytes_per
    hop_s = float(meta.get("band_hop_s", 0.25))
    win_s = float(meta.get("band_window_s", 1.0))
    hop = max(1, int(round(hop_s * fps)))
    print("[3] storage per SECOND of video, full grid "
          f"({ny*nx:,} blocks, {args.nscales} scales):")
    print(f"    full-rate scalogram : {_fmt_bytes(full_rate_per_frame*fps)}/s")
    print(f"    hop-subsampled ({hop_s}s hop): "
          f"{_fmt_bytes(full_rate_per_frame*fps/hop)}/s")
    print(f"    multiband (4 bands) : "
          f"{_fmt_bytes(ny*nx*4*bytes_per*fps/hop)}/s")
    print(f"    (for reference) raw u/v/speed float16: "
          f"{_fmt_bytes(ny*nx*3*2*fps)}/s")
    for hours in (1, 3):
        secs = hours * 3600
        print(f"    --> {hours}h clip: full-rate="
              f"{_fmt_bytes(full_rate_per_frame*fps*secs)}, "
              f"hop={_fmt_bytes(full_rate_per_frame*fps/hop*secs)}, "
              f"multiband={_fmt_bytes(ny*nx*4*bytes_per*fps/hop*secs)}")
    est_full_grid_s = (ny * nx * T) / max(rate, 1e-9)
    print(f"    compute to scalogram the full grid once (this clip): "
          f"~{est_full_grid_s:.1f}s at measured rate\n")

    # -- 4. figure -------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib unavailable: {e}; skipping figure)")
        cache.close()
        return 0

    tvec = np.arange(T) / fps
    band_lo, band_hi = 12.0, 24.0  # the single band this cache actually holds
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2, 1]})

    axes[0].plot(tvec, pooled[:, active], lw=0.8, color="#1f77b4")
    axes[0].set_ylabel(f"{args.channel}\n(replicate mean)")
    axes[0].set_title(f"Replicate {tile_ids[active]} — raw channel vs Morlet "
                      f"scalogram vs the single cached {band_lo:g}-{band_hi:g} Hz band")

    sg = pooled_sg[:, :, active]                        # (F,T)
    logp = np.log10(sg + 1e-6)
    im = axes[1].pcolormesh(tvec, freqs, logp, shading="nearest", cmap="magma")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("frequency (Hz)")
    axes[1].axhline(band_lo, color="cyan", lw=0.8, ls="--")
    axes[1].axhline(band_hi, color="cyan", lw=0.8, ls="--")
    fig.colorbar(im, ax=axes[1], label="log10 power", pad=0.01)

    band_mask = (freqs >= band_lo) & (freqs <= band_hi)
    band_series = sg[band_mask].sum(axis=0) if band_mask.any() else np.zeros(T)
    total_series = sg.sum(axis=0)
    axes[2].plot(tvec, total_series, lw=0.8, color="#444", label="all-band power")
    axes[2].plot(tvec, band_series, lw=0.8, color="cyan",
                 label=f"{band_lo:g}-{band_hi:g} Hz band only")
    axes[2].set_ylabel("power")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right", fontsize=8)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "poc_expanded_cache.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[4] figure -> {os.path.abspath(out)}")

    cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
