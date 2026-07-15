"""The flow pass: video -> preprocess -> flow -> block reduce -> cache.

Runs in two stages.

Stage 1 (streaming, expensive): decode each frame once, preprocess it, compute
flow against the previous frame, reduce to the block grid, append (u, v, speed)
to disk. This is O(frames) with a large constant and is the only part that
touches the decoder.

Stage 2 (post-hoc, cheap): band-power needs a time series, which stage 1 does
not have until it finishes. So band-power is computed afterwards by reading the
cached `speed` array back and running a sliding-window FFT along the time axis.
Reading it back costs seconds; recomputing flow would cost hours.

Qt-free by design: the GUI drives this from a worker thread and receives progress
through a plain callback.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from core import cache as cache_mod
from core.config import PipelineConfig
from core.features import cached_feature_names
from core.flow import create_backend, reduce_to_blocks
from core.preprocess import Preprocessor, sample_frames_for_background
from core.video import VideoSource

ProgressFn = Callable[["Progress"], None]


@dataclass
class Progress:
    stage: str
    done: int
    total: int
    elapsed_s: float
    message: str = ""

    @property
    def frac(self) -> float:
        return self.done / self.total if self.total else 0.0

    @property
    def eta_s(self) -> float:
        if self.done <= 0:
            return float("nan")
        rate = self.elapsed_s / self.done
        return rate * (self.total - self.done)


class Cancelled(Exception):
    pass


def run_pipeline(
    video_path: str,
    cfg: PipelineConfig,
    cache_root: str,
    duration_s: float | None = None,
    progress: ProgressFn | None = None,
    should_cancel: Callable[[], bool] | None = None,
    backend: str | None = None,
    cache_key_suffix: str = "",
) -> str:
    """Run the full pass and return the cache key.

    duration_s=None processes the whole video (full mode). A value processes only
    the first N seconds (test mode) into a separately-keyed scratch cache, so a
    test run never collides with or invalidates a completed full run.
    """
    t_start = time.perf_counter()
    should_cancel = should_cancel or (lambda: False)

    def emit(stage: str, done: int, total: int, msg: str = "") -> None:
        if progress:
            progress(Progress(stage, done, total,
                              time.perf_counter() - t_start, msg))
        if should_cancel():
            raise Cancelled()

    src = VideoSource(video_path)
    info = src.info
    fps = info.fps

    n_frames = info.frame_count
    if duration_s is not None:
        n_frames = min(n_frames, max(2, int(round(duration_s * fps))))

    key = cfg.cache_key(info.video_hash) + cache_key_suffix

    pre = Preprocessor(cfg.preprocess, info.width, info.height)
    if cfg.preprocess.bg_subtract == "median":
        emit("background", 0, 1, "Sampling frames for background model")
        pre.fit_background(
            sample_frames_for_background(src, cfg.preprocess.bg_median_samples))

    if cfg.preprocess.registration != "off":
        ref = src.frame_at(src.time_to_frame(cfg.preprocess.registration_ref_time_s))
        if ref is not None:
            pre.set_reference(ref)

    block = cfg.flow.block_size
    ny, nx = pre.height // block, pre.width // block
    if ny == 0 or nx == 0:
        raise ValueError(
            f"Block size {block} exceeds the working frame "
            f"({pre.width}x{pre.height}). Lower the block size."
        )

    # Flow is defined between consecutive frames, so N frames yield N-1 fields.
    # We prepend a zero field so the arrays are frame-aligned: index t is the
    # motion arriving at frame t. Off-by-one here would shift every time series
    # relative to the video by one frame.
    n_flow = n_frames
    dtype = cfg.features.dtype

    meta = {
        "video_path": video_path,
        "video_hash": info.video_hash,
        "fps": fps,
        "n_frames": n_flow,
        "src_width": info.width,
        "src_height": info.height,
        "work_width": pre.width,
        "work_height": pre.height,
        "downsample": pre.scale,
        "block_size": block,
        "grid": [ny, nx],
        "dtype": dtype,
        "compression": cfg.features.compression,
        "compression_level": cfg.features.compression_level,
        "bands": [{"lo_hz": b.lo_hz, "hi_hz": b.hi_hz} for b in cfg.features.bands],
        "band_window_s": cfg.features.window_s,
        "band_hop_s": cfg.features.hop_s,
        "features": cached_feature_names(cfg),
        "config": cfg.to_dict(),
        "test_mode": duration_s is not None,
        "duration_s": n_flow / fps,
    }

    cache = cache_mod.create_cache(cache_root, key, meta, backend=backend)
    try:
        for name in ("u", "v", "speed"):
            cache.create_array(name, (n_flow, ny, nx), dtype)
        if cfg.features.cache_coherence:
            cache.create_array("coherence", (n_flow, ny, nx), dtype)
        if cfg.features.cache_divergence_curl:
            cache.create_array("divergence", (n_flow, ny, nx), dtype)
            cache.create_array("curl", (n_flow, ny, nx), dtype)

        # -- stage 1: streaming flow -----------------------------------------
        fb = create_backend(cfg.flow)
        buf_u = np.zeros((cache_mod.DEFAULT_CHUNK_FRAMES, ny, nx), np.float32)
        buf_v = np.zeros_like(buf_u)
        buf_s = np.zeros_like(buf_u)
        buf_at = 0
        write_t0 = 0

        def flush() -> None:
            nonlocal buf_at, write_t0
            if buf_at == 0:
                return
            u, v, s = buf_u[:buf_at], buf_v[:buf_at], buf_s[:buf_at]
            cache.write("u", write_t0, u.astype(dtype))
            cache.write("v", write_t0, v.astype(dtype))
            cache.write("speed", write_t0, s.astype(dtype))
            if cfg.features.cache_coherence:
                coh = np.hypot(u, v) / (s + 1e-6)
                cache.write("coherence", write_t0,
                            np.clip(coh, 0, 1).astype(dtype))
            if cfg.features.cache_divergence_curl:
                div = np.gradient(u, axis=2) + np.gradient(v, axis=1)
                curl = np.gradient(v, axis=2) - np.gradient(u, axis=1)
                cache.write("divergence", write_t0, div.astype(dtype))
                cache.write("curl", write_t0, curl.astype(dtype))
            write_t0 += buf_at
            buf_at = 0

        prev = None
        for i, frame in src.iter_frames(0, n_frames):
            g = pre.apply(frame)
            if prev is None:
                # Frame 0 has no predecessor: zero flow, keeping arrays aligned.
                buf_u[buf_at] = 0.0
                buf_v[buf_at] = 0.0
                buf_s[buf_at] = 0.0
            else:
                flow = fb.compute(prev, g)
                u, v, s = reduce_to_blocks(flow, block, fps)
                buf_u[buf_at], buf_v[buf_at], buf_s[buf_at] = u, v, s
            prev = g
            buf_at += 1

            if buf_at == buf_u.shape[0]:
                flush()
            if i % 20 == 0:
                emit("flow", i + 1, n_frames,
                     f"Optical flow ({cfg.flow.backend}) at "
                     f"{pre.width}x{pre.height}")
        flush()
        fb.close()
        emit("flow", n_frames, n_frames, "Flow complete")

        # -- stage 2: band-power ---------------------------------------------
        if cfg.features.bands:
            _compute_band_power(cache, cfg, fps, n_flow, ny, nx, dtype, emit)

    finally:
        src.release()
        cache.close()

    return key


def _sliding_windows(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """(T, B) -> (n_win, win, B) view via strides. No copy."""
    T = x.shape[0]
    n_win = 1 + (T - win) // hop
    s0, s1 = x.strides
    return np.lib.stride_tricks.as_strided(
        x, shape=(n_win, win, x.shape[1]), strides=(s0 * hop, s0, s1),
        writeable=False,
    )


def _compute_band_power(cache, cfg: PipelineConfig, fps: float, n_frames: int,
                        ny: int, nx: int, dtype: str, emit) -> None:
    """Sliding-window band-power from the cached speed time series.

    Band power lives on its own time axis -- one value per hop, not per frame --
    because the STFT window already smears it over window_s seconds. Storing it
    per frame would inflate it by hop_s*fps with no extra information.
    """
    win = max(4, int(round(cfg.features.window_s * fps)))
    hop = max(1, int(round(cfg.features.hop_s * fps)))
    if n_frames < win:
        win = max(4, n_frames // 2)
        hop = max(1, win // 4)

    n_win = 1 + (n_frames - win) // hop
    freqs = np.fft.rfftfreq(win, d=1.0 / fps)
    w = np.hanning(win).astype(np.float32)
    norm = fps * np.sum(w ** 2)

    # Band power is stored as float32 regardless of the cache's dtype setting.
    # It is a sum of PSD values over a band and can easily exceed float16's
    # 65504 ceiling on a wide band or high-contrast footage -- when it does,
    # float16 silently saturates to +inf and every histogram built on it is
    # ruined. Band arrays are per-window (one value per hop, not per frame), so
    # they are a tiny fraction of the cache and paying float32 for them costs
    # almost nothing.
    band_dtype = "float32"
    band_names = [b.label() for b in cfg.features.bands]
    band_masks = [
        (freqs >= b.lo_hz) & (freqs <= b.hi_hz) for b in cfg.features.bands
    ]
    for name in band_names:
        cache.create_array(name, (n_win, ny, nx), band_dtype)

    extra_flat = cfg.features.cache_spectral_flatness
    extra_osc = cfg.features.cache_direction_oscillation
    if extra_flat:
        cache.create_array("spectral_flatness", (n_win, ny, nx), "float32")
    if extra_osc:
        cache.create_array("direction_oscillation", (n_win, ny, nx), "float32")

    cache.meta["n_band_windows"] = n_win
    cache.meta["band_freqs_hz"] = [float(f) for f in freqs]
    cache.write_meta()

    # Process a stripe of block-rows at a time.
    #
    # Get this budget wrong and the full clip dies. The windowed view has shape
    # (n_win, win, blocks) -- it does NOT scale with the number of frames, it
    # scales with the number of WINDOWS, and at the default 0.25 s hop there are
    # four windows per second. On the 8.5-minute reference clip that is 2036
    # windows x 60 frames/window x 3645 blocks = 445M float32 = 1.8 GB for the
    # detrended copy alone, and the FFT output on top of it. A budget that counts
    # only `win` and `nx` (as an earlier version of this did) happily hands you
    # the whole grid and then runs the machine out of memory.
    #
    # So bound the actual allocation: elements per block-row is n_win * win * nx.
    BUDGET_ELEMS = 20_000_000          # ~80 MB per float32 working array
    elems_per_row = max(1, n_win * win * nx)
    rows_per_chunk = max(1, min(ny, BUDGET_ELEMS // elems_per_row))

    for r0 in range(0, ny, rows_per_chunk):
        r1 = min(ny, r0 + rows_per_chunk)
        speed = cache.read_rows("speed", r0, r1).astype(np.float32)
        B = (r1 - r0) * nx
        flat = np.ascontiguousarray(speed.reshape(n_frames, B))

        wins = _sliding_windows(flat, win, hop)          # (n_win, win, B)
        seg = wins - wins.mean(axis=1, keepdims=True)    # detrend: kill DC
        seg = seg * w[None, :, None]
        # np.fft.rfft promotes to complex128 regardless of input dtype, so the
        # power array would be float64 -- twice the memory for precision that
        # means nothing here. Cast straight back down.
        spec = np.fft.rfft(seg, axis=1)
        psd = (np.abs(spec).astype(np.float32) ** 2) / np.float32(norm)

        df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        for name, mask in zip(band_names, band_masks):
            if not mask.any():
                bp = np.zeros((psd.shape[0], B), np.float32)
            else:
                bp = psd[:, mask, :].sum(axis=1) * df
            cache.write_partial_rows(name, bp.reshape(-1, r1 - r0, nx),
                                     r0, r1, band_dtype)

        if extra_flat:
            p = psd[:, 1:, :] + 1e-12
            gm = np.exp(np.mean(np.log(p), axis=1))
            am = np.mean(p, axis=1)
            sf = np.clip(gm / (am + 1e-12), 0, 1)
            cache.write_partial_rows("spectral_flatness",
                                     sf.reshape(-1, r1 - r0, nx), r0, r1, "float32")

        if extra_osc:
            u = cache.read_rows("u", r0, r1).astype(np.float32).reshape(n_frames, B)
            v = cache.read_rows("v", r0, r1).astype(np.float32).reshape(n_frames, B)
            n = np.hypot(u, v) + 1e-6
            ux, uy = u / n, v / n
            dot = np.zeros_like(ux)
            dot[1:] = ux[1:] * ux[:-1] + uy[1:] * uy[:-1]
            dw = _sliding_windows(np.ascontiguousarray(dot), win, hop)
            osc = dw.mean(axis=1)
            cache.write_partial_rows("direction_oscillation",
                                     osc.reshape(-1, r1 - r0, nx), r0, r1, "float32")

        emit("bandpower", r1, ny, f"Band power over {len(band_names)} band(s)")

    emit("bandpower", ny, ny, "Band power complete")
