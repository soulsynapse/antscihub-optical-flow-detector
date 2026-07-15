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
from dataclasses import dataclass, replace
from typing import Callable

import cv2
import numpy as np

from core import cache as cache_mod
from core.config import PipelineConfig
from core.features import cached_feature_names
from core.flow import (create_backend, forward_backward_error,
                       reduce_scalar_to_blocks, reduce_to_blocks)
from core.preprocess import Preprocessor
from core.replicates import ReplicateLayout, build_layout
from core.video import ReplicateVideoSource, VideoSource

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


def _flow_support_pixels(cfg: PipelineConfig) -> int:
    """Private synthetic edge support; never sourced outside the replicate."""
    if cfg.flow.backend == "farneback":
        return max(4, min(32, int(cfg.flow.fb_winsize)))
    return 16


def _flow_atlas_geometry(layout: ReplicateLayout, support: int
                         ) -> tuple[tuple[int, int], dict[int, tuple[slice, slice]]]:
    """Pack privately supported crops into one solver image.

    Batching removes the large fixed overhead of invoking Farnebäck/DIS once per
    replicate per frame. Every core is still separated from every other core by
    its own reflected support on both sides plus an all-zero guard gap.
    """
    gap = max(2, 2 * support)
    width = max(t.work_width + 2 * support for t in layout.tiles)
    height = sum(t.work_height + 2 * support for t in layout.tiles) + \
        gap * max(0, len(layout.tiles) - 1)
    cores = {}
    y = 0
    for tile in layout.tiles:
        cores[tile.replicate_id] = (
            slice(y + support, y + support + tile.work_height),
            slice(support, support + tile.work_width),
        )
        y += tile.work_height + 2 * support + gap
    return (height, width), cores


def _pack_flow_atlas(images: dict[int, np.ndarray], layout: ReplicateLayout,
                     support: int, shape: tuple[int, int]) -> np.ndarray:
    atlas = np.zeros(shape, dtype=np.float32)
    gap = max(2, 2 * support)
    y = 0
    for tile in layout.tiles:
        g = images[tile.replicate_id]
        if g.shape != (tile.work_height, tile.work_width):
            raise ValueError(
                f"Replicate {tile.replicate_id} produced {g.shape}, expected "
                f"{(tile.work_height, tile.work_width)}.")
        border = cv2.BORDER_REFLECT_101 \
            if g.shape[0] > 1 and g.shape[1] > 1 else cv2.BORDER_REPLICATE
        padded = cv2.copyMakeBorder(
            g, support, support, support, support, border)
        atlas[y:y + padded.shape[0], :padded.shape[1]] = padded
        y += padded.shape[0] + gap
    return atlas


def _region_divergence_curl(u: np.ndarray, v: np.ndarray,
                            layout: ReplicateLayout
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Spatial derivatives independently inside every packed replicate tile."""
    div = np.zeros_like(u, dtype=np.float32)
    curl = np.zeros_like(v, dtype=np.float32)
    for tile in layout.tiles:
        y0, x0, y1, x1 = tile.atlas_bbox
        us = u[:, y0:y1, x0:x1]
        vs = v[:, y0:y1, x0:x1]
        # np.gradient needs at least two values on an axis. A one-block-wide
        # replicate has no resolvable spatial derivative on that axis, so its
        # contribution is correctly zero.
        du_dx = np.gradient(us, axis=2) if us.shape[2] > 1 else np.zeros_like(us)
        dv_dx = np.gradient(vs, axis=2) if vs.shape[2] > 1 else np.zeros_like(vs)
        du_dy = np.gradient(us, axis=1) if us.shape[1] > 1 else np.zeros_like(us)
        dv_dy = np.gradient(vs, axis=1) if vs.shape[1] > 1 else np.zeros_like(vs)
        div[:, y0:y1, x0:x1] = du_dx + dv_dy
        curl[:, y0:y1, x0:x1] = dv_dx - du_dy
    return div, curl


def run_pipeline(
    video_path: str,
    cfg: PipelineConfig,
    cache_root: str,
    duration_s: float | None = None,
    progress: ProgressFn | None = None,
    should_cancel: Callable[[], bool] | None = None,
    backend: str | None = None,
    cache_key_suffix: str = "",
    replicates: list[dict] | None = None,
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

    replicates = list(replicates or [])
    block = cfg.flow.block_size
    scale = cfg.preprocess.resolve_downsample(info.width)
    layout = build_layout(replicates, info.width, info.height, scale, block)
    key = cfg.cache_key(info.video_hash, layout.geometry_hash) + cache_key_suffix

    # Preprocessing is stateful, so every replicate owns an independent
    # instance. The explicit full-frame-derived scale preserves the same
    # physical pixel scale across differently sized boxes.
    pre_cfg = replace(cfg.preprocess, downsample=scale, mask_path=None)
    source_mask = None
    if cfg.preprocess.mask_path:
        source_mask = cv2.imread(cfg.preprocess.mask_path, cv2.IMREAD_GRAYSCALE)
        if source_mask is None:
            raise IOError(f"Could not read mask: {cfg.preprocess.mask_path}")
        if source_mask.shape != (info.height, info.width):
            source_mask = cv2.resize(source_mask, (info.width, info.height),
                                     interpolation=cv2.INTER_NEAREST)

    preprocessors: dict[int, Preprocessor] = {}
    for tile in layout.tiles:
        x0, y0, x1, y1 = tile.source_box
        mask_crop = source_mask[y0:y1, x0:x1] if source_mask is not None else None
        preprocessors[tile.replicate_id] = Preprocessor(
            pre_cfg, x1 - x0, y1 - y0, mask_image=mask_crop)

    if cfg.preprocess.bg_subtract == "median":
        emit("background", 0, 1,
             "Sampling a separate background model for each replicate")
        sample_count = min(cfg.preprocess.bg_median_samples, info.frame_count)
        sample_idxs = np.linspace(0, max(0, info.frame_count - 1),
                                  num=sample_count, dtype=int)
        samples = {tile.replicate_id: [] for tile in layout.tiles}
        for idx in sample_idxs:
            frame = src.frame_at(int(idx))
            if frame is None:
                continue
            for tile in layout.tiles:
                x0, y0, x1, y1 = tile.source_box
                samples[tile.replicate_id].append(frame[y0:y1, x0:x1])
        for tile in layout.tiles:
            preprocessors[tile.replicate_id].fit_background(
                samples[tile.replicate_id])

    if cfg.preprocess.registration != "off":
        ref = src.frame_at(src.time_to_frame(cfg.preprocess.registration_ref_time_s))
        if ref is not None:
            for tile in layout.tiles:
                x0, y0, x1, y1 = tile.source_box
                preprocessors[tile.replicate_id].set_reference(ref[y0:y1, x0:x1])

    ny, nx = layout.atlas_grid
    support_px = _flow_support_pixels(cfg)
    flow_atlas_shape, flow_core_slices = _flow_atlas_geometry(layout, support_px)
    roi_decoder = None
    decoder_name = "opencv-full-frame"
    decoder_fallback = None
    try:
        roi_decoder = ReplicateVideoSource(video_path, layout, n_frames)
        decoder_name = "ffmpeg-roi-gray"
    except (FileNotFoundError, OSError, RuntimeError) as e:
        decoder_fallback = f"{type(e).__name__}: {e}"

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
        "work_width": nx * block,
        "work_height": ny * block,
        "work_pixels_per_frame": layout.work_pixels_per_frame,
        "downsample": layout.scale,
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
        "processing_scope": "replicate",
        "replicate_layout_version": 1,
        "replicate_geometry_hash": layout.geometry_hash,
        "replicate_tiles": [tile.to_meta() for tile in layout.tiles],
        "synthetic_flow_support_px": support_px,
        "flow_solver_atlas_shape": list(flow_atlas_shape),
        "source_padding_px": 0,
        "decoder": decoder_name,
        "decoder_fallback": decoder_fallback,
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
        if cfg.features.cache_fb_error:
            cache.create_array("fb_error_p90", (n_flow, ny, nx), dtype)
        if cfg.features.cache_texture:
            cache.create_array("texture_min_eigen", (n_flow, ny, nx), dtype)

        # -- stage 1: streaming flow -----------------------------------------
        flow_backend = create_backend(cfg.flow)
        buf_u = np.zeros((cache_mod.DEFAULT_CHUNK_FRAMES, ny, nx), np.float32)
        buf_v = np.zeros_like(buf_u)
        buf_s = np.zeros_like(buf_u)
        buf_fb_error = np.zeros_like(buf_u) if cfg.features.cache_fb_error else None
        buf_texture = np.zeros_like(buf_u) if cfg.features.cache_texture else None
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
            if buf_fb_error is not None:
                cache.write("fb_error_p90", write_t0,
                            buf_fb_error[:buf_at].astype(dtype))
            if buf_texture is not None:
                cache.write("texture_min_eigen", write_t0,
                            buf_texture[:buf_at].astype(dtype))
            if cfg.features.cache_coherence:
                coh = np.hypot(u, v) / (s + 1e-6)
                cache.write("coherence", write_t0,
                            np.clip(coh, 0, 1).astype(dtype))
            if cfg.features.cache_divergence_curl:
                div, curl = _region_divergence_curl(u, v, layout)
                cache.write("divergence", write_t0, div.astype(dtype))
                cache.write("curl", write_t0, curl.astype(dtype))
            write_t0 += buf_at
            buf_at = 0

        previous_atlas = None
        try:
            frame_iter = roi_decoder.iter_frames() if roi_decoder is not None \
                else src.iter_frames(0, n_frames)
            for i, frame in frame_iter:
                # Clear separator/unused atlas cells before filling this frame.
                buf_u[buf_at].fill(0)
                buf_v[buf_at].fill(0)
                buf_s[buf_at].fill(0)
                if buf_texture is not None:
                    buf_texture[buf_at].fill(0)
                if buf_fb_error is not None:
                    buf_fb_error[buf_at].fill(0)

                processed: dict[int, np.ndarray] = {}
                for tile in layout.tiles:
                    rid = tile.replicate_id
                    x0, y0, x1, y1 = tile.source_box
                    owned = roi_decoder.crop(frame, rid) \
                        if roi_decoder is not None else frame[y0:y1, x0:x1]
                    g = preprocessors[rid].apply(owned)
                    processed[rid] = g
                    ay0, ax0, ay1, ax1 = tile.atlas_bbox

                    if buf_texture is not None:
                        g8 = np.clip(g, 0, 255).astype(np.uint8)
                        eig = cv2.cornerMinEigenVal(g8, blockSize=3, ksize=3)
                        buf_texture[buf_at, ay0:ay1, ax0:ax1] = \
                            reduce_scalar_to_blocks(
                                eig, block, statistic="mean",
                                include_partial=True)

                current_atlas = _pack_flow_atlas(
                    processed, layout, support_px, flow_atlas_shape)
                if previous_atlas is not None:
                    forward_atlas = flow_backend.compute(
                        previous_atlas, current_atlas)
                    error_atlas = None
                    if buf_fb_error is not None:
                        backward_atlas = flow_backend.compute(
                            current_atlas, previous_atlas)
                        error_atlas = forward_backward_error(
                            forward_atlas, backward_atlas)

                    for tile in layout.tiles:
                        ay0, ax0, ay1, ax1 = tile.atlas_bbox
                        core_sl = flow_core_slices[tile.replicate_id]
                        flow = np.asarray(forward_atlas[core_sl], np.float32)
                        u, v, s = reduce_to_blocks(
                            flow, block, fps, include_partial=True)
                        buf_u[buf_at, ay0:ay1, ax0:ax1] = u
                        buf_v[buf_at, ay0:ay1, ax0:ax1] = v
                        buf_s[buf_at, ay0:ay1, ax0:ax1] = s
                        if buf_fb_error is not None and error_atlas is not None:
                            fb_error = np.asarray(error_atlas[core_sl], np.float32)
                            buf_fb_error[buf_at, ay0:ay1, ax0:ax1] = \
                                reduce_scalar_to_blocks(
                                    fb_error, block, statistic="p90",
                                    include_partial=True)
                previous_atlas = current_atlas
                buf_at += 1

                if buf_at == buf_u.shape[0]:
                    flush()
                if i % 20 == 0:
                    extra = " + backward consistency" \
                        if cfg.features.cache_fb_error else ""
                    emit("flow", i + 1, n_frames,
                         f"Per-replicate optical flow "
                         f"({cfg.flow.backend}{extra}); {len(layout.tiles)} "
                         f"replicates, {layout.work_pixels_per_frame:,} owned "
                         f"working pixels/frame")
            flush()
        finally:
            flow_backend.close()
        emit("flow", n_frames, n_frames, "Flow complete")

        # -- stage 2: band-power ---------------------------------------------
        if cfg.features.bands:
            _compute_band_power(cache, cfg, fps, n_flow, ny, nx, dtype, emit)

    finally:
        if roi_decoder is not None:
            roi_decoder.release()
        src.release()
        cache.close()

    # meta.json exists from the moment construction starts so progress and
    # failures are inspectable.  Only mark it complete after every backend write
    # has succeeded and the store has closed; Preprocessing & Flow will recompute a cache left
    # false by cancellation or an exception instead of trying to load missing
    # arrays from it.
    cache.meta["complete"] = True
    cache.meta["runtime_s"] = time.perf_counter() - t_start
    cache.write_meta()

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
