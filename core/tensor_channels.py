"""Precompute the structure-tensor temporal channels a feature cache does not
store, so an explorer can read them like any other per-block time series.

The flow pipeline caches motion (u, v, speed) and optionally texture, but not the
temporal-change family the tool-3 explorer needs:

    intensity   mean block intensity            -> amplitude variance (windowed)
    change      <I_t^2> per block               -> fast change energy (J_tt)
    appearance  <r^2>, r = I_t + grad(I).v      -> change no motion explains
    texture     spatial min-eigen               -> read from cache if present
    tensor_speed Lucas-Kanade speed from J       -> independent flow-speed read

``appearance`` uses the flow ALREADY in the cache (block flow, upsampled), not a
fresh solve, so the residual is measured against exactly the motion the pipeline
committed to -- the same "no second flow implementation to mistrust" contract the
other explorers keep.

Preprocessing replays downsampling, grayscale conversion, temporal denoising and
contrast normalization. Denoising is exactly reproducible because extraction
streams the same frames in the same order. Registration, background subtraction
and within-replicate masks need reference/model assets that the cache does not
retain, so those steps are omitted and the returned metadata flags the channels
as approximated when any of them were configured.
"""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from core.config import PipelineConfig
from core.flow import reduce_scalar_to_blocks
from core.preprocess import Preprocessor
from core.structure_tensor import (flow_from_tensor, spatial_min_eigen,
                                   tensor_products)
from core.timing import Timer
from core.video import VideoSource, prefetch

CHANNELS = ("intensity", "change", "appearance", "texture", "tensor_speed")
CHANNEL_VERSION = 3     # bump when the extraction math changes (sidecar key)

# Shared always-off timer so _reduce can time itself without a per-call branch.
_NO_TIMER = Timer("", enabled=False)


def _tiles_from_meta(meta: dict) -> list[dict]:
    """Replicate tiles as (source_box, atlas_bbox, grid, work dims), with a
    whole-frame fallback so a non-replicate cache still works."""
    ny, nx = map(int, meta["grid"])
    raw = meta.get("replicate_tiles")
    if not raw:
        sw = int(meta.get("src_width", meta.get("work_width", nx)))
        sh = int(meta.get("src_height", meta.get("work_height", ny)))
        return [{"id": None, "source_box": (0, 0, sw, sh),
                 "atlas_bbox": (0, 0, ny, nx),
                 "work_width": nx * int(meta["block_size"]),
                 "work_height": ny * int(meta["block_size"])}]
    tiles = []
    for i, t in enumerate(raw):
        y0, x0, y1, x1 = map(int, t["atlas_bbox"])
        tiles.append({
            "id": t.get("id", i),
            "source_box": tuple(map(int, t["source_box"])),
            "atlas_bbox": (y0, x0, y1, x1),
            "work_width": int(t.get("work_width", (x1 - x0) * int(meta["block_size"]))),
            "work_height": int(t.get("work_height", (y1 - y0) * int(meta["block_size"]))),
        })
    return tiles


def _stream_channels(video_path: str, meta: dict, *, sigma: float = 2.0,
                     start: int = 0, n: int | None = None,
                     cached_uv: tuple[np.ndarray, np.ndarray] | None = None,
                     cached_texture: np.ndarray | None = None,
                     denoise: str | None = None, progress=None) -> dict:
    """Stream a frame window [start, start+n) and return per-block channel arrays.

    Geometry comes from ``meta`` (the same shape a feature cache carries); the
    video is streamed directly, so this needs no cache -- it is the seam that lets
    the tensor/scalogram path run on a bare video.

    ``cached_uv`` is the pipeline's block flow (u, v) in px/s, indexed by absolute
    frame, used for the appearance residual; pass ``None`` to measure the residual
    against the tensor's OWN per-pixel flow instead (px/frame, no cache needed).
    ``cached_texture`` is read straight through when present, else computed.
    ``denoise`` overrides the config's temporal-denoise mode (the live/windowed
    path forces it off, since denoise is stateful from frame zero and cannot be
    reproduced starting mid-clip).

    Returns a dict of the ``CHANNELS`` arrays (length = clamped window n) plus
    ``meta`` describing how they were built.
    """
    fps = float(meta["fps"])
    block = int(meta["block_size"])
    ny, nx = map(int, meta["grid"])
    total = int(meta["n_frames"])
    start = max(0, min(int(start), max(0, total - 1)))
    n = (total - start) if n is None else max(0, min(int(n), total - start))
    scale = float(meta.get("downsample", 1.0))
    tiles = _tiles_from_meta(meta)

    base_cfg = PipelineConfig.from_dict(meta.get("config", {})).preprocess
    denoise = base_cfg.denoise if denoise is None else denoise
    # Replay every step that can be reconstructed exactly by a sequential pass.
    # Registration/background models need fitted assets the cache does not retain,
    # so they are dropped and the result is flagged approximated when they were on.
    pre_cfg = replace(base_cfg, downsample=scale, mask_path=None,
                      registration="off", bg_subtract="off", denoise=denoise)
    approximated = (base_cfg.mask_path is not None or
                    base_cfg.registration != "off" or
                    base_cfg.bg_subtract != "off" or
                    denoise != base_cfg.denoise)

    out = {k: np.zeros((n, ny, nx), np.float32) for k in CHANNELS}
    if n == 0:
        out["meta"] = {"fps": fps, "block": block, "grid": (ny, nx),
                       "n_frames": 0, "channel_version": CHANNEL_VERSION,
                       "approximated": approximated, "sigma": sigma,
                       "window_start": start}
        return out

    pres = {t["id"]: Preprocessor(pre_cfg, t["source_box"][2] - t["source_box"][0],
                                  t["source_box"][3] - t["source_box"][1])
            for t in tiles}
    prev_g: dict = {t["id"]: None for t in tiles}

    # For a window that does not start at frame zero, read one preceding frame to
    # seed prev_g, so the first stored frame carries motion. This is only valid
    # because denoise is forced off in the windowed path -- a stateful denoise
    # would need every frame from zero to reach the correct state here.
    seed = 1 if start > 0 else 0
    first = start - seed
    count = n + seed

    tm = Timer("extract_channels")
    done = 0            # stored frames, so a cancelled pass can report how far it got
    src = VideoSource(video_path)
    frames = None
    try:
        # The decode is driven by hand rather than with a plain ``for`` so the
        # generator's own work (seek + read + colour convert) lands in its own span
        # instead of being invisibly folded into the loop body. Behind ``prefetch``
        # that decode runs on its own thread, so the span now measures how long
        # this loop *waits* for a frame -- near zero when the overlap is working,
        # and still the honest number when decode is the bottleneck.
        frames = prefetch(src.iter_frames(first, count))
        while True:
            with tm.span("decode"):
                nxt = next(frames, None)
            if nxt is None:
                break
            i, frame = nxt
            oi = i - start                     # <0 for the seed frame (not stored)
            for t in tiles:
                rid = t["id"]
                x0, y0, x1, y1 = t["source_box"]
                ay0, ax0, ay1, ax1 = t["atlas_bbox"]
                with tm.span("preprocess"):
                    g = pres[rid].apply(frame[y0:y1, x0:x1])
                th, tw = ay1 - ay0, ax1 - ax0

                gp = prev_g[rid]
                if oi >= 0:
                    out["intensity"][oi, ay0:ay1, ax0:ax1] = _reduce(g, block, th, tw, tm)
                    if cached_texture is not None:
                        out["texture"][oi, ay0:ay1, ax0:ax1] = \
                            cached_texture[i, ay0:ay1, ax0:ax1]
                    if gp is not None:
                        it = g - gp
                        # Form and spatially smooth the complete 3-D tensor at a
                        # small scale, solve LK per pixel, then reduce speed to
                        # blocks. Solving once per block would couple the aperture
                        # problem to the user's display/block size.
                        with tm.span("tensor_products"):
                            prods = tensor_products(gp, g)
                        with tm.span("tensor_blur"):
                            J = np.stack([cv2.GaussianBlur(prods[k], (0, 0), sigma)
                                          for k in range(6)])
                        with tm.span("flow_solve"):
                            uv = flow_from_tensor(J)             # px/frame
                            tensor_speed = np.hypot(uv[..., 0], uv[..., 1]) * fps
                        out["tensor_speed"][oi, ay0:ay1, ax0:ax1] = \
                            _reduce(tensor_speed, block, th, tw, tm)
                        out["change"][oi, ay0:ay1, ax0:ax1] = \
                            _reduce(J[2], block, th, tw, tm)
                        # Appearance residual r = I_t + grad(I).v. Against cached
                        # block flow (px/s -> px/frame, upsampled) when given, else
                        # against the tensor's own per-pixel flow (already px/frame).
                        with tm.span("appearance"):
                            if cached_uv is not None:
                                U, V = cached_uv
                                ub = cv2.resize(U[i, ay0:ay1, ax0:ax1] / fps,
                                                (g.shape[1], g.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)
                                vb = cv2.resize(V[i, ay0:ay1, ax0:ax1] / fps,
                                                (g.shape[1], g.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)
                            else:
                                ub, vb = uv[..., 0], uv[..., 1]
                            iy, ix = np.gradient(g)
                            r = it + ix * ub + iy * vb
                        out["appearance"][oi, ay0:ay1, ax0:ax1] = \
                            _reduce(r * r, block, th, tw, tm)
                        if cached_texture is None:
                            with tm.span("texture"):
                                mineig = spatial_min_eigen(J)
                            out["texture"][oi, ay0:ay1, ax0:ax1] = \
                                _reduce(mineig, block, th, tw, tm)
                prev_g[rid] = g
            if oi >= 0:
                done = oi + 1
            if progress and (oi >= 0) and (oi % 20 == 0):
                # The progress hook re-enters the GUI thread and can supersede or
                # cancel the pass, so it is timed apart from the actual work.
                with tm.span("progress_cb"):
                    progress(oi + 1, n)
    finally:
        try:
            # Close before releasing: the decode thread holds the VideoCapture,
            # and closing joins it. Releasing first would free a capture still in
            # use by a cancelled pass's producer. Nested so that a close() that
            # raises cannot skip the release and leak the handle -- the live
            # surface starts a pass on every knob edit, so a leak per pass adds up.
            if frames is not None:
                frames.close()
        finally:
            src.release()
        # Logged from the finally so a pass cancelled mid-stream still reports its
        # spans. A knob edit supersedes the running extraction by raising inside
        # the progress callback, so during tuning -- exactly when these numbers are
        # wanted -- most passes leave by the exception path. ``done`` distinguishes
        # a partial line from a complete one.
        tm.log(frames=n, done=done, tiles=len(tiles), grid=f"{ny}x{nx}",
               block=block, scale=f"{scale:.3f}")

    if progress:
        progress(n, n)
    out["meta"] = {"fps": fps, "block": block, "grid": (ny, nx), "n_frames": n,
                   "channel_version": CHANNEL_VERSION, "approximated": approximated,
                   "sigma": sigma, "window_start": start}
    return out


def extract_channels(cache, sigma: float = 2.0, max_frames: int | None = None,
                     progress=None) -> dict:
    """Stream the clip once and return per-block (T, ny, nx) channel arrays.

    ``cache`` is an open feature cache (its meta drives geometry and points at the
    video). Appearance is measured against the cached block flow, and cached
    texture is read straight through when present -- the full-clip, cache-backed
    contract. ``progress(done, total)`` is called if given.
    """
    meta = cache.meta
    U = np.asarray(cache.read("u"), np.float32)
    V = np.asarray(cache.read("v"), np.float32)
    cached_texture = None
    if "texture_min_eigen" in set(meta.get("features", [])):
        cached_texture = np.asarray(cache.read("texture_min_eigen"), np.float32)
    return _stream_channels(meta["video_path"], meta, sigma=sigma, start=0,
                            n=max_frames, cached_uv=(U, V),
                            cached_texture=cached_texture, progress=progress)


def extract_channels_live(video_path: str, meta: dict, *, start: int = 0,
                          n: int | None = None, sigma: float = 2.0,
                          progress=None) -> dict:
    """Cacheless windowed extraction: geometry from ``meta``, appearance against
    the tensor's own flow, temporal denoise forced off (stateful, can't reproduce
    mid-clip). This is what feeds the live scalogram surface."""
    return _stream_channels(video_path, meta, sigma=sigma, start=start, n=n,
                            cached_uv=None, cached_texture=None, denoise="off",
                            progress=progress)


def load_or_extract_channels(cache, sidecar_path: str | None = None,
                             progress=None, **kwargs) -> dict:
    """extract_channels with an on-disk sidecar so re-opens are instant.

    The sidecar is keyed by frame count and channel version; a mismatch (cache
    rebuilt, extraction math changed) is ignored and recomputed rather than
    trusted, so a stale sidecar can never feed an explorer the wrong numbers.
    """
    import os

    n = int(cache.meta["n_frames"])
    if sidecar_path and os.path.exists(sidecar_path):
        try:
            data = np.load(sidecar_path, allow_pickle=True)
            if (int(data["n_frames"]) == n and
                    int(data["channel_version"]) == CHANNEL_VERSION):
                res = {k: data[k] for k in CHANNELS}
                res["meta"] = {"fps": float(data["fps"]), "block": int(data["block"]),
                               "grid": tuple(data["grid"]), "n_frames": n,
                               "channel_version": CHANNEL_VERSION,
                               "approximated": bool(data["approximated"]),
                               "sigma": float(data["sigma"])}
                if progress:
                    progress(n, n)
                return res
        except (KeyError, ValueError, OSError):
            pass  # unreadable/old sidecar -> recompute

    res = extract_channels(cache, progress=progress, **kwargs)
    if sidecar_path:
        try:
            os.makedirs(os.path.dirname(sidecar_path), exist_ok=True)
            m = res["meta"]
            np.savez_compressed(
                sidecar_path, n_frames=m["n_frames"], fps=m["fps"],
                block=m["block"], grid=np.asarray(m["grid"]),
                channel_version=m["channel_version"], approximated=m["approximated"],
                sigma=m["sigma"], **{k: res[k] for k in CHANNELS})
        except OSError:
            pass  # read-only location -> just skip caching
    return res


def _reduce(field: np.ndarray, block: int, th: int, tw: int,
            tm: Timer = _NO_TIMER) -> np.ndarray:
    """Block-mean a tile field and force it to the tile's (th, tw) atlas shape.

    Timing lives inside rather than at the five call sites, so every reduction in
    a pass lands in one ``block_reduce`` span without wrapping each caller."""
    with tm.span("block_reduce"):
        r = reduce_scalar_to_blocks(field, block, "mean", include_partial=True)
    if r.shape != (th, tw):
        # Geometry drift between preprocess output and cached grid: crop/pad so the
        # channel aligns with the cached flow rather than silently misindexing.
        out = np.zeros((th, tw), np.float32)
        h, w = min(th, r.shape[0]), min(tw, r.shape[1])
        out[:h, :w] = r[:h, :w]
        return out
    return r.astype(np.float32)
