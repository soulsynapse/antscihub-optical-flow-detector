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
from core.video import VideoSource

CHANNELS = ("intensity", "change", "appearance", "texture", "tensor_speed")
CHANNEL_VERSION = 3     # bump when the extraction math changes (sidecar key)


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


def extract_channels(cache, sigma: float = 2.0, max_frames: int | None = None,
                     progress=None) -> dict:
    """Stream the clip once and return per-block (T, ny, nx) channel arrays.

    ``cache`` is an open feature cache (its meta drives geometry and points at the
    video). ``progress(done, total)`` is called if given. Returns a dict of the
    ``CHANNELS`` arrays plus ``meta`` describing how they were built.
    """
    meta = cache.meta
    fps = float(meta["fps"])
    block = int(meta["block_size"])
    ny, nx = map(int, meta["grid"])
    n = int(meta["n_frames"])
    if max_frames is not None:
        n = min(n, max_frames)
    scale = float(meta.get("downsample", 1.0))
    tiles = _tiles_from_meta(meta)

    base_cfg = PipelineConfig.from_dict(meta.get("config", {})).preprocess
    # Replay every step that can be reconstructed exactly by a sequential pass.
    # Temporal denoising is stateful but deterministic from frame zero, unlike
    # registration/background models whose fitted assets are not cached.
    pre_cfg = replace(base_cfg, downsample=scale, mask_path=None,
                      registration="off", bg_subtract="off")
    approximated = (base_cfg.mask_path is not None or
                    base_cfg.registration != "off" or
                    base_cfg.bg_subtract != "off")

    # Cached block flow (px/s -> px/frame), for the appearance residual.
    U = np.asarray(cache.read("u"), np.float32)
    V = np.asarray(cache.read("v"), np.float32)

    feats = set(meta.get("features", []))
    cached_texture = None
    if "texture_min_eigen" in feats:
        cached_texture = np.asarray(cache.read("texture_min_eigen"), np.float32)

    out = {k: np.zeros((n, ny, nx), np.float32) for k in CHANNELS}
    pres = {t["id"]: Preprocessor(pre_cfg, t["source_box"][2] - t["source_box"][0],
                                  t["source_box"][3] - t["source_box"][1])
            for t in tiles}
    prev_g: dict = {t["id"]: None for t in tiles}

    src = VideoSource(meta["video_path"])
    try:
        for i, frame in src.iter_frames(0, n):
            for t in tiles:
                rid = t["id"]
                x0, y0, x1, y1 = t["source_box"]
                ay0, ax0, ay1, ax1 = t["atlas_bbox"]
                g = pres[rid].apply(frame[y0:y1, x0:x1])
                th, tw = ay1 - ay0, ax1 - ax0

                out["intensity"][i, ay0:ay1, ax0:ax1] = _reduce(g, block, th, tw)
                if cached_texture is not None:
                    out["texture"][i, ay0:ay1, ax0:ax1] = \
                        cached_texture[i, ay0:ay1, ax0:ax1]

                gp = prev_g[rid]
                if gp is not None:
                    it = g - gp
                    # Match the proof of concept: form and spatially smooth the
                    # complete 3-D tensor at a small scale, solve LK per pixel,
                    # then reduce the resulting speed to cache blocks.  Solving
                    # only once per block would couple the aperture problem to
                    # the user's display/block size.
                    prods = tensor_products(gp, g)
                    J = np.stack([cv2.GaussianBlur(prods[k], (0, 0), sigma)
                                  for k in range(6)])
                    uv = flow_from_tensor(J)
                    tensor_speed = np.hypot(uv[..., 0], uv[..., 1]) * fps
                    out["tensor_speed"][i, ay0:ay1, ax0:ax1] = \
                        _reduce(tensor_speed, block, th, tw)
                    out["change"][i, ay0:ay1, ax0:ax1] = \
                        _reduce(J[2], block, th, tw)
                    # residual r = I_t + grad(I).v against upsampled cached flow.
                    ub = cv2.resize(U[i, ay0:ay1, ax0:ax1] / fps, (g.shape[1], g.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                    vb = cv2.resize(V[i, ay0:ay1, ax0:ax1] / fps, (g.shape[1], g.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                    iy, ix = np.gradient(g)
                    r = it + ix * ub + iy * vb
                    out["appearance"][i, ay0:ay1, ax0:ax1] = _reduce(r * r, block, th, tw)
                    if cached_texture is None:
                        out["texture"][i, ay0:ay1, ax0:ax1] = \
                            _reduce(spatial_min_eigen(J), block, th, tw)
                prev_g[rid] = g
            if progress and (i % 20 == 0):
                progress(i + 1, n)
    finally:
        src.release()

    if progress:
        progress(n, n)
    out["meta"] = {"fps": fps, "block": block, "grid": (ny, nx), "n_frames": n,
                   "channel_version": CHANNEL_VERSION, "approximated": approximated,
                   "sigma": sigma}
    return out


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


def _reduce(field: np.ndarray, block: int, th: int, tw: int) -> np.ndarray:
    """Block-mean a tile field and force it to the tile's (th, tw) atlas shape."""
    r = reduce_scalar_to_blocks(field, block, "mean", include_partial=True)
    if r.shape != (th, tw):
        # Geometry drift between preprocess output and cached grid: crop/pad so the
        # channel aligns with the cached flow rather than silently misindexing.
        out = np.zeros((th, tw), np.float32)
        h, w = min(th, r.shape[0]), min(tw, r.shape[1])
        out[:h, :w] = r[:h, :w]
        return out
    return r.astype(np.float32)
