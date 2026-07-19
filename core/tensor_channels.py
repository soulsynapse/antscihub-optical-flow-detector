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
from core.video import (ClipAtlasSource, ReplicateVideoSource, VideoSource,
                        prefetch)

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


class _MetaTile:
    """The four attributes ``ReplicateVideoSource`` reads off a tile.

    ``_tiles_from_meta`` yields dicts (it also has to serve a whole-frame
    fallback that no real ``ReplicateLayout`` can express), so the ROI decoder
    gets this adapter rather than a reconstructed ``ReplicateTile`` -- there is
    no layout to rebuild here, only geometry that meta already carries.
    """

    __slots__ = ("replicate_id", "source_box", "work_width", "work_height")

    def __init__(self, key: int, t: dict):
        self.replicate_id = key
        self.source_box = t["source_box"]
        self.work_width = t["work_width"]
        self.work_height = t["work_height"]


class _MetaLayout:
    def __init__(self, tiles: list[_MetaTile]):
        self.tiles = tiles


def _roi_layout(tiles: list[dict]) -> _MetaLayout:
    """Adapt meta tiles for the ROI decoder, keyed by list position.

    Position, not ``t["id"]``: the whole-frame fallback tile has id ``None``,
    and ``crop`` keys with ``int(...)``. The caller pairs a tile with its atlas
    slot by position anyway, so this cannot drift.
    """
    return _MetaLayout([_MetaTile(i, t) for i, t in enumerate(tiles)])


def _open_roi(video_path: str, tiles: list[dict], first: int, count: int,
              fps: float):
    """``(decoder, frame_iter)`` for the FFmpeg ROI path, or ``(None, None)``.

    The first frame is pulled here rather than in the loop so a decoder that
    builds fine but fails on contact -- a filter graph FFmpeg rejects, a codec it
    cannot seek -- falls back to full-frame decode instead of killing the pass.
    Only construction is guarded in the pipeline's copy of this; the live surface
    starts a pass on every knob edit, so a hard failure there is far more costly.
    """
    try:
        roi = ReplicateVideoSource(video_path, _roi_layout(tiles), count,
                                   start=first, fps=fps)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return None, None

    inner = roi.iter_frames()
    try:
        head = next(inner)
    except (StopIteration, OSError, RuntimeError, ValueError):
        # StopIteration included deliberately: a decoder that yields nothing is
        # as useless as one that raises, and both should fall back rather than
        # hand the caller an empty pass.
        roi.release()
        return None, None

    def frames():
        # A generator, not itertools.chain, so close() propagates through the
        # yield from and shuts the FFmpeg reader down -- prefetch relies on that.
        yield head
        yield from inner

    return roi, frames()


def _open_clips(clip_paths: list[str], tiles: list[dict], first: int,
                count: int, fps: float):
    """``(decoder, frame_iter)`` for pre-transcoded clips. **Never falls back.**

    The deliberate difference from :func:`_open_roi`: a clip decoder that fails
    raises instead of quietly reverting to whole-source decode. The fallback is
    right for the ROI path, where both routes read the same source pixels and
    only speed differs. It is wrong here. Below ``lossless`` a clip's pixels are
    *not* the source's (``FINDINGS.md`` section 10), and the caller has already
    folded the clips' provenance key into whatever it caches -- so a silent
    fallback would file source-decoded numbers under a clip-decoded identity,
    which is exactly the confusion ``provenance_key`` exists to prevent.
    """
    # verify_sizes=False: callers reach this through
    # ``channel_source.resolve_clip_paths``, which has already checked every
    # clip's recorded size against its replicate box using the manifest, for
    # free. Probing the files again would cost ~42 ms per pass on 6 clips --
    # against a ~160 ms decode, on a surface that starts a pass per knob edit.
    clips = ClipAtlasSource(clip_paths, _roi_layout(tiles), count, start=first,
                            fps=fps, verify_sizes=False)
    inner = clips.iter_frames()
    try:
        head = next(inner)
    except StopIteration:
        clips.release()
        raise RuntimeError("clip decode yielded no frames")
    except BaseException:
        clips.release()
        raise

    def frames():
        yield head
        yield from inner

    return clips, frames()


def _stream_channels(video_path: str, meta: dict, *, sigma: float = 2.0,
                     start: int = 0, n: int | None = None,
                     cached_uv: tuple[np.ndarray, np.ndarray] | None = None,
                     cached_texture: np.ndarray | None = None,
                     clip_paths: list[str] | None = None,
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
                       "window_start": start,
                       # Present on both exits: an empty window was asked for and
                       # delivered, which is not the same as a decode ending
                       # early, and a consumer reading meta["truncated"] should
                       # not have to know which return path produced its dict.
                       "truncated": False}
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
    # Pre-transcoded clips when the caller has them: the decoder then reads files
    # that are already only replicate pixels, which is the one thing that moves
    # the decode floor (~25x, FINDINGS.md section 10). Otherwise ROI decode:
    # FFmpeg crops, greyscales and downsamples each replicate box in its own
    # process, so the ~92% of a 5.3K frame no replicate owns never crosses into
    # Python -- though the decoder still paid for it. Falls back to full-frame
    # OpenCV when even that is unavailable.
    if clip_paths is not None:
        roi, frames = _open_clips(clip_paths, tiles, first, count, fps)
    else:
        roi, frames = _open_roi(video_path, tiles, first, count, fps)
    src = None if roi is not None else VideoSource(video_path)
    try:
        # The decode is driven by hand rather than with a plain ``for`` so the
        # generator's own work (seek + read + colour convert) lands in its own span
        # instead of being invisibly folded into the loop body. Behind ``prefetch``
        # that decode runs on its own thread, so the span now measures how long
        # this loop *waits* for a frame -- near zero when the overlap is working,
        # and still the honest number when decode is the bottleneck.
        frames = prefetch(frames if roi is not None
                          else src.iter_frames(first, count))
        while True:
            with tm.span("decode"):
                nxt = next(frames, None)
            if nxt is None:
                break
            i, frame = nxt
            oi = i - start                     # <0 for the seed frame (not stored)
            for ti, t in enumerate(tiles):
                rid = t["id"]
                x0, y0, x1, y1 = t["source_box"]
                ay0, ax0, ay1, ax1 = t["atlas_bbox"]
                with tm.span("preprocess"):
                    # On the ROI path the crop is already gray and already at the
                    # tile's work size, so Preprocessor's downsample/grayscale
                    # steps collapse to no-ops and the remaining steps (normalize,
                    # mask, ...) run on identical input. Same geometry either way,
                    # because both derive it from the same source_box and scale.
                    owned = (roi.crop(frame, ti) if roi is not None
                             else frame[y0:y1, x0:x1])
                    g = pres[rid].apply(owned)
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
            (src if src is not None else roi).release()
        # Logged from the finally so a pass cancelled mid-stream still reports its
        # spans. A knob edit supersedes the running extraction by raising inside
        # the progress callback, so during tuning -- exactly when these numbers are
        # wanted -- most passes leave by the exception path. ``done`` distinguishes
        # a partial line from a complete one.
        tm.log(frames=n, done=done, tiles=len(tiles), grid=f"{ny}x{nx}",
               block=block, scale=f"{scale:.3f}",
               src="clips" if clip_paths is not None else
                   ("roi" if roi is not None else "full"))

    # A decoder that ended early leaves the tail of every channel at zero. Return
    # a SHORT window rather than a full-length one padded with zeros: a short
    # window is self-describing, whereas zeros are indistinguishable from
    # "measured, and nothing happened" -- i.e. a silent false negative, on the
    # detection default (``change``) above all. Reachable in practice: a clip
    # truncated by a crash or a full disk during the cut passes
    # ``verify_manifest`` (which checks a clip exists, not how long it is), and a
    # 20-of-64-frame clip was measured yielding 36 all-zero frames of 40 reported
    # as real. The ROI and full-frame paths get the same treatment; they can hit
    # it too, just far less often.
    short = done < n
    if short:
        for k in CHANNELS:
            out[k] = out[k][:done]
        n = done
    if progress:
        progress(n, n)
    out["meta"] = {"fps": fps, "block": block, "grid": (ny, nx), "n_frames": n,
                   "channel_version": CHANNEL_VERSION, "approximated": approximated,
                   "sigma": sigma, "window_start": start, "truncated": short,
                   # The spans are already measured for the log line; returning
                   # them too is what lets the cost model be built from passes
                   # the user actually ran instead of from asserted constants.
                   # Only on the complete path: a cancelled pass unwinds through
                   # the finally above and never reaches here, so a partial pass
                   # can never be mistaken for a timing sample.
                   "timing": {"wall": tm.elapsed, "spans": dict(tm.totals),
                              "frames": n, "scale": scale, "block": block}}
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
                          clip_paths: list[str] | None = None,
                          progress=None) -> dict:
    """Cacheless windowed extraction: geometry from ``meta``, appearance against
    the tensor's own flow, temporal denoise forced off (stateful, can't reproduce
    mid-clip). This is what feeds the live scalogram surface.

    ``clip_paths`` (aligned with ``meta['replicate_tiles']`` by position) reads
    pre-transcoded per-replicate clips instead of the source. The math is
    untouched -- only where the pixels come from changes. Callers should go
    through ``channel_source.live_channel_source``, which resolves the paths from
    a manifest and verifies the geometry still matches; passing raw paths here
    skips those checks."""
    return _stream_channels(video_path, meta, sigma=sigma, start=start, n=n,
                            cached_uv=None, cached_texture=None,
                            clip_paths=clip_paths, denoise="off",
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
