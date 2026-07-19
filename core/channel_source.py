"""Decouple the scalogram/tensor path from the feature cache.

The scalogram explorer historically took an open feature cache and read both the
geometry contract (``meta``) and the flow arrays from it. But the tensor channels
it detects on -- change, tensor_speed, intensity, and the appearance residual --
need only the geometry and the video, not the expensive flow solve. A
``ChannelData`` carries exactly that: cache-meta-shaped geometry plus per-block
channel time series, from EITHER an existing cache or a live windowed pass over a
bare video.

  * ``cache_channel_source`` -- today's behaviour. All five channels, including
    the pipeline's cached flow ``speed`` and appearance measured against cached
    flow.
  * ``live_channel_source`` -- geometry via ``build_layout`` (cheap, no flow),
    then a windowed structure-tensor pass. Appearance is measured against the
    tensor's own flow; cached-flow ``speed`` is absent. This is the seam that lets
    the explorer open any video, any window, with no cache.

See docs/expanded_cache_plan.md and the branch plan for the larger restructure.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from core.replicates import build_layout
from core.tensor_channels import (_tiles_from_meta, extract_channels_live,
                                  load_or_extract_channels)
from core.video import VideoSource

# Channels a live (cacheless) source provides, in the explorer's UI order. All are
# video-derived; appearance rides the tensor's own flow. The pipeline's cached
# flow ``speed`` is added only by the cache-backed source.
LIVE_CHANNELS = ("change", "appearance", "tensor_speed", "intensity")


@dataclass
class ChannelData:
    """Geometry + per-block channel time series, independent of any cache.

    ``meta`` is cache-meta-shaped (fps, grid, block_size, replicate_tiles, ...) so
    the explorer reads it exactly as it read a cache's meta. ``channels`` maps a
    channel name to its ``(T, ny, nx)`` array. ``window_start`` is the absolute
    video frame the T axis begins at (0 for full-clip sources), letting the video
    overlay map a plot frame back to a decode frame.
    """
    meta: dict
    channels: dict[str, np.ndarray]
    window_start: int = 0
    approximated: bool = False

    @property
    def available(self) -> set[str]:
        return set(self.channels)

    @property
    def n_frames(self) -> int:
        return int(self.meta["n_frames"])


def cache_channel_source(cache, sidecar_path: str | None = None,
                         progress=None) -> ChannelData:
    """Today's five-channel, cache-backed source: structure-tensor channels via
    the sidecar-memoized extractor plus the pipeline's cached flow ``speed``."""
    ch = load_or_extract_channels(cache, sidecar_path=sidecar_path,
                                  progress=progress)
    channels = {
        "change": np.asarray(ch["change"], np.float32),
        "appearance": np.asarray(ch["appearance"], np.float32),
        "tensor_speed": np.asarray(ch["tensor_speed"], np.float32),
        "intensity": np.asarray(ch["intensity"], np.float32),
        "speed": np.asarray(cache.read("speed"), np.float32),
    }
    return ChannelData(meta=cache.meta, channels=channels, window_start=0,
                       approximated=bool(ch["meta"].get("approximated", False)))


def synth_live_meta(video_path: str, cfg, replicates: list[dict], *,
                    width: int, height: int, fps: float,
                    frame_count: int) -> dict:
    """Build a cache-meta-shaped geometry contract for a bare video + config,
    without running (or writing) any flow cache. ``n_frames`` is the FULL clip;
    the extraction window is applied separately."""
    scale = cfg.preprocess.resolve_downsample(width)
    block = cfg.flow.resolve_block_size(scale)
    layout = build_layout(replicates, width, height, scale, block)
    ny, nx = layout.atlas_grid
    return {
        "video_path": video_path,
        "backend": "live",
        "fps": float(fps),
        "n_frames": int(frame_count),
        "block_size": block,
        "grid": [int(ny), int(nx)],
        "src_width": int(width),
        "src_height": int(height),
        "work_width": int(nx * block),
        "work_height": int(ny * block),
        "downsample": float(scale),
        "features": [],
        "replicate_tiles": [t.to_meta() for t in layout.tiles],
        "replicate_geometry_hash": layout.geometry_hash,
        "processing_scope": "replicate",
        "config": cfg.to_dict(),
    }


def resolve_clip_paths(manifest, clip_dir: str, replicates: list[dict],
                       tiles: list[dict]) -> list[str]:
    """Clip paths in ``tiles`` order, after checking they may be used at all.

    Matched by replicate id rather than by list position. Both lists do come out
    of ``build_layout`` over the same replicates and so happen to align today,
    but a positional match would fail *silently* if that ever stopped being true,
    and the failure mode is a replicate's pixels being attributed to another
    replicate's detections. An id lookup raises instead.

    ``verify_manifest`` runs on every call: it re-checks the geometry hash and
    the source's identity, and it is cheap by construction (a quick head+tail
    signature, not a re-hash of an 11 GB file) precisely so that a live surface
    starting a pass on every knob edit can afford it.
    """
    from core.pretranscode import verify_manifest

    verify_manifest(manifest, clip_dir, replicates)
    paths = []
    for t in tiles:
        entry = manifest.clip_for(t["id"])
        x0, y0, x1, y1 = (int(v) for v in t["source_box"])
        if tuple(int(v) for v in entry.source_box) != (x0, y0, x1, y1):
            raise ValueError(
                f"replicate {t['id']} box {t['source_box']} does not match the "
                f"clip's {entry.source_box}; re-run the pre-transcode")
        # The frame size the clip is supposed to hold, checked here because it
        # is free -- the manifest already recorded it. ClipAtlasSource would
        # otherwise probe each file to learn the same thing (41.8 ms for 6
        # clips), and a wrong size is the one mismatch its filter graph would
        # silently absorb by rescaling rather than reject.
        if (entry.width, entry.height) != (x1 - x0, y1 - y0):
            raise ValueError(
                f"clip {entry.filename} records {entry.width}x{entry.height} "
                f"but replicate {t['id']}'s box is {x1-x0}x{y1-y0}; "
                "re-run the pre-transcode")
        paths.append(os.path.join(clip_dir, entry.filename))
    return paths


def live_channel_source(video_path: str, cfg, replicates: list[dict], *,
                        start: int = 0, n: int | None = None,
                        width: int | None = None, height: int | None = None,
                        fps: float | None = None, frame_count: int | None = None,
                        manifest=None, clip_dir: str | None = None,
                        channels: Iterable[str] | None = None,
                        progress=None) -> ChannelData:
    """Cacheless windowed source: geometry from ``build_layout``, channels from a
    structure-tensor pass over frames [start, start+n). Video info is read from
    the file unless all of width/height/fps/frame_count are supplied (the GUI
    passes them to avoid reopening the decoder).

    Pass a ``manifest`` (and the ``clip_dir`` holding its clips) to decode
    pre-transcoded per-replicate clips instead of the source. The channel math is
    identical; what changes is that the decoder stops paying for the ~92% of each
    frame no replicate owns. The manifest's ``provenance_key`` lands in the
    returned meta as ``clip_provenance`` and MUST be folded into the key of
    anything cached downstream -- below ``lossless`` these are different pixels
    from the source's, so a clip-derived result and a live-crop-derived one are
    different measurements (``FINDINGS.md`` section 10). ``PipelineConfig.cache_key``
    takes it as an optional third argument; note that nothing passes it yet, so
    this is an obligation on the next caller to cache clip-derived output rather
    than a guarantee already in force.

    ``channels`` restricts the pass to the channels named (default: all of
    ``LIVE_CHANNELS``). The returned ``ChannelData`` then **carries only those
    keys** -- an uncomputed channel is absent, not present-and-NaN, so
    ``ChannelData.available`` and everything gating on it (the explorer's channel
    checkboxes, ``detect_channel_region``) keep meaning "this is real data".
    ``meta['channels_computed']`` records the request either way.
    """
    if None in (width, height, fps, frame_count):
        with VideoSource(video_path) as src:
            info = src.info
            width, height = info.width, info.height
            fps, frame_count = info.fps, info.frame_count
    full_meta = synth_live_meta(video_path, cfg, replicates, width=width,
                                height=height, fps=fps, frame_count=frame_count)
    clip_paths = None
    if manifest is not None:
        if not clip_dir:
            raise ValueError("a manifest needs the clip_dir holding its clips")
        clip_paths = resolve_clip_paths(manifest, clip_dir, replicates,
                                        _tiles_from_meta(full_meta))
        # The manifest's rational rate, not the float OpenCV reported: the clip
        # seek divides a frame index by this, and a rounded rate lands
        # progressively earlier as the index grows (FINDINGS.md section 3 trap 2).
        full_meta = {**full_meta, "fps": manifest.fps,
                     "clip_provenance": manifest.provenance_key(),
                     "clip_dir": clip_dir,
                     "clip_quality": manifest.quality,
                     "clip_quality_rms": manifest.quality_rms}
    want = (set(LIVE_CHANNELS) if channels is None
            else set(channels) & set(LIVE_CHANNELS))
    if not want:
        raise ValueError(f"no live channels requested; pick from {LIVE_CHANNELS}")
    res = extract_channels_live(video_path, full_meta, start=start, n=n,
                                clip_paths=clip_paths, channels=want,
                                progress=progress)
    win = int(res["meta"]["n_frames"])
    wstart = int(res["meta"]["window_start"])
    # The explorer's T axis is the window, so advertise the window length as
    # n_frames; keep the full-clip geometry otherwise.
    # ``truncated`` rides along: the window is already short, but a consumer that
    # only sees a length cannot tell "you asked for a short window" from "the
    # decode ended early and this is less data than you asked for".
    meta = {**full_meta, "n_frames": win, "window_start": wstart,
            "truncated": bool(res["meta"].get("truncated", False)),
            "channels_computed": sorted(want)}
    # Carried through so the downsample decision tool can price the lever from
    # passes that actually ran on this machine and this footage. Absent when the
    # pass was empty or timing was disabled.
    if res["meta"].get("timing"):
        meta["timing"] = res["meta"]["timing"]
    arrays = {k: np.asarray(res[k], np.float32) for k in LIVE_CHANNELS
              if k in want}
    return ChannelData(meta=meta, channels=arrays, window_start=wstart,
                       approximated=bool(res["meta"].get("approximated", False)))


def _reduce_stack(field: np.ndarray, block: int, th: int, tw: int) -> np.ndarray:
    """Batched block-mean of a ``(T, H, W)`` per-pixel field to ``(T, th, tw)``.

    Numerically identical to ``tensor_channels._reduce`` (include_partial nan-mean
    then crop/pad to the tile's atlas shape) but reduces the whole time axis in one
    vectorized pass instead of frame-by-frame. Partial edge cells always hold at
    least one real pixel, so no all-NaN mean arises."""
    n = field.shape[0]
    if block <= 1:
        red = field.astype(np.float32, copy=False)
        ny, nx = field.shape[1], field.shape[2]
    else:
        hh, ww = field.shape[1], field.shape[2]
        ny = (hh + block - 1) // block
        nx = (ww + block - 1) // block
        x = field.astype(np.float32, copy=False)
        ph, pw = ny * block - hh, nx * block - ww
        if ph or pw:
            x = np.pad(x, ((0, 0), (0, ph), (0, pw)), constant_values=np.nan)
        x = x.reshape(n, ny, block, nx, block).transpose(0, 1, 3, 2, 4)
        x = x.reshape(n, ny, nx, block * block)
        red = np.nanmean(x, axis=3, dtype=np.float32).astype(np.float32)
    if (ny, nx) == (th, tw):
        return red
    out = np.zeros((n, th, tw), np.float32)         # geometry drift: crop/pad
    yy, xx = min(th, ny), min(tw, nx)
    out[:, :yy, :xx] = red[:, :yy, :xx]
    return out


def reduce_channel_data(pp: ChannelData, cfg, replicates: list[dict]
                        ) -> ChannelData:
    """Re-reduce a pixel-level (``block_size=1``) live ChannelData to the block grid
    in ``cfg`` WITHOUT re-extracting.

    The per-pixel structure-tensor solve is block-size independent, so a Block
    change is only the cheap block-mean over already-extracted per-pixel fields.
    The result is identical block-for-block to a fresh ``live_channel_source`` at
    that block (the block=1 atlas stores the exact working-resolution field, and
    reducing it is the same nan-mean the extractor would have run)."""
    src = pp.meta
    if int(src.get("block_size", 0)) != 1:
        raise ValueError("reduce_channel_data expects a block_size=1 source")
    w, h = int(src["src_width"]), int(src["src_height"])
    # Track the scale the cached pixels were actually extracted at, not the one
    # in cfg: this reduces an existing atlas, so its own meta is authoritative.
    block = cfg.flow.resolve_block_size(float(src["downsample"]))
    if block <= 1:
        return pp                                    # already pixel-level
    fps = float(src["fps"])
    n = int(src["n_frames"])
    ws = int(getattr(pp, "window_start", src.get("window_start", 0)))
    meta_n = synth_live_meta(src["video_path"], cfg, replicates, width=w, height=h,
                             fps=fps, frame_count=n)
    ny_a, nx_a = map(int, meta_n["grid"])
    # Both tile lists come from build_layout over the same replicates (sorted by
    # id), so they align by position: block=1 pixel regions -> block-N atlas cells.
    src_tiles = _tiles_from_meta(src)
    dst_tiles = _tiles_from_meta(meta_n)
    channels = {}
    for name, arr in pp.channels.items():
        arr = np.asarray(arr, np.float32)
        out = np.zeros((n, ny_a, nx_a), np.float32)
        for st, dt in zip(src_tiles, dst_tiles):
            sy0, sx0, sy1, sx1 = st["atlas_bbox"]
            dy0, dx0, dy1, dx1 = dt["atlas_bbox"]
            field = arr[:, sy0:sy1, sx0:sx1]
            out[:, dy0:dy1, dx0:dx1] = _reduce_stack(field, block,
                                                     dy1 - dy0, dx1 - dx0)
        channels[name] = out
    meta = {**meta_n, "n_frames": n, "window_start": ws}
    return ChannelData(meta=meta, channels=channels, window_start=ws,
                       approximated=pp.approximated)
