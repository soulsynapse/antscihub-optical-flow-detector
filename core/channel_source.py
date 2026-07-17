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

from dataclasses import dataclass

import numpy as np

from core.replicates import build_layout
from core.tensor_channels import extract_channels_live, load_or_extract_channels
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
    layout = build_layout(replicates, width, height, scale, cfg.flow.block_size)
    ny, nx = layout.atlas_grid
    block = int(cfg.flow.block_size)
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


def live_channel_source(video_path: str, cfg, replicates: list[dict], *,
                        start: int = 0, n: int | None = None,
                        width: int | None = None, height: int | None = None,
                        fps: float | None = None, frame_count: int | None = None,
                        progress=None) -> ChannelData:
    """Cacheless windowed source: geometry from ``build_layout``, channels from a
    structure-tensor pass over frames [start, start+n). Video info is read from
    the file unless all of width/height/fps/frame_count are supplied (the GUI
    passes them to avoid reopening the decoder)."""
    if None in (width, height, fps, frame_count):
        with VideoSource(video_path) as src:
            info = src.info
            width, height = info.width, info.height
            fps, frame_count = info.fps, info.frame_count
    full_meta = synth_live_meta(video_path, cfg, replicates, width=width,
                                height=height, fps=fps, frame_count=frame_count)
    res = extract_channels_live(video_path, full_meta, start=start, n=n,
                                progress=progress)
    win = int(res["meta"]["n_frames"])
    wstart = int(res["meta"]["window_start"])
    # The explorer's T axis is the window, so advertise the window length as
    # n_frames; keep the full-clip geometry otherwise.
    meta = {**full_meta, "n_frames": win, "window_start": wstart}
    channels = {k: np.asarray(res[k], np.float32) for k in LIVE_CHANNELS}
    return ChannelData(meta=meta, channels=channels, window_start=wstart,
                       approximated=bool(res["meta"].get("approximated", False)))
