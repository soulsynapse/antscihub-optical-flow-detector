"""Chunked, compressed, memory-mappable feature cache.

Two backends implement one interface so they can be benchmarked head to head
(see scripts/benchmark_storage.py). The winner is the default; see STORAGE_CHOICE
at the bottom for the decision and the numbers behind it.

Layout, both backends:

    <cache_root>/<cache_key>/
        meta.json           config, video info, block grid, band time axis
        u, v, speed, ...    (T, ny, nx) arrays, chunked along time

Chunking is time-major -- (T_chunk, ny, nx) with the full spatial grid in each
chunk. Two access patterns have to be fast and they pull in opposite directions:

  * Tab 2 scrubbing reads one frame, all blocks   -> wants whole spatial planes.
  * Tab 3 reads one ROI's time series, all frames -> wants whole time columns.

A full-spatial-plane chunk spanning many frames serves the first perfectly and
the second acceptably (an ROI read touches every chunk, but only T/T_chunk of
them, each decompressed once). The reverse layout would make scrubbing -- the
interactive, latency-critical path -- decompress the entire clip. Time-major
wins, which is also what the handoff specifies.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Iterable

import numpy as np

# Frames per chunk. At a 83x46 block grid in float16 this is ~1.5 MB per chunk
# uncompressed: big enough that zstd has something to work with, small enough
# that a single-frame scrub read does not drag in megabytes it will not use.
DEFAULT_CHUNK_FRAMES = 256


class IncompleteCacheError(RuntimeError):
    """A cache directory exists, but its pipeline run did not finish cleanly."""


class FeatureCacheBase(ABC):
    """Common interface. Arrays are (T, ny, nx), time-major."""

    def __init__(self, root: str, meta: dict):
        self.root = root
        self.meta = meta
        # Read-through chunk cache. A single-frame read has to decompress the
        # whole 256-frame chunk containing it; without this, scrubbing pays that
        # cost on every frame even though consecutive frames almost always live
        # in the same chunk. With it, a scrub within a chunk is a memory slice.
        self._chunk_cache: "OrderedDict[tuple[str, int], np.ndarray]" = OrderedDict()
        self._chunk_cache_max = 12  # ~20 MB at the default grid and chunk size

    def _cached_chunk(self, name: str, chunk_idx: int) -> np.ndarray:
        key = (name, chunk_idx)
        hit = self._chunk_cache.get(key)
        if hit is not None:
            self._chunk_cache.move_to_end(key)
            return hit
        t0 = chunk_idx * DEFAULT_CHUNK_FRAMES
        t1 = min(self.n_frames_of(name), t0 + DEFAULT_CHUNK_FRAMES)
        arr = np.asarray(self._dataset(name)[t0:t1])
        self._chunk_cache[key] = arr
        if len(self._chunk_cache) > self._chunk_cache_max:
            self._chunk_cache.popitem(last=False)
        return arr

    def n_frames_of(self, name: str) -> int:
        """Length of a feature's time axis. Band-power features live on the
        coarser window axis, so this is not always n_frames."""
        return int(self._dataset(name).shape[0])

    def invalidate_chunk_cache(self) -> None:
        self._chunk_cache.clear()

    # -- metadata passthrough -------------------------------------------------

    @property
    def fps(self) -> float:
        return float(self.meta["fps"])

    @property
    def n_frames(self) -> int:
        return int(self.meta["n_frames"])

    @property
    def block_size(self) -> int:
        return int(self.meta["block_size"])

    @property
    def grid(self) -> tuple[int, int]:
        return tuple(self.meta["grid"])

    @property
    def band_hop_s(self) -> float:
        return float(self.meta.get("band_hop_s", 0.0))

    @property
    def feature_names(self) -> list[str]:
        return list(self.meta["features"])

    def times_s(self) -> np.ndarray:
        return np.arange(self.n_frames) / self.fps

    # -- band-power time axis -------------------------------------------------

    def band_frame_index(self, frame_idx: int | np.ndarray):
        """Map a frame index to the nearest band-power window index.

        Band-power lives on its own coarser time axis (one value per STFT hop,
        not per frame) because storing it per frame would multiply its size by
        hop_s*fps for no added information -- the STFT window already smears it
        over window_s seconds. Callers that want it aligned to frames go through
        here, and the mapping is nearest-window, which is exact at window centers
        and never off by more than hop_s/2.
        """
        hop_frames = max(1, int(round(self.band_hop_s * self.fps)))
        n_win = int(self.meta.get("n_band_windows", 1))
        win_frames = int(round(float(self.meta.get("band_window_s", 1.0)) * self.fps))
        centers = np.arange(n_win) * hop_frames + win_frames // 2

        # Round to the NEAREST window centre. searchsorted alone returns the next
        # centre at or after the frame, which biases every band-power lookup
        # forward by up to a full hop -- a systematic lag between the band-power
        # track and the video it is supposed to describe.
        idx = np.searchsorted(centers, frame_idx)
        idx = np.clip(idx, 0, n_win - 1)
        prev = np.clip(idx - 1, 0, n_win - 1)
        take_prev = np.abs(centers[prev] - frame_idx) <= np.abs(centers[idx] - frame_idx)
        return np.where(take_prev, prev, idx)

    # -- abstract storage ops -------------------------------------------------

    @abstractmethod
    def create_array(self, name: str, shape: tuple, dtype: str) -> None: ...

    @abstractmethod
    def _dataset(self, name: str):
        """The raw backend array object. Both h5py and zarr support numpy-style
        slice assignment, so the shared write helpers below work on either."""

    @abstractmethod
    def write(self, name: str, t0: int, arr: np.ndarray) -> None: ...

    def read_rows(self, name: str, r0: int, r1: int) -> np.ndarray:
        """Read a stripe of block-rows across all time, without materializing the
        whole array. Cuts peak RAM in the band-power stage by ny/rows_per_chunk."""
        return np.asarray(self._dataset(name)[:, r0:r1, :])

    def write_partial_rows(self, name: str, arr: np.ndarray, r0: int, r1: int,
                           dtype: str) -> None:
        """Write a stripe of block-rows across all time. Used by the band-power
        stage, which chunks over space (not time) because the windowed FFT
        expands the data by window/hop and cannot hold the whole grid at once."""
        self._dataset(name)[:, r0:r1, :] = arr.astype(dtype)

    @abstractmethod
    def read(self, name: str, t0: int = 0, t1: int | None = None) -> np.ndarray: ...

    @abstractmethod
    def read_frame(self, name: str, frame_idx: int) -> np.ndarray: ...

    @abstractmethod
    def close(self) -> None: ...

    def size_on_disk(self) -> int:
        total = 0
        for dirpath, _, files in os.walk(self.root):
            for f in files:
                total += os.path.getsize(os.path.join(dirpath, f))
        return total

    def write_meta(self) -> None:
        path = os.path.join(self.root, "meta.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)
        os.replace(tmp, path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ZarrCache(FeatureCacheBase):
    def __init__(self, root: str, meta: dict, mode: str = "r"):
        super().__init__(root, meta)
        import zarr
        self._zarr = zarr
        os.makedirs(root, exist_ok=True)
        self._store = zarr.open_group(os.path.join(root, "data.zarr"), mode=mode)

    def _codec(self):
        from zarr.codecs import BloscCodec
        c = self.meta.get("compression", "zstd")
        if c == "none":
            return []
        return [BloscCodec(cname=c, clevel=int(self.meta.get("compression_level", 5)),
                           shuffle="shuffle")]

    def create_array(self, name: str, shape: tuple, dtype: str) -> None:
        t_chunk = min(DEFAULT_CHUNK_FRAMES, shape[0]) or 1
        self._store.create_array(
            name=name, shape=shape, chunks=(t_chunk, shape[1], shape[2]),
            dtype=dtype, compressors=self._codec(), overwrite=True,
        )

    def _dataset(self, name: str):
        return self._store[name]

    def write(self, name: str, t0: int, arr: np.ndarray) -> None:
        self._store[name][t0:t0 + arr.shape[0]] = arr

    def read(self, name: str, t0: int = 0, t1: int | None = None) -> np.ndarray:
        a = self._store[name]
        t1 = a.shape[0] if t1 is None else t1
        return np.asarray(a[t0:t1])

    def read_frame(self, name: str, frame_idx: int) -> np.ndarray:
        chunk = self._cached_chunk(name, frame_idx // DEFAULT_CHUNK_FRAMES)
        return chunk[frame_idx % DEFAULT_CHUNK_FRAMES]

    def close(self) -> None:
        pass


class HDF5Cache(FeatureCacheBase):
    def __init__(self, root: str, meta: dict, mode: str = "r"):
        super().__init__(root, meta)
        import h5py
        os.makedirs(root, exist_ok=True)
        try:
            import hdf5plugin  # noqa: F401  (registers blosc filters)
            self._have_blosc = True
        except ImportError:
            self._have_blosc = False
        self._h5py = h5py
        self._f = h5py.File(os.path.join(root, "data.h5"), mode)

    def _compression_kwargs(self) -> dict:
        c = self.meta.get("compression", "zstd")
        if c == "none":
            return {}
        if self._have_blosc:
            import hdf5plugin
            cname = {"zstd": "zstd", "lz4": "lz4"}.get(c, "zstd")
            return dict(**hdf5plugin.Blosc(
                cname=cname, clevel=int(self.meta.get("compression_level", 5)),
                shuffle=hdf5plugin.Blosc.SHUFFLE))
        # gzip is the only filter guaranteed present in stock h5py. It is much
        # slower than blosc/zstd, which is exactly what the benchmark measures.
        return dict(compression="gzip", compression_opts=4, shuffle=True)

    def create_array(self, name: str, shape: tuple, dtype: str) -> None:
        t_chunk = min(DEFAULT_CHUNK_FRAMES, shape[0]) or 1
        if name in self._f:
            del self._f[name]
        self._f.create_dataset(
            name, shape=shape, dtype=dtype,
            chunks=(t_chunk, shape[1], shape[2]),
            **self._compression_kwargs(),
        )

    def _dataset(self, name: str):
        return self._f[name]

    def write(self, name: str, t0: int, arr: np.ndarray) -> None:
        self._f[name][t0:t0 + arr.shape[0]] = arr

    def read(self, name: str, t0: int = 0, t1: int | None = None) -> np.ndarray:
        d = self._f[name]
        t1 = d.shape[0] if t1 is None else t1
        return d[t0:t1]

    def read_frame(self, name: str, frame_idx: int) -> np.ndarray:
        chunk = self._cached_chunk(name, frame_idx // DEFAULT_CHUNK_FRAMES)
        return chunk[frame_idx % DEFAULT_CHUNK_FRAMES]

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


_BACKENDS = {"zarr": ZarrCache, "hdf5": HDF5Cache}

# STORAGE DECISION: Zarr.
#
# Measured by scripts/benchmark_storage.py on a full-clip-shaped cache
# (3 features x 30600 frames x 45x81 blocks, float16, blosc/zstd level 5),
# which is exactly the 8.5-minute 5.3K test clip at the default settings:
#
#   backend  comp     size    write     rnd scrub  seq scrub   ROI ts   full read
#   zarr     zstd   497.7 MB   34 MB/s   17.26 ms   0.092 ms   107 ms     0.10 s
#   hdf5     zstd   497.7 MB   52 MB/s    8.03 ms   0.056 ms   245 ms     0.27 s
#
# Reading the table:
#
#   * Size is IDENTICAL. Both drive the same blosc/zstd codec over the same
#     chunk geometry, so this was never going to separate them. (The 1.3x ratio
#     is low because float16 flow data is already dense -- the mantissa bits are
#     close to incompressible. Do not expect more.)
#
#   * Scrubbing does not separate them either, once the read-through chunk cache
#     in FeatureCacheBase is in play. Stepping frame to frame -- what dragging
#     the scrub bar and playback actually do -- costs under 0.1 ms on both,
#     because consecutive frames hit the same decompressed chunk. HDF5's 2x edge
#     on a cold random seek (8 vs 17 ms) is real but is paid only on a discrete
#     jump, where it is invisible next to decoding the video frame itself.
#
#   * The ROI time-series read is what actually separates them, and Zarr wins it
#     by 2.3x (107 vs 245 ms). This is the one pattern that CANNOT be cached
#     away: pulling one block's full time column touches every chunk in the clip
#     by construction. Tab 3 does this on every ROI click.
#
# Two things the benchmark does not measure also point to Zarr, and they are the
# reason this is not a close call:
#
#   1. Failure isolation. Zarr's directory store is a pile of independent chunk
#      files, so one interrupted write does not truncate a monolithic container.
#      The GUI opens only caches whose pipeline run completed, but their metadata
#      and surviving chunks remain inspectable for diagnosis. HDF5 is
#      single-writer/single-process and a killed writer can leave the whole file
#      unreadable.
#
#   2. Partial-failure survival. If a long pass is killed halfway, a Zarr store
#      is still readable up to the last written chunk. A truncated HDF5 file is
#      frequently not readable at all.
#
# HDF5 remains implemented and selectable -- it is faster to write and to seek
# cold, and if the concurrency requirement ever goes away it would be defensible.
STORAGE_CHOICE = "zarr"


def cache_dir(cache_root: str, cache_key: str) -> str:
    return os.path.join(cache_root, cache_key)


def _rmtree_retry(path: str, attempts: int = 8) -> None:
    """Delete a cache directory, retrying on transient Windows lock errors.

    Zarr writes thousands of small chunk files and directories. On Windows,
    Defender or the search indexer routinely holds one for a second or two after
    we close it, and rmtree then fails with WinError 5 -- often on a directory
    that is already empty. The lock always clears; it just needs longer than one
    immediate retry. Read-only attributes are cleared on the way as well, since
    that is the other common cause of the same error code.
    """
    def _on_error(func, target, exc_info):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            raise

    for i in range(attempts):
        try:
            shutil.rmtree(path, onexc=_on_error)
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(0.25 * 2 ** i)


def create_cache(cache_root: str, cache_key: str, meta: dict,
                 backend: str | None = None) -> FeatureCacheBase:
    backend = backend or STORAGE_CHOICE
    root = cache_dir(cache_root, cache_key)
    if os.path.exists(root):
        _rmtree_retry(root)
    os.makedirs(root, exist_ok=True)
    # Presence of meta.json means "inspectable", not "safe to load".  A run can
    # be cancelled after writing only u/v/speed while metadata already advertises
    # a later band-power array.  The pipeline flips this only after all writes and
    # the backend close have succeeded.
    meta = {**meta, "backend": backend, "complete": False}
    cache = _BACKENDS[backend](root, meta, mode="w")
    cache.write_meta()
    return cache


def open_cache(cache_root: str, cache_key: str) -> FeatureCacheBase:
    root = cache_dir(cache_root, cache_key)
    meta_path = os.path.join(root, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No cache at {root}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    backend = meta.get("backend", STORAGE_CHOICE)
    cache = _BACKENDS[backend](root, meta, mode="r")

    missing = []
    for name in meta.get("features", []):
        try:
            cache.n_frames_of(name)
        except (KeyError, OSError):
            missing.append(name)
    # Caches made before the completion flag was introduced remain compatible
    # when every advertised array is present.  Explicit false always means the
    # producing run was cancelled/failed, even if it happened after arrays were
    # created but before all chunks were written.
    if meta.get("complete") is False or missing:
        cache.close()
        detail = f" Missing arrays: {', '.join(missing)}." if missing else ""
        raise IncompleteCacheError(
            f"Cache '{cache_key}' is incomplete and cannot be opened.{detail} "
            "Run the same Test/Full pass again to replace it.")

    # Re-register the band-power features this cache actually holds, so the UI's
    # histogram list reflects the cache rather than the current config.
    #
    # First drop every band from any PREVIOUSLY opened cache. The registry is
    # module-global, so without this a cache built with a 12-24 Hz band leaves
    # `bandpower_12-24Hz` registered forever; open a new cache with a different
    # band and the UI still offers the old name, whose data is not in this cache
    # -- which is exactly the KeyError "can only come from the cache" crash.
    from core.features import REGISTRY, register_band
    for stale in [n for n in REGISTRY if n.startswith("bandpower_")]:
        del REGISTRY[stale]
    for b in meta.get("bands", []):
        register_band(f"bandpower_{b['lo_hz']:g}-{b['hi_hz']:g}Hz",
                      b["lo_hz"], b["hi_hz"])
    return cache


def cache_exists(cache_root: str, cache_key: str) -> bool:
    return os.path.exists(os.path.join(cache_dir(cache_root, cache_key), "meta.json"))


def cache_is_complete(cache_root: str, cache_key: str) -> bool:
    """Whether a cache is safe to offer as an alternative to recomputation.

    Legacy caches have no explicit flag; for those, array presence is the best
    available completion evidence.  New caches must carry ``complete: true``.
    """
    try:
        cache = open_cache(cache_root, cache_key)
    except (FileNotFoundError, IncompleteCacheError, KeyError, OSError, ValueError):
        return False
    cache.close()
    return True


def read_meta(cache_root: str, cache_key: str) -> dict | None:
    p = os.path.join(cache_dir(cache_root, cache_key), "meta.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def delete_cache(cache_root: str, cache_key: str) -> None:
    root = cache_dir(cache_root, cache_key)
    if os.path.exists(root):
        _rmtree_retry(root)


def purge_stale_test_caches(cache_root: str, video_hash: str,
                            keep_key: str | None = None) -> list[str]:
    """Delete test caches for this video other than `keep_key`.

    Test caches are scratch by definition -- they exist to answer "are these
    settings any good?" and are worthless the moment the settings change.
    Accumulating one per parameter tweak silently fills the disk with hundreds of
    megabytes apiece, so only the current one is kept. Full-pass caches are never
    touched: those are the expensive artefact the whole tool exists to produce.
    """
    removed = []
    for c in list_caches(cache_root):
        if not c.get("test_mode"):
            continue
        if c.get("video_hash") != video_hash:
            continue
        if keep_key is not None and c["key"] == keep_key:
            continue
        try:
            delete_cache(cache_root, c["key"])
            removed.append(c["key"])
        except OSError:
            pass
    return removed


def list_caches(cache_root: str) -> list[dict]:
    out = []
    if not os.path.isdir(cache_root):
        return out
    for key in os.listdir(cache_root):
        mp = os.path.join(cache_root, key, "meta.json")
        if os.path.exists(mp):
            with open(mp) as f:
                out.append({"key": key, **json.load(f)})
    return out


# -- size estimation (drives the Tab 1 "cost of this option" labels) ----------

def estimate_cache_bytes(cfg, width: int, height: int, n_frames: int,
                         fps: float) -> dict[str, int]:
    """Per-feature uncompressed byte estimate for a given config.

    Tab 1 shows these inline next to each expansion checkbox, so the user sees
    the cost before committing to an hour of compute. Compression typically
    takes 2-4x off these numbers for flow data, but we quote the uncompressed
    figure as the conservative bound.
    """
    from core.features import cached_feature_names

    scale = cfg.preprocess.resolve_downsample(width)
    w = max(1, int(round(width * scale)))
    h = max(1, int(round(height * scale)))
    block = cfg.flow.block_size
    ny, nx = h // block, w // block
    itemsize = 2 if cfg.features.dtype == "float16" else 4

    per_frame_plane = ny * nx * itemsize
    hop_frames = max(1, int(round(cfg.features.hop_s * fps)))
    n_windows = max(1, n_frames // hop_frames)

    out: dict[str, int] = {}
    for name in cached_feature_names(cfg):
        if name.startswith("bandpower_") or name in (
                "spectral_flatness", "direction_oscillation"):
            out[name] = per_frame_plane * n_windows
        else:
            out[name] = per_frame_plane * n_frames
    return out


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"
