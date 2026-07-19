"""Headless whole-video detection: the GUI's commit pass, with no Qt in scope.

This is the core half of Batch J (T24). ``gui.explorers.live_scalogram_surface
._ProcessWorker`` is the reference implementation and this must stay numerically
identical to it -- the whole point of the tuned-preview/commit split is that a
detection found headlessly can be re-opened in the explorer and look the same.
Both roads therefore run the same two calls, ``live_channel_source`` then
``detect_channel_region``, and neither reimplements any detector math.

Two things here deliberately differ from the GUI worker, both because the batch
case is many-regions-per-video where the interactive case is one:

  * **Extraction is paid once per video, not once per region.** The channel pass
    produces the whole atlas -- every replicate's blocks -- and
    ``detect_channel_region`` merely slices a region's columns out of it. The GUI
    extracts per commit because the user commits one region at a time; a batch
    run over six replicates would otherwise decode the same video six times for
    one video's worth of pixels.
  * **fps comes from the extraction meta, not from the decoder probe.** When
    clips are in play the manifest's rational rate is authoritative and the
    float OpenCV reports is not (``FINDINGS.md`` section 3 trap 2), and the
    wavelet bank is built from fps. ``detect_channel_region(freqs=None)`` reads
    ``meta['fps']``, so this passes nothing and lets it.

**Truncation is a hard error by default.** A decode that ends early yields a
short track, so every "no detection" past the cut point is unexamined rather
than examined-and-clear -- a silent false negative, and the failure this
codebase spends the most effort refusing. Interactively that can be a warning in
the status bar because a human is watching; across a fan-out of hundreds of jobs
nobody reads the logs of the runs that "succeeded". So a truncated pass fails the
job unless the caller opts in, and the shortfall is recorded in the result either
way.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, replace

import numpy as np

from core.channel_source import LIVE_CHANNELS, live_channel_source
from core.config import PipelineConfig
from core.detection import DetectionResult, detect_channel_region
from core.replicates import validate_replicates
from core.video import VideoSource

# The detector's default channel. ``change`` is J[2] -- squared frame
# differencing per block -- and is the one the explorer opens on.
DEFAULT_CHANNEL = "change"


class BatchError(RuntimeError):
    """A job that cannot be run or trusted. Distinct from a bug: these are
    expected operator errors (bad params, missing sidecar, short decode) and the
    CLI turns them into an exit code rather than a traceback."""


# -- params ------------------------------------------------------------------
# The tuned settings travel as JSON, because tuning happens in the explorer and
# running happens on a node with no display. The shape is
# ``ScalogramExplorer.detection_params()`` minus ``region_index`` (which the
# batch path iterates rather than selects).
#
# Band endpoints are the one awkward part: they are frequently infinite, and
# ``Infinity`` is not JSON. Rather than emit the non-standard token, an
# unbounded endpoint is written and read as ``null``. That also reads correctly
# to a human: "no lower bound", not "a very large negative number".

_BAND_KEYS = ("freq_band_hz", "value_band", "count_band")


def _band_from_json(v, key: str) -> tuple[float, float]:
    if v is None:
        return (-math.inf, math.inf)
    # Shape-checked before indexing so a hand-edited params file that says
    # `"value_band": 5` gets an operator-readable error, not a TypeError
    # traceback out of len().
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise BatchError(f"{key} must be a pair of endpoints (or null), got {v!r}")
    # Each endpoint's null takes the sign that WIDENS the band -- a null lower
    # bound is -inf, a null upper bound is +inf. Signing both the same way would
    # produce an inverted band that matches nothing, i.e. a detector that never
    # fires: the silent-false-negative shape again.
    lo = -math.inf if v[0] is None else float(v[0])
    hi = math.inf if v[1] is None else float(v[1])
    if lo > hi:
        raise BatchError(f"{key} is inverted: {lo} > {hi}")
    return (lo, hi)


def _band_to_json(band) -> list:
    return [None if not math.isfinite(float(v)) else float(v) for v in band]


def normalize_params(d: dict) -> dict:
    """Validate a params dict and put it in the shape ``detect_channel_region``
    wants. Accepts ``null`` (or a missing key) for an unbounded band."""
    ch = d.get("channel_attr", DEFAULT_CHANNEL)
    if ch not in LIVE_CHANNELS:
        raise BatchError(
            f"channel_attr {ch!r} is not one of {LIVE_CHANNELS}. (The cached-flow "
            "'speed' channel needs a pipeline cache and is not on the live path.)")
    win = int(d.get("detect_window", 1))
    if win < 1:
        raise BatchError(f"detect_window must be >= 1, got {win}")
    out = {"channel_attr": ch, "detect_window": win,
           "centered": bool(d.get("centered", False))}
    for k in _BAND_KEYS:
        out[k] = _band_from_json(d.get(k), k)
    lo, hi = out["freq_band_hz"]
    # -inf is "no lower bound" (band_indices handles it), not a negative
    # frequency; only a finite negative edge is meaningless.
    if math.isfinite(lo) and lo < 0:
        raise BatchError(f"freq_band_hz lower bound is negative: {lo}")
    return out


def params_to_json(p: dict) -> dict:
    return {"channel_attr": p["channel_attr"],
            "detect_window": int(p["detect_window"]),
            "centered": bool(p["centered"]),
            **{k: _band_to_json(p[k]) for k in _BAND_KEYS}}


def load_params(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return normalize_params(json.load(f))


# -- config overrides --------------------------------------------------------

def config_from_overrides(downsample: float | None = None,
                          block_size: int | None = None,
                          base: PipelineConfig | None = None) -> PipelineConfig:
    """A config with the two command-line overrides applied.

    Shared by every headless entry point rather than living in one of them: the
    validation here is policy, not argument parsing, and a second CLI that
    resolved these differently would produce results that disagree with the
    first while recording the same provenance.

    ``downsample`` is deliberately not a free knob -- it decides which
    behaviours are detectable at all -- so it is left at the config default
    unless a caller explicitly overrides it.
    """
    cfg = base or PipelineConfig()
    if downsample is not None:
        if not 0 < downsample <= 1.0:
            raise BatchError("--downsample must be in (0, 1]")
        cfg = replace(cfg, preprocess=replace(cfg.preprocess,
                                              downsample=float(downsample)))
    if block_size is not None:
        if block_size < 1:
            raise BatchError("--block-size must be >= 1")
        cfg = replace(cfg, flow=replace(cfg.flow, block_size=int(block_size)))
    return cfg


# -- replicates --------------------------------------------------------------

def load_replicates(path: str) -> list[dict]:
    """Replicate boxes from a sidecar. Accepts both shapes in the tree: a bare
    list, and the ``{"replicates": [...]}`` wrapper the GUI's exporter writes."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    reps = d.get("replicates") if isinstance(d, dict) else d
    if not isinstance(reps, list):
        raise BatchError(f"{path} holds no replicate list")
    validate_replicates(reps)      # raises on overlap / bad boxes, not advisory
    return reps


def default_replicate_path(video_path: str) -> str | None:
    """The sidecar next to a video, by the convention the tree already uses."""
    base = os.path.splitext(video_path)[0]
    for cand in (base + ".rois.json", base + ".replicates.json", base + ".json"):
        if os.path.exists(cand):
            return cand
    return None


# -- results -----------------------------------------------------------------

@dataclass
class RegionDetection:
    """One region's detector output. ``result`` holds the full per-frame series
    (and the retained band power); the summary fields are what a batch consumer
    reads without loading arrays."""
    replicate_id: int
    region_index: int
    result: DetectionResult
    intervals: list[tuple[int, int]]

    @property
    def n_detected_frames(self) -> int:
        return int(sum(e - s for s, e in self.intervals))

    def to_summary(self) -> dict:
        return {"replicate_id": self.replicate_id,
                "region_index": self.region_index,
                "n_intervals": len(self.intervals),
                "n_detected_frames": self.n_detected_frames,
                "intervals": [[int(s), int(e)] for s, e in self.intervals]}


@dataclass
class BatchResult:
    """One video's headless pass. ``truncated`` and the two frame counts are the
    coverage record: a consumer must be able to tell "examined and clear" from
    "never examined" without re-reading the video."""
    video_path: str
    params: dict
    regions: list[RegionDetection]
    meta: dict
    requested_frames: int
    covered_frames: int
    truncated: bool
    extract_seconds: float
    detect_seconds: float
    replicate_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    # The window as REQUESTED, which is not recoverable from the resolved
    # counts. A completed `--frames 1000` pass and a completed to-the-end pass
    # over a 1000-frame video both end up with requested == covered == 1000, so
    # only the raw request distinguishes "I asked for a thousand frames" from
    # "I asked for the whole video". Without it a resumed batch retains a
    # deliberately short smoke-test pass as if it were the complete answer.
    # `requested_n is None` means "to the end". See core/shard.py's skip logic.
    start_frame: int = 0
    requested_n: int | None = None

    @property
    def coverage(self) -> float:
        return (self.covered_frames / self.requested_frames
                if self.requested_frames else 0.0)

    def to_summary(self) -> dict:
        m = self.meta
        return {
            "video_path": self.video_path,
            "replicate_path": self.replicate_path,
            "params": params_to_json(self.params),
            "coverage": {
                "start_frame": int(self.start_frame),
                "requested_n": (None if self.requested_n is None
                                else int(self.requested_n)),
                "requested_frames": self.requested_frames,
                "covered_frames": self.covered_frames,
                "fraction": self.coverage,
                "truncated": self.truncated,
            },
            "timing": {"extract_seconds": round(self.extract_seconds, 3),
                       "detect_seconds": round(self.detect_seconds, 3)},
            # Provenance: everything needed to tell two results apart. The clip
            # keys are present only when the pass read pre-transcoded clips, and
            # matter because below `lossless` those are not the source's pixels
            # (FINDINGS.md section 10) -- a clip-derived and a source-derived
            # result are different measurements of the same footage.
            "provenance": {
                "fps": float(m["fps"]),
                "downsample": float(m["downsample"]),
                "block_size": int(m["block_size"]),
                "grid": [int(v) for v in m["grid"]],
                "src_width": int(m["src_width"]),
                "src_height": int(m["src_height"]),
                "replicate_geometry_hash": m.get("replicate_geometry_hash"),
                "channels_computed": m.get("channels_computed"),
                "clip_provenance": m.get("clip_provenance"),
                "clip_quality": m.get("clip_quality"),
                "config": m.get("config"),
            },
            "regions": [r.to_summary() for r in self.regions],
            "warnings": list(self.warnings),
        }


# -- the pass ----------------------------------------------------------------

def _region_indices(meta: dict, only: list[int] | None) -> list[tuple[int, int]]:
    """(region_index, replicate_id) pairs to detect over.

    Index is position in ``meta['replicate_tiles']``, which is what
    ``region_blocks_and_grid`` slices by; id is the stable handle an operator
    names on the command line. Selecting by id and resolving here means a job
    spec does not silently retarget when a replicate is added to the sidecar.
    """
    tiles = meta.get("replicate_tiles") or []
    pairs = [(i, int(t.get("id", i))) for i, t in enumerate(tiles)]
    if only is None:
        return pairs
    by_id = {rid: i for i, rid in pairs}
    missing = [r for r in only if r not in by_id]
    if missing:
        raise BatchError(
            f"replicate id(s) {missing} are not in this layout "
            f"(have {sorted(by_id)})")
    # De-duplicated, because the summary is keyed by replicate id and the .npz
    # is too: a repeated id would list the same region twice in the JSON while
    # writing its arrays once, leaving the two halves of one result disagreeing
    # about how many regions there are.
    seen: set[int] = set()
    uniq = [r for r in only if not (r in seen or seen.add(r))]
    return [(by_id[r], r) for r in uniq]


def run_video(video_path: str, replicates: list[dict], cfg: PipelineConfig,
              params: dict, *, regions: list[int] | None = None,
              manifest=None, clip_dir: str | None = None,
              start: int = 0, n: int | None = None,
              allow_truncated: bool = False,
              replicate_path: str | None = None,
              progress=None) -> BatchResult:
    """Extract one channel over the clip, then run the tuned detector over each
    selected region. The GUI commit pass for every replicate, at one video's
    decode cost.

    Raises ``BatchError`` on a short decode unless ``allow_truncated``.
    """
    # Idempotent: normalizing an already-normalized dict is a no-op, so callers
    # may hand this either a raw JSON dict or the output of normalize_params.
    params = normalize_params(params)
    with VideoSource(video_path) as src:
        info = src.info
        w, h, fps, fc = info.width, info.height, info.fps, info.frame_count
    # Rejected here rather than allowed to produce an empty pass: a start past
    # the end would otherwise return a zero-length, zero-detection result that
    # is indistinguishable from "examined the whole video, found nothing".
    if start < 0 or (fc and start >= fc):
        raise BatchError(
            f"--start {start} is outside {os.path.basename(video_path)}, "
            f"which has {fc} frames")

    t0 = time.perf_counter()
    cd = live_channel_source(
        video_path, cfg, replicates, start=start, n=n,
        width=w, height=h, fps=fps, frame_count=fc,
        manifest=manifest, clip_dir=clip_dir,
        channels=[params["channel_attr"]], progress=progress)
    extract_s = time.perf_counter() - t0

    # What the file actually holds PAST THE OFFSET. Both terms matter and each
    # was wrong once: asking for more frames than exist is the end of the video
    # rather than a short decode (so the request is clamped), and a window that
    # starts at `start` can never deliver more than `fc - start` (so the offset
    # is subtracted even when n is None). Without the latter, every --start run
    # without an explicit --frames failed the truncation guard -- and failed
    # only after paying for the whole extraction. Truncation is the decode
    # delivering less than the file claims, which is cd.meta['truncated'].
    available = max(0, fc - start)
    requested = available if n is None else min(int(n), available)
    covered = int(cd.n_frames)
    truncated = bool(cd.meta.get("truncated")) or covered < requested
    warnings: list[str] = []
    if truncated:
        pct = f" ({covered / requested:.1%})" if requested else ""
        msg = (f"decode covered {covered} of {requested} requested frames"
               f"{pct}; everything after frame "
               f"{start + covered} is UNEXAMINED, not clear")
        if not allow_truncated:
            raise BatchError(msg + " -- pass allow_truncated to accept it")
        warnings.append(msg)

    t1 = time.perf_counter()
    out: list[RegionDetection] = []
    for idx, rid in _region_indices(cd.meta, regions):
        # freqs=None on purpose: detect_channel_region then builds the bank from
        # cd.meta['fps'], which is the manifest's rational rate when clips are in
        # play rather than the decoder's rounded float.
        res = detect_channel_region(
            cd, idx, params["channel_attr"],
            freq_band_hz=params["freq_band_hz"],
            value_band=params["value_band"],
            count_band=params["count_band"],
            detect_window=params["detect_window"],
            centered=params["centered"])
        out.append(RegionDetection(replicate_id=rid, region_index=idx,
                                   result=res,
                                   intervals=res.detected_intervals()))
    detect_s = time.perf_counter() - t1

    return BatchResult(video_path=video_path, params=params, regions=out,
                       meta=cd.meta, requested_frames=requested,
                       covered_frames=covered, truncated=truncated,
                       extract_seconds=extract_s, detect_seconds=detect_s,
                       replicate_path=replicate_path, warnings=warnings,
                       start_frame=int(start),
                       requested_n=None if n is None else int(n))


# -- output ------------------------------------------------------------------

def write_result(res: BatchResult, out_path: str, *,
                 save_series: bool = True, save_band_power: bool = False) -> None:
    """Write the JSON summary, and (unless disabled) an ``.npz`` beside it with
    the per-frame series.

    ``band_power`` is (T, B) per region and is the large one -- an hour at 30 fps
    over a few hundred blocks runs to hundreds of MB per region -- so it is
    opt-in. It is the only array that makes value-band and detection-window
    re-tuning possible without a fresh pass, which is why it is offered at all.
    """
    summary = res.to_summary()
    if save_series:
        npz = os.path.splitext(out_path)[0] + ".npz"
        arrays = {}
        for r in res.regions:
            p = f"r{r.replicate_id}"
            arrays[f"{p}_count"] = np.asarray(r.result.count, np.float32)
            arrays[f"{p}_windowed"] = np.asarray(r.result.windowed, np.float32)
            arrays[f"{p}_gate"] = np.asarray(r.result.gate, np.float32)
            arrays[f"{p}_clump"] = np.asarray(r.result.clump, np.float32)
            if save_band_power:
                arrays[f"{p}_band_power"] = np.asarray(r.result.band_power,
                                                       np.float32)
        np.savez_compressed(npz, **arrays)
        summary["series_path"] = os.path.basename(npz)
        summary["series_includes_band_power"] = bool(save_band_power)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    os.replace(tmp, out_path)      # atomic: a killed job leaves no torn summary
