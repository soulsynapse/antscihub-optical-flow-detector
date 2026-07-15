# Archived initial project handoff

> Archived on 2026-07-15. This was the original design brief and is preserved for
> historical context. It assumes full-frame flow and automatic ROI discovery,
> both of which were superseded by the measured replicate-first workflow. See the
> [current README](../../README.md), [design decisions](../decisions.md) and
> [next steps](../next-steps.md). Nothing below is a current implementation
> contract.

# General optical flow behavior detection tool

## Context

Building a domain-general desktop tool for detecting animal behaviors in
video via dense optical flow. The workflow:

1. Load a video, apply configurable preprocessing, compute dense flow, and
   cache a set of per-pixel/per-block features (magnitude, angle,
   coherence, band-powers, etc.).
2. Discover ROIs by selecting ranges on feature histograms — pixels matching
   the selected ranges cluster into candidate regions.
3. Classify behaviors within (or independent of) those ROIs using the same
   histogram-range-selection model applied to a broader feature set,
   including temporal features. Behavior definitions save to a library and
   reload across projects.

Core interaction model, everywhere: **Lightroom-style live histograms with
draggable range selectors.** The user picks a range on each feature's
histogram; overlays and detected regions update live to show what matches
the intersection. This should feel like Lightroom's tone-curve/HSL panel:
histograms always visible, always reflecting the current filter state.

Behavior specifications should be a small logical tree (AND/OR nodes with
range leaves), not a flat AND, so more complex signatures — harmonics,
either-or conditions — can be expressed. Flat AND is the common case; the
tree just gives headroom.

The tool is domain-general. First use case is grasshopper wingbeat
detection (signature: high band-power at ~20 Hz, small spatial extent,
low net flow), but it must work equally for ant fanning, walking gaits,
grooming, hovering, or any behavior with a distinctive flow signature. Do
not hardcode assumptions about frequency bands, spatial scales, or the
number of behaviors of interest.

## Reference project

Scaffold the UI framework, tab structure, video playback, hotkeys, and
"low/high slider window" style from
https://github.com/soulsynapse/antscihub_color_detector_v2 (also available
in the workspace — inspect it directly for framework, layout, and playback
conventions, and match them). Where that project uses click-to-sample-color,
this tool uses click-to-inspect-detected-ROI. Keep the histogram-driven
interaction philosophy but generalize from color channels to arbitrary
flow-derived features.

## Cache sizing philosophy

The feature cache stores pre-filter data so histogram-range tuning in
Tabs 2/3 is interactive. Naive settings (full resolution, small blocks,
fat feature set, float32, no compression) produce tens of GB per 5-minute
clip. Sensible defaults trim this by 50–100× with no meaningful loss for
most use cases. Design the tool to **default to a lean cache and let the
user expand it if detection is unreliable**, not the other way around.

Lean defaults (target: hundreds of MB for a 5-minute 1080p clip):
- Downsample the video to half-resolution before flow computation.
- Block size 16×16.
- Minimal feature set stored to disk: magnitude, angle, and a small
  configurable band-power set (default a single band appropriate to the
  behavior, e.g. 15–30 Hz for wingbeat).
- float16 storage for all features.
- Chunked, compressed storage (blosc/zstd). Chunk along the time axis so
  per-ROI extraction and histogram updates are fast.

Expansion options, exposed in Tab 1 with clear labeling as "expand cache
if detection is unreliable":
- Full or higher resolution.
- Smaller block size (8×8 or per-pixel).
- Additional stored features (coherence, divergence, curl, rolling
  stats, extra band-power ranges, dominant frequency, spectral flatness,
  direction oscillation index).
- float32 storage.
- Reduced or no compression.

Each expansion option in the UI should show its estimated cache size and
compute-time cost inline, computed from the current video's resolution
and duration, so the user can see the tradeoff before running.

**Derived features computed on demand from cached ones don't need to be
stored.** E.g. rolling stats can be recomputed from magnitude, extra
band-powers from magnitude, etc. Only cache the features that are either
(a) expensive to compute from scratch (band-powers with long windows) or
(b) fundamental (magnitude, angle). The feature registry should mark
which features are cached vs. on-demand.

## Program structure — three tabs

### Tab 1: Preprocessing & Flow Computation

Purpose: load video, define what the flow computation sees, run the
(expensive) flow computation, cache derived features to disk.

Capabilities:
- Load video; read fps, resolution, duration.
- Configurable preprocessing pipeline, each step toggleable, all with
  sensible defaults so a first-time user can hit "Run test" immediately:
    - Spatial ROI mask (exclude regions like ceiling, shelving, glare).
      Default: none.
    - Frame registration for camera motion. Feature-based (ORB +
      homography to a reference frame) and phase-correlation options.
      Default: off, since some footage doesn't need it — but the (?)
      help should be explicit that behaviors in the same temporal band
      as camera motion require this on.
    - Temporal denoising (rolling median or Gaussian). Default: off.
    - Background subtraction (temporal median or configurable BG model).
      Default: off.
    - Downsampling factor. **Default: 0.5** (half-res).
    - Contrast / illumination normalization. Default: off.
- Optical flow configuration. **Pluggable flow backend** behind a common
  interface:
    - Farnebäck (CPU) — default, adequate for coarse-scale behaviors
      like wingbeat.
    - DIS (CPU) — better for subtle motion.
    - RAFT (GPU) — best for fine motion (antennal movement, grooming).
  Detect GPU availability at startup and enable RAFT accordingly;
  degrade gracefully with a clear message if the user selects a backend
  that isn't available. Block size in this config, default 16×16.
- Feature extraction. Features are defined declaratively in a feature
  registry (adding a new feature is a matter of registering it, not
  editing the UI). Each entry declares: name, compute function, whether
  it's cached to disk or computed on demand, and any dependencies on
  other features.
  Starter cached set (defaults):
    - flow magnitude
    - flow angle
    - band-power in a user-configurable list of frequency bands
      (default: one band, 15–30 Hz)
  Starter on-demand set (available in Tabs 2/3 without expanding cache):
    - rolling mean/std of magnitude
    - dominant frequency of magnitude (from cached magnitude time series)
  Optional cached expansions (user opts in explicitly, with size cost
  shown):
    - angular coherence in local neighborhood
    - divergence, curl
    - additional band-powers
    - spectral flatness
    - direction oscillation index (autocorrelation of angle)
- **Test mode**: process the first N seconds (default 10, settable),
  writes to a scratch cache. Jumps directly into Tab 2 with the partial
  cache so the user can decide whether to commit to the full pass or
  return here to adjust.
- **Full mode**: process the whole video. Persist the feature cache to
  disk, keyed by (video_hash, preprocessing_config, flow_config) so
  reopening skips recomputation. Progress bar with ETA.
- Every input has a (?) button opening a small dialog with: what the
  parameter does, effect of raising/lowering, defaults rationale, and
  how to tell from the downstream histograms whether it's set well.
- Settings export/import: full preprocessing + flow + feature config to
  JSON. Include a version field and the target video's hash so imports
  can warn if config was tuned on different footage.

### Tab 2: ROI Discovery

Purpose: from cached features, find spatially and temporally stable
regions where "something of interest is happening" — before committing
to a specific behavior label.

Capabilities:
- Show a video preview with a live overlay of "matched pixels" (pixels
  currently satisfying the ROI feature filter).
- Feature-histogram panel: for each feature the user opts in, show a
  histogram with a draggable range selector (a "keep this range" band,
  Lightroom-style). Histograms show the distribution across all
  frames/pixels in the current cache. Range selectors update overlay live.
  On-demand features (not in the cache) are computed lazily and cached
  in RAM per session as the user opts them into the histogram panel.
- Spatial post-filtering: minimum region area, morphological open/close,
  temporal-stability requirement (region must persist for N frames).
- Automatic ROI extraction: run the current filter over the full cache,
  extract connected components meeting spatial+temporal criteria, assign
  stable IDs. ROI list appears in a side panel with thumbnails.
- **Click any ROI** to inspect: show its bounding box on the video,
  per-feature time series for the ROI (mean over ROI pixels), and PSD of
  key features. Same panel as Tab 3 will use.
- All standard video playback controls and hotkeys from the reference
  project (play/pause, frame step, scrub bar, jump-to-time).
- Save/load ROI configurations. Export ROI list as JSON + mask PNGs +
  per-ROI feature time series (CSV or HDF5).
- If the user finds their filter can't isolate the behavior with the
  current cached features, provide a clear one-click path back to Tab 1
  to enable additional cached features (with size cost shown) and rerun.

### Tab 3: Behavior Classification

Purpose: define behaviors as filters over feature time series (and
optional spatial features), apply to ROIs or globally, export detections.

Capabilities:
- Behavior library: user-defined behaviors, each with a name, color, and
  specification tree (AND/OR of feature range constraints). Behaviors
  save/load individually so a "wingbeat.json" or "walking.json" can move
  between projects.
- Behavior editor: build the AND/OR tree; for each leaf (a feature +
  range), show the corresponding histogram with the range selector on it.
  Histograms in this tab reflect the *time-series* distribution over ROIs,
  not the pixel-space distribution used in Tab 2 (so band-power histograms
  are per-ROI-per-window here, per-pixel-per-frame there).
- Temporal criteria per behavior: minimum sustained duration, minimum
  gap-fill (bridge short dropouts), and any smoothing on the binary trace.
- Live: for the selected behavior, video overlay colors ROIs by whether
  they currently match. A timeline strip beneath the video shows behavior
  state per ROI across the whole video (like a raster plot).
- Click an ROI → inspection panel with:
    - all feature time series (raw and smoothed),
    - PSDs of magnitude and angle,
    - binary behavior traces for every defined behavior,
    - which specific filter constraints are met/unmet at the current
      timestamp (so the user can debug "why isn't this classified as X").
- Export: per-ROI CSV/HDF5 with feature time series and per-behavior
  binary traces. Also a summary CSV: for each ROI × behavior, total time,
  bout count, mean bout duration.
- Behavior definitions export/import separately from the pipeline config.

## Cross-cutting requirements

- All feature computation is cacheable and keyed by
  (video_hash, preprocessing_config, flow_config) so retuning downstream
  filters is instant.
- **Storage format**: chunked and compressed, memory-mappable. HDF5
  (h5py + blosc) and Zarr are both reasonable — benchmark both on a
  representative clip (write throughput, random-access read latency for
  scrubbing, compressed size on disk) and pick based on results.
  Document the choice in the code. Chunk along the time axis.
- All histograms in Tabs 2 and 3 honor the current filter state: when a
  range on histogram A moves, histogram B's density updates to show only
  pixels/ROIs matching A. This is what makes multi-feature tuning
  tractable — the user can see whether the intersection is well-separated
  from noise.
- Time everywhere is in seconds and Hz, never frames. Frame indices only
  in tooltips.
- ROI IDs are stable across parameter changes when possible (spatial
  overlap tracking), so retuning a filter doesn't lose per-ROI notes.
- Ask clarifying questions before writing code if any of this is
  ambiguous. Recommend a framework based on the reference project and
  confirm before scaffolding. Suggest optimizations (GPU flow, further
  downsampled compute) if the naive path won't hit reasonable performance.

## Suggested build order

1. Inspect the reference project. Confirm framework and UI patterns.
2. Build the feature registry with cached/on-demand distinction. Get
   preprocessing + flow + lean-cache pipeline working headless on a
   short clip. Verify histograms of each feature look sensible.
   Benchmark HDF5 vs Zarr here and commit to one.
3. Tab 1 UI with test-mode/full-mode, pluggable flow backend, and
   inline size/time estimates on cache-expansion options.
4. Tab 2 with live histogram-and-overlay tuning, ROI extraction, and
   the one-click "expand cache" path back to Tab 1.
5. Tab 3 with behavior tree editor, timeline strip, and export.
6. Settings/behavior import/export and (?) help dialogs last.
7. Ship a "wingbeat" behavior definition as a bundled example.
