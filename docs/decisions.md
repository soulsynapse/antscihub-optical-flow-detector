# Current design decisions

Status: current as of 2026-07-15. This document describes the live configuration
version 3 workflow. The archived handoffs are historical inputs, not overrides.

## Replicates come before flow

The experimental units are known before processing, so the user draws or imports
replicate boxes first. The old automatic ROI-discovery tab was shelved: on the
reference footage, camera motion and motion on static structures made automatic
pixel-space discovery harder to tune than the biologically meaningful replicate
layout.

Boxes have exclusive ownership. Positive-area overlaps are rejected because an
overlapped source pixel would otherwise be processed twice and could leak one
animal's motion into two isolates. Edge-touching is allowed.

Only geometry participates in cache identity. Labels, baseline intervals,
pixels/mm and body length are analysis metadata and may change without
recomputing flow. Adding, deleting, moving or resizing a box invalidates the
matching cache.

## Crop exactly, create support synthetically, then discard it

FFmpeg emits the exact replicate rectangles. Before dense flow, each crop gets a
private reflected border derived only from its own edge pixels. This gives the
solver numerical context without reading a neighboring replicate. The border is
removed before block reduction.

The final partial block on a drawn right or bottom edge is retained with its real
pixel count. Padding never changes ownership, and a threshold failure never
causes underlying continuous flow values to be discarded.

We do not use source-image padding around a replicate. That would be harmless for
widely separated grasshopper tubes but unsafe for close ant isolation assays.

## Preprocessing state belongs to one replicate

Registration, temporal denoising, background models and normalization each have
independent state per replicate. A single video-wide normalization was rejected
because replicate illumination and contrast can differ even within one frame.

Per-frame scalar normalization was also rejected as a default. It forces every
frame toward the same distribution and can erase the temporal amplitude that
behavior detection is supposed to measure.

CLAHE remains opt-in. It is spatially local and adapts its mapping on every frame,
which can help subtle ambient drift but can also create apparent motion in weakly
textured areas. It must earn its place through marked-video validation rather
than being enabled as a universal correction.

The optional within-replicate mask remains available for a fixed nuisance inside
a useful box. It does not reduce FFmpeg output geometry, flow-solver geometry or
cache size, and hard mask boundaries can corrupt nearby flow. It is an advanced
exclusion tool, not a second ROI system.

## Cache raw evidence; derive tunable judgments

The fundamental cached flow quantities are block-mean `u`, block-mean `v` and
mean per-pixel speed. Caching an ordinary mean angle would introduce a circular
statistics bug at the 0°/360° seam. `atan2(v, u)` gives the correct block direction,
and `|mean vector| / mean speed` gives angular coherence at no additional storage
cost.

Thresholds, unit conversion, spatial medians and baseline-relative features stay
outside the raw cache so their effects remain visible and adjustable.

The per-replicate noise reference is fixed over time. An explicit quiescent
interval supplies its p99 when available. The automatic fallback takes a spatial
p25 for every frame and then one temporal p99 of that series. It does not divide
each frame by its own noise estimate.

## Diagnostics are evidence, not automatic rejection

Forward/backward consistency is cached as a continuous per-block p90 residual.
Backward vectors are not retained, and no error threshold is baked into cached
flow. The option adds another dense-flow solve and is intended for sampled QC,
not routine processing of hundreds of hours.

Texture is cached as block-mean `cornerMinEigenVal`; a per-frame,
per-replicate percentile is derived later. This keeps low-texture masking
tunable and prevents one high-texture replicate from defining another
replicate's percentile scale.

Spatial medians, divergence, curl, texture ranks and relative-background
features are computed separately inside each packed replicate tile. Sparse atlas
separators are storage coordinates, never biological neighbors.

## CPU FFmpeg is the supported decode path

The performance gain came from decoding once and moving only small grayscale ROI
pixels into Python, not from CUDA. The measured 10-second DIS production run took
6.6 seconds with CPU FFmpeg.

CUDA/NVDEC is not required. The 5312×2988 MPEG-4 Part 2 reference stream exceeds
the [NVDEC MPEG-4 resolution limit](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html),
and hardware decode can lose its advantage when frames must return to CPU
preprocessing and DIS/Farnebäck. GPU flow through RAFT is a separate optional
capability.

The current fallback is OpenCV full-frame decode. It preserves functionality but
is substantially slower, so improving ordinary FFmpeg discovery is more valuable
for general users than maintaining a custom CUDA-FFmpeg setup guide.

## Zarr and sparse atlases are deliberate

Zarr was faster for the repeated time-series access pattern, even though HDF5
wrote and cold-sought faster. Each replicate grid is packed into one regular
time-major atlas so Zarr reads remain efficient. Metadata maps atlas tiles back
to exact source-frame boxes, and every spatial feature respects those tile
boundaries.

## User-interface defaults follow the common path

Draw mode and fixed-size stamping start enabled because most experimental layouts
use repeated boxes. The first drag sets the stamp size; later clicks place boxes.

Processing completion and cache opening stay on Preprocessing & Flow. The
application only redirects to Replicates when valid geometry is missing. Opening
a cache enables Behavior Classification without assuming which inspection step
the user wants next.

## Cache compatibility is explicit

Version 3 cache keys include replicate geometry as well as video, preprocessing,
flow and feature settings. A per-replicate cache refuses to open against missing
or different boxes. Incomplete caches are never presented as complete results.
Older full-frame caches should be rebuilt rather than silently interpreted as
ROI-first data.
