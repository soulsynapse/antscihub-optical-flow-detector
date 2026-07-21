# Current design decisions

Status: current as of 2026-07-20. This document describes the live configuration
version 3 workflow. The archived handoffs are historical inputs, not overrides.

**The flow cache has been removed.** The **tensor/scalogram detection path** is
now the only detection path — it runs the whole isolate-a-behavior loop without an
optical-flow cache (see "The tensor/scalogram path runs without a flow cache" and
"Commit means running the detector, not building a cache" near the end). The
flow-pipeline and cache sections below ("Cache raw evidence", "Cache precision
stops at float16", "Zarr and sparse atlases", "Cache compatibility is explicit")
describe that removed subsystem and are retained only as historical rationale.

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

FFmpeg emits the exact replicate rectangles with `crop:exact=1`, so an odd-valued
box origin is never silently rounded to an even boundary. That rounding shifted
the window by up to a whole source pixel, and — being inconsistent frame to frame
— dense flow read the sub-pixel offset as real translation. Before dense flow,
each crop then gets a private reflected border derived only from its own edge
pixels, giving the solver numerical context without reading a neighboring
replicate. The border is removed before block reduction.

The final partial block on a drawn right or bottom edge keeps its real pixel
count, and every downstream area count now weights it by that valid area. Clump
size, passing-block count and detection strength treat a one-pixel-tall edge
sliver as the fraction of a block it actually is, so a row of slivers can no
longer masquerade as a real clump (`core.replicates.block_weight_plane`). A
threshold failure never causes underlying continuous flow values to be discarded.

The synthetic reflected border is a known compromise, not a settled ideal: a
mirror is a poor input for a translation estimator, and on difficult footage it is
the main reason CLAHE edge contrast becomes phantom edge speed. We still do not
read source pixels from *another* replicate box — that would import a neighbor's
motion and break isolate independence. Replacing the mirror with a small real
source halo drawn only from background between boxes (Voronoi-clipped so it never
enters a neighbor) is planned; see KNOWN_ISSUES.md and next-steps §9.

## Preprocessing state belongs to one replicate

Registration, temporal denoising, background models and normalization each have
independent state per replicate. A single video-wide normalization was rejected
because replicate illumination and contrast can differ even within one frame.

Per-frame z-score normalization is now the default (it previously defaulted off).
It is boundary-safe — a single global per-frame mean/std rescale, with none of
CLAHE's per-tile edge behavior — and near-neutral for the common case of a small
target in a larger box, where its global statistics barely move frame to frame,
while still countering slow ambient drift. The earlier concern that per-frame
scalar normalization can erase temporal amplitude is not discarded; it bounds
where z-score is trusted. When a target fills most of its box, its own motion
shifts the per-frame statistics and the rescaling can inject inter-frame contrast
changes that flow misreads. That regime must be confirmed in the normalization
validation (next-steps §1), and `off` stays available for it.

CLAHE is opt-in and now carries an explicit warning in the flow tab. Its per-box,
per-frame local equalization has a known replicate-boundary artifact: truncated
edge-tile histograms amplify into phantom edge speeds, and because the tile grid
shifts with any added context it perturbs every block, not only the edge. On the
reference clip, replicate 23 peaked at 861 px/s with CLAHE versus 48 with it off.
It must earn its place through marked-video validation rather than being a
universal correction. See KNOWN_ISSUES.md.

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

## Cache precision stops at float16

Flow blocks store as float16 by default; float32 is available but doubles size for
precision the data does not carry. The right precision is set by the intrinsic
uncertainty of a block flow estimate (order 5–10%): float16's ~0.05% relative
quantization sits far below that, so it is effectively lossless for this signal,
while float32 mostly stores noise more precisely.

Going below float16 was considered and rejected. An 8-bit float (E4M3) tops out at
448 and would clip the fastest block speeds outright — edge artifacts alone reach
~860 px/s — and its ~6–12% relative step is comparable to the real measurement
uncertainty, so it would add noise where float16 adds none, worst exactly in the
low-speed baseline the standardized noise references depend on. It also has no
native numpy support in the current stack. Band power is stored as float32
regardless, because it sums PSD over a band and can exceed float16's range. When
cache size is the real constraint, block size is the correct lever: it scales
storage roughly quadratically and trades spatial resolution that can be reasoned
about, not precision that cannot.

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

## The tensor/scalogram path runs without a flow cache

The structure-tensor channels an explorer detects on — `tensor_speed`, change
energy, intensity, and an appearance residual — need only geometry and the video,
not the optical-flow solve. The flow cache only ever fed two of them: the
cached-flow-speed channel and the residual measured against *cached* flow. So the
detection loop was decoupled from the cache.

`core.channel_source.ChannelData` carries what the scalogram explorer needs —
cache-meta-shaped geometry plus per-block channel time series — from a live
windowed pass over a bare video (`live_channel_source`). The cache-backed source
(`cache_channel_source`) was removed with the rest of the flow-cache subsystem;
the live pass is now the only source. The explorer takes a `ChannelData`, not a
cache. The appearance residual on the live path is measured against the tensor's
**own** per-pixel flow (`I_t + ∇I·v` with `v` from `flow_from_tensor`), not cached
`u`/`v`, which removes the last flow-array dependency; this is the
`flow_residual(J)` flavor anticipated in the expanded-cache plan.

The preprocessing controls that actually feed the tensor channels — `downsample`,
`block_size`, `normalize` — are live knobs on the surface. All three are upstream
of the tensor solve, so changing one re-extracts the window rather than
transforming the existing result; that cost is accepted because a window is short.
`normalize` is a per-frame pixel op (z-score is ~invariant for `tensor_speed`, a
gradient-ratio solve, and reshapes change/intensity); `block_size` is a geometry
op whose expensive per-pixel tensor solve is actually block-size independent (only
the final block reduction depends on it — a latent optimization, not yet needed).
`registration` and `bg_subtract` do not apply to this path (they need fitted
assets the pass does not reconstruct) and are flagged approximated. **Temporal
denoise is forced off** for windowed extraction: it is stateful from frame zero,
so an arbitrary mid-clip window cannot reproduce its state. This is what makes
true random-access windows honest — pick any 10 s and pay only for those frames.

## Commit means running the detector, not building a cache

For a behavior isolable by a band on a tensor channel, "commit" is running the
tuned detector over the whole clip and navigating the result — not writing a flow
cache. *Process whole video* streams the clip once and applies the detector,
retaining only the per-block band power `(T, B)` and the detection tracks. It
never materializes the full `(F, T, B)` scalogram cube (~17 GB for a long clip):
`core.wavelet.morlet_band_power` derives the zero-pad length from the whole bank's
largest scale so a band sum matches a full-cube slice exactly, yet transforms only
the band's scales, chunked over blocks — ~440 MB for one channel over a whole clip.

The detection formulas (in-band count, windowed mean over D, positive gate,
per-frame clump) live in `core/detection.py` as pure functions that **both** the
explorer preview and the whole-clip pass call. This is deliberate: if the two
forked, you could navigate to a detection found over the whole clip, re-open its
window to verify, and see a different result — which would make the tool
untrustworthy at exactly the moment it matters. The whole-clip result becomes a
navigation timeline (gate plus largest-clump strength); clicking a detection loads
its window back for verification. Because the band power is retained, value-band
and detection-window re-tuning is instant, while a frequency-band or channel
change requires a fresh pass.

Divergence and curl — two of the flow-derived features a cache once justified —
are now tensor-path DERIVED channels (`core.channels` velocity-gradient family:
`vel_divergence`, `vel_vorticity`, `vel_shear`), taken from the block flow `u`/`v`
the structure tensor already solves, so they need no cache. Coherence and
direction oscillation are not yet exposed. The flow cache and the flow-cache-only
Behavior tab are both removed (see the top of this document). The velocity-gradient
channels are built but not yet validated on marked footage.

## Explorers stay partitioned by measurement domain

The three sibling explorers deliberately divide the signal space: the speed
explorer reads per-frame motion magnitude, the coherent-flow explorer reads the
flow vector integrated over a window, and the structure tensor explorer reads
temporal intensity change integrated over a window. Porting speed detection
into the structure tensor explorer was considered and rejected.

Detection channels in an explorer are per-block density heatmaps with a
draggable tail band, never spatial means: a replicate is mostly empty space, so
a mean buries the sparse behaving blocks that are the signal. Bands belong on
heavy-tailed nonnegative energies where "threshold the tail" is meaningful;
bounded ratios (appearance fraction, speed disagreement) are kept as band-less
diagnostic views because a ratio lets a barely-above-floor block at 1.0 outrank
a real event at 0.7 with a hundred times the energy.

A speed band in the tensor explorer would duplicate the speed explorer as a
strictly worse copy — its clump sweep lacks the gap-bridged clustering — and
would create two disconnected speed thresholds with no answer to which one is
real. It also sits outside the explorer's organizing axis: everything
detection-related there flows through the integration-window prefix sums, and
per-frame speed would either silently ignore the window slider or become a
windowed speed mean, a worse version of what the coherent-flow explorer already
does by integrating the vector (so opposing motions cancel instead of averaging
into fake sustained speed). The tensor explorer's existing speed channels
(tensor vs cached speed and their disagreement) are validation reads that flag
where brightness-constancy or large-displacement assumptions fail; they mark
where to distrust the other channels, and overloading them as detection targets
would blur that role. Processing cost was explicitly *not* a factor: cached
speed is already loaded there, so a speed heatmap would be nearly free — the
objection is redundancy and semantics, not compute.

Cross-channel conjunctions ("high appearance energy AND low speed") are the
behavior tree's job, not an explorer's: explorers calibrate individual
`RangeLeaf` thresholds, and combination logic composes downstream. For joint
inspection, the explorers already share AppState frame sync for side-by-side
use, and frame-indexed CSV export covers offline joint analysis. The one
accepted follow-up in this space is converting the speed-disagreement channel
into a density heatmap — its per-block distribution exists nowhere else, unlike
speed's.

## User-interface defaults follow the common path

Draw mode and fixed-size stamping start enabled because most experimental layouts
use repeated boxes. The first drag sets the stamp size; later clicks place boxes.

The tabs are Replicates → Preprocessing (live) → Flow cache (commit) → Behavior
Classification, in that order, reflecting that the tensor detection loop is the
primary path and the flow pass is a demoted, optional commit. Loading a video
enables the first three tabs and lands on Replicates; Behavior stays disabled
until a completed flow cache is opened. The live preprocessing surface builds
lazily — only when it is shown with a video and at least one replicate box, and it
rebuilds only when the video or replicate geometry actually changes, so editing
boxes on another tab does not trigger a windowed extraction.

## Cache compatibility is explicit

Version 3 cache keys include replicate geometry as well as video, preprocessing,
flow and feature settings. A per-replicate cache refuses to open against missing
or different boxes. Incomplete caches are never presented as complete results.
Older full-frame caches should be rebuilt rather than silently interpreted as
ROI-first data.
