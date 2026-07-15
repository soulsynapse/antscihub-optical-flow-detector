# Optical Flow Behavior Detector

Domain-general detection of animal behaviors from dense optical flow, driven by
Lightroom-style histogram range selection over flow-derived features.

    python main.py

Requires the venv in `.venv` (rebuild with `python -m venv .venv` and
`pip install -r requirements.txt` — the one in git history was built for a
different path and does not work after the folder rename).

## The three tabs

1. **Preprocessing & Flow** — load a video, configure preprocessing and the flow
   backend, run the expensive pass, cache features to disk. Test mode processes
   the first N seconds and drops you into Tab 2 with the partial cache.
2. **ROI Discovery** — cross-filtered feature histograms with draggable ranges, a
   live matched-block overlay, and connected-component ROI extraction with IDs
   that survive a filter retune.
3. **Behavior Classification** — an AND/OR spec tree over feature ranges, temporal
   criteria, an ethogram raster, and CSV/HDF5 export.

Time is in **seconds** and frequency in **Hz** everywhere. Frame indices appear
only in tooltips and in one clearly-labelled export column.

## Design decisions worth knowing

### What is cached, and why it is not what the spec said

The spec called for caching `magnitude` and `angle`. This caches the **block-mean
flow vector** (`u`, `v`) and the **block-mean speed** (mean of per-pixel |flow|)
instead. Same three arrays, same disk cost, strictly more information:

- Averaging `angle` over a block is a **circular-mean bug**. A block whose pixels
  point at 1° and 359° averages to 180° — the exact opposite of the truth.
  Averaging the vector components and taking `atan2` afterwards is the correct
  magnitude-weighted circular mean.
- `|mean vector| / mean|vector|` is **angular coherence**, free. It is ~1 for rigid
  translation and ~0 for motion that cancels within a block. A wingbeat is exactly
  the low-coherence, high-speed, low-net-flow case — so the feature that most
  directly encodes the target signature costs nothing.

Consequently `angle`, `coherence`, `net_speed`, `divergence`, `curl`, rolling
stats, `dominant_freq`, `spectral_flatness` and `direction_oscillation` are all
**derived on demand** and are available in Tabs 2/3 without expanding the cache.
The "expand cache" options only save recomputation; they do not unlock anything.

### Storage: Zarr

Benchmarked head-to-head against HDF5 on full-clip-shaped data
(`scripts/benchmark_storage.py`). Identical compressed size; HDF5 is faster to
write and to cold-seek; **Zarr is 2.3× faster on the ROI time-series read**, which
is the one access pattern that cannot be cached away (it touches every chunk by
construction). Scrubbing is a non-issue on both once the read-through chunk cache
is in place — under 0.1 ms. Zarr also isolates interrupted writes to independent
chunks; the GUI refuses to open an incomplete cache, but its surviving metadata
and chunks remain diagnosable. Full reasoning in `core/cache.py`.

### Standardization without destroying the raw signal

The cache-time standardization options are deliberately limited to evidence that
cannot be recovered without the frames:

- **Forward/backward error is a diagnostic, not a production default.** It
  computes forward and backward flow at full working
  resolution, evaluates `||F(x) + B(x + F(x))||` pixelwise, and caches the p90
  residual per block. It is continuous and unthresholded. Enabling it roughly
  doubles the number of dense-flow solves; it does not replace `u`, `v`, or
  `speed`. Leave it off for corpus-scale processing.
- **Texture strength** runs `cornerMinEigenVal` on each preprocessed frame and
  caches its block mean. The derived `texture_percentile` feature makes the
  per-frame low-texture cutoff tunable (`>= 0.25` is the proposed starting point).

Everything cheap remains analysis-time and explicitly named so raw and adjusted
results can be compared: `median3_*` features and per-replicate speed divided by
an explicit quiescent-baseline p99. When no clean baseline exists, the automatic
fallback takes the spatial p25 in every frame and uses the temporal p99 of that
series as one fixed reference for the whole replicate. It does not renormalize
each frame independently or pretend the spatial p25 itself is a noise p99.

Each replicate box can also carry source-pixels/mm, body length, and its own
baseline interval in the per-video ROI sidecar. Physical-unit features account
for cache downsampling at read time; cached pixel-space values are never modified.

CLAHE remains an opt-in preprocessing mode. OpenCV CLAHE uses fixed parameters
but derives local tile mappings from every frame; it is spatially local, not a
time-fixed flat-field correction. This can help uneven/slowly drifting illumination
but can also amplify low-texture noise, so it should be compared against `off` on
representative footage instead of enabled blindly.

Honest number: flow data in float16 only compresses **~1.3×**. The mantissa bits
are near-random. Do not expect compression to rescue an over-large cache; fix that
with downsampling or block size.

### Nyquist

**You cannot measure any frequency above half the frame rate**, and content above
it does not politely vanish — it aliases down and imitates real signal inside your
band. The reference footage is 59.94 fps, so the ceiling is 29.97 Hz and the
spec's default 15–30 Hz band sits *on* that limit. The tool proposes a band from
the loaded video's fps (12–24 Hz at 60 fps), warns above 80% of Nyquist, and
refuses above it. A grasshopper wingbeat near 20 Hz is recoverable at ~3 samples
per cycle — marginal but real. Anything genuinely faster needs a faster camera.

### Defaults tuned to the footage, not to a spec constant

- **Downsample** is derived from resolution to hit a ~1300 px working width, not
  fixed at 0.5. The spec's cache math assumed 1080p; the reference clip is 5.3K
  (7.6× the pixels). A fixed 0.5 would mean ~950 MB per feature and a multi-hour
  pass. Auto gives 0.25 here, 0.5 at 1080p.
- **Morphological opening defaults to OFF.** Opening with radius r erodes with a
  (2r+1)² element, so it deletes any region smaller than that. On the reference
  footage a candidate region is 1–4 blocks, and radius 1 destroyed **100%** of
  them. "Small spatial extent" is part of the signature this tool exists to find;
  opening is hostile to it, and `min_area_blocks` already removes small components
  without shrinking the large ones.
- **ROI tracking uses centroid distance as well as bbox IoU.** A 1-block region
  that moves one block has an IoU of exactly 0 with its own previous position, so
  an IoU-only tracker starts a new ROI every frame and nothing ever meets the
  minimum-duration test.

## Measured performance (reference clip, 5312×2988 @ 59.94 fps, 8.5 min)

| | |
|---|---|
| Working resolution | 1300×731 (downsample 0.245) |
| Block grid | 45×81 |
| Farnebäck throughput | ~4.6 frames/s |
| Full pass | **~110 min** |
| Cache on disk | **~550 MB** |

The standardization validation intentionally enabled full-frame forward/backward
error and texture at block size 4. On a 599-frame (10 s) sample it took 276.8 s
(~2.2 source frames/s, ~28× slower than real time) and wrote ~289 MB. A naïve
510-second projection is about 3.9 hours and 14.7 GB for that one video. That
configuration exists to evaluate the diagnostics; it is not feasible for hundreds
of hours. The six stored replicate boxes cover only 8.27% of the full frame, so a
future production path should compute flow in padded replicate crops and keep
full-frame forward/backward validation as a sampled quality-control pass.

DIS is roughly 1.7× faster than Farnebäck at equal or better quality on subtle
motion, and is worth trying first if the full pass is too slow.

## ⚠ The reference footage has camera motion

Measured on `Videos/Raw/GX010050c2_02_18_26.MP4`:

- **83.5%** of blocks have flow correlating > 0.5 with the whole-frame mean.
- Whole-frame net flow is **0.67×** the mean per-block speed.
- The global motion is concentrated at **0.2–1.2 Hz** (a slow pan or drift, not
  vibration).

That means the flow field is dominated by the camera, and ROI extraction on
speed/coherence lands on **static PVC pipes and shelf edges** rather than animals.
**Turn frame registration on** (`phase` handles a slow pan and is cheap) before
trusting any detection from this clip.

The one piece of good news: only ~5% of the camera-motion power falls in the
12–24 Hz band, so band-power filtering is partially insulated from it. Registration
is still the right fix.

## Scripts

    python scripts/smoke_test.py         # headless pipeline + feature sanity check
    python scripts/gui_smoke_test.py     # drives the whole GUI offscreen
    python scripts/benchmark_storage.py  # HDF5 vs Zarr
    python scripts/render_tabs.py        # render each tab to screenshots/
    .venv\Scripts\python.exe scripts\validate_standardization.py \
      --video Videos\Stabilized\stab_GX010050c2_02_18_26.MP4 \
      --layout replicates.json --duration 10 --block 4

## Adding a feature

Register it in `core/features.py`. Declare `kind="derived"` with a compute
function and it appears automatically as an opt-in histogram in Tabs 2 and 3 —
no UI code to touch. Only add `kind="cached"` if it genuinely cannot be recovered
from `u`, `v`, `speed` and the band-powers.
