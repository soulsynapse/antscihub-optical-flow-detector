# Expanded cache: dynamics, identity, and unsupervised behavior mapping

Status: proposed (2026-07-16). This is a forward plan, not live configuration.
See `decisions.md` for the current v3 workflow this extends.

## Why

Two references drive this:

1. **Geng et al. 2023 (ZeChat)** — Cell Reports Methods. An unsupervised
   behavioral profiler in the Berman et al. (MotionMapper) lineage. Per animal,
   per frame it builds a "merged image" (frame-difference silhouette masking a
   direction+velocity-colored dense-flow image), autoencodes it to a latent
   vector -> first 40 PCs, then applies a **Continuous Wavelet Transform**
   (Morlet, 25 scales, 0.38-5 Hz) to each PC time series to inject dynamics,
   then t-SNE -> Gaussian-PDF -> watershed into 80 behavior categories.
2. **color_detector_v2** — the ancestor tool: an HSV `inRange` mask (with
   hue-wraparound) used as a color detector. `../color_detector_v2/core/mask_builder.py`.

The transferable insight is not the deep autoencoder. Our block-reduced
structure-tensor channels (change/appearance/tensor_speed/amplitude-variance)
are already a hand-designed **energetic feature basis** — the thing the CAE is
trained to discover, for the *motional/rhythmic* part of behavior. What we are
missing to run the Berman recipe is (a) the multi-scale time-frequency
representation of those channels, and (b) an identity channel.

## What this buys, and its one hard limit

Block-level channels + wavelet scalograms capture **energetic and rhythmic
dynamics** well and **postural configuration** poorly — block reduction discards
sub-block body shape. So:

- Energetic/rhythmic behaviors (movement bursts, oscillation, struggle,
  wingbeat, tremor): the block-channel + wavelet path gives the ZeChat
  unsupervised map with no network to train. This is the target.
- Postural/configural behaviors: need finer per-animal crops; block caching does
  not substitute.

### Eulerian field vs Lagrangian tracking (the TREX division of labor)

This tool is **Eulerian** — it measures at fixed locations (blocks), no
identity, no correspondence. TREX-class trackers are **Lagrangian** — they
follow individuals, keep identity, extract posture. They fail in opposite
regimes:

| | TREX (Lagrangian) | this cache (Eulerian) |
|---|---|---|
| unit | the individual | the fixed block |
| gives | trajectory, identity, posture | energetic/rhythmic/appearance dynamics per site |
| overlap | breaks (identity swaps) | survives (no segmentation needed) |
| position | its job | cannot (block reduction discards it) |

Do **not** cache a motion centroid or otherwise re-implement tracking. Cache the
overlap-robust field features in lab-frame block coordinates, and let external
tracks *sample* the cache: given a TREX trajectory `(x,y,theta)(t)`, a downstream
reader pools the blocks under a body-centered window per frame -> per-individual
energetic series -> wavelet -> embed. That is ZeChat's architecture with a much
stronger tracker than ZeChat had. Requires block size finer than the animal;
for the ant case the user can set it as fine as needed.

## Proposed cache additions

The caching principle stays "cache what needs the decode; derive what is a
function of cached series."

**Needs the decode -> compute in the single streaming pass:**

- `intensity` — block-mean of the preprocessed frame. Needed for amplitude
  variance; currently only in the tensor sidecar (a second decode).
- Structure-tensor block components `(xx,yy,tt,xy,xt,yt)` (or the derived
  channels change/appearance/tensor_speed) — folds the tensor into the flow
  pass, killing the second decode the explorer currently pays.
  - The one loss: `sigma` (pixel-scale pre-smoothing before block reduction)
    becomes a build-time commitment, since it happens below the block grid and
    cannot be a post-hoc knob. `sigma=2` fixed is expected to be fine.
  - Two "appearance" flavors: `flow_residual(J)` (from cached components, free)
    vs residual-against-cached-flow (needs frames). Prefer the former; it is the
    tensor's own brightness-constancy residual and removes the second decode.
- `color_density` — per-block fraction of pixels inside an HSV range
  (mask_builder logic), block-reduced. An **identity** channel orthogonal to
  motion ("the marked animal is here"). Cheap, streaming-friendly, cannot be
  derived post-hoc. Bounded ratio -> band-less per the detection-channel design;
  the raw matched-pixel count is the heavy-tailed bandable quantity.

**Function of cached series -> derive post-hoc (persist the scalogram like band
power; do not re-decode):**

- **Per-channel Morlet scalogram** — the dynamics layer. Generalizes the current
  fixed band power (band power = scalogram summed over a band). Implemented
  FFT-based (no pywt; scipy.signal.cwt is gone in scipy 1.18).
  - STORAGE WARNING (measured in the POC): a full per-block, per-frame scalogram
    is `n_scales x frames x blocks` = `n_scales` times a raw channel. At 60 fps
    and 20 scales this is ~20x per channel per frame — not cacheable for
    multi-hour footage. The cache-appropriate forms are:
      1. **hop-subsampled** scalogram (one column per hop, like band power) —
         ~`fps*hop` smaller; the STFT window already smears finer detail.
      2. **multiband** — a coarse scalogram of K>2 log-spaced bands (a modest
         extension of today's single band), storing K band powers instead of a
         full scalogram.
      3. **sparse** — full-rate scalogram only inside detected/active blocks.
  - The POC computes full-rate on a sample to show the *signal*, then reports the
    storage math to pick the cached form.

## Phased plan

1. **Log the plan** (this doc + a project memory). — task #1
2. **POC on the stabilized hopper cache** (`scripts/poc_expanded_cache.py`):
   load tensor channels, compute Morlet scalograms per replicate and per-block on
   one tile, measure build time + storage, extrapolate to multi-hour, and show a
   figure where the scalogram exposes rhythmic structure the single 12-24 Hz band
   misses. Decide hop-subsampled vs multiband vs sparse from the measured sizes.
   — task #2
3. **Explorer** for the expanded cache, in the same idiom as the structure-tensor
   explorer (`gui/explorers/scalogram_explorer.py`, launcher
   `scripts/scalogram_explorer.py`): video + channel overlay on the left, the
   custom-painted plot stack on the right — channel sparkline, a `ScalogramPlot`
   whose draggable band rides the FREQUENCY axis ("which rhythm"), a per-block
   band-power `DensityPlot` with a VALUE band ("which blocks"), and a
   "# blocks in band" bar. The Morlet math lives in `core/wavelet.py` (shared
   with the POC). The two stacked bands are the explicit contrast with the
   structure-tensor explorer's single fixed band. Nothing is written to the
   cache; per-block scalograms are derived on region/channel change. — task #3

Later (out of scope until 1-3 validate): fold the winning cached form into
`run_pipeline`; add the color channel with a picker sharing color_detector_v2's
settings; wire external-track sampling for per-individual embedding.
