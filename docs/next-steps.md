# Prioritized next steps

This is the working backlog after the ROI-first and standardization pass. It
records both the useful next experiment and the nearby tempting shortcut that
would produce misleading or hard-to-support behavior.

## 1. Validate normalization across marked replicates and conditions

Run `off`, CLAHE and z-score preprocessing on representative marked windows from
the difficult stabilized video, using its actual replicate boxes. Add footage
with different ambient levels, species, arena textures and animal-to-box size
ratios. Compare within-replicate temporal continuity, between-replicate behavior
separation, false-positive motion and manual-mark agreement.

Success means a setting improves transfer across replicates/videos without
flattening behavior amplitude or raising low-texture false positives.

z-score is the current default because it is boundary-safe and near-neutral for a
small target in a larger box; this validation must confirm it does not flatten
amplitude when a target fills its box (see decisions: normalization). Do not
re-enable CLAHE globally based on one visually improved frame — it has a known
replicate-edge artifact (KNOWN_ISSUES.md). Do not fit one normalization across the
whole video, which mixes unequal replicates.

## 2. Turn the reference clip into an accuracy regression

Use the existing per-video marks and saved boxes to report precision, recall,
false-positive time and bout-boundary error for representative behaviors. Keep a
small reproducible crop recipe rather than committing the large research video.
Record feature-series correlations and scale ratios when decode, preprocessing or
flow code changes.

The current tests establish mathematical and geometry contracts, and the 10-second
benchmarks establish throughput. They do not yet establish biological detection
accuracy.

Do not optimize thresholds against only one replicate or report a visually
plausible overlay as validation.

## 3. Let Test mode choose a representative interval

The current Test mode always starts at frame zero. Add a start time or a selected
preview interval so users can test a window containing known behavior or a known
ambient transition. Preserve enough warm-up for temporal denoising, background
models and spectral windows.

Do not implement arbitrary frame skipping as a substitute. Optical flow,
registration state and band power depend on consecutive frames and well-defined
time spacing.

## 4. Add a resumable corpus runner after accuracy stabilizes

For hundreds of hours, support a manifest of video, ROI sidecar and settings;
skip complete matching caches; identify incomplete work; and bound the number of
parallel workers. Benchmark CPU saturation, memory, disk write rate and total
cache volume before selecting a worker count.

At the measured reference settings, 100 hours is about 66 single-process compute
hours and potentially hundreds of gigabytes of cache. Batch execution therefore
needs storage planning and explicit lean feature presets.

Do not launch one unconstrained process per video. FFmpeg, OpenCV flow and Zarr
writes already use substantial CPU, memory bandwidth and disk bandwidth, so naïve
parallelism can reduce total throughput or exhaust storage.

## 5. Make ordinary FFmpeg installation nearly automatic

First add a startup capability report with the exact executable/version and the
active fallback. Evaluate using the platform wheels from `imageio-ffmpeg` as an
[application-managed CPU executable](https://github.com/imageio/imageio-ffmpeg)
while still allowing an explicit system FFmpeg override. Verify that the bundled
build contains the crop, scale, format, split and vstack filters used by the ROI
stream, and review binary licensing and update policy.

Do not make NVDEC a requirement. It is NVIDIA-only, codec- and
resolution-dependent, and does not accelerate the current reference stream.
Do not transcode research footage solely to unlock NVDEC: transcoding costs time,
can alter subtle motion evidence and creates another provenance problem.

## 6. Benchmark GPU flow only where accuracy needs it

Compare DIS with RAFT on marked fine-motion cases such as antennae or grooming.
Measure end-to-end time, GPU memory, transfer cost and accuracy—not model
inference alone. If RAFT clearly wins, document a separate optional PyTorch/CUDA
[installation profile](https://docs.pytorch.org/get-started/locally/).

Do not replace the dependable CPU backend merely because a GPU is present. GPU
decode and GPU optical flow are separate decisions, and normal OpenCV Python
wheels do not provide a turnkey CUDA build.

## 7. Improve cache compatibility and cleanup UX

Surface cache schema/config version, replicate geometry hash, decoder path and
feature inventory in the cache list. Offer deliberate cleanup for stale test,
incomplete and pre-version-3 caches. A missing array should produce a clear
incompatibility explanation rather than a low-level Zarr `KeyError`.

Do not silently synthesize a missing cached feature or open a geometrically
mismatched cache. Either operation could make an old result look valid.

## 8. Revisit the within-replicate mask only with real demand

If users repeatedly need irregular arenas or fixed exclusions, move the mask
controls under an Advanced section and consider whether the mask should influence
block ownership. Validate edge behavior first.

Do not advertise it as a speed optimization in the current implementation, and
do not remove it solely for conceptual neatness while it still covers real
within-box nuisances.

## 9. Replace the synthetic reflected halo with a neighbor-safe source halo

The private reflected border (decisions: "create support synthetically") is a
mirror, and a mirror is a poor input for a translation estimator — it is the main
reason CLAHE edge contrast becomes phantom edge speed on difficult footage.
Replace it with a small real source halo, sized to the normalization/CLAHE need
(roughly one block, not the full flow-support width), decoded around each box and
discarded after the solve so only the core is cached. On replicate 23, eight real
working pixels already drop the artifact from 861 to ~53 px/s.

The halo must not break replicate isolation. Clip it against neighbors with a
Voronoi rule: it may read any pixel outside every box, including the gap between
two boxes up to the perpendicular bisector, but never crosses into another box's
interior. Fall back to reflected padding only on the strip facing a genuinely
adjacent neighbor. This also lets the CLAHE-tiling question be revisited — test
`tileGridSize=(1,1)` per box against the halo before committing.

Do not size the halo for full Farnebäck support if that reaches into a neighbor;
the isolation contract (`validate_replicates`) stays. See KNOWN_ISSUES.md.

## Approaches deliberately not on the near-term path

- Returning to full-frame flow as the production default. The measured cost is
  disproportionate when experimental units occupy a small fraction of the frame.
- Allowing overlapping replicate ownership without a new semantic model. It
  double-counts pixels and breaks isolate independence.
- Reading real pixels from *another replicate box* as flow padding. This imports a
  nearby ant and breaks isolate independence. Reading true background between boxes,
  Voronoi-clipped so it never enters a neighbor, is the planned halo (§9) and is
  safe; reflected support remains the fallback where boxes are genuinely adjacent.
- Storing thresholded/binary flow or permanently filtered vectors. Those choices
  prevent later inspection and threshold sweeps.
- Running forward/backward diagnostics on every frame of the full corpus. Sampled
  QC provides evidence at a tractable cost.
- Treating atlas coordinates as image coordinates. Every spatial operation must
  continue to respect replicate tile boundaries.
