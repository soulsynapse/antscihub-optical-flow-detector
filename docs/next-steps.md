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

Do not enable CLAHE globally based on one visually improved frame. Do not fit one
normalization across the whole video, and do not normalize every frame to a fixed
scalar distribution. Those approaches either mix unequal replicates or remove
the temporal contrast the classifier needs.

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

## Approaches deliberately not on the near-term path

- Returning to full-frame flow as the production default. The measured cost is
  disproportionate when experimental units occupy a small fraction of the frame.
- Allowing overlapping replicate ownership without a new semantic model. It
  double-counts pixels and breaks isolate independence.
- Reading real pixels outside a replicate as flow padding. This can import a
  nearby ant; private reflection provides support without leakage.
- Storing thresholded/binary flow or permanently filtered vectors. Those choices
  prevent later inspection and threshold sweeps.
- Running forward/backward diagnostics on every frame of the full corpus. Sampled
  QC provides evidence at a tractable cost.
- Treating atlas coordinates as image coordinates. Every spatial operation must
  continue to respect replicate tile boundaries.
