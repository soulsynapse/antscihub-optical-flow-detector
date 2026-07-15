# Optical Flow Behavior Detector

Desktop research software for defining animal behavior from dense optical-flow
features. The current workflow is replicate-first: the user identifies the
experimental units, the pipeline processes only those owned pixels, and behavior
definitions are tuned with histogram ranges and AND/OR logic.

This is active research software. The cache format and workflow changed
substantially in configuration version 3; caches made by the earlier full-frame
pipeline should be rebuilt rather than treated as interchangeable.

## Quick start

Use a virtual environment. On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

On macOS or Linux, replace `.\.venv\Scripts\python.exe` with
`.venv/bin/python`.

[FFmpeg](https://ffmpeg.org/download.html) on `PATH` is strongly recommended.
The application uses ordinary CPU FFmpeg to decode once and emit only cropped,
downsampled grayscale replicate pixels. **CUDA and NVDEC are not required.** If
FFmpeg is unavailable, the application falls back to OpenCV full-frame decoding,
which is numerically comparable but much slower on high-resolution footage.

RAFT is a separate, optional GPU optical-flow backend. It requires a CUDA-enabled
PyTorch and torchvision installation and is intentionally not part of the base
requirements. Farnebäck and DIS work on CPU.

## Workflow

1. **Replicates** — open a video and draw or import the non-overlapping source
   boxes that own each animal or isolate. Drawing and fixed-size stamping start
   enabled: drag once to establish a size, then click to place matching boxes.
   Labels, calibration and quiescent-baseline intervals live here.
2. **Preprocessing & Flow** — configure per-replicate preprocessing and the flow
   backend. Test mode processes the first N seconds into a separate scratch
   cache; full mode processes the clip. Completion opens the cache but leaves the
   user on this tab.
3. **Behavior Classification** — define behavior with a tree of AND/OR feature
   ranges, temporal cleanup criteria and per-replicate thresholds; inspect the
   ethogram and export CSV/HDF5 results.

Time is in **seconds** and frequency in **Hz** throughout. Frame indices appear
only where an exact source-frame reference is useful.

Replicate boxes and manual marks are stored beside the video as
`<video>.rois.json` and `<video>.marks.json`. Feature caches are separate and
live under `.cache/`; loading a new video never silently opens a prior cache.

## Current processing model

- Replicate boxes are the processing and ownership boundary. Positive-area
  overlap is rejected; edge-touching is valid.
- FFmpeg decodes the compressed stream once, then crops, area-downsamples,
  converts to grayscale and packs the exact replicate rectangles before pixels
  enter Python.
- Every replicate has its own stateful preprocessor. Normalization,
  registration, denoising and background models are therefore not shared between
  differently illuminated experimental units.
- Dense flow receives temporary reflected support made only from that
  replicate's edge pixels. The support is removed before block reduction, so
  neighboring animals cannot enter through padding.
- Partial blocks at drawn right/bottom edges are retained. Padding improves the
  numerical support of flow; it never expands the biological ownership region.
- Replicate block grids are packed into a sparse Zarr atlas for efficient reads.
  Derivatives, spatial medians, texture percentiles and relative-background
  calculations operate separately within each tile.
- Cache identity includes video content, processing settings, feature settings
  and replicate geometry. Renaming a replicate or changing calibration/baseline
  metadata does not recompute flow; moving, adding or deleting a box does.

The optional **within-replicate mask** is not another compute ROI. It only
suppresses a fixed nuisance inside one or more boxes and does not reduce decoded
area, solver work or cache size. Most projects should leave it unset.

## Standardization without destroying the raw signal

The cache stores the block-mean vector (`u`, `v`) plus the mean per-pixel speed.
It does not store block-mean angles: ordinary averaging across the 0°/360° seam
is a circular-mean error. Angle, net speed and angular coherence can instead be
derived correctly from the cached vector and speed.

Cache-time options are limited to measurements that need the source frames:

- **Forward/backward error** stores a continuous per-block p90 consistency
  residual. It does not store a second complete flow field or reject pixels. It
  is a costly diagnostic and should normally be used on sampled quality-control
  windows, not an entire corpus.
- **Texture strength** stores block means from `cornerMinEigenVal`. The derived
  `texture_percentile` keeps the low-texture cutoff visible and tunable.

Cheap or subjective operations remain analysis-time features: spatial median
variants, physical-unit conversion, and speed relative to a fixed
per-replicate quiescent-baseline p99. When no explicit baseline exists, the
fallback derives one fixed reference for the whole replicate; it does not
renormalize every frame.

CLAHE is opt-in. It is applied independently per replicate, but its local mapping
is still recomputed for every frame. That can help gradual ambient illumination
changes, but it can also turn low-texture intensity noise into apparent motion.
It should be validated against `off` on representative marked footage rather
than made a universal default. Video-wide normalization and per-frame scalar
renormalization are deliberately avoided because they either mix unequal
replicates or destroy temporal amplitude.

The complete rationale, including rejected alternatives, is in
[Current design decisions](docs/decisions.md).

## Storage choice

Zarr was selected after benchmarking it against HDF5 with
`scripts/benchmark_storage.py`. Compressed sizes were similar; HDF5 wrote and
cold-sought faster, while Zarr was 2.3× faster for the repeated ROI time-series
read that drives this application. Time-major chunks also leave interrupted
caches diagnosable. Incomplete caches are not opened as valid results.

Float16 flow data compresses only about 1.3× because its mantissa bits are close
to random. Downsampling, block size and owned ROI area are the meaningful cache
controls; compression cannot rescue an oversized processing plan.

## Measured performance

Reference: local stabilized video `stab_GX010050c2_02_18_26.MP4`, 5312×2988 at
59.94 fps, using its six saved replicate boxes.

| Configuration | 10 s runtime | Relative to real time | 10 s cache |
|---|---:|---:|---:|
| Old full-frame Farnebäck + FB diagnostic | 276.8 s | 27.7× slower | 289 MB |
| ROI-first Farnebäck + FB diagnostic | 33.2 s | 3.3× slower | 25.0 MB |
| ROI-first Farnebäck production | ~14.8 s projected from 5 s | ~1.5× slower | ~20 MB |
| ROI-first DIS production | **6.6 s** | **1.5× faster** | ~18 MB |

The six boxes contain 78,774 owned working pixels per frame, 8.27% of the
source. Private reflected support brings the solver atlas to 166,980 pixels. At
the measured DIS rate, the 510-second source projects to about 5.6 minutes and
roughly 0.9 GB with block size 4, texture and one cached band. One hundred hours
projects to about 66 processing-hours in a single process; safe parallel corpus
throughput still needs to be benchmarked.

The FFmpeg path was checked against the OpenCV path. Farnebäck replicate p95
series correlated 0.990–0.996 and median scale differed by at most 1.7%. The
ROI-first result correlated 0.950–0.990 with the original full-frame replicate
p95 series, with median scale ratios of 0.985–1.015.

Runtime scales with owned area plus the perimeter cost of private synthetic
support. Layouts containing many tiny ant boxes can therefore spend a larger
fraction of work on edge support than the six-box grasshopper layout.

## Frequency and camera-motion constraints

No method can recover frequency above the Nyquist limit. At 59.94 fps the limit
is 29.97 Hz, so the old 15–30 Hz suggestion sits directly on the boundary. The
tool proposes 12–24 Hz at 60 fps, warns above 80% of Nyquist and rejects bands
above Nyquist. Faster behavior needs faster capture.

The corresponding raw reference clip contains strong slow camera motion:

- 83.5% of blocks had flow correlating above 0.5 with the whole-frame mean.
- Whole-frame net flow was 0.67× the mean per-block speed.
- Most global-motion power was at 0.2–1.2 Hz; only about 5% fell in 12–24 Hz.

Registration is therefore still important for unstabilized footage, even though
high-frequency band power is partly insulated from slow drift.

## Documentation

- [Known issues and deferred design decisions](KNOWN_ISSUES.md)
- [Documentation index](docs/README.md)
- [Current design decisions](docs/decisions.md)
- [Prioritized next steps and approaches to avoid](docs/next-steps.md)
- [Archived initial project handoff](docs/archive/initial-project-handoff.md)
- [Archived standardization handoff](docs/archive/standardization-handoff.md)

The archived handoffs explain how the project arrived here, but they are not
current implementation contracts.

## Validation and development

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\smoke_test.py
.\.venv\Scripts\python.exe scripts\benchmark_storage.py
.\.venv\Scripts\python.exe scripts\validate_standardization.py `
  --video Videos\Stabilized\stab_GX010050c2_02_18_26.MP4 `
  --layout replicates.json --duration 10 --block 4
```

The large validation videos are local research data and are not part of the
repository.

To add a feature, register it in `core/features.py`. A `kind="derived"` feature
with a compute function appears without changing the GUI. Add a cached feature
only when it cannot be recovered from `u`, `v`, `speed`, existing band powers or
metadata without revisiting the source frames.
