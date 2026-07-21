# Optical Flow Behavior Detector

Desktop research software for defining animal behavior from dense per-block
motion features. The workflow is replicate-first: the user identifies the
experimental units, the pipeline processes only those owned pixels, and a
behavior is isolated by placing a frequency band and a value band on a
structure-tensor channel.

This is active research software. **The optical-flow cache subsystem has been
removed** — detection now streams from the video and writes nothing to disk.
Documents describing a `.cache/` directory, cached features or a Behavior
Classification tab describe a subsystem that no longer exists; they are retained
only as historical rationale (see [Current design decisions](docs/decisions.md)).

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

## Workflow

The application has two tabs.

1. **Replicates** — open a video and draw or import the non-overlapping source
   boxes that own each animal or isolate. Drawing and fixed-size stamping start
   enabled: drag once to establish a size, then click to place matching boxes.
   Click selects and zooms, dragging repositions, and moving a box that has
   already been processed requires an explicit acknowledgement that its
   measurements are discarded. Labels, calibration and quiescent-baseline
   intervals live here.
2. **Preprocessing (live)** — tune preprocessing directly against the detector.
   Press `Play ▶` and the clip streams forward while the Morlet scalogram,
   per-block densities and detection track fill in behind the frontier. Drag a
   frequency band to pick the rhythm and a value band to isolate the behaving
   blocks; `downsample`, `block_size` and `normalize` are live knobs. The strip
   at the bottom is the whole clip's seeker — click into any examined region to
   verify a detection against the footage. `Process whole video ▶` runs the same
   detector over the entire clip and fills the *same* accumulator, so a committed
   pass and a live one are not two different pictures.

Time is in **seconds** and frequency in **Hz** throughout. Frame indices appear
only where an exact source-frame reference is useful.

Replicate boxes and manual marks are stored beside the video as
`<video>.rois.json` and `<video>.marks.json`.

## Live tensor detection

The detector that isolates a behavior — a per-block "detection / no-detection" —
is built entirely from structure-tensor channels streamed from the video. Nothing
is precomputed and nothing is cached, so the whole loop runs on a bare video.

**Base fields** come out of one decode + tensor pass
(`core/tensor_channels.py`): `intensity`, `change` (`J_tt`, squared frame
differencing), `appearance` (the residual `I_t + ∇I·v` that the tensor's own flow
cannot explain), `texture` (`cornerMinEigenVal`), `tensor_speed`, and the signed
flow components `u`, `v`. Channel selection is a real cost lever — each field's
prerequisites are resolved per pass, so asking for one channel does not pay for
four.

**Derived channels** (`core/channels.py`) are pure functions of those fields and
never re-decode. The velocity-gradient family decomposes `∇v` into its three
2-D invariant parts — `vel_divergence` (trace), `vel_shear` (deviatoric strain
rate) and `vel_vorticity` (antisymmetric) — computed per atlas region so a
spatial derivative never crosses a replicate seam. `∇v` is translation-invariant
by construction, so it measures configural change with no tracker.

The explorer's detection channels are the nonnegative energies (`change`,
`appearance`, `tensor_speed`, `intensity`, `vel_shear`) on a log axis with a
draggable tail band. `vel_divergence` and `vel_vorticity` are signed: linear
axis, no threshold band, diagnostic overlays only.

### The live axis is the whole video

`core/live_track.py` holds a clip-length accumulator of `count` / `clump` /
`gate` plus a per-frame stamp of the settings that produced it. Consequences
worth knowing:

- **Coverage is painted honestly, in three states**: unexamined (bare trough),
  examined-and-quiet (baseline rule lit), and examined-under-other-settings
  (desaturated). "Nobody looked" is never rendered as "nothing happened".
- **Whole-clip per-block band power is retained** (~45 MB at 30k frames × 377
  blocks, capped at 1 GiB), so re-tuning the value band, count band or detection
  window is instant and never grays out the clip. Changing the channel,
  frequency band, grid or scale does invalidate it.
- **Cone-of-influence contaminated frames are not written.** `coi_trim` derives
  the trim from the band's lowest frequency; a later overlapping window, for
  which those frames are interior, supplies them. The detector therefore never
  produces frontier-transient numbers, and no detector-side mask is needed.
- Detection runs on the worker thread, not the display path: the display drops
  windows while it is busy, which is right for a display and would put holes in
  an accumulator.

Per-frame series are cheap at whole-clip length; the per-block `(F, T, B)` cube
is not, so densities and the scalogram cube stay windowed around the cursor and
are drawn padded onto the shared axis with the uncovered tail hatched.

**Channel selection is a live-performance knob, not only a cost knob.** On
`GX010047c2` (6 tiles, block 64, scale 1.0), `intensity` + `change` runs ~80 fps
(3.3× realtime), but adding `appearance` drops it to ~18 fps (0.75× — behind
playback), because the flow solve is ~24% of the pass.

**Validation status.** The wingbeat band is validated: on a 152-bout
hand-verified flying corpus (`rep3_intermittent_crop`), a [13.66–25 Hz] band
separates flying near-perfectly and is frequency-specific, and `butter`+`filtfilt`
matches Morlet. That contrast is *saturated* (flight is gross whole-crop motion),
so it validates the band but does not rank channels finely. The
velocity-gradient channels and the occupancy claim (present-but-still vs
animal-absent) are **not** validated — both need a postural/still corpus that
does not exist yet. See [next steps](docs/next-steps.md).

## Headless batch

Tuning happens in the explorer; running happens on nodes with no display.

```powershell
# one video
.\.venv\Scripts\python.exe -m cli.detect VIDEO --params tuned.json --out results\GX0100.json

# a shard of a file list; launch N copies differing only in --shard i/N
.\.venv\Scripts\python.exe -m cli.run "footage\*.MP4" --params tuned.json --shard 3/8 --out-dir results\
```

`--params` is `ScalogramExplorer.detection_params()` as JSON minus
`region_index`. Boxes come from each video's `.rois.json` sidecar. Partitioning
is a stride over the sorted list, so no shard consults any other and a preempted
shard can simply be relaunched. Finished videos are skipped on re-run only when
the recorded params and the resolved scale and block size match — a retuned band
recomputes rather than returning yesterday's detections.

Exit codes are the job runner's contract: **0** success, **1** at least one video
failed, **2** bad usage.

## Current processing model

- Replicate boxes are the processing and ownership boundary. Positive-area
  overlap is rejected; edge-touching is valid.
- FFmpeg decodes the compressed stream once, then crops, area-downsamples,
  converts to grayscale and packs the exact replicate rectangles before pixels
  enter Python.
- Every replicate has its own stateful preprocessor. Normalization,
  registration and denoising are therefore not shared between differently
  illuminated experimental units. (Background-model subtraction has stubs but is
  forced off on the tensor path: a fitted background is an asset the pipeline
  cannot reconstruct from the frames.)
- Dense flow receives temporary reflected support made only from that
  replicate's edge pixels. The support is removed before block reduction, so
  neighboring animals cannot enter through padding.
- Partial blocks at drawn right/bottom edges are retained and weighted by their
  real pixel area, so a row of one-pixel slivers cannot masquerade as a clump.
- Replicate block grids are packed into a sparse atlas so derivatives and
  spatial statistics operate separately within each tile.

**`downsample` and `block_size` are separate levers and stay separate.**
`downsample` shortens every per-pixel stage (buys **compute**) and carries the
warning that it may decide what is detectable; `block_size` only coarsens the
grid (buys **storage**) and leaves the per-pixel math untouched. They are never
fused into one "quality" slider. Downsampling is opt-in, default scale 1.0: a
pipeline that silently downsamples has already decided which behaviors are
detectable, and coarser resolution must be *demonstrated* per behavior and
species, never assumed.

Changing `block_size` rescales the detector's count band
(`core.detection.rescale_count_band`), because `inband_count` is a raw block
count and a region holds ~29 blocks at block 64 against ~377 at block 16 — the
same band otherwise means something ~13× different and the detector would
silently stop firing.

The optional **within-replicate mask** is not another compute ROI. It only
suppresses a fixed nuisance inside one or more boxes and does not reduce decoded
area or solver work. Most projects should leave it unset.

## Normalization without destroying the raw signal

CLAHE is opt-in and `zscore` is the default. CLAHE is applied independently per
replicate, but its local mapping is still recomputed for every frame. That can
help gradual ambient illumination changes, and it can also turn low-texture
intensity noise into apparent motion — on the reference clip a replicate peaked
at 861 px/s with CLAHE against 48 px/s without, where the actual translation was
~0.08 working pixels (see [Known issues](KNOWN_ISSUES.md) §1). It should be
validated against `off` on representative marked footage rather than made a
default.

Video-wide normalization and per-frame scalar renormalization are deliberately
avoided because they either mix unequal replicates or destroy temporal
amplitude. The quiescent-baseline interval on the Replicates tab is still
collected, but its only consumer — the flow cache's baseline-relative speed
feature — went with the cache; it is recorded metadata today, not an active
normalization.

Block-mean **angles** are never averaged: ordinary averaging across the 0°/360°
seam is a circular-mean error. Angle, net speed and angular coherence are derived
from the signed `(u, v)` components instead.

The complete rationale, including rejected alternatives, is in
[Current design decisions](docs/decisions.md).

## Measured performance

At scale 1.0 a pass is **math-bound, not decode-bound**: `tensor_blur` (~52%) and
`tensor_products` (~28%) dominate, the producer thread already overlaps decode,
and making decode 25× cheaper (per-replicate pre-transcoded clips) is **1.06×
end to end**. Do not prioritize decode work off an isolated decode number.

Headless throughput on `GX010047c2_02_17_26.MP4` — 7 replicates, 5312×2988,
23.976 fps, scale 1.0, `change` only (`scripts/bench_worker_scaling.py`):

| workers | extract fps | speedup | vs. real time | slowest worker |
|---|---:|---:|---:|---:|
| 1 | 36.6 | 1.00× | 1.53× | 16.4 s |
| 2 | 60.2 | 1.64× | 2.51× | 19.9 s |
| 4 | 102.7 | 2.81× | 4.28× | 23.4 s |
| 8 | **131.6** | **3.60×** | **5.49×** | 36.5 s |
| 16 | 130.9 | 3.58× | 5.46× | 73.3 s |

**The useful per-node figure is 8 workers** on a 32-logical-core box. N=16
delivers the same aggregate throughput with every worker taking twice as long.
3.6× from 32 cores is not what compute-bound work looks like; the plateau's shape
suggests a memory-bandwidth ceiling, which is a hypothesis, not a measured
attribution. At 5.4× realtime per node, 3000 h of footage is ~555 node-hours —
roughly 23 nodes for a 24 h turnaround.

Runtime scales with owned area plus the perimeter cost of private synthetic
support, so layouts of many tiny boxes spend a larger fraction of work on edge
support than a six-box grasshopper layout does.

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

- [Measurements and hard-won conclusions](FINDINGS.md) — load this before
  optimizing anything or re-running a sweep. Several conclusions are
  counter-intuitive enough that they were reached wrongly at least once.
- [Working plan](todo.md) — open items, standing decisions, remaining batches.
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
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest tests\test_detection.py -v
.\.venv\Scripts\python.exe scripts\bench_worker_scaling.py
.\.venv\Scripts\python.exe scripts\channel_lab.py
```

`scripts/channel_lab.py` (with `scripts/run_lab.py`) is the offline channel
comparison: it scores candidate channels against a marked corpus on the tail
statistic and shares its registry with `core/channels.py`.

The large validation videos are local research data and are not part of the
repository.

To add a channel, register it in `core/channels.py`:

```python
@channel("my_channel", needs=("u", "v"))
def _my_channel(fields, meta):
    ...  # -> (T, ny, nx)
```

`needs` names the base fields the extraction pass must produce; only the declared
fields are loaded. A channel is a pure function of already-computed fields — if
it cannot be recovered without revisiting the source frames, it belongs in
`core/tensor_channels.py` as a base field instead, and that costs a real pass.
