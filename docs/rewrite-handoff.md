# Rewrite handoff: replicate-native behavior detector

Status: proposed clean rewrite based on the functionality and measurements in
the current application as of 2026-07-21.

This is an implementation handoff, not a migration plan. The rewrite does not
need to read the current ROI, tuning, track, marks, manifest, or replicate-home
formats. Old results do not constrain the new schemas. The current program
remains the behavioral and numerical reference while the replacement is built.

The replacement should be built in a fresh repository whose implementation
environment cannot read this checkout. This document and an intentionally small
contract bundle are the transfer boundary. The old source is an external
black-box oracle for validation, not a library of code or patterns to port.

## 1. Objective

Rebuild the application around one decisive boundary:

> Splitting a source recording into replicates is an ingest operation. Detection
> opens and processes exactly one isolated replicate clip as though that clip
> were the only video in the project.

The analyzer must never receive the original multi-replicate source, a list of
replicate rectangles, or an atlas of several replicate tiles. It must not offer a
runtime choice between source pixels and replicate clips. Its input is one
replicate package containing one grayscale clip, and its processing geometry is
the full clip.

This is primarily a performance and correctness requirement, not a UI
reorganization. The current GUI can select one replicate while the extraction
plan still contains every replicate tile. It therefore filters the result after
paying the multi-replicate work. The rewrite must make that state impossible.

The other governing rule is:

> There is one headless scientific engine. Interactive preview, live playback,
> whole-clip processing, and batch execution are different schedules and output
> sinks for that engine, never alternate implementations of it.

## 2. The performance result that governs the design

The concrete reference case is:

- Source: `stab_GX010050c2_02_18_26.MP4`.
- The source layout contains eight boxes, replicate ids 20 through 27.
- Replicate 27 is source box `(3698, 113, 4109, 541)`, producing a 411 x 428
  clip with 30,579 frames.
- Test settings: replicate 27, downsample 1.0, block size 4, `tensor_speed`.
- Processing through the multi-replicate source/GUI path is approximately 1.5x
  realtime.
- Opening replicate 27's grayscale footage directly and applying the same scale,
  block, and channel runs at more than 20x realtime.

The roughly 13x observed ratio is much too large to dismiss as paint overhead.
The current source-driven plan builds geometry for all boxes and extracts all
replicate tiles even when the detector is scoped to one selected region. The
isolated-clip run has one decoder, one rectangle, one block grid, and no atlas.
It exposes the throughput of the engine when it is asked to do only the work the
user selected.

Treat this as a release-gating benchmark. The rewrite is not successful if it
has a cleaner module layout but gives this speed back.

### Performance acceptance test

Keep a repeatable benchmark command for both the headless engine and a
GUI-launched pass:

```powershell
detector benchmark `
  --package <rep27-package> `
  --channel tensor_speed `
  --downsample 1.0 `
  --block-size 4 `
  --frames 30579
```

Record decoded frames per second, tensor frames per second, band-power frames
per second, end-to-end frames per second, peak resident memory, and output
coverage. Do not report only decoder speed.

Required properties:

1. The engine plan reports exactly one input clip and one full-frame grid.
2. Adding other replicate packages beside it does not change its plan, runtime,
   memory, or numerical result.
3. A GUI-launched whole-clip job sustains at least 90% of the same headless job's
   throughput when plotting is throttled normally.
4. Plot visibility, selected plot, window length, and playback rate do not alter
   engine throughput beyond a small bounded IPC/display cost.
5. Processing N replicates means N independent single-clip jobs. Parallelism is
   chosen by the batch scheduler, never created inside one tensor pass.

The current >20x observation is a target to preserve and verify, not yet a
portable guarantee across machines or codecs.

## 3. Product boundary: two applications, one package format

The rewrite should expose two clearly separate workflows. They may ship in one
installer and may link to each other, but they must not share live process state.

### 3.1 Repository and clean-room boundary

Create a new repository rather than a branch or new top-level package in this
one. The rewrite repository must not have this repository as a submodule,
dependency, adjacent readable checkout, editable install, copied history, or
search path. Do not copy modules and refactor them after arrival; that would
preserve the assumptions the rewrite is intended to remove.

The rewrite implementer receives only:

- This handoff.
- The new replicate-package schema once approved.
- Small redistributable video fixtures designed around named contracts.
- Expected numerical arrays/summaries produced in advance.
- A benchmark record for the rep27 case, including machine and command details.
- A list of user-visible acceptance tests.

The implementer does not receive:

- The old Python source.
- Git history or archived plans.
- Current GUI object structure.
- Current sidecar schemas beyond what this handoff says is intentionally being
  discarded.
- Existing tests whose fixtures mock implementation details rather than public
  behavior.

Validation should be role-separated. A validator that can access both programs
runs the same contract inputs through each and reports differences. The rewrite
implementer gets the input, expected output, tolerance, and failure description,
not the old code that produced it. When old behavior is scientifically wrong or
architecturally coupled, update the written contract explicitly instead of
asking the new implementation to match it.

This is not a legal clean-room exercise; it is an engineering discipline to
prevent accidental transplantation of historical structure.

### 3.2 Replicate Splitter

The splitter opens the original recording and exists only to define experimental
units and build isolated inputs.

It retains the useful parts of the current Replicates tab:

- Open and scrub the source video.
- Draw non-overlapping boxes.
- After the first box, click to stamp additional boxes at the selected box's
  fixed size.
- Select and zoom to a box; right-click returns to the overview.
- Drag to reposition an unbuilt box.
- Rename, delete, clear, save, and load the current layout while preparing it.
- Assign stable replicate ids and display colors.
- Optionally record source pixels/mm, body length in mm, and a quiescent interval.
- Calibrate by drawing an animal-length line and a known-length ruler line.
- Choose a clip encoding profile and see estimated/actual size.
- Build all replicate clips with progress, cancellation, and clear per-replicate
  failures.

The build output is one immutable package per replicate. Building packages is a
commit boundary. Moving a box afterward does not rewrite or reinterpret the old
package; it creates a new package identity on the next build. The UI can retain
or delete the old package explicitly, but the analyzer never needs “retired
geometry” logic.

The splitter must crop at exact integer source coordinates. It records the
source rectangle and source identity for provenance, but the resulting package
is self-contained and analyzable when the original source is unavailable.

### 3.3 Replicate Analyzer

The analyzer opens exactly one replicate package. It may also open a standalone
grayscale video by creating a minimal package whose replicate is the entire
frame.

It does not contain:

- A Replicates tab.
- Multiple editable rectangles.
- A replicate selector or region index.
- Sparse atlas packing or atlas separators.
- A `Use ROI clips` checkbox.
- Source-versus-clip fallback.
- Geometry hashes derived from a current source-side ROI list.
- Cross-replicate track, tuning, or mark handover.

For the analyzer, `(0, 0, width, height)` is the only owned rectangle. A block
grid is local to that image. No spatial derivative can cross into another
replicate because another replicate is not present.

The splitter can offer `Open in Analyzer` for a completed package, but this
launches a separate analyzer process. Closing or editing the splitter cannot
restart, invalidate, slow, or otherwise affect an analyzer job.

## 4. Replicate package

A package is a directory, not merely a pathname remembered by the splitter.
Suggested shape:

```text
stab_GX010050c2_02_18_26_rep27/
  replicate.toml
  video.mkv
  analysis/
    tuning.json
    track.npz
    detections.json
  logs/
```

`replicate.toml` is authoritative package metadata. JSON is also acceptable;
the important part is one versioned schema and atomic writes. It should include:

```text
schema_version
package_id                    # UUID, never inferred from label
replicate_id
label
created_utc

video_file                    # package-relative
width
height
pixel_format
frame_count_decoded
fps_num
fps_den
duration derived from the rational rate

encoding_profile              # lossless/high/standard, exact codec arguments
clip_sha256

source_name                   # provenance only
source_quick_signature
source_sha256 if paid for during build
source_box_xyxy

source_pixels_per_mm optional
body_length_mm optional
quiescent_interval_s optional
```

The analyzer trusts package dimensions and timing only after checking them
against the clip. A mismatch is an invalid package, not permission to infer new
metadata silently. Use the manifest's exact rational rate for all frame/time
conversion even when a decoder exposes a rounded float.

The new system needs no compatibility reader for the existing `.rois.json`,
`.pretranscode.json`, per-replicate home, tuning, track, or marks schemas.

### Encoding policy

The current profiles are useful starting points:

- High: H.264 CRF 12.
- Standard: H.264 CRF 18.
- Lossless: FFV1.

The package must make pixel provenance visible because lossy clips are different
measurements from live crops. In the rewrite there is no runtime ambiguity—the
package clip is always the measurement source—but the encoding profile remains
part of every result's provenance.

Prefer a decoder-friendly grayscale representation. Benchmark H.264 grayscale,
FFV1 gray8/gray16 where appropriate, and an intra-only profile against the real
tensor pass. Select the default by end-to-end throughput, storage, and signal
agreement, not isolated decode speed. Do not transcode again during analysis.

## 5. One headless engine

The engine is a GUI-free Python package with no import of PyQt. Its public
request should be an immutable value:

```python
@dataclass(frozen=True)
class RunSpec:
    package_id: UUID
    clip_sha256: str
    fps_num: int
    fps_den: int
    frame_range: FrameRange
    preprocess: PreprocessSpec
    grid: GridSpec
    channel: ChannelSpec
    detector: DetectorSpec
    pipeline_version: str
```

The central API should be equivalent to:

```python
engine.run(
    package: ReplicatePackage,
    spec: RunSpec,
    schedule: Iterable[Segment],
    sink: ResultSink,
    cancellation: CancellationToken,
) -> RunSummary
```

Every execution mode calls this API:

- A preview supplies a short segment and an in-memory sink.
- Live play supplies forward segments and a rolling-window plus track sink.
- Process-whole-video supplies a planned segment sequence and a persistent
  track sink.
- The CLI supplies one or more packages and file sinks.
- Tests use collecting or deliberately failing sinks.

There must be no GUI implementation of extraction, preprocessing, channel
selection, Morlet power, detection, coverage, or provenance.

### 5.1 Fixed stage graph

```text
ReplicatePackage
  -> grayscale decoder
  -> per-replicate preprocessing
  -> tensor base fields
  -> derived channel
  -> per-block temporal band power
  -> value-band occupancy
  -> detection-window mean
  -> count-band gate
  -> largest spatial clump
  -> track/result sink
```

The graph is single-replicate throughout. Arrays do not carry a region axis.
Coordinates are ordinary clip coordinates, not atlas coordinates.

Storage policy is orthogonal to the graph. An array may be held in RAM, spooled
to a temporary memory map, or persisted as an artifact without changing any
stage or reopening the video through a different path. Crossing a memory budget
may change placement and chunk size; it may not change semantics.

### 5.2 Base fields and channels to retain

The current tensor pass produces these base fields as needed:

- `intensity`: mean block intensity.
- `change`: block mean `J_tt`, squared temporal intensity difference.
- `appearance`: residual energy `I_t + grad(I) dot v` using the tensor's own
  solved velocity.
- `texture`: minimum-eigenvalue spatial structure diagnostic.
- `tensor_speed`: magnitude of tensor-derived flow in pixels/second.
- `u`, `v`: signed tensor-derived velocity components.

Retain the visible channels:

- Change energy `J_tt`.
- Appearance energy.
- Tensor speed.
- Intensity.
- Shear strain rate, derived from `u,v` and usable for detection.
- Signed divergence and signed vorticity as diagnostic plots without tail
  detection bands.

Channel prerequisites must remain declarative. A selected `change` pass must not
compute flow components. Selecting `appearance`, `tensor_speed`, or a velocity
gradient pays for the tensor flow solve. “All channels” remains an explicit,
off-by-default comparison mode because it is materially more expensive.

### 5.3 Preprocessing to retain

- Downsample in `(0, 1]`, default 1.0.
- Block size independent from downsample, with an automatic option that keeps a
  stable source-pixel footprint.
- Normalization choices `off`, per-frame z-score, and CLAHE.
- Per-frame/per-replicate preprocessing only; there is never cross-replicate
  normalization.
- Temporal denoise may be used only when its state can be reproduced honestly.
  Arbitrary random-access previews must not pretend to reproduce a stateful
  filter initialized at frame zero.
- Registration and background subtraction remain off unless the rewrite adds
  explicit fitted assets to the package. Do not expose inert configuration.

Keep downsample and block size as separate scientific levers. Downsample changes
the per-pixel evidence and compute cost. Block size changes spatial aggregation,
memory, and the denominator of the detection count. When block size changes,
convert an existing count band using actual old/new block counts or warn that it
cannot be converted; never leave the same displayed number with a new meaning.

### 5.4 Detection semantics to retain

For one selected nonnegative channel:

1. Compute per-block Morlet power over a draggable frequency band.
2. Count blocks whose band power lies in the selected value band.
3. Average that count over a trailing or centered detection window `D`.
4. Gate where the windowed count lies in the selected count band.
5. Compute the largest 8-connected in-band block clump per frame.

Use the same pure functions for a displayed window and a whole-clip run. Retain
band power when affordable so value band, count band, and detection-window
changes are instant. A channel or frequency-band change requires a new channel
or spectral pass unless a correctly keyed upstream artifact already exists.

Frequency is in Hz and time is in seconds throughout. Cap the draggable Morlet
bank below Nyquist structurally; do not rely only on a warning.

Cone-of-influence-contaminated edge frames must remain uncommitted until another
overlapping segment supplies them as interior frames.

## 6. Analyzer user experience

The analyzer should preserve the successful interaction model while removing
multi-replicate concepts.

### 6.1 Open and inspect

- Open a replicate package or standalone grayscale clip.
- Show package label, source provenance, dimensions, exact fps, duration,
  encoding profile, and calibration if present.
- Display and scrub the full isolated clip.
- Space plays/pauses; arrows step a frame; Shift+arrows step approximately one
  second; Home/End jump to clip ends.
- Display time in seconds, with exact frame numbers available in tooltips or
  diagnostics.

### 6.2 Live tuning strip

Retain:

- Window start and window length.
- Downsample and its measured decision helper.
- Block size, including Auto.
- Normalization.
- Selected-channel-only versus All channels.
- Reset of preprocessing and detector bands while retaining the selected
  channel.
- Play/stop live processing.
- Process-video action and process-plan settings.
- Clear progress, achieved fps, realtime multiplier, cancellation state, and
  failure messages.

Remove:

- `Inherit` from another tab. The analyzer's own playhead supplies the start.
- `Use ROI clips`. The package clip is always used.
- Replicate selector, pooled regions, boxes belonging to other replicates, and
  any region-index translation.

### 6.3 Explorer

Retain the current useful views and gestures:

- Video with the selected per-block overlay and a visible `DETECTED` badge.
- Morlet scalogram with draggable frequency band.
- Per-block density heatmaps with draggable value bands for nonnegative
  channels.
- Signed linear diagnostic views for divergence and vorticity.
- Windowed in-band block-count plot with draggable count band.
- Largest-clump and detection readouts.
- Collapsible plots that do not render while collapsed.
- Sticky but resettable plot ranges.
- Scrubbing and playback synchronized with the whole-clip navigator.
- Backpressure: if a plot transform is still building, skip display refreshes
  rather than queueing stale work indefinitely.

### 6.4 Whole-clip navigator and accumulated track

Retain the three honest coverage states:

- Unexamined.
- Examined and quiet under current settings.
- Examined under other settings and therefore stale.

Retain click-to-seek, detected intervals, current/stale styling, save detections,
and instant re-derivation after cheap threshold changes.

Retain process schedules:

- Continuous from frame zero.
- From the current playhead to the end.
- Binary-split sampling with progressively distributed coverage.
- Uniform chunk sampling.
- Fill gaps/stale coverage only.

Sampling modes must state plainly that unprocessed footage is unknown, not
negative. Coverage is measured against the verified decodable frame count.

The rewrite may persist tuning, tracks, and detections for the new package, but
it does not need to import existing files. Persistence must be atomic and keyed
by `RunSpec` identity.

## 7. GUI/process isolation

Do not run scientific jobs as `QThread` subclasses owned by transient widgets.
The safest design is for the GUI to launch the exact headless executable as a
worker subprocess.

Suggested control protocol:

```text
GUI -> worker: start RunSpec + segment plan
GUI -> worker: cancel job_id
worker -> GUI: accepted job_id + resolved plan
worker -> GUI: progress counters and timing
worker -> GUI: throttled display window snapshots
worker -> GUI: track deltas with RunSpec hash
worker -> GUI: completed RunSummary or structured failure
```

Use newline-delimited JSON for control messages initially. Transfer large arrays
through shared memory or temporary `.npy`/memory-map artifacts identified by the
message; do not serialize multi-megabyte arrays through Qt signals.

The worker owns its decoder and numerical threads. The GUI owns only display
frames and presentation state. Plot refresh should be capped and lossy: the
newest snapshot replaces an unpainted older snapshot. Scientific output is
lossless and uses a separate sink, so dropped display updates can never create
holes in the detection track.

On cancellation, the worker stops at a defined frame boundary, flushes only
completed output, emits its final coverage, and exits. On window close, the GUI
cancels and joins worker processes before destroying its model. A late message
with an old `job_id` or `RunSpec` hash is ignored.

## 8. Headless and batch behavior

The CLI should be the primary interface and the GUI's backend, not a secondary
port.

Suggested commands:

```powershell
# Build packages from a source and a new splitter layout.
detector split SOURCE.MP4 --layout layout.json --out packages\

# Tune/smoke-test one isolated package.
detector run PACKAGE --params tuned.json --start 120s --duration 30s

# Process one whole package.
detector run PACKAGE --params tuned.json --out results\rep27.json

# Process a corpus of package directories with bounded parallelism.
detector batch "packages\*" --params tuned.json --workers 8 --out results\
```

Retain:

- Explicit channel, downsample, block-size, start, and frame/duration overrides.
- Exit code 0 for success, 1 when any input failed, and 2 for bad usage.
- Per-package failure isolation.
- Deterministic sorting and sharding/worker assignment.
- Resume only when the complete `RunSpec` and input identity match.
- Atomic summary writes.
- Optional per-frame series and opt-in band-power output.
- Provenance, timing, verified coverage, warnings, and detected intervals in the
  summary.

Simplify batch execution by treating packages as independent jobs. There is no
shared source manifest to resolve during detection and no multi-region result.
One package produces one replicate result.

## 9. Identity, provenance, and artifacts

Every scientific artifact carries or is keyed by:

```text
package_id
clip_sha256
exact rational fps
decoded frame count
encoding profile
preprocessing settings
downsample
block size and resolved grid
channel name and channel implementation version
frequency band
value band
count band and its block-count denominator
detection window and centered/trailing mode
pipeline/software version
coverage intervals
```

Separate four kinds of state:

1. Package facts: immutable clip identity, timing, geometry, and provenance.
2. Scientific tuning: settings that define a measurement.
3. Session presentation: selected plot, collapsed state, playhead, playback rate.
4. Derived artifacts: channel arrays, band power, track, and detections.

Presentation state must never participate in a scientific cache key. Scientific
settings must never be inferred from what a widget happens to display.

Derived artifacts should be immutable by identity. A retune produces a new
identity or a new cheap derivation from a compatible upstream artifact. Do not
overwrite an old artifact and rely on a `stale` flag to explain mixed contents.

## 10. Testing strategy

### 10.1 Numerical contracts

- Window and whole-clip results agree on their uncontaminated overlap.
- GUI-launched and direct CLI jobs produce identical arrays and summaries.
- RAM and memory-mapped execution produce identical results.
- Chunk boundaries do not change values outside the defined cone of influence.
- Derived-channel prerequisite planning computes exactly the needed base fields.
- Block-band re-denomination uses actual grid counts.
- Signed channels never acquire tail-detection bands accidentally.

### 10.2 Video/time contracts

Build small committed fixtures for:

- Odd crop coordinates.
- Rational rates such as `60000/1001` and `24000/1001`.
- A container frame count larger than its decodable count.
- A deliberately truncated file.
- Partial right/bottom blocks.
- Gray8 and gray16 inputs.
- A lossy and a lossless package generated from the same source rectangle.

Time-to-frame conversion always uses integer arithmetic over the rational rate
where possible. A requested full pass that decodes fewer than the verified count
fails unless truncation is explicitly accepted.

### 10.3 Lifecycle contracts

- No worker process or thread remains alive after each test.
- Cancel during decode, tensor solve, spectral transform, and artifact write.
- Close the analyzer while each job phase is active.
- Restart rapidly and prove old job messages cannot change the new run.
- Kill a worker and prove the GUI reports failure without treating partial
  coverage as quiet footage.

### 10.4 Performance contracts

- Pin the rep27 benchmark above.
- Benchmark each visible channel separately and All channels.
- Benchmark GUI visible, GUI minimized, and direct CLI.
- Fail or prominently flag a regression that makes GUI-launched throughput less
  than 90% of direct headless throughput.
- Assert the planned pixel count is independent of unrelated packages in the
  corpus.
- Track peak memory as well as fps so a speedup cannot silently buy unbounded
  allocation.

### 10.5 Scientific validation

Keep scientific accuracy separate from software equivalence. Use a small new
marked corpus to report precision, recall, false-positive duration, and bout-edge
error for known behavior. The current flying result supports the wingbeat band,
but it is a saturated contrast and does not validate every channel or species.

## 11. Implementation sequence

### Phase 0: freeze contracts

- Freeze this repository and record its exact revision and environment.
- Record the rep27 commands, settings, hardware, fps, timings, and current output.
- Export several short numerical fixtures and expected outputs from the current
  headless engine into a standalone contract bundle.
- Review the bundle to ensure it contains no old source or implementation-shaped
  test doubles.
- Create the fresh rewrite repository in an environment that cannot read this
  checkout.
- Write the package schema and `RunSpec` in the new repository before engine
  implementation.

### Phase 1: package builder

- Build a headless exact-crop splitter.
- Verify rational fps and decodable frame count.
- Emit one self-contained package per replicate.
- Add the Splitter GUI only after the command is correct.

### Phase 2: single-clip headless engine

- Port the current generator-based tensor extraction without atlas support.
- Port channel prerequisites and derived channels.
- Port shared detection math and process-plan scheduling.
- Establish >20x-class rep27 throughput before adding a GUI.

### Phase 3: artifacts and resumability

- Implement `RunSpec` hashing, atomic result writes, coverage, and optional band
  power persistence.
- Add RAM/memory-map policy behind the same stage interfaces.
- Implement single-package CLI and bounded batch execution.

### Phase 4: analyzer GUI

- Implement the subprocess protocol.
- Add playback, tuning controls, explorer plots, and navigator as clients.
- Enforce the GUI/headless throughput ratio continuously.

### Phase 5: scientific and operational validation

- Run route-equivalence, lifecycle, failure-injection, and performance suites.
- Validate on a new marked corpus.
- Only then choose defaults beyond the conservative current ones.

Do not begin with the complete GUI and “wire the engine in later.” The headless
engine and package contract are the product core; the GUI is one client.

## 12. Deliberate simplifications from the current application

The rewrite explicitly does not carry forward:

- Legacy sidecar or cache compatibility.
- Automatic adoption of old marks, tuning, tracks, or replicate homes.
- Multi-replicate atlases in the analyzer.
- Source-video fallback during replicate analysis.
- Runtime source/clip selection.
- Region-index-based storage.
- Retired-geometry generations inside the analyzer.
- Qt-owned scientific worker threads.
- Dead configuration for consumers that no longer exist.
- Historical flow-cache and Behavior-tab concepts.

Manual annotation and saved detections may be rebuilt against the new package
identity, but they start empty. A source box change creates a new package rather
than asking old annotations to survive a changed subject/geometry claim.

## 13. Current limitations that remain scientifically open

Do not accidentally claim these are solved by the rewrite:

- CLAHE can create strong boundary artifacts; z-score remains the conservative
  default.
- Synthetic reflected support is not real image context. With isolated clips,
  real outside-box context no longer exists unless the splitter deliberately
  packages a private halo. Any halo design must define which pixels belong to
  the replicate and discard halo results from output.
- Tensor-derived velocity-gradient channels are not broadly validated on marked
  fine-posture footage.
- Present-but-still versus animal-absent occupancy needs an appropriate corpus.
- Downsampling sensitivity is behavior- and species-specific. It cannot be
  chosen globally from compute cost alone.
- The current supervised-signature idea remains future work and should be added
  only after the single-channel engine is stable and validated.

## 14. Definition of done

The rewrite is ready to replace the current application when all of the
following are true:

- A source video can be split into independently portable replicate packages.
- Each package opens and analyzes with no original source or neighboring
  replicate present.
- Preview, live, whole-clip, GUI-launched, and CLI runs all use the same engine.
- The rep27 scale-1/block-4/tensor-speed benchmark preserves the isolated-clip
  speed class and GUI launch remains within 90% of direct headless throughput.
- Adding neighboring packages cannot affect one package's numerical result or
  runtime plan.
- All output carries exact timing, input, settings, software, and coverage
  identity.
- Cancellation, shutdown, truncation, and stale-result tests pass without leaked
  workers or silently unexamined footage.
- The analyzer retains the present tuning, visualization, navigation, sampling,
  and export functionality without reintroducing multi-replicate state.

If a future feature appears to require the analyzer to know about multiple
replicates, implement it above the package jobs as corpus orchestration or result
comparison. Do not put multiple replicates back inside the scientific hot path.
