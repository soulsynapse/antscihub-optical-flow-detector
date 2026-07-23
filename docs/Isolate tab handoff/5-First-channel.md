# 5 — Add the first Isolate channel

Status: implementation handoff for the fifth Isolate-tab milestone.

This milestone turns the accepted temporal source and spatial grid into one real
scientific result: block-mean encoded-luma intensity over the selected looping
window. It adds the first cancellable GUI scientific job and one concrete
channel panel. It does not create a general processing framework.

The required result is:

```text
immutable working-window request
+ immutable resolved working grid
+ fixed rgb24-to-intensity rule
    -> bounded headless intensity computation
    -> one complete time x block result
    -> one Isolate channel panel
    -> player-synchronized absolute-frame cursor
```

The user must be able to select a window, resolve a grid, run intensity, inspect
the values through time and space, seek with the channel cursor, and safely
supersede obsolete work.

## 1. Precedence and implementation gate

This handoff follows:

- `1-Build-the-player.md` and its accepted implementation.
- The accepted media-service implementation and diagnostics.
- The corrected and accepted implementation of `3-Working-window.md`.
- The corrected and accepted implementation of `4-Working-grid.md`.
- The rewrite's current active-asset, player, Isolate-state, worker, and
  lifecycle contracts.
- The fixed geometry and intensity conventions in
  `docs/sieve-scientific-computation-contract.md`.

Do not begin milestone 5 until milestones 3 and 4 have been implemented,
manually validated, and accepted. In particular, do not build channel code on a
provisional grid or silently choose the grid-setting asset-switch policy on this
milestone's behalf.

Names and illustrative types in this document are not a demand that the rewrite
copy an oracle class layout. Preserve the required behavior through the
rewrite's smallest current seam.

### 1.1 Required rewrite-side divergence report

Before implementation, update `.isolate-state-divergence.md` with:

- The accepted milestone-4 settings and resolved-grid types.
- Which owner snapshots temporal and spatial scientific inputs.
- How recorded active-asset dimensions are compared with dimensions probed by
  the request-local working-window source.
- The exact native `rgb24` batch shape, byte order, stride, mutability, and
  source-conversion provenance available to a consumer.
- The existing source cancellation checks and final-outcome behavior.
- The smallest current worker/supersession seam in Isolate, if one exists.
- Current player position/seek signals and the right-hand placeholder layout.
- Any current rewrite behavior that conflicts with the fixed intensity,
  downsample, block, validity, or publication rules below.

The rewrite should use that report to adapt ownership and placement. It should
not create a parallel source, grid, player clock, or settings model to imitate
oracle examples.

If the `rgb24` source does not preserve enough conversion provenance to identify
how source color became RGB, report that explicitly. This milestone may record
the current media service and `rgb24` conversion implementation as part of its
scientific identity; it must not claim that the RGB bytes are verified native
luma.

## 2. Outcome

At completion:

1. A Qt-independent caller can compute intensity from the accepted working
   window and grid without importing GUI modules.
2. The computation consumes bounded native `rgb24` batches and never uses the
   display preview raster.
3. Each source frame produces exactly one `(grid_rows, grid_columns)` block
   plane aligned with the accepted working grid.
4. The complete result records absolute frames, rational timebase, source and
   working geometry, grid identity, intensity conversion, outcome, and processed
   span.
5. Isolate exposes one explicit **Compute intensity** action, progress/cancel
   state, and one channel panel.
6. The panel displays time-by-block values on a fixed meaningful intensity
   scale and synchronizes its cursor with the player.
7. Asset, window, downsample, or block changes make running and completed
   results obsolete immediately. No obsolete worker may publish as current.
8. Player seeks and ordinary playback-position changes move the cursor without
   recomputing intensity.

## 3. Scope boundary

Implement:

- One fixed channel id, `intensity`.
- Deterministic conversion of delivered `rgb24` pixels to canonical `[0,1]`
  intensity.
- Area downsampling to the already-resolved working dimensions.
- Mean reduction over the already-resolved grid, including partial edge cells.
- A small immutable computation request and result.
- Bounded synchronous headless computation with progress, cancellation, and a
  final outcome.
- One GUI worker that owns one request-local source stream.
- GUI generation or request tokens used only to reject obsolete publication.
- One compact time-by-block panel with a player-synchronized cursor.
- Focused numerical, lifecycle, stale-result, and offscreen GUI tests.

Do not implement:

- Normalization, including z-score or CLAHE.
- `change`, tensor, flow, appearance, texture, speed, or derived channels.
- Previous-frame context or frame pairing; intensity at frame `t` uses only
  frame `t`.
- A channel registry, plugin interface, add/remove UI, or configurable channel
  graph.
- Morlet transforms, value bands, selected blocks, detection, events, marks, or
  a result timeline.
- A current-frame intensity overlay on the player.
- Whole-asset execution, processing plans, recipes, CLI/HPC execution, export,
  or persistence.
- A pixel cache, channel cache, job directory, or result artifact store.
- Background subtraction, registration, masks, temporal denoising, or fitted
  preprocessing assets.
- Automatic reruns on every control change.
- Repairs to unrelated display-decoder, asset-registration, derivation,
  preview-aspect, or media-benchmark issues listed in the divergence ledger.

Stop after the first channel is integrated and validated. Do not begin
normalization or the second channel.

## 4. Required distinctions

Keep these concepts separate:

```text
native source frame
  one immutable rgb24 frame delivered by the working-window source

working intensity frame
  canonical float intensity after RGB conversion and area downsampling

block plane
  one mean intensity value per resolved grid cell for one absolute frame

channel result
  ordered block planes plus identity, geometry, time, validity, and outcome

GUI job token
  mutable publication identity; not scientific identity or provenance

player preview
  width-capped display bytes; never scientific input
```

Also distinguish:

- Requested span from processed span.
- Recorded content identity from content verified during this run.
- Successful source delivery from successful channel computation.
- A cancelled/failed prefix from a complete current result.
- Cursor position from computation settings.

## 5. Scientific intensity contract

### 5.1 Input

The only accepted input plane is the milestone-3 native-size `rgb24` plane:

```text
shape: H x W x 3
channel order: R, G, B
sample type: uint8
sample interval: [0,255]
coordinates: active asset native coordinates
```

A batch must be exact-sized and immutable for the lifetime promised by the
working-window contract. A short or malformed frame fails the channel job; it
must not be padded, cropped, resized to a nearby shape, or filled with zero.

The computation must first require:

```text
resolved_source.width  == captured_grid.source_width
resolved_source.height == captured_grid.source_height
```

A mismatch is a structured stale-input/geometry failure. Never silently choose
one extent.

### 5.2 RGB to encoded-luma intensity

For each delivered RGB code triplet:

```text
I_raw = (0.299*R + 0.587*G + 0.114*B) / 255
```

Requirements:

- Evaluate the weighted sum in at least `float32`; the conformance reference
  evaluates in `float64`.
- Store working pixels as `float32`.
- The result interval is `[0,1]`.
- Do not apply gamma linearization. This is encoded luma-like intensity, not
  physical linear-light luminance.
- Do not quantize the weighted sum back to an 8-bit gray code.
- Do not substitute OpenCV's implicit BGR conversion or a backend-default
  grayscale transform without proving it implements this rule.
- Non-finite output fails the frame. It is never replaced with zero.

Record a stable conversion id such as:

```text
sieve.intensity.rgb601_float.v1
```

The id names this exact post-`rgb24` rule. Result provenance must also retain the
working-window plane descriptor and media-service conversion identity/status so
the result does not imply that the RGB codes were source-native luma.

This fixed rule is intentionally small enough to make the first channel
reproducible. A future source-native luma plane, explicit color-management
policy, linear-light representation, or alternate color channel is a different
node and implementation identity.

### 5.3 No normalization

Milestone 5 has exactly one normalization state:

```text
normalization = off
```

Do not add a disabled combo box for future modes. Do not rescale each frame,
each block, or the complete window for display convenience. The panel's color
mapping uses the fixed scientific `[0,1]` interval.

## 6. Working-resolution contract

Use the accepted milestone-4 resolved working dimensions:

```text
Ww = resolved_grid.working_width
Hw = resolved_grid.working_height
```

For each native intensity frame, produce one `Hw x Ww` frame by exact area
averaging of source-pixel footprints. This is a shrinking or identity operation;
upsampling is an error.

The resolved effective scales remain:

```text
sx = Ww / W
sy = Hw / H
```

Use resolved integer dimensions, not the requested decimal downsample, as the
pixel-operation target. This prevents geometry and pixels from independently
rounding to different shapes.

The area reducer must be deterministic at odd sizes and non-integer ratios.
Using a measured backend fast path is allowed only when it conforms to the
reference fixtures and retains the same boundary behavior. Backend name and
version belong in computation provenance until equivalence is established.

Do not:

- Resize the display preview and call it working data.
- Resize independently per block.
- Stretch to an even width or height for an encoder constraint.
- Crop remainder pixels to make dimensions divisible by the block size.
- Change downsample automatically because a window is large.

## 7. Block reduction and partial cells

Use the accepted milestone-4 grid exactly:

```text
R = grid.rows
C = grid.columns
b = grid.block_size_working_px
```

For cell `(r,c)`:

```text
y0 = r*b
y1 = min((r+1)*b, Hw)
x0 = c*b
x1 = min((c+1)*b, Ww)
area = (y1-y0)*(x1-x0)
value[t,r,c] = mean(I_working_t[y0:y1, x0:x1])
```

Requirements:

- Use only owned pixels; no padding enters the statistic.
- Accumulate the conformance reference in `float64`.
- Store values as `float32`.
- Preserve the grid's fractional edge-cell weights:
  `area/(b*b)`.
- A valid finite RGB frame produces finite values for every nonempty cell.
- Empty cells are invalid geometry and must have been rejected by the grid
  resolver.
- Flattening for presentation is row-major only:
  `block_index = r*C + c`.

Partial-cell weight does not alter the cell's mean. It records how much nominal
block area the cell owns for later density and occupancy calculations.

## 8. Headless request and result

Keep the first channel contract concrete. A small immutable request may contain:

```text
working_window_request
resolved_grid
channel_id = intensity
intensity_conversion_id
normalization = off
implementation_id
execution batch size
optional cancellation token/callback
```

Execution batch size and cancellation polling are execution settings, not
scientific identity, unless testing shows they change valid numerical results.

The request must not contain:

- A PyQt object, widget value, or player raster.
- A GUI generation or publication token.
- Mutable active-asset/session objects.
- A general channel registry or future graph.

The complete immutable result contains at least:

```text
asset id and recorded content identity
content-verification status
requested absolute half-open frame span
processed absolute half-open frame span
exact rational fps
source width and height
working width and height
effective sx and sy
requested downsample intent
resolved block size, rows, and columns
partial-cell weights or compact geometry sufficient to reproduce them
channel id and conversion id
normalization id
implementation/backend identity
absolute frame indices or an equivalent contiguous-span invariant
values shaped T x R x C, float32
final outcome
```

For a successful result:

```text
T == requested_stop - requested_start
processed_span == requested_span
frame_indices are contiguous absolute indices
values.shape == (T,R,C)
all values are finite and in [0,1]
```

Do not serialize pixels, values, or results in this milestone.

## 9. Coverage, validity, and incomplete execution

Intensity is a scientific computation, so successfully reduced frames may be
called processed for this channel. They are not quiet, detected, or negative.

The headless execution outcome distinguishes at least:

```text
completed
cancelled
source_truncated
source_failed
computation_failed
```

The headless layer may expose a processed prefix with its honest outcome for
diagnostics and later consumers. The milestone-5 GUI publishes a channel panel
as current only for a complete `completed` result covering the complete captured
request.

On cancellation, truncation, or failure:

- Never pad the missing tail.
- Never render a missing tail as zero intensity.
- Never preserve an old result under new settings as though it were current.
- Close the request-local source.
- Report the structured outcome and processed prefix/span.
- Leave the GUI with no current result for the failed request.

The result panel does not build the later three-state result timeline. It only
knows whether one captured window has one complete current intensity result.

## 10. Bounded execution and cancellation

The headless computation is synchronous and Qt-independent. It consumes the
working-window stream batch by batch:

```text
open request-local working-window stream
for each bounded rgb24 batch
    check cancellation
    validate shape and absolute indices
    convert RGB to intensity
    area-downsample each frame
    reduce each frame to the resolved grid
    append one bounded result batch
    report progress
finalize outcome
close stream
```

The complete window's `T x R x C` values may be retained in memory. Native or
working pixel frames remain batch-bounded and are released after reduction.

Cancellation must be observed between bounded decode/compute operations. A
consumer must not need to wait for the complete window before cancellation is
noticed. Explicit close, normal completion, cancellation, source failure, and
computation failure all release the request-local media session.

The headless computation does not implement latest-request policy. That policy
belongs to GUI orchestration.

## 11. GUI job lifecycle and supersession

Milestone 5 introduces the first GUI scientific worker. Use one worker for one
immutable captured request:

```text
current GUI state
    -> snapshot working-window request
    -> snapshot resolved grid
    -> validate dimension equality
    -> assign GUI job token
    -> run headless intensity computation on worker
```

The worker owns its request-local working-window stream. It must not share the
player's `MediaSession`, decoder, timer, or preview bytes.

### 11.1 Explicit action

Add one action labelled **Compute intensity**. It captures the current inputs
and starts the job. Add a **Cancel** action only while a job is active.

Do not automatically start a scientific job merely because:

- An asset opened.
- The temporal window changed.
- Downsample or block intent changed.
- The user switched tabs.
- The player sought or advanced.

Automatic recomputation can be considered after the first path is validated.

### 11.2 Job identity

The scientific input key includes all captured upstream facts that can change
the result:

```text
registered asset/content identity and status
requested temporal span
native source extent and rational timebase
rgb24 plane/conversion identity
resolved working dimensions
downsample intent
resolved block intent and grid geometry
intensity conversion id
normalization off
implementation/backend identity
```

The GUI job token is separate and answers only whether a signal may still update
the current UI.

### 11.3 Supersession rules

Immediately invalidate the current GUI job token and request cancellation when:

- The active asset changes.
- The looping window start or stop changes.
- Downsample intent or resolved working dimensions change.
- Block intent or resolved grid changes.
- A new **Compute intensity** action supersedes an active job.
- Isolate is closed.

A player seek or ordinary playback tick does not invalidate the result.

Late progress, completion, failure, or cancellation signals from an obsolete
token must not:

- Replace the current panel.
- Clear a newer result.
- Change newer progress/status text.
- Close or reposition the player's media session.

Do not use forceful thread termination. If a new job is requested while an old
worker is stopping, keep only the newest pending request and start it after the
old worker has actually released its source and terminated. Check worker
shutdown success; do not copy the unsafe start/interrupt/timeout pattern listed
in the divergence ledger.

## 12. Isolate controls and status

Use a compact control/status area near the grid controls or channel panel:

```text
[Compute intensity] [Cancel]
Intensity: idle
```

During a job, report bounded factual progress such as:

```text
Intensity: processing 84 / 240 frames
```

On completion:

```text
Intensity: current — frames [1200,1440), grid 7 x 8
```

On cancellation/failure, report the outcome without leaving a partial result
painted as current. Do not add throughput, completion-time estimates, or a
reusable processing benchmark casually. If a reusable estimate is introduced,
it must follow the accepted benchmark-diagnostic decision in
`.isolate-handoff-practices`.

The action is enabled only when:

- A valid registered active asset exists.
- The temporal window is valid.
- The grid is valid.
- No worker is in the non-interruptible portion of shutdown.

It is not gated on whether the asset has more than one frame.

## 13. First channel panel

Use the existing right-hand channel placeholder for one concrete intensity
panel. Do not replace it with a general dock, card registry, or add/remove
framework.

The panel displays a raster with:

```text
x axis: absolute frame/time across the captured window
y axis: row-major block index 0 .. R*C-1
cell value: block-mean intensity in [0,1]
color scale: fixed [0,1]
```

Required visible context:

- Channel name: Intensity.
- Absolute window `[start,stop)`.
- Source and working dimensions.
- Grid rows, columns, and resolved block size.
- A stable legend for `[0,1]`; no per-window autoscaling.
- A vertical cursor for the player's current absolute frame when it lies within
  the computed span.

The panel should rasterize the value array efficiently. Do not allocate one Qt
graphics item or widget per frame/block cell.

### 13.1 Cursor synchronization

Use the existing player/Isolate position as the one clock:

- Player seek, scrub, step, and playback update the channel cursor.
- The cursor uses absolute frame identity, not a percentage.
- When the player is outside the computed window, hide or clearly detach the
  cursor; do not clamp it to an edge and imply the edge frame is current.
- Clicking a valid time column in the channel panel seeks the player to that
  absolute frame through the existing Isolate seek path.
- Moving only the cursor never starts, cancels, or recomputes the channel.

Do not create a second timer or independent playback position in the panel.

### 13.2 Spatial interpretation

The y axis is a compact representation, not an intuitive 2-D spatial view.
Provide a minimal row/column readout on hover or selection:

```text
frame t
block (r,c)
intensity value
owned working bounds
partial-cell weight
```

Do not add current-frame block highlighting or a player intensity overlay in
this milestone. Those interactions belong to later value-filtering work.

## 14. State and invalidation

Keep these states explicit:

```text
no active input
ready, no result
processing
stopping/superseding
complete current result
cancelled
failed
```

Do not represent a stale result by merely changing a label while leaving it
visually indistinguishable from current data. On any upstream change, remove or
visibly disable the old raster immediately.

A captured immutable request does not change when widgets change. The running
worker may finish, but its token prevents publication.

Presentation-only changes do not invalidate scientific data, including:

- Panel resize.
- Color-map choice, if one fixed alternative is later added.
- Grid-overlay visibility.
- Tab switching.

Milestone 5 does not persist any of these states.

## 15. Errors

Surface structured errors at the narrowest owner:

- Missing/unregistered asset or identity mismatch: working-window resolution.
- Probed source extent differs from captured grid extent: channel admission.
- Malformed `rgb24` batch or noncontiguous absolute indices: source/channel
  boundary.
- Invalid working/grid geometry: grid resolver or channel admission.
- Non-finite intensity or block output: channel computation.
- Source truncation/failure: working-window outcome propagated through the
  channel outcome.
- Worker shutdown failure: GUI lifecycle error, not a scientific zero result.

An error message should identify the captured asset, requested span, and failed
stage without dumping large arrays or backend command lines into the panel.

## 16. Oracle-derived guidance

The oracle currently obtains intensity as a cheap prerequisite inside
`core.tensor_channels`, block-reduces it, and avoids the tensor solve when
intensity alone is requested. Those are useful behavioral and performance
observations, not APIs for the rewrite.

### 16.1 Required behavioral lessons

- Intensity uses only the current frame.
- Selecting intensity alone must not compute temporal differences, gradients,
  tensor products, Gaussian tensor integration, flow, or other channels.
- Downsample changes pixel evidence and compute cost.
- Block size changes spatial aggregation.
- Partial blocks contribute their owned-pixel mean and retain fractional area.
- Absolute frame identity and selected window are explicit.

### 16.2 Performance hints to test, not inherit blindly

Potential low-risk optimizations include:

- Convert a bounded RGB batch with vectorized array operations.
- Use a conforming area-resize backend rather than Python pixel loops.
- Reduce complete blocks through reshape/sum and handle only the final ragged
  row/column separately.
- Reuse precomputed compact grid bounds and weights for every frame.
- Emit progress at a bounded human-readable rate rather than once per block.

For each optimization:

1. Pin the reference numerical result first.
2. Measure the rewrite before and after on the same representative input.
3. Change one hypothesis at a time.
4. Retain it only if it improves the measured bottleneck without changing
   results outside accepted tolerances.
5. Record useful negative results.

Do not add integral-image, cache, GPU, or parallel-frame complexity without a
measurement showing it is needed for this increment.

## 17. Suggested implementation shape

The smallest coherent shape is:

```text
Qt-independent application/science module
    IntensityRequest
    IntensityResult
    compute_intensity(request, cancellation, progress)

Isolate GUI
    snapshots temporal request + resolved grid
    owns GUI job token and one worker
    publishes only complete current results

IntensityPanel
    paints T x (R*C) raster
    follows and requests seeks through existing player position
```

The rewrite may use different names or combine small values where its accepted
contracts already do so. Do not introduce `ChannelRegistry`, `Pipeline`,
`Executor`, `ArtifactStore`, or a general `ProcessingController` for one fixed
channel.

## 18. Automated tests

Use tiny deterministic lossless fixtures. Compare against a simple independent
reference implementation, not the optimized production reducer calling itself.

### 18.1 RGB conversion

Pin at least:

```text
black   (0,0,0)       -> 0
white   (255,255,255) -> 1
red     (255,0,0)     -> 0.299
green   (0,255,0)     -> 0.587
blue    (0,0,255)     -> 0.114
```

Also test mixed codes, array order, dtype, `[0,1]` bounds, nonmutation of source
bytes, and rejection of malformed/non-finite input at the applicable boundary.

### 18.2 Area downsampling

Test:

- Scale `1.0` preserves dimensions and values.
- Odd native dimensions resolve to the milestone-4 working dimensions.
- A hand-computed non-integer area case matches the reference.
- A one-pixel working dimension remains valid.
- No even-dimension coercion occurs.
- Upsampling and geometry mismatch fail.
- Different execution batch sizes produce the same working frames.

### 18.3 Block reduction

Test:

- Exact division into full cells.
- Partial right edge.
- Partial bottom edge.
- Partial bottom-right corner.
- Block larger than one working dimension but valid through one partial cell.
- Mean uses only owned pixels.
- Partial weight is `owned_area/(b*b)`.
- Row-major flattening round-trips to `(r,c)`.
- Values and weights agree with an independent float64 reference.

### 18.4 Request and result

Test:

- `[2,5)` produces frames `2,3,4` and shape `(3,R,C)`.
- A one-frame window produces one intensity plane.
- Exact rational fps is retained.
- Source, working, grid, conversion, normalization, implementation, identity,
  and verification provenance are present.
- Complete success requires exact requested coverage.
- Source truncation preserves only its honest prefix and is not completed.
- Cancellation and computation failure carry distinct outcomes.
- Headless imports do not import PyQt.

### 18.5 Bounded lifecycle

Test:

- Native/working pixel memory is batch-bounded.
- Normal completion closes the request-local source.
- Cancellation closes it.
- Explicit close and failure close it.
- Cancellation is observed before the complete window is processed.
- Progress is monotonic, bounded, and ends exactly on successful completion.
- Intensity-only execution never calls tensor, flow, Morlet, or detection code.

### 18.6 GUI supersession

Run Qt tests with:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:QT_QPA_FONTDIR = "C:/Windows/Fonts"
```

Test:

- **Compute intensity** snapshots immutable temporal and grid inputs.
- A completed current job populates the panel once.
- Asset change rejects late progress, failure, and success from the old job.
- Window change does the same.
- Downsample and block changes do the same.
- A second compute request supersedes the first without two source-owning
  workers running concurrently.
- An obsolete cancellation/failure cannot clear a newer result.
- Closing Isolate cancels and verifies worker shutdown.
- A failed worker wait is surfaced rather than silently ignored.
- Player display decode remains independent throughout.

### 18.7 Panel and cursor

Test:

- Raster axes map absolute frames and row-major blocks correctly.
- Fixed `[0,1]` colors do not autoscale between results.
- Player positions inside the window move the cursor to the exact frame.
- Positions outside hide/detach the cursor rather than clamp it.
- Clicking a column seeks through the existing Isolate path.
- Cursor movement does not launch scientific work.
- Panel resize does not invalidate the result.
- Hover/readout maps back to correct `(frame,r,c)`, bounds, value, and weight.

### 18.8 Existing regressions

The accepted working-window, working-grid, player, active-asset, media-service,
and cleanup suites continue to pass. Do not weaken their lifecycle or
provenance assertions to make this milestone pass.

## 19. Manual acceptance

After automated tests pass, stop and return this milestone for user validation.

The manual path is:

1. Open a registered asset in Isolate and choose a short looping window.
2. Confirm the accepted working-grid readout and optional overlay are correct.
3. Click **Compute intensity** and confirm progress advances without freezing
   player controls.
4. Confirm the completed panel shows the exact window, dimensions, grid, fixed
   `[0,1]` legend, and spatial/time data.
5. Play, scrub, and step within the window. Confirm the panel cursor follows the
   exact absolute frame without recomputation.
6. Click several channel columns and confirm the player seeks to those frames.
7. Seek outside the computed window and confirm the channel cursor does not
   clamp misleadingly to an edge.
8. Start another computation and cancel it. Confirm no partial tail appears as
   zero and the previous request is not reported current.
9. Start a computation, then change the window. Confirm the old result
   disappears immediately and late old signals do not repaint it.
10. Repeat while changing downsample or block intent and then while changing
    active asset.
11. Compute a one-frame window and confirm it produces one valid block plane.
12. Close the window during a job and confirm clean worker/source shutdown.

Do not judge normalization, motion response, value filtering, spectral
behavior, or detection in this milestone; none is implemented.

## 20. Definition of done

This milestone is complete only when:

- Milestones 3 and 4 are accepted prerequisites.
- The rewrite-side divergence report is refreshed and dispositioned.
- One Qt-independent intensity computation consumes the accepted native
  working-window source and resolved grid.
- The exact RGB-to-`[0,1]` conversion is pinned and identified.
- Downsampling is area-based and targets the resolved working dimensions.
- Block means use owned pixels and retain partial-cell weights.
- The result preserves absolute time, rational rate, geometry, identity,
  verification status, implementation provenance, processed span, and outcome.
- Execution is bounded and cancellable and closes its source on every exit.
- The GUI has one explicit compute action, one worker, and one channel panel.
- GUI publication rejects every obsolete progress/result/error path.
- Worker shutdown is verified and does not copy the known display-thread race.
- The panel uses a fixed `[0,1]` scale and one player clock.
- Player seeks update the cursor without recomputation, and channel clicks seek
  through the existing player path.
- Incomplete work is never padded or presented as a complete current result.
- No normalization, other channels, transforms, detection, result timeline,
  persistence, caches, recipes, whole-asset execution, or general channel
  framework were introduced.
- Focused headless and offscreen GUI tests pass.
- The user has manually validated and accepted the milestone.

Stop here. Normalization is milestone 6 and is not authorized by completion of
this milestone.
