# 4 — Build the Isolate working grid

Status: implementation handoff for the fourth Isolate-tab milestone.

This milestone adds the spatial geometry that later channels will use. The user
can choose a working scale and block size, see the resolved source, working, and
grid dimensions, and optionally inspect the grid over the current player frame.

It does not compute a channel:

```text
active asset extent + downsample intent + block-size intent
    -> resolved working extent
    -> resolved block grid with partial-edge ownership
    -> geometry readout and player overlay
```

Changing these controls changes geometry only. It must not decode the selected
working window, create a scientific result, or imply that any frame has been
examined.

## 1. Precedence and implementation gate

This is a later, user-authorized increment in the Isolate rebuild. For the work
explicitly in scope, it follows:

- `1-Build-the-player.md` and its accepted implementation.
- The rewrite's accepted media-service implementation and diagnostics.
- The corrected and accepted implementation of `3-Working-window.md`, including
  its review corrections.
- The rewrite's current active-asset, media, and Isolate-state contracts.
- The fixed geometry conventions in the scientific-computation contract where
  they are already accepted by the rewrite.

Do not begin this milestone until the previous working-window milestone has been
accepted. Do not implement against an uncorrected handoff merely because its
number is earlier.

Names and illustrative types in this document are not a demand that the rewrite
copy an oracle class layout. Preserve the required behavior through the
rewrite's smallest current seam.

### 1.1 Required rewrite-side divergence report

Before implementation, inspect the current rewrite and report where it differs
from this handoff's assumptions. At minimum, report:

- How the current active asset exposes source width and height.
- What milestone 3 actually implemented for source requests and resolved media
  metadata.
- Where Isolate-local window and tuning state currently live.
- Whether a downsample or grid value object already exists.
- Whether any current default, validation rule, or persisted setting conflicts
  with the values below.
- How the player maps active-asset coordinates to its painted image, including
  letterboxing, zoom, device-pixel ratio, or a display-size cap.
- Whether the player already has a safe presentation-overlay extension point.
- Whether changing an Isolate control currently triggers media decoding or
  worker activity.

Classify each mismatch as:

```text
compatible naming/shape difference
current implementation that should be preserved
missing capability needed by this milestone
true behavior conflict requiring a user decision
```

Use the report to adapt the handoff to current rewrite ownership. Do not create a
parallel settings model, coordinate transform, or overlay framework just to
imitate the examples here.

If the rewrite already has a scientifically coherent and tested dimension
rounding rule that differs at exact half-pixel ties, report it before changing
it. The important contract is one deterministic rule shared by resolution,
pixel resizing, tests, readouts, and overlays.

## 2. Outcome

At the end of this increment:

- Isolate has a downsample control whose default is `1.0`.
- Isolate has a block-size control expressed in working pixels.
- The block control supports an automatic source-footprint-tracking intent and
  shows the integer working-pixel block size to which it resolves.
- Headless code can resolve the active asset extent and those settings into one
  compact immutable working-grid description.
- The description records source dimensions, requested scale, resolved working
  dimensions, block intent, resolved block size, row/column count, and
  partial-edge geometry.
- The UI shows the resolved geometry in plain language.
- A presentation-only grid overlay can be shown over the player without
  changing or replacing the displayed scientific source frame.
- Control changes update the geometry and overlay immediately without decoding
  the selected temporal window.

No per-frame or per-block scientific values exist yet. A block is only a
spatial ownership cell.

## 3. Scope boundary

Implement only:

- A small headless settings and resolution boundary for working spatial
  geometry.
- Explicit downsample intent in `(0, 1]`, default `1.0`.
- Explicit working-pixel block sizes `>=1`.
- An automatic block intent that approximately holds a 64-source-pixel block
  footprint as downsample changes.
- Deterministic source-to-working dimension resolution.
- Ceil-sized rows and columns.
- Exact owned bounds and fractional area for partial right and bottom blocks.
- Isolate-local controls and resolved-geometry readout.
- A toggleable, presentation-only player grid overlay.
- Focused headless and GUI tests.
- A short manual acceptance path.

Do not implement:

- Pixel downsampling or frame preprocessing.
- A second working-window decoder or source.
- Intensity, change, tensor, flow, color, texture, or learned channels.
- Block means or any other block value.
- Normalization.
- Channel panels, plots, heatmaps, value overlays, or block highlighting.
- Static value bands.
- Morlet transforms, scalograms, or frequency bands.
- Detection counts, gates, clumps, marks, or events.
- Scientific workers, queues, cancellation, or latest-result publication.
- Whole-asset processing.
- Result coverage, persistence, recipes, CLI/HPC execution, or artifacts.
- The oracle's downsample cost-sweep dialog.
- Corpus runtime/storage projections.
- Automatic selection of a “best” downsample or block size.
- A general settings registry, graph planner, plugin API, or rendering system.

Do not add placeholders for those features. Stop after the grid geometry and its
overlay are user-validated.

## 4. Required distinctions

The UI and headless model must keep these concepts separate:

```text
source extent
  Native active-asset pixel dimensions W x H.

requested downsample s
  Scientific intent in (0, 1]. It changes the pixel evidence later supplied to
  channels.

working extent
  Resolved dimensions Ww x Hw after applying s.

requested block intent
  Either auto source-footprint tracking or an explicit integer in working
  pixels.

resolved block side b
  The integer working-pixel block side used by this grid.

block grid
  R rows x C columns covering the complete working extent, including partial
  right and bottom cells.

display projection
  A presentation transform from source/working geometry to the painted player.
  It is not scientific geometry.
```

Downsample and block size are separate scientific levers:

- Downsample changes which pixel evidence is retained and later compute cost.
- Block size changes spatial aggregation, localization, block count, and later
  result size.

Do not label block size in source pixels when it is an explicit working-pixel
value. Do not label a display pixel distance as either source or working
geometry.

## 5. Downsample contract

The requested downsample factor is:

```text
0 < s <= 1
```

Required behavior:

- `1.0` means native working dimensions and is the default.
- The application never silently chooses a coarser scale merely because an
  asset is large.
- A value outside `(0, 1]` fails validation in the headless model.
- The requested factor and resolved dimensions remain separately visible.
- Different requested factors may resolve to the same dimensions on very small
  assets; that is valid and must not produce zero-sized geometry.

The default matters scientifically. Downsampling averages or discards spatial
evidence and can change whether a behavior is detectable. The control may
explain that tradeoff, but this milestone must not claim to know which scale is
adequate for a species, behavior, or project.

### 5.1 Working dimensions

For source dimensions `W,H > 0`, resolve:

```text
Ww = max(1, round(W * s))
Hw = max(1, round(H * s))
```

Use the rewrite's already-established deterministic rounding convention if one
exists and report any difference from the oracle before implementation. If no
convention exists, use Python's integer `round` behavior consistently and pin
exact half-tie cases in tests.

The future pixel resize must produce exactly `Ww x Hw`. The geometry resolver,
future downsampler, grid, readout, and overlay may not each calculate dimensions
differently.

Area downsampling remains the accepted future scientific method when `s < 1`.
This milestone resolves its output geometry only; it does not resize pixels.

Do not derive `s` from a fixed target width. A pre-cropped child and another
asset with the same owned dimensions should resolve identically under the same
explicit settings.

## 6. Block-size contract

Blocks are square in working-pixel coordinates.

The block control has two intents:

```text
auto
explicit b_requested >= 1 working pixels
```

### 6.1 Explicit block intent

For an explicit integer:

```text
b = b_requested
```

An explicit value remains fixed in working pixels when downsample changes. Its
approximate source-pixel footprint therefore changes with `s`.

The headless model rejects zero, negative, fractional, boolean-as-integer, or
otherwise malformed explicit block sizes.

Do not make the oracle widget's old upper limit of 64 a domain contract. The UI
may choose a practical input range that covers the active asset, but the
headless rule is `b>=1`. A block larger than one or both working dimensions is
valid and resolves to a one-cell extent along that axis.

### 6.2 Automatic block intent

The automatic intent tracks an approximately stable source-pixel footprint:

```text
base_source_block = 64
b = max(1, round(base_source_block * s))
```

The UI must display the resolved value, for example:

```text
Block: auto (16 working px)
```

Do not show only “auto.” A user must be able to see the geometry actually in
effect.

The automatic rule is not an automatic downsample decision. It does not change
`s`. It adjusts only the working-pixel block side so that changing downsample can
approximately preserve localization in source coordinates.

Rounding means the source footprint is approximate, especially at very small
scales. Record:

```text
block_intent = auto
base_source_block = 64
resolved_block_size = b
```

Do not serialize only `b` in a way that loses whether the user selected auto or
an explicit value. Persistence itself remains out of scope, but the in-memory
state must preserve the distinction for later work.

If the rewrite already has a different accepted automatic rule or default,
report the divergence rather than silently replacing it.

## 7. Grid and partial-block contract

For resolved working dimensions `Ww,Hw` and block side `b`:

```text
R = ceil(Hw / b)
C = ceil(Ww / b)
```

For block row `r` and column `c`:

```text
y0 = r * b
y1 = min((r + 1) * b, Hw)
x0 = c * b
x1 = min((c + 1) * b, Ww)

owned_width  = x1 - x0
owned_height = y1 - y0
owned_area   = owned_width * owned_height
nominal_area = b * b
area_weight  = owned_area / nominal_area
```

Required behavior:

- The grid covers every working pixel exactly once.
- Blocks do not overlap.
- No padding pixel belongs to a block.
- The last column remains present when `Ww` is not divisible by `b`.
- The last row remains present when `Hw` is not divisible by `b`.
- The bottom-right block combines both partial fractions.
- Full blocks have weight `1`.
- Partial blocks have weight in `(0,1)`.
- Grid and block coordinates are half-open.
- Metadata uses `(x,y)` while array axes later use `[y,x]`.

The area weight is geometry, not a detection value. It is established now so
later block counts and connected components cannot accidentally treat a
one-pixel edge sliver as a full block.

Do not compute block means in this milestone. Later reducers must average only
finite owned pixels and must not include synthetic padding.

### 7.1 Compact representation

The resolved grid must remain compact.

Do not allocate one Python object, JSON record, or Qt item per block merely to
represent a regular grid. At block size 1, that would create one object per
working pixel before any science runs.

A compact description can retain:

```text
source_width, source_height
requested_downsample
work_width, work_height
block_intent
resolved_block_size
rows, columns
```

and derive any block's bounds and area weight by formula.

If the rewrite already has a small immutable extent/grid type, extend or reuse
it. Do not build a general spatial-artifact hierarchy for this milestone.

## 8. Source and working coordinate relationship

Scientific block ownership is defined in working pixels. The player displays the
active asset in source coordinates.

For presentation, map a working boundary to source-coordinate space using the
resolved extents:

```text
source_x = working_x * W / Ww
source_y = working_y * H / Hw
```

Keep these boundary coordinates as floating-point presentation values until the
paint step. Do not repeatedly round through:

```text
working -> source integer -> widget integer -> device integer
```

That can shift or collapse thin edge blocks.

This projection is a visualization of which working cells cover the asset. It
does not claim that a downsampling kernel assigns one exact integer source
rectangle to each working pixel.

The complete outside grid boundary is exactly the active asset extent:

```text
[0,W) x [0,H)
```

The player/display adapter then applies its existing source-to-painted-image
transform. Letterbox margins are never part of the grid.

## 9. Headless geometry boundary

The geometry resolver is independent of Qt, player widgets, and media decoding.

A representative call is:

```text
resolve_working_grid(
    source_width,
    source_height,
    downsample_intent,
    block_intent,
) -> resolved working grid
```

This shape is illustrative. Reuse current rewrite types when they already carry
the necessary meaning.

The resolved value contains at least:

```text
source width and height
requested downsample
working width and height
block intent: auto or explicit
automatic source-footprint base when auto
resolved integer block side
rows and columns
the deterministic rules/version needed to interpret it
```

It provides or permits pure calculation of:

```text
block_bounds(row, column)
block_area(row, column)
block_area_weight(row, column)
working_to_source_boundary(x, y)
```

Resolution is synchronous and cheap. It:

- Opens no media decoder.
- Requests no frames.
- Starts no worker or timer.
- Imports no Qt.
- Writes no files.
- Produces no result coverage.

The active asset is responsible for authoritative source dimensions. Do not ask
the caller to supply dimensions that conflict with the resolved asset and then
silently choose one.

## 10. Isolate controls and readout

Add only the controls required to manipulate and verify this geometry.

The exact widget classes and placement should follow the current rewrite. A
representative presentation is:

```text
Downsample  [1.000]
Block       [auto (64)]
[x] Show grid

Source 411 x 428 px
Working 411 x 428 px
Grid 7 rows x 7 columns
Edge cells: right 27 px, bottom 44 px
```

At `s=0.5` with automatic blocks, it may read:

```text
Working 206 x 214 px
Block auto (32 working px; about 64 source px)
Grid 7 rows x 7 columns
Edge cells: right 14 px, bottom 22 px
```

Required UI behavior:

- Downsample defaults to `1.0`.
- Block intent defaults to automatic source-footprint tracking.
- The displayed automatic block value changes when downsample changes.
- Source, working, block, and grid units are explicit.
- The resolved readout changes immediately with either control.
- Invalid text or values do not create invalid geometry.
- The controls remain usable on a one-frame asset; this is spatial state.
- A clear tooltip says downsampling can remove detectable evidence.
- The UI makes no runtime or storage promise from the setting alone.

Do not add:

- A “Run,” “Prepare,” or “Process” button.
- A channel card or blank plot labelled as future data.
- A quality score.
- A recommended scale.
- An estimated runtime based only on dimensions.
- The oracle's corpus size, body-length, or cost-frontier dialogs.

The existing empty channel area may remain empty.

### 10.1 Defaults and reset

For this milestone:

```text
downsample = 1.0
block intent = auto
base source block = 64
show grid = a presentation choice
```

Use the rewrite's current reset/open-asset behavior. Do not introduce a new
sidecar or settings file solely for these controls.

On asset change, resolve geometry against the new asset's dimensions. An old
resolved grid may not remain painted over the new asset.

If the rewrite already restores Isolate-local settings, preserve its accepted
ownership and report how these two settings will participate. Do not invent
persistence in the oracle handoff.

## 11. Grid overlay

The overlay exists to let the user verify geometry before any channel exists.
It is a player presentation layer, not a modified frame.

Required behavior:

- The user can show and hide it.
- It is drawn over the currently displayed active-asset frame.
- It uses the same source-to-display transform as the frame.
- It remains aligned during widget resize, letterboxing, zoom, and device-pixel
  ratio changes supported by the current player.
- It updates when downsample, block intent, resolved block size, or active asset
  changes.
- It does not seek, decode, replace, resize, recolor, or copy the scientific
  working-window pixels.
- It draws only geometry: no heat, values, detections, or selected blocks.
- Partial right and bottom cells are visibly bounded by the true asset edge.

Use a thin, legible neutral line style that does not imply positive/negative
classification. Follow the rewrite's current visual language rather than
copying an oracle color.

### 11.1 Dense-grid behavior

The overlay must remain responsive when the resolved grid is too dense to draw
meaningfully.

Do not create one persistent Qt graphics item per block. Prefer compact line
generation during paint or the rewrite's existing equivalent.

When projected grid spacing is below a legible display threshold:

- Preserve the true headless geometry.
- Keep the resolved numerical readout accurate.
- Avoid drawing thousands of indistinguishable internal lines.
- Show a small presentation-only indication such as “grid too dense at this
  zoom” if the overlay would otherwise appear broken.
- Draw the asset/grid outside boundary if it remains useful.

The exact legibility threshold is a presentation choice, not a scientific
parameter and not part of grid identity. Pin the policy with GUI tests without
asserting a biological meaning.

If the current player supports zoom, zooming in may make more real grid lines
legible. Do not invent zoom in this milestone if it does not already exist.

## 12. State and invalidation

Downsample and block intent are Isolate-local scientific settings. The resolved
grid is derived state.

In this milestone:

- A control change resolves a new grid.
- The readout and overlay update.
- No working-window pixels are consumed.
- No scientific result is marked stale because no scientific result exists.
- No result or coverage artifact is created.

Later channels will need to capture the resolved grid with their input settings,
and changing either setting will supersede those channel results. Do not build
that invalidation machinery now.

GUI generation tokens remain in GUI orchestration. They are not fields in the
reusable headless geometry request.

## 13. Relationship to the working-window source

The two boundaries remain separate:

```text
working-window source
  temporal selection and native pixel delivery

working-grid resolver
  spatial scale and block ownership
```

This milestone may snapshot both in one Isolate controller state view for later
use, but it must not make the geometry resolver decode pixels.

The next real channel may combine them:

```text
native working-window batches
  -> area downsample to resolved working extent
  -> per-block computation over resolved owned cells
```

That is future behavior, not an instruction to implement it now.

The player continues to use its display representation. The grid overlay uses
geometry projection, not a downsampled scientific frame.

## 14. Errors and edge cases

Fail clearly in the headless resolver for:

- Nonpositive source dimensions.
- Nonfinite, zero, negative, or greater-than-one downsample.
- Invalid block intent.
- Explicit block size below one.
- A resolved state that violates internal grid invariants.

Required edge behavior:

- A `1 x 1` asset resolves at every valid scale to a `1 x 1` working extent.
- An explicit block larger than the working image produces one cell along the
  affected axis.
- A scale so small that rounded width or height would be zero clamps that
  dimension to one.
- Exact divisibility produces no partial edge on that axis.
- Partial width and height combine multiplicatively at the bottom-right cell.
- Changing active assets clears the old overlay before or atomically with
  painting the new geometry.
- Closing Isolate releases no geometry worker because none should exist.

Do not silently coerce malformed headless inputs. A spinbox may prevent invalid
user gestures before constructing settings.

## 15. Oracle-derived guidance

The following are lessons from the oracle, not instructions to copy its
implementation:

### 15.1 Keep scale and block intent separable

The oracle found it valuable to let automatic blocks track the scale so the
downsample control could approximately change compute without also changing
source-space localization.

Preserve the behavior, but do not copy:

- `PipelineConfig`.
- `PreprocessConfig`.
- `FlowConfig`.
- The oracle's live-surface widget layout.
- Atlas packing or multi-replicate geometry.

The rewrite's active asset is already the complete processing world. There is no
replicate atlas in this grid.

### 15.2 Preserve partial-block area now

The oracle once treated thin edge blocks as full cells in later counts. That
allowed a row of slivers to masquerade as substantial occupied area. Establish
the real ownership fraction in the geometry contract before any reducer or
detector can make that mistake.

This is a correctness requirement, not a performance optimization.

### 15.3 Do not silently downsample large assets

The oracle's older width-targeted automatic scale made a pre-cropped asset and
an equivalent owned region resolve differently. The accepted behavior is
explicit scale with native `1.0` as default.

Do not reintroduce a fixed-target-width default in the rewrite.

### 15.4 Avoid premature cache and replay work

The oracle later optimized block-only changes by reusing decoded pixels and
per-pixel intermediates. That is relevant only after real channel computation
exists and measurement shows the block reduction is the remaining cost.

This milestone has no channel pixels to cache. Do not add:

- Decoded-frame caches.
- Per-pixel channel caches.
- Memmaps.
- Block re-reduction.
- Cache invalidation keys.

Carry the geometry cleanly so that later measured reuse remains possible.

## 16. Benchmark and diagnostic boundary

This milestone does not introduce a reusable performance estimate.

Do not extend:

```powershell
sieve media benchmark
```

to claim grid-overlay or scientific-processing performance. The accepted media
diagnostic measures the media-service representation; an overlay is a different
layer.

Ordinary GUI timing used to prevent an obvious paint regression may remain a
test or disposable implementation investigation. If the project later decides
that users need a reusable estimate of visible Isolate rendering or scientific
grid throughput, follow the accepted benchmark-diagnostic decision:

- Put measurement logic in an importable package module.
- Expose it through a deliberate supported CLI or UI surface.
- Provide human and versioned machine-readable results.
- Identify representation, sample/cache conditions, environment, statistics,
  and measured versus extrapolated conclusions.
- Require explicit user action for expensive work.

Do not build that product diagnostic as part of this geometry milestone.

## 17. Suggested implementation shape

Fit the work into the rewrite's current packages. A small shape may be:

```text
headless geometry module
  settings/intents
  resolve source extent -> working extent and block grid
  pure block-bound and weight calculations

Isolate-local state/controller
  own current downsample and block intent
  resolve against current active asset
  expose current resolved grid to presentation

player presentation
  paint optional grid through the existing source-to-display transform
```

Do not add a pipeline base class, artifact registry, generic renderer, or worker
framework.

Prefer a compact immutable resolved value. It should be cheap to compare and
safe to snapshot later with a channel request.

The geometry module must not import:

- PyQt.
- Player widgets.
- Media decoding.
- Channel or detection modules.

The overlay must not become the owner of the scientific grid. It receives a
resolved grid and paints it.

## 18. Automated tests

Use small, hand-calculable dimensions. Geometry tests do not need video decode.

### 18.1 Downsample validation and resolution

Test:

- `s=1.0` preserves source dimensions.
- A representative fractional scale resolves both axes by the declared rule.
- Very small valid scales clamp each resolved dimension to at least one.
- Zero, negative, greater-than-one, NaN, and infinite scales fail.
- Requested scale and resolved dimensions are both retained.
- The same source dimensions and settings resolve deterministically.
- If exact half ties are possible, tests pin the accepted rounding behavior.

### 18.2 Block intent

Test:

- Explicit block 16 resolves to 16 at scales `1.0`, `0.5`, and `0.25`.
- Auto resolves to 64, 32, and 16 at those scales.
- Auto never resolves below one.
- Explicit zero, negative, fractional, and boolean values fail.
- Auto intent remains distinguishable from explicit 64 after resolution.
- A block larger than the working extent remains valid.

### 18.3 Grid dimensions

Test:

- `Ww=32,Hw=16,b=16` resolves to `R=1,C=2`.
- `Ww=20,Hw=40,b=16` resolves to `R=3,C=2`.
- Exact divisibility produces only full blocks.
- Nondivisible dimensions retain the last row and column.
- `R*C` is not materialized as a list merely to resolve the grid.

### 18.4 Owned bounds and fractional area

For `Ww=20,Hw=40,b=16`, test:

```text
full interior weight = 1
right-edge weight = 4/16
bottom-edge weight = 8/16
bottom-right weight = (4*8)/(16*16) = 1/8
```

Also test:

- Every returned bound is half-open and within the working extent.
- Adjacent bounds touch without overlap or gaps.
- Summed owned area equals `Ww*Hw`.
- Full cells have exact weight one.
- No padding contributes area.
- Invalid row or column indices fail clearly.

### 18.5 Coordinate projection

Test:

- Working origin maps to source origin.
- Working right/bottom extent maps exactly to source right/bottom extent.
- Interior block boundaries map monotonically.
- A downsampled partial edge still terminates at the true asset boundary.
- Projection uses resolved extents rather than `1/s` alone, avoiding mismatch
  after dimension rounding.

### 18.6 Controller/state behavior

Test:

- Opening an asset resolves the default grid from its dimensions.
- Changing downsample updates working dimensions and auto block text.
- Changing explicit block size updates rows and columns without requesting a
  frame or consuming a working-window batch.
- Switching assets replaces the resolved geometry and cannot retain the old
  overlay.
- No control change creates examined or processed coverage.
- No worker/thread is created for grid resolution.

### 18.7 Overlay behavior

Use the rewrite's established GUI test environment.

Test:

- Overlay off paints no grid lines.
- Overlay on aligns its outer boundary with the painted source image, not the
  widget or letterbox.
- Representative interior lines match the resolved grid.
- Partial right and bottom cells terminate at the true image edge.
- Resizing and any existing zoom keep alignment.
- Changing settings repaints geometry without media decode.
- A dense grid uses the presentation density policy rather than creating one Qt
  item per cell.
- The overlay contains no heat/value/detection state.

Do not make pixel-perfect antialiasing snapshots the only proof of alignment.
Test the transformed boundary coordinates directly and use a small screenshot or
grab as a supplemental regression check.

### 18.8 Player regression

Retain existing player tests for:

- Exact displayed frame identity.
- Seeking, scrubbing, stepping, and looping.
- Display-size media requests.
- Asset switching and stale-frame rejection.

The grid overlay must not cause a second decode of the displayed frame merely
because it repaints.

## 19. Manual acceptance

After automated tests pass, stop and return this milestone for validation.

1. Open an ordinary source asset and switch to Isolate.
2. Confirm the default readout reports native working dimensions at
   downsample `1.0`.
3. Turn on the grid and confirm its outside boundary aligns with the visible
   video rather than any letterbox margin.
4. Choose an explicit block size that leaves partial right and bottom cells.
5. Confirm those cells end at the actual video edge and are visibly narrower or
   shorter than full cells.
6. Change downsample while block is explicit and confirm the working dimensions
   and grid count update.
7. Select automatic block intent and change downsample through `1.0`, `0.5`, and
   `0.25`.
8. Confirm the resolved block readout changes approximately `64`, `32`, `16`
   working pixels and the source-space grid remains approximately stable.
9. Scrub, play, resize, and use any existing zoom behavior with the overlay on.
10. Confirm the grid stays aligned and the player remains responsive.
11. Select a very dense grid and confirm the UI remains responsive and explains
    any presentation suppression without changing the numerical grid readout.
12. Switch to a derived child asset and confirm source and working dimensions
    now describe the child itself.
13. Change controls rapidly and confirm no complete working-window decode,
    channel computation, plot, or success/coverage state appears.

Do not judge pixel downsampling or block values in this milestone; neither is
computed yet.

## 20. Definition of done

This milestone is complete only when:

- The rewrite-side divergence report was reviewed and any true conflicts were
  resolved before implementation.
- A non-Qt resolver produces deterministic working and block geometry.
- Downsample defaults to native `1.0` and never changes automatically.
- Explicit block size is in working pixels.
- Automatic block intent exposes its resolved working-pixel value and tracks an
  approximately 64-source-pixel footprint.
- The complete working extent is covered exactly once.
- Partial right and bottom blocks retain their real bounds and fractional area.
- The resolved grid remains compact rather than allocating one object per cell.
- Source, working, and display coordinates remain distinct.
- The Isolate readout makes dimensions, units, intent, and resolved values
  visible.
- The optional grid overlay aligns through the player's existing display
  transform and remains responsive for dense grids.
- Changing geometry performs no working-window decode and creates no scientific
  coverage or result.
- Existing player behavior has not regressed.
- No channel, normalization, detector, persistence, recipe, cache, or general
  executor was added.
- Focused automated tests pass.
- The user has reviewed and accepted the visible interaction.

Stop here. The next possible handoff is the first real channel, but completion
of this milestone does not authorize beginning it.
