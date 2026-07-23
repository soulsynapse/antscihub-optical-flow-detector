# Review — 4 Build the Isolate working grid

Originally reviewed on `2026-07-23 14:53:33 -07:00` at commit `4b09232`.
Reviewed and updated after milestone 3 on `2026-07-23 15:16:37 -07:00` at
commit `01b256f`.

Status: architectural and current-rewrite review of `4-Working-grid.md`.

## Verdict

The handoff is scientifically sound and substantially better aligned with the
rewrite than the original working-window handoff was.

Its strongest decisions are:

- Native `1.0` is the downsample default.
- Downsample and block intent remain separate controls.
- Automatic blocks expose their resolved working-pixel size.
- Working dimensions and block geometry resolve headlessly without decoding.
- Ceil-sized grids preserve all right and bottom edge pixels.
- Partial blocks carry their actual area fraction before any detector exists.
- The grid representation remains compact rather than allocating one object per
  block.
- The overlay is presentation-only and does not become scientific evidence.
- Control changes do not consume the working-window source.
- Dense-grid presentation is bounded.
- Channels, preprocessing, workers, persistence, caches, and processing remain
  deferred.

The core geometry formulas are safe to implement.

The handoff needs a few current-checkout clarifications before implementation.
Most are not scientific changes; they prevent an implementer from inventing
abstractions around GUI capabilities the rewrite does not have.

The largest sequencing fact is now:

> Milestone 3 is implemented in commit `5bffca7`, but still awaits user
> acceptance.

The required post-milestone-3 divergence check is complete and recorded in
`.isolate-state-divergence.md`. Milestone 4 remains queued only until milestone
3 is accepted, the corrected milestone-4 documents are accepted, and the
asset-switch setting policy is chosen.

## Current-checkout evidence

### Present

- `ActiveAsset` carries registered source width, height, content hash, and
  sidecar/media references.
- `application.working_window` provides Qt-free request, resolved-source,
  plane, batch, stream, provenance, and outcome contracts.
- `IsolateSession.snapshot_working_window_request()` captures registered
  identity and temporal bounds without opening scientific media.
- `IsolateSession` owns temporal window state, playback, the display
  `MediaSession`, and its decoder thread.
- `IsolateTab` owns the widgets and connects active-asset changes to
  `IsolateSession`.
- `IsolatePlayer` paints a capped RGB preview into one letterboxed
  `image_rect()`.
- `IsolatePlayer.image_rect()` is already directly testable.
- Qt painting uses logical widget coordinates, so normal device-pixel scaling is
  handled by the painter.
- The right-hand channel area remains an explicit empty placeholder.
- Existing tests cover letterboxing, one-frame assets, temporal controls,
  display backpressure, and decode cleanup.

### Absent

- A headless working-grid settings or resolved-geometry type.
- A downsample or block-size setting in Isolate.
- A grid readout.
- A grid overlay input or paint branch in `IsolatePlayer`.
- A native-source-coordinate transform in `IsolatePlayer`.
- Zoom behavior in `IsolatePlayer`.
- An Isolate overlay-hiding shortcut.
- Scientific-settings persistence or reset behavior.
- A general display adapter or overlay framework.

These absences do not require a broad new architecture. The current player and
tab already provide a small place to add this milestone.

## Important correction 1: use the implemented milestone-3 seams precisely

The handoff correctly says milestone 3 must be accepted first. That requirement
is material, not ceremonial.

The current rewrite now contains:

```text
WorkingWindowRequest
ResolvedWorkingWindow
WorkingWindowStream
IsolateSession.snapshot_working_window_request()
```

`WorkingWindowRequest` carries registered identity and half-open temporal
bounds, not source dimensions or spatial tuning. `ResolvedWorkingWindow`
contains probed native dimensions and recorded identity status, but obtaining it
opens a request-local media session. Grid resolution must not open that source.
It should use primitive registered `ActiveAsset.width,height` for immediate UI
geometry, then require a later channel to compare its resolved source dimensions
with its captured grid before combining pixels and cells.

Required disposition:

1. Accept the implemented milestone 3.
2. Keep grid geometry in a Qt-independent application module.
3. Keep one plain grid-settings value in `IsolateTab`.
4. Do not expand the temporal GUI request snapshot into a general settings
   controller.
5. Choose the asset-switch policy, then implement milestone 4.

This review approves the geometry contract and the current package/ownership
seam, subject to the remaining user decisions above.

## Important correction 2: map normalized grid boundaries directly into the current player rectangle

The handoff describes:

```text
working boundary
  -> native source coordinate
  -> existing source-to-display transform
```

The current `IsolatePlayer` does not have that second transform.

It stores:

```text
frame_size
  dimensions of the capped display preview

image_rect()
  the letterboxed widget rectangle in which that preview is painted
```

`frame_size` is not the native active-asset extent. Replacing it with native
dimensions would break `QImage` construction and preview painting.

For the current player, the smallest correct grid projection is:

```text
view_x = image_rect.left + (working_x / work_width) * image_rect.width
view_y = image_rect.top  + (working_y / work_height) * image_rect.height
```

This is equivalent to mapping through native source coordinates because both
spaces cover the complete active asset, but it does not require a new
source-coordinate API.

Required invariants:

- Headless geometry still resolves from native registered `W,H`.
- Overlay projection uses normalized working boundaries.
- The outside grid boundary is exactly `image_rect()`.
- Preview width and height never become scientific working dimensions.
- Letterbox margins remain outside the grid.
- The display preview's even-height rounding cannot shift the normalized grid.

Do not create a `DisplayAdapter` or copy `FrameView` solely to obtain this
mapping. If a small pure helper makes coordinate testing clearer, keep it local
to `IsolatePlayer` or the grid-presentation module.

## Important correction 3: use the native registered extent, not `IsolatePlayer.frame_size`

The current player receives the width-limited preview dimensions from
`IsolateDecodeThread`. On a wide asset, those are deliberately not the native
dimensions.

The headless resolver must receive primitive native source width and height from
the current `ActiveAsset` snapshot. The milestone-3
`WorkingWindowRequest` snapshot does not carry dimensions and must not be opened
merely to obtain them.

For this milestone:

- Registered `ActiveAsset.width,height` are sufficient to resolve visible
  geometry.
- Passing those primitive values into the headless resolver does not require
  the resolver to import the PyQt-bearing `ActiveAsset` module.
- Grid-setting changes must not reprobe or reopen media.
- A later scientific channel must confirm that its working-window resolved
  native dimensions match the resolved grid before combining pixels and cells.

The grid itself does not verify current media bytes or frame decodability. It
must not relabel registered sidecar dimensions as verified current evidence.
Milestone 3 owns source identity and media-fact checks.

If the player successfully displays media whose current dimensions disagree
with the registered asset, that is an asset/media consistency issue, not a
reason for the grid to silently switch to preview dimensions.

## Important correction 4: keep grid settings out of the current temporal `IsolateSession`

The handoff allows an “Isolate-local state/controller” to own the settings.
In the current rewrite, `IsolateSession` is specifically the Qt playback and
temporal-window model. It owns:

- A mutable display `MediaSession`.
- A decode thread.
- Playback timers.
- Current/displayed frame state.
- The looping frame window.

Putting headless grid resolution into that class would make scientific settings
depend on the display/media lifecycle and would make non-Qt reuse less clear.

The smallest current shape is:

```text
headless module
  WorkingGridSettings or equivalent
  ResolvedWorkingGrid or equivalent
  pure resolution and block-bound formulas

IsolateTab
  owns the current plain settings value
  presents controls
  resolves against primitive native dimensions
  passes the resolved grid to IsolatePlayer

IsolatePlayer
  paints an optional resolved grid
```

The names are illustrative. A new QObject controller is not required.

Do not make widget values the only source of truth. Keep one plain settings
value that future channel orchestration can snapshot without reading spinboxes.
The widgets present and edit that value.

Milestone 3 established a temporal request-snapshot method on
`IsolateSession`, not a general Isolate settings owner. It does not supersede
the `IsolateTab` ownership above.

## Important correction 5: define the currently absent asset-switch behavior

The handoff says to use the rewrite's current reset/open-asset behavior, but the
rewrite has no downsample or block settings today. There is no existing behavior
to preserve.

The review recommends the following simplest explicit rule, but it is not
accepted behavior until the user chooses it:

```text
on Isolate-tab construction
  downsample intent = 1.0
  block intent = auto

on active-asset change in the same application session
  retain those requested intents
  resolve a new grid against the new native extent
  clear the old painted grid before showing the new asset

on application restart
  return to defaults
```

This keeps settings Isolate-local, allows parent/child comparison under one
intent, and adds no persistence.

Do not add a sidecar, `QSettings` key, per-asset tuning store, or reset button in
this milestone. Later persistence can deliberately decide whether settings are
global, session-local, or per asset.

Resetting to defaults on every asset switch is the other cheap, coherent policy.
The handoff now exposes this as an explicit implementation gate rather than
silently treating this review's recommendation as a decision.

## Important correction 6: “no worker” means no additional grid worker

The loaded Isolate player already owns `IsolateDecodeThread`. Tests cannot
correctly assert that no thread exists while an asset is open.

The required invariant is narrower:

> Resolving or repainting the grid creates no additional worker and sends no
> additional frame request to the existing display decoder.

Test this through observable behavior:

- Count display-decoder requests before and after a grid change.
- Confirm the count does not increase.
- Confirm the current displayed frame does not change.
- Confirm no working-window source is opened.
- Confirm no grid-specific `QThread`, timer, or asynchronous task is created.

Do not write a brittle process-wide thread-count assertion.

## Important correction 7: add one grid-specific player seam, not a general overlay framework

`IsolatePlayer.paintEvent()` currently paints only:

1. The dark background.
2. A message when no image exists.
3. The display image.

There is no current Isolate overlay abstraction. The Replicates `FrameView` has
region overlays and Shift-to-hide behavior, but it has different interaction
and source-crop responsibilities.

The smallest safe addition is an input such as:

```text
set_grid(resolved_grid_or_none, visible)
```

or equivalent state owned by `IsolatePlayer`, followed by a grid-specific paint
branch after `drawImage`.

Requirements:

- The player receives resolved geometry; it does not resolve scientific
  settings.
- Repainting the grid calls `update()` only.
- Clearing or switching assets clears the old resolved grid.
- No `FrameView` inheritance or shared interactive overlay framework is
  introduced.
- No heatmap, value, selection, or detection layer is anticipated in the API.

Future scientific overlays may eventually justify a presentation model. One
line grid does not.

## Important correction 8: do not invent zoom or overlay shortcuts

The current `IsolatePlayer` has no zoom and no overlay-hiding shortcut.

Interpret the handoff's conditional zoom language literally:

- Test letterboxing and resize, which exist.
- Do not add zoom for grid acceptance.
- Do not copy Replicates' Shift-to-hide shortcut unless separately requested.
- The explicit “Show grid” control is sufficient for this milestone.

Normal Qt device-pixel scaling does not require a new scientific coordinate
system. Paint and test in the player's logical coordinates.

## Important correction 9: grid controls must not inherit `can_loop`

The current temporal controls are enabled with:

```text
can_loop
```

which is false for a one-frame asset.

Working-grid settings are spatial and must remain usable for a valid one-frame
asset. Do not place them in the same enable/disable branch as Play, window
start, or window length.

A suitable current rule is:

```text
grid controls enabled when a valid active asset extent exists
overlay paints when an image is available and Show grid is on
```

This preserves the handoff's one-frame requirement.

If media loading fails, keep the media error visible. A headless grid may still
resolve from registered dimensions, but the UI must not imply that an overlay or
scientific pixel path succeeded.

## Important correction 10: dense-grid suppression should remain local presentation state

The handoff correctly avoids painting thousands of indistinguishable lines. In
the current player, the simplest implementation is:

- Calculate projected horizontal and vertical spacing from `image_rect()`.
- Draw internal lines only when the relevant spacing meets the presentation
  threshold.
- Always keep the resolved numerical readout truthful.
- Optionally draw a short “grid too dense to display” note inside or adjacent to
  the image rectangle.

Do not:

- Change the real grid.
- Increase block size automatically.
- Draw every Nth line without labelling that it is a presentation sample.
- Send resize events back into scientific settings.
- Persist the density threshold.

The threshold is not grid identity. It belongs in player presentation code and
tests.

## Geometry decisions that are safe as written

### Native `1.0` default

The rewrite should not silently discard evidence to make large footage cheaper.
There is no current conflicting downsample default.

### Deterministic working dimensions

The rewrite has no existing dimension resolver, so the handoff's deterministic
Python `round` fallback is available. Pin exact half ties so future code does
not accidentally use `floor`, `ceil`, or a different resize shape.

### Explicit and automatic block intents

An explicit block remains fixed in working pixels. Automatic intent tracks:

```text
max(1, round(64 * s))
```

The resolved integer must remain visible. Preserve the tagged intent separately
from its resolved value.

A simple tagged value or `int | None` can express this. Do not create a class
hierarchy for two cases.

### Ceil-sized grid and partial cells

These formulas are correct:

```text
R = ceil(Hw / b)
C = ceil(Ww / b)
```

and:

```text
area_weight = owned_area / (b*b)
```

Every working pixel remains owned exactly once. No padding enters later
statistics.

### Compact grid representation

Store the regular-grid parameters and derive bounds. This is especially
important at block size one.

The overlay needs at most row/column boundary lines, not one rectangle or Qt
item per cell.

### No pixel resize yet

This increment resolves the dimensions that future area downsampling will
produce. It does not need to decode or resize a frame merely to make the grid
visible.

### No new benchmark surface

The grid readout is geometry, not a performance estimate.

Do not add overlay timing to `sieve media benchmark`; it answers a different
question. A disposable paint regression measurement is sufficient for
implementation work unless the project later accepts a reusable GUI diagnostic
surface.

## UI placement clarification

The current Isolate layout has:

- A left player.
- A right channel placeholder.
- A transport/window row.
- A timeline and status line.

The handoff should not force the grid controls into the empty channel container
or replace “No channels added yet.” That area is still reserved for the first
real channel.

Add a compact “Working grid” control/readout area using the current layout
without redesigning the splitter. Exact placement is user-facing and should be
validated manually.

The visible information can remain small:

```text
Downsample 1.000
Block auto (64 working px)
Show grid
Source W x H -> Working Ww x Hw -> Grid R x C
partial right/bottom dimensions when present
```

Do not bring over the oracle's cost dialog, normalization controls, channel
selector, clip-routing controls, or processing status.

## Testing disposition

### Headless geometry tests

The handoff's proposed tests are appropriate:

- Downsample validation and deterministic resolution.
- Explicit versus automatic block intent.
- Ceil-sized rows and columns.
- Exact partial bounds and weights.
- Sum of owned area equals `Ww*Hw`.
- Compact representation.
- Normalized coordinate projection.

No video fixture, PyQt import, or media session is needed for these tests.

### Current GUI tests

Add focused tests for:

- Grid settings enabled independently of `can_loop`.
- One-frame asset support.
- Native asset extent used even though player preview is capped.
- Normalized grid boundaries mapped into `image_rect()`.
- Letterbox margins excluded.
- Resize preserves alignment.
- Show/hide updates without decode.
- Asset switch clears old geometry and resolves new geometry.
- Dense-grid suppression stays presentational.
- Existing `channels_empty` placeholder remains.
- No extra display-frame request occurs on setting or visibility changes.

The current player has no zoom, so no zoom test belongs in this milestone.

Use direct coordinate assertions as the primary proof. A grabbed image may be a
supplemental visual regression, not the only alignment test.

### Existing regressions to retain

Keep passing:

- Player frame identity and backpressure.
- Timeline/window behavior.
- Letterboxing.
- One-frame asset display.
- Asset-switch stale-result rejection.
- Decoder cleanup.
- Active-tab shortcuts.

Run GUI tests with:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
```

## Risk classification

### Potentially expensive if interpreted incorrectly

1. Building a general display adapter or overlay framework.
2. Copying `FrameView` interaction into `IsolatePlayer`.
3. Putting scientific geometry resolution into the media-owning
   `IsolateSession`.
4. Allocating one Python or Qt object per block.
5. Persisting settings before their later ownership is known.
6. Starting milestone 4 before the implemented milestone-3 package seam is
   accepted.

### Correctness risks if left ambiguous

1. Using capped preview dimensions as native source dimensions.
2. Mapping source integers through multiple rounded transforms.
3. Resetting or retaining settings unpredictably on asset switch.
4. Treating registered dimensions as verified current media facts.
5. Letting partial edge cells count as full cells.
6. Coupling grid control availability to temporal looping.
7. Letting a grid repaint trigger another media decode.

### Cheap to extend later

1. Settings persistence.
2. A reset action.
3. Zoom-aware denser grid display after zoom exists.
4. Additional presentation overlays.
5. A supported GUI performance diagnostic if users need one.

### Correctly deferred

1. Working-pixel production.
2. Block reducers and values.
3. Channels and plots.
4. Normalization.
5. Scientific workers and supersession.
6. Result invalidation and coverage.
7. Caches, recipes, whole-asset execution, and exports.

## Current implementation issues observed during this review

These issues were found at rewrite commit `4b09232`. The milestone-3 changes
through current commit `01b256f` did not address them. They are not caused by
the working-grid handoff and should not be “fixed” by moving their
responsibilities into grid code.

### High: an obsolete decode error can defeat the latest valid request

`IsolateDecodeThread.run()` treats successful and failed obsolete requests
differently.

After a successful read, it checks whether a newer frame is pending and discards
the obsolete result:

```text
decode old frame
new request arrives
pending request exists
discard old success
```

In the `SieveError` path, it emits `decode_failed` without checking whether a
newer request is pending. The signal contains the asset generation but no frame
or request identity.

`IsolateSession._decode_failed()` then closes the complete media session for any
error in the current asset generation.

The resulting failure path is:

```text
old seek A is in flight
new seek B becomes current
A fails
old error is emitted as current
Isolate closes the decoder and media
B never runs
```

This violates the current latest-request policy even before scientific workers
exist.

Add a regression in which an old in-flight frame fails after a newer frame is
queued. The old error must be discarded or explicitly associated with its frame
and rejected by the session. Do not let it overwrite or close the newer request.

The Replicates decode thread has the same missing pending-request check in its
error branch, although its receiver currently reports the stale error rather
than closing the complete session.

### High: decoder-thread shutdown has a start/interrupt race and ignores timeout

Both GUI decode owners follow this pattern:

```text
thread.stop()
thread.wait(3000)
discard thread reference
close MediaSession
```

They do not inspect whether `wait(3000)` succeeded.

There is also a race:

1. The worker removes a request from its condition mailbox.
2. The GUI calls `stop()`.
3. `MediaSession.interrupt()` sees no decoder process yet.
4. The worker starts FFmpeg after that interrupt.
5. The GUI waits three seconds and then continues even if the worker remains
   inside the read.

The GUI can then close the same media session and schedule the `QThread` for
deletion while its `run()` method is still active.

The existing interaction diagnostic reports that one sampled close ended with
`decoder is None`; that only proves the owner dropped the reference. It does not
prove the thread stopped or that the FFmpeg process exited.

Required hardening should:

- Make stop state observable immediately before decoder creation.
- Serialize or otherwise make decoder start/interrupt ownership safe.
- Treat a failed `wait()` as a real cleanup failure rather than continuing
  silently.
- Verify the worker stopped and the FFmpeg process exited.
- Apply the same lifecycle rule to Replicates and Isolate.

### High scientific-integrity risk: derivation can trust a stale content hash

For an existing asset, `ActiveAssetController.open_asset()` reads the sidecar but
does not verify that the current media bytes still match
`media.content_sha256`. That may be acceptable for inexpensive browsing if the
identity is clearly treated as recorded.

The more serious path is derivation:

- `DerivationService.plan()` calls `AssetService.verify(..., level="metadata")`.
- Metadata verification compares file size and dimensions but does not hash the
  current media.
- The derivation then uses the recorded sidecar content hash as parent lineage
  identity.

If media is replaced while retaining the same size and dimensions, a new child
can be attributed to a parent content hash that was not the bytes actually
decoded.

Before scientific derivation or processing publishes lineage, require an
accepted content-verification policy or explicitly record that identity was only
metadata-checked. Do not present the stored SHA-256 as verified current content
when it was not recalculated.

This does not mean every small viewer open must hash an 11 GB file. Browsing,
processing admission, and published lineage may legitimately use different
verification costs, but their status must be explicit.

### Medium: opening an unregistered video performs a full hash on the GUI thread

`MainWindow.open_asset()` calls `ActiveAssetController.open_asset()` directly.
For a raw video without a sidecar, the controller calls
`AssetService.initialize()`, which synchronously:

- Probes the media.
- Calculates SHA-256 over the complete file.
- Writes the sidecar.

All of this occurs before `open_asset()` returns to the Qt event loop. Large raw
videos can therefore make the application appear frozen with no progress or
cancellation.

Either require deliberate prior registration or move first-time registration to
observable cancellable work. Do not hide the cost while retaining synchronous
GUI execution.

Registered assets still incur two synchronous `MediaSession` probes because
both workflow tabs eagerly open independent sessions. That two-tab lifecycle is
already documented as an unresolved product decision; benchmark it before
changing it, and do not share their mutable decoder merely to remove duplication.

### Medium: the display preview can distort aspect ratio

`MediaSession.scaled_dimensions()` rounds the scaled height and then forces it
to an even integer while keeping width fixed.

The current regression test pins:

```text
native 18 x 14
max width 9
display result 9 x 8
```

The source aspect ratio is about `1.286`; the display result is `1.125`.

The output is raw `rgb24`, so an even height is not generally required by the
output pixel format. The rule can visibly stretch some capped previews and makes
source-to-display assumptions less obvious.

Confirm whether the active FFmpeg scale path has a real even-dimension
constraint. If not, preserve aspect ratio instead of carrying an encoder-style
constraint into raw display output. Update the test to pin aspect preservation,
not the distortion.

The working-grid review's normalized overlay mapping will align with whichever
preview is painted, but it should not be used to legitimize a distorted preview.

### Medium: the supported media benchmark underreports provenance

The accepted product diagnostic exists in an importable package and has human
and JSON CLI surfaces. Its basic tests pass.

Its result still falls short of the accepted reporting decision in several
places:

- `content_sha256` is reported without saying it is recorded rather than
  content-verified for this run.
- `frame_count_status` collapses packet, container, and duration-derived counts
  into `estimated`.
- The package/SIEVE version is absent from environment metadata.
- Default human output omits source codec, native dimensions, rational rate,
  sample counts, and software/backend environment even though the decision says
  every supported estimate identifies them.
- Tests assert only a small subset of the required report and do not pin those
  provenance fields.

The numerical estimate is honestly scoped to media decode and excludes GUI
paint and science. Preserve that strength while completing the accepted result
schema and human report.

## Validation performed for this review

The current focused suites pass:

```text
tests/test_working_window.py
tests/test_isolate_player.py
tests/test_domain_and_services.py

49 passed in 10.46s
```

Passing tests do not negate the findings above; the stale-error path,
start/interrupt race, timeout result, same-size stale-content lineage case, and
aspect-preservation expectation are not covered by those tests.

## Recommended disposition

The geometry and scope in `4-Working-grid.md` are safe.

Before implementation:

1. Accept the implemented milestone 3.
2. Accept the updated handoff and review.
3. Choose retain or reset behavior for grid intents across asset switches.
4. Preserve the current-player and ownership clarifications above.

The smallest implementation for the current rewrite should remain:

```text
plain Qt-independent settings
    -> compact resolved grid

IsolateTab
    -> owns session-local settings and readout
    -> passes resolved geometry to the player

IsolatePlayer
    -> maps normalized boundaries into image_rect()
    -> paints an optional bounded grid
```

No wider architectural rewrite is needed. After these corrections and the
post-milestone-3 divergence refresh, the handoff is suitable for implementation.
