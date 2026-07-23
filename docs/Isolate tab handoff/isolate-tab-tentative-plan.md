# Isolate tab — tentative handoff plan

Status: non-binding roadmap.

This file lists plausible small increments for rebuilding the oracle Processing
tab as the rewrite's Isolate tab. It is not an implementation contract for all
of them.

Each increment still requires its own user-approved handoff. The order and
boundaries may change after hands-on validation or when the rewrite's current
implementation reveals a cleaner seam. Completing one increment does not
authorize beginning the next.

Existing handoffs:

1. `1-Build-the-player.md`
2. `2-Media-service-handoff.md`
3. `3-Working-window.md`
4. `4-Working-grid.md`
5. `5-First-channel.md`

## Current handoff state and gates

The current rewrite snapshot returned to the oracle was commit `01b256f`.
Milestone 3 was implemented in rewrite commit `5bffca7`, but the returned
material still describes it as awaiting user acceptance.

The current implementation sequence is gated as follows:

```text
accept milestone 3
  -> choose retain/reset behavior for grid intent across asset switches
  -> implement and accept corrected milestone 4
  -> refresh milestone-5 divergence against that implementation
  -> revise, implement, and accept milestone 5
```

Milestone 5's known revision gates are:

- Choose canonical encoded luma or a distinctly named post-`rgb24` Rec.601
  intensity representation.
- Add explicit retained-result memory admission before opening scientific media.
- Compose the accepted working-window source outcome instead of duplicating its
  lifecycle states.
- Define a race-safe scientific-worker creation, cancellation, close, and
  termination handshake.
- Pin numerical conformance to the accepted scientific representation rather
  than raw equality with the oracle's historical grayscale scale.

`.isolate-state-divergence.md` separately tracks stale display-decode errors,
decoder shutdown, recorded-versus-verified content identity, first-registration
GUI blocking, preview aspect distortion, and media-benchmark provenance. Those
findings do not change the numbered feature order. A finding becomes a milestone
prerequisite only when that milestone touches the affected boundary; in
particular, milestone 5 must not copy the unsafe decoder lifecycle and must
preserve identity-verification status.

## Candidate next increments

### 3 — Define the working window (implemented; user acceptance pending)

Establish the non-Qt request for loading the selected frame window:

- Stable resolved asset/content reference.
- Half-open frame range.
- Rational timebase resolved from authoritative asset metadata.
- One explicit native-resolution plane supported by the current media service.
- Synchronous bounded delivery with decoded/delivered span and source outcome.
- GUI generations and latest-request publication kept outside the reusable
  headless request.

Do not compute a scientific channel yet.

The rewrite reports that this narrow pixel-delivery seam is implemented with
Qt-free requests, resolved source facts, request-local native-resolution
`rgb24` streaming, explicit outcomes and provenance, and a pure GUI request
snapshot. Decoded pixels must not be reported as scientifically examined
coverage. Acceptance remains a user gate before milestone 4 begins.

### 4 — Build the working grid (handoff and review refreshed; implementation gated)

Add only the spatial working geometry:

- Explicit downsample control, defaulting to native `1.0`.
- Working-pixel block-size control with visible automatic source-footprint
  tracking.
- Source-pixel and working-pixel dimensions.
- Compact ceil-sized geometry with fractionally weighted partial right and
  bottom blocks.
- Optional block-grid overlay on the player that does not decode the working
  window.

Do not add pixel preprocessing, a channel, or detection.

Downsample and block size remain separate: downsample changes the pixel evidence
and compute cost, while block size changes spatial aggregation. The rewrite
completed the required post-milestone-3 divergence refresh. Before
implementation, the user must accept milestone 3 and choose whether downsample
and block intent are retained or reset across active-asset changes.

### 5 — Add the first channel (handoff and review written; revision pending)

Add one real, inexpensive channel end to end, probably block-mean intensity:

- Compute it headlessly over the looping window.
- Display its time/block data in one channel panel.
- Synchronize its cursor with the player.
- Cancel or supersede it when the window or upstream settings change.

Do not create a channel registry or general add/remove framework yet. Let the
first real channel expose the smallest useful panel and data contract.

Before implementation, revise the handoff using
`5-First-channel-review.md`, then refresh its rewrite-side divergence against
the accepted milestone-4 types, ownership, geometry, worker, and player seams.
The scientific representation, result-memory admission, source-outcome
composition, and worker lifecycle must be resolved explicitly.

### 6 — Add normalization

Add the preprocessing modes actually needed by the first channels:

- `off`.
- Per-frame z-score.
- CLAHE only if it remains appropriate for this increment.

Requirements:

- Apply normalization to scientific input pixels, never to a display raster.
- Record which normalization produced the displayed channel.
- Recompute the selected window when normalization changes.
- Compare results against small oracle fixtures.

CLAHE may deserve a separate handoff because of its known signal and boundary
artifacts.

### 7 — Add change energy

Port the oracle's `change` / `J_tt` path:

- Correct frame pairing.
- Explicit temporal alignment and validity.
- Block reduction.
- Selected-channel-only computation.
- A channel panel and current-frame overlay.

This is the first channel that visibly responds to behavior and exercises
multi-frame channel state.

### 8 — Add static value filtering

Add a value band directly to a channel without requiring a frequency transform:

```text
channel
  -> value band
  -> selected blocks
```

Display:

- The selected value interval.
- Selected blocks on the current frame.
- A per-frame selected-block count.

Do not add Morlet processing in this increment.

Making this path real before the spectral path helps prevent the Isolate
implementation from assuming that every valid signal must have a frequency band.

### 9 — Add the Morlet view

Port the oracle spectral representation:

```text
channel
  -> Morlet power
  -> frequency band
  -> frequency-band power
```

Add:

- The windowed scalogram.
- The draggable frequency band.
- Cone-of-influence display/validity behavior.

Stop before implementing the complete detector.

### 10 — Add value density and block highlighting

Port:

- Per-block band-power density.
- A draggable value band.
- Current-frame in-band block highlighting.
- Cached display layers where measurement proves them useful.

Require numerical agreement between the preview computation and the oracle
contract on small fixtures. Performance changes remain benchmarked hypotheses.

### 11 — Add the windowed detector

Add the remaining preview chain:

```text
selected blocks
  -> block count
  -> centered or trailing detection window
  -> count band
  -> gate
  -> largest spatial clump
```

The shared formulas remain headless and independent of widget visibility.

At the end of this increment, the looping window should reproduce the essential
oracle signal-isolation workflow, while both static and Morlet representations
remain possible.

### 12 — Build the result timeline

Extend the bottom timeline to display accumulated result state:

- Unexamined.
- Examined and quiet.
- Detected.
- Current versus stale settings.
- Click-to-seek and looping-window repositioning.

Initially, the timeline may accumulate only the windows the user has actually
previewed. It must not present uncovered footage as negative.

### 13 — Process the whole asset

Add whole-asset execution through the same headless channel and detector
functions used by the preview:

- Continuous execution.
- Structured progress.
- Cancellation.
- Segment-by-segment result delivery.
- Explicit validity and cone-of-influence handling.
- Preview/whole-run numerical agreement.

Do not add alternative processing schedules until the continuous path is working
and validated.

### 14 — Add processing plans

After continuous whole-asset processing works, add:

- From the current position.
- Binary-split sampling.
- Uniform sampling.
- Fill uncovered or stale gaps.
- Skip coverage produced under the current settings.

Partial runs must preserve finished work and report honestly which footage
remains unexamined.

### 15 — Save and restore isolation

Add settings and result persistence only after the working state and
invalidation behavior are known:

- Scientific settings separate from GUI presentation state.
- Atomic writes.
- Active asset and implementation identity.
- Coverage and validity.
- Retained intermediates where their cost is acceptable.
- Explicit stale/current compatibility.

This increment does not own validation, presentation rendering, derived assets,
or detection stacking.

### 16 — Expose the headless recipe

Make the proven isolation settings executable noninteractively:

- Validate settings.
- Resolve them against one active asset.
- Run the same computation from CLI/HPC.
- Emit structured progress and results.
- Compare GUI-launched and direct headless results.

Do not add consumer-specific validation or rendering options to the isolation
recipe. Later consumers should operate on stable result artifacts.

## Current recommended near-term path

The most useful sequence after the player and media-service work currently
appears to be:

```text
accept working window
  -> choose grid asset-switch policy
  -> implement and accept working grid
  -> revise and implement intensity
  -> normalization
  -> change energy
  -> static value filtering
  -> Morlet representation
```

This order remains tentative. Its purpose is to keep each increment visible and
testable, make static filtering real before spectral assumptions harden, and
delay broad recipe/CLI generalization until at least two genuinely different
isolation paths exist.
