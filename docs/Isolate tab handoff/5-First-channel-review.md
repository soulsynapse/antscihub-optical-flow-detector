# Review — 5 Add the first Isolate channel

Reviewed on `2026-07-23` against:

- `5-First-channel.md`.
- The rewrite state returned in `.isolate-state-divergence.md`, reported at
  commit `01b256f`.
- `docs/rewrite-handoff-v2.md`.
- `docs/sieve-scientific-computation-contract.md`.
- The oracle's current intensity extraction and live-channel behavior.

Status: architectural, scientific-contract, and forward-looking review of
`5-First-channel.md`.

The rewrite repository source is not present in this oracle checkout. Current
rewrite facts below are therefore limited to the returned divergence report and
the milestone-4 review; package names and exact current code paths must be
confirmed by the rewrite before implementation.

## Verdict

The increment boundary is good.

Adding one inexpensive block-time channel before normalization, temporal
channels, transforms, detection, persistence, and whole-asset execution is the
right way to expose the first real scientific seam without prematurely building
the complete architecture.

The handoff's strongest decisions are:

- It consumes the accepted headless working-window source rather than the
  player preview.
- It treats one selected asset as the complete processing domain.
- It reuses the accepted working grid and checks probed source dimensions
  before combining pixels with cells.
- It preserves downsample and block size as separate scientific settings.
- It retains partial edge cells and their actual area.
- It keeps intensity frame-local and avoids tensor, flow, and previous-frame
  work.
- It introduces one explicit computation action rather than silently running
  scientific work on every edit.
- It keeps GUI job tokens separate from scientific identity.
- It makes obsolete progress, success, failure, and cancellation signals
  harmless.
- It keeps the player clock authoritative for the channel cursor.
- It publishes only complete current-window data in the milestone-5 GUI.
- It does not introduce a channel registry, cache, graph executor, persistence,
  CLI recipe, or whole-asset execution prematurely.

Those choices match the rewrite's one-asset, headless-computation,
explicit-execution, and bounded-lifecycle intent.

The handoff is not yet ready to implement unchanged. It needs five material
corrections:

1. Resolve whether the channel is source encoded luma or a named post-`rgb24`
   Rec.601 representation.
2. Add explicit result-memory admission; bounded pixel batches alone do not make
   the computation memory-bounded.
3. Reuse the accepted working-window outcome/lifecycle rather than creating a
   parallel source outcome hierarchy.
4. Define a safe worker stop handshake without inheriting the rewrite's known
   decoder start/interrupt race.
5. State the correct oracle comparison target, because the current oracle
   intensity values are not numerically identical to the new scientific
   contract.

After those corrections and a post-milestone-4 rewrite-side divergence refresh,
the increment is suitable for implementation.

## Current-rewrite evidence available to this review

### Reported present

- A Qt-free `application.working_window` package.
- `WorkingWindowRequest` with registered identity and absolute half-open
  temporal bounds.
- `ResolvedWorkingWindow` with probed native-resolution dimensions and extent
  provenance.
- Request-local synchronous `WorkingWindowStream` delivery.
- Immutable native-resolution `rgb24` batches.
- Explicit source outcomes.
- `IsolateSession.snapshot_working_window_request()` as a pure GUI adapter.
- A separate player `MediaSession` and display decoder.
- One letterboxed `IsolatePlayer.image_rect()`.
- An empty right-hand channel placeholder.
- Registered `ActiveAsset` dimensions and recorded content identity.

### Reported absent before milestone 4

- Accepted working-grid settings and resolved geometry.
- A scientific GUI worker in Isolate.
- A GUI supersession contract for scientific results.
- A channel result type or channel panel.
- Normalization or scientific pixel preprocessing in the rewrite.
- Persistence, result artifacts, recipes, and whole-asset analysis.
- A general channel registry or computation graph.

Milestone 5 must be reviewed again against the actual accepted milestone-4
implementation. The current report cannot certify its type names, settings
owner, geometry snapshot method, asset-switch policy, or dimension-rounding
implementation.

## Important correction 1: choose the intensity representation honestly

This is the largest scientific issue in the handoff.

The handoff currently says:

```text
I_raw = (0.299*R + 0.587*G + 0.114*B) / 255
conversion id = sieve.intensity.rgb601_float.v1
```

That is deterministic and potentially valid as a named channel. It is not
automatically the same quantity as the governing scientific contract's
canonical encoded luma.

The contract requires source color interpretation to resolve:

- Declared color matrix.
- Declared range.
- Decoder conversion.
- Output bit depth.
- An explicit policy when source color metadata is ambiguous.

For BT.709, BT.2020, limited-range YUV, or material whose metadata FFmpeg
interprets differently, decoding to RGB and then applying fixed Rec.601 weights
does not generally reconstruct the source encoded-luma codes. Calling both
quantities simply `intensity` would make later oracle comparisons and
cross-asset results look more equivalent than they are.

Also correct the terminology:

```text
native-resolution rgb24
```

is accurate. `rgb24` is a decoder-produced representation, not necessarily a
codec-native or source-native plane. Avoid phrases such as “native RGB plane”
when they could imply the latter.

### Acceptable path A — canonical encoded luma

If the rewrite can expose encoded luma with explicit matrix/range/conversion
provenance through a narrow extension of the accepted working-window source,
then milestone 5 may implement the scientific contract's canonical
`intensity`.

This path must:

- Define exactly which decoder output plane/code range is delivered.
- Preserve source matrix and range resolution.
- Normalize legal black/white codes to `[0,1]`.
- Clip declared excursions as specified by the scientific contract.
- Make ambiguous metadata an explicit policy/error rather than a backend
  default hidden in provenance.
- Keep native-resolution delivery and request-local source ownership.

Do not build a general plane registry merely to add this one representation.

### Acceptable path B — named post-RGB representation

If milestone 3 deliberately supports only `rgb24`, keep the handoff's fixed
formula but treat it as a distinct named representation:

```text
sieve.channel.rgb601_intensity.v1
```

or an equivalently explicit id.

Then:

- Do not call it the canonical source encoded-luma node.
- State that FFmpeg/media-service color conversion occurred first.
- Retain that conversion's available metadata and implementation identity.
- Treat missing source conversion metadata as known provenance uncertainty.
- Use the same named representation as the upstream basis for later
  normalization and change-energy work unless a deliberate profile change is
  accepted.
- Update the scientific profile or add a named profile/node contract before
  presenting it as the built-in reference `intensity`.

Path B is the smaller milestone-5 implementation. Path A is closer to the
current scientific contract. The user/rewrite must choose before coding; the
implementation must not drift between them based on which FFmpeg or image
conversion helper is convenient.

### Required tests

Primary-color unit tests are necessary but insufficient. Add a small
color-managed fixture or synthetic code-value test that distinguishes:

- Full-range Rec.601.
- Limited-range input.
- BT.709 versus BT.601 interpretation.
- Recorded conversion metadata from content-verified conversion metadata.

The result must identify which interpretation was actually used.

## Important correction 2: make result memory explicitly bounded

The handoff correctly bounds decoded and working pixel batches, but then allows:

```text
values shaped T x R x C, float32
```

for an arbitrary headless working-window request.

That result uses:

```text
value_bytes = T * R * C * 4
```

before array/container, raster, and transient display overhead. A bounded
decoder with an unbounded retained result is not a memory-bounded computation.
The rewrite's governing direction explicitly requires large windows and whole
assets to remain streamable without requiring full-video arrays in memory.

Milestone 5 does not need a cache, memory map, artifact store, or chunked result
format. It does need admission.

Required disposition:

1. Compute the exact result shape and estimated retained bytes before starting
   the job.
2. Apply one explicit accepted in-memory result budget or a pre-existing
   Isolate-window bound that is demonstrably sufficient.
3. Refuse the job before decoding when the result cannot be retained safely.
4. Report the requested shape, estimated bytes, budget, and refusal reason.
5. Keep the budget an execution/resource policy, not scientific identity.
6. Include the display raster and any unavoidable copy in the practical GUI
   admission estimate, or document why its bounded representation does not
   scale with the full scientific array.

For example:

```text
T=600
R=7
C=8
values=134,400 bytes
```

is comfortably small. A very long headless request at a dense grid may not be.

The accepted behavior should be:

```text
small selected window
  -> retain complete T x R x C values

request exceeding in-memory result budget
  -> structured resource refusal before source open
```

Later whole-asset execution can stream into a deliberate artifact or bounded
accumulator. Milestone 5 should not invent that mechanism.

Add tests for:

- Exact byte estimation.
- Boundary exactly at the budget.
- One byte/element over the budget.
- Refusal before `WorkingWindowStream` or `MediaSession` creation.
- No integer overflow in shape/byte arithmetic.
- GUI status that distinguishes resource refusal from decode failure.

## Important correction 3: compose existing source outcomes and keep one small result envelope

The handoff proposes:

```text
completed
cancelled
source_truncated
source_failed
computation_failed
```

The rewrite already reportedly has explicit `WorkingWindowStream` outcomes.
Milestone 5 should not create a second, subtly different source-state machine
that copies truncation, cancellation, and failure semantics.

The channel layer needs only to distinguish:

```text
source outcome
channel computation outcome
processed channel span
```

Required behavior:

- Preserve or embed the exact accepted source outcome.
- Add channel-stage failure only for conversion, downsample, reduction, or
  result assembly failures.
- Do not translate a specific source failure into a lossy channel enum/string.
- Derive the processed channel prefix from frames actually reduced, not merely
  frames delivered by the source.
- Require source completion plus exact processed coverage for a complete GUI
  result.
- Close the accepted stream through its established lifecycle; do not add a
  competing close owner.

The first result also establishes fields later panels will naturally need:

```text
channel id
units
absolute frame span/timebase
grid geometry
values T x R x C
validity/processed span
scientific settings/provenance
```

It is reasonable for the concrete type to remain `IntensityResult`. Do not add
a channel registry or general graph. But avoid naming fields or panel APIs so
specifically that normalization or change energy would need to redefine
absolute time, grid order, units, validity, or provenance.

This is a small stable data envelope, not a speculative framework.

## Important correction 4: define the scientific worker handshake before copying a QThread pattern

The handoff correctly rejects forceful termination and requires newest-request
publication. That is necessary but not sufficient.

The rewrite already reports a display-worker race:

```text
GUI requests stop
interrupt sees no FFmpeg process yet
worker starts FFmpeg afterward
timed wait expires
owner drops reference and closes shared state
```

Milestone 5 must not reproduce that race with a request-local scientific source.

Before implementation, the rewrite must identify:

- Which thread creates and exclusively owns `WorkingWindowStream`.
- How cancellation becomes visible before source/media construction.
- Whether a blocked media read can be interrupted safely from another thread.
- Which thread closes the stream.
- Which signal proves terminal outcome and resource release.
- How the GUI verifies thread termination before starting a superseding job or
  destroying the owner.
- What happens when shutdown exceeds its expected bound.

The smallest safe ownership is:

```text
GUI
  owns job token and cancellation request

worker
  exclusively creates, consumes, and closes request-local stream
  checks cancellation before source creation and between bounded operations
  emits exactly one terminal outcome after close

GUI
  starts newest pending job only after terminal outcome and verified thread exit
```

If media interruption must occur cross-thread, the media service must already
make that operation safe. Do not reach into a worker-owned decoder through a
new private GUI backchannel.

The unrelated Replicates/Isolate display-worker defects do not have to be fixed
inside milestone 5, but the new worker may not copy their lifecycle. If one
small reusable lifecycle primitive is genuinely required to make all three safe,
report that scope expansion before implementation instead of hiding it in the
channel patch.

Required race tests:

- Cancel before worker entry.
- Cancel after worker entry but before stream construction.
- Cancel while source construction is in progress, if reachable.
- Cancel during a bounded read.
- Supersede while the old job is closing.
- Close Isolate during each phase.
- Terminal signal occurs only after source close.
- Failed thread termination is retained as an owned cleanup failure; the thread
  reference is not silently discarded.

## Important correction 5: define oracle agreement against the right contract

The oracle provides valuable behavior and performance evidence:

- Intensity is frame-local.
- Intensity-only planning avoids tensor/flow work.
- Area downsampling precedes block reduction.
- Blocks average owned pixels and retain partial edges.
- Downsample and block size remain separate.
- Absolute frame alignment matters.

The oracle's current implementation is not a direct numerical gold standard for
the handoff as written.

Current oracle details include:

- `core.preprocess.Preprocessor` generally produces grayscale in an
  approximately `[0,255]` scale, not canonical `[0,1]`.
- OpenCV conversion operates on BGR and uses its own conversion implementation.
- The optimized block reducer accumulates some means in `float32`.
- The current tensor path has historical atlas/replicate compatibility behavior
  that the single-asset rewrite must not inherit.

Therefore:

- Use the scientific contract or the newly accepted named RGB representation as
  the numerical reference.
- Use the oracle for structural invariants and carefully labelled comparative
  fixtures.
- Do not require raw equality to current oracle `intensity` arrays without an
  explicit unit/conversion normalization.
- If an oracle fixture is used, state exactly which source representation,
  conversion, scale, and tolerance make the comparison meaningful.
- Do not port atlas padding/cropping fallback behavior into the rewrite.

The handoff should say explicitly whether conformance is:

```text
exact
```

for integer geometry, spans, masks, weights, and outcomes, and:

```text
within the scientific contract's accepted floating tolerance
```

for area resampling and means. Backend equivalence should not be defined by
vague visual similarity.

## Important correction 6: preserve future normalization and temporal channels without implementing them

The handoff correctly defers normalization and change energy. A few boundaries
must remain forward-compatible:

- Normalization belongs after intensity/downsample and before block reduction.
  Do not put the only reusable seam after block means, because per-frame z-score
  must see the complete working pixel frame.
- `normalization=off` is part of the milestone-5 scientific settings. It should
  not become an intensity-specific hardcoded fact that later requires changing
  result identity semantics.
- Intensity output is valid at frame 0 and needs no previous frame.
- A later change channel describes `t-1 -> t`, is indexed at `t`, and may need
  input context outside the owned output span. Do not turn intensity's
  one-input-frame/one-output-frame convenience into a universal channel
  interface.
- Pixel input span, channel output span, and owned result span should remain
  conceptually distinct even when all three happen to match for intensity.
- The first panel's absolute-frame axis and result envelope should tolerate
  later channels with invalid/context-dependent boundary samples.

These are boundary cautions only. Do not implement normalization modes, context
fetching, general validity masks, or a channel graph in milestone 5.

## Important correction 7: keep scientific values separate from panel contrast

The fixed `[0,1]` scientific scale is correct and must remain in the result.
However, a literal fixed linear black-to-white raster may make small natural
variations hard to inspect.

For milestone 5, the simplest accepted presentation can remain a fixed linear
`[0,1]` map. The review recommends making the separation explicit:

```text
scientific value
  always stored in canonical channel units

presentation transfer
  maps those values to display colors only
```

Do not normalize or autoscale the scientific array to improve contrast.
Conversely, do not make a later presentation-only contrast control part of
scientific identity or trigger recomputation.

The panel should label:

- Channel id/name.
- Units or normalized interval.
- Presentation mapping.
- Absolute frame/time axis.
- Row-major block mapping.

The proposed time-by-flattened-block raster is acceptable for the first
increment. It is compact, makes every block inspectable, and does not prejudge
the later spatial-selection UI. The hover readout is important because a flat
row number is otherwise spatially opaque.

## Headless and GUI ownership review

The intended dependency remains:

```text
accepted WorkingWindowRequest
+ accepted ResolvedWorkingGrid
+ accepted intensity representation/settings
                |
                v
Qt-independent bounded channel computation
                |
                v
immutable block-time result
                |
        queued result/progress signals
                |
                v
Isolate-owned current-result state
                |
                v
one presentation panel
```

Keep these rules:

- The headless computation imports no PyQt.
- The worker adapts a synchronous computation; it does not move scientific
  formulas into GUI code.
- `IsolateTab` or its current rewrite equivalent owns the current scientific
  request/result because milestone 4 reportedly keeps spatial intent there.
- `IsolateSession` remains the temporal player/media owner unless the accepted
  milestone-4 implementation establishes a narrower immutable combined
  snapshot.
- The panel consumes an immutable result and the existing position signal.
- The panel does not read widget state to reinterpret an old result.
- Tabs do not call one another directly.
- The player decoder and scientific decoder remain independent request-local
  media lifecycles.

If milestone 4 introduces a small immutable Isolate scientific-settings
snapshot, milestone 5 should compose it. Do not move mutable widgets,
`ActiveAsset`, or `IsolateSession` into the Qt-free computation merely to avoid
copying primitive facts.

## Supersession and invalidation review

The handoff's invalidation set is correct:

```text
asset identity
temporal window
working dimensions/downsample
block intent/resolved grid
intensity representation
normalization
implementation identity
```

The following do not invalidate scientific results:

```text
playhead position
panel resize
tab selection
grid overlay visibility
presentation-only color mapping
```

Two clarifications:

1. Compare immutable captured keys, not the current widgets, when accepting a
   result. A token prevents stale publication, while the captured key makes the
   reason inspectable and testable.
2. Invalidation should detach the panel from the old result immediately, but
   the worker still owns its old immutable inputs until it terminates. Do not
   mutate or clear objects the worker is reading.

The “newest pending compute starts only after the old worker stops” rule is
appropriate. A single replaceable pending request avoids both concurrent
scientific decoders and a queue of obsolete work.

## Coverage and validity review

The handoff correctly refuses to call source delivery scientific examination.
Once a frame has passed:

```text
RGB validation
-> accepted intensity conversion
-> area downsample
-> block reduction
```

it is processed for this channel.

Keep:

- Requested source span.
- Delivered source span/outcome.
- Processed channel span.
- Complete-current status.

Do not add:

- Quiet/detected semantics.
- Negative-result claims.
- Whole-asset coverage.
- Current-versus-stale timeline painting.
- General detector validity.

For milestone 5's finite `uint8` input and no masks, one valid decoded frame
should produce a finite value for every nonempty grid cell. If path A introduces
an encoded-luma plane with invalidity or ambiguous metadata, its admission and
validity behavior must be explicit rather than silently dropping pixels.

## Performance review

Intensity is deliberately the cheapest useful end-to-end channel. A correct
implementation should not invoke:

- Previous-frame decode context.
- Spatial derivatives.
- Tensor products or integration.
- Flow solve.
- Morlet transforms.
- Detection.
- Other channels.

The handoff's optimization suggestions are appropriately labelled as hints.
One additional caution is needed:

> Do not decode a complete native-resolution batch, convert the complete batch
> to `float64`, retain it, then allocate complete float working and block
> batches simultaneously merely because each individual container is bounded.

Peak memory depends on simultaneous live buffers, not only final result size.
Measure or calculate:

```text
decoded rgb batch
+ conversion working buffer
+ resized working buffer
+ reduced block batch
+ retained result
+ display copy
```

Prefer frame-at-a-time processing inside each bounded source batch if batch-wide
vectorization causes a large peak without a demonstrated throughput benefit.

Do not add a reusable throughput estimate to this milestone unless it follows
the accepted benchmark-diagnostic contract. Development timing in tests or
profiling remains appropriate.

## Automated-test additions required by this review

In addition to the handoff's tests, require:

### Representation and provenance

- BT.601 versus BT.709/limited-range case resolves according to the chosen path.
- `rgb24` is described as native-resolution decoder output, not source-native
  luma.
- Result identity distinguishes canonical encoded luma from named post-RGB
  Rec.601 intensity.
- Missing/ambiguous conversion provenance follows the accepted policy.

### Memory admission

- Result byte estimate is exact.
- Over-budget work is refused before opening scientific media.
- Peak-buffer reasoning is covered by a deterministic bound or targeted
  diagnostic.
- Large integer dimensions cannot wrap the estimate.

### Outcome composition

- The exact source outcome is retained.
- A delivered frame not yet reduced is not in processed channel coverage.
- Source failure and channel-stage failure remain distinguishable.
- Completion requires both source completion and exact channel coverage.

### Worker lifecycle

- Cancellation before source construction.
- Cancellation during source ownership.
- Supersession during close.
- Owner destruction during each phase.
- Exactly one terminal publication after source close.
- No new job starts until verified old-worker exit.

### Oracle/conformance

- Contract reference and current-oracle comparison are labelled separately.
- Unit/range conversion is explicit for any oracle fixture.
- Integer geometry/coverage matches exactly.
- Floating output uses one declared tolerance.

### Forward boundaries

- Presentation contrast changes do not mutate or recompute scientific values.
- Result/panel absolute-frame behavior can represent invalid or absent future
  boundary samples without redefining the axis.
- No PyQt import enters the scientific module.

## Manual-acceptance additions required by this review

Add these checks to the handoff's manual path:

1. Use an asset with known color metadata and show which intensity
   representation and conversion provenance the result reports.
2. Attempt a deliberately over-budget window/grid combination and confirm it is
   refused before processing with a useful byte estimate.
3. Supersede a job immediately after clicking **Compute intensity**, exercising
   cancellation before or during source construction.
4. Close Isolate while a job is stopping and confirm no thread/process warning,
   late repaint, or leaked FFmpeg process remains.
5. Confirm a failed/cancelled job distinguishes source outcome from channel
   computation outcome.
6. Confirm the panel legend describes both scientific units and presentation
   mapping.

These supplement rather than replace the original short-window, cursor,
one-frame, cancellation, and active-asset-change checks.

## Current unrelated rewrite findings

The milestone-4 review reported:

- Stale display-decode errors can defeat newer requests.
- Display decoder shutdown has a start/interrupt race.
- Derivation may publish lineage from a recorded but unverified hash.
- First-time registration can block the GUI while hashing.
- Capped display previews may distort aspect ratio.
- The supported media benchmark underreports provenance.

They remain separately routed in `.isolate-state-divergence.md`.

Milestone 5 must account for two of them at its boundary:

- Do not copy the unsafe worker lifecycle into the scientific worker.
- Preserve recorded-versus-verified identity status in the channel result.

It should not absorb the unrelated fixes unless the rewrite reports that one
small prerequisite change is required for safe implementation.

## Recommended corrected implementation shape

After milestones 3 and 4 are accepted:

```text
accepted temporal request
+ accepted resolved grid
+ chosen intensity representation
    -> validate geometry and estimate result memory
    -> refuse before source open if over budget
    -> worker creates request-local source
    -> bounded headless convert/downsample/reduce
    -> source outcome + processed channel span
    -> complete immutable current result
    -> one time x block panel
```

GUI ownership:

```text
Isolate owner
    captured scientific key
    current job token
    at most one active worker
    at most one newest pending request
    current complete result

Intensity panel
    immutable result
    existing player position/seek path
    presentation-only raster mapping
```

This remains a small milestone. None of the corrections requires a channel
registry, graph executor, cache, persisted artifact, whole-asset runner, or
normalization UI.

## Recommended disposition

Revise `5-First-channel.md` before implementation:

1. Choose canonical encoded luma or a distinctly named post-`rgb24` Rec.601
   representation.
2. Replace “native rgb24” terminology with “native-resolution decoded
   `rgb24`” where representation provenance matters.
3. Add exact result-memory admission and pre-source refusal.
4. Compose the accepted source outcome instead of duplicating it.
5. Pin the safe worker creation/cancellation/close/termination handshake.
6. Clarify the numerical oracle/conformance target and tolerance.
7. Preserve future normalization-before-blocking and temporal-context
   boundaries without implementing them.
8. Keep scientific values distinct from presentation contrast.
9. Refresh rewrite-side divergence after milestone 4 is implemented and
   accepted.

Subject to those corrections, the first-channel direction is sound and aligned
with the rewrite's forward-looking architecture.

Do not begin implementation from the uncorrected handoff, and do not proceed to
normalization merely because this review exists.
