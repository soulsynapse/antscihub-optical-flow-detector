# 3 — Define the Isolate working window

Status: implementation handoff for the third Isolate-tab milestone.

This milestone establishes the headless boundary through which later Isolate
computations obtain pixels for the selected time window. It does not add a
scientific channel or change what the player displays.

The result is a small, testable request/result contract:

```text
active asset + selected half-open frame range + requested media plane(s)
    -> resolved working-window source
    -> bounded batches of correctly identified pixels
```

“Working window” means the selected temporal input to future computation. It
does **not** mean that the entire interval must be decoded and retained as one
array.

## 1. Precedence and starting point

This is a later, user-authorized increment in the Isolate rebuild. For the work
explicitly in scope, it follows:

- `1-Build-the-player.md` and its review.
- The rewrite's implemented active-asset and media-service contracts.
- `2-Media-service-handoff.md` and its review where media performance work has
  already been accepted.
- The current rewrite v2 and scientific-computation contracts for asset
  identity, media-plane semantics, time, coordinates, and validity.

Do not replace a working v2 contract merely to match an illustrative type or
method name in this handoff. Names and pseudocode below describe required
information and behavior, not a frozen public API.

Older oracle code is evidence about behavior and possible optimizations. It is
not authoritative over the rewrite's newer correctness boundaries.

## 2. Outcome

At the end of this increment, headless code can request the current active
asset's full-frame pixels over any valid, nonempty, asset-bounded half-open frame
range.

The request explicitly identifies:

- The active asset using the rewrite's existing stable asset identity.
- The requested absolute frame range `[start, stop)`.
- The asset's exact rational timebase.
- One or more named media planes.
- A unique request or generation identity.

The resolved source returns bounded frame batches carrying:

- Absolute frame indices.
- Exact time coordinates derived from the rational timebase.
- The requested plane buffers and their semantic descriptors.
- Array shapes, dtypes, and value domains.
- Coordinates in the active asset's own pixel space.
- Explicit valid and examined coverage.
- Truncation, cancellation, and failure information.

The working-window source is independent of Qt and can be exercised without
constructing the Isolate tab. The Isolate controller can resolve its current
selection into this request, but no channel consumes the pixels yet.

## 3. Scope boundary

Implement only:

- A headless working-window request and resolved-result boundary.
- Resolution against the existing active-asset and media metadata contracts.
- Bounded sequential delivery of the requested frames and media planes.
- Exact temporal, spatial, plane, and validity metadata on returned data.
- Cancellation and generation/supersession behavior.
- Minimal Isolate-controller hookup needed to form a request from the current
  asset and selected window.
- Focused tests and a small headless diagnostic or test fixture.

Do not implement:

- Downsampling.
- A working block grid.
- Block-size controls.
- Normalization or color-space preprocessing.
- Intensity, change, tensor, optical-flow, or other scientific channels.
- Static value filtering.
- Morlet transforms, scalograms, frequency bands, or cone-of-influence rules.
- Detection, thresholds, gates, counts, clumps, marks, or events.
- A channel registry, graph planner, plugin system, or general executor.
- Whole-asset processing.
- Result persistence, coverage accumulation between requests, or caches of
  scientific artifacts.
- Recipe/settings serialization.
- CLI or HPC commands.
- Validation, presentation rendering, export, or detection stacking.
- A new asset identity system or a competing media service.

Do not add speculative parameters for those later features to the
working-window request.

Stop after this contract is integrated and validated. Do not begin the working
grid or first channel.

## 4. Ownership boundary

The working-window component owns:

- Validating and resolving a requested temporal window.
- Asking the existing media service for declared pixel planes.
- Delivering bounded, correctly described pixel batches.
- Reporting what was actually delivered and valid.
- Releasing its own in-flight work and temporary buffers.

It does not own:

- Active-asset selection.
- The player's playhead, loop timer, or display raster.
- Media probing or low-level decoder implementation.
- Scientific preprocessing or channel computation.
- Long-lived result artifacts.
- GUI presentation.

The intended dependency direction is:

```text
Isolate player/window state
          |
          v
Isolate controller -----> working-window request
                                  |
                                  v
                         existing media service
                                  |
                                  v
                    typed, bounded pixel batches
```

Later grids and channels may depend on the working-window boundary. The
working-window component must never import or call those consumers.

## 5. Request contract

A request contains the following semantics. Reuse existing v2 value objects
where they already express them correctly.

```text
asset_ref
asset_generation
start_frame
stop_frame
requested_planes
request_id
```

Execution-only limits such as a batch-size or memory budget may be supplied
through an existing execution context. They are not scientific settings and
must not change the values delivered.

### 5.1 Asset identity

`asset_ref` is the same stable identity used by the active-asset layer. A path
alone is insufficient if the rewrite already distinguishes asset identity from
location or records content/provenance identity.

The request targets the active asset itself:

- A source asset is decoded in its own coordinates.
- A derived child is decoded from the child's own media in the child's
  coordinates.
- The working-window service does not reopen a parent and recreate a child crop
  at request time.

`asset_generation` represents the active-asset/session generation already used
by the application, or an equivalent monotonic invalidation token. It prevents
late data from a previous active asset from being published into the new one.

Do not invent a second canonical definition of asset identity solely for
Isolate.

### 5.2 Frame range

The range is an absolute, integer, half-open interval:

```text
[start_frame, stop_frame)
```

with:

```text
0 <= start_frame < stop_frame <= verified_available_stop
```

The underlying model accepts any nonempty asset-bounded interval. The player's
initial UI range is not a durable limit on this contract.

For a request `[17, 20)`, the only requested frames are `17`, `18`, and `19`.
No conversion through rounded display seconds is allowed.

If verified availability is still being established by the existing media
layer, use that layer's current explicit resolution behavior. Do not block
first use merely to reproduce an oracle frame-count assumption, and do not
pretend an unverified container count is exact.

### 5.3 Rational time

The authoritative rate is the existing exact pair:

```text
fps_num / fps_den
```

Frame `t` has the exact presentation time:

```text
t * fps_den / fps_num seconds
```

The request or its resolved form must retain the rational pair. Returned batches
carry absolute frame indices, from which exact times can be derived without
accumulating floating-point drift.

Floating-point seconds may be supplied for display or diagnostics, but they are
not authoritative identifiers and must not be fed back into scientific frame
selection.

### 5.4 Requested media planes

Planes are requested by explicit semantic identity. Examples may include:

- Encoded luma.
- RGB with a declared color interpretation.
- A declared native codec plane.
- Alpha, when present and explicitly requested.

The actual supported plane identifiers come from the rewrite's media contract.
Do not create an Isolate-only alias with ambiguous meaning such as `gray`,
`image`, or `display_frame`.

The media service must not silently substitute:

- RGB for requested encoded luma.
- A display-sized preview for native-size evidence.
- A parent asset's pixels for a child asset.
- A derived color space for a decoder plane.

HSV, Lab, normalized intensity, linear-light luminance, and similar derived
representations remain explicit later preprocessing nodes. They are not hidden
inside the working-window source.

The first diagnostic may request only one inexpensive plane supported by the
current backend. Supporting every conceivable plane is not required in this
increment.

## 6. Resolved request

Validate the request before starting substantial decoding. The resolved form
records at least:

```text
stable asset identity and generation
verified or currently authoritative available frame interval
resolved requested frame interval
exact rational timebase
resolved plane descriptors
active-asset width and height
spatial coordinate reference
request identity
execution limits used for delivery
```

Resolution must fail clearly when:

- The asset cannot be resolved or opened.
- The range is empty, inverted, negative, or outside verified availability.
- A requested plane is unsupported or ambiguous.
- Media metadata needed to interpret the plane is ambiguous under the current
  media policy.
- The request refers to a stale asset generation.

Do not silently clamp a malformed headless request. The GUI controller may
clamp user gestures before constructing the request, as already required by the
player handoff.

## 7. Batch/result contract

The delivery shape may follow the rewrite's existing media-service conventions.
A representative batch has these semantics:

```text
request_id
asset identity and generation
absolute_frame_indices
exact rational timebase
plane_id -> pixel buffer
plane descriptors
array shape and dtype
active-asset spatial coordinates
validity
```

Requirements:

- Frame indices are absolute within the active asset, not offsets within the
  selected window or batch.
- Batches are ordered by increasing frame index for ordinary sequential
  delivery.
- Every delivered frame lies inside `[start_frame, stop_frame)`.
- Plane arrays agree with the declared frame and spatial axes.
- Plane buffers are contiguous when the downstream media contract requires it.
- Buffer ownership and lifetime are explicit. A consumer must not observe a
  buffer being mutated when the decoder advances.
- Native-size full-frame planes use the active asset extent:
  `[0, width) x [0, height)`.
- Metadata uses `(x, y)` while arrays use `[y, x]`.
- A display letterbox, widget size, device-pixel ratio, or zoom state never
  changes scientific pixel coordinates.

If batches contain multiple requested planes, their frame axis and asset
identity must agree. A plane that cannot be produced is an explicit failure; it
is not omitted silently.

## 8. Validity, coverage, truncation, and failure

Returned data distinguishes at least:

- Requested coverage.
- Successfully decoded/examined coverage.
- Valid coverage.
- Cancellation.
- Truncation or decode failure.

Unknown or missing frames are not converted into black pixels, zeros, quiet
signal, or negative detections.

For an ordinarily decoded full-frame plane, validity may be represented
compactly as valid frame intervals plus a declaration that the complete plane
extent is valid. If the backend can report partial or spatial invalidity, retain
it explicitly rather than overwriting the pixels. Do not introduce a large
per-pixel mask when the current media contract and evidence do not require one.

Decoder exhaustion before the resolved requested stop is truncation:

- Preserve batches already delivered.
- Report the final successfully decoded frame or covered stop.
- Mark all remaining requested frames unexamined.
- Do not report the request as complete.

A decode error carries the media layer's stable error code and enough structured
context to identify the asset, request, range, plane, and failing frame. The
working-window layer should add context without replacing a more specific
underlying cause.

## 9. Bounded memory and streaming

Do not require:

```text
frames x full_height x full_width x all_planes
```

to fit in RAM.

The source delivers bounded batches or an equivalent bounded iterator/stream.
At any time, memory use must be limited by the accepted execution policy plus
documented decoder and consumer buffers.

Required properties:

- Batch size is an execution choice, not a change in scientific meaning.
- Changing batch size returns the same frame identities and pixel values.
- Backpressure or a bounded queue prevents an unbounded producer backlog.
- Cancellation can be observed between bounded units of work.
- Released batches do not remain retained by an accidental session-wide list.

It is acceptable for a tiny test or explicitly bounded caller to assemble
batches into one array. That convenience must not define the core contract or
be used automatically for an arbitrary player-selected interval.

Do not add a persistent pixel cache in this milestone. If the existing media
service already has a safe in-memory cache, reuse it through its public
contract; do not duplicate it in the working-window layer.

## 10. Cancellation and latest-request behavior

The Isolate controller treats the current request as generation-scoped and
latest-only.

A request becomes obsolete when any of these changes:

- Active asset.
- Selected frame range.
- Requested media planes.
- The controller or application closes.

When a request is superseded:

1. Signal cancellation through the existing cancellation mechanism.
2. Stop producing new batches as promptly as the bounded media operation
   permits.
3. Release queued buffers and request-local resources.
4. Ignore any late batch whose request id or asset generation is no longer
   current.
5. Never let an obsolete completion or error overwrite the state of the newer
   request.

Cancellation is not truncation and is not success. Report it distinctly.

Rapid window dragging must not build an unbounded queue of complete-window
decodes. The controller may defer starting expensive work until a consumer
exists. In this milestone there is no scientific consumer, so do not eagerly
decode every window change merely to prove that the hookup exists.

## 11. Relationship to player decoding

Player display requests and working-window scientific requests have different
semantics:

```text
player
  latest requested display frame, optimized for interactive latency

working window
  ordered bounded delivery over an explicit frame interval
```

They may reuse the same media service and safe shared caches. They must not
corrupt each other's position, generation, or request ordering.

Do not require a second physical decoder if the rewrite already supports the two
request patterns correctly through one service. Conversely, do not force a
single mutable decoder cursor to serve both patterns if that causes seeking,
stale-frame, or lifecycle bugs. Decoder ownership remains an implementation
choice to inspect and benchmark.

Within one media-session generation and one resolved representation, passive
consumers must not accidentally cause duplicate decodes of the same request.
This is not permission to build a general cross-job frame cache.

The media-service performance hints remain hypotheses:

- Protect correctness and lifecycle first.
- Benchmark before and after a proposed optimization.
- Retain only measured improvements.
- Do not copy an oracle technique that breaks the rewrite's current contract.

## 12. Minimal Isolate hookup

Add only enough controller behavior to prove that the current GUI selection can
form the same headless request:

- Read the shared active asset identity and generation.
- Read the Isolate-local `[window_start, window_stop)`.
- Obtain the exact rational timebase from authoritative asset metadata.
- Accept an explicit plane request from a headless caller or diagnostic.
- Resolve or cancel work outside the widgets.

The player and timeline widgets must not decode or accumulate a working window.
They continue to own presentation and interaction only.

Because there is no channel yet:

- Do not add fake plots or channel cards.
- Do not continuously decode the complete selection after every GUI gesture.
- Do not add a new permanent `Prepare`, `Process`, or `Run` button solely for
  this plumbing milestone.
- Do not show a success state implying that scientific isolation has run.

A development diagnostic may report the resolved range, planes, batches,
coverage, and cancellation state. Keep it headless and structured so tests can
assert it; it is not the future user-facing CLI recipe.

## 13. Suggested implementation shape

Fit this work into the rewrite's existing packages. The following split is
illustrative:

```text
domain or processing boundary
  WorkingWindowRequest
  ResolvedWorkingWindow
  FrameBatch
  WorkingWindowOutcome

headless service
  resolve(request)
  iter_batches(resolved, cancellation)

Isolate controller
  form request from active asset + local window
  own current request id
  cancel/supersede obsolete work
  reject stale results
```

Prefer existing types for:

- Asset references and generations.
- Rational rates and frame spans.
- Media-plane ids and descriptors.
- Cancellation.
- Structured errors.
- Coverage intervals.

Do not create a broad pipeline base class in order to implement this split. A
small typed function/service is sufficient until real channels demonstrate what
additional lifecycle is necessary.

## 14. Automated tests

Use deterministic, tiny media fixtures with known per-frame content. At least
one fixture should encode its frame number into the pixels so off-by-one and
seek errors are observable.

### 14.1 Request resolution

Test:

- `[2, 5)` resolves to frames `2`, `3`, and `4`, never `5`.
- Empty, inverted, negative, and out-of-range requests fail before decoding.
- A valid one-frame headless interval is accepted by this boundary.
- The player's initial maximum window length is not enforced by the headless
  model.
- A direct child asset resolves to the child's media and coordinate extent.
- A stale asset generation is rejected.
- An unsupported or ambiguous plane request fails explicitly.

### 14.2 Time and coordinates

Test:

- A fixture at `60000/1001` retains that exact rational pair.
- Frame `t` maps to `t * 1001 / 60000` seconds without accumulated drift.
- Returned indices remain absolute when the requested range starts after zero.
- Full-frame arrays have the declared `[frame, y, x, ...]` axes.
- Spatial extent is the active asset's `[0, width) x [0, height)`.
- Widget resizing, letterboxing, and zoom state cannot alter returned pixels or
  scientific coordinates.

### 14.3 Plane semantics

For each plane supported in this milestone, test:

- The returned plane id and descriptor match the request.
- Shape, dtype, bit depth or numeric range, and color/range interpretation are
  explicit.
- Known fixture pixels agree with the declared interpretation.
- Requesting one plane does not silently materialize or return a different one.
- A native-size request cannot be satisfied by a cached display-sized raster.

### 14.4 Batching and memory

Test:

- Different batch sizes return identical ordered frame identities and pixels.
- The final short batch is handled correctly.
- No batch crosses the requested stop.
- The queue and number of live batches remain bounded with a deliberately slow
  consumer.
- Released buffers are not mutated by later decoder reads.
- The implementation does not retain all decoded frames after iteration.

Use a direct invariant or instrumentation for boundedness rather than relying
only on a process-wide memory measurement.

### 14.5 Validity and truncation

Test:

- A successful request reports complete examined and valid coverage.
- Early decoder exhaustion reports partial coverage and truncation.
- Remaining frames are unexamined, not invalid zeros.
- A media failure preserves the underlying stable error code and adds request
  context.
- A partial result is never labelled complete.

### 14.6 Cancellation and supersession

Test:

- Cancelling during iteration stops further delivery within a bounded amount of
  work.
- Cancellation is distinct from success, failure, and truncation.
- A new range supersedes an older range.
- Switching assets invalidates pending results from the old generation.
- A late old batch, completion, or error cannot become current.
- Rapidly replacing requests does not grow an unbounded queue.
- Closing the controller releases requests, buffers, workers, and media
  resources it owns.

### 14.7 Player regression

Test:

- Existing player frame accuracy and loop boundaries remain unchanged.
- Forming or cancelling a working-window request does not move the player
  playhead.
- Player and window-source requests cannot corrupt one another's decoder
  position or generation.
- With no scientific consumer, ordinary window dragging does not trigger a full
  selected-window decode.

## 15. Headless diagnostic

Provide a focused test helper or development diagnostic that can:

1. Resolve one asset and `[start, stop)` range.
2. Request one declared media plane.
3. Consume every returned batch.
4. Emit structured facts:
   - Asset identity and generation.
   - Requested and resolved frame range.
   - Rational timebase.
   - Plane descriptor.
   - Per-batch frame bounds and shapes.
   - Examined and valid coverage.
   - Final outcome.
5. Optionally hash fixture frames or plane buffers for repeatability.

This diagnostic is an implementation-validation surface, not a supported
analysis CLI. Do not add recipe parsing, job directories, export choices, or
HPC behavior here.

## 16. Manual acceptance

After automated tests pass, stop and return this milestone for validation.

The manual check is intentionally small because this increment is primarily
headless:

1. Open an ordinary source asset in Replicates and switch to Isolate.
2. Confirm the existing player still seeks, loops, and scrubs normally.
3. Change the selected window several times, including rapid dragging.
4. Confirm the UI remains responsive and does not begin decoding every frame in
   each transient selection.
5. Run the focused diagnostic on the settled window and inspect its structured
   range, timebase, plane, shape, and coverage.
6. Open a derived child asset and repeat the diagnostic.
7. Confirm the child reports its own media dimensions and coordinates.
8. Start a deliberately slow diagnostic, change the active asset or range, and
   confirm the obsolete request cancels or is ignored without stale output
   becoming current.

Do not judge scientific channel values in this milestone; none exist yet.

## 17. Definition of done

This milestone is complete only when:

- A non-Qt caller can resolve and consume an explicit active-asset frame window.
- The range is absolute, integer, half-open, and asset-bounded.
- Exact rational time is preserved.
- Media planes and their interpretations are explicit.
- Returned batches carry frame identity, axes, coordinates, shape, dtype, and
  validity/coverage.
- Delivery is bounded and does not require a full-window allocation.
- Cancellation and supersession prevent stale asset or range results from
  becoming current.
- Player interaction and frame accuracy have not regressed.
- The implementation does not introduce grids, channels, recipes, persistence,
  exports, or a general pipeline framework.
- Automated tests pass.
- The user has reviewed and accepted the milestone.

Stop here. The next possible handoff is the working grid, but it is not
authorized by completion of this one.
