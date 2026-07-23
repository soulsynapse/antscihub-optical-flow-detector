# Review — 3 Define the Isolate working window

Reviewed and updated against the current checkout on
`2026-07-23 14:41:35 -07:00` at commit `7707968`.

Status: architectural and current-rewrite review of
`3-Working-window.md`. The implementation handoff has been corrected to reflect
this review.

## Verdict

The oracle handoff identified the right long-term seam, but its original form
was unsafe for this checkout. It described a small pixel-delivery boundary while
implicitly requiring several future systems that do not exist:

- Headless asset generations and request ids.
- General named and multi-plane media delivery.
- General validity and coverage artifact types.
- Asynchronous cancellation and latest-only orchestration.
- A GUI integration despite there being no scientific consumer.
- A diagnostic that somehow shared private live state with another Qt process.

The corrected milestone is intentionally smaller:

```text
stable registered asset/content reference
+ absolute half-open frame span
+ one native rgb24 plane
+ bounded execution options
    -> resolved source facts
    -> synchronous closeable iterator
    -> immutable native frame batches
    -> explicit delivered span and final source outcome
```

It stops before the working grid, first channel, GUI computation worker, or
scientific result architecture.

## Current-checkout evidence

The review was checked against the implementation rather than relying on the
oracle's state descriptions.

### Present

- Asset sidecars contain stable `asset_id` and
  `media.content_sha256`.
- `ActiveAsset` contains paths, dimensions, rational fps, duration, kind, and
  parent navigation metadata.
- `MediaSession` probes media, calculates exact `Fraction` timestamps, and
  returns immutable FFmpeg `rgb24` bytes.
- Adjacent `read_frame_rgb` calls reuse one mutable FFmpeg decoder subprocess.
- Native decode remains the default; Isolate display explicitly requests a
  width-limited preview.
- `IsolateSession` owns a private GUI generation counter.
- `IsolateDecodeThread` coalesces display requests and rejects stale
  generations.
- The existing test direction includes deterministic lossless FFV1 footage.

### Absent

- A headless `WorkingWindowRequest` or frame-span value object.
- A headless asset-generation contract.
- General media-plane ids or descriptors.
- Multi-plane, luma, alpha, or native-codec-plane delivery.
- A scientific cancellation worker or queue.
- General validity and coverage intervals.
- A scientific executor, recipe, or result artifact.

The corrected handoff asks only for the smallest additions justified by the
current media backend.

## Corrections incorporated into the handoff

### 1. GUI generations do not belong in the headless request

The original handoff put:

```text
asset_generation
request_id
```

into the request, resolved metadata, batches, outcome, diagnostic, and tests.

`IsolateSession._generation` is meaningful only to one live Qt controller. It
rejects late display-thread signals; it is not stable asset identity and cannot
be reproduced by a CLI, test process, or later HPC job.

The corrected boundary separates:

```text
headless source request
  stable registered asset/content reference
  half-open frame span
  rgb24 plane

future GUI job envelope
  GUI generation
  publication request id
  cancellation ownership
  latest-result policy
```

The second layer remains deferred until a real GUI scientific consumer exists.
A cancellation predicate may still be supplied as an execution option to the
synchronous source.

### 2. Decoded delivery is not scientific examination

The original handoff repeatedly combined decoded and examined coverage.

No channel exists in this milestone. Decoding a frame does not establish that
SIEVE processed it, found it quiet, or examined it for a signal.

The source may report:

```text
requested span
delivered/decoded span
full-frame buffer validity
final source outcome
```

These terms remain reserved for later scientific results:

```text
examined
processed
quiet
detected
```

Missing frames are never converted to black pixels, zeros, or negative
detections.

### 3. The first plane contract matches the media API that exists

The original handoff described named multi-plane delivery, encoded luma, native
codec planes, alpha, and plane maps.

The current media API is:

```python
read_frame_rgb(frame, max_width=None) -> bytes
```

The corrected milestone therefore supports exactly one explicit plane:

```text
rgb24
native active-asset dimensions
shape [frame, y, x, channel]
dtype uint8
value domain [0, 255]
channel order R, G, B
immutable bytes
```

The smallest descriptor needed to make those semantics explicit is appropriate.
A plane registry and multi-plane map are not.

The source must not pass the player's `max_width=1280` preview cap.

### 4. Frame extent has provenance, not assumed verification

The original handoff simultaneously required:

```text
stop_frame <= verified_available_stop
```

and prohibited a full decoded count before first use. The current rewrite
cannot guarantee both.

`MediaSession.frame_count` may come from:

1. Decoded count.
2. Packet count.
3. Container count.
4. Duration multiplied by rational fps.

The corrected handoff records both the declared extent and its exact provenance.
Requests are admitted against that declared extent. The source then reports the
span it actually delivered.

Packet, container, and duration-derived counts must not be relabeled as verified
decodable coverage. A small-window request must not trigger a full-video count.

### 5. Resolution owns timebase and media facts

If a caller identifies an asset, also accepting caller-supplied width, height,
fps, or plane interpretation creates conflicting sources of truth.

The caller supplies:

```text
asset/content reference
frame span
plane id
```

Resolution obtains:

- Media path.
- Width and height.
- Exact rational fps.
- Declared extent and provenance.
- Plane descriptor.

Returned absolute frame indices plus the rational pair are sufficient to derive
exact times. A floating-point timestamp array is not required.

### 6. Synchronous iteration is the current bounded-delivery contract

The original handoff's queue, slow-consumer, worker, and latest-only language
could be read as requiring an asynchronous producer.

The current `MediaSession` already returns one immutable frame at a time and
retains its decoder for adjacent reads. A synchronous iterator therefore has
natural backpressure and needs no producer thread or bounded queue.

Batch size may change grouping only. The implementation need not concatenate a
whole selected interval.

The corrected contract also requires explicit close/context-manager behavior.
The source owns a request-local media session and closes it on exhaustion,
cancellation, failure, explicit close, or early consumer exit through the
context-managed path.

Cancellation is checked between bounded reads. It does not promise interruption
inside one already-blocking FFmpeg read without a worker and thread-safe
interrupt lifecycle.

### 7. Scientific iteration does not share the player's mutable decoder

One `MediaSession` owns one mutable FFmpeg cursor, and `interrupt()` terminates
that decoder.

Sharing the live player session would allow:

- Scientific reads to reposition the display decoder.
- Player scrubbing to disrupt sequential source delivery.
- Source cancellation to terminate display work.

The safe current default is a separate request-local `MediaSession` over the
same resolved media. That reuses the existing media service without implying
concurrent-reader safety or building a competing decoder system.

### 8. The original manual acceptance path was not executable

The original path asked a standalone headless diagnostic to consume a running
GUI's private settled window and react when that GUI changed asset or range.
It defined no in-process action, IPC bridge, or supported CLI surface through
which that could happen.

The corrected acceptance model is executable:

- Run the diagnostic with explicit registered asset, span, plane, and batch
  inputs.
- Exercise cancellation with a deterministic headless token.
- Test a pure GUI snapshot adapter without launching source work.
- Confirm existing player behavior and confirm window gestures do not start
  scientific decoding.

Live GUI supersession is deferred until the first real scientific worker and
consumer.

## Additional current-checkout corrections found during review

### 9. Qt independence requires an import boundary

`ActiveAsset` is a frozen dataclass, but it shares
`application.active_asset` with `ActiveAssetController`, and that module imports
PyQt.

A headless source that imports `ActiveAsset` from that module is not actually
Qt-independent. The corrected handoff requires headless request and source types
to avoid that module. A thin GUI adapter may extract primitive stable identity
and window values into the headless request.

Moving the immutable dataclass to a headless module later is an implementation
option, not a requirement to redesign active-asset selection.

### 10. Recorded content identity is not verified current content

`AssetService.inspect()` validates the sidecar schema and returns the recorded
`content_sha256`; it does not rehash the current media file.

The corrected contract distinguishes recorded identity from explicitly verified
identity. It does not hash an arbitrarily large asset merely to resolve a small
window.

The source requires a registered sidecar and checks the request's expected
asset id and recorded content hash against it. It must not silently initialize
an unregistered raw video or claim that reading the recorded hash verified the
current bytes. It compares cheap current facts—file size, width, height, and
rational fps—with the sidecar and reports identity status as recorded.
Explicit content verification remains a separate `AssetService` operation.

### 11. Extent status should retain its source

A binary estimated/verified flag loses useful distinctions between decoded,
packet, container, and duration-derived counts.

The corrected handoff requires provenance-specific status. This keeps current
admission policy explicit and lets a later whole-asset milestone decide which
sources are sufficient without retroactively changing this contract.

### 12. The final outcome needs an accessible API

A Python generator can return a value through `StopIteration.value`, but an
ordinary `for` loop discards it. “Return or expose final outcome” was therefore
not a sufficient API requirement.

The corrected handoff requires a closeable/context-managed stream whose final
outcome remains inspectable after completion. It does not freeze a class name or
build a general execution framework.

Resolution errors raise before streaming. Runtime decode errors set a failed
outcome, close the request-local media session, and re-raise the structured
`SieveError`. Cancellation and explicit truncation terminate iteration normally
and remain distinguishable through the required outcome.

### 13. Current short reads do not distinguish EOF from decode failure

`MediaSession.read_frame_rgb()` currently raises the same
`FRAME_DECODE_FAILED` error whenever it receives fewer bytes than one complete
frame. It stops the decoder before preserving a structured clean-EOF versus
decoder-error distinction.

The working-window layer must not classify an ambiguous error by parsing message
text or assuming empty stderr means EOF.

The corrected policy is:

- Explicit media-layer clean EOF before requested stop may produce
  `truncated`.
- An ambiguous current short read preserves the delivered prefix and produces
  `failed`.
- A minimal structured EOF reason may be added to `MediaSession` if it is small
  and testable.
- Broad decoder redesign is not required by this milestone.

### 14. Successful `rgb24` delivery does not justify general validity machinery

The current backend returns a complete exact-sized frame or raises an error. It
does not report partial spatial validity.

For this milestone, a successfully delivered frame has a valid full-frame
buffer. Failed delivery remains an error. Per-pixel masks and general coverage
interval types remain deferred until a backend or channel demonstrates a need.

## Testing disposition

### Headless source tests

Cover:

- Half-open validation and absolute indices.
- Registered identity mismatch and unregistered-asset rejection.
- Exact rational timebase.
- Extent provenance.
- Native `rgb24` descriptor, shape, and values.
- Batch-size invariance and final short batches.
- Synchronous boundedness and immutable yielded buffers.
- Context-managed cleanup on every exit path.
- Cancellation between bounded reads.
- Accessible final outcome.
- Explicit EOF truncation versus ambiguous short-read failure.
- Child media and child coordinates.

Use lossless deterministic fixtures for exact pixel comparisons. Temporal
identity and pixel equivalence remain separate assertions.

### GUI adapter tests

Cover:

- Current asset and local window create the expected immutable request.
- Later GUI changes do not mutate an existing request.
- Snapshotting and dragging a window do not consume the source.
- Existing player display behavior remains unchanged.

Run Qt tests with `QT_QPA_PLATFORM=offscreen`.

Do not create a scientific worker merely to satisfy stale-result tests for a
feature that does not yet run.

## Durable decisions retained from the oracle

- Absolute half-open frame ranges.
- Exact rational frame timing.
- Child assets decoded from their own media and coordinates.
- Native scientific pixels kept separate from display previews.
- Bounded delivery rather than full-window allocation.
- No persistent scientific pixel cache.
- Downsampling and the working grid remain deferred.
- Channels, filtering, detection, persistence, recipes, export, and a general
  executor remain deferred.

## Recommended disposition

The corrected `3-Working-window.md` is now suitable to hand to an implementation
agent, subject to user acceptance.

Implementation should remain a small reversible foundation:

```text
registered stable identity
+ half-open frame span
+ native rgb24
    -> resolved facts
    -> request-local synchronous stream
    -> immutable batches
    -> delivered span and explicit outcome
```

No broader processing architecture is justified by this milestone.
