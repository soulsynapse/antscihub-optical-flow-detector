# 3 — Define the Isolate working window

Reviewed and updated against the current checkout on
`2026-07-23 14:41:35 -07:00` at commit `7707968`.

Status: corrected implementation handoff for the third Isolate-tab milestone.

This milestone establishes the small headless boundary through which a later
Isolate channel can obtain native pixels for a selected time window. It does
not add a scientific channel, computation worker, or user-facing processing
action.

The required result is:

```text
registered asset reference
+ absolute half-open frame span
+ native rgb24 plane
+ bounded execution options
    -> resolved source facts
    -> synchronous bounded frame batches
    -> explicit final source outcome
```

“Working window” means the selected temporal input to future computation. It
does not mean that all selected frames are decoded or retained as one array.

## 1. Precedence and current starting point

This handoff follows:

- `1-Build-the-player.md` and its review.
- The implemented active-asset and Isolate player contracts.
- `2-Media-service-handoff.md`, its review, and the accepted media performance
  work.
- The current source and tests in this checkout.

Oracle contracts and pseudocode are evidence, not current APIs. Do not create a
future executor contract merely because the oracle assumed one already existed.

### Present in this checkout

- Asset sidecars contain `asset_id`, `media.content_sha256`, media location,
  dimensions, rational fps, duration, and lineage.
- `ActiveAsset` is an immutable GUI-facing snapshot, but it does not contain
  the active media's `content_sha256`.
- `ActiveAsset` and `ActiveAssetController` currently share a module that
  imports PyQt. A headless working-window module must not import that module.
- `MediaSession.read_frame_rgb(frame, max_width=None)` returns immutable
  `rgb24` bytes.
- Adjacent reads through one `MediaSession` reuse one mutable FFmpeg decoder
  cursor.
- Calling `read_frame_rgb` without `max_width` requests native dimensions.
- The Isolate player explicitly requests a width-limited display
  representation. That representation is not scientific evidence.
- `IsolateSession._generation` is a private GUI lifecycle token used to reject
  late display results.
- Ordinary `MediaSession` construction does not fully decode the asset merely
  to verify its final frame.

### Not present

- A headless working-window request or frame-span type.
- A general media-plane API or descriptor registry.
- Encoded-luma, alpha, native-codec-plane, or multi-plane delivery.
- A scientific worker, queue, cancellation system, or latest-only publisher.
- General validity, coverage, recipe, executor, or result-artifact types.

The implementation should add only the narrow types and behavior required
below.

## 2. Scope

Implement:

- A Qt-independent request containing a registered asset reference, an
  absolute half-open frame span, and one explicit plane id.
- Resolution through the existing asset and media services.
- A descriptor for the one supported plane, native `rgb24`.
- A synchronous, bounded, closeable stream of immutable frame batches.
- Exact rational timebase and active-asset coordinates in resolved metadata.
- Explicit extent provenance and delivered frame span.
- Cancellation checks between bounded decode operations.
- A final source outcome that remains inspectable after iteration.
- A pure GUI adapter that snapshots the current asset and Isolate-local window
  into the headless request without starting a decode.
- Focused headless tests, small GUI adapter regressions, and a development
  diagnostic or test helper.

Do not implement:

- A GUI computation worker, producer thread, cross-thread queue, or mailbox.
- GUI job generation, request publication, or latest-result orchestration.
- Continuous working-window decoding on window changes.
- Downsampling, a working block grid, or block-size controls.
- Normalization, color-space preprocessing, or scientific channels.
- Filtering, transforms, detections, gates, events, or marks.
- A plane registry, multi-plane batches, or speculative plane aliases.
- Persistent pixel caches or scientific result artifacts.
- Recipes, job directories, export, CLI/HPC processing, or a graph executor.
- A new canonical asset identity system.

Stop after this contract is integrated and validated. Do not begin the working
grid or first scientific channel.

## 3. Ownership and dependency boundaries

The headless working-window source owns:

- Resolving a registered asset reference.
- Validating the requested frame span against the declared extent.
- Opening and closing its own request-local `MediaSession`.
- Delivering ordered native `rgb24` frames in bounded batches.
- Reporting resolved facts, delivered span, cancellation, and failure.

It does not own:

- Active-asset selection or GUI lifecycle generations.
- The player's playhead, display decoder, timer, or preview raster.
- Scientific computation or claims that footage was examined.
- Long-lived caches, results, or presentation.

The dependency direction is:

```text
IsolateSession and current ActiveAsset
               |
               | pure snapshot; no decode
               v
headless WorkingWindowRequest
               |
               v
headless asset resolution
               |
               v
request-local MediaSession
               |
               v
bounded immutable rgb24 batches
```

The headless request, resolved metadata, batches, stream, and outcome must live
in modules that can be imported without importing PyQt. The GUI adapter may
depend on both the GUI snapshot and the headless request type.

## 4. Request contract

The request contains:

```text
asset_ref
expected_asset_id
expected_content_sha256
start_frame
stop_frame
plane_id = rgb24
```

`asset_ref` identifies an existing registered asset, normally through its asset
sidecar path. A raw unregistered video is not silently initialized by the
working-window source or diagnostic.

`expected_asset_id` and `expected_content_sha256` reuse the identity already
stored in the sidecar. They protect an immutable request from silently resolving
to a different sidecar identity later. They are not a second asset identity
scheme.

The GUI adapter may obtain the active content hash through the existing asset
inspection path, or the implementation may add that existing sidecar field to
the immutable GUI snapshot. Do not import the PyQt-bearing
`application.active_asset` module into the headless source.

Do not put these values in the headless request:

- `IsolateSession._generation`.
- A GUI publication or request id.
- Width, height, fps, duration, or frame-count claims supplied by the caller.
- Widget size, zoom, device-pixel ratio, or the display preview width.
- Future scientific settings.

Batch size and a cancellation predicate are execution options supplied when the
stream is opened. They do not change the scientific request or delivered pixel
values.

## 5. Frame-span contract

The requested interval is absolute, integer, and half-open:

```text
[start_frame, stop_frame)
```

For `[17, 20)`, the requested frames are exactly `17`, `18`, and `19`.

Validate before substantial decoding:

```text
0 <= start_frame < stop_frame <= declared_stop
```

The headless model accepts any nonempty span admitted by the asset's declared
extent. It does not inherit the player's initial window length, UI minimum, or
UI maximum.

Do not silently clamp malformed headless requests. The GUI already settles user
gestures before taking a request snapshot.

## 6. Identity and media resolution

Resolution must:

1. Require an existing valid asset sidecar.
2. Resolve its media path through `AssetService`.
3. Confirm that the sidecar's `asset_id` and recorded `content_sha256` match the
   request's expected identity.
4. Compare cheap current facts—file size, width, height, and rational fps—with
   the sidecar and fail on a mismatch.
5. Open a new request-local `MediaSession`.
6. Obtain current duration, media metadata, and declared extent from that
   session.
7. Resolve the single supported plane descriptor.

The sidecar content hash is a recorded content identity. Reading it is not
equivalent to rehashing and verifying the current media bytes. Resolved metadata
must describe the identity status honestly:

```text
recorded
```

Ordinary small-window resolution must not hash or fully decode a large video
merely to label it verified. This milestone reports the sidecar identity as
recorded. Explicit content verification remains a separate `AssetService`
operation and is not smuggled into ordinary window-open latency.

Resolution fails clearly for:

- A missing or invalid sidecar.
- Unreachable media.
- Mismatched expected asset or content identity.
- Invalid requested bounds.
- An unsupported plane id.
- Media metadata insufficient to describe native `rgb24`.

## 7. Extent provenance

`MediaSession.frame_count` may currently come from:

1. Decoded-frame count.
2. Packet count.
3. Container frame count.
4. Duration multiplied by rational fps.

Carry both the admitted `declared_stop` and its provenance in resolved metadata.
Use explicit provenance such as:

```text
decoded_count
packet_count
container_count
duration_estimate
```

Do not collapse all four into “verified.” In particular, packet and container
counts do not establish that every indexed frame can be decoded successfully.

The working-window source validates requests against the declared extent and
reports what it actually delivered. It does not run a full-video count merely
to admit a small request.

## 8. Exact time and coordinates

Resolution obtains the authoritative rational pair:

```text
fps_num / fps_den
```

The exact presentation time for absolute frame `t` is:

```text
t * fps_den / fps_num seconds
```

Batches carry absolute frame indices plus the resolved rational pair or an
unambiguous reference to it. A floating-point timestamp array is not required
and must not become the authoritative identity for frame selection.

Native full-frame coordinates are:

```text
x in [0, width)
y in [0, height)
```

Metadata uses `(x, y)`. Array semantics use `[frame, y, x, channel]`.

A derived child is a complete processing world. Resolve and decode the child's
own media at the child's own dimensions. Never reopen the parent and recreate a
crop at request time.

## 9. Supported plane

This milestone supports one explicit plane:

```text
plane_id: rgb24
resolution: native active-asset width and height
per-frame shape: [height, width, 3]
batch axes: [frame, y, x, channel]
dtype: uint8
value domain: [0, 255]
channel order: R, G, B
buffer: immutable bytes
coordinates: active-asset pixel coordinates
backend: FFmpeg through MediaSession
```

Record relevant probed source color metadata, including unknown values, and the
fact that FFmpeg produced `rgb24`. Missing optional source color tags are not
silently relabeled as known, but they also do not automatically require a
general color-policy subsystem in this milestone.

The source must call `read_frame_rgb` without the player's `max_width=1280`
display cap.

Do not add encoded luma, alpha, native codec planes, plane maps, or a registry.
Later channels may justify those extensions without changing frame, time,
coordinate, or bounded-delivery semantics.

## 10. Batch and stream contract

A batch carries at least:

```text
absolute_frame_indices
immutable rgb24 frame buffers
frame_count
per-frame shape
plane descriptor or resolved-plane reference
```

Requirements:

- Indices increase in ordinary sequential order.
- Every index lies inside `[start_frame, stop_frame)`.
- No batch crosses `stop_frame`.
- A final short batch is valid.
- Changing batch size changes grouping only, not indices or pixel values.
- The implementation need not concatenate frames into one large byte array.
- Previously yielded immutable buffers are never mutated by later reads.
- The stream does not retain all previously yielded batches.

Use a synchronous stream. It has natural backpressure because no next batch is
decoded until the consumer requests it.

The stream must be explicitly closeable and usable as a context manager. It
owns the request-local `MediaSession` and closes it in `finally` on:

- Normal exhaustion.
- Cancellation.
- Decode failure.
- Explicit close.
- Early consumer abandonment when used through the required context-manager
  path.

Cancellation is checked between bounded frame reads or batches. This milestone
does not promise that a token can interrupt one already-blocking FFmpeg read;
that would require worker/thread lifecycle which remains deferred.

## 11. Outcome and source truth

The stream exposes a final outcome after termination without requiring callers
to recover a generator's `StopIteration.value`.

The outcome distinguishes:

```text
complete
cancelled
truncated
failed
```

It records:

- Requested span.
- Delivered contiguous prefix or delivered span.
- Final outcome kind.
- Structured underlying `SieveError`, when present.
- The frame at which delivery stopped, when applicable.

Request-resolution errors raise before a stream is returned. A runtime decode
failure sets the stream's failed outcome, closes its media session, and
re-raises the structured `SieveError`. Cancellation and explicit clean-EOF
truncation end iteration normally; the caller distinguishes them from success
by inspecting the required outcome.

Decoded delivery is not scientific examination. Do not use these terms for
source output:

```text
examined
processed
quiet
detected
```

For the current `rgb24` backend, a successful exact-sized frame establishes a
valid full-frame buffer. A failed frame is not replaced by zeros, black pixels,
or a validity mask. Do not build general spatial-validity machinery until a
backend can actually report partial spatial validity.

### EOF versus decode failure

The current `MediaSession` reports a short raw-frame read as
`FRAME_DECODE_FAILED`; it does not expose whether FFmpeg reached clean EOF or
failed decoding.

Therefore:

- Report `truncated` only when the media layer supplies an explicit clean-EOF
  or equivalent structured reason before the requested stop.
- Otherwise preserve the delivered prefix and report `failed` with the
  underlying error.
- Do not infer truncation from error-message text or an empty stderr string.
- A minimal structured EOF reason may be added to `MediaSession` if it can be
  implemented and tested without broad decoder redesign.

Cancellation is neither success nor truncation.

## 12. Player relationship and minimal GUI hookup

Player display requests and working-window requests have different semantics:

```text
player
  latest requested display frame
  width-limited representation
  mutable display-session decoder

working window
  ordered bounded delivery
  native rgb24 representation
  request-local media session
```

Do not pass the live player's `MediaSession` into the source. A working-window
read would reposition its mutable decoder cursor, and cancellation could
terminate the player's in-flight decode.

The only GUI hookup in this milestone is a pure snapshot operation:

```text
current registered asset identity
+ current Isolate [window_start, window_stop)
    -> immutable WorkingWindowRequest
```

Changing GUI state later must not mutate a previously created request. Taking a
snapshot must not open the working-window stream or decode frames.

Do not add a permanent Prepare, Process, or Run button. Do not add plots,
channel cards, success indicators, or fake processing states.

GUI job generations, request ids, cancellation ownership, stale-result
rejection, and cross-thread queues belong to the first milestone with a real
GUI scientific consumer.

## 13. Development diagnostic

Provide either a focused test helper or a development-only diagnostic accepting:

```text
registered asset reference
start
stop
plane = rgb24
batch size
optional deterministic cancellation point
```

It reports structured:

- Resolved asset id and recorded content identity.
- Identity-verification status.
- Media path.
- Requested span.
- Declared extent and provenance.
- Width, height, and exact rational fps.
- `rgb24` descriptor.
- Per-batch absolute indices and shapes.
- Delivered span.
- Final outcome and structured error, if any.

This is an implementation-validation surface, not the future processing CLI.
Do not add recipe parsing, export, job directories, or HPC behavior.

Because no cross-process bridge to live Qt state exists, the diagnostic does
not claim to consume another running GUI process's current private window.

## 14. Tests

Use deterministic tiny lossless fixtures. Reuse the numbered FFV1 fixture
direction for exact same-backend pixel comparisons and off-by-one detection.

### Headless request and resolution

Test:

- `[2, 5)` delivers only `2`, `3`, and `4`.
- Negative, empty, inverted, and out-of-declared-extent spans fail before
  frame decoding.
- A valid one-frame headless interval is accepted.
- GUI window-length limits do not constrain the headless model.
- A missing or unregistered sidecar fails without creating one.
- Expected asset-id or content-hash mismatch fails.
- Caller-supplied media facts do not exist in the request.
- `rgb24` is accepted and another plane id fails explicitly.

### Extent, time, and coordinates

Test:

- Each available count source retains its provenance.
- Duration-derived extent is labeled an estimate.
- Exact `60000/1001` or another non-integer fixture rate remains rational.
- Returned frame indices remain absolute when the request starts after zero.
- Native shapes and coordinates match the active asset.
- A child resolves its own media and dimensions.

### Batching, lifecycle, and outcomes

Test:

- Different batch sizes return identical ordered indices and bytes.
- The final short batch is correct.
- Yielded buffers remain immutable after subsequent reads.
- The stream retains only bounded request-local data.
- Normal exhaustion, cancellation, explicit close, failure, and early consumer
  exit close the request-local `MediaSession`.
- Cancellation is observed between bounded reads and yields a cancelled
  outcome.
- Explicit clean EOF yields truncation and a delivered prefix.
- Ambiguous current-media short read yields failure rather than guessed
  truncation.
- The final outcome remains accessible after iteration.

### GUI adapter and player regression

Run Qt tests with:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
```

Test:

- Current active asset and Isolate-local window produce the expected immutable
  headless request.
- Later asset/window changes do not mutate the earlier request.
- Taking or changing a window snapshot never starts a working-window decode.
- Existing player seek, loop, scrub, preview sizing, generation rejection, and
  cleanup tests continue to pass.

Do not create a GUI computation worker solely to test stale scientific results.

## 15. Manual acceptance

After automated tests pass, stop and return this milestone for user validation.

The executable manual path is:

1. Run the development diagnostic on a registered source asset and inspect its
   identity, extent provenance, timebase, native shape, batches, delivered
   span, and outcome.
2. Repeat on a derived child and confirm the child's media and dimensions.
3. Run the deterministic cancellation diagnostic and confirm a cancelled
   outcome and clean resource release.
4. Open the GUI and confirm the existing Isolate player still seeks, loops,
   scrubs, and uses its display-sized representation.
5. Change the GUI window rapidly and confirm no scientific working-window
   decode starts.

Live “change the GUI while a scientific job runs” acceptance is deferred until
a real GUI channel worker and consumer exist.

## 16. Definition of done

This milestone is complete only when:

- A non-Qt caller can resolve and consume a registered asset frame window.
- Importing the source contract does not import PyQt.
- The range is absolute, integer, half-open, and validated without clamping.
- Stable recorded asset/content identity and its verification status are
  explicit.
- Declared extent and provenance are explicit.
- Exact rational time and native asset coordinates are preserved.
- Only explicit native `rgb24` is supported.
- Delivery is synchronous, bounded, immutable, and independently closeable.
- The final source outcome remains inspectable.
- Decoded delivery is not mislabeled scientific examination.
- The GUI hookup snapshots state without launching processing.
- The player continues to use its independent display session and preview.
- No grids, channels, workers, queues, caches, recipes, persistence, exports,
  or general pipeline framework were introduced.
- Headless and offscreen GUI tests pass.
- The user has reviewed and accepted the milestone.

Stop here. The working grid is a later handoff and is not authorized by
completion of this milestone.
