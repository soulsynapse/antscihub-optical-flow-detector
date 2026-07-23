# Review — 3 Define the Isolate working window

Status: architectural and current-rewrite review of `3-Working-window.md`.

## Verdict

The handoff has the right long-term seam, but it is not safe to give to an
implementation agent unchanged.

Its strongest decisions are:

- The scientific source is headless and Qt-independent.
- The range is absolute, integer, half-open, and expressed in asset frames.
- A derived child is decoded from its own media in its own coordinates.
- Native scientific pixels remain separate from display-sized rasters.
- Pixel delivery is bounded rather than requiring one full-window allocation.
- The player, timeline, and working-window source have separate ownership.
- Downsampling, blocks, channels, transforms, detection, persistence, and
  recipes remain out of scope.
- The milestone explicitly stops before the working grid or first channel.

Those are durable decisions and fit both the rewrite's current player and the
later scientific-computation contract.

The unsafe part is that the handoff calls this a small pixel-delivery boundary
while also requiring several systems that the rewrite does not currently have:

- Headless asset generations and request ids.
- General named and multi-plane media delivery.
- A resolved-request layer.
- Batch and outcome artifact types.
- Valid and examined coverage.
- Cancellation and latest-request orchestration.
- A controller integration with no scientific consumer.
- A diagnostic that somehow shares live GUI state.

That combination risks building the first slice of a speculative executor
before the first channel has shown what it actually needs. The handoff should be
narrowed to one synchronous, bounded, native-resolution source contract, with
GUI supersession kept outside it.

## Important correction 1: keep controller generations out of the headless
## request

The handoff currently places both:

```text
asset_generation
request_id
```

inside `WorkingWindowRequest`, resolved requests, batches, outcomes, diagnostics,
and headless tests.

That conflicts with the handoff practices:

> Interactive generation tokens and latest-request policy stay in GUI
> orchestration; reusable headless contracts use stable asset/content identity.

It also conflicts with the current rewrite. `ActiveAsset` has no asset
generation. `IsolateSession._generation` is a private GUI-lifecycle counter used
to reject late `IsolateDecodeThread` signals. It is meaningful only inside that
particular live controller instance. A CLI, HPC process, test, or later
scientific worker cannot use it to establish asset identity.

Use two layers:

```text
headless source request
  stable asset/content reference
  half-open frame span
  requested supported plane

GUI job envelope
  controller generation
  request id, if needed
  cancellation token
  latest-result publication policy
```

The headless source should validate the asset and decode the requested evidence.
The GUI controller should decide whether a returned batch or outcome is still
current. Do not make the reusable source understand the current Qt session.

The current rewrite's `ActiveAsset` contains `asset_id`, paths, dimensions, and
timebase, but not `content_sha256`. The sidecar does contain the content hash.
The implementation should reuse the existing asset inspection path to resolve
the stable asset/content identity. It should not treat the private GUI
generation as a substitute, and it should not invent another asset identity
scheme.

Cancellation may still be a reusable execution concern. Pass a cancellation
token or predicate to iteration. Do not place transient cancellation identity
inside the scientific request.

## Important correction 2: decoded coverage is not examined coverage

The handoff repeatedly combines:

```text
successfully decoded/examined coverage
```

and asks the working-window outcome and diagnostic to report “examined”
coverage.

No scientific channel exists in this milestone. Decoding a frame does not mean
that SIEVE examined it for a signal, produced a valid channel value, or found it
quiet. If this terminology hardens into an artifact now, later result timelines
could incorrectly present pixel delivery as analysis coverage.

Use:

```text
requested frame span
delivered/decoded frame span
pixel validity, when the decoder can establish it
final source outcome
```

Reserve:

```text
examined
processed
quiet
detected
```

for later scientific results whose required nodes actually completed.

Early EOF can report a delivered prefix and a truncated source outcome. The
undelivered suffix remains not decoded by this request. It is also unexamined,
but that is a later consumer's conclusion, not coverage owned by the pixel
source.

This correction does not weaken the rule that missing frames must never become
black pixels, zeros, or negative detections.

## Important correction 3: match the first plane contract to the media service
## that actually exists

The handoff describes one or more requested named planes, multi-plane batch
alignment, encoded luma, native codec planes, alpha, ambiguity checks, and plane
maps:

```text
plane_id -> pixel buffer
```

Those are valid long-term scientific-computation goals, but they are not the
current rewrite media API.

The current `MediaSession` provides:

```python
read_frame_rgb(frame, max_width=None) -> bytes
```

The native scientific representation presently available is FFmpeg `rgb24`.
There is no media-plane id type, plane registry, multi-plane request, luma
decoder path, plane descriptor type, or batch API.

Do not make this milestone implement the general decoder contract merely
because the future platform document describes it. Support exactly one
explicitly named native-resolution plane in this increment:

```text
rgb24
shape [frame, y, x, channel]
dtype uint8
value domain [0, 255]
channel order R, G, B
active-asset coordinates
FFmpeg/backend and relevant source color metadata recorded
```

It is reasonable to introduce the smallest plane id and descriptor needed to
prevent `rgb24` from becoming an ambiguous unnamed byte string. It is not
reasonable yet to build multi-plane maps, alpha behavior, encoded-luma support,
or a general plane registry.

Phrase the future boundary as:

> The request names one plane supported by the current media backend. This
> milestone supports native-resolution `rgb24`. Later channels may justify
> additional plane ids or multi-plane requests without changing frame, time,
> coordinate, or validity semantics.

The first channel can then reveal whether encoded luma is actually required and
whether adding it improves correctness or measured performance.

Do not silently feed `max_width=1280` player previews into this source. The
working-window path must call the native-size media path.

## Important correction 4: resolve the provisional frame-extent contradiction

The handoff requires:

```text
0 <= start_frame < stop_frame <= verified_available_stop
```

and says out-of-range requests fail before decoding. It also says not to block
first use when verified availability is still being established.

The current rewrite cannot satisfy both requirements. `MediaSession.frame_count`
uses, in order:

1. Decoded count, when already available.
2. Packet count, when already available.
3. Container count.
4. Duration multiplied by rational fps.

Ordinary `MediaSession` construction does not run a full decoded-frame count.
Its extent can therefore be estimated. There is no current
`verified_available_stop` value against which every request can be rejected
before decoding.

Keep the distinction established by the player review:

```text
declared/navigable extent
  Current authoritative metadata for request admission, possibly estimated.

delivered decodable extent
  What this source request actually returned.

verified whole-asset extent
  A separately measured bound when later whole-asset processing requires it.
```

For this milestone:

- Validate negative, empty, inverted, and obviously out-of-declared-extent
  requests before decoding.
- Carry the extent value and its status, such as estimated or verified, in the
  resolved metadata.
- If the admitted request reaches real EOF early, return truncation and the
  delivered prefix.
- Do not label an estimated bound verified.
- Do not run a full-video count merely to open or validate a small window.

If the project instead wants every scientific request to require a verified
decoded count, that is a separate product decision with a potentially large
open-time cost. The current handoff must not imply both policies at once.

The tests should exercise both a known verified extent and an estimated extent
that truncates during decode.

## Important correction 5: timebase and media facts belong in resolution, not
## duplicated caller input

The outcome section says the request explicitly identifies the asset's exact
rational timebase. The minimal controller hookup likewise says to obtain the
timebase and form the request.

If the request already identifies an asset whose sidecar and media probe are
authoritative, accepting caller-supplied `fps_num`, `fps_den`, width, height, or
plane interpretation creates two sources of truth. The source would then need a
policy for mismatches such as:

```text
request says 60/1
asset sidecar says 60000/1001
live probe says 60000/1001
```

The headless caller should supply asset identity/reference, frame span, and
plane id. Resolution should obtain and validate:

- Content identity.
- Media path.
- Width and height.
- Rational fps.
- Current extent and extent status.
- Plane descriptor.

Returned batches retain absolute indices and the resolved exact timebase. A
caller may calculate exact presentation times from those values. Do not require
a separate floating-point timestamp array when frame indices plus the rational
pair are sufficient.

## Important correction 6: a synchronous iterator is the appropriate current
## streaming contract

The handoff mentions backpressure, bounded queues, slow-consumer queue tests,
workers, queued buffers, cancellation, and latest-only replacement. An
implementation agent could reasonably infer that this milestone requires an
asynchronous producer.

The current `MediaSession` already retains one sequential FFmpeg process for
adjacent `read_frame_rgb` calls and returns immutable `bytes`. The smallest
correct headless source is therefore a synchronous iterator:

```text
open/resolve source
for each bounded batch:
    check cancellation
    read adjacent native rgb24 frames
    yield immutable batch
close source
return or expose final outcome
```

A synchronous iterator has natural backpressure: it cannot produce the next
batch until the consumer asks for it. It needs no queue, producer thread, Qt
signal, or latest-only mailbox. Test boundedness by instrumenting the source's
owned buffers and by confirming it does not retain yielded batches, not by
requiring a slow-consumer queue test.

Batch size may remain an execution option. A batch of one is valid. The
implementation should not concatenate an arbitrary selected window merely to
make batches look array-like.

Controller concurrency can be added when a real channel consumes this iterator.
At that point the GUI worker owns:

- The cancellation token.
- The current generation/request id.
- The rule for ignoring stale output.
- Any bounded cross-thread queue.

Do not prebuild that worker while this milestone explicitly has no consumer.

## Important correction 7: do not share the current player's mutable decoder
## cursor accidentally

The handoff correctly leaves physical decoder ownership open in the abstract.
The current rewrite, however, has one mutable FFmpeg decoder cursor per
`MediaSession`, and `interrupt()` terminates that decoder.

Passing the live player `MediaSession` directly to a sequential working-window
iterator would allow:

- A scientific read to reposition the player's decoder.
- Player scrubbing to stop sequential window delivery.
- Cancelling a working request to terminate the player's in-flight display
  read.

For the implementation that exists today, the safe default is a separate
request-local `MediaSession` over the same resolved asset media. This is reuse of
the existing media service, not a competing service architecture.

If the current media-performance work introduces an explicit session-sharing or
independent-reader contract before this milestone begins, inspect and use that
new contract. Do not infer concurrent-reader safety from the fact that both
paths use the same class.

The prior “one decode per resolved display request” diagnostic does not require
sharing one mutable decoder between display and science. These are different
representations and consumers.

## Important correction 8: the proposed manual acceptance path is not
## executable as written

The handoff requires the user to:

1. Change a window in the GUI.
2. Run a headless diagnostic “on the settled window.”
3. Start a slow diagnostic.
4. Change the live GUI asset or range.
5. Observe that the diagnostic cancels or becomes stale.

At the same time, it forbids a button or CLI surface and does not define any
bridge through which a separate headless diagnostic receives the live
`IsolateSession` state. A standalone script cannot observe a private in-process
Qt generation or automatically react to window changes in another process.

Choose one coherent acceptance model.

### Recommended model for this milestone

Keep it wholly headless:

- Provide a development script or focused test helper accepting an asset
  reference, `start`, `stop`, plane, and batch size.
- Print resolved identity, extent status, rational timebase, plane descriptor,
  per-batch absolute frame spans/shapes, delivered span, and final outcome.
- Exercise cancellation through an explicit test cancellation token.
- In the GUI, verify only that existing player behavior is unchanged and that
  window dragging launches no working-window decode.
- Test a small pure method that snapshots the current active asset and
  `[window_start, window_stop)` into the headless request fields, without
  launching work.

Then defer live supersession and “change the GUI while a scientific job is
running” to the first channel milestone, where an actual GUI consumer and worker
exist.

An alternative is to add a deliberately temporary in-process developer action
that launches the diagnostic. That would make the cancellation acceptance
possible, but it creates GUI orchestration with no user feature and is not the
smaller choice.

## Current rewrite facts the implementation handoff should state explicitly

The review must be interpreted against the current `antscihub-SIEVE` checkout,
not only the oracle's future contracts.

### Already present

- `ActiveAsset` provides asset id, sidecar/media paths, dimensions, rational
  fps, duration, kind, and parent navigation metadata.
- The asset sidecar provides `content_sha256`.
- `MediaSession` probes media and provides exact `Fraction` timestamps.
- Adjacent `read_frame_rgb` calls reuse one FFmpeg decoder process.
- Native decode returns immutable `rgb24` bytes.
- `IsolateSession` owns the GUI window and a private generation counter.
- `IsolateDecodeThread` already coalesces display requests and rejects stale
  generations.
- Player display decoding uses a width-limited preview path.

### Not present

- A headless `WorkingWindowRequest` or frame-span value object.
- A stable headless asset-generation concept.
- General media-plane ids or descriptors.
- Encoded-luma, alpha, or native-codec plane requests.
- Multi-plane batch delivery.
- A scientific cancellation token or worker.
- General validity or coverage interval types.
- A scientific executor, recipe, or result artifact.

The handoff should ask for the smallest additions needed today rather than
writing the “not present” list as though all items were already established
rewrite contracts.

## Testing clarifications

### Use the existing lossless fixture direction

The rewrite already uses FFV1 fixtures. Keep exact pixel comparisons on a
lossless deterministic fixture with an explicitly selected output format.
Frame-number encoding remains useful for seek and off-by-one errors.

Do not use exact hashes from a lossy fixture as a cross-backend plane contract.
If a later backend intentionally changes conversion behavior, temporal identity
and pixel equivalence remain separate assertions.

### Separate source tests from controller tests

Headless source tests should cover:

- Half-open range validation.
- Absolute frame indices.
- Exact rational timebase.
- Native `rgb24` descriptor, shape, and values.
- Batch-size invariance.
- Final short batch.
- Synchronous boundedness.
- Immutable yielded buffers.
- Explicit cancellation checks between bounded units.
- Early EOF and structured decode failure.
- Child media and child coordinates.

Pure controller/request-snapshot tests should cover:

- Current active asset and local window produce the expected source request.
- Changing GUI state does not mutate an already-created request.
- Window dragging does not eagerly consume the source.

Live stale-result tests belong with the first GUI computation worker. Do not
create a worker solely to satisfy tests for a feature this handoff says not to
run.

### Do not overclaim widget-independence tests

A headless source has no access to widget size, letterboxing, zoom, or
device-pixel ratio. Its native-size test already establishes that those GUI
properties cannot affect its output.

One GUI regression asserting that the player continues to use its display path
is useful. A large matrix of widget resize/zoom tests is not needed for the
headless source milestone.

## Decisions that are safe as written

### Half-open absolute frame ranges

`[17, 20)` meaning frames 17, 18, and 19 is the right contract. It matches the
player model, future coverage intervals, and scientific time alignment.

### Child assets are complete processing worlds

A derived child must be decoded from its own media and use:

```text
[0, child_width) x [0, child_height)
```

The working-window source must not reopen the parent and recreate a crop. This
preserves the central performance and ownership goal of the rewrite.

### Native scientific pixels are not display pixels

The player may request a width-limited RGB preview. The working-window source
must use native asset dimensions. This is a necessary boundary even before
downsampling is added.

### Downsampling and the working grid remain deferred

The source should return native evidence. The next milestone can define
downsample and block geometry without forcing the media layer to guess future
scientific settings.

### Bounded delivery is required

Later windows and whole assets may exceed memory. Establishing an iterator now
is appropriate as long as it remains a small synchronous source rather than the
beginning of a speculative general executor.

### No persistent pixel cache

There is no measured need for a new scientific pixel cache in this milestone.
The iterator should release request-local decoder and buffer ownership
promptly.

## Risk classification

### Potentially expensive if implemented as written

1. Embedding GUI request generations in the reusable headless contract.
2. Building a general multi-plane media API before one required plane has been
   exercised by a channel.
3. Creating an asynchronous worker, queue, and latest-only controller for a
   source with no consumer.
4. Building validity and coverage artifact machinery while calling decode
   coverage “examined.”
5. Sharing the player's mutable decoder cursor with scientific iteration.
6. Requiring a full decoded-frame count before every small window request.

### Correctness risks if left ambiguous

1. Treating an estimated frame extent as verified.
2. Letting caller-supplied timebase metadata disagree with the resolved asset.
3. Returning the 1280-pixel display preview as scientific evidence.
4. Treating missing source color metadata as either silently harmless or
   universally fatal without an explicit current policy.
5. Reporting early EOF as successful complete coverage.
6. Allowing pixel delivery to imply quiet or examined footage.

### Cheap to extend later

1. Additional named plane ids.
2. Multi-plane batches.
3. Batch sizes larger than one.
4. A GUI computation worker and bounded cross-thread queue.
5. Latest-only publication ids around that worker.
6. Richer validity representations if a backend can produce partial spatial
   validity.
7. A supported CLI or HPC recipe.

### Correctly deferred

1. Downsampling and block geometry.
2. Normalization and derived color spaces.
3. Intensity, change, tensor, flow, or learned channels.
4. Static and spectral filtering.
5. Detection and event construction.
6. Persistent scientific artifacts and accumulated result coverage.
7. General graph planning, plugin execution, recipes, CLI, and HPC.

## Recommended disposition

Revise `3-Working-window.md` before implementation.

The corrected milestone should require:

```text
stable resolved asset/content identity
+ absolute half-open frame span
+ one explicit native rgb24 plane
+ optional execution batch size and cancellation token
    -> synchronous bounded iterator
    -> immutable native-resolution frame batches
    -> delivered span and structured final source outcome
```

Keep controller generation, request ids, latest-only publication, and GUI worker
lifecycle outside the headless request. In this milestone, add only a pure
request snapshot from current Isolate state; do not launch window decoding on
GUI gestures.

Replace “examined coverage” with “delivered/decoded span,” reconcile estimated
versus verified frame extent, and make the diagnostic an explicit development
script or test helper whose inputs can actually be supplied.

After those corrections, the handoff becomes a small, reversible foundation for
the working grid and first channel without prematurely implementing the future
executor.
