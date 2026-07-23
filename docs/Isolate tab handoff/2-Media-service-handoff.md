# 2 — Media-service performance hints

Status: performance handoff for the v2 media service and Isolate player.

The rewrite's current media viewer is noticeably slower than the oracle
application's viewer. This document records mechanisms that helped the oracle
feel responsive and proposes benchmarks for locating the rewrite's actual
bottleneck.

These are **hints, not implementation instructions**.

Do not replace a correct v2 mechanism merely because the oracle used something
different. Do not weaken asset identity, rational timekeeping, cancellation,
decoder lifecycle, frame accuracy, portability, or error reporting to reproduce
an old optimization. Benchmark the rewrite before a change and after the same
change on the same media. Keep a change only when it produces a meaningful
end-to-end improvement without breaking behavior.

The oracle is useful evidence, not a design authority.

## 1. Objective

Make these Isolate-player actions feel immediate:

- Opening an asset and seeing its first frame.
- Playing forward at the asset's native rate.
- Stepping one frame at a time.
- Clicking a distant point in the timeline.
- Scrubbing rapidly and seeing the final requested frame promptly.
- Moving the looping window.
- Looping from the window end to its beginning.
- Switching to another asset.

The benchmark target is not merely high decoder throughput. A viewer can decode
quickly in isolation and still feel slow because it reopens the decoder, seeks
for adjacent frames, queues obsolete scrub requests, converts full-resolution
pixels unnecessarily, or repaints hidden widgets.

Measure both:

1. The media service by itself.
2. The complete GUI path from user request to the frame being painted.

## 2. Preserve correctness first

Before performance work, establish tests for:

- The requested frame is the displayed frame.
- Sequential playback neither duplicates nor skips frames unexpectedly.
- A random seek does not drift progressively with timestamp rounding.
- The final scrub request wins over every superseded request.
- A late frame from an old asset cannot appear after an asset switch.
- The first and final decodable frames are handled correctly.
- Looping never displays a frame outside the selected half-open window.
- Cancellation and close release the decoder and any helper threads/processes.
- Letterboxing and source-coordinate mapping remain correct.
- Decode errors remain visible and do not trigger an undeclared backend or
  parent-asset fallback.

If a proposed speedup fails one of these checks, reject it even if its timing is
better.

## 3. Benchmark before changing the implementation

Create one repeatable benchmark command or script in the rewrite. It should be
usable without manually operating the GUI and should also have an end-to-end GUI
mode.

Suggested commands:

```powershell
sieve media benchmark ASSET --mode service --json
sieve media benchmark ASSET --mode viewer --offscreen --json
```

Equivalent test scripts are acceptable. The important requirement is stable,
machine-readable measurements.

Record:

```text
asset id and content identity
codec, container, dimensions, bit depth, and rational fps
verified decodable frame count
decoder backend and exact backend configuration
display size
warm or cold run
operation being measured
latency distribution, not only mean
number of requested, decoded, delivered, painted, dropped, and superseded frames
peak resident memory
CPU utilization where available
software and dependency versions
```

Run each scenario several times. Report the median and a tail measurement such
as p95; a viewer that is usually fast but pauses badly every few seeks still
feels slow.

### 3.1 Media corpus

Use at least:

1. A small deterministic synthetic fixture with a visible and machine-verifiable
   frame number encoded into every frame.
2. A representative isolated replicate child.
3. A long-GOP source file on which random seeks are expensive.
4. A high-resolution source representative of real project footage.
5. A high-frame-rate clip, preferably near 120 fps, because costs hidden at
   24–30 fps become obvious against an 8.33 ms frame budget.
6. A short clip for testing loop-boundary and end-of-stream behavior.

Keep these files and their expected frame hashes stable. Do not compare two
implementations using different transcoded copies and call the result a decoder
comparison.

### 3.2 Required scenarios

#### Open-to-first-frame

Measure from the open request until the first correct frame is painted:

- Cold process and cold asset.
- Same process reopening the asset.
- Source asset and child asset.

Break the time down into asset resolution, media probe, decoder creation, first
decode, display conversion, and paint.

#### Sequential playback

Starting at frame zero and at a nonzero frame:

- Play at native rate for at least 10 seconds.
- Repeat without timer throttling to measure maximum sequential throughput.
- Report achieved footage-seconds per wall-second.
- Report missed deadlines, decoded frames, painted frames, and dropped display
  updates separately.

The native-rate test answers “does the viewer keep up?” The unthrottled test
answers “how much headroom exists?” They are not interchangeable.

#### Adjacent stepping

Request `N`, then `N+1`, repeatedly for at least 100 frames. Report per-step
latency and verify the service did not perform a random seek for every step.

Also test alternating `N`, `N+1`, `N`, because backward stepping may
legitimately require a different strategy.

#### Random seeking

Use a fixed deterministic list of positions near:

- The beginning.
- The middle.
- The end.
- Several presumed long-GOP boundaries.

Measure cold and warm seek latency and verify each returned frame against the
numbered fixture or expected hash.

#### Rapid scrub

Simulate a realistic pointer drag that emits at least 100 intermediate
positions over approximately one second and ends on a known frame.

Record:

- Time from release to the final frame being painted.
- Number of intermediate requests.
- Number of actual seeks/decodes.
- Number of frames painted after they were already obsolete.
- Whether the final requested frame was displayed.

The desired behavior is latest-request responsiveness, not decoding every frame
the pointer crossed.

#### Loop boundary

Loop a short window repeatedly:

- A window whose beginning is near a keyframe.
- A window whose beginning is likely inside a GOP.
- A two-frame minimum window.
- A window ending at the asset's final frame.

Measure the ordinary sequential frame latency separately from the
`stop - 1 -> start` loop seek. A visible pause once per loop is a different
problem from generally slow playback.

#### Asset switch and cancellation

Start a slow random seek, switch assets immediately, and verify:

- The old request is cancelled or ignored.
- The new asset's first frame appears promptly.
- No old frame paints afterward.
- The old decoder and helper resources close.

#### Resize and repaint

Hold one decoded frame and resize/repaint the viewer repeatedly without
requesting new media. This isolates GUI conversion and paint cost from decoding.

Also change frames without resizing to measure the ordinary
decode-to-display-conversion path.

#### Hidden-view overhead

Play or step in the active view while another media widget exists but is hidden.
Verify the hidden widget causes no duplicate decode and negligible repaint work.

## 4. Oracle hint: keep one decoder session open

The oracle opens one `VideoCapture` in `VideoSource` and retains it for the
session. It does not reopen the video for every frame.

Why this may help:

- Decoder initialization, probing, and keyframe indexing are not repaid on every
  request.
- The decoder retains its current stream position.
- Adjacent reads can continue sequentially.

What to benchmark:

- Reused-session versus reopen-per-request first-frame latency.
- Memory and handle use after repeated asset switches.
- Whether the rewrite already reuses a session and the perceived delay comes
  from another layer.

Do not adopt:

- A process-global decoder shared by unrelated jobs.
- A decoder handle that outlives its asset session.
- Direct widget ownership that prevents reliable cancellation or close.

The v2 media service should retain its existing lifecycle and ownership
discipline.

## 5. Oracle hint: distinguish sequential reads from random seeks

The oracle tracks the decoder's next expected frame. Its `frame_at(N)` path
performs a plain `read()` when `N` is exactly the decoder's current position and
seeks only when it is not.

This was load-bearing on long-GOP footage. The oracle notes that an accidental
seek per frame reduced playback to only a few frames per second because the
decoder repeatedly jumped to an earlier keyframe and decoded forward.

One measured oracle case at 1920×1080 and 120 fps took approximately:

- `2.70 ms` for an adjacent sequential decode.
- Roughly `400 ms` for some random seeks on the representative GoPro footage.

These are historical measurements from one machine and media set, not v2
acceptance thresholds.

What to benchmark:

- Adjacent `read` versus explicit seek-to-next-frame.
- Whether the rewrite's current abstraction unknowingly resets decoder position
  or creates a new decode command for every displayed frame.
- Forward stepping, forward playback, backward stepping, and loop restart
  separately.

Do not change a frame-accurate seek implementation until the replacement has
passed frame-hash tests across the clip.

## 6. Oracle hint: decode a requested display frame once

The oracle once had multiple tabs independently ask for the same frame. The
first request decoded it and left the decoder at `N+1`; the second asked for
`N`, missed the sequential fast path, and caused a full random seek.

Its shared display path therefore retains:

- The most recently decoded full-resolution frame and its index.
- A downscaled display copy of that same frame and its index.

Every consumer of that displayed frame receives the already-decoded result.

For v2, interpret this narrowly:

- Within one Isolate asset session, the player, timeline-related previews, and
  future overlays should not independently decode the same frame.
- Do not reintroduce cross-tab mutable state or share one decoder between
  independent scientific jobs.

What to benchmark:

- Count actual decoder reads for one requested GUI frame.
- Add the timeline, metadata display, and empty channel area one by one and
  verify none causes another decode.
- Compare no frame cache, a one-frame cache, and any existing rewrite cache.

A one-frame cache may be sufficient. Do not build a large cache without evidence
that revisit patterns justify its memory and invalidation complexity.

## 7. Oracle hint: coalesce scrub requests

Qt sliders can emit a value for every intermediate position during a drag. The
oracle originally decoded each one. On footage with approximately 400 ms random
seeks, this created a long backlog that looked like the application was slowly
replaying the entire drag.

The oracle uses a short, single-shot settle timer:

- Update the time/frame readout immediately.
- Remember only the latest requested frame.
- Restart a roughly `60 ms` timer on each intermediate value.
- Decode after the handle has settled.

The exact `60 ms` is only a hint.

The rewrite may instead use cancellation tokens, a latest-only request mailbox,
or a decoder worker that discards superseded results. Benchmark candidate
strategies under the rapid-scrub scenario.

Required semantics:

- Obsolete requests do not build an unbounded queue.
- A superseded decode result cannot overwrite a newer one.
- The final requested frame appears promptly after release.
- The visible time readout may update before the expensive decode finishes.

Do not debounce ordinary sequential playback or one-frame keyboard stepping so
aggressively that those interactions feel delayed.

## 8. Oracle hint: coalesce repaint requests too

The oracle's `VideoPanel.refresh()` allows only one repaint request per event-loop
turn. Multiple signals describing one frame change therefore collapse into one
decode/convert/paint cycle.

It also returns without doing display work when the panel is not visible.

What to benchmark:

- Number of refresh requests versus actual paints for one frame change.
- CPU cost with the Isolate tab visible and hidden.
- Whether state changes produce duplicate frame conversions even when only one
  decode occurs.

Do not skip required state updates for hidden widgets. Skip expensive rendering,
not authoritative model changes.

## 9. Oracle hint: reduce to display resolution before display-only work

The oracle's real source footage is approximately 5312×2988 while the viewer is
much smaller. It keeps the decoded source frame available when native pixels are
needed, but creates a display copy capped at 1280 pixels wide.

Its historical measurement found that copying, blending, converting, and
displaying overlays at full 5K resolution cost roughly `250 ms/frame`. Performing
display work at 1280-pixel width reduced pixel traffic by about 15× and brought
the viewer close to realtime.

What to benchmark:

- Full-resolution versus widget-sized or capped display conversion.
- Several caps, including the actual device-pixel-ratio-adjusted widget size.
- Resize frequency and the cost of rebuilding a display buffer.
- Visual equality at the displayed size.

Preserve:

- Full-resolution scientific pixels outside the display path.
- Accurate source-coordinate mapping.
- Native-detail behavior if a future zoom/focus view requires it.
- Aspect ratio and color interpretation.

Do not globally downsample the media source or pass display pixels into
scientific computation as a performance shortcut.

## 10. Oracle hint: avoid unnecessary full-frame copies and conversions

The oracle display path currently performs a BGR-to-RGB conversion and creates a
Qt-owned image copy before painting. The copy is deliberate: it prevents a
`QImage` or `QPixmap` from referring to NumPy memory whose lifetime has ended.

Possible rewrite improvements include:

- Requesting a display-ready plane from the media service.
- Reusing a stable display buffer.
- Avoiding repeated BGR/RGB conversion when the backend can produce the desired
  layout.
- Avoiding an extra copy before a later mandatory copy.

These are only candidates. A superficially attractive zero-copy `QImage` can
become a use-after-free or display corruption bug.

Benchmark the complete conversion-and-paint candidate, not isolated array
operations, and retain explicit ownership tests.

## 11. Oracle hint: keep frame conversion out of `paintEvent`

The oracle converts a newly decoded frame into a `QPixmap` once in `set_frame`.
`paintEvent` draws the stored pixmap into an aspect-fitted destination rectangle.
Repainting an unchanged frame does not reconvert BGR to RGB or reconstruct the
pixmap.

What to benchmark:

- Repainting an unchanged frame.
- Installing a new frame at a fixed widget size.
- Resizing with the same frame.
- Whether the rewrite reconstructs image objects on every paint.

Cache only what has a clear invalidation key: frame identity, display conversion
parameters, and possibly target size. Incorrect image caches are worse than slow
ones.

## 12. Oracle hint: keep playback latest-only

The oracle's timer advances the intended frame rather than queuing an independent
long-running task per tick. Elsewhere in the oracle, display refreshes are
allowed to drop while scientific computation must not; the same distinction is
useful here.

For a viewer:

- It is acceptable to skip an obsolete paint when the UI falls behind.
- It is not acceptable to silently corrupt frame numbering or scientific
  coverage.
- A timer should not create a growing queue of pending decodes.

Benchmark at 24, 30, 60, and 120 fps where fixtures permit. Report deadline
misses and achieved footage/wall-time ratio.

Consider an in-view rate meter like the oracle's:

```text
1.00x realtime
0.72x realtime
```

The oracle measures footage seconds advanced per wall second over approximately
one-second windows and resets the measurement after pause. Such a meter is
useful diagnostics, but it is optional for this milestone and must not itself
become significant per-frame work.

## 13. Oracle hint: bounded prefetch may help, but is not automatically a viewer fix

The oracle uses a producer thread with a small queue for sequential scientific
extraction. On one 479-frame, seven-replicate, 5.3K processing benchmark,
prefetch improved `15.14 s` to `10.21 s` by overlapping decode with downstream
math.

That result does not prove that UI playback needs prefetch:

- The viewer may have little work to overlap.
- A prefetched queue can consume large amounts of RAM.
- Random scrubbing invalidates queued sequential frames.
- Looping repeatedly may favor a different strategy.
- Cancellation and decoder ownership become more complicated.

If testing prefetch:

- Keep the queue small.
- Include queued, producer-held, and consumer-held frames in the memory estimate.
- Ensure close joins the producer before releasing the decoder.
- Forward decoder failures.
- Flush or generation-stamp queued frames after seeks and asset switches.

Compare it against the simpler persistent sequential decoder before retaining
it.

## 14. Oracle hint: do not blame the media service without isolating paint work

The oracle diagnosed a visible playback slowdown that initially looked like
channel computation. End-to-end timings showed that per-frame overlay and plot
painting were responsible after the channel data became available.

Relevant oracle measurements included:

- Pre-overlay display path: `4.6 ms/frame`.
- Old overlay path: `12.2 ms/frame`.
- Optimized overlay path: `6.8 ms/frame`.
- Overlay disabled: `4.7 ms/frame`.

The optimization was found by timing complete candidate implementations; an
isolated microbenchmark had incorrectly exonerated the expensive operation.

For the rewrite, instrument at least:

```text
request queue delay
seek/decode
display resize
color conversion
Qt image/pixmap construction
overlay preparation
paint
end-to-end request-to-paint
```

Do not rewrite the decoder because the combined player is slow until these
stages show that decode or seeking is actually the limiting part.

## 15. Oracle hint: hidden and collapsed visuals should not paint

The oracle found that merely making a plot geometrically small did not make it
cheap. Truly collapsed plots skip `update()` and return early from `paintEvent`.
Similarly, hidden video panels skip their refresh path.

This matters more when channels are added to Isolate, but it is worth preserving
in the player now:

- An invisible Isolate tab should not repaint video.
- An empty Channels area should not schedule frame work.
- Future collapsed channel panels should not recompute display rasters merely
  because the playhead moved.

Do not use widget visibility as a substitute for scientific scheduling. This
hint concerns display work only.

## 16. Things in the oracle not to copy blindly

The oracle's current `VideoSource` is intentionally small and contains shortcuts
that are inappropriate as a v2 contract:

- It stores fps as a float rather than an authoritative rational.
- It falls back to 30 fps when the backend reports no rate.
- It uses OpenCV's container frame-count property directly.
- Its quick video hash is not a full content identity.
- It has limited structured error reporting.
- Its random seek behavior inherits the backend's semantics.
- Its GUI cache is centralized in a broad application state object because that
  suited the old two-tab architecture.

V2 should retain:

- Exact rational timing.
- Verified asset and media metadata.
- Explicit backend identity.
- Structured failures.
- Active-asset/session boundaries.
- Latest-request cancellation and generation checks.

Copy the performance idea only when it fits those contracts.

## 17. Recommended experiment order

Run the baseline suite first, then test one hypothesis at a time:

1. Confirm whether the decoder/media session is reopened per frame.
2. Confirm whether adjacent playback uses sequential decode.
3. Count actual decodes per requested displayed frame.
4. Add or verify latest-only scrub coalescing.
5. Separate decode time from resize/conversion/paint time.
6. Test early display-resolution reduction.
7. Coalesce duplicate refresh requests and skip hidden paints.
8. Test stable frame/pixmap caching.
9. Test bounded prefetch only if sequential decode still leaves useful work
   unoverlapped.
10. Profile future overlays and channel plots separately from media decode.

After each candidate:

```text
run correctness tests
run the same benchmark corpus
compare service and viewer results
inspect peak memory and resource cleanup
keep or revert the candidate
record the result, including rejected ideas
```

Do not stack several speculative optimizations before measuring. If performance
improves, the responsible change should be identifiable.

## 18. Suggested acceptance criteria

Set numeric thresholds only after collecting a baseline on the target
development machine. At minimum, require:

- Correct frames for every seek fixture.
- No stale frame after superseding a request or switching assets.
- No growing request queue during a scrub.
- Final scrub frame painted within an agreed p95 latency.
- Native-rate playback at 1.0× on representative assets where raw sequential
  decode plus display work fits the frame budget.
- A documented explanation for assets whose codecs or dimensions make 1.0×
  impossible.
- No duplicate decode for one displayed-frame request.
- Hidden media widgets produce no material decode or paint load.
- Memory remains bounded during playback, scrubbing, looping, and repeated asset
  switches.
- All decoder/helper resources close cleanly.

Do not set “must match the oracle” as an acceptance criterion. The rewrite may
have stronger correctness and lifecycle guarantees, a different decoder, or
different display requirements. Its goal is responsive, measured behavior under
its own contracts.

## 19. Definition of done

This handoff is complete when:

- The rewrite has a repeatable service-level and viewer-level benchmark.
- Baselines exist for the representative corpus.
- The slow stage or stages are identified with timings.
- Oracle ideas are tested individually rather than transplanted wholesale.
- Kept changes show a before/after improvement.
- Rejected changes and their measurements are recorded.
- Frame accuracy, rational timing, cancellation, asset switching, and cleanup
  remain correct.
- The user validates that the real Isolate player feels responsive.

Stop after improving and validating the media service. Do not use this
performance work as permission to begin channel or isolation processing.
