I'm writing this while waiting for the usage window to renew.

- remove the 'band' info from the orange text at the top.
- remove the live preprocessing text at the top - it is eating like 1cm of space for no reason
- add a thing to log how long processing steps that were taken actually take between steps so we can see where to optimize the app
- we should figure out which parts that take a long time can be parallelized (for cpu) or done faster on gpu. this is *mostly* the extraction steps.
- if an extraction is happening (either the 10s or the full), those buttons should change to 'stop' so that i can adjust settings without having to close everything down or wait for it to finish
- we can shelve flow cache tab for now (essentially stop maintaining it for now)
- we can also shelve the behavior tab for now (no need to maintain either, but might be useful later for inspiration)
- the plots themselves can actually be the main source of lag for the program if I set something to 1 and downsampling is weak enough, even for 10 seconds it can be a massive result. this is likely even more true when the 'full processed video' has a way to scrub through the footage, which is a tab we probably have to work on.
- if fixed size stamp is enabled, it should be the stamp of the currently selected replicate. if I draw a new box, that replicate should be selected (and thus the current stamp).
- if i click a replicate on the replicate tab, it should (1) highlight that replicate (on the right) and (2) zoom in on that replicate. If I drag a replicate's box, it should let me reposition that box on the overlay. If I right click a replicate box, it should remove it - but only on the replicate tab.
- per block band power by channel currently displays black blocks for the plots when not populated - those should be collapsed down to the size of the checkbox so they don't take so much space. the name for the plot itself is the height of the checkbox, so this works out nicely.
- in line with that, i should be able to collapse and expand the other plots. e.g., blocks in band should have [+] next to it, so that it expands, but it should be collapsed by default. if they're collapsed, it shouldn't render the plot, saving memory.
- change the positive detection graph to just be green bands that are overlaid on the windowed # blocks in band.
- if a detection happens, even if I'm holding shift, it should have a green background (bolded black text) in the bottom right of the viewer box that says 'DETECTED'
- when the whole video is processed, it shouldn't be resetting the detection threshold bands when i move around the video
- it seems like when i select process the whole video, it *might* be processing change energy no matter what, even if only tensor speed is what was passed over to 'process the whole video'
- even if downsampling is auto, it should say how much it downsampled. so if it says auto it should be like, auto (number).
- we can do away with the replicate drop menu, it is entirely redundant with the clicking navigation
- it seems like there's some kind of runaway memory issue on the replicate tab somewhere.
- we need to make a 'viewer' tab that ingests the output of fully processing the video, but has none of the stuff from the preprocessing tab - it just says when there was a detection, and when there was not. This should have the output bar that comes up at the bottom once a video is fully processed (currently), but then a second one that is zoomed in on a like, 1 minute window where the scrubber is currently. It should also have the ability to pass it *back* to the preprocessing tab in the middle of wherever it was paused so I can see how it processes it, rather than the current version where whenever I click on the timeline it will start the processing right away.
- (23) ROI pre-transcode: cut each source once into per-replicate clips at working resolution + a manifest, so every later pass decodes ~1/16 the pixels instead of re-decoding 5.3K. Decode is the throughput wall; this is the one lever that removes it.
- (24) headless batch driver: a no-GUI entry point (video + replicate JSON -> detection results) and file-level job partitioning, so a run fans out across HPC nodes. Scaling is linear across nodes and flat across cores on one box.
- (25) replicate-aware auto downsample: resolve the scale from the widest replicate rather than the source width. Correctness, not preference - a pre-cropped video and an equivalent uncropped box currently resolve to different physical scales, so their results are not comparable. Sweep DEFAULT_TARGET_WIDTH first to price it.

---

## Execution plan (batched for cheap context)

Items above are referenced by their line number in this file. Batches are grouped
by *file locality*, so each one loads one or two files and can be done in a fresh
context. Do one batch, review, commit, `/clear`. Run python via
`.venv\Scripts\python.exe`.

Verified anchors (so a fresh session skips the search):
- top strip / workers / whole-video commit: `gui/explorers/live_scalogram_surface.py`
  (groupbox title L182, downsample spin L206-215, `_build_cfg` L283, `extract` L301,
  `process_whole_video` L433, workers L47-105)
- orange status line composition: `gui/explorers/scalogram_explorer.py:830 _update_status`
  (the `band … Hz` part is L844-845); plot construction L581-645;
  `_apply_selected_plot_ui` L672 already calls `dp.set_expanded(...)`, so a collapse
  mechanism exists on the density plots and can be generalized.
- auto downsample resolution: `core/config.py:73 resolve_downsample`
- extraction entry point + progress hook: `core/channel_source.py:109 live_channel_source`
  → `extract_channels_live` in `core/tensor_channels.py`

### Batch A — strip cosmetics (todo 3, 4, 19) · 1 file + 1 line elsewhere
Drop `band … Hz` from `_update_status`; it is already drawn on the scalogram plot
title. Replace the `QGroupBox("Live preprocessing · no flow cache")` with a bare
`QWidget` + margins to reclaim the vertical band. Downsample spinbox: compute
`PreprocessConfig(downsample=None).resolve_downsample(width)` once in `__init__`
and set `setSpecialValueText(f"auto ({scale:.3f})")`. Cheapest batch; do it first.

### Batch B — cancellable extraction (todo 7) · live_scalogram_surface + tensor_channels
Give the worker threads a `cancel()` that sets a flag checked in the per-frame
progress callback; raise a `_Cancelled` inside the callback and swallow it in
`run()`, emitting nothing. Extract/Process buttons become Stop while their pass
runs. Prereq for comfortably tuning anything else, so do it early.

### Batch C — step timing (todo 5) then hot-spot work (todo 6)
Add a small `core/timing.py` context manager that accumulates named spans and logs
one line per pass (decode / preprocess / tensor solve / block reduce / wavelet /
detect). Instrument `extract_channels_live` and `detect_channel_region` only.
**Land C before 6** — do not guess at parallel/GPU targets without the numbers.
Likely candidates once measured: frame decode overlapped with the tensor solve, and
the per-pixel eigen-solve.

**Why this is the keystone batch, not a comfort item.** The real blocker is
replicate-aware auto downsample (see `tests/test_auto_downsample.py`, shelved with
a full writeup). Resolving the scale from the widest replicate rather than the
source width is ~4x linear / ~16x pixels through the per-pixel tensor solve, which
is what made a previous attempt unusable. Extraction already crops per tile
(`_stream_channels` builds a `Preprocessor` per `source_box`), so cropping does not
offset it — the tile is exactly what gets scaled up.

That shelved behaviour is a **correctness** gap, not a quality preference: today a
pre-cropped video and an equivalent uncropped box resolve to different physical
scales, so detection results are not comparable across those two workflows. The
`auto (0.245)` now shown by Batch A is the source-width number, i.e. provisional.

Before reaching for GPU, measure the cheaper lever: `DEFAULT_TARGET_WIDTH` (1300)
is what forces the 16x. Detection reads *block-level* band power over time, so the
spatial resolution it truly needs is set by localization, not by target width.
Halving the per-replicate target is 4x rather than 16x and may cost no sensitivity
at all. Sweep target width against detection output on a known clip first; only
parallelize the residual.

### Batch D — plot cost + collapse (todo 13, 14, 10) · scalogram_explorer only
Generalize the existing expand/collapse into a `[+]` header on every plot, collapsed
by default, and make a collapsed plot skip its paint *and* drop its cached array.
Unpopulated per-channel heatmaps collapse to checkbox height. This is also the fix
for todo 10 (plot render is the lag source at block=1), so treat them as one batch.

### Batch E — detection readout (todo 15, 16) · scalogram_explorer + video_panel
Replace the separate positive-detection plot with green bands overlaid on the
windowed-#-blocks-in-band plot. Add a `DETECTED` badge (green bg, bold black) in the
viewer box's bottom-right that survives the shift-held path.

### Batch F — whole-video correctness (todo 17, 18) · live_scalogram_surface + channel_source
Two suspected bugs, both need confirming before fixing: (17) detection threshold
bands reset on navigation — check the `_focus_frame` → `extract` → `_swap_explorer`
view-state capture/restore round trip. (18) change energy may be computed even when
only tensor_speed was requested — `LIVE_CHANNELS` looks unconditional in
`live_channel_source`, so the commit pass likely computes every channel regardless of
`channel_attr`. If confirmed, thread the selected channel through. 18 is a real
speed win, so it pairs naturally with Batch C.

### Batch G — replicate tab (todo 11, 12, 20, 21) · tab2_replicates + video_panel
Click-to-select + zoom, drag to reposition, right-click to delete (that tab only),
fixed stamp follows selection, new box selects itself, drop the redundant dropdown.
The memory leak (21) is likely an overlay/pixmap not released per selection — chase
it last, with the rest already simplified.

### Batch H — viewer tab (todo 22) · new file, largest
New tab consuming a `DetectionResult`: full-clip detection bar plus a ~1 min zoomed
bar following the scrubber, and an "open here in preprocessing" handoff that seeks
the live surface without auto-triggering a pass. Depends on B (cancel) and F (state
retention). Do last.

### Status (append only — item line numbers above must not shift)

- **Batch A — done**, commit `56fe5a6`. Also rewrote
  `tests/test_live_scalogram_surface.py`, which had been written against a
  decoded-`frame_window` cache that `90da164` replaced with the per-pixel
  channel cache; it never passed in a commit.
- **Batch B — done**, commit `40c7cac`. `_StreamWorker` base gives both passes a
  `cancel()` polled from the progress tick. Two deviations from the plan above:
  a cancelled pass emits `cancelled` (emitting nothing leaves the buttons stuck
  disabled), and a knob edit mid-extract *supersedes* that pass rather than
  being dropped — live extract only, never the whole-video commit.
  Fixed in passing: `closeEvent` did not stop the debounce timers, so a knob
  edited just before a close fired `extract()` into a dying widget. That was the
  pre-existing ~1-in-3 Qt crash (exit 9) in the full suite.
  **Open question:** a stop during the detector phase discards a completed
  `DetectionResult`, because `detect_channel_region` has no cancel point and the
  trailing `if self._cancel` throws the finished result away. Delivering it may
  be better — the expensive extraction is already paid for.
- **Batch C (todo 5) — done.** New `core/timing.py`: one `Timer` per pass, named
  spans, one log line at the end. Spans are preallocated reused objects (entered
  ~30k times a pass), so a name must not nest inside itself. On by default;
  `OFD_TIMING=0` disables. Logger `ofd.timing` self-attaches a stderr handler and
  sets `propagate = False`, so `caplog` cannot see it — `tests/test_timing.py`
  captures on the named logger instead. Instrumented `_stream_channels` (decode /
  preprocess / tensor_products / tensor_blur / flow_solve / appearance / texture /
  block_reduce / progress_cb) and the detector.
  Three review fixes applied: `detect_over_blocks` now logs *after* the
  count/gate/clump chain (logging before it hid the per-frame clump loop);
  `_stream_channels` logs from its `finally` with a `done=` count, so a pass
  superseded by a knob edit still reports its spans; guarded `blocks.shape[0]`.

  **Measured (todo 6 is now a numbers question, not a guess).** 4 replicates,
  block=16, `.venv\Scripts\python.exe`, `Videos/Stabilized/`:

  - `GX010047c2` 5312x2988 @23.98, auto scale 0.2447, 479 frames (20 s): **33.6 s
    total, ~1.7x slower than realtime.** decode 26% · preprocess 18% ·
    block_reduce 14% · tensor_blur 14% · flow_solve 10% · tensor_products 7% ·
    appearance 4% · texture 2%.
  - `rep3_intermittent_crop` 462x456 @59.94, scale 1.0, 599 frames (10 s): 3.3 s.
    block_reduce 32% · tensor_blur 21% · flow_solve 15%; decode only 5%.
  - Detection is **free**: 0.04 s for T=479, B=627 — morlet 87%, chain 13%. All
    optimization effort belongs in extraction. Whole-clip clump was the suspected
    cost and is not one at these sizes.

  **What the numbers say, against the plan's guesses.** The plan expected the
  per-pixel eigen-solve to dominate; it is 10%. On the full-size clip **decode +
  preprocess is 44%** — I/O and resize, not math, and both are pure producer work
  that overlaps with the tensor stage. So the first lever is a one-frame-deep
  producer thread (OpenCV decode releases the GIL), worth up to ~40% for no math
  change and no GPU. Second lever: `block_reduce` is 14% here and 32% on the
  small clip, called 5x per frame per tile via the frame-by-frame `_reduce` —
  and `core/channel_source.py:_reduce_stack` is ALREADY a vectorized (T,H,W)
  version of the identical math. Reducing per tile-window instead of per frame
  reuses code that exists. Only after both is GPU worth pricing.
  The `DEFAULT_TARGET_WIDTH` sweep the plan calls for is still unrun and still
  the right way to price the replicate-aware-downsample correctness gap.

- **Batch C's hot-spot half (todo 6) — done, both levers landed.**

  1. **Producer-thread decode.** New `core.video.prefetch(iterable, depth=2)`
     runs any iterator on a worker thread. `_stream_channels` wraps
     `src.iter_frames` in it. The `decode` span now means *wait for a frame*,
     not decode cost — near zero when the overlap works, still honest when
     decode is the bottleneck. `close()` joins the producer with **no timeout**,
     on purpose: the caller releases the VideoCapture the instant close returns,
     so a timeout could only hand back a capture still being read from. Cancel
     stress-tested at 10 cut points on the 5.3K clip — no lingering threads, no
     stale-handle crash, clean pass afterwards.
  2. **Block reduce.** `reduce_scalar_to_blocks(..., "mean")` no longer pads the
     whole plane with NaN, transposes it, and runs `nanmean` (three full-size
     temporaries). New `_block_mean` reshapes to (ny, block, nx, block) and
     reduces axes 1 and 3 — a view, no copy, even for ragged tiles. Microbench
     730x1300 b=16: **3.46 ms → 1.01 ms (3.4x)**. NaN in the field falls back to
     the old nanmean route, so masked-input semantics are preserved.

  **Measured, 7 replicates, 5312x2988, 479 frames (20 s), grid 41x5, scale 0.245**
  (controlled A/B in one session; not comparable to the 33.6 s figure above,
  which used a different replicate set):

  - prefetch off: **15.14 s** — decode 64% · block_reduce 11% · preprocess 9%
  - prefetch on: **10.21 s** — decode 39% · block_reduce 16% · preprocess 13%

  ~33% off the pass. Prefetch hides ~5.6 s of the 9.6 s decode — exactly the
  amount of consumer work available to overlap it with. Note the consumer spans
  read slightly *higher* with prefetch on (GIL contention); net is still a clear
  win. Small clip (462x456, decode only 3%): 9.15 s → 9.23 s, i.e. neutral, with
  decode fully hidden (0.28 s → 0.02 s). No regression where there is nothing to
  overlap.

  Review fixes applied: unbounded join (was a 5 s timeout that silently broke
  the close-joins-the-producer guarantee); `frames.close()` nested so a raising
  close cannot skip `src.release()`; `except Exception` not `BaseException`, so
  a KeyboardInterrupt is not re-raised inside the consumer's cancel machinery,
  with a terminator now put on every exit path; deduped the ragged-cell loop
  shared by the mean and p90 branches. New `tests/test_prefetch.py` (7 tests,
  ordering / error forwarding / thread shutdown) and a ragged-grid test covering
  both statistics — the `include_partial` p90 branch had no test at all.
  `tests/test_channel_source.py` re-reduction check relaxed from `rtol=0` to
  `rtol=1e-6`: it was asserting bit-identical float32 summation order between
  the batched and per-frame routes, which was never the contract.

- **Next lever is decode itself, now 39% and the largest single span.** The pass
  decodes full 5312x2988 BGR frames and discards ~92% of the pixels.
  `core.video.ReplicateVideoSource` already exists and does exactly the right
  thing — FFmpeg crops, greyscales and downsamples each replicate box before the
  data crosses the process boundary — but it is wired into `core/pipeline.py`,
  not the live tensor path. Reusing it here subsumes both `decode` and
  `preprocess` (52% combined) and needs no new math. It does change where
  preprocessing happens, so it is its own batch, not a fold-in. Price that
  before considering GPU.
  The `DEFAULT_TARGET_WIDTH` sweep is still unrun and still the right way to
  price the replicate-aware-downsample correctness gap.

- **ROI decode on the live path — done.** `ReplicateVideoSource` now serves the
  tensor path, not just `core/pipeline.py`. Two additions made it usable there:
  a windowed `start=` (input `-ss`, absolute frame indices out) and a first-frame
  probe in `_open_roi` so a decoder that builds but fails on contact falls back
  to full-frame instead of killing the pass. `_MetaTile`/`_roi_layout` adapt the
  meta tile dicts, keyed by list position because the whole-frame fallback tile
  has `id=None`.

  **Measured, same clip/replicates as the prefetch A/B above** (5312x2988, 479
  frames, 7 replicates, scale 0.245):

  - full-frame: **10.41 s** — decode 35% · preprocess 27% · block_reduce 19%
  - ROI decode: **4.10 s** — block_reduce 31% · decode 22% · tensor_blur 11%

  **2.54x.** decode+preprocess went 63% → 27%, exactly the span the note above
  predicted this would subsume. The pass is now ~5x faster than realtime; at the
  start of Batch C it was 1.7x *slower*. Small clip (462x456, scale 1.0, nothing
  to crop): 8.43 s → 7.78 s, no regression. **`block_reduce` is now the largest
  span** and is the next lever if throughput still matters.

  **It is also more accurate than what it replaced, which was not expected.**
  Three-way comparison against a float32 reference (gray in float, resize in
  float, no uint8 round trip), gate disagreement averaged over 7 regions x 3
  thresholds: roi-vs-full **8.07%**, full-vs-reference **7.36%**, roi-vs-reference
  **2.37%**. So the ~8% the two decoders disagree by is mostly the *existing*
  full-frame path's own quantization error, and the ROI path is ~3x closer to
  ground truth. Two reasons: output is `gray16le`, so area-averaging a 4x4 patch
  of 8-bit samples is not re-rounded to 8 bits (0.528 → 0.364 grey-levels RMS);
  and at scale 1.0 FFmpeg reads the luma plane directly where OpenCV round-trips
  yuv420p → BGR → gray (0.69 RMS + 1.28 bias).
  **The `format=gray16le` must come AFTER `scale` in the filter graph** — before
  it, swscale converts first and scales a gray16 input, measuring 3.80 RMS, far
  worse than plain 8-bit. That ordering is load-bearing and easy to "tidy" wrong.

  Seek verified frame-accurate at 0/1/7/100/373/1000/5000/11000 on both a 23.976
  and a 59.94 fps clip. Cancel re-stress-tested at 7 cut points: no lingering
  threads, no orphaned ffmpeg processes, clean pass afterwards.

  Review fixes applied: **the seek divided the frame index by a caller-supplied
  fps, so a rounded value shifted the window** — 24.0 for 24000/1001 lands 3
  frames early by frame 11000, silently; the container's own rate is now read and
  overrides anything passed in (regression test confirmed to fail against the old
  behaviour). Also: `Preprocessor._downsample` skips the resize when the input is
  already at target size (the ROI path made it a 610 us/tile/frame no-op, verified
  bit-identical); merged duplicated except clauses. New
  `tests/test_replicate_video_source.py` (11 tests) — this decoder had **no**
  coverage at all despite `core/pipeline.py` already depending on it.

  **Two things left open, deliberately.** (1) Only the *first* frame is probed,
  so an FFmpeg failure at frame 9000 of a whole-video commit still kills the pass
  rather than falling back — a mid-stream retry needs `prev_g` and the partial
  output arrays reset, which is its own change. (2) The flow cache key does not
  include the decoder or its bit depth, so **caches built before this change must
  be rebuilt before comparing detection results across it** — the numbers moved
  for a good reason but the key does not know that.

- **Throughput is now decode-bound, and that is a hardware wall.**
  **SCOPE (added later): this holds at the OLD default scale ~0.245 only. The
  Batch K decision (scale 1.0) moves the bottleneck back onto math** — H.264
  decode is whole-frame and crop is a post-decode filter, so the decode floor
  stays ~3.8 s per 479 frames either way:

  | | extract | ~decode | ~math |
  |---|---|---|---|
  | scale 0.245 | 4.69 s | ~3.8 s | ~0.9 s (**19%**) |
  | scale 1.0 | 19.41 s | ~3.8 s | ~15.6 s (**80%**) |

  So "do not optimize the math" below is correct for the old default and WRONG
  under the new one; compute levers (Batch L) are worth having again. The
  hardware-wall conclusion about multi-process scaling still stands regardless,
  since it is about decode.

  Original entry, measured after the ROI batch, invalidating the "optimize the
  top span" instinct at the scale it was measured at:

  | 479 frames (20 s of 5312x2988) | |
  |---|---|
  | full extraction pass | 4.39 s |
  | **pure ffmpeg decode, no filters, output discarded** | **3.82 s** |

  Everything the tool does — tensor solve, block reduce, wavelets, detection —
  adds **13%** on top of decode. So `block_reduce` reading "31% of spans" is an
  artifact: those spans accumulate while the loop waits on a frame. Optimizing it
  gains ~nothing. Same for GPU on the tensor math — it is already free in the
  shadow of decode. **Do not re-derive this by staring at the span table.**

  **Multi-process does not scale**, because one ffmpeg already spreads H.264
  decode across the whole box (`-threads 1` is 40.3 s vs 3.8 s auto):

  | workers | CPU decode | GPU decode (`-hwaccel cuda`) |
  |---|---|---|
  | 1 | 4.6x realtime | 6.8x |
  | 4 | 7.8x | 10.4x |
  | 8 | 7.6x | **10.8x (ceiling)** |

  Aggregate throughput is flat past ~2-4 workers either way. The RTX 4060 helps
  ~38% but has one NVDEC engine and hits its own ceiling. At 10.8x realtime,
  3000 h of footage is **~11.6 days on this machine** — so single-box tuning is
  not the path. GPU decode is a real but small win; it is a flag
  (`-hwaccel cuda` before `-i`), not a project, if it is ever wanted.

- **`DEFAULT_TARGET_WIDTH` sweep — RUN. It refutes the plan's cost model, and
  concludes the constant is in the wrong units.** `scripts/sweep_target_width.py`,
  `GX010047c2` 5312x2988, 7 replicates, 479 frames (20 s), channel
  `tensor_speed`. Method: block_size tracks the target width (64/32/16/8) so the
  **block grid is identical (41x5) at every rung** — localization is held fixed
  and the only variable is how many working pixels average into each block.
  Thresholds are matched by quantile per rung (value band at q90 of that rung's
  own band power, gate at q85 of its own windowed count).

  **Read the timing column, not the `sig_corr` columns.** `sig_corr` is retained
  below only because the withdrawal note explains what it cannot show; it is a
  stability metric, NOT a detection-quality metric — see "What this sweep does
  NOT establish". Every durable conclusion here rests on wall clock.

  | width | block | scale | rel.px | **extract** | sig_corr 6-10 Hz | sig_corr 1-3 Hz |
  |---|---|---|---|---|---|---|
  | 5200 | 64 | 0.979 | 1.00 | **19.1 s** | 1.000 (ref) | 1.000 (ref) |
  | 2600 | 32 | 0.490 | 0.25 | **6.8 s** | 0.925 | 0.962 |
  | 1300 (old default) | 16 | 0.245 | 0.06 | **4.6 s** | 0.767 | 0.884 |
  | 650 | 8 | 0.122 | 0.02 | **4.1 s** | 0.519 | 0.794 |

  **1. Downsampling below the current default buys almost nothing — the plan's
  cost model is wrong.** The plan assumed "halving the per-replicate target is 4x
  rather than 16x", i.e. that width is a strong compute lever downward. Measured:
  1300 -> 650 is a 4x pixel cut for **~13% wall time**. The pass is pinned near
  the decode floor at these scales, so there is nothing left to save. This is pure
  wall clock and does not depend on any sensitivity claim.

  **2. Upward, the pixel cost IS paid in full — but off a baseline decode
  dominated.** 5200 is 16.7x the pixels of 1300 for **~4.1x** the wall time, and
  2600 is 4x the pixels for ~1.45x. The sub-linear *total* is Amdahl, not a shadow
  that absorbs the work: decode is a fixed ~3.8 s floor that dominated the 4.69 s
  baseline, while the math itself scaled roughly in proportion (~0.9 s -> ~15.6 s,
  i.e. ~16x). So at scale 1.0 the math is paid at full freight — which is exactly
  why compute levers (Batch L) become interesting again, and why the "~16x makes
  the live surface unusable" claim that shelved Batch K is still wrong: the wall
  cost is 4.1x, not 16x.

  Findings 1 and 2 are the durable half — both are wall clock — and together they
  say: **there is no throughput argument for downsampling below the current
  default, and the cost of abandoning it upward is 4.1x rather than 16x.** Neither
  depends on the sensitivity question below.

  **What this sweep does NOT establish — the sensitivity half is withdrawn.**
  An earlier version of this entry read the `sig_corr` column as sensitivity loss
  and concluded 1300 was under-resolved. That inference does not hold:

  - `sig_corr` is the region-MEAN band power over every block in a replicate,
    including empty background. On sparse events it largely measures whether the
    noise floors agree, not whether events are detected. It is a *stability*
    metric, not a detection-quality metric.
  - Scoring detection quality needs labelled events. `marks.json` currently holds
    **one** span, so that evaluation cannot be run yet at all.
  - `GX010047c2` is adversarially hard by design and is used as a stress case, so
    degradation measured on it does not generalize to ordinary footage in either
    direction.

  The frequency-dependent spread (1-3 Hz degrades less than 6-10 Hz at every
  rung) is consistent with small fast structures being averaged away, which is
  mechanistically plausible — but it is a hypothesis this metric cannot confirm,
  and the 6-10 Hz band's top edge sits above the 0.8*Nyquist reliability warning
  `validate_bands` already emits at this clip's 23.98 fps.

  **3. The real conclusion: `DEFAULT_TARGET_WIDTH` is in the wrong units, so no
  sweep of it can have a transferable answer.** `1300 / source_width` makes the
  physical working scale a function of how the camera was framed — same animal
  and same behaviour at a different sensor resolution or crop resolves to a
  different scale. The tool has to work across fps, resolution, behaviour, and
  species, so any constant tuned against one clip is overfitted by construction.
  This is why every previous attempt to "price" the constant stalled.

  What transfers is **organism-relative**: how many working pixels the smallest
  moving structure of interest spans. That reframing is an argument *for* Batch K
  rather than a prerequisite of it — a replicate box is one arena holding one
  animal, so resolving scale from replicate width is a proxy for
  pixels-per-animal, i.e. it moves scale from sensor-relative to
  organism-relative units. The quantity to choose is then a per-replicate working
  width with a documented physical meaning (per-species preset, or derived from
  arena calibration where it exists), NOT a global frame width. Batch K should be
  specified that way, and Batch I's manifest should record the resolved
  per-replicate scale and the rule that produced it, so footage transcoded at
  different scales stays comparable.

  **Still open, and blocked on labels rather than on measurement:** how many
  pixels-per-animal the detector actually needs. That is a question about the
  organism and the behaviour, answerable only against labelled events across more
  than one species/fps. Timings above vary run-to-run by ~25% (2600 measured
  5.3 s and 6.8 s); the ratios are sound, the absolute seconds are not.

- **Batch K — done. Downsampling is opt-in, scale 1.0 is the default, and
  `block_size` tracks the scale.** `DEFAULT_TARGET_WIDTH` and the auto branch of
  `resolve_downsample` are gone; `PreprocessConfig.downsample=None` now means
  1.0. New `BASE_BLOCK_SOURCE_PX = 64` + `FlowConfig.resolve_block_size(scale)`;
  `FlowConfig.block_size=None` means "track the scale", an explicit int still
  wins. Call sites resolve block *from* scale (`cache.py`, `pipeline.py`,
  `channel_source.py`, `tab1_flow.py`) — each already had `scale` in hand.

  **The tracking question was the one real fork, and it was decided on detection
  grounds, not aesthetics.** The plan's two-lever table makes `downsample` and
  `block_size` sound independent, which reads like an argument for a static
  block default. It is the opposite: blocks are in *working* pixels, so a static
  block means the scale knob silently moves the grid too. Detection reads
  per-block band power and gates on per-region block *counts*, so a moving grid
  changes detection output for two reasons at once and neither is attributable.
  Tracking is what makes the scale knob a single-variable experiment — which is
  exactly what Batch M's empirical panel needs in order to mean anything.

  **Verified end to end**, `GX010047c2` 5312x2988, 7 replicates, 120 frames:

  | setting | scale | block | grid | extract |
  |---|---|---|---|---|
  | default | 1.000 | 64 | 41x5 | 4.20 s |
  | downsample 0.5 | 0.500 | 32 | 41x5 | 2.15 s |
  | downsample 0.25 | 0.250 | 16 | 41x5 | 1.51 s |
  | explicit block 16 | 1.000 | 16 | 146x19 | 6.29 s |

  Grid holds at 41x5 across the scale sweep (matching the Batch K table's row 3
  prediction) while compute moves ~3.5x; the explicit override reaches the finer
  grid on demand. Full suite green (98 passed).

  `tests/test_auto_downsample.py` rewritten as the plan asked: it now pins the
  *invariant* (a pre-cropped video and an equivalent uncropped box resolve
  identically) rather than `resolve_replicate_downsample`, plus grid-invariance
  under the scale sweep, so it survives the eventual organism-relative mode.

  **Three review findings fixed, all in `tab1_flow.py`, all caused by this
  change reinterpreting `None`** — worth recording because the shelved-tab rule
  ("don't fix them if a change breaks them") was overridden deliberately here:
  `_apply_config` crashed with `TypeError` on `setValue(None)` for any config
  saved at defaults; the downsample spin's `p.downsample if p.downsample else
  0.05` mapped a default config to **0.05, a silent 20x downsample** — precisely
  the failure this batch exists to prevent, arriving through the back door; and
  tab1's `"auto"` sentinel now meant 1.0 while sitting next to a 0.05 minimum,
  so it was removed. Also: `reduce_channel_data` now takes the scale from the
  cached atlas's own meta rather than from `cfg`, since it is reducing data that
  already records what it was extracted at.

  **Left open:** `resolve_downsample(src_width)` keeps a parameter it ignores,
  because every call site still passes one. Harmless but misleading — it invites
  the inference that scale still depends on framing. Worth removing when the
  organism-relative mode lands, since that mode needs *replicate* geometry, not
  source width, i.e. a different signature anyway.

  **Batch M is now the blocking item, exactly as the plan warns.** K alone ships
  a knob users will refuse on principle; the `auto (0.245)` text Batch A added is
  gone, and nothing yet shows what downsampling buys. Ship M next.

### Batch I — ROI pre-transcode (todo 23) · new file + `channel_source`
The one lever that removes the decode wall. Cut each source once into
per-replicate clips plus a manifest (geometry, scale, source hash, fps at full
precision), and teach the extraction path to consume the manifest instead of the
original. Every later pass then decodes ~1/16 the pixels.

**Why this beats the live ROI crop that already exists.** H.264 decode is
whole-frame: the `crop` in `ReplicateVideoSource`'s filter graph runs *after* the
frame is decoded, so the live path still pays full decode cost and only saves the
downstream per-pixel work. That is why the decode floor measures ~3.8 s per 479
frames regardless of crop or scale. Pre-transcoding makes the *stored file* small,
so subsequent decodes genuinely decode fewer pixels — it is the only thing here
that moves that floor.

**Crop is lossless; scale is lossy. Batch I should do only the crop by default.**
Under the Batch K decision, the pre-transcode must cut to the replicate box at
**scale 1.0** unless the user has explicitly asked for downsampling. Cropping
discards only pixels no replicate owns, so it cannot change any detection result
— it is a pure decode win and is safe to apply unconditionally. Rescaling would
bake a sensitivity decision into artifacts that cost a full re-decode to
regenerate, which is exactly what the principle forbids. The manifest therefore
records the resolved per-replicate scale *and the rule that produced it*, so
clips transcoded under different settings are never silently compared. This also
makes I the main throughput lever now that scale is pinned at 1.0: for these
~297 px replicates the crop alone is ~1/16 the pixels off the decoder.

**It moves the bottleneck to the math, which is the point.** The already-cropped
`rep3_intermittent_crop` (462x456) spends 3-5% on decode and ~7.8 s on math — a
different regime entirely. So this must land *before* any math/threading work:
afterwards CPU math parallelizes cleanly across processes, where decode does not.
Reuses the `ReplicateVideoSource` filter-graph geometry that already exists.
Watch: the `format=gray16le` after `scale` ordering, and keep fps at full
precision in the manifest (see the seek bug fixed in the ROI batch).

### Batch J — headless batch driver (todo 24) · new CLI + `core` only
No-GUI entry point: video + replicate JSON -> detection results on disk, plus
file-level job partitioning. This is what gets linear scaling, because each node
brings its own decode capacity. Pairs with I (a batch run should consume
pre-transcoded ROI clips).

**Throughput assumption updated by the Batch K decision — the old "3000 h / 10x /
50 nodes ~ 6 h" no longer applies.** That used the 10.8x-realtime decode ceiling
measured at the old default scale. At scale 1.0 a single-process pass is ~1.03x
realtime (19.41 s per 19.97 s of footage) because the work is now math-bound, not
decode-bound. The compensation is that **math-bound work parallelizes across
processes where decode did not** — the flat-past-4-workers ceiling above was a
decode property. So per-node throughput needs re-measuring at scale 1.0 with N
workers before any node-count estimate is quoted; do not carry the 10x figure
forward.

Storage per the measured table under Batch K: ~289 MB/h of 24 fps footage at
block 64 (the (n, ny, nx) float32 x 5 channel arrays), rising to ~3.6 GB/h if
block_size does not track scale. Either way a multi-hour clip wants chunked writes
rather than one in-memory pass.

### Batch K — downsampling becomes an opt-in cost lever (todo 25) · `core/config.py` + tests

**DECISION (supersedes the auto-downsample framing entirely).** Default to **no
downsampling, scale 1.0**. Downsampling stops being an automatic behaviour keyed
to frame width and becomes an explicit user-wielded knob for reducing compute or
storage.

**Why, and this is the governing principle for the whole tool:** if the pipeline
silently downsamples by default, it has already decided which behaviours are
detectable — *the tool defines the data collected rather than the other way
around.* Coarser resolution may well be sufficient for a given behaviour, but
that has to be **demonstrated per behaviour/species, never assumed**. Detection
sensitivity is a scientific result about the organism, not a default constant.

Consequences, in the order they bite:

1. **`DEFAULT_TARGET_WIDTH` and `resolve_downsample`'s auto branch go away** as
   defaults. `PreprocessConfig.downsample=None` should mean 1.0, not
   `1300/src_width`. The `auto (0.245)` text Batch A added to the UI goes with
   it. This also closes the original correctness gap for free: at scale 1.0 a
   pre-cropped video and an equivalent uncropped box trivially agree, because
   neither is rescaled. `tests/test_auto_downsample.py` should be rewritten to
   assert *that* invariant rather than `resolve_replicate_downsample`.
2. **The eventual denomination is pixels per body length**, roughly — an
   organism-relative unit, so a target transfers across sensor and framing. Not
   needed to ship the default-1.0 change; needed once the knob grows a "target a
   metric" mode. Per-species presets or arena calibration are the likely inputs.
3. **The knob must be legible about what it buys.** A user choosing 50% must be
   able to see which steps shorten and by how much. `core/timing.py` already
   emits exactly these spans, so the cost model can be built from measured data
   rather than asserted. **This is Batch M, and it is not optional polish** — an
   opt-in knob with no visible upside just gets refused, which trades silent
   degradation for silently infeasible projects. Ship M with K.

**Scale and block size are separable levers, and they pay for different things
— do not conflate them in the UI.**

| lever | shortens | cost axis |
|---|---|---|
| `downsample` (scale) | decode(ROI) · preprocess · tensor_products · tensor_blur · flow_solve · appearance · texture — every per-pixel stage | **compute** |
| `block_size` | output grid cells, hence cache size, wavelet and detection | **storage** |

Scale leaves the block grid alone if `block_size` tracks it, and `block_size`
leaves per-pixel compute alone entirely. A user who is storage-limited wants
`block_size`; a user who is compute-limited wants `downsample`. Presenting one
"quality" slider would hide that.

**Priced (from the sweep above):** scale 1.0 on `GX010047c2` costs ~4.1x the
current default's wall time, landing at ~1.03x realtime single-process. That is the price of the
principle, and it is accepted deliberately. The answer to it is more nodes
(Batch J) and less decode (Batch I), not a quieter default.

**What `block_size` does when scale goes to 1.0 (resolved below).** Going to scale
1.0 without touching `block_size=16` also multiplies the output grid, because
blocks are measured in *working* pixels. On `GX010047c2` (7 replicates, ~297 px
wide):

**Measured** (479 frames, 7 replicates, extrapolated to 24 fps):

| | extract | grid | storage/h | 3000 h |
|---|---|---|---|---|
| today: scale 0.245, block 16 | 4.69 s | 41x5 = 205 | 289 MB | ~0.9 TB |
| scale 1.0, block 16 | 22.69 s | 139x19 = 2641 | 3.6 GB | **~11 TB** |
| **scale 1.0, block 64** | **19.41 s** | 41x5 = 205 | 289 MB | **~0.9 TB** |
| scale 0.245, block 64 | 4.06 s | 20x2 = 40 | 56 MB | ~0.2 TB |

**RESOLVED: block_size tracks scale by default (row 3).** It keeps today's
storage and today's localization while still feeding the per-pixel tensor solve
full-resolution pixels — blocks cover the same source area as today, but fine
motion reaches the solve before being averaged away, which is where the
sensitivity argument lives. Row 2's finer localization is a separate opt-in
decision at ~12x storage, not something to inherit silently from the scale
change.

**The two levers are NOT interchangeable — measured, not asserted.** Rows 1 vs 3
vs 2 separate them cleanly:

| lever | compute | storage |
|---|---|---|
| `block_size` 16 -> 64 | **-13%** | -5x to -13x |
| `downsample` 1.0 -> 0.245 | **-4.8x** | -13x |

`block_size` is close to a pure *storage* lever: 13x less data for 13% less
compute, because the per-pixel stages run at working resolution regardless of
block size. A compute-limited user gets almost nothing from it; only
`downsample` touches that. So the UI must expose both separately — a
storage-limited user reaching for a single fused "quality" slider would pay a
sensitivity cost they had no need to pay. Only the `downsample` lever carries
the "this may decide what is detectable" warning.

### Batch L — quiet-tile gating (DEFERRED, design recorded) · `core/tensor_channels.py`
Wanted, but explicitly not now. Recorded so the reasoning is not re-derived.

**The idea.** TRex/TGrabs (Walter & Couzin 2021) and idtracker.ai reduce data by
subtracting a background model, thresholding to blobs, and running everything
downstream on blobs only. The same *data reduction* is available here without any
background model, because the structure tensor already computes the discriminant:
`change` is `<I_t^2>` per block, documented as `J_tt` (`core/tensor_channels.py:8`)
— squared frame differencing, i.e. the crudest form of background subtraction, as
a first-class channel.

**Gate within the frame, not with a lagged rolling window.** The pass runs
`tensor_products -> tensor_blur -> flow_solve -> appearance -> texture ->
block_reduce`. `I_t^2` falls out of `tensor_products`, which is early and cheap
(~7% of the measured profile), while `flow_solve` / `appearance` / `texture` are
the expensive per-pixel work downstream. So: compute products everywhere, then
skip the remaining stages for block-tiles below threshold and write a known-quiet
value. No fitted asset, no lag.

**Must be tile gating, not pixel masking.** Masking background pixels saves
nothing — the per-pixel stages are vectorized over the whole tile and a mask does
not skip work in a dense array op. A scattered gather breaks the math outright,
since `tensor_blur` is a spatial convolution and the structure tensor is a
neighbourhood operator. Only *contiguous block-tile* gating both saves work and
preserves the fixed grid the atlas, `block_weight_plane` and
`region_blocks_and_grid` all assume.

**Why this fits the Eulerian design and TRex's blob extraction does not.** TRex is
Lagrangian: the blob IS the unit of analysis, so non-blob pixels are genuinely
nothing. Here detection is per-block band power on a fixed grid, and background
blocks are the baseline the statistics are computed against (in-band block *count*
per region, clump area). A gated block therefore needs a defined contribution to
the count and the clump — it cannot simply vanish.

**Do this measurement first: tile occupancy.** Gating pays only if most tiles are
quiet most of the time. Computable today from the `change` channel of an ordinary
pass, no new machinery: what fraction of block-cells fall below a candidate
threshold, across footage types. ~90% quiet means gating removes most of the math;
~40% means bookkeeping overhead eats it. Same shape of cheap up-front measurement
the width sweep turned out to need.

**Threshold validation is unusually tractable here, and is not optional.** The
gate and the detector read the SAME quantity, so the gate can be expressed in
units of the detector's own sensitivity floor ("gate at X% of floor") rather than
as an independent calibration. Use that. The hazard is the one the Batch K
decision names: gating decides what is detectable. Its failure mode is *worse*
than downsampling's — downsampling loses small/fast structures and fails
uniformly, whereas a variance/change gate loses low-contrast, slow, subtle motion
and fails **selectively on quiet behaviour**, which is often the behaviour of
interest (stillness, antennal movement, slow postural shift). Deleted signal is
indistinguishable from "nothing happened". Off by default, same as downsampling.

**Note on existing code:** `bg_subtract` (median/MOG2) has stubs in
`core/preprocess.py` and config fields in `core/config.py:61`, but is not built
out and is deliberately forced off for the tensor path
(`core/tensor_channels.py:176`), which flags results `approximated` when it was
on — background models are fitted assets the cache cannot reconstruct. Do not
build on it. The `J_tt` route sidesteps that objection entirely, since it is
recomputed from the frames like everything else.

### Batch M — the downsampling decision tool (new window) · new `gui/downsample_dialog.py`
Pops up from the downsample control. Design approved; not yet built.

**The problem it exists to solve.** Batch K makes downsampling opt-in and off by
default, which is right, but a bare "downsample?" knob produces exactly one
behaviour: *"I don't want my data to be worse, so I won't touch it."* That is not
a decision, it is an avoidance, and it makes large projects infeasible for no
examined reason. Many projects genuinely need full resolution; **many do not**,
and for those the difference is whether the project happens at all. The tool's job
is to make the middle ground reachable and legible so the choice is strategic.

**REQUIRED: this reasoning must be written out in the window itself, in prose, on
open.** Not a tooltip, not docs. It states plainly that (a) downsampling can be
what makes a project computationally feasible, and (b) it is deliberately NOT
assumed on the user's behalf, precisely so that computationally expensive projects
are not silently thrown away by a default. Both halves, together — one without the
other produces either the avoidance above or the silent degradation Batch K
forbids. Write it as real explanatory text.

**Layout — split (approved).**

```
┌─ Downsampling — what you gain and lose ──────────────────┐
│ Replicate: [rep17 ▾]      Corpus: [3000] hours           │
├───────────────────────────┬──────────────────────────────┤
│  PROJECT TIME / STORAGE   │   WHAT THE ANIMAL LOOKS LIKE │
│ 125d┤●                    │  ┌────────┐    ┌────┐        │
│     │ ╲                   │  │        │    │    │        │
│  60d┤  ●╲                 │  │  1.00  │    │0.50│        │
│     │    ╲●___            │  │ 297 px │    │148 │        │
│  30d┤        ●━━━●━━━●    │  └────────┘    └────┘        │
│     └─┬───┬───┬───┬───┬   │   full          pick         │
│      1.0 .75 .50 .25 .12  │                              │
│              ▲ KNEE       │  ▓▓ what the detector sees ▓▓│
│   past here: lose pixels, │  ┌────────┐    ┌────┐        │
│   save almost no time     │  │ change │    │chg │        │
├───────────────────────────┴──────────────────────────────┤
│ AT 0.50 →  4.2 px per body length  ·  62 d  ·  0.9 TB    │
│ AT 1.00 →  8.4 px per body length  ·  125 d ·  0.9 TB    │
│                         [ Use 0.50 ]  [ Keep 1.00 ]      │
└──────────────────────────────────────────────────────────┘
```

**The knee is the single most valuable thing here and is invisible today.** Wall
time has a hard decode floor, so below some scale the user gives up resolution and
buys *nothing*: measured, 1300 -> 650 is a 4x pixel cut for ~13% wall time. Nobody
can guess that. Marking it turns the question from "how brave am I" into "here is
the efficient frontier, pick a point on it." Compute the knee from the measured
cost model rather than hardcoding it — `core/timing.py` already emits the spans
per stage, and the two-lever table under Batch K says which stages each knob
shortens.

**Both levers, kept visually distinct** (see the Batch K table): `downsample` is
the compute lever and carries the "may decide what is detectable" warning;
`block_size` is the storage lever and does not. Do not fuse them into one
"quality" slider — a storage-limited user would pay a sensitivity cost for
nothing.

**Calibration sub-tool — broader than just body length.** The user should be able
to draw a line against a **known fiducial** (ruler, scale bar, arena feature of
known size) to establish px/mm, AND draw a line along the animal to establish body
length, and similar measurements in the same spirit. Feeds the existing fields
rather than inventing new ones: `core/roi.py:83-84` already has `pixels_per_mm`
and `body_length_mm`, and `core/roi.py:418` already establishes the convention
`working_px_per_mm = pixels_per_mm * downsample`. So "working px per body length"
at a given scale is exactly `pixels_per_mm * body_length_mm * scale`. Writing back
also benefits exports and `speed_body_lengths_s`, which read the same fields.
Fall back to working px across the replicate box when uncalibrated — geometric,
always available, just not organism-relative.

**Empirical panel — scoped, and it is the point of the whole thing.** Run the
detector at each candidate scale over the loaded window and report events found.
Label it exactly: *evidence on THIS clip and THIS behaviour, not a general
sensitivity guarantee.* This is the mechanism by which Batch K's "must be
demonstrated per behaviour/species, never assumed" actually gets satisfied — it
turns the principle into something a user can execute in a minute instead of an
instruction they cannot act on. Costs a pass per scale; open the window instantly
and populate this panel asynchronously (Batch B's `_StreamWorker` + `cancel()`
already give the pattern).

**Do NOT invent a quality score.** No single number summarizing "how much worse".
Everything shown is either measured wall clock, measured storage, a rendered
image, or an event count on a named clip. The withdrawn `sig_corr` reading under
the sweep entry is the cautionary case: an aggregate that looks authoritative and
does not mean what it appears to.

**Anchors.** Dialog pattern: `gui/mask_dialog.py` (94 lines, `QDialog` +
`FrameView` with `draw_enabled` / `box_drawn`) is the closest existing model and
also the precedent for a draw-on-frame tool. `FrameView` is in
`gui/video_panel.py`. Launch point: next to the downsample spin box in
`gui/explorers/live_scalogram_surface.py:206-215`. Detector entry:
`core/detection.py:167 detect_channel_region`. Cost model input: `core/timing.py`
spans plus the measured tables under Batch K.

**Depends on K** (the scale-1.0 default and the two-lever split are what this
tool presents). Build after K, and it pairs naturally with J, since the corpus-
hours field is the same feasibility question a batch run asks.

**Build the panels as reusable components, not baked into this dialog.** Batch N
is the same tool for `block_size` and should share all of it — cost model,
corpus projection, frontier curve, evidence panel, empirical panel, calibration.
Knowing that up front is the difference between one shared widget set and two
divergent dialogs.

### Batch N — the same decision tool for block size (stub, not yet elaborated)
A sibling pop-out for `block_size`, **built on shared machinery with Batch M** —
build M's pieces as reusable components rather than baking them into one dialog:
the cost model over `core/timing.py` spans, the corpus-hours input and its
time/storage projection, the frontier curve with its knee marker, the
render-at-each-setting evidence panel, the scoped empirical detection panel, and
the calibration sub-tool. Same discipline: explanatory prose on open, and no fused
quality score.

The one substantive difference, which sets what its "what you lose" panel shows:
`downsample` and `block_size` lose *different things*. Downsampling loses detail
within a block (the per-pixel field feeding the tensor solve is coarser);
block size loses **spatial localization** (fewer, larger grid cells, so clump area
and where-in-the-arena resolution degrade) while the per-pixel math is untouched.
So N's evidence panel is about grid granularity and clump resolution, not image
sharpness. Per the Batch K table it is the storage lever (-13x storage for -13%
compute), and it does not carry downsampling's "may decide what is detectable"
warning in the same form.

Elaborate when M is built and it is clear which parts genuinely generalized.

### Suggested order (revised after the sweep + the Batch K decision)
**K -> M -> I -> J**, and J is now materially more urgent than it was.

K first: it is small (delete an auto branch, flip a default, rewrite one test
file) and it is the principle everything else has to respect — landing I or J
first would bake the old default into artifacts or into a batch run.

**M immediately after K, not later.** K alone is a half-shipped change: it makes
downsampling opt-in and off by default, which without M leaves users with a knob
they will refuse to touch on principle, and therefore with projects that do not
fit their compute. M is what converts the default from an avoidance into a
decision. Shipping K without M is arguably worse than shipping neither.

Then I, which under the new default is the *only* remaining decode lever and is
lossless, so it costs nothing scientifically. Then J.

**Why J moved up.** The default just got ~4x more expensive by deliberate choice,
so single-process throughput drops to ~1.03x realtime and is no longer close to
sufficient for 3000 h. Fan-out is what pays for the principle. Note the
compensation recorded under J: at scale 1.0 the work is math-bound, and math
parallelizes across processes where decode did not.

D/E (interactive polish) slip behind J unless day-to-day use becomes the
near-term milestone; they live in `gui/explorers/scalogram_explorer.py`, the
widget inside **tab 2 · Preprocessing (live)** (`tab_live_preprocess.py` ->
`live_scalogram_surface.py`), not the shelved flow-cache tab. F is still worth
folding into whichever lands first, since 18 is a real speed win.

Not blocking any of the above, and explicitly deferred: **Batch L** (quiet-tile
gating — wanted, design recorded, not now), the pixels-per-body-length
denomination, and the per-behaviour sensitivity study that would justify any
non-1.0 default. The latter two need labelled events; `marks.json` has one span.
L becomes more attractive after K, since K is what makes math the bottleneck
again.

### Shelved (todo 8, 9)
`gui/tab1_flow.py` and `gui/tab3_behavior.py` are unmaintained. Do not fix them if a
change breaks them; just note it. Consider renaming to `_shelved_*` (the repo already
has that convention) once nothing imports them.
