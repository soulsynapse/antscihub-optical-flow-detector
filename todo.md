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

- **Throughput is now decode-bound, and that is a hardware wall.** Measured after
  the ROI batch, and it invalidates the "optimize the top span" instinct:

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

### Batch I — ROI pre-transcode (todo 23) · new file + `channel_source`
The one lever that removes the decode wall. Cut each source once into
per-replicate clips at working resolution plus a manifest (geometry, scale,
source hash, fps at full precision), and teach the extraction path to consume the
manifest instead of the original. Every later pass then decodes ~1/16 the pixels.

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
brings its own decode capacity: 3000 h / 10x / 50 nodes ~ 6 h. Pairs with I (a
batch run should consume pre-transcoded ROI clips). Also needs a memory note: the
channel arrays are (n, ny, nx) float32 x 5 — ~354 MB per hour of 24 fps footage,
so a multi-hour clip wants chunked writes rather than one in-memory pass.

### Batch K — replicate-aware auto downsample (todo 25) · `core/config.py` + tests
Was already flagged as the keystone under Batch C and has its own shelved writeup
in `tests/test_auto_downsample.py`. Restated as a numbered item because it is a
**correctness** gap, not a tuning preference, and it is the only remaining one:
today a pre-cropped video and an equivalent uncropped box resolve to different
physical scales, so results are not comparable across those two workflows. The
`auto (0.245)` in the UI is the source-width number, i.e. provisional.
Sweep `DEFAULT_TARGET_WIDTH` (1300) against detection output on a known clip
before paying the ~16x pixels; halving the per-replicate target is 4x rather than
16x and may cost no sensitivity at all. Passes are now ~5x realtime, so the sweep
that was previously painful is quick.

- **Then: Batch D/E** if perceived UI lag matters more than throughput. Note D/E
  are `gui/explorers/scalogram_explorer.py`, which is the widget inside **tab 2 ·
  Preprocessing (live)** (`tab_live_preprocess.py` -> `live_scalogram_surface.py`),
  not the shelved flow-cache tab.

### Suggested order
K (correctness, cheap, blocks nothing) -> I (removes the decode wall) -> then
either J (HPC fan-out) or D/E (interactive polish), depending on whether the next
milestone is a big batch run or day-to-day use. F is worth folding into whichever
comes first, since 18 is a real speed win.

### Shelved (todo 8, 9)
`gui/tab1_flow.py` and `gui/tab3_behavior.py` are unmaintained. Do not fix them if a
change breaks them; just note it. Consider renaming to `_shelved_*` (the repo already
has that convention) once nothing imports them.
