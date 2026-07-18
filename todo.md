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

### Shelved (todo 8, 9)
`gui/tab1_flow.py` and `gui/tab3_behavior.py` are unmaintained. Do not fix them if a
change breaks them; just note it. Consider renaming to `_shelved_*` (the repo already
has that convention) once nothing imports them.
