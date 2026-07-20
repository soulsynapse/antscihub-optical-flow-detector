# Optical flow detector — working plan

**How to use this file.** Items have stable ids (`T1`, `T2`, …) — reference those,
never line numbers, so this file can be edited freely. Batches are grouped by
*file locality*, so each loads one or two files and fits in a fresh context. Do
one batch, review, commit, `/clear`. Run python via `.venv\Scripts\python.exe`.

**`FINDINGS.md` is the companion file** and holds every measurement and hard-won
conclusion — deliberately kept out of here so this plan stays cheap to load each
session. Load it before optimizing anything, re-running a sweep, or touching code
flagged there as load-bearing. Several of its conclusions are counter-intuitive
enough that they were reached wrongly at least once; the batch specs below point
to the relevant section rather than restating it.

---

## 1. Open items

| id | item |
|---|---|
| ~~**T10**~~ | ~~Plots are themselves the main source of lag at block=1 with weak downsampling.~~ **FIXED** — see Batch D. Measured **3.2x** cheaper scrub collapsed. The premise was half wrong: plot cost tracks *widget pixel area*, not block count, so the win is the same at every block size (`FINDINGS.md` §17). |
| ~~**T11**~~ | ~~If fixed-size stamp is on, it should be the stamp of the *currently selected* replicate.~~ **FIXED** — see Batch G. `stamp_size` became a property derived from the selection; the fix was to delete the second copy of the state, not to sync it. |
| ~~**T12**~~ | ~~Clicking a replicate should highlight + zoom to it; dragging its box should reposition it; right-click should delete it.~~ **FIXED, with the right-click half deliberately REFUSED** — see Batch G. Right-click is `back_requested` on five other screens, and T12's own zoom is what makes this tab need an un-zoom. |
| ~~**T13**~~ | ~~Per-block band power by channel draws black blocks when unpopulated — collapse those to checkbox height.~~ **FIXED** — see Batch D. |
| ~~**T14**~~ | ~~All plots should collapse/expand via `[+]`, collapsed by default, and a collapsed plot must not render.~~ **FIXED** — see Batch D. |
| ~~**T27**~~ | ~~The four **detection-sweep** plots draw the same black slab T13 removed from the density plots.~~ **FIXED** — see Batch E. Not the two-line change this entry claimed: the mechanism lived on `DensityPlot` and is keyed on `matrix`, so it had to move up to `MiniPlot` behind an overridable `_is_empty()`. |
| ~~**T28**~~ | ~~Collapsed-by-default (T14) hides `count_w_plot`, which carries the **detection threshold band**.~~ **RESOLVED: exempt** — see Batch E. T15 settled it, by putting the detection readout on that same plot. |
| ~~**T15**~~ | ~~Replace the positive-detection graph with green bands overlaid on the windowed #-blocks-in-band plot.~~ **FIXED** — see Batch E. |
| ~~**T16**~~ | ~~On a detection, show a `DETECTED` badge (green bg, bold black) bottom-right of the viewer box.~~ **FIXED** — see Batch E; the shift-held exemption is pinned by a pixel test. |
| ~~**T17**~~ | ~~Whole-video processing resets the detection threshold bands when navigating.~~ **FIXED** — see Batch F. |
| ~~**T18**~~ | ~~"Process whole video" computes every channel regardless of the selected one.~~ **FIXED** — see Batch F. |
| **T20** | ~~Drop the replicate dropdown — redundant with click navigation.~~ **RESCOPED, and it is not in the file this entry assumed** (`FINDINGS.md` §19). There is no dropdown on the replicate tab. The widget is `region_combo` in **five explorers**, where it is not redundant with click navigation but *is* the selection state — `active_region_index` reads `currentData()`, click-to-select is `setCurrentIndex`, and it carries a **pooled scope (`-1`) with no click gesture at all**. Deleting it means rehoming three things. Belongs with the explorers, not Batch G. |
| **T21** | Suspected runaway memory issue somewhere on the replicate tab. **Not reproduced by inspection** — `_refresh_list` was the obvious suspect and is clean (14x14 swatch pixmaps, `list.clear()` releases the items). Needs an actual measurement, not more code reading; do not guess at a fix. |
| ~~**T22**~~ | ~~New **viewer tab** consuming a fully-processed video.~~ **DROPPED** (2026-07-19) — scope, not value. A convenience shell over a `DetectionResult` that nothing else depends on; never started. Re-open only if day-to-day review actually stalls without it. |
| ~~**T23**~~ | ~~**ROI pre-transcode**: cut each source once into per-replicate clips + manifest.~~ **DONE** — see Batch I. Works, and **the premise it was justified on turned out false**: the 25x isolated decode win is **1.06x end to end** (`FINDINGS.md` §16). Not the lever that moves the floor at scale 1.0. |
| ~~**T24**~~ | ~~**Headless batch driver**: no-GUI entry point, file-level partitioning, N-worker throughput.~~ **DONE** — all three slices landed; throughput measured in `FINDINGS.md` §14 (**~130 fps / 5.4x realtime, ceiling at 8 workers**). |
| **T29** | **Streaming live surface** — display immediately, process forward continuously, plots fill as frames arrive, instead of extract-a-window-then-block. Architecture decided 2026-07-20; see Batches P/Q. Foundations landed in `e11a4cb` (`core/stream_buffer.py`); **Batch P landed** (`stream_channel_planes` + `ChannelPlan`, `FINDINGS.md` §22, cost measured at nil). **Batch Q slices 1 and 2 landed** (`LiveStreamWorker`; the `Live ▶` surface + `set_channel_data`, §23). **Slice 3 — the CWT with the cone of influence — is what remains.** |
| **T30** | **The mark corpus is not a corpus.** `marks.json` holds **one** 0.39 s span, and **no animal-absent spans at all** — so occupancy's whole claim (present-but-still ≠ empty) cannot be tested, and neither can any supervised channel comparison. Needs flying / still-with-animal / animal-absent / wingbeat, several each, across replicates. `rep6-no-flying` is a ready-made negative control. **This blocks T31 and T32, and no amount of channel-building substitutes for it.** |
| **T31** | **Validate occupancy and the ratio channels against marked footage**, using the **tail statistic, not region means** (`FINDINGS.md` §20 — this was got wrong once already). Gates Batch S. |
| **T32** | **Supervised signature fitting.** With a real corpus, fit LDA / L1-logistic per labelled behaviour over the (channel × spatial scale × temporal scale) grid. The weight vector *is* an interpretable, tunable signature and drops into a template detector with the existing `DetectionResult` shape. Cheap, and it tells you which channels earn their place before the unsupervised map is built around them. |
| **T33** | **Rebuild marking on the streaming surface** (Batch R). `gui/timeline.py` still has the full span model; `gui/_shelved/tab3_behavior.py` has the persistence and label picker. |
| ~~**T26**~~ | ~~**The container frame count is not the decodable frame count.**~~ **FIXED** — see Batch O. The cut and a full-length headless pass both run clean on `GX010047c2` now; `FINDINGS.md` §15 records the fix, §16 the clip-backed throughput it unblocked. |

## 2. Standing decisions

**The governing principle (from the Batch K decision).** If the pipeline silently
downsamples by default, it has already decided which behaviours are detectable —
*the tool would define the data collected rather than the other way around.*
Coarser resolution may well suffice for a given behaviour, but that must be
**demonstrated per behaviour/species, never assumed**. Detection sensitivity is a
scientific result about the organism, not a default constant. The same rule
governs any future quiet-tile gating (Batch L).

**Scale and block size are separate levers and must stay separate in the UI.**

| lever | shortens | cost axis | carries the "may decide what is detectable" warning |
|---|---|---|---|
| `downsample` (scale) | every per-pixel stage | **compute** | **yes** |
| `block_size` | grid cells → cache size, wavelet, detection | **storage** | no |

Never fuse them into one "quality" slider: a storage-limited user would pay a
sensitivity cost for nothing.

**Do NOT invent a quality score.** No single number summarizing "how much worse".
Show measured wall clock, measured storage, a rendered image, or an event count on
a named clip. The withdrawn `sig_corr` reading (`FINDINGS.md` §4) is the
cautionary case: an aggregate that looked authoritative and did not mean what it
appeared to.

**Decode is not the throughput lever; math is.** Measured, not assumed. At scale
1.0 (K's default) `tensor_blur` + `tensor_products` are ~80% of a pass, the
producer thread already overlaps decode, and Batch I's 25x cheaper decode buys
**1.06x end to end** (`FINDINGS.md` §16). Do not prioritize decode work off the
isolated decode number again — that number is real and nearly irrelevant. The one
untested case is **N=8 clip-backed** (carried to Batch L below), where 8-way
decode contention meets §14's suspected memory-bandwidth ceiling.

**The clip-provenance cache obligation (from Batch I, still unenforced).**
`cfg.cache_key(video_hash, geometry_hash, provenance_key=None)` has a third
provenance axis that **no caller passes**. Nothing caches a clip-derived result
today, so it is an available guard, not an active one — but the day something
does, omitting it collides a source-derived and a clip-derived result in one
cache entry, silently. The key travels as `meta["clip_provenance"]` out of
`live_channel_source`.

*Recommended fix, cheap and worth doing before it bites — make the omission
impossible rather than documented.* Drop the default and make the parameter
**required and keyword-only**:

```python
def cache_key(self, video_hash, replicate_geometry_hash, *, provenance_key): ...
```

Every call site must then write `provenance_key=None` deliberately, which is a
claim ("this pass read the source") rather than an oversight. Cost is three call
sites — `core/pipeline.py:174`, `scripts/validate_standardization.py:114` — plus
`tests/test_clip_extraction.py`. (A third was `gui/tab1_flow.py`, now retired to
`gui/_shelved/`; leave it broken if this lands.)
The *hashing* behaviour must not change: `None` stays omitted from the blob, so
every pre-clip cache keeps its key. This is the T17 shape handled early — state
that quietly stops meaning what it meant — and the same remedy class as bumping
`PRETRANSCODE_VERSION`: make the silent case loud at the boundary.

**Shelved (T8, T9) — done.** `tab1_flow.py` and `tab3_behavior.py` moved to
`gui/_shelved/`, joining the two files that already used the flat `_shelved_*`
prefix (the old prefix convention is retired; the package replaces it). Nothing
imports them. `MainWindow` is down to two tabs, and `tests/test_cache_naming.py`
was deleted with the `_test_cache_suffix` it covered. Don't fix anything in
`gui/_shelved/` if a change breaks it; just note it.

## 3. Done

| batch | what | commit |
|---|---|---|
| A | strip cosmetics (T3, T4, T19) | `56fe5a6` |
| B | cancellable extraction (T7) | `40c7cac` |
| C | step timing (`core/timing.py`) + hot-spot work (T5, T6) | — |
| — | producer-thread decode, faster block reduce | — |
| — | ROI decode on the live path | — |
| — | `DEFAULT_TARGET_WIDTH` sweep (refuted the plan's cost model) | — |
| K | downsampling opt-in, scale 1.0 default, block tracks scale (T25) | `34ef8ec` |
| M | downsampling decision tool — all five slices, **closed** | `ecd942d`…`3ec1e5a` + calibration |
| I | ROI pre-transcode: the cut, the manifest, the wiring (T23) — **closed, see below** | — |
| J | headless batch driver + sharding + throughput (T24) | — |
| O | decodable frame count (T26) | `cba9824` |
| D | plot collapse, empty-plot collapse (T10, T13, T14) | `0b112a6` |
| E | detection readout (T15, T16, T27, T28) | `4fdf8b0` |
| N (item 2) | count-band re-denomination + the three permanent controls | `1d1f958` |
| G (part) | replicate direct manipulation (T11, T12) — **T20, T21 still open** | — |
| — | occupancy + conjunction channels, live stream ring buffer | `e11a4cb` |

**Closed batch specs live in `docs/archive/closed-batches.md`** (D, E, F, I, J, O
archived 2026-07-20). Their durable output is `FINDINGS.md`; the specs are kept
only because several were wrong in instructive ways.

Their durable output is `FINDINGS.md`. Everything else about them has been deleted.

**Batch B left one open question:** a stop during the detector phase discards a
completed `DetectionResult`, because `detect_channel_region` has no cancel point
and the trailing `if self._cancel` throws the finished result away. Delivering it
may be better — the expensive extraction is already paid for.

**Batch K left one loose end:** `resolve_downsample(src_width)` keeps a parameter
it ignores. Harmless but misleading — it invites the inference that scale still
depends on framing. Remove when the organism-relative mode lands, which needs
*replicate* geometry and therefore a different signature anyway.

---

## 4. Findings — see `FINDINGS.md`

| section | why you would open it |
|---|---|
| 1. Throughput | decode-bound vs math-bound; why the span table misleads; multi-process ceiling |
| 2. Landed optimizations | prefetch, block reduce, ROI decode, and what each bought |
| 3. ROI decode | it is *more accurate* than what it replaced — plus three load-bearing traps |
| 4. Target-width sweep | the cost model it refuted, and its withdrawn sensitivity half |
| 5. Scale vs block size | the storage/compute tables; why block tracks scale |
| 6. Cost model and knee | `t(s) = F + M·s²`, `s* = √(F/M)`, and why one pass cannot fit it |
| 7. Detection panel | why it was removed and must not be rebuilt in that form |
| 8. Calibration | the fiducial cancels; two ownership traps that each bit once |
| 9. GUI bugs | the Qt exit-9 crash, the silent 20x downsample, the false pass |
| 10. ROI pre-transcode | the 25x decode win, why lossless saves no storage, and four traps |
| 11. Per-channel extraction | the measured 1.59x (not the ~4x this plan asserted), and why the placeholder is zero-length |
| 12. The headless path | per-video extraction, the two clocks, why truncation fails the job |
| 13. File-level partitioning | the four silent ways a shard examines less than it claims; resume by identity, not existence |
| 14. N-worker throughput | the ~130 fps / 8-worker ceiling and the node count to quote |
| 15. Decodable frame count | claim vs measurement, and why a tolerance is not a fix |
| 16. Clip-backed throughput | why the 25x decode win is 1.06x end to end, and where it may still be real |
| 17. Plot collapse | the 3.2x, why it does not scale with block size, and the detector the collapse nearly disarmed |
| 18. Detection readout | why the gate moved onto the band's own plot, the 1 px floor, and the badge's shift-peek exemption |
| 19. Replicate direct manipulation | where T20's dropdown actually lives, why right-click-to-delete was refused, and two silent-write drag bugs |
| 22. Batch P streaming generator | the measured nil cost, why absolute fps is not comparable across sessions, and the three traps in the new seam |
| 23. Batch Q continuous surface | the cube-build livelock, the 1-frame window that renders, the backwards-drifting cursor, and why a slice with no observable behaviour means the boundary is wrong |

---

## 5. Remaining batches
### ~~Batch P — streaming extraction generator~~ · `core/tensor_channels.py`
**LANDED. Full write-up in `FINDINGS.md` §22.** What Batch Q needs to know:

- **`stream_channel_planes(video_path, plan)`** yields
  `(absolute_index, {channel: (ny, nx)})` per frame and RETURNS the pass
  metadata (`StopIteration.value`) — `n_frames`, `truncated`, timing spans.
  Absolute indices, contiguous, every wanted channel present on every frame:
  the dict feeds `StreamBuffer.append` with no adapter, which is pinned by a
  test rather than asserted here.
- **`plan_channel_stream(meta, ...) -> ChannelPlan`** is decode-free and
  resolves the window, geometry and gating. **Build this first and size the ring
  from it** — that is the whole reason it exists, so the clamp of `start`/`n`
  against `n_frames` is not computed a second time next to the buffer.
- The windowed API did not change, and neither did `CHANNEL_VERSION`.
- **The refactor cost nothing**, measured back-to-back against the pre-refactor
  file: 79.5 fps either way. But §22's warning applies — absolute fps drifts ~15%
  between sessions on this machine, so do not compare a Q measurement against a
  number taken today.

---

### Batch Q — continuous live surface · `gui/explorers/live_scalogram_surface.py`

**Goal.** Replace extract-a-window-then-block with: display immediately, process
forward continuously, plots fill in as frames arrive.

`_LiveExtractWorker` appends Batch P's frames to a `core.stream_buffer.StreamBuffer`
and publishes `frontier` by queued signal; the GUI asks the worker for windows and
never touches the buffer (the buffer is not internally synchronised, on purpose —
see its module docstring).

**Slice 1 of 3 LANDED (2026-07-20): the worker.** `gui/stream_worker.py` now holds
`StreamWorker` (moved out of the surface so both files could share it without a
cycle) and the new **`LiveStreamWorker`**, with `tests/test_stream_worker.py` (9).
What the remaining slices can assume:

- It owns the `StreamBuffer` and fills it from `stream_channel_planes`, emitting
  `advanced(start, frontier, fps)` at ≤10 Hz plus **one unconditional final
  emission** — the throttle must never blur "processing" into "done".
- **`request_latest(n, channels, token)` parks a request; it does not return
  data.** The GUI cannot read the buffer, by construction: servicing happens on
  the worker thread between frames and comes back on `window_ready`. Newest
  request replaces an unserved one — no queue, because the consumer is a repaint
  timer and a backlog would redraw windows the user has already left.
- `advanced` publishes **`start` as well as `frontier`**, because past capacity
  the island is no longer `[0, frontier)`. A progress readout that assumes it is
  will claim history the ring has already dropped.
- `pass_meta` (with `truncated`) is readable after `done` — the difference
  between "the clip ended" and "the decoder stopped early".
- `capacity` is passed IN, computed by the caller from the plan via
  `capacity_for_budget`. Keeping the geometry arithmetic out of the worker is
  the whole reason `ChannelPlan` exists.

**Trap found while testing, and it will bite the next slice too:** a synthetic
clip runs a 400-frame pass in ~0.12 s, which is shorter than one publish
interval — so any test that reacts to a mid-pass `advanced` is racing the pass,
not testing it. Park the request before `start()` where determinism matters.

**Slice 2 of 3 LANDED (2026-07-20): the surface.** Full write-up in
`FINDINGS.md` §23. **It was NOT file-local** — `ScalogramExplorer` derives `T`,
regions, `freqs` and the cube cache at construction and had no in-place update,
so the slice needed `set_channel_data` there too. What slice 3 can assume:

- **`Live ▶`** runs `LiveStreamWorker` to the end of the clip and updates the
  hosted explorer in place at ~1 Hz; `_show_live_window` is the seam, and
  `self._live_window` holds the last served window.
- **`ScalogramExplorer.set_channel_data(cd)`** accepts a new span under a
  **fixed** geometry and refuses a grid/fps change (that is still a rebuild).
  Tuning, bands and selection survive it. The cursor is carried by **absolute
  video frame**, with `follow_latest()` for the live default.
- **Updates are gated on `explorer.is_building()`.** Do not remove this: the
  cube worker cannot be cancelled and will not relaunch while one is in flight,
  so an ungated 1 Hz update makes every cube arrive stale and the scalogram
  **never appears at all** (§23 trap 3).
- **`_MIN_LIVE_FRAMES`** stops a 1-2 frame window from being transformed as if
  it were a time series. It sits on the consumer and should move into
  `request_latest` alongside the worker's zero-row guard.
- The realtime-ratio readout is live, so the "falling behind playback"
  requirement below is met.

**Slice 3 remains:** the ~1 Hz bounded-trailing-window CWT **with the cone of
influence drawn**. The COI is still the one thing that must not ship wrong.

**Decided (2026-07-20), do not relitigate without a reason:**

- **One island.** Seeking forward past the frontier abandons the span and restarts
  there. No gap list, no backfill worker. Backward seek inside the ring is instant;
  before it is just another restart.
- **Ring buffer, oldest dropped**, capacity from `sysmem.budget_bytes`. This is what
  keeps a fine block grid usable — block 16 on a 30k-frame clip is ~1.8 GB for two
  channels.
- **The CWT recomputes on a ~1 Hz timer over a bounded trailing window**, never over
  the growing island. `morlet_band_power` is O(F·T log T·B); re-running it over 30k
  frames every tick is not interactive.

**The one thing that must not ship wrong: draw the cone of influence.** Morlet at
w0=6 has e-folding support ≈ 1.46/f seconds — ~2.9 s at 0.5 Hz, ~0.07 s at 20 Hz.
So the newest seconds of the scalogram are zero-padding artifact at low frequencies
and fine at high ones, and progressive fill is *naturally frequency-dependent*.
Rendering that edge as data is the plausible-looking-wrong-number failure this
codebase designs against everywhere else, and it is the easiest thing here to ship
by accident. Fade or hatch the wedge.

**Throughput sets a UI requirement, measured 2026-07-20 on `GX010047c2`, 6 tiles,
block 64, scale 1.0:** `intensity`+`change` runs ~80 fps (**3.3× realtime**), but
adding `appearance` drops it to ~18 fps (**0.75× — falls behind playback**), because
the flow solve is ~24% of the pass. Continuous processing is therefore only possible
for a channel subset, and the surface must show the frontier falling behind rather
than showing stale plots as current. Channel selection is now a live-performance
knob, not only a cost knob.

---

### Batch R — marking, rehomed onto the streaming surface · `gui/timeline.py` + surface

**Goal.** Get labelling back, on the live surface rather than the retired tab.

Mostly rehoming, not writing: `gui/timeline.py` already has the whole span model
(`marks`, `marks_to_dict`, `set_marks_from_dict`, middle-drag to lay spans,
per-label colours), and `docs/archive/`-era `gui/_shelved/tab3_behavior.py` has the
persistence, the label picker and the video-scoped sidecar path
(`state.video_sidecar("marks")`). Read both before writing anything.

**Preserve the scoping contract.** Marks live next to the video and are scoped to
it; opening a different clip must load ITS marks or none, never inherit the previous
clip's. That is the one behaviour the shelved tab got right and is easy to lose.

**T30 is the point of this batch** — the corpus, not the widget. A marking UI with
nothing marked in it does not unblock anything.

---

### Batch G — replicate tab (~~T11~~, ~~T12~~, T20, T21) · `tab2_replicates` + `video_panel`
**T11 and T12 landed. T20 left the batch, T21 is untouched — G is NOT closed.**
Full write-up in `FINDINGS.md` §19; what matters here:

**T11 — LANDED.** `stamp_size` is now a property derived from the selection
rather than a stored value written by the last box *drawn*. Those were two
pieces of state answering one question and they diverged whenever you selected
an older box: the highlight said one thing, the next click placed another size.
Same remedy class as T17 and the count-band re-denomination — delete the second
copy rather than keep it in step.

**T12 — LANDED, except the right-click, which was refused on purpose.**
`FrameView` grows an opt-in `box_drag_enabled` (off by default, so the four
explorers, `mask_dialog` and tab3 keep clicking *through* their boxes) plus
`box_grabbed` / `box_clicked` / `box_moved`. Press selects, release-in-place
zooms, drag repositions. Right-click stays `back_requested` = un-zoom: it means
that on five other screens, and T12's own zoom is what makes this tab need one,
so honouring the item would have made one button mean "go back" on five screens
and "destroy a replicate" on the sixth. Delete stays on the button.

**The zoom is the box's `frac` verbatim — the same call the explorers make.** A
margin was built first and removed: same replicate at a different magnification
is not comparable by eye, which is the whole reason to zoom. Its stated
justification ("no room left to drag into") was also false, and the test that
now pins dragging at full zoom exists because the no-margin choice makes that
extrapolation load-bearing.

**Two silent-write bugs, both found by review and confirmed with real mouse
events**, both shaped "writes a change the user did not make": `moved` and the
displacement came from separate sources (reachable via Qt's move-event
compression — a release that reads as a 150 px drag while the cached delta is
zero, firing `_rebuild_rois` and a sidecar write for a no-op), and a chorded
right-click mid-drag committed a partial drag under a coordinate mapping the
un-zoom had already changed. `tests/test_replicate_tab.py`, 26 tests.

**T34 — moving a PROCESSED replicate is now gated. LANDED (2026-07-20).**
Leaving the Replicates tab latches every box then present as `processed`
(persisted in the sidecar); dragging a latched box raises a confirmation whose
button reads *"I recognize the existing replicate data will be lost"*, and
accepting clears the flag so it does not warn twice about data already gone.

Three choices worth not re-deriving:
- **A gate, not a freeze.** Freezing would force delete-and-redraw for a small
  correction, and redrawing loses the label, colour and standardization along
  with the geometry — so the safe-looking option would cost *more* state than
  the dangerous one.
- **Latched on leaving the tab, and only there.** That is the only route to the
  surfaces that consume the layout, so it strictly precedes any pass; a second
  trigger on the pass itself would be redundant. Boxes drawn after a latch are
  fresh until the next one — the flag is per replicate.
- **`_load` strips the flag, `_load_sidecar` keeps it.** An imported plate layout
  has not been processed against *this* clip, and carrying the lock across would
  be the ghosting-between-videos failure `_load_sidecar` exists to prevent.

*The regression it caused, since it is the T11 trap again:* the first cut called
`_refresh_list` to update the badge. That clears the list and the selection with
it, and `stamp_size` is **derived** from the selection — so every move silently
emptied the stamp. Badges are now edited in place (`_sync_lock_badges`). Any
future per-row indicator must do the same. `tests/test_replicate_tab.py`, 42.

**Still open in this batch:**
- **T20** — left the batch entirely; see the item above and §19. It is explorer
  work, and a state-ownership change rather than a deletion.
- **T21** (memory) — untouched, and still **must not be guessed at**. The plan's
  own note stands: `_refresh_list` was inspected and is clean. It needs a
  measurement. Nothing in this batch's changes bears on it either way.

---

### Batch L — quiet-tile gating (DEFERRED, design recorded) · `core/tensor_channels.py`
Wanted, explicitly not now. Recorded so the reasoning is not re-derived. Becomes
more attractive after K, since K is what makes math the bottleneck again.

**The idea.** TRex/TGrabs (Walter & Couzin 2021) and idtracker.ai reduce data by
subtracting a background model, thresholding to blobs, and running everything
downstream on blobs only. The same *data reduction* is available here with no
background model, because the structure tensor already computes the discriminant:
`change` is `<I_t^2>` per block, documented as `J_tt` — squared frame differencing,
the crudest form of background subtraction, as a first-class channel.

**Gate within the frame, not with a lagged rolling window.** `I_t^2` falls out of
`tensor_products`, which is early and cheap (~7%), while `flow_solve` /
`appearance` / `texture` are the expensive downstream work. Compute products
everywhere, then skip the rest for block-tiles below threshold and write a
known-quiet value. No fitted asset, no lag.

**Must be tile gating, not pixel masking.** A mask does not skip work in a dense
array op, and a scattered gather breaks the math outright — `tensor_blur` is a
spatial convolution and the structure tensor is a neighbourhood operator. Only
*contiguous block-tile* gating both saves work and preserves the fixed grid the
atlas, `block_weight_plane` and `region_blocks_and_grid` assume.

**Why TRex's blob extraction does not transfer.** TRex is Lagrangian: the blob IS
the unit of analysis, so non-blob pixels are nothing. Here detection is per-block
band power on a fixed grid, and background blocks are the baseline the statistics
are computed against. A gated block needs a *defined* contribution to the count and
the clump — it cannot simply vanish.

**Measure tile occupancy first.** Computable today from the `change` channel of an
ordinary pass: what fraction of block-cells fall below a candidate threshold,
across footage types. ~90% quiet means gating removes most of the math; ~40% means
bookkeeping eats it.

**Second cheap measurement, inherited from closed Batch I: clip-backed throughput
at N=8.** It belongs here rather than with I because it asks L's question, not
I's — §14 suspects memory bandwidth is the 8-worker ceiling, and 8 processes
decoding 1/12 the pixels is the one configuration that would separate a
bandwidth ceiling from a decode-contention one. Single-process says 1.06x
(§16); the multi-worker case is untested and is the configuration that actually
runs a corpus. A bench run on `scripts/bench_worker_scaling.py` with clip paths,
not a batch. **If it comes back ~1x, the memory-bandwidth hypothesis firms up and
L is the only remaining lever** — which is the result that would matter most.

**Threshold validation is tractable and not optional.** The gate and the detector
read the SAME quantity, so the gate can be expressed in units of the detector's own
sensitivity floor ("gate at X% of floor"). The hazard is worse than downsampling's:
downsampling loses small/fast structures and fails *uniformly*, whereas a change
gate loses low-contrast, slow, subtle motion and fails **selectively on quiet
behaviour** — stillness, antennal movement, slow postural shift — which is often
the behaviour of interest. Deleted signal is indistinguishable from "nothing
happened". Off by default.

**Do not build on `bg_subtract`.** Its median/MOG2 stubs exist but are deliberately
forced off for the tensor path (results flagged `approximated`), because background
models are fitted assets the cache cannot reconstruct. The `J_tt` route sidesteps
that entirely, being recomputed from the frames like everything else.

---

### Batch N — block size: three small things, **not** a sibling decision tool
**Rescoped.** This was specced as a pop-out dialog for `block_size` built on
Batch M's components (`core/cost_model.py`, `gui/cost_panels.py`,
`core/scale_sweep.py`, `core/scale_render.py`). That framing is wrong and the
work is much smaller than a dialog.

**Why the dialog does not transfer: block size has no knee.** M earned its
complexity because downsampling has a genuine curve — `t(s) = F + M·s²`,
`s* = √(F/M)` (§6) — so a frontier plot has a shape to show and an optimum to
mark. Block size does not: per §5 compute is **flat** (−13% across the whole
range) while storage falls as ~1/block². That is monotonic in one axis. There is
no tradeoff to visualize, so `FrontierPlot` and `cost_model` — the two components
that made M worth building — have nothing to do here. Inheriting them would be
machinery for a problem shape that is not present.

Nor does the "what you lose" panel transfer. Downsampling coarsens the per-pixel
field feeding the tensor solve, so the image is visibly blurrier and
`RenderStrip` shows it. Block size leaves the per-pixel math **untouched** and
loses **spatial localization** instead — fewer, larger cells. There is no
blurrier image to render.

**What is actually wanted, in place of the dialog:**

1. **State the storage cost.** One sentence, already measured in §5: ~0.9 TB per
   3000 h at block 64 against ~11 TB at block 16. This is the whole decision for
   most users and it does not need a plot.
2. ~~**Warn that changing block size invalidates the tuned detector.**~~
   **LANDED (`1d1f958`), and as a conversion rather than a warning.** The
   hazard, which is why this was the load-bearing item: `inband_count` produces
   a **raw block count** and `detect_gate` compares it against raw `count_band`
   endpoints with no normalization by region size (`clump` is in block units
   too). A region holds ~29 blocks at block 64 and ~377 at block 16, so **the
   same `count_band` means something ~13x different** — `[20, ∞)` is meaningful
   at block 16 and unreachable at block 64, and the detector does not clamp or
   fail when that happens, it just silently stops firing. Same class as T17.

   `core.detection.rescale_count_band` now carries a band across the change.
   The factor comes from **actual region block counts, not the block-size ratio
   squared** — those disagree whenever the grid does not divide the working
   frame evenly, and the actual count is what the detector compares against.
   `capture_view_state` records `count_denom` (block size, region blocks, and
   the region index, which is load-bearing because replicate tiles are not
   uniform and a rebuilt explorer resets its selection). This plan offered the
   re-scale "if that is cheap"; it was.
3. **Possibly a grid overlay** on the live view showing cell size against the
   animal, which is the honest form of "what you lose" for a localization lever.

Items 1 and 3 remain open, and neither blocks anything.

**Still inherit M's required prose pattern** wherever this surfaces: state
plainly both that the lever can be what makes a project feasible, *and* that it
is deliberately not assumed on the user's behalf. One half without the other
produces either avoidance or silent degradation.

---
## 6. Order

**I → J → O → D → E → F closed (specs archived). N item 2 landed. G is half done:
T11 and T12 landed, T20 left the batch, T21 is untouched.**

---

### The current thread (2026-07-20): live streaming, then a corpus, then signatures

This is what the last session was actually about, and it supersedes the throughput
thread below as the near-term order.

1. ~~**Batch P** — streaming generator in `tensor_channels`.~~ **DONE**, §22.
2. **Batch Q** — continuous live surface on `core/stream_buffer.py`. **Slices 1
   (the worker) and 2 (the surface + `set_channel_data`) landed; slice 3 (CWT +
   cone of influence) is next.**
3. **Batch R + T30** — marking rehomed, and **an actual corpus laid down**. The
   widget without the corpus unblocks nothing.
4. **T31** — validate occupancy/ratios against that corpus, on the tail statistic.
5. **T32** — supervised signature fitting; decides which channels earn their place.
6. **Batch S** (unwritten, gated on T31) — velocity gradient tensor: strain rate and
   vorticity from the block flow. `∇v` is **translation-invariant by construction**,
   so it measures posture change with no tracker and no body frame — the one
   configural read available in the low-resolution regime. `features.py` already has
   `_divergence`/`_curl` for the flow path; neither is exposed on the tensor path.
   Needs `(u, v)` exposed as a channel, which `flow_from_tensor` computes and
   `tensor_channels` currently discards in favour of the magnitude.

**Deliberately NOT next, and why:**

- **Wavelet phase** (inter-block coherence, instantaneous frequency). Real, but
  `morlet_band_power` is built to never materialise the complex cube — that is what
  makes whole-clip passes memory-bounded — and retaining phase breaks that contract
  for a speculative payoff. It also needs enough blocks per animal that two
  oscillators land in different blocks, which is not the current operating point.
- **Multi-lag temporal gradients** (Δ ∈ {1,2,4}). The slow-motion argument is real in
  principle but **cannot be demonstrated on the current corpus** — every labelled
  behaviour is fast or zero. Building it blind is exactly what the Batch K principle
  forbids. The half that *does* bite is `tensor_speed` being biased low for fast
  motion (the gradient method is a first-order linearisation), which argues for a
  narrow lag set aimed at displacement range, not a pyramid aimed at slow behaviour.
  Note the trap either way: a lag-Δ difference is a comb filter with **zeros at every
  multiple of f_s/Δ** — at 60 fps, Δ=4 is blind at 15 Hz — so lags must be chosen as a
  set whose nulls cover each other, never singly.

---

**The older throughput thread, still open and still not a build:**

1. **T21 (memory), the only open item on the replicate tab.** Still needs a
   *measurement*, and the plan's standing instruction not to guess at a fix has
   survived two sessions of inspection. Batch G did not touch it and does not
   bear on it. If it is real it wants a run under `tracemalloc`, not a reading.
2. **T20**, now understood to be explorer work — see its item and §19. Not
   urgent: the combo is redundant *as navigation* but is currently the only
   route to pooled scope, so it is a rehoming job, not a deletion.
3. **Batch L's two cheap measurements** (tile occupancy from the `change`
   channel; clip-backed throughput at N=8), which remain the plan's real
   throughput question. Take both before building anything.

Batch N items 1 and 3 are one sentence and a maybe; neither blocks anything.

**Scope cut, 2026-07-19.** Batch H (T22, viewer tab) is **dropped** — unbuilt,
self-scoped as the largest batch, a convenience shell nothing depends on. Batch I
(T23) is **closed rather than continued**, on the measurement rather than on
taste: §16 priced its win at 1.06x end to end. Both removals are about the same
thing — this plan grew two large items justified by numbers taken in isolation
(25x decode) or by no number at all (the viewer tab), and neither survived being
asked what it buys the pipeline.

**What that leaves as the real throughput lever: Batch L (quiet-tile gating)**,
because at scale 1.0 the pass is math-bound. Its two prerequisites are both cheap
measurements, not builds, and are recorded in L: **tile occupancy from the
`change` channel**, and **clip-backed throughput at N=8** (inherited from I).
Take both before building anything.

**The file-locality hazard has now fired FOUR times — treat the "· file"
annotations as guesses.** Batch Q slice 2 is the fourth: specced as file-local to
`live_scalogram_surface.py`, it needed `set_channel_data` on `ScalogramExplorer`
as well, because that class derives its whole time axis at construction. §23
records the cheap tell that generalizes: **if a slice has no observable
behaviour, the boundary is in the wrong place** — noticing that is cheaper than
re-deriving the file list up front.
 D's stated locality was wrong (`MiniPlot` lived in a
file the spec did not name). E only escaped because D had already moved it. G
made it three: **T20's dropdown is not on the replicate tab at all**, and worse
than being in the wrong file, it is not the *kind of thing* the item said —
authoritative state in five explorers rather than a redundant convenience on
one tab (§19). The rule has grown a second half: check what a widget derives
from **and where its state actually lives** before trusting the locality line.
An item that names a file is asserting two things, and the second one is the
one that has been wrong every time.

Explicitly deferred and not blocking: **Batch L**, the pixels-per-body-length
denomination, and the per-behaviour sensitivity study that would justify any
non-1.0 default. The latter two need labelled events; `marks.json` has one span.
