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
| ~~**T20**~~ | ~~Delete the `region_combo` dropdown in the explorers.~~ **DONE.** The "five explorers" was stale — it was **one** combo, in `scalogram_explorer.py`, and it was the *hub*: click-to-select and right-click-to-pool routed through `setCurrentIndex` → `currentIndexChanged` → `_on_region_changed`. Replaced with a single `_set_active_region(new_index)` sink the gestures (and `select_region` for session restore) write directly; the combo became a passive `region_lbl` readout. The combo also carried the region→`tuning_changed` relay, so the sink now emits it. `_on_region_changed` deleted, `QComboBox` import dropped. `test_view_state._select` repointed at `select_region`. 148 GUI tests green. |
| **T21** | Suspected runaway memory issue somewhere on the replicate tab. **Not reproduced by inspection** — `_refresh_list` was the obvious suspect and is clean (14x14 swatch pixmaps, `list.clear()` releases the items). Needs an actual measurement, not more code reading; do not guess at a fix. |
| ~~**T22**~~ | ~~New **viewer tab** consuming a fully-processed video.~~ **DROPPED** (2026-07-19) — scope, not value. A convenience shell over a `DetectionResult` that nothing else depends on; never started. Re-open only if day-to-day review actually stalls without it. |
| ~~**T23**~~ | ~~**ROI pre-transcode**: cut each source once into per-replicate clips + manifest.~~ **DONE** — see Batch I. Works, and **the premise it was justified on turned out false**: the 25x isolated decode win is **1.06x end to end** (`FINDINGS.md` §16). Not the lever that moves the floor at scale 1.0. |
| ~~**T24**~~ | ~~**Headless batch driver**: no-GUI entry point, file-level partitioning, N-worker throughput.~~ **DONE** — all three slices landed; throughput measured in `FINDINGS.md` §14 (**~130 fps / 5.4x realtime, ceiling at 8 workers**). |
| ~~**T29**~~ | ~~**Streaming live surface** — display immediately, process forward continuously, plots fill as frames arrive, instead of extract-a-window-then-block.~~ **DONE — Batches P and Q are both closed (2026-07-20).** Foundations `e11a4cb` (`core/stream_buffer.py`); Batch P `73ac827` (`stream_channel_planes` + `ChannelPlan`, §22, cost measured at nil); Batch Q slices 1–5 `aefa8cc` → `249fce1` → `8f2d198` → `d944cad` → `c7634c1` (§23, §24). T35 closed by slice 4. Suite 655 passing on a clean run. |
| ~~**T30**~~ | ~~The mark corpus is not a corpus.~~ **CORPUS LAID DOWN (2026-07-20).** `Videos/Stabilized/rep3_intermittent_crop.marks.json` holds **152 hand-verified Flying bouts** (scrubbed repeatedly — Kendrick reports more accurate than hand-labelled), 49% of frames, with real not-flying between them. Enough to validate the wingbeat band (T31). **Still missing for the occupancy claim:** still-with-animal / animal-absent spans — see the **Occupancy** item. `rep6-no-flying` is the ready negative control. |
| **T31** | **Validate channels against marked footage on the tail statistic** (`FINDINGS.md` §20). **DONE for flying-vs-not (2026-07-20):** wingbeat band [13.66–25 Hz] separates flying near-perfectly, is **frequency-specific** (low [1–5 Hz] band degrades — intensity drops to chance), and **`butter`+`filtfilt` ≈ Morlet** — measured via the channel lab (now `scripts/channel_lab.py`). **Caveat: the contrast is saturated** (flight is gross whole-crop motion), so it validates the *band* but does not rank channels finely. |
| **T32** | **Supervised signature fitting. Large — its own branch.** Fit LDA / L1-logistic per labelled behaviour over the (channel × spatial scale × temporal scale) grid. The weight vector *is* an interpretable, tunable signature and drops into a template detector with the existing `DetectionResult` shape. Tells you which channels earn their place before the unsupervised map is built around them. |
| **Occupancy** (future) | **Test present-but-still vs animal-absent on the tail statistic** — the half of `FINDINGS.md` §20 the rep3 corpus cannot reach (the animal is always present). Needs still-with-animal / absent spans; `rep6-no-flying` is the ready negative control. Deferred. |
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

**The clip-provenance cache obligation (from Batch I).** `PipelineConfig.cache_key`
was **DELETED** — the flow cache was its only consumer, and after the teardown it
had zero production callers (its named call sites `core/pipeline.py` and
`scripts/validate_standardization.py` are themselves gone). So there is no key to
thread a `provenance_key` through today. The obligation survives as data, not code:
the clip's provenance travels as `meta["clip_provenance"]` out of
`live_channel_source`, and the day something first caches a clip-derived result it
MUST fold that into the new cache key — below `lossless` a clip-derived and a
source-derived result are different measurements (`FINDINGS.md` §10). Re-introduce
the key with `provenance_key` **required and keyword-only** so the omission is a
deliberate claim, not an oversight.

**Shelved (T8, T9) — now DELETED.** `tab1_flow.py` and `tab3_behavior.py` were
shelved to `gui/_shelved/` and then deleted outright with the flow-cache teardown
(the whole `gui/_shelved/` package is gone). `MainWindow` is down to two tabs, and
`tests/test_cache_naming.py` was deleted with the `_test_cache_suffix` it covered.

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
| G (part) | replicate direct manipulation (T11, T12) — T20 done, **T21 still open** | — |
| — | occupancy + conjunction channels, live stream ring buffer | `e11a4cb` |
| P | streaming extraction generator (`stream_channel_planes`, `ChannelPlan`) | `73ac827` |
| Q | continuous live surface, all five slices (T29, T35) — **closed** | `aefa8cc`…`c7634c1` |

**Closed batch specs live in `docs/archive/closed-batches.md`** (D, E, F, I, J, O
archived 2026-07-20). Their durable output is `FINDINGS.md`; the specs are kept
only because several were wrong in instructive ways.

Their durable output is `FINDINGS.md`. Everything else about them has been deleted.

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
| 24. Batch Q continuous plots | the tier table, why the pooled Morlet is NOT the expensive call, the percentile that re-opened a closed decision, and the hatch that inverted its own purpose |
| 25. Batch S routing | the three route outcomes and why the middle one cannot raise, the rational-fps fix the streaming path never inherited, and why the clips checkbox is not persisted |
| 26. Batch S retired geometries | why a gray-out is not enough for a moved box, how the ruling made the slice local, and the in-memory cache that undid the retire |
| 27. The teardown crash | it was ours, not the machine; the two ways a cube thread outlived its widget, and why instrumenting the PRECONDITION beat chasing the crash |

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

**Slice 3 LANDED (2026-07-20): the cone of influence, plus the time axis it
needs.** `core.wavelet.coi_efolding_s` / `coi_edge_samples` derive the wedge from
`morlet_scales` rather than a baked constant, and `ScalogramPlot` fades it toward
the plot background at both ends. Two things worth not re-deriving:

- **The plan's own constant was wrong.** This file said the e-folding support is
  "≈1.46/f seconds". Torrence & Compo give τ = √2·s with s = 0.96805/f at w0=6,
  i.e. **1.369/f** — 2.74 s at 0.5 Hz, not 2.9. ~7% too large. Deriving it from
  `morlet_scales` in code is what makes the discrepancy moot.
- **`ScalogramPlot`'s time axis was broken from the file's first commit**
  (`6466687`), and the COI could not have been built on it. `col = (arange(w)*T)
  // T` cancels to `arange(w)`: the heatmap gathered the first `w` FRAMES and
  stretched them over the full width, showing ~5% of a whole-clip cube while the
  cursor mapped over all `T`. A transposition of `DensityPlot`'s scatter, which
  divides correctly. It hid precisely the newest edge the COI is about.

**T35 — CLOSED 2026-07-20 by the whole-video rebuild below**, in the cheap way
this section predicted: `core.live_track.coi_trim` derives the trim from
`coi_efolding_s` at the BAND'S LOWEST frequency, and `WholeVideoTrack.write`
simply does not write the contaminated edge frames. A later overlapping window,
for which those frames are interior, supplies them. The detector never produces
the frontier-transient numbers, so there is nothing to mask. `trim_head` /
`trim_tail` overrides keep the clip's true ends, which are edges of the DATA
rather than of an arbitrary cut. **No detector-side mask was added**, as this
section instructed.

**The original statement of T35, kept because it is the clearest description of
the failure class:**

**T35 — the DETECTOR still consumes the cone it now fades. OPEN, and the most
important thing slice 3 did not close.** `detect_channel_region` →
`morlet_band_power` → `inband_count` → `detect_gate` reads every frame of the
window, edges included. At 24 fps the newest 66 frames are padding response at
0.5 Hz, so a low-frequency edge transient can push `inband_count` over
`count_band` and report a detection **at the frontier that vanishes when the
window slides past it**. Fading the wedge on the display while the detector
trusts the same values makes the picture honest and the numbers not — which is
the failure this slice existed to prevent, moved one layer down.

*The redesign is where this gets fixed, and it fixes it cheaply:* with an
accumulating whole-video series, a window's COI-contaminated frames are simply
**not written** into the accumulator, and a later overlapping window — for which
those frames are interior rather than edge — supplies them. Coverage then means
"computed from frames the transform could actually see", which is the honest
definition anyway. Do NOT paper over it with a detector-side mask.

**REDESIGN (2026-07-20, from use): the live axis is the WHOLE VIDEO.** Live mode
should match the post-commit detection strip (`DetectionNavigator`) rather than a
sliding window — plots continuous, moving with time, skipping around generating
parts of the same whole-video picture, so navigation is whole-video throughout.
This supersedes the two decisions immediately below; both are struck rather than
deleted, because the reasoning that produced them is still what bounds the build.

**What makes it affordable, and the split it forces.** Per-FRAME series are
free at whole-video length — gate/clump are ~120 KB each at 30k frames, the
pooled scalogram ~2.9 MB — so those carry the whole-video axis and fill
progressively. **Per-BLOCK data cannot**: the `(F,T,B)` cube runs to several GB,
which is the whole reason the ring buffer and `sysmem.budget_bytes` exist. So
densities and the cube stay windowed around the cursor. The CWT still *computes*
over a bounded window; only the axis it is painted onto grew.

**The hazard this creates, and it is the standing one.** `_Strip` paints
unfilled regions `(18,18,22)` — identical to a computed `gate 0, clump 0`. Under
progressive fill that makes **unexamined indistinguishable from examined-and-quiet**
(§10 traps 7/8; the governing principle). A coverage mask is therefore the first
piece of this build, not a polish item.

- ~~**One island.** Seeking forward past the frontier abandons the span and restarts
  there. No gap list, no backfill worker. Backward seek inside the ring is instant;
  before it is just another restart.~~ **REVERSED.** Skipping around keeps earlier
  segments and the strip carries several. What made one island look right was the
  cost of retaining per-block history; the per-frame series that the whole-video
  axis actually needs are cheap enough that the argument does not apply to them.
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

**I → J → O → D → E → F closed (specs archived). N item 2 landed. P and Q closed
(2026-07-20) — T29 and T35 with them. G is half done: T11 and T12 landed, T20
left the batch, T21 is untouched.**

**Corpus laid down and the wingbeat band validated (T30, T31 — 2026-07-20).**
Marking rehoming onto the old span model is **dropped** — the future marking
surface will look different and is out of scope here. **The velocity gradient
tensor channel is DONE** (shipped in `9d843c0`; see the current thread below) —
built but not yet validated for want of a postural corpus. Signature fitting
(T32) is its own branch.

**Batch S is CLOSED (2026-07-21): all six slices landed.** It is where
per-replicate ownership was made real on disk, and it ran ahead of everything
else here because slices 1–3 each fixed a *silent cross-replicate overwrite*
rather than adding a feature. Slice 6 closed the last of that class: a moved box
no longer presents one animal's labelled behaviour as describing whatever the
rectangle now covers.

**So the plan has no live thread.** What remains, in the order the file argues
for: **Batch L's two cheap measurements** (tile occupancy from the `change`
channel; clip-backed throughput at N=8) — both measurements, not builds, and
together they decide whether quiet-tile gating is the only remaining throughput
lever. Then **T21** (the replicate-tab memory suspicion, which needs a
`tracemalloc` run and must not be guessed at), **T32** (supervised signature
fitting, its own branch), and the deferred **Occupancy** half of §20, which is
blocked on a still-with-animal / animal-absent corpus rather than on code.

---

### Batch Q slice 4 — the whole-video rebuild. LANDED 2026-07-20 · `d944cad`

**What it is.** The live axis became the whole video. The strip at the bottom is
now the clip's SEEKER and is always visible; it fills as detection runs, and it
paints three states that used to be two.

**New: `core/live_track.py` (`WholeVideoTrack`, `TrackStamp`, `coi_trim`).**
Clip-length `count`/`clump`/`gate` plus a per-frame `stamp_id`. Read the module
docstring before touching it; the four properties it exists to hold are stated
there. Tests: `tests/test_live_track.py` (23).

**The stamp split, decided with the user and NOT to be re-litigated silently.**
A `TrackStamp` holds only what invalidates the retained BAND POWER — channel,
frequency band, grid, region, downsample, block size. The value band, count band
and detection window are deliberately OUT, because whole-clip `(T, B)` band
power is retained (~45 MB at 30k x 377, capped at `_TRACK_BP_CAP` = 1 GiB) and
those three re-derive over existing coverage instead. **A count-band nudge must
never gray out the clip.**

**Coverage is the first piece, and it is honest.** Three painted states:
unexamined (bare trough, no baseline rule), examined-and-quiet (baseline rule
LIT), examined-under-other-settings (desaturated). The baseline rule is what
keeps "nobody looked" out of "nothing happened" — §10 traps 7/8. Not decoration.

**A stale region's gate is FROZEN** (`_derive_gate` runs over `current`, not
`covered`). Found in review: re-gating stale frames combines a `count` computed
under one value band with a count band chosen later, and made the gray bars move
while tuning knobs that provably do not apply to them.

**Detection runs on the WORKER thread** (`LiveStreamWorker.request_detect` /
`detected`, `_serve_detect`). Not a performance choice: the DISPLAY path drops
windows while the explorer is transforming, which is right for a display and
fatal for an accumulator — coverage would get holes wherever the GUI was busy,
and a hole is indistinguishable from examined-and-quiet. There is also a final
post-loop serve, or the tail of every pass stays unexamined.

**Requests overlap by `_DETECT_OVERLAP_COI` (2.5) cones.** Windows that merely
abut leave a 2*coi seam unexamined at every join — a permanent gap created
purely by the request cadence.

**Both producers go through `_track_write`.** The commit pass now fills the SAME
accumulator (decided with the user) rather than a parallel `DetectionResult`.
`write` RAISES on a block-count mismatch, which in a Qt signal handler is a
crash, so the guard converts it to a reported False; `_DETECT_DROP_ALARM`
escalates a sustained run of drops, because a strip that never fills while
progress looks healthy reads exactly like a quiet clip.

**The file-locality hazard fired a FIFTH time, as predicted.** Specced as
`live_scalogram_surface.py` + a new core module; it also needed
`gui/stream_worker.py` (detection on the worker thread), `core/detection.py`
(`region_grid_from_meta` — the clump grid without the data) and
`gui/explorers/scalogram_explorer.py` (`seek_absolute` + a `frame_moved` signal,
because the strip is a seeker and nothing emitted an ABSOLUTE cursor).

**CLOSED by slice 5 below.** ~~The continuous-plots half of the user's ask.~~

---

### Batch Q slice 5 — continuous plots. LANDED 2026-07-20 · `c7634c1`

**Full write-up in `FINDINGS.md` §24.** The design call slice 4 deferred is
settled and built. Three things this section exists to stop being re-derived:

**1. The premise of the slice-4 spec was wrong, and in the user's favour.** That
section said *"The user asked for the scalogram at ~1 Hz (fine as-is) but the
OTHER TWO CONTINUOUS"* and built a two-cadence design around the scalogram being
necessarily slow. **The user did not ask for 1 Hz** — confirmed directly:
*"i didn't ask for it at 1hz, i said it would be whatever."* And it is not slow:
`scalo_plot` is fed by `morlet_power(pooled)` over a single `(T,)` series, **8 ms
at T=30k**. It is the per-BLOCK `(F,T,B)` cube — 0.44 s to 6 s — that is
expensive, and the only plots it feeds are the DENSITIES. So all three plots the
split was designed around go continuous together at 10 Hz (~12 ms total) and no
split between them was needed at all. *A cost attributed to the wrong widget
produced a whole design; measure which call site is expensive before designing
around it.*

**2. `is_building()` gating is GONE, and its docstring now says so.** It existed
because a cube arriving after its span moved was DISCARDED (§23 trap 3). Cubes
are now RETAINED and tagged with the span they cover (`_sg_span`), drawn at that
span by `DensityPlot.set_matrix(m, axis_off, axis_total)` with the remainder
hatched. A late cube became late data correctly placed, so the gate had nothing
left to protect and was only throttling the fast plots to the slow transform's
rate. **Its own trap, found in review:** once cubes outlive their span,
"cached" stops meaning "current" — `_request_cube` returning early on a cache
HIT would make the first cube of a pass the only one ever built. Hence
`_cube_is_current`.

**3. Registration, not staleness — the design call itself.** The slice-4 spec
framed the risk as a stale scalogram needing to be "visibly marked". That
understates it: these plots stack, share a width and are read down a column, so a
cube over `[0,8000)` beside traces over `[0,9500)` puts frame 7800 above frame
8300. A badge leaves that misregistration intact. Decided with the user: **pad
onto the shared axis, hatch the uncovered tail**, reusing slice 4's coverage
vocabulary. The lag becomes a geometric fact instead of a claim.

**The hatch inverted its own purpose on the first cut** (§24): derived from which
columns received a frame rather than from the span, it painted 236 of 300
*covered* columns as unexamined — destroying signal, strictly worse than the
state it prevents. It passed the full suite, because both registration tests used
spans denser than the failure needs. **"Covered" and "received a frame" are the
same question only when T ≥ plot width**, which no live window early in a pass
satisfies.

**`_ov_scale` moved to the slow cadence.** `np.percentile` is a full sort of
`T*B` — 53 ms at T=30k, more than three times the rest of the fast path — and
the line already carried a comment that it was frozen so scrubbing would not
re-percentile in the hot path. A 10 Hz caller reintroduced exactly that, one
layer up. Holding it steady is also the better picture, by §19's argument.

**Producer cost was measured too**, because the comment being replaced was about
the window COPY, not the plots: 1.8% of worker time at 10 Hz, worst case. Not a
constraint. *The rate was nearly justified by answering a different question than
the comment it overrode asked.*

Suite 655 passing. `tests/test_live_stream.py` gained 6.

---

**Also open, found in review and deliberately not fixed:** dragging the strip
runs `seek_absolute` -> `_apply_frame` -> `_redraw_video`, i.e. a video decode
per mouse-move. Pre-existing scrub behaviour, not introduced here, but the strip
makes it much easier to hit.

### The teardown crash was NOT environmental. FIXED 2026-07-21 · `FINDINGS.md` §27

**Everything below this heading is the record of getting it wrong for four
sessions, kept deliberately.** The "not ours" reading was reasonable at every
step and was still false. Two real defects in `ScalogramExplorer`'s cube-thread
lifetime let a `QThread` outlive the widget that owned it; Qt destroying a
running thread is an access violation, not an exception, which is why it
surfaced as *"843 passed"* followed by a faulthandler dump and a non-zero exit.

**What the four sessions' evidence actually showed.** Deselect-plus-isolate
proved only that the crash needs full-suite *context* — which is exactly what a
timing-dependent thread race needs, so the proof pair was consistent with the
bug the whole time. "The diff did not touch `gui/stream_worker.py`" was also
true and also irrelevant: the leaked threads are `_ScalogramWorker`s, and the
`_serve_pending` frame in the dump was a *different* thread that faulthandler
prints alongside the crashing one, because it dumps every thread.

**The generalizable lesson:** an intermittent crash that survives four
"unrelated diff" arguments is evidence about *reachability*, not about
ownership. The cheap decisive move was never attempted until now — instrument
for the PRECONDITION (a QThread still running when its test ends) rather than
try to reproduce the crash. That took one throwaway pytest plugin and found 14
tests leaking a running thread on the first run.

**How the fix is verified, and its limit.** The crash could NOT be reproduced on
demand — ~35 clean full runs this session against one firing early on, so the
rate is far below the "roughly half" originally recorded. So the fix is verified
against the **precondition**, not the crash: the leak probe went from 45 leak
events with ~14 tests leaving a *running* thread, to 28 events of which **every
one is a finished-then-deleted thread and none is still running**. Both fixes
also have regression tests that were confirmed to fail with their own fix
reverted, and only their own. If the dump ever recurs, this note is wrong again
and the probe is in the session log.

---

**The superseded environment note, kept because the reasoning is instructive:**
this machine throws
`Windows fatal exception: access violation` in `_serve_pending` during
`tests/test_stream_worker.py` on roughly half of full-suite runs. User reports it
is an aggressive university endpoint blocker; re-running gets through. An attempt
to confirm it on a clean tree was INVALID (the `git stash push -u` failed on
`.claude/` permissions and left the tree modified) — so it is unconfirmed, not
proven environmental. **It did NOT fire on the 2026-07-20 closing run** (655
passed, 57 subtests, 40.6 s, clean tree at `c7634c1`), which is consistent with
the intermittent story and still does not confirm the cause.

**2026-07-21 (Batch S slice 3) adds the cleanest data point so far, and it is
evidence AGAINST the crash being ours.** It fired once mid-session, in the same
`_serve_pending` frame, on a session that never touched `gui/stream_worker.py`.
Immediately after: 764 passed with `test_stream_worker.py` deselected, that file
passed **3/3 in isolation**, and two subsequent full runs passed clean (780).
So the crash needs the full-suite context and does not reproduce on the file
alone — still unexplained, still not reason to chase it, and the deselect-plus-
isolate pair is the cheap way to prove a session did not cause it.

**It fired again on Batch S slice 4 (2026-07-21), same frame, same outcome:** a
session touching only `core/pretranscode.py`, `core/channel_source.py` and
`cli/pretranscode.py`. 777 passed deselected, the file passed in isolation, and
an immediate full re-run passed 793. Three sessions, three unrelated diffs — the
"not ours" reading is now the only one the evidence supports.

**Slice 5 (2026-07-21) adds a fourth, and sharpens what it is.** The full suite
passed 817 twice; a third run reported **817 passed** and *then* died at
interpreter shutdown (pytest exit 5 with a faulthandler dump, no failed test).
So it is a teardown-time crash, not a test failure, which is why "817 passed"
and a non-zero exit code appear together. Proof pair again: 16/16 in isolation,
801 deselected. This session did not touch `gui/stream_worker.py` at all.
**Read the LAST line of the run, not the exit code.**

**Slice 6 (2026-07-21) is the fifth firing, and the first with a caveat worth
stating.** Same shape exactly: two full runs reported **840 passed**, a third
reported 840 passed and then died at teardown with the faulthandler dump and
exit 5. Proof pair again: **824 passed deselected, 16/16 in isolation.** The
caveat: unlike the previous four, this session DID touch a file the stream
worker is wired to (`live_scalogram_surface.py`, the `discard_replicate_track`
seam) — though not `gui/stream_worker.py` and nothing on the `_serve_pending`
path the dump names. So the evidence is one notch weaker than "never touched
it", and still points the same way.

---

### The current thread (2026-07-20): live streaming, then a corpus, then signatures

This is what the last session was actually about, and it supersedes the throughput
thread below as the near-term order.

1. ~~**Batch P** — streaming generator in `tensor_channels`.~~ **DONE**, §22.
2. ~~**Batch Q** — continuous live surface on `core/stream_buffer.py`.~~
   **CLOSED 2026-07-20, all five slices committed:** 1 (worker, `aefa8cc`),
   2 (surface + `set_channel_data`, `249fce1`), 3 (COI, `8f2d198`), 4 (the
   whole-video rebuild — `core/live_track.py`, coverage, staleness, the strip
   as seeker, `d944cad`), 5 (continuous plots — the axis registration and the
   10 Hz rate, `c7634c1`). See `FINDINGS.md` §23 and §24.
3. ~~**Batch R + T30** — marking rehomed, and an actual corpus laid down.~~
   **T30 DONE (2026-07-20): rep3 flying / not-flying corpus, 152 hand-verified
   bouts.** Marking rehoming is dropped — the future marking surface will look
   different, so its plan was removed rather than carried.
4. ~~**T31**~~ **DONE (2026-07-20)** — wingbeat band validated on the tail
   statistic via the channel lab (`scripts/channel_lab.py`, `scripts/run_lab.py`).
   Frequency-specific, `butter` ≈ Morlet; contrast saturated so it validates the
   band, not a fine channel ranking. Occupancy half deferred (see Open items).
5. **T32** — supervised signature fitting. **Large; its own branch.**
6. ~~**Velocity gradient tensor channel**~~ **DONE — shipped in `9d843c0`** (the
   Phase-1 commit, before the flow-cache teardown; this line was never updated).
   `∇v` decomposed into its three 2-D invariant parts — `vel_divergence` (trace),
   `vel_shear` (deviatoric strain-rate magnitude), `vel_vorticity` (antisymmetric)
   — in `core/channels.py`, per atlas region so a gradient never crosses a
   replicate seam. Translation-invariant by construction, so it measures configural
   change with no tracker. `(u, v)` ARE now exposed as base fields (`LIVE_CHANNELS`,
   `tensor_channels.py:502` — the old "discards in favour of the magnitude" claim is
   stale); the derived channels fold in via `with_derived_channels`. Wired end to
   end: live-surface extraction resolves base fields then derives, `vel_shear` is a
   first-class detection channel, the explorer menu carries all three (signed axis +
   no band for divergence/vorticity, per detection-channel design), stream worker
   derives `vel_shear` before the transform. Tests: `test_channels.py`,
   `test_channel_source.py`, `test_stream_worker.py` (34 green). **Not validated on
   marked footage** — the rep3 corpus is saturated flying-vs-not and cannot rank a
   configural channel (T31 caveat); a still/postural corpus is the prerequisite,
   same gap as the deferred Occupancy item.

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
2. ~~**T20**~~ **DONE** — the dropdown was rehomed onto the click/right-click
   gestures (a `_set_active_region` sink) and deleted; see its item above.
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

**The file-locality hazard has now fired FIVE times — treat the "· file"
annotations as guesses.** Batch Q slice 2 was the fourth: specced as file-local
to `live_scalogram_surface.py`, it needed `set_channel_data` on
`ScalogramExplorer` as well, because that class derives its whole time axis at
construction. Slice 4 was the fifth and needed three extra files (see its
section). §23 records the cheap tell that generalizes: **if a slice has no
observable behaviour, the boundary is in the wrong place** — noticing that is
cheaper than re-deriving the file list up front.
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
non-1.0 default. The latter two need labelled events — **the "`marks.json` has
one span" that used to stand here is stale**: T30 laid down 152 hand-verified
Flying bouts on 2026-07-20. What they still lack is *variety*, not volume; the
corpus is one behaviour on one clip and saturated (T31's caveat), so it cannot
price a sensitivity curve across behaviours.

---

## Batch S — per-replicate homes (started 2026-07-21, slices 1–3 landed)

The replicate box, not the video, is the unit of ownership on disk. A home is
`<stem>_rep01/` beside the source, holding that replicate's track, marks, tuning
and — optionally — a transcoded clip of just that region. The cut
(`core/pretranscode.py`, already built and CLI-only) becomes a pure speed-up that
drops a file into a directory that already exists, rather than a mode that
changes the layout.

**Three rulings this batch rests on. Do not relitigate without new evidence.**

1. **Homes are named by id, never by label.** `core/replicates.py:68`
   (`_canonical_geometry`) already decided labels are not identity. Naming by
   label would make a rename a directory move with every path inside it
   shifting, and duplicate labels a filesystem collision. With id naming a
   rename is one string in `rois.json` and duplicates are cosmetic.
2. **Nothing deletes a home.** Moving a box warns and enumerates; deleting a box
   stops referring to the directory. `TrackStamp` carries geometry per frame and
   `track_store` refuses a mismatch, so staleness is *detected* — deleting would
   do destructively and irreversibly what that already does safely, and would
   take a curated marks corpus with it on a 2% nudge.
3. **`next_id` is monotonic and persisted.** This is what makes (2) safe: a
   retired id is never reissued, so an orphaned home can never be silently
   adopted by a later box.

### Landed
- **Slice 1** · `core/replicate_home.py` (new), `gui/tab2_replicates.py`.
  Homes on box draw; `next_id` persisted; imports renumbered onto fresh ids;
  `_confirm_move` enumerates and deletes nothing; delete prompts only when the
  home holds something; duplicate labels warn rather than block.
- **Slice 2** · `gui/track_store.py`, `gui/explorers/live_scalogram_surface.py`.
  **The bug was bigger than the file path**: one `WholeVideoTrack` per clip meant
  replicate 3's pass overwrote replicate 2's `count`/`clump`/`gate` frame-for-
  frame *in memory*, with only the stamp id changing — the strip stayed
  plausible. Now one track per region (`_tracks`, `_active_region`), swapped in
  `_sync_track` before `set_stamp`, flushed to its own home on handover.
  Legacy sidecars are adopted once, only by the region named in their own stamp.
  `region_index` and `replicate_id` are passed separately on purpose — they
  coincide only while ids run 1..N with nothing deleted.

- **Slice 3 · schema split. LANDED 2026-07-21** · `gui/marks_store.py`,
  `gui/tuning_store.py`, `gui/explorers/live_scalogram_surface.py`.
  Built as ruled below. Three things worth not re-deriving:

  **The view handover is where the count-band hazard actually lives.** Splitting
  storage alone is not enough: tune replicate A, switch to B, and the debounced
  `_save_tuning` files A's `count_band` under B. So `_activate_region` saves the
  outgoing view before the switch and applies the incoming one, mirroring the
  track handover it already performed. `apply_view_state` turned out to be built
  for this — `_converted_count_band` already converts a band across differing
  region block counts.

  **The handover applies the view WITHOUT `frame`, and a test caught it.** The
  cursor is a position in the CLIP, not in a replicate (slice 4 made the strip
  the whole video's seeker), so re-applying a remembered frame yanked the view
  back from wherever the user had just navigated —
  `test_focusing_a_detection_while_stopped_loads_the_window` failed with
  `15 != 20`. The initial restore still carries `frame`, through
  `_pending_state`, where there is no navigation to fight.

  **`_save_view` writes only from a LIVE explorer.** `_saved_view` is a cache
  whose region is whatever was active when it was filled; with no explorer to
  confirm that against the region being switched away from, writing it could
  file one replicate's band under another's name — the very failure the split
  exists to prevent. Skipping costs nothing, since the debounce already wrote it.

  Also: `add_detection_spans` no longer assigns a colour (per-replicate
  insertion order would restart in every home), and `save_palette` runs AFTER
  the spans land, since a slot reserved for a failed save shifts every later
  label's colour. Suite 780 green.

- **Slice 4 · cut into the home. LANDED 2026-07-21** · `core/pretranscode.py`,
  `core/channel_source.py`, `cli/pretranscode.py`. Built as specced:
  `clip_filename(stem, id, layout)` returns `<stem>_rep01/<stem>.mkv` by
  default (`CLIP_LAYOUTS = ("home", "flat")`, `--layout` on the CLI),
  `ClipEntry.filename` widened to a POSIX relative path, no version bump,
  `clip_layout` out of `provenance_key` — pinned by a test that cuts the same
  source both ways and asserts the keys are equal. Four things worth not
  re-deriving:

  **The layout is NOT a manifest field.** Each `ClipEntry.filename` is the
  record, and a second copy of the same fact is the state duplication T11/T17
  kept deleting. It also happens to be what makes the flat→home change
  backward-compatible for free: a basename is a valid relative path, so old
  manifests read and resolve unchanged.

  **`clip_path(clip_dir, filename)` is the only supported way back to a path,
  and `os.path.join` on the whole string is specifically wrong.** It yields
  `dir\rep01/clip.mkv` on Windows, which *opens fine and so passes every test*
  while being unprintable and uncomparable. The POSIX-on-both-platforms rule on
  the stored string matters more: a backslash is a legal POSIX filename
  character, so a Windows-written path would resolve on Linux as one
  absurdly-named file rather than failing — a missing clip reported against a
  manifest that verifies.

  **`_cleanup` removes clips, never directories** (ruling 2). A failed cut
  leaves an empty home; the alternative risks a curated corpus on a cancelled
  transcode. Pinned by a test.

  **A flat→home re-cut orphans the old clips silently**, doubling the storage
  the CLI's own report just quoted. `cli._superseded` names them after a
  successful cut rather than deleting them — `-y` already replaced everything
  the new cut claims, and a CLI that removes video files it was not asked about
  is the worse trade. Not in the spec; found by asking what `--overwrite` does
  across a layout change.

  *Locality, seventh firing, mildly:* the spec named `clip_filename` and the
  CLI. It also needed `core/channel_source.py`, the one other place that joined
  `clip_dir` to `entry.filename`. Small, but the same shape — the spec named the
  producer of a string and not its consumers.

### The slice 3 rulings, kept because the reasoning still binds slices 4–6
- **Slice 3 · schema split — RULED 2026-07-21, all three questions answered,
  and built (see Landed above).**
  - **Tuning: `view` per-replicate, everything else per-video.** The home holds
    `{view: …}` — channel, freq band, value band, count band, detect window,
    selection — which is exactly `TrackStamp`'s invalidation set. `strip`
    (downsample, block, normalize, window), `process` and `region_index` stay
    beside the video as source properties. `count_band` settles it on its own:
    §N item 2's `count_denom` rescaling exists *because* a raw block count means
    something ~13x different per region, so it cannot be per-video.
  - **Marks: read-through, never migrate.** A home with no marks file falls back
    to the per-video `.marks.json`, **filtered** to its own region index — filter,
    never reshape, so the 152-bout rep3 corpus is read where it lies and no
    curated file is ever rewritten. Writes always go to the home; the legacy file
    goes inert once a home copy exists. This is `track_store.load_track`'s
    adopt-once discipline minus the adoption, and it is the safer half: marks
    carry no stamp naming their region, so an actual migration would have to
    infer index==tile-order-position — the inference slice 2 exists to forbid.
  - **`colors` stays per-video** (palette-by-insertion-order is a per-clip
    display contract: one behaviour label is one colour across replicates).
    It therefore keeps living in the per-video `.marks.json`, and the palette
    write is **load-modify-save touching only the `colors` key**, preserving
    `spans`/`provenance`/unknown keys verbatim. That file is a curated corpus.
  - **`provenance` moves per-replicate**, which dissolves rather than changes the
    "honest record" caveat in `marks_store`'s docstring: label-keyed overwrite
    was only lossy because two regions shared one file.
  - **The box `frac` joins `provenance`.** One copy, in the place that already
    records what produced a span — NOT a second top-level `geometry` key. Marks
    written between slice 3 and slice 6 must be able to say which rectangle they
    were labelled against, or they inherit the legacy corpus's own defect.

### Open
*(nothing — slice 6 closed 2026-07-21, and Batch S with it.)*

- **Slice 6 · retired geometries. LANDED 2026-07-21. It REVERSES a ruling above.**

  **Built as ruled.** `core/replicate_home.py` grew `retire_current` /
  `restore_generation` / `list_generations` / `next_generation`;
  `gui/tab2_replicates.py` retires on an accepted move and offers
  *Older geometries…*; `FrameView` grew a separate `ghost_boxes` list.
  Four things worth not re-deriving:

  **The rulings SHRANK the slice, and the reason generalizes.** Predicted to
  span `rois.json`'s schema, `replicate_home`, `tab2_replicates` and both
  stores. Home-authoritative geometry removed the schema change, and the three
  stores never learned generations exist — they write to the home *root* via
  `home_path`, and retiring moves the fileset out from under them. **Putting
  the new concept below the layer that would have had to know about it is what
  made it local**, and that was a consequence of the ruling, not of the build.

  **`ghost_boxes` is a separate list on `FrameView`, not a flag on `boxes`.**
  The index carried by `box_grabbed`/`box_clicked`/`box_moved` is a *position
  in `boxes`* that the tab uses to index `self.replicates` — the contract
  `_redraw_boxes` documents. A non-interactive entry mixed in would shift every
  later index and silently move the wrong replicate. Keeping them apart makes
  "not selectable" structural: `_box_at` cannot see the ghosts at all.

  **The in-memory hazard this section predicted was REAL, reachable, and is
  fixed.** `LiveScalogramSurface.closeEvent` flushes its track on purpose (a
  replicate edit rebuilds the surface, and that flush is how an accumulated
  whole-video pass survives) — so after a retire it wrote old-rectangle band
  power straight back into the home root the retire had just emptied, **undoing
  the retire one tab switch later**. `TrackStamp` would have made the next load
  refuse it, so it showed gray rather than lying; detectable-but-wrong was not
  good enough. Now `AppState.replicate_retired(int)` → `_on_replicate_retired`
  → `discard_replicate_track`, which drops that replicate's track *without*
  flushing and re-reads the root. **The signal is per-replicate, not
  `rois_changed`**: one box moving must not throw away a neighbour's
  accumulated pass. The restore path emits it too — a swap retires the current
  fileset on the way past, so the staleness is identical.
  *`_sync_track` was the wrong reload route* — it reloads only as a side effect
  of pushing a stamp and returns early when there is no stamp yet, leaving
  `_track` pointing at the object just dropped. `_activate_region` directly,
  with `_active_region` cleared first (which also skips its outgoing flush —
  the two needs want the same assignment).

  **The generation counter is derived, and `_rebuild_rois` is where the listing
  refreshes.** Not `_redraw_boxes`, which runs on every selection change and
  would put a directory listing per replicate in a paint-adjacent path;
  `_rebuild_rois` is the one funnel every mutation of the box list already
  passes through.

  *Found in review, both in the ghost paint:* `QColor.hue()` returns **-1 for
  an achromatic colour** and a box sidecar can carry `#ffffff`, so the HSV
  desaturation was replaced with a blend toward mid-gray; and the dashed pen
  was being reused for the label text, whose glyph outlines dash too.

  Suite 840 green. New: `RetiredGeometryTest` (10) in `test_replicate_tab.py`,
  `GenerationTest` (11) in `test_replicate_home.py`, `RetiredTrackDiscardTests`
  (4) in `test_live_stream.py`.

  **The original statement, kept because the reasoning still binds:**
  Slice 3's spec said "marks survive a box move (ruled, and already reflected in
  `_confirm_move`'s text)". **The user reversed this on 2026-07-21:** the marks
  survive *the deletion*, not *the move* — they stay valid **for the geometry
  they were labelled against**, and nothing may let the user believe what they
  are now looking at is current. `_confirm_move`'s docstring and `_MOVE_ANYWAY`
  both assert the old ruling and must be rewritten with it.

  **The reason is stronger than "computed under old settings", and it is the
  user's:** after a move there is no reason to believe those marks are remotely
  detectable under the new rectangle — *or that they were ever centred on the
  replicate the box now names*. A moved box can land on a different animal, or
  on nothing. So carrying marks forward as current is not a staleness problem
  (which the gray-stamp machinery already handles); it is **attributing one
  animal's labelled behaviour to another**, the same failure class as the track
  collision slice 2 fixed, and the reason a gray-out is not sufficient here.
  This is also why the old `_confirm_move` argument inverts: it reasoned that
  deleting "would take a hand-curated marks corpus with it on a 2% nudge", and
  that is still true — but the answer is to RETIRE the corpus with its
  rectangle, not to keep showing it against a rectangle it never described.

  What the user asked for, in their terms: old boxes stay **recoverable**, **sit
  in place** (still drawn, at their old rectangle), **labelled distinctly**
  (`OLD_<id>` or similar), and a move is **roll-back-able — bringing the
  detections back with it**. Explicitly: *"I don't want it to be a surprise to
  the user."*

  **The design tension this has to resolve, and it is the whole slice.** Batch
  S ruling 3 makes `next_id` monotonic so a retired id is never reissued, and
  ruling 2 refuses to delete a home so a 2% nudge cannot take a corpus with it.
  Retiring the *id* on every move would satisfy "old work stays put" but
  re-creates exactly the orphaning ruling 2 forbids. So the generation must sit
  **inside** the home, not beside it — the id keeps meaning "this animal", and a
  generation means "this rectangle". Sketch, not yet ruled:

      GX010047_rep01/
        GX010047.track.npz        <- current geometry
        GX010047.marks.json
        GX010047.tuning.json
        old_002/                  <- retired generation, with its own frac
          …

  **RULED 2026-07-21, all three questions answered.** The rulings shrink the
  slice rather than growing it — see the locality note at the end.

  - **The retired rectangle lives IN THE HOME and is authoritative there.**
    `old_NNN/geometry.json` carries the frac beside the marks and track it
    describes. Slice 3 already ruled the box `frac` joins marks `provenance`,
    so a `retired: [...]` list in `rois.json` would be a second copy of the same
    fact — the state duplication T11, T17 and T34 each had to delete. The
    Replicates tab discovers generations by listing the home.
    **Consequence: `rois.json`'s schema does not change at all.**
  - **`OLD_` boxes are drawn on the Replicates tab ONLY**, dashed and
    desaturated, and are not selectable. The tab is the layout editor, so
    "this used to be here" belongs there. The explorers route a detection pass
    into whatever region is *active*; a selectable retired box on those surfaces
    could aim a fresh pass at a retired generation, which re-creates the
    cross-attribution failure slice 2 fixed, one level down.
  - **Rollback is per-replicate, to ANY generation, and it is a swap.**
    Restoring `old_002` moves its contents up to the home root and retires the
    displaced current fileset as a new generation, so restore is a move in the
    other direction rather than a second mechanism. Disk-backed, therefore it
    survives a reload — a session-scoped undo would show a box labelled
    recoverable with no way left to recover it.

  **The generation counter is DERIVED, not persisted, and that is the one place
  this departs from `next_id`.** Next generation is `max(existing old_NNN) + 1`.
  `next_id` had to be persisted because a deleted box leaves *no trace* in
  `rois.json`, so the counter was the only memory of it; a retired generation
  leaves a **directory**, which is itself the trace. The filesystem enforces
  monotonicity here for free, and the reissue hazard is the same failure class
  (a reissued generation would adopt a dead rectangle's marks) — so it must be
  derived from the listing, never from a count.

  **Current geometry is UNNUMBERED.** It sits at the home root and takes a
  number only when it is retired; `rep1 (gen 3)` is not a thing the layout can
  say. That is what keeps the stores untouched.

  *Locality, ninth firing — inverted for once.* The spec predicted
  `rois.json`'s schema, `core/replicate_home.py`, `gui/tab2_replicates.py` and
  both stores. The rulings remove three of those: the schema is unchanged
  (home-authoritative), and `marks_store` / `tuning_store` / `track_store` all
  write to `home_path(...)` = the home ROOT, so retiring is a move of the root
  fileset *underneath* them and they never learn a generation exists. The
  hazard's real form here is the other direction: the surface holds
  `_tracks[region]` **in memory** across a retire on another tab, and flushing
  that on handover would write old-rectangle data into the new generation's
  root. `TrackStamp` makes it come back gray rather than pass as current, so it
  is detected, not silent — but check it before assuming two files.
- ~~**Slice 5 · routing + cost model.**~~ **LANDED 2026-07-21.** Full write-up
  in `FINDINGS.md` §25. Built as specced, plus three things the spec did not
  reach:

  **`core/source_route.py` is the one place that asks.** `resolve_source` is
  derived per call, never recorded. Three outcomes: no manifest → source (with
  a `reason`); a manifest that is **not this video's** → also source, because
  for this video none exists; this video's manifest failing `verify_manifest` →
  **raise** (ruled by the user). The middle case is the one `verify_manifest`
  provably cannot catch — it validates against the source *the manifest names*
  — and `cli/pretranscode` and `core/framecount` had each rediscovered it
  separately, so `framecount.manifest_describes` was made public and is now the
  single implementation.

  **Found in review, and it is §3 trap 2 by a new road:** the Batch Q streaming
  path builds its own meta via `synth_live_meta` and so **never inherited**
  `live_channel_source`'s substitution of the manifest's *rational* fps.
  `ClipAtlasSource` seeks by dividing a frame index by that rate, so a rounded
  24.0 for 24000/1001 lands 3 frames early by frame 11000 — the right window
  length from the wrong place, silently. Harmless until something streamed from
  clips; it would have gone live with the checkbox. The route is now resolved
  **before** the plan, because on a clip route it supplies the plan's rate.

  **The GUI is OPT-IN and the checkbox is not persisted** (`Use ROI clips`,
  greyed until a manifest exists). A restored downsample is the same
  measurement at a remembered setting; a restored *source* is a different set
  of pixels re-armed silently in a later session — §10. For the same reason a
  stale manifest **refuses the pass** rather than falling back: the tick is a
  claim about which pixels the numbers came from.

  **Cost model:** `PassSample` gained `source_kind` (`clips:<provenance_key>`,
  not a bare `"clips"` — two qualities are two floors) *and* `channels`, and
  `CostModel.fit` raises on a mix. `channels` was already recorded on
  `ScalePass` for this exact reason and was being **dropped** on the way to
  `PassSample`, so guarding only the axis the spec named would have shipped a
  guard that refuses one contamination and permits its neighbour. The GUI's
  `_cost_samples` key had to grow both axes too, or a re-sweep after a cut
  overwrites the source regime entry by entry.

  Suite 817 green. New: `tests/test_source_route.py` (12), plus cost-model,
  shard and surface tests.

**Locality warning (the hazard has now fired six times).** Slice 2 was specced
as `track_store.py` plus a call site; it needed a real restructure of
`live_scalogram_surface.py` because `self._track` was a single attribute read
from ~25 places.

**Slice 3 was the sixth, and it fired in a new way worth naming: the spec listed
DATA files, not code files.** "`tuning.json` and `marks.json` each mix per-video
and per-replicate content" reads as a schema edit in two stores, and the two
stores were indeed the easy half. The work was in `live_scalogram_surface.py`,
because a schema split is only inert until you ask *who holds the value between
writes* — `_saved_view` is an in-memory cache that outlives a region switch, so
separating the files without separating the handover would have kept the bug and
merely given it two filenames. **A spec that names a file FORMAT is making the
same claim as one that names a file, and it has now been wrong the same way:
ask what holds the state, not where the state is written.**

Treat slice 6's file list as a guess. Its own section already flags it as
spanning four files, which is a prediction, not a reprieve. Slice 4 fired the
hazard a seventh time, in the smallest possible way — see its entry.

**Slice 6 was the ninth and the first to fire INWARD.** The spec predicted four
files and took two, because the rulings put the new concept *below* the layer
that would have had to know about it (see its entry). The hazard reappeared in
the other direction instead: the file the spec did not name,
`live_scalogram_surface.py`, held the replicate's track **in memory** and
flushed it on close, undoing the retire. **The generalization: when a ruling
makes a change local on disk, ask what holds the same state in RAM** — a
storage layer can be moved out from under its writers, but not out from under a
cache that will write again.

**Slice 5 was the eighth, and it names the rule for a whole class of specs.**
It named two things (`resolve_source(video)` and `core/cost_model.py:89`) and
took nine files. The generalization: the spec named a function to ADD and a
field to ADD, and **adding a field to a value object is never local** — every
producer of that object and every consumer that groups or keys by it moves with
it. `PassSample` gained two fields; that reached `scale_sweep` (the producer),
`cost_model` (the guard), and the surface's sample key, grouping and tests.
