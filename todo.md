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
| **T10** | Plots are themselves the main source of lag at block=1 with weak downsampling — even 10 s can be a massive result. Worse once the full-video tab can scrub. |
| **T11** | If fixed-size stamp is on, it should be the stamp of the *currently selected* replicate. Drawing a new box should select it (and set the stamp). |
| **T12** | Replicate tab: clicking a replicate should highlight + zoom to it; dragging its box should reposition it; right-click should delete it (that tab only). |
| **T13** | Per-block band power by channel draws black blocks when unpopulated — collapse those to checkbox height (the plot name is already that tall). |
| **T14** | All plots should collapse/expand via `[+]`, collapsed by default, and a collapsed plot must not render (saves memory). |
| **T15** | Replace the positive-detection graph with green bands overlaid on the windowed #-blocks-in-band plot. |
| **T16** | On a detection, show a `DETECTED` badge (green bg, bold black) bottom-right of the viewer box — must survive the shift-held path. |
| **T17** | Whole-video processing resets the detection threshold bands when navigating. It should not. **CONFIRMED, cause found — see Batch F.** |
| **T18** | "Process whole video" computes every channel regardless of the selected one. **CONFIRMED — see Batch F.** |
| **T20** | Drop the replicate dropdown — redundant with click navigation. |
| **T21** | Suspected runaway memory issue somewhere on the replicate tab. **Not reproduced by inspection** — `_refresh_list` was the obvious suspect and is clean (14x14 swatch pixmaps, `list.clear()` releases the items). Needs an actual measurement, not more code reading; do not guess at a fix. |
| **T22** | New **viewer tab** consuming a fully-processed video: detection/no-detection only, full-clip bar plus a ~1 min bar zoomed on the scrubber, and a hand-off back into preprocessing at the paused position *without* auto-triggering a pass. |
| **T23** | **ROI pre-transcode**: cut each source once into per-replicate clips + manifest, so later passes decode ~1/16 the pixels. The only lever that moves the decode floor. |
| **T24** | **Headless batch driver**: no-GUI entry point (video + replicate JSON → detection results) and file-level job partitioning, so runs fan out across nodes. |

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

**Shelved (T8, T9).** `gui/tab1_flow.py` and `gui/tab3_behavior.py` are
unmaintained. Don't fix them if a change breaks them; just note it. Consider
renaming to `_shelved_*` once nothing imports them. (Overridden once, deliberately,
in Batch K — see `FINDINGS.md` §9.)

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

---

## 5. Remaining batches

### Batch D — plot cost + collapse (T13, T14, T10) · `scalogram_explorer` only
Generalize the existing expand/collapse into a `[+]` header on every plot,
collapsed by default; a collapsed plot skips its paint *and* drops its cached
array. Unpopulated per-channel heatmaps collapse to checkbox height. This is also
the fix for T10, so treat them as one batch. `_apply_selected_plot_ui` already
calls `dp.set_expanded(...)`, so a collapse mechanism exists and can be generalized.

### Batch E — detection readout (T15, T16) · `scalogram_explorer` + `video_panel`
Replace the separate positive-detection plot with green bands overlaid on the
windowed-#-blocks-in-band plot. Add a `DETECTED` badge (green bg, bold black) in
the viewer box's bottom-right that survives the shift-held path.

### Batch F — whole-video correctness (T17, T18) · `scalogram_explorer` + `tensor_channels`
Both bugs are now **confirmed by inspection**, and the file locality moved — neither
fix is where this batch originally guessed.

**T17 — the original hypothesis was wrong.** `extract()` captures view state
unconditionally (`live_scalogram_surface.py:567`) and `_focus_frame` goes through
`extract()`, so the round trip is *not* dropping the capture. The actual cause is
that `capture_view_state` (`scalogram_explorer.py:373`) carries `channel`, `frame`,
`sweep_win`, `centered` and `freq_band` — but **not `value_band` or `count_band`**,
the two detection threshold bands. `detection_params()` reads all three from live
widgets, so the state exists; it simply is not in the captured dict, so every
rebuild reverts those two to defaults. Fix is local: add both to `capture_view_state`
and restore them in `apply_view_state`. Note `apply_view_state` ends with
`_on_freq_band_committed()`, so restore the bands *before* that call or the
recompute runs against defaults.

**T18 — confirmed, and bigger than "computes change energy anyway".**
`extract_channels_live` takes no channel selector at all, so the whole commit pass
computes `flow_solve`, `appearance` and `texture` for every window regardless of
`channel_attr`; `live_channel_source` then materializes all of `LIVE_CHANNELS`
(`channel_source.py:137`). The dependency structure is what makes this worth doing:

| selected channel | needs |
|---|---|
| `intensity`, `change` | `tensor_products` only (~7% per `FINDINGS.md` §1) |
| `tensor_speed` | + `flow_solve` |
| `appearance` | + `flow_solve`, + residual |
| `texture` | its own spatial min-eigen |

So selecting `change` — the detection default — should skip nearly all the
downstream math. Thread a channel set through `extract_channels_live` and write a
defined placeholder for unselected channels rather than omitting keys, so the
explorer's channel checkboxes still know they exist but are unavailable
(`_channel_available` already gates on this).

### Batch G — replicate tab (T11, T12, T20, T21) · `tab2_replicates` + `video_panel`
Click-to-select + zoom, drag to reposition, right-click to delete (that tab only),
fixed stamp follows selection, new box selects itself, drop the redundant dropdown.
The memory leak (T21) is likely an overlay/pixmap not released per selection —
chase it last, with the rest already simplified.

### Batch H — viewer tab (T22) · new file, largest
New tab consuming a `DetectionResult`: full-clip detection bar plus a ~1 min zoomed
bar following the scrubber, and an "open here in preprocessing" handoff that seeks
the live surface without auto-triggering a pass. Depends on B (cancel) and F (state
retention). Do last.

### Batch I — ROI pre-transcode (T23) · new file + `channel_source`  ← **IN PROGRESS**
Cut each source once into per-replicate clips plus a manifest (geometry, scale,
source hash, fps at full precision), and teach extraction to consume the manifest.

**Landed:** `core/pretranscode.py` + `tests/test_pretranscode.py` — the cut, the
manifest, `provenance_key()`, and verification. Measured on `GX010047c2`:
**~25x faster decode**, 1/12.1 the pixels. See `FINDINGS.md` §10 for the codec
measurements and why the quality default is what it is.

**Landed — the wiring:** `video.ClipAtlasSource` (one ffmpeg process, N clip
inputs, same atlas contract as `ReplicateVideoSource` via the shared
`_AtlasStream` / `_tile_tail`), `extract_channels_live(clip_paths=...)`, and
`live_channel_source(manifest=, clip_dir=)`, which resolves paths by replicate
id, re-verifies geometry and source identity per pass, and puts
`clip_provenance` in the meta. `cfg.cache_key` takes a `provenance_key` —
**omitted from the blob when absent**, so every pre-clip cache keeps its key.
Tests in `tests/test_clip_extraction.py`.

Four things came out of it, all in `FINDINGS.md` §10. **Lossless clips are not
bit-exact against the live crop** (8-bit store vs the live path's gray16le,
≤0.494 grey levels — do not write a test asserting equality). A per-channel
sensitivity table that turns "quality belongs in the provenance key" from
asserted into measured. And two real bugs, both silent-false-negative shaped:
**`vstack` at its default `shortest=0` freezes a replicate** whose clip is short,
and `_stream_channels` **zero-padded a short decode** while reporting full
length. Both fixed; windows now trim and carry a `truncated` flag.

**Landed — the review fixes.** A `/code-review high` over the wiring slice found
three real defects, all silent-shaped, all fixed:

- **The cut had no `-fps_mode passthrough`.** FFmpeg's default output mode is
  `cfr`, which duplicates or drops frames to force a constant rate — breaking
  *clip frame N is source frame N*, the invariant every seek rests on. Same
  failure as `FINDINGS.md` §3 trap 2 by a different road, and invisible to a
  `testsrc` fixture because that source is already CFR. `PRETRANSCODE_VERSION`
  is now **2**, so old cuts are refused rather than silently mixed — a v1 clip
  is not guaranteed frame-aligned with its source.
- **Nothing recorded how long a clip was.** `verify_manifest` checked existence,
  so a clip truncated by a crash or a full disk was rediscovered at decode time,
  per pass, forever. `ClipEntry` now carries `frame_count` (checked once at the
  cut against the source's, which also catches a re-timed cut) and `size_bytes`
  (re-checked every pass — an `os.stat` costs microseconds against the 41.8 ms
  a `VideoCapture` re-probe of 6 clips measured). The decode-time trim stays as
  the second line of defence for the ROI/full-frame paths, which have no
  manifest at all.
- **`scale_sweep` priced a truncated pass as full-length**, dividing real wall
  time by the *requested* frame count. A 20-of-64 pass reported a third of its
  true per-frame cost — biased toward "downsampling is free", the one direction
  §6 says the cost model must never err in. Now uses the delivered count and
  carries `truncated`, which `usable` excludes from any fit.

**Still open:** nothing on the tensor path. `run_pipeline` accepts no manifest —
deliberately, since the flow cache is not the detection path on this branch. Wire
it there only if Batch J needs it.

**One unenforced obligation, carried to J.** `cfg.cache_key` accepts a
`provenance_key` third argument and **no caller passes it**, because nothing
caches a clip-derived result yet. The first code that does must thread
`meta["clip_provenance"]` into it; otherwise a source-derived and a clip-derived
result collide in one cache entry. Available guard, not an active one.

**Why this beats the live ROI crop that already exists.** H.264 decode is
whole-frame: the `crop` in `ReplicateVideoSource`'s filter graph runs *after*
decode, so the live path still pays full decode cost. Pre-transcoding makes the
*stored file* small, so later decodes genuinely decode fewer pixels. It is the only
thing that moves the ~3.8 s floor (`FINDINGS.md` §1).

**Crop is lossless; scale is lossy — do only the crop by default.** Cropping
discards pixels no replicate owns, so it cannot change any detection result and is
safe unconditionally. Rescaling would bake a sensitivity decision into artifacts
that cost a full re-decode to regenerate, which the governing principle forbids.
The manifest records the resolved per-replicate scale **and the rule that produced
it**, so clips transcoded under different settings are never silently compared.
For these ~297 px replicates the crop alone is ~1/16 the pixels off the decoder.

Reuses the `ReplicateVideoSource` filter-graph geometry. **Watch the three traps in
`FINDINGS.md` §3** — especially `format=gray16le` after `scale`, and fps at full
precision in the manifest.

### Batch J — headless batch driver (T24) · new CLI + `core` only
No-GUI entry point plus file-level job partitioning. This is what gets linear
scaling, because each node brings its own decode capacity. Pairs with I (a batch
run should consume pre-transcoded ROI clips).

**Do not carry the old "3000 h / 10x / 50 nodes ≈ 6 h" figure forward** — it used
the decode ceiling at the old default scale. At scale 1.0 a single process is
~1.03x realtime because the work is math-bound. Re-measure per-node throughput at
scale 1.0 with N workers before quoting any node count (`FINDINGS.md` §1).

Storage ~289 MB/h at block 64, rising to ~3.6 GB/h if block does not track scale.
Either way a multi-hour clip wants chunked writes rather than one in-memory pass.

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

### Batch N — the same decision tool for block size (stub)
A sibling pop-out for `block_size`, built on Batch M's components
(`core/cost_model.py`, `gui/cost_panels.py`, `core/scale_sweep.py`,
`core/scale_render.py`, `gui/calibration_dialog.py`) rather than a divergent second
dialog. Same discipline: explanatory prose on open, no fused quality score.

**The substantive difference, which sets what its "what you lose" panel shows:**
downsampling loses detail *within* a block (the per-pixel field feeding the tensor
solve is coarser); block size loses **spatial localization** (fewer, larger cells,
so clump area and where-in-the-arena resolution degrade) while the per-pixel math
is untouched. So N's evidence panel is about grid granularity and clump resolution,
not image sharpness. Per `FINDINGS.md` §5 it is the storage lever and does **not**
carry downsampling's "may decide what is detectable" warning in the same form.

Also inherit M's required prose pattern: state plainly both that the lever can be
what makes a project feasible, *and* that it is deliberately not assumed on the
user's behalf. One half without the other produces either avoidance or silent
degradation.

---

## 6. Order

**I → J**, then D/E/F/G/H as day-to-day use demands.

I is next: under the scale-1.0 default it is the *only* remaining decode lever and
it is lossless, so it costs nothing scientifically. Then J — the default got ~4x
more expensive by deliberate choice, so fan-out is what pays for the principle.

D/E (interactive polish) sit behind J unless day-to-day use becomes the near-term
milestone; they live in `gui/explorers/scalogram_explorer.py`, the widget inside
**tab 2 · Preprocessing (live)**, not the shelved flow-cache tab. **F is worth
folding into whichever lands first, since T18 is a real speed win.**

Explicitly deferred and not blocking: **Batch L**, the pixels-per-body-length
denomination, and the per-behaviour sensitivity study that would justify any
non-1.0 default. The latter two need labelled events; `marks.json` has one span.
