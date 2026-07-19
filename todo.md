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
| ~~**T17**~~ | ~~Whole-video processing resets the detection threshold bands when navigating.~~ **FIXED** — see Batch F. |
| ~~**T18**~~ | ~~"Process whole video" computes every channel regardless of the selected one.~~ **FIXED** — see Batch F. |
| **T20** | Drop the replicate dropdown — redundant with click navigation. |
| **T21** | Suspected runaway memory issue somewhere on the replicate tab. **Not reproduced by inspection** — `_refresh_list` was the obvious suspect and is clean (14x14 swatch pixmaps, `list.clear()` releases the items). Needs an actual measurement, not more code reading; do not guess at a fix. |
| **T22** | New **viewer tab** consuming a fully-processed video: detection/no-detection only, full-clip bar plus a ~1 min bar zoomed on the scrubber, and a hand-off back into preprocessing at the paused position *without* auto-triggering a pass. |
| **T23** | **ROI pre-transcode**: cut each source once into per-replicate clips + manifest, so later passes decode ~1/16 the pixels. The only lever that moves the decode floor. |
| **T24** | **Headless batch driver**: no-GUI entry point (video + replicate JSON → detection results) and file-level job partitioning, so runs fan out across nodes. **Entry point and partitioning both landed** (Batch J); only the N-worker throughput measurement is still open. |

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
| 11. Per-channel extraction | the measured 1.59x (not the ~4x this plan asserted), and why the placeholder is zero-length |
| 12. The headless path | per-video extraction, the two clocks, why truncation fails the job |
| 13. File-level partitioning | the four silent ways a shard examines less than it claims; resume by identity, not existence |

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

### Batch F — whole-video correctness (~~T17~~, T18) · `scalogram_explorer` + `tensor_channels`
T18 remains; **T17 is landed**. The file locality moved — neither fix was where
this batch originally guessed.

**T17 — LANDED.** The original hypothesis was wrong: `extract()` captures view
state unconditionally and `_focus_frame` goes through `extract()`, so the round
trip was not dropping the capture. The real cause was that `capture_view_state`
carried `channel`, `frame`, `sweep_win`, `centered` and `freq_band` — but **not
the two detection threshold bands**, which `detection_params()` reads off live
widgets. The state existed; it was simply never in the captured dict, so every
rebuild reverted both to defaults.

Both now travel, and the per-channel value bands travel **for every channel, not
just the selected one** — switching channels after a rebuild found the others
wiped by the same omission. Endpoints are carried **raw**: `None` means "never
placed" and `set_band_active` seeds it lazily, so collapsing it to ±inf on
capture would convert "unset" into an explicit unbounded threshold. The restore
sits *before* `apply_view_state`'s closing `_on_freq_band_committed()`, or the
recompute would run against the defaults it exists to replace. Round trip is
covered by `tests/test_view_state.py`, including an old state dict lacking the
new keys.

**T18 — LANDED.** `channels=` now threads through `extract_channels_live` and
`live_channel_source`; `_ProcessWorker` passes the single `channel_attr` it is
about to detect on. The live preview deliberately still computes all four, so a
channel toggle there stays instant instead of triggering a re-extract — the
waste this fixes was only ever on the commit pass.

**The plan's own estimate here was wrong and is corrected in `FINDINGS.md` §11.**
"Selecting `change` should skip nearly all the downstream math" overstated it:
`change` is `J[2]`, so it still needs `tensor_products` *and* `tensor_blur`, and
the blur is the single largest span in a pass (32%). Measured **1.59x**, not the
implied ~4x — and that on a decode-trivial synthetic, so real footage will show
less. `intensity` is the outlier at 6.0x, being the only channel that touches no
tensor at all.

Two things fell out, both recorded in §11: the unselected-channel placeholder is
**zero-length, not zero-filled** (a zero-filled array is the standing
false-negative shape, and a full-length one costs ~88 MB/channel/hour for data
nobody asked for), and `scale_sweep` now takes and records the same `channels`,
because a cost model fitted across a four-channel and a one-channel sample reads
the difference as scale.

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
**Still unenforced after J's first slice** — the headless summary *records*
`clip_provenance`, but writing a result file is not caching, so no caller has
appeared yet.

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

### Batch J — headless batch driver (T24) · new CLI + `core` only  ← **IN PROGRESS** (throughput measurement only)
No-GUI entry point plus file-level job partitioning. This is what gets linear
scaling, because each node brings its own decode capacity. Pairs with I (a batch
run should consume pre-transcoded ROI clips).

**Landed — the single-video entry point.** `core/batch.py` (the logic, no
argparse) + `cli/detect.py` (`python -m cli.detect VIDEO --params tuned.json`),
with `tests/test_batch_cli.py`. Consumes the video, a replicate sidecar, tuned
params as JSON, and optionally a Batch I manifest. Writes a summary JSON plus an
`.npz` of per-frame series; `--band-power` additionally retains the `(T,B)` cube,
which is what makes value-band and window re-tuning possible without a fresh
pass. Exit codes are a contract for a job runner: 0 / 1 untrusted / 2 usage.

Details and measurements in `FINDINGS.md` §12. Four things worth knowing here:

- **Extraction is per video, not per region** — the atlas already holds every
  replicate's blocks, so N regions cost one decode. The GUI's per-commit extract
  is right for one-region-at-a-time and wrong for a batch.
- **A truncated pass fails the job** rather than warning. Two arithmetic traps in
  that guard were found by review, both dangerous-direction: coverage ignoring
  `--start` failed every offset run *after* paying full extraction, and an
  unclamped over-long `--frames` would have failed every job that passed a
  generous count — which trains operators into habitual `--allow-truncated` and
  disarms the guard.
- **`freqs` comes from the extraction meta**, so clip-backed passes build the
  wavelet bank from the manifest's rational fps. The GUI worker still uses the
  decoder's float; they differ only for clip-backed passes.
- **The GUI-agreement test asserts arrays, not intervals.** A summary can agree
  while the series underneath has drifted.

**Landed — file-level partitioning.** `core/shard.py` + `cli/run.py`
(`python -m cli.run "footage/*.MP4" --params tuned.json --shard 3/8`), with
`tests/test_shard.py`. A shard is `sorted(list)[i::N]` — stateless, so a runner
launches N identical commands differing only in `--shard`, and a preempted shard
is relaunched rather than reconciled. Stride rather than contiguous chunks,
because file size correlates with position within a session. A failed video does
not stop its shard-mates; the shard exits 1 with the failures named in a report.
Finished videos are skipped on re-run, but only against a *matching identity*
(params, resolved scale, resolved block, frame window) — never mere existence.
Pre-transcoded clips (Batch I) are discovered per video from the manifest naming
convention, since a manifest belongs to one source and cannot be a single flag.

**Details in `FINDINGS.md` §13, and it is worth reading before touching this.**
Every defect review found had one shape — *a shard that exits 0 having examined
less footage than asked* — and none would have failed a test that only checked
the happy path. Colliding basenames across session directories (the normal
multi-session case) silently **skipped** a video rather than merely overwriting
it, because resume and naming are individually reasonable and interact badly. A
platform-dependent sort key (`normcase`) partitions differently on Windows and
Linux, so a mixed fan-out leaves footage examined by nobody. And a completed
`--frames` smoke test is indistinguishable from a completed full pass in the
resolved counts, so it was reused as one — `requested_n` and `start_frame` are
now recorded and are part of the resume identity.

**Still open in J:** the N-worker throughput measurement below. Single process on
one small-ROI replicate at scale 1.0 measured **~88 fps / ~1.47x realtime** — a
real number but *not* a basis for a node count, being one replicate on a 512x512
working ROI.

**Extension point, deliberately not built:** a job-spec runner (a JSON list of
per-job video/replicates/manifest/params/out). `--shard` over a file list covers
the uniform case, which is the one that exists today; the spec form is what
per-video *differing* params would need, and nothing needs that yet.

**No Qt on this path, enforced by a test** (`NoQtImportTest`): a transitive PyQt
import turns every headless job into a startup crash on a display-less node, and
the `core` → `gui` import chain makes that easy to break by accident.

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
