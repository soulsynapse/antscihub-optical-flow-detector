# Findings — measurements and conclusions that must not be re-derived

Companion to `todo.md`, which holds the plan. This file holds the expensive part:
measurements that cost real time to obtain, and several conclusions that are
**counter-intuitive enough that they were arrived at wrongly at least once**.

Read the relevant section before optimizing anything, re-running a sweep, or
"tidying" code flagged here as load-bearing. Do not re-derive these by staring at
a span table — two of them explicitly refute what the span table appears to say.

Contents:
1. [Throughput: decode-bound vs math-bound](#1-throughput)
2. [Landed optimizations](#2-landed-optimizations)
3. [ROI decode, and three load-bearing traps](#3-roi-decode)
4. [The target-width sweep, and its withdrawn half](#4-the-target-width-sweep)
5. [Scale vs block size: storage and compute](#5-scale-vs-block-size)
6. [The cost model and the knee](#6-the-cost-model-and-the-knee)
7. [Why the empirical detection panel was removed](#7-why-the-detection-panel-was-removed)
8. [Calibration](#8-calibration)
9. [GUI bugs worth remembering](#9-gui-bugs-worth-remembering)

---

## 1. Throughput

**Decode-bound at low scale, math-bound at scale 1.0.** H.264 decode is
whole-frame and crop is a post-decode filter, so the decode floor is ~3.8 s per
479 frames (5312x2988) **regardless of crop or scale**:

| | extract | ~decode | ~math |
|---|---|---|---|
| scale 0.245 | 4.69 s | ~3.8 s | ~0.9 s (**19%**) |
| scale 1.0 | 19.41 s | ~3.8 s | ~15.6 s (**80%**) |

**At the old default scale, optimizing the math was pointless.** A full extraction
pass was 4.39 s against **3.82 s for pure ffmpeg decode with no filters and output
discarded** — everything the tool does (tensor solve, block reduce, wavelets,
detection) added **13%**. A span table reading "`block_reduce` 31%" was an artifact
of the loop waiting on frames; optimizing it would have gained ~nothing.

**Under the scale-1.0 default this is reversed and compute levers matter again.**
Both halves are easy to get wrong from the span table alone.

**Multi-process does not scale**, because one ffmpeg already spreads decode across
the whole box (`-threads 1` is 40.3 s vs 3.8 s auto):

| workers | CPU decode | GPU decode (`-hwaccel cuda`) |
|---|---|---|
| 1 | 4.6x realtime | 6.8x |
| 4 | 7.8x | 10.4x |
| 8 | 7.6x | **10.8x (ceiling)** |

Aggregate throughput is flat past ~2-4 workers either way. The RTX 4060 helps ~38%
but has one NVDEC engine and hits its own ceiling. At 10.8x realtime, 3000 h is
**~11.6 days on this machine** — single-box tuning is not the path. GPU decode is a
flag (`-hwaccel cuda` before `-i`), not a project, if it is ever wanted.

**Caveat that matters for Batch J:** these are decode numbers at the *old* scale.
At scale 1.0 the work is math-bound and **parallelizes across processes where
decode did not**, so per-node throughput must be re-measured before quoting any
node count. Do not carry the old "3000 h / 10x / 50 nodes ≈ 6 h" forward.

## 2. Landed optimizations

- **Prefetch (producer-thread decode).** 15.14 s → **10.21 s** (~33%) on 479
  frames / 7 replicates / 5312x2988. Hides ~5.6 s of the 9.6 s decode — exactly
  the consumer work available to overlap it with. Consumer spans read slightly
  *higher* with prefetch on (GIL contention); net is still a clear win. Neutral on
  an already-cropped clip (9.15 s → 9.23 s), with decode fully hidden.
  `close()` joins the producer with **no timeout, on purpose**: the caller releases
  the VideoCapture the instant close returns, so a timeout could only hand back a
  capture still being read from.
- **Block reduce.** `reduce_scalar_to_blocks(..., "mean")` reshapes to
  (ny, block, nx, block) and reduces axes 1 and 3 — a view, no copy, even for
  ragged tiles — instead of padding the whole plane with NaN, transposing, and
  running `nanmean` (three full-size temporaries). 730x1300 b=16:
  **3.46 ms → 1.01 ms (3.4x)**. NaN in the field falls back to the old nanmean
  route, so masked-input semantics are preserved.
- **ROI decode on the live path.** 10.41 s → **4.10 s (2.54x)**; decode+preprocess
  went 63% → 27%. The pass went from 1.7x *slower* than realtime to ~5x faster.
  No regression on an already-cropped clip (8.43 s → 7.78 s).

## 3. ROI decode

**It is also *more accurate* than the full-frame path it replaced, which was not
expected.** Gate disagreement vs a float32 reference (gray in float, resize in
float, no uint8 round trip), averaged over 7 regions x 3 thresholds:

| comparison | disagreement |
|---|---|
| roi-vs-full | 8.07% |
| full-vs-reference | 7.36% |
| **roi-vs-reference** | **2.37%** |

So the ~8% the two decoders disagree by is mostly the *existing* full-frame path's
own quantization error, and the ROI path is ~3x closer to ground truth. Two
reasons: output is `gray16le`, so area-averaging a 4x4 patch of 8-bit samples is
not re-rounded to 8 bits (0.528 → 0.364 grey-levels RMS); and at scale 1.0 FFmpeg
reads the luma plane directly where OpenCV round-trips yuv420p → BGR → gray
(0.69 RMS + 1.28 bias).

### Three traps, all load-bearing

1. **`format=gray16le` must come AFTER `scale` in the filter graph.** Before it,
   swscale converts first and scales a gray16 input, measuring **3.80 RMS** — far
   worse than plain 8-bit. That ordering is easy to "tidy" wrong.
2. **Seek must read the container's own frame rate**, never a caller-supplied fps.
   The seek divided the frame index by a passed-in fps, so a rounded value shifted
   the window: 24.0 for 24000/1001 lands **3 frames early by frame 11000**,
   silently. Verified frame-accurate at 0/1/7/100/373/1000/5000/11000 on both a
   23.976 and a 59.94 fps clip after the fix.
3. **The flow cache key does not include the decoder or its bit depth**, so
   **caches built before this change must be rebuilt** before comparing detection
   results across it. The numbers moved for a good reason but the key does not
   know that.

**Still open, deliberately:** only the *first* frame is probed, so an FFmpeg
failure at frame 9000 of a whole-video commit kills the pass rather than falling
back — a mid-stream retry needs `prev_g` and the partial output arrays reset.

## 4. The target-width sweep

`scripts/sweep_target_width.py`, `GX010047c2` 5312x2988, 7 replicates, 479 frames,
channel `tensor_speed`. Block size tracks the target width so the **block grid is
identical (41x5) at every rung** — localization is held fixed and the only
variable is how many working pixels average into each block.

| width | block | scale | rel.px | **extract** |
|---|---|---|---|---|
| 5200 | 64 | 0.979 | 1.00 | **19.1 s** |
| 2600 | 32 | 0.490 | 0.25 | **6.8 s** |
| 1300 (old default) | 16 | 0.245 | 0.06 | **4.6 s** |
| 650 | 8 | 0.122 | 0.02 | **4.1 s** |

**1. Downsampling below the old default buys almost nothing — the plan's cost
model was wrong.** The plan assumed "halving the per-replicate target is 4x rather
than 16x", i.e. that width is a strong downward compute lever. Measured: 1300 →
650 is a **4x pixel cut for ~13% wall time**. The pass is pinned near the decode
floor at these scales.

**2. Upward the pixel cost IS paid in full**, but off a decode-dominated baseline:
5200 is 16.7x the pixels of 1300 for **~4.1x** the wall time. The sub-linear total
is Amdahl, not a shadow absorbing the work — the math itself scaled ~16x
(~0.9 s → ~15.6 s). So abandoning downsampling costs **4.1x, not the 16x** that
once shelved Batch K.

**3. `DEFAULT_TARGET_WIDTH` was in the wrong units, so no sweep of it could have a
transferable answer.** `1300 / source_width` makes the physical working scale a
function of how the camera was framed — the same animal and behaviour at a
different sensor resolution or crop resolves to a different scale. Any constant
tuned against one clip is overfitted by construction. What transfers is
**organism-relative**: how many working pixels the smallest moving structure of
interest spans.

### The sensitivity half is WITHDRAWN

An earlier reading took the `sig_corr` column as sensitivity loss and concluded
1300 was under-resolved. That does not hold:

- `sig_corr` is the region-**MEAN** band power over every block in a replicate,
  including empty background. On sparse events it largely measures whether the
  noise floors agree. It is a *stability* metric, **not** a detection-quality one.
- Scoring detection quality needs labelled events. `marks.json` holds **one** span,
  so that evaluation cannot be run at all yet.
- `GX010047c2` is adversarially hard by design, so degradation measured on it does
  not generalize in either direction.

Timings vary run-to-run by ~25% (2600 measured both 5.3 s and 6.8 s) — the ratios
are sound, the absolute seconds are not.

**Still open, blocked on labels rather than measurement:** how many
pixels-per-animal the detector actually needs. That is a question about the
organism, answerable only against labelled events across more than one species/fps.

## 5. Scale vs block size

479 frames, 7 replicates, extrapolated to 24 fps:

| | extract | grid | storage/h | 3000 h |
|---|---|---|---|---|
| old: scale 0.245, block 16 | 4.69 s | 41x5 = 205 | 289 MB | ~0.9 TB |
| scale 1.0, block 16 | 22.69 s | 139x19 = 2641 | 3.6 GB | **~11 TB** |
| **scale 1.0, block 64 (current default)** | **19.41 s** | 41x5 = 205 | 289 MB | **~0.9 TB** |
| scale 0.245, block 64 | 4.06 s | 20x2 = 40 | 56 MB | ~0.2 TB |

**The two levers are NOT interchangeable — measured, not asserted:**

| lever | compute | storage |
|---|---|---|
| `block_size` 16 → 64 | **-13%** | -5x to -13x |
| `downsample` 1.0 → 0.245 | **-4.8x** | -13x |

`block_size` is close to a pure *storage* lever: 13x less data for 13% less
compute, because the per-pixel stages run at working resolution regardless of block
size. A compute-limited user gets almost nothing from it; only `downsample` touches
that. Hence the UI must expose both separately.

**Block tracks scale by default, decided on detection grounds, not aesthetics.**
A two-lever table makes them sound independent, which reads like an argument for a
static block default. It is the opposite: blocks are in *working* pixels, so a
static block means the scale knob silently moves the grid too. Detection reads
per-block band power and gates on per-region block *counts*, so a moving grid
changes detection output for two reasons at once and neither is attributable.
Tracking is what makes scale a single-variable experiment.

Verified end to end (`GX010047c2`, 7 replicates, 120 frames): grid holds at 41x5
across a scale sweep (1.0 / 0.5 / 0.25 → 4.20 s / 2.15 s / 1.51 s) while compute
moves ~3.5x; an explicit `block_size=16` override reaches the finer 146x19 grid on
demand at 6.29 s.

## 6. The cost model and the knee

Cost is **`t(s) = F + M·s²`** — decode is whole-frame so `F` is fixed, everything
downstream is per-pixel. Measured against `GX010047c2` (7 replicates, 120 frames),
the quadratic reproduces real wall time to **within ±2% across a 4x scale range**
(1.0 → 4.30 s, 0.5 → 2.17 s, 0.25 → 1.56 s). That is what justifies the form.

The user's real question is an elasticity ("does 1% less resolution buy 1% less
time?"), and `E(s) = 2Ms²/(F+Ms²) = 1` exactly when `M·s² = F`, i.e.
**`s* = √(F/M)`: the knee is where per-pixel math cost equals the decode floor.**
Parameter-free, and it independently reproduces the ~0.50 an approved mockup had
marked by eye when fed the measured numbers (3.8 s floor, 15.6 s math → 0.494).

### A single-pass fit is unusable and must stay withheld

Prefetch hides decode so thoroughly that the `decode` span read **0.01 s of a
4.30 s pass (0%)**, so a one-pass span split sees *no floor*:

| | floor | per-pixel | knee |
|---|---|---|---|
| one pass (provisional) | 0.04 ms/f | 35.77 ms/f | 0.05 |
| fitted (3 scales) | 11.79 ms/f | 24.09 ms/f | **0.70** |

Wrong by ~300x on the floor, ~5.7x optimistic on cost at scale 0.25, and wrong in
the direction that **makes aggressive downsampling look free** — precisely the
choice the window exists to stop being made carelessly. A plot that wrong is worse
than no plot, because it looks measured.

### Two more traps in the same machinery

**The live surface extracts at `block_size=1`** whenever the per-pixel cache fits,
and block=1 does no reduction: `block_reduce` measured **62%** of such a pass
against ~15% at block 64. Measured directly, the live block=1 pass costs **11.0 s**
against **4.2 s** for the same scale at block 64 — the frontier was overstating a
batch run by ~2.6x. Hence the sweep runs a pass per scale at the *production* block.

**Group cost samples by regime, not by raw block.** With `auto` the block tracks
the scale by design, so grouping on the resolved block files a five-scale sweep as
five one-sample groups and leaves the model provisional forever — in exactly the
workflow the sweep exists to serve. The regime is recorded from the config that ran
(`FlowConfig.block_size`: `None` tracked, a number pinned), **never inferred** from
the resolved block: for any pinned block there is a scale where tracking picks the
same number (64 at 1.0, 32 at 0.5), and inferring misfiles that pass.

Related: whether a live pass runs at block=1 depends on a per-pixel budget that
scales with s², so **dragging Downsample to compare scales is itself what splits
samples across regimes.** Pick the regime with the most distinct scales, ties to
the newest.

### Storage does NOT fall with downsampling, and is not flat either

`round(64*s)` and `ceil(dim*s/block)` round independently, so the cell count
jitters non-monotonically:

- 7-replicate 5312x2988: exactly 205 cells / 26.4 GB per 100 h from 1.0 down to
  0.15, then **240 cells (30.9 GB) at 0.10**. Atlas packing absorbs the rounding.
- single full-frame replicate: 539 GB at 1.0/0.75/0.50, **564 GB** at 0.35,
  498 GB at 0.15, **615 GB** at 0.10 — a 19% spread.

So the claim to make is "does not reliably fall, and sometimes rises", not "flat".
**A user downsampling to save disk can pay MORE.**

**Storage must be counted over the packed atlas**, not the sum of per-tile boxes —
205 cells against 175 on this clip, a ~17% under-report.

## 7. Why the detection panel was removed

An empirical panel ran the tuned detector at each candidate scale and reported
events kept/lost. It reads as the sensitivity evidence the governing principle asks
for, and it is not. Value and count bands are **absolute**, and downsampling
averages pixels before differencing, so per-block band power falls with scale and a
fixed threshold catches less whether or not the behaviour is still resolved.

Measured with a discriminating tuning (value band at the p95 of reference band
power):

| scale | 1.00 | 0.75 | 0.50 | 0.35 | 0.25 |
|---|---|---|---|---|---|
| frames | 60 | 55 | 52 | 31 | 13 |

with **zero added frames at any scale**. That one-directional signature is what
threshold drift looks like, not lost structure. The table could not distinguish
"0.25 is too coarse to resolve the behaviour" from "your threshold needs re-tuning
at 0.25" — precisely the withdrawn `sig_corr` failure: an authoritative-looking
number that does not mean what it appears to.

**Do not rebuild it in that form.** Detectability is decided in the live surface
and the whole-video pass. If revisited, the fix is a **re-tune-at-scale row** that
holds a tuning-invariant fixed (equal gate fraction, or the band at the same
percentile of that scale's own band power) and reports **both** rows, separating
the two causes. Never fold it into a single number.

**What replaced it: the render strip.** The actual working image the tensor solve
receives, plus the frame-difference field the `change` channel is built from, at
each scale. Tiles share one display size and one brightness range, normalised
against the **largest** scale — per-tile auto-contrast would make every scale look
equally vivid and hide the amplitude loss completely, the same class of error as
`sig_corr`. Drawn with NEAREST upscaling so a coarser scale reads as blockier
rather than smaller. It shows the *mechanism* rather than a verdict; a picture does
not claim to have counted anything. Nothing there returns a scalar, deliberately.

## 8. Calibration

**The fiducial cancels out of the resolution question, exactly.** Working px per
body length is `pixels_per_mm * body_length_mm * scale`; substituting the two
measurements collapses it to **`body_line_px * scale`**. So an animal line alone
answers it; a ruler is needed only to store the result in millimetres for exports
and cross-clip comparison. Pinned as an invariant in `tests/test_calibration.py`
across three rulers and three scales.

**Line placement error is `sqrt(2)/zoom` source px.** A 10 px animal measured at
0.25x is **±57%**. Reported per line and warned past 10%, which is where 0.50 and
0.35 stop being distinguishable in the units the frontier is read in. This is a
propagated *measurement* error, not a judgement about the footage.

### Two ownership traps that each bit once

**`build_layout` sorts replicates by id; a replicate list is in draw order.**
Anything indexing `layout.tiles` and a replicate list with the *same* index must
sort first — use `core.replicates.in_tile_order`. This bit once: the render strip
showed one replicate's picture beside another's px-per-body-length, and would have
written a calibration onto the wrong animal.

**`AppState.set_replicate_specs` *copies* the replicate dicts.** A widget built
earlier holds references that go stale on any edit, and a geometry-keyed rebuild
does not catch it because calibration is not geometry. Hence
`LiveScalogramSurface.refresh_replicate_metadata`, which re-reads all non-geometry
fields and refuses to copy geometry across. This bit once: the downsample window
read "uncalibrated" for a calibrated replicate for a whole session.

## 9. GUI bugs worth remembering

- **`closeEvent` not stopping the debounce timers** let a knob edited just before a
  close fire `extract()` into a dying widget. That was the pre-existing ~1-in-3 Qt
  crash (exit 9) in the full suite.
- **Batch K's reinterpretation of `None`** broke three things in the shelved
  `tab1_flow.py`, including a downsample spin whose
  `p.downsample if p.downsample else 0.05` mapped a default config to **0.05, a
  silent 20x downsample** — precisely the failure Batch K exists to prevent,
  arriving through the back door. The shelved-tab rule was overridden deliberately
  for that one.
- **The default state produced a false pass.** A freshly built explorer has both
  bands wide open, so the gate is on for every frame; every scale then reported
  "loses nothing" — five rows of that conclusion produced by a detector
  discriminating nothing, in the window whose job is to prevent it. Guard
  comparisons against an all-on or all-off reference gate.
- **A stopped sweep left its remaining rows reading "waiting" forever**, so a
  half-populated table looked live.
- **A reference row projected "—" forever** because it lands while only one scale
  has been timed and the model is still provisional. Re-render rows against the
  final model when a sweep ends.

---

## Method note

Nearly every entry above was found by **driving the thing**, not by tests: the
false pass, the wrong-replicate index, the stale surface, the 300x floor error, the
17% storage under-report, the unreachable provisional branch. Tests then pinned
them. The order matters — several of these produce *plausible, authoritative-looking
output* rather than a crash, and a test written from the same wrong mental model
would have passed.

## 10. ROI pre-transcode

Measured on `GX010047c2_02_17_26.MP4` (5312x2988 HEVC, 24000/1001, 0.157 bits/px,
3.55 GB), 6 replicates covering 8.27% of frame area.

**The decode win, which is the whole point.** 300 frames of one replicate:

| path | time | speedup |
|---|---|---|
| source HEVC + crop (today) | 4.03 s | 1.0x |
| ffv1 lossless clip | 0.16 s | **25.4x** |
| crf18 clip | 0.14 s | 28.6x |

**Codec choice barely affects decode speed** — 25.4x lossless vs 28.6x lossy, 12%
apart. Decode cost tracks *pixels*, not bytes. There is therefore no
speed-vs-fidelity trade to agonize over; the trade is purely storage.

### Storage vs fidelity, and why lossless is not the default

Luma error against a bit-exact reference, and full-clip size for all 6 replicates:

| quality | grey-level RMS | clips | total disk |
|---|---|---|---|
| lossless (ffv1) | 0.000 | 131% of source | 2.31x |
| high (crf 12) | 1.041 | 24% | **1.24x** ← default |
| standard (crf 18) | 1.540 | 11% | 1.11x |
| x265 lossless | 0.000 | **130%** | — |
| ffvhuff | 0.000 | **199%** | — |

**Bit-exact lossless saves essentially no storage**, and two lossless codecs come
out *larger than the 5.3K source*. Lossless must encode the source's inherited
HEVC artifacts and sensor noise verbatim, and noise does not compress.

The default is `high`, deliberately, against the initial instinct to force
lossless. The sources are *already* a lossy re-encode (`stab_` = a stabilization
generation removed from the sensor), so byte-exact preservation of an
already-degraded intermediate buys less than it appears to; and a result that
flips between crf 12 and lossless was fragile regardless, which is better
surfaced than hidden. For reference, `FINDINGS.md` §3's ROI path achieves 0.364
RMS, so crf 12 at 1.041 is ~2.9x that — real, and the reason quality is recorded
rather than assumed.

**What survives that argument.** The `change` channel is `<I_t^2>`, squared frame
differencing, and it is the detection default. Lossy *inter-frame* coding
perturbs precisely the frame-to-frame quantity that channel measures, rather than
degrading generically. So quality is in `Manifest.provenance_key()` and clips cut
at different quality must never share a cache entry. Whether a given behaviour is
robust across these settings is now **measurable end to end** (extraction consumes
the manifest) — **measure it, per behaviour, rather than assuming either way.**

### "Lossless" clips are still not bit-exact against the live crop

Found while wiring extraction onto the manifest, and it invalidates the obvious
test. `lossless` names the *encode*, not the route. The live path converts the
source's yuv luma straight to `gray16le` and keeps the sub-8-bit precision that
the limited→full range conversion produces — **81% of luma values are not
multiples of 257**, i.e. genuinely not representable in 8 bits. A clip stores
8-bit gray, so it rounds by up to **0.494 grey levels** before any codec runs.

Nothing is wrong here and nothing needs fixing: that 0.494 max (≈0.29 RMS for
uniform rounding) sits *below* §3's 0.364 RMS for the live ROI path itself, and
far below the 1.041 the `high` default already accepts. But it means an
extraction test must never assert clip == live, and the first one written did.

**How the error reaches the channels.** 160x120 `testsrc`, 2 replicates, 12
frames, RMS difference clip-vs-live normalized by each channel's own scale:

| quality | change | appearance | tensor_speed | intensity |
|---|---|---|---|---|
| lossless | 0.0049 | 0.0093 | 0.0111 | **0.00011** |
| high | 0.0164 | 0.0273 | 0.0441 | **0.00014** |
| standard | 0.0438 | 0.0770 | 0.0552 | **0.00033** |

Two things to read off it. Agreement degrades monotonically with quality, so the
knob means what it claims. And **`intensity` is 100-500x less affected than the
differencing channels** — it is a block mean, and averaging destroys exactly the
noise that squared frame differencing amplifies. That is the measured form of the
argument above for keeping `quality` in `provenance_key()`, which until now was
asserted from the structure of `<I_t^2>` rather than observed.

Caveat: `testsrc` at 160x120 is close to a worst case — high spatial frequency,
small blocks, nothing to average over. Real footage will differ, probably
favourably. These numbers order the presets; they are not an error budget.

### Lossless cost is distributed backwards from intuition

| replicate | bits/px |
|---|---|
| rep6-no-flying | **3.695** ← most expensive |
| rep2-backlit-flying-whole-time | 2.981 |
| rep1-mostly-flying-some-pause | **1.921** ← among the cheapest |

Motion is *cheaper*: motion blur is smooth and compresses well, while a still
frame is mostly per-pixel noise. **Do not extrapolate a corpus from one
replicate** — rep3 alone reads 1.602 bpp and projects 86% of source, a third
under the true 131%. That error was made once here.

### Traps found while building it

1. **Both ffmpeg pipes must be drained concurrently.** Reading stdout for
   `-progress` while deferring stderr until exit deadlocks as soon as ffmpeg
   emits ~64 KB of stderr. It would hang the multi-minute whole-source runs and
   never reproduce on a short test clip. A thread drains stderr.
2. **ffmpeg emits a progress block only ~every 0.5 s**, so any clip small enough
   for a fast test finishes inside one block and reports only the final frame. An
   end-to-end test of progress parsing therefore passes whether or not parsing
   works. The parser is unit-tested against canned lines instead. A first attempt
   at an end-to-end version was vacuous in exactly this way.
3. **`frame=` does NOT aggregate across outputs** — it matches source frames even
   with 6 outputs mapped. This was assumed to be a bug and "fixed" wrongly before
   being checked; it is not one.
4. Gray-only encoding saves just ~6% here, not the ~33% the chroma planes suggest:
   this backlit footage is nearly monochrome already. Still worth taking, since
   the pipeline reads only luma.
5. **A clip decoder must not fall back to whole-source decode on failure.** The
   fallback is right for the ROI path (`_open_roi`), where both routes read the
   same source pixels and only speed differs. It is wrong for clips: below
   lossless the pixels genuinely differ, and the caller has already folded the
   clips' `provenance_key` into what it caches — so a silent fallback would file
   source-decoded numbers under a clip-decoded identity, the exact confusion that
   key exists to prevent. `_open_clips` raises.
6. **Match clips to tiles by replicate id, not list position.** Both lists come
   out of `build_layout` over the same replicates and do align today, but a
   positional match would fail *silently* if that ever stopped holding, and the
   failure mode is one replicate's pixels being attributed to another's
   detections. An id lookup raises instead.
7. **`vstack` needs `shortest=1` when its inputs are separate files.** This is
   the worst bug found in the batch. At the default `shortest=0`, vstack does not
   end the stream when an input runs out — it *repeats that input's last frame*
   while the others advance. With N clips as N inputs, one clip cut short by a
   crash or a full disk therefore **freezes its replicate**: identical
   consecutive frames, so `change == 0`, read downstream as "measured, and
   nothing moved" — for that replicate alone, while every other replicate looks
   healthy, and with no symptom reported anywhere. Precisely the failure mode the
   governing principle forbids: deleted signal indistinguishable from "nothing
   happened". `shortest=1` ends the stream instead, and `_stream_channels` now
   trims the window to what actually decoded and flags `truncated`.
   `ReplicateVideoSource` needs no such flag — its inputs are one split of one
   decode and cannot differ in length. `verify_manifest` does not close this:
   it checks that a clip exists, not how long it is.
8. **Zero-padding a short decode was the same bug one level up.** Independently
   of vstack, `_stream_channels` left the tail of every channel at zero when the
   decode ended early and still reported the full window length. It now returns a
   short window. All three decode paths get this; clips just make it reachable.
9. Probing clip dimensions per pass is not free: opening a `VideoCapture` per
   clip measured **41.8 ms for 6 clips**, against a ~160 ms clip decode and 1.4 ms
   for the whole manifest verification — on a surface that starts a pass per knob
   edit. The manifest already records each clip's size, so `resolve_clip_paths`
   checks it for free and `ClipAtlasSource` takes `verify_sizes=False`.

### One measurement trap, recorded because it nearly caused a wrong fix

Investigating trap 7, a truncated clip showed "17 of 21 frames all-zero in rep0's
`change`" — which looks damning and nearly justified a manifest schema change to
record per-clip frame counts. **The control refuted it**: healthy clips show
18/20 and the *live source* shows 16/20 on the same box. The all-zero frames are
`testsrc` content (a mostly-static region at block granularity), not truncation.
Run the healthy-path control before attributing an absolute number to the change
under test — the real defect here was visible only in the *reported window
length* (40 → 20), never in the zero count.
