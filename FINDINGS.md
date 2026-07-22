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
10. [ROI pre-transcode](#10-roi-pre-transcode)
11. [Per-channel extraction (T18)](#11-per-channel-extraction-t18)
12. [The headless path (T24)](#12-the-headless-path-t24-batch-j-first-slice)
13. [File-level partitioning (T24)](#13-file-level-partitioning-t24-batch-j-second-slice)
14. [N-worker throughput, and the node count](#14-n-worker-throughput)
15. [The container frame count is not the decodable frame count](#15-the-container-frame-count-is-not-the-decodable-frame-count)
16. [Clip-backed throughput: the 25x decode win is ~1.06x end to end](#16-clip-backed-throughput-the-25x-decode-win-is-106x-end-to-end)
17. [Plot collapse](#17-plot-collapse-t10-t13-t14-batch-d)
18. [Detection readout](#18-detection-readout-t15-t16-t27-t28-batch-e)
19. [Replicate direct manipulation](#19-replicate-direct-manipulation-t11-t12-batch-g)
20. [Region means measure the tile, not the animal — and live-streaming throughput](#20-region-means-measure-the-tile-not-the-animal)

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

**That re-measurement is now done — see §14, and it only half-confirms the
caveat above.** The math-bound pass does scale further than decode's 2-4 worker
plateau, but it plateaus too, at N≈8 and **3.6x** — and its ceiling in realtime
terms (5.4x) is *below* this table's decode-only ceiling (7.6x), because the
whole pass is decode plus math and the table prices only the decode.

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

### Changing block size silently re-denominates the tuned detector

The consequence of the grid moving, and the reason the tracking decision above is
load-bearing rather than tidy. **`count_band` and `clump` are in raw block
units.** `inband_count` returns a block *count*, `detect_gate` compares it
straight against the `count_band` endpoints, and `largest_clump_per_frame`
returns an area in blocks. Nothing is normalized by how many blocks the region
happens to hold.

So the same numeric threshold means ~13x different things across the block range:

| block | atlas grid | blocks/replicate (7 reps) |
|---|---|---|
| 64 (default) | 41x5 = 205 | **~29** |
| 16 | 139x19 = 2641 | **~377** |

A `count_band` of `[20, ∞)` is a reasonable threshold at block 16 and
**unreachable** at block 64, where a region holds only ~29 blocks in total. The
detector does not fail, warn, or clamp — it simply stops firing. This is the same
shape as T17 (tuned state that quietly stops meaning what it meant) and the same
standing hazard as everything else in this file: the failure presents as *no
detections*, which is indistinguishable from *nothing happened*.

Two things follow. **Any UI that exposes `block_size` must convert the tuned
bands, or refuse to change it without saying what it invalidates** — this is the
load-bearing item in Batch N and the reason that batch is not the cost dialog it
was originally specced as. And **a `count_band` is only comparable across runs at
equal block size**, which is why resolved block size is already part of the
resume identity in §13; a summary carrying one block size must never be reused
for a request at another.

(Note: the 479-frame table above records the block-16 grid as 139x19 and the
120-frame verification as 146x19. Grid does not depend on frame count, so one of
the two is misrecorded. It does not affect anything here — the ratio is ~13x
either way — but do not treat either as exact without re-deriving it.)

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

---

## 11. Per-channel extraction (T18)

The whole-video commit detects on one channel via `detect_channel_region`'s
single `channel_attr`, but computed all four `LIVE_CHANNELS` regardless. It now
takes a `channels=` selection; the live preview deliberately does **not** use it,
because there a channel toggle must stay instant rather than trigger a re-extract.

### What it is worth, measured

Synthetic clip, 7 replicate-sized boxes (~297 px) on 1600x1200, 120 frames,
scale 1.0, block 64, grid 41x5, ROI decode. **Not the reference footage** —
`GX010047c2` is not on this machine.

| selection | wall | speedup | stages run |
|---|---|---|---|
| all four | 3.34 s | 1.00x | everything |
| `change` | 2.10 s | **1.59x** | products, blur |
| `tensor_speed` | 2.64 s | 1.27x | + flow_solve |
| `appearance` | 2.66 s | 1.26x | + flow_solve, residual |
| `intensity` | 0.51 s | **6.02x** | preprocess + reduce only |

Full-pass spans: `tensor_blur` 1.06 s (32%), `tensor_products` 0.54 s (16%),
`flow_solve` 0.49 s (15%), `appearance` 0.38 s (11%), `block_reduce` 0.31 s (9%).

**This refutes the plan's own estimate.** `todo.md` asserted that selecting
`change` "should skip nearly all the downstream math". It does not: `change` is
`J[2]`, so it needs `tensor_products` **and** `tensor_blur`, and the blur is the
largest single span in the pass. What `change` actually skips is `flow_solve`,
the appearance residual and the min-eigen — real, but 1.59x rather than the ~4x
the phrasing implied. The dependency table in the plan was right; the summary
sentence under it was not.

Two cautions on the number. Decode here is ~0% of wall time (synthetic mp4v,
tiny frames); on real 5.3K footage decode is a large fixed share, so the
end-to-end win will be **smaller** than 1.59x. And `intensity`'s 6.02x is not a
general result — it is the only channel that forms no tensor at all.

### The placeholder is zero-LENGTH, not zero-filled

`_stream_channels` still returns every `CHANNELS` key (the dict shape is fixed,
and `load_or_extract_channels`' sidecar depends on that), but an unselected
channel's array is `(0, ny, nx)`.

Zero-*filled* was rejected for the reason §10 traps 7 and 8 both turn on: zero is
a real measurable value on every one of these channels, so a zero-filled array is
indistinguishable from "measured, and nothing happened". Full-length NaN was the
first fix and was also rejected — it is honest but costs the memory the selection
exists to save: ~88 MB per channel per hour of 30 fps footage at a 41x5 grid, so
~354 MB of placeholder on a one-hour change-only commit. A zero-length array
cannot be mistaken for data, cannot silently broadcast against a real one, and
costs nothing.

`meta["channels_computed"]` is the authoritative list. `live_channel_source`
drops the placeholders entirely, so at the `ChannelData` layer — and only there —
key presence is equivalent, which is what keeps `ChannelData.available` and the
explorer's `_channel_available` meaning "this is real data".

### The cost model has to be told which selection it priced

`scale_sweep.measure_scale` took no channel selection, so it priced a
four-channel pass while the run it models now computes one. Left alone, that
inflates the fixed term `F` in `t(s) = F + M·s²` by up to 59% and moves the knee
`s* = √(F/M)` toward "downsampling is free" — the one direction §6 says this
model must never err in. It now takes `channels=` and `ScalePass` **records**
them, for the same reason it records `truncated`: wall time depends on a second
axis that a consumer comparing scales cannot recover from the number alone, and a
fit mixing a four-channel sample with a one-channel one reads the difference as
scale.

---

## 12. The headless path (T24, Batch J first slice)

`core/batch.py` + `cli/detect.py`. The GUI commit worker
(`live_scalogram_surface._ProcessWorker`) is the reference implementation and the
headless path runs the same two calls, `live_channel_source` then
`detect_channel_region`, so neither road reimplements detector math.
`tests/test_batch_cli.py::test_matches_gui_worker_path` asserts array equality on
`band_power`/`count`/`windowed`/`gate`/`clump` rather than on interval counts — a
summary can agree while the series underneath has drifted.

### Extraction is per video, not per region

The GUI extracts once per commit because a user commits one region at a time. A
batch run over six replicates that copied that would decode the same video six
times for one video's worth of pixels: the channel pass already produces the
whole atlas, and `detect_channel_region` only slices a region's block columns out
of it. Measured on the 2-replicate synthetic, two regions cost **1.0x** one
region's extraction, and detection is ~0.2% of the pass.

### Two clocks disagree, and the manifest's is the right one

`_ProcessWorker` passes `default_freqs(fps)` built from the decoder's float fps.
When clips are in play, the authoritative rate is the manifest's rational one
(§3 trap 2), and the wavelet bank is built *from* fps. The headless path passes
`freqs=None` so `detect_channel_region` reads `meta["fps"]`, which
`live_channel_source` has already replaced with the manifest's. The GUI worker is
the inconsistent one here; it matters only for clip-backed passes, where the two
rates differ.

### Truncation is a job failure, not a warning

Interactively a short decode can be a status-bar note because a human is
watching. Across a fan-out nobody reads the logs of runs that "succeeded", and
frames past a cut are **unexamined, not examined-and-clear** — the standing
false-negative shape. So a truncated pass raises unless `allow_truncated`, and
the shortfall is recorded in the result either way.

Two arithmetic traps in that guard, both found by review and each wrong in the
dangerous direction once:

- **Coverage must be measured against what the file holds past the offset.**
  `requested` was the whole clip regardless of `start`, so a `--start N` run with
  no `--frames` could never reach it and aborted as truncated — after paying the
  full extraction (238 s on a 30 479-frame clip before the error appeared).
- **An over-long request is not truncation.** Asking for more frames than the
  video has is the end of the video. Left unclamped it would fail every job that
  passed a generous `--frames`, which trains operators to pass
  `--allow-truncated` habitually and thereby disarms the guard entirely.

### Single-process throughput at scale 1.0

`rep3_intermittent_crop` (1 replicate, 8x8 grid at block 64, i.e. 512x512 working
px), `change` only, scale 1.0: **~88 frames/s**, against the clip's 59.94 fps —
**~1.47x realtime**. Split `tensor_blur` 53%, `tensor_products` 31%,
`preprocess` 11%, `block_reduce` 3%, decode ~0%.

This is math-bound, as §1 predicts at scale 1.0, and it is one replicate on a
small ROI — **do not extrapolate it to a node count.** The figure Batch J needs
is N-worker throughput on real multi-replicate footage, which this is not.

### The provenance obligation from Batch I is still unenforced

The summary records `clip_provenance`, but nothing here caches a clip-derived
result, so `cfg.cache_key`'s third argument still has no caller. Unchanged from
§10: the first code to cache clip-derived output must thread it.

## 13. File-level partitioning (T24, Batch J second slice)

`core/shard.py` + `cli/run.py`. A shard is `sorted(list)[i::N]` — stateless, so
a job runner launches N identical commands differing only in `--shard i/N` and
a preempted shard is relaunched rather than reconciled.

**Stride, not contiguous chunks.** Footage is named in capture order and file
size correlates strongly with position within a session, so `paths[i*k:(i+1)*k]`
hands one shard the long files and another the short ones. Both forms are
deterministic; only the stride balances.

### Every bug found here had the same shape: a shard that exits 0 having examined less footage than asked

Four distinct defects, all caught by review rather than by a failing test, and
all of them silent. They are recorded together because the *pattern* is the
finding — in a fan-out, "this shard completed the list it saw" is not evidence
that the list was right, and nothing downstream re-checks it.

**Output paths keyed on the basename collide, and the collision is invisible.**
Cameras number files per card, so `s1/GX010047.MP4` and `s2/GX010047.MP4` in one
run is the *normal* multi-session case. Both mapped to
`GX010047.detections.json`. The second video then found the first's summary,
matched on params and scale, and was reported **"skipped (up to date)"** — its
footage never decoded, the run exiting 0. Note the interaction: resume turned an
overwrite (bad) into a silent skip (worse), because the two features are
individually reasonable. `assign_output_names` now extends colliding stems
leftward with parent directories (`s1_GX010047`), computed over the **whole**
list so the mapping is independent of which shard a video lands in and identical
on every node. Unique stems keep the plain name `cli/detect.py` writes.

**`os.path.normcase` as a sort key makes the partition platform-dependent.**
It lowercases on Windows and is the identity on POSIX. `['B.mp4','a.mp4','C.mp4',
'd.mp4']` sorts `a,B,C,d` on Windows and `B,C,a,d` on Linux, so a Windows shard
0/2 covers `{a,C}` while a Linux shard 1/2 covers `{C,d}`: **`B.mp4` is examined
by nobody and `C.mp4` is decoded twice**, both exiting 0. normcase is still
correct for *de-duplication* — it is the filesystem's own rule for whether two
spellings are one file, and case-folding there would silently drop one of
`a.mp4`/`A.mp4` on POSIX where both can exist. So the two uses genuinely need
different keys, which is why one function had it right and wrong at once.

**Glob inherits the filesystem's case rules, for the same reason.** `*.mp4`
matches `GX010047.MP4` on Windows and not on Linux — and GoPro writes uppercase.
`_ci_pattern` rewrites each alphabetic character to a two-case class so the match
is platform-independent. It must skip path segments with no glob magic: a literal
`C:` rewritten to `[cC]:` is no longer a drive letter, which broke every absolute
Windows path on the first attempt.

**A completed short pass is indistinguishable from a completed full pass in the
resolved counts.** `--frames 1000` and a to-the-end run over a 1000-frame video
both end with `requested == covered == 1000, truncated == False`. So a shard run
once as a `--frames` smoke test and then re-run full-length was **skipped as up
to date**, leaving the smoke test standing as the video's detections. Only the
*raw* request distinguishes them, so `BatchResult` now records `requested_n`
(`None` = to the end) and `start_frame`, and both are part of the resume
identity. A summary lacking them is treated as not current and recomputed — the
safe direction.

### Resume must compare identity, not existence

The general rule the above is one instance of: skipping on output existence
alone silently returns results computed under *different* settings. The skip
requires the recorded params, resolved scale, resolved block size, and frame
window all to match. Two directions each cost something and only one is
recoverable: a false *stale* re-decodes work needlessly, a false *current*
returns a wrong answer forever. Resolved rather than raw settings are compared
for exactly this reason — `downsample=None` and `downsample=1.0` are the same
scale, and calling them stale would re-decode hours for a difference that does
not exist.

### Silence is the failure mode to design against, not just wrongness

Three guards exist only to convert silence into noise, none of which change any
result: a glob matching nothing raises, a directory holding no videos raises
(the wrong-nesting-level typo — videos one folder deeper contributed *nothing*
and the run still succeeded), and a missing pre-transcode manifest under an
explicit `--clip-dir` warns and records `used_clips: false` per video rather
than quietly decoding the source at ~25x the cost across a provenance boundary
(§10 — below `lossless` those are not the same pixels).

`--shard i/N` is 0-based and `i == N` is rejected by name, because an operator
launching `1/N`..`N/N` otherwise never runs shard 0 and every job reports
success.

### Failure isolation

A failed video records its error and the shard continues; the shard exits 1 at
the end with the failed videos **enumerated by name** in the report. Aborting at
the first failure would discard the decode already paid for by every earlier
video in the shard. The two exit codes are kept distinct on purpose: **2** is
bad usage (retrying on every node is pointless) and **1** is a failed job (it is
not).

## 14. N-worker throughput

The number Batch J was missing. §12's single-process figure was one replicate on
a 512x512 ROI and said in as many words not to extrapolate it; this is
`GX010047c2_02_17_26.MP4` — **7 replicates, 5312x2988, 23.976 fps** — at the
scale-1.0 default, `change` only, 600 frames per worker, measured by
`scripts/bench_worker_scaling.py`.

| workers | wall fps | extract fps | speedup | realtime | slowest worker |
|---|---|---|---|---|---|
| 1 | 35.2 | 36.6 | 1.00x | 1.53x | 16.4 s |
| 2 | 58.3 | 60.2 | 1.64x | 2.51x | 19.9 s |
| 4 | 99.6 | 102.7 | 2.81x | 4.28x | 23.4 s |
| 8 | 128.9 | **131.6** | **3.60x** | **5.49x** | 36.5 s |
| 16 | 129.1 | 130.9 | 3.58x | 5.46x | 73.3 s |

**The ceiling is ~130 fps / ~5.4x realtime, reached at N=8 on a 32-logical-core
box.** N=16 delivers the *same* aggregate throughput with every worker taking
twice as long — past the knee, added workers only redistribute a fixed budget.
So the useful per-node figure is **8 workers**, and launching one per core wastes
15 of them while doubling every job's latency, which is worse than neutral: a
preempted long job loses more work.

**Single-process on real footage is 36.6 fps, not §12's 88.** Same scale, same
channel — the difference is 7 replicates on a 41x5 block grid rather than 1 on
8x8. Extraction is per video (§12), so replicate count is close to free in
*decode* and very much not free in *math*.

**Do not read the 3.6x as "math-bound work parallelizes".** §1's caveat
predicted it would, and directionally it does — decode alone is flat past ~2-4
workers, this reaches 8. But 3.6x from 32 cores is not what compute-bound work
looks like, and the plateau is flat rather than gradual. The likely constraint is
**memory bandwidth**: `tensor_blur` (52% of the pass) and `tensor_products` (28%)
are large-array elementwise/convolution passes with low arithmetic intensity, and
those saturate a shared memory bus long before they saturate cores. This is a
hypothesis consistent with the shape of the curve, **not a measured attribution**
— confirming it needs a bandwidth counter or a cache-blocked reimplementation,
neither of which anything here depends on yet.

**Consequences for a node count.** At 5.4x realtime per node, 3000 h of footage
is ~555 node-hours: **~23 nodes for a 24 h turnaround**, ~70 for 8 h. That is
per *node*, not per core, and it assumes each node has its own memory bus — which
is exactly why the fan-out unit being a whole machine (`--shard i/N` over a file
list, §13) is the right granularity and why more workers per node is not the
lever. Treat these as the same machine's numbers: a node with more memory
channels moves the ceiling and one with fewer moves it down.

**Detection is still ~0.5% of the pass** (0.047 s against 8.60 s over 7
replicates), so extraction throughput and job throughput are the same number.
That equivalence is assumed by the table's `extract fps` column and stops holding
if a future channel makes the detector expensive.

**Two throughputs are reported because they bracket the truth.** `wall` includes
each worker's interpreter and numpy startup — real cost, but amortized to nothing
over a job of thousands of frames rather than this benchmark's 600. `extract`
divides by the *slowest* worker's in-process extraction time, which excludes
startup but retains all contention. A real long job approaches `extract`. The
harness gives each worker its own frame window rather than the same one, because
N workers over identical ranges share the OS page cache and flatter every N > 1;
offsets are free (seeking to frame 8000 cost 8.43 s against 8.60 s at frame 0).

**Two limits of the method, found by review after the numbers were taken.**
Neither is worth re-running the sweep over, but both bound how tightly the table
should be read, and the first becomes serious if anyone re-uses this harness at
low scale. (a) **Each level covers a different span** — offsets are `i * stride`,
so the N=1 baseline is only the first 600 frames while N=16 spans 9600. Harmless
here because decode is ~0% of a math-bound pass and tensor cost is
content-independent; **confounding at low scale**, where §1 says decode dominates
and decode cost does vary with content. (b) **`extract fps` flatters scaling
slightly**, dividing by `max(extract_seconds)` when workers start staggered by
interpreter startup, so the true overlap window is longer. Visible as `wall_s`
minus `slowest_extract_s` — 74.4 vs 73.3 at N=16, ~1.5% — and it grows with N.
Both are documented in the script's docstring.

**The clip-backed half of this measurement was not obtained**, because the
pre-transcode cut fails on this footage — see §15.

## 15. The container frame count is not the decodable frame count

Found while trying to obtain §14's clip-backed half, which is why that half is
missing. **The pre-transcode cut fails outright on `GX010047c2_02_17_26.MP4`**,
the project's own primary test footage:

```
GX010047c2_02_17_26__rep14.mkv holds 11308 frames but the source has 11328;
the cut re-timed the stream, so clip frame N is no longer source frame N
```

**The guard is right to exist and wrong here.** The stream was not re-timed.
The source container advertises 11328 video packets (`ffprobe -count_packets`
agrees: `nb_frames=11328`, `nb_read_packets=11328`) but **only 11308 frames
decode** — `ffmpeg -i src -f null -` with no filters, no crop and no
`-fps_mode` reports `frame=11308`. The 20 missing frames are lost in the
*source*, before anything this project does touches it. The cut wrote one frame
out per frame in, which is exactly the invariant the guard is meant to protect.

**Why the guard could not see that.** Its comment asserts "both counts come from
the same `CAP_PROP_FRAME_COUNT` estimator, so they are comparable even where that
estimator is approximate." That is false, and it is the whole bug. For the
H.264/MP4 source OpenCV returns the container's *claim* (11328); for the
transcoded Matroska clip it returns the true frame count (11308). Two different
quantities behind one property name — so the guard compares a claim against a
measurement and reports the difference as a defect in the cut.

### Alignment is not affected, and this was verified rather than argued

The dangerous reading of a 20-frame gap is that frames were dropped *scattered*
through the file, which would mean clip frame N ≠ source frame N and would break
every seek — the exact failure the guard names. It is not what happened. Whole
file, `rep14`'s box, source crop vs clip decoded to raw gray:

| comparison | mean abs diff (grey levels) |
|---|---|
| clip[N] vs source[N], sampled every 500 frames | **0.35 – 0.43** |
| the same at the last three frames | 0.75 – 0.85 |
| clip[N] vs source[N-1] (1-frame shift) | **2.39** |
| clip[N] vs source[N-2] | 3.09 |

The aligned residual is crf-12 encode noise; a one-frame shift is ~6x larger, so
the separation is unambiguous and holds from frame 0 to frame 11307. **The cut is
frame-accurate on real footage** — the first time Batch I's central invariant has
been checked against anything but a `testsrc` fixture, which could not have shown
this because a synthetic source has no undecodable packets.

### The same root cause independently breaks the source path

This is not a pre-transcode-only bug, and that is easy to miss because the cut
fails loudly first. `core/batch.run_video` computes `available = fc - start` from
the same OpenCV count, so **any pass that reaches the true end of this video
fails the truncation guard**:

```
--start 11000 --frames 400
error: decode covered 308 of 328 requested frames (93.9%);
everything after frame 11308 is UNEXAMINED, not clear
```

A full-length headless pass over this video therefore cannot complete without
`--allow-truncated` — which §12 identifies by name as the thing that "trains
operators into habitual `--allow-truncated` and thereby disarms the guard." The
guard is fail-closed, so this is a false *positive*, not the silent-false-negative
shape most of this file catalogues. It is still the more corrosive failure in
practice, because the standing workaround for it is to switch the real guard off.

### It is a per-file property, not a corpus-wide one

| video | packets | decoded |
|---|---|---|
| `GX010047c2_02_17_26.MP4` | 11328 | **11308** |
| `GX320051c2_02_19_26.MP4` | 32040 | 32040 |
| `stab_GX010050c2_02_18_26.MP4` | 30579 | 30579 |
| `rep3_intermittent_crop.MP4` | 30579 | 30579 |

One of four. So it cannot be dismissed as a quirk of one bad export, and it also
cannot be handled by a blanket constant — whatever the fix is, it has to derive
the count per file rather than assume a relationship between the two numbers.

### The fix (Batch O, landed)

The honest framing is that **the container count is a claim and the pipeline
treated it as a measurement.** A tolerance would have been wrong twice over: it
is a magic constant, and it would blind the guard to exactly the small
re-timings it exists to catch. So the true count became a recorded fact instead.

`core/framecount.py` resolves a decodable-frame count from, best first, a Batch I
manifest → a `.framecount.json` sidecar → the container's claim, and returns
`verified` alongside the number. Every result records which was used.

**The cut's guard was replaced, not adjusted.** It compared each clip's true
count against the source's claim; it now checks two separate things, because one
comparison could not see both failures:

- **Re-timing** comes from ffmpeg's own `dup_frames`/`drop_frames` in the
  `-progress` stream. This is strictly better than what it replaced, and not
  merely equivalent: `frame=` counts frames **encoded**, so a `cfr` conversion
  that duplicated frames reports its inflated length and the clip on disk agrees
  with it. The old count comparison could not have caught the failure it was
  written to catch. These counters are direct evidence.
- **Truncation** compares each clip against ffmpeg's `frame=`. Both sides are now
  measurements of the same quantity, so it is exact, and it passes on
  `GX010047c2`.
- **A cut that gave up part way** — found by review, and it is the one that
  nearly went out. Dropping the claim comparison also dropped the only thing
  separating *"the container over-claims by 20"* (fine) from *"ffmpeg died at
  frame 5000 of 11328 and exited 0"* (catastrophic: the manifest would record
  5000 as the source's true length, `resolve_frame_count` would return it as
  **verified**, and 6308 unread frames would be reported examined-and-clear).

**The discriminator is stderr, and it was measured rather than assumed.** On
`GX010047c2`, `ffmpeg -loglevel error -i src -map 0:v:0 -f null -` emits **zero
bytes of stderr and exits 0** while decoding 20 fewer frames than advertised.
Those 20 packets are not decode *errors*; they are simply not decodable frames.
A decode that genuinely fails part way does emit at that level. So *"came up
short **and** said something"* is evidence of a real failure and provably does
not fire on the legitimate case — where a tolerance on the size of the gap would
be a magic constant and would blind the guard to the small re-timings it exists
to catch.

The general lesson, which is the one worth carrying: **a bare shortfall carries
no information here.** Any guard keyed on the size of the gap is either the
original bug (refuse the legitimate case) or the regression above (accept the
fatal one). The signal has to come from somewhere other than the two counts.

`PRETRANSCODE_VERSION` is **3**. This is a bump rather than an added field
because `Manifest.frame_count` *changed meaning* — claim → measurement. A v2
manifest's `frame_count` reads as a decodable count and is not one, and a field
whose name survives a change of meaning is the T17 shape exactly.

**On the source path, one thing was wrong beyond the denominator.** Fixing
`available` alone was not enough: `run_video` passed the operator's raw `--frames`
to the extractor, which sets its own `truncated` flag against what it was asked
for. So `--start 11000 --frames 400` reported **"308 of 308 requested (100.0%)"
and failed anyway**. Both halves have to be clamped against the same length or
the guard contradicts itself in a single sentence. Measured, after: that pass now
exits 0.

**The remedy offered on a shortfall now depends on `verified`, and that is the
load-bearing part of the batch.** Against an unverified denominator the guard
points at `python -m cli.count_frames` — which *resolves* the ambiguity — and
mentions the override second. Pointing at `--allow-truncated` to work around a
false alarm is precisely how §12's warning comes true: tell someone to switch off
a real guard and they leave it off. An unverified count is also recorded as a
warning on a *clean* pass, because 100% coverage against an unmeasured
denominator is a different claim from 100% against a measured one.

**Counting costs a full decode and there is no way around it** — measured 1m36s
on the 11 GB `GX010047c2`. That is why it is written to a sidecar and why the cut
writes one as a byproduct: the cut decodes the whole source anyway, so it gets
the number for free.

**A synthetic fixture cannot test the real case, and this is named rather than
papered over.** `testsrc` has no undecodable packets, so its claim and its true
count always agree — which is exactly why the bug survived Batch I's test suite.
The provenance logic is tested on stand-in files; the arithmetic is verified
against real footage by hand (above).

---

## 16. Clip-backed throughput: the 25x decode win is ~1.06x end to end

Batch O unblocked the cut on `GX010047c2`, which is what §14's missing
clip-backed half was waiting on. Measured immediately afterwards, and the result
is not what Batch I's framing predicts.

Same video (7 replicates, 5312x2988), scale 1.0, `change` only, 2000 frames from
frame 0, single process, `python -m cli.detect` with and without `--manifest`:

| path | extract | throughput |
|---|---|---|
| source (live ROI crop) | 54.6 s | 36.6 fps |
| pre-transcoded clips | 51.7 s | **38.7 fps** |

**1.06x, against a decode measured at 25x faster in isolation (§10).** The
decode win is real and is not the thing that was limiting the pass. Two reasons,
both already in this file and neither previously joined up:

- **Decode is already overlapped.** The producer thread (§2) hides decode behind
  the math, so the pass only pays the part that does not fit underneath. The
  span table shows this directly: `decode` reads 0.01 s / 0% on both paths,
  because what it measures is the *blocking wait*, not the work.
- **The pass is math-bound at scale 1.0**, and §14 already suspected memory
  bandwidth: `tensor_blur` 53% + `tensor_products` 27% = 80% of the span, both
  low-arithmetic-intensity passes over large arrays. Removing 92% of the decoded
  pixels does not touch either.

**This does not retire the pre-transcode**, and it is not evidence the cut was
not worth building. What it retires is Batch I's headline claim as stated in
`todo.md` — *"the only lever that moves the decode floor"* is true and is no
longer the interesting sentence, because at scale 1.0 on this footage **decode is
not the floor**. §1's "decode-bound vs math-bound" split is footage- and
scale-dependent, and Batch K moved this workload across it by making scale 1.0
the default. The cut's value now rests on the cases it still plainly wins:
storage-local working sets, machines where the source is on slow or remote disk,
and any future pass at a smaller scale where math shrinks and decode does not.

**Untested, and the case most likely to differ: N-worker.** §14's ceiling is 8
workers on a 32-core box with a suspected memory-bandwidth wall. Eight processes
each decoding a full 5.3K frame contend for exactly that resource, so the clip
win may be substantially larger in the configuration that actually runs a corpus
than in the single-process one measured here. **Hypothesis, not measured** — the
honest reading of the table above is "single process only".

### Clip-backed and source-backed detections agreed exactly

Worth recording because it is the first end-to-end check of the thing §10 could
only reason about. Same params, both paths, `change` at crf 12:

| window | source | clips |
|---|---|---|
| 2000 frames from 0 | 14000 detected frames | **14000** |
| 308 frames from 11000 | 2156 | **2156** |

§10 measured a per-channel sensitivity table and declined to say whether any
given *behaviour* survives the quality setting. On this footage, at these tuned
thresholds, it does — exactly. That is one video and one parameter set, so it is
not a general result about crf 12, but it is a real one and it is the first
evidence in either direction.

## 17. Plot collapse (T10, T13, T14, Batch D)

**The measurement.** Scrubbing 120 frames of a synthetic clip through
`ScalogramExplorer`, everything collapsed vs everything expanded:

| grid | all collapsed | all expanded | ratio |
|---|---|---|---|
| 8x8 blocks, T=1200 | 160 ms | 513 ms | **3.2x** |
| 48x48 blocks, T=1200 | 180 ms | 588 ms | **3.3x** |

**The ratio barely moves with grid size, and that is the informative part.**
The plan (T10) framed the lag as a *block=1 / weak-downsampling* problem, which
implies the cost scales with the number of blocks. It does not.
`DensityPlot._density_image` bins into a `w x h` raster, so its cost is set by
the widget's **pixel area**, not by `blocks x frames` — its own docstring says
so, and this measurement confirms it. The panel is expensive because it holds
~11 plots that each repaint on every cursor move, not because any one of them
handles a lot of data. So collapsing helps *uniformly*, at every block size,
and "shrink the grid to make the UI faster" was never the lever it looked like.

Total collapsed panel height is 198 px against ~800 expanded, which is the
other half of why a collapsed panel is cheap: less area to blit.

**A collapsed plot must skip the *paint*, not merely the geometry.** Setting a
widget to 18 px still repaints it. The mechanism therefore returns early from
`paintEvent` and drops the cached `QImage` (`_release_render_cache`), and
`set_cursor`/`set_series` skip `update()` entirely so no paint is even queued.
Only the QImage is dropped, never the matrix: the matrix is reconstructible only
from a cube the explorer may have evicted, so discarding it could cost a full
rebuild to reopen a plot.

**The bug this batch nearly shipped, and the rule that prevents it.**
`_refresh_densities` skipped the band-power sum for collapsed plots. Combined
with collapsed-by-default (T14), that meant the **selected** channel was skipped
too — and `_recompute_counts` / `_recompute_clump` read that matrix as the
**detector's input**, not as a picture. Result: the explorer opened with a cube
built and cached, and the entire detection sweep (count, windowed count,
detect, clump) reading zero-length. A silent false negative, arrived at from a
pure-UI change, and self-sustaining: the empty matrix re-armed the
auto-collapse-when-empty rule (T13) that caused it, so the plot could never
populate.

> **Visibility decides what is DRAWN. It must never decide what is COMPUTED.**

The selected channel is now exempt from the skip. This is the same shape as
T17 (tuned state quietly stops meaning what it meant) reached from a different
direction, and it is worth stating because the tempting version of this
optimization — "don't compute what you don't show" — is *correct for four of
the five density plots and catastrophic for the fifth*.

`tests/test_plot_collapse.py::CollapseDoesNotDisarmTheDetectorTest` covers it,
and was confirmed to fail against the unfixed code rather than merely passing
against the fixed code. A test that only asserted geometry — which is what a
"collapse plots" batch invites — would have passed throughout.

**Not fixed, and deliberately.** `_recompute_clump` runs an O(T) connected-
components pass even when `clump_plot` is collapsed. Skipping it would be a
real saving, but the clump series is a detection quantity and the failure above
is precisely what skipping a detection quantity looks like. It stays until
someone establishes that nothing downstream reads it.

**A GUI-test trap worth adding to §9.** `QApplication.instance() or
QApplication([])` inside a helper *function* creates an app that nothing holds a
reference to; it is garbage collected on return and the next `QWidget`
constructor aborts the process with "Must construct a QApplication before a
QWidget". That surfaces as a bare **exit 9 with no failing test named** — the
same signature as the `closeEvent`/debounce crash already recorded in §9, from
an unrelated cause. It only appears when the module runs *alone*; under the full
suite another module has already made an app that outlives it, so the file
passes. Hold the `QApplication` at module scope.

---

## 18. Detection readout (T15, T16, T27, T28, Batch E)

Interactive polish with no measurement attached, so this section records the
decisions and the three traps rather than numbers.

**T15 -- the gate moved onto the plot whose band produces it.** The separate
0/1 `detect_plot` is gone; `count_w_plot` now shades positive frames green
behind its series. The reason is not screen real estate: the gate is computed
against *that plot's own band*, so a threshold drag and its consequence now
share one x-axis. Previously the user dragged a handle on one plot and read the
result on another, correlating two axes by eye to see whether the drag had done
anything.

**The gate is owned by the explorer, not by a widget.** `ScalogramExplorer.detect`
holds it and `_set_detect` publishes it to its two pictures (the shading and the
badge). Reading the gate back off a plot -- which the old code effectively did,
`detect_plot.y` being the only copy -- means a change to a *drawing* can change
what the detection *is*. That is the Batch D failure (section 17) in a second
form, and the existing regression test now asserts `ex.detect`, not a widget.

**Three traps, all found in review or by writing the test:**

- **A single positive frame is 0.03 px wide** on a 400 px plot over a 14000
  frame clip. The series is enveloped down to pixel columns, but a detection
  must not be: rounding it away draws a clean clip over a real detection.
  Spans are drawn in frame coordinates with a **1 px floor**. This is the
  standing silent-false-negative shape and is the one thing in E worth a test.
- **A stale mask invents events.** `set_series` now clears `_detect_mask`.
  Held across a replicate switch, the previous clip's detections would shade
  the new one -- indistinguishable from a real result, and wrong in the
  direction that *adds* events rather than losing them. Batch D's
  empty-rather-than-stale rule, second application.
- **Runs touching either end of the clip** are closed by padding the mask with
  a zero on both sides before differencing. A naive edge-diff drops a run that
  reaches the final frame, and the end of a clip is where a truncated event
  sits.

**T16 -- the badge is exempt from shift-to-peek, and that is the whole point.**
`DETECTED` (green fill, bold black) is painted bottom-right of the drawn frame,
deliberately OUTSIDE `FrameView`'s `_overlays_hidden` guard. Shift-to-peek hides
annotations so the raw pixels can be judged, which is exactly the moment the
user is deciding whether a detection is real -- the worst possible moment to
withdraw the detector's verdict. It also annotates nothing (it reports a
per-frame result rather than marking a place in the image), so it cannot occlude
what peek exists to reveal. Pinned by a test that renders the widget and
inspects pixels; confirmed to fail when the badge is moved inside the guard.

**The badge colour is duplicated, not shared.** `speed_explorer` imports
`video_panel`, so `video_panel` cannot import `DETECT` back. `DETECT_BADGE` is
a second literal with a test asserting the two are equal -- the cheapest thing
that stops the shading and the badge drifting into two different greens for one
gate.

**T27 needed the mechanism moved, not a two-line call.** The plan recorded this
as "`set_auto_collapse_empty` exists, two-line change". It did exist, but on
`DensityPlot`, keyed on `self.matrix` -- and the four sweep plots are
`MiniPlot`/`PixelBarPlot`, which carry a series and no matrix. The flag, its
setter and `_refresh_auto_collapse` moved up to `MiniPlot` behind an overridable
`_is_empty()` (series-based by default, matrix-based on `DensityPlot`).

**T28 -- `count_w_plot` is exempt from collapsed-by-default, and T15 is what
settles it.** It already carried the detection threshold band, the primary
tuning control; since T15 it also carries the detection readout. Collapsing it
put both the control and its result one click away on open, on the one panel
whose purpose is to show them. Following T14 literally was defeating T14's
reason for existing. The *auto*-collapse still applies: with no series there is
nothing to tune against, and it opens itself when data arrives.

**Pre-existing encoding damage, found and fixed alongside this batch.**
`gui/explorers/scalogram_explorer.py` carried a UTF-8 BOM and **18
double-encoded characters** -- real UTF-8 bytes once decoded as cp1252 and
re-encoded, so `…` was stored as `â€¦`, `—` as `â€”`, `·` as `Â·`, `×` as `Ã—`.
Six of them were in *user-visible status strings* (`"startingâ€¦"`, the
`"â€” all replicates (click one to select) â€”"` combo entry). Repaired by
targeted replacement, not a whole-file `encode('cp1252').decode('utf-8')` round
trip: the file is mixed, so a blanket reversal corrupts the characters that were
already correct.

**Two traps worth keeping, because both cost time here:**

- **`Set-Content -Encoding utf8` on Windows PowerShell 5.1 writes a BOM.** That
  is how the BOM most likely arrived, and this session reproduced it exactly --
  a scripted edit to `video_panel.py` added one, visible only as a phantom
  first-line diff. Use `[System.IO.File]::WriteAllText` with
  `UTF8Encoding($false)`, or write from Python.
- **The PowerShell console mis-decodes UTF-8 when *reading*.** `Get-Content` on
  this file showed `â€"` for every correct em-dash, which led to an initial
  conclusion that `FINDINGS.md` was corrupted throughout. It is not: every
  non-ASCII character in this file (`—`, `§`, `→`, `≈`, `√`, `±`, `∞`, `←`, `≠`)
  is intentional and correctly stored. **Verify encoding claims by reading the
  bytes in Python, never from console output** -- the display artifact and the
  real defect look identical in a terminal.

## 19. Replicate direct manipulation (T11, T12, Batch G)

Interactive work, so this records decisions and traps rather than numbers.

**T20 is not in the file the batch named, and is not a deletion.** The plan says
"drop the replicate dropdown -- redundant with click navigation", filed under
`tab2_replicates`. There is no dropdown on the replicate tab; it has a
`QListWidget`. The widget meant is `region_combo`, and it exists in **five
explorers**, where it is not redundant with click navigation but *is* the
selection state: `active_region_index` is read off `currentData()`,
`_on_video_clicked` is implemented as `setCurrentIndex(findData(i))`, and
`_clear_region_focus` returns to a **pooled scope (`-1`) that has no click
gesture at all**. Deleting the combo therefore means rehoming three things, not
removing one. Third time the Order section's warning has paid: check what a
widget derives from, and where the state actually lives, before trusting a
per-batch "· file" annotation.

**T12's right-click-to-delete was refused, deliberately.** Right-click is
`back_requested` in four explorers and tab3. T12 also introduces the zoom, which
is what makes the replicate tab *need* an un-zoom -- so honouring the item as
written would have made one button mean "go back" on five screens and "destroy a
replicate" on the sixth, one misfire from silent data loss on the only screen
where the boxes are authored. Delete stays on the button. **The item was the
right instinct at the wrong altitude**: it wanted a fast delete, and it was
written before the zoom it shares a batch with existed.

**The zoom is the box's `frac` verbatim, with no margin, because the explorers'
is.** A margin was implemented first and was wrong: this view and an explorer's
show the same replicate, so any difference in magnification makes them
non-comparable by eye, which is the entire reason to zoom. The rationale offered
for the margin -- "no empty space left to drag the box into" -- was also false:
`_delta_to` uses `_frac_of(clip=False)` and Qt grabs the mouse on press, so a
drag extrapolates past the widget edge and works at full zoom. That is now
pinned by a test, because the no-margin decision makes it load-bearing.

**T11 removed a copy of state rather than syncing it.** `stamp_size` was stored
and written by the last box *drawn*; the selection was separate. The two
diverged the moment you selected an older box -- the highlight said one thing
and the next click placed another size -- and same-size replicates are the whole
point (a fixed "min blocks" only means one thing if the boxes match). It is now
a property derived from the selection, so the label, the highlight, and what a
click places cannot disagree. Same remedy class as T17 and the count-band
re-denomination: state that quietly stops meaning what it meant.

**Two silent-write bugs found by review, both confirmed by driving real mouse
events, both in the "writes a change the user did not make" direction:**

- **`moved` and the displacement had separate sources.** The release decided
  *whether* it was a drag from `e.pos()` but applied *how far* from
  `_move_delta`, cached by the last move event. Qt compresses move events under
  load, so press-then-release with none in between is reachable: the release
  reads as a 150 px drag while the cached delta is still zero, rewrites the box
  to where it already was, and fires `_rebuild_rois()` and a sidecar write for a
  reposition that never happened. Both now come from the release event.
- **A chorded right-click mid-drag committed a partial one.** Right-press
  returned early without clearing `_move_idx`, so the right *release* fell into
  the move path and repositioned the box using whatever delta the drag had
  reached -- and `back_requested`'s un-zoom had already changed the coordinate
  mapping that delta was measured in, so it landed somewhere never dragged to.
  Right-click during a drag now cancels it and is not also a "back".

**This tab matches replicates by POSITION, where `_source_box` matches by id.**
The index in `box_grabbed` / `box_clicked` / `box_moved` is a position in the
list `_redraw_boxes` publishes, and `_on_box_moved` indexes `self.replicates`
with it. That holds only because the comprehension is 1:1 and in order, and it
is the *opposite* convention to `_source_box`, which matches by `replicate_id`
precisely because `build_layout` re-sorts. Filtering that comprehension -- to
hide boxes outside the zoom, the obvious future optimization -- would silently
move the wrong replicate. The contract is commented at the point it is created.

---

## 20. The band-power playback lag is the video OVERLAY, not the plots

**The symptom.** On the live surface, playback runs ~realtime until the change
energy cube lands and ~1/3 realtime after. The obvious reading -- "the band
power channels are expensive" -- is wrong twice over: the cube is already built
by then (it is what "loaded in" means), and the plots are not what got slower.

**The plots were ruled out by measurement, not by argument.** §17 had already
priced the panel, and the `DensityPlot._data_range` memo took out the last of
it. The decisive test was making the SELECTED channel's heatmap collapsible
again (Batch N had stripped its `[+]`) purely so the two states could be
compared live: collapsed, playback was still slow. That is worth keeping as a
standing capability -- a permanent plot is a plot you cannot A/B.

**What actually turns on.** The "highlight blocks in band" branch of
`_redraw_video` is gated on `m.size and ... m.shape[1] == total`, i.e. on the
selected channel's density matrix being populated. Before the cube it is
skipped entirely; from `_on_cube_ready` onward it runs on every frame. It is
the ONLY per-frame work whose on/off point is exactly the cube's arrival, which
is what makes the symptom so sharply staged.

Measured through the real `_redraw_video`, 1600x900 source, 40x50 blocks:

| | ms/frame |
|---|---|
| pre-cube (branch skipped) | 4.6 |
| post-cube, old code | 12.2 |
| post-cube, new code | 6.8 |
| post-cube, highlight unchecked | 4.7 |

The branch itself: **7.7 ms -> 2.0 ms, 3.9x.**

**The cost was `np.copyto(where=)`, and that is the counter-intuitive part.**
The branch does five things and the expensive one is not the arithmetic. A
component-level micro-benchmark put `np.copyto(roi, blended, where=...)` at
0.06 ms and pointed at `addWeighted` and `findContours` instead -- it was
**wrong**, because it passed `roi` as both source and destination and numpy
short-circuited the copy. Only end-to-end candidate timings found it. `where`
takes a full-size boolean that has to be broadcast to three channels and walked
elementwise; `cv2.copyTo` takes the single-channel mask directly.

> A micro-benchmark whose result is suspiciously good is measuring something
> other than what you think. Time whole candidate implementations.

`findContours` was the intuitive suspect and is a red herring: tracing at
display resolution costs only 0.7 ms of the 6.4, and moving it to block
resolution is not worth the coordinate-scaling code.

**Rejected: restricting the work to the bbox of passing blocks.** It measured
no better than the simple fix (1.9 vs 2.0 ms) because a scattered mask has a
bbox covering most of the tile, and it adds a block->display coordinate
conversion whose failure mode is a highlight on the wrong pixels.

**Also removed:** the `np.zeros_like(roi)` tint, a full display-resolution
uint8 image allocated per region per frame to hold a constant. Now cached by
shape on the explorer.

**The fix is pixel-identical and is tested that way.** A speed fix that alters
the picture is worse than the lag, because the picture is what the value band
is tuned against by eye. `tests/test_highlight_overlay.py` holds the old
implementation verbatim as an oracle -- deliberately duplicated rather than
shared, so it cannot drift with the code under test -- and checks equality
across mask densities, the 0.5/0.5 rounding boundaries, and a NON-CONTIGUOUS
sliced ROI view (which is what the call site actually passes, and the one case
where `cv2.copyTo`'s in-place write could have differed). Confirmed to fail
against a sabotaged tint rather than merely to pass against the fixed code.

### The budget is 8.33 ms, not 33 ms, and that is why this was ever visible

`output.mp4` is **120 fps**. Every per-frame cost on this panel is priced
against an **8.33 ms** budget, of which a sequential decode already takes
**2.70 ms** (1920x1080, and the `frame_at` fast path is working -- this is
decode, not seek). That leaves ~5.6 ms for the overlay and every plot repaint
combined.

This is the single most useful number in this section, and not knowing it is
what made the first diagnosis so slow. At 30 fps none of the costs below would
be visible at all; at 120 fps a 2 ms regression is a quarter of the frame. Any
future "is this fast enough" question on the live surface has to name the
clip's fps before it means anything.

**A rate meter now sits beside the timestamp** (`rate_lbl`), comparing footage-
seconds advanced against wall-seconds elapsed over a window that closes on the
first tick past one wall second. It reads ~1.00x when the panel keeps up and
cannot read above that, since playback is timer-driven at the clip's fps -- it
measures whether the per-frame work fits the budget, not unthrottled
throughput. The window restarts on resume so a pause is not charged against
zero frames advanced.

### MiniPlot was the second half, and the collapse A/B could not have found it

With the overlay fixed, playback measured a consistent **0.7x**. The remaining
cost was `MiniPlot.paintEvent` rebuilding its decimated envelope on every
paint -- a Python loop calling `np.nanmax` once per pixel column -- for a
series that had not changed between frames. **1.70 ms/paint at T=1200, against
0.11 ms for the DensityPlot beside it.**

> The cheap-looking sparkline was 15x the cost of the heatmap it sat under.

**Why the collapse A/B missed it.** The plot carrying this cost is
`count_w_plot`, which is PERMANENT (no `[+]`, Batch N) and becomes populated at
exactly the moment the cube lands -- the same trigger as the overlay. Only the
selected density heatmap could be collapsed, so the experiment that correctly
exonerated the heatmaps could not reach the one plot that mattered. A
non-collapsible plot is a plot that cannot be ruled out.

Memoised on a version counter with a single writer (`_bump_series_version`,
called by `set_series` and by the two subclasses that write `self.y` directly),
and the loop replaced by `np.fmax.reduceat`. The polyline is cached too -- it
was a second per-column Python loop. `PixelBarPlot._bar_image` got the same
treatment; it had no cache at all.

| plot | before | after |
|---|---|---|
| MiniPlot (T=1200) | 1.70 ms | **0.20 ms** |
| PixelBarPlot (T=1200) | 0.63 ms | **0.10 ms** |
| DensityPlot (T=1200, B=2000) | 0.11 ms | 0.13 ms |

**One deliberate divergence from the old loop**, pinned in
`tests/test_miniplot_envelope.py`: an all-NaN column made `np.nanmax` emit NaN
(plus a RuntimeWarning), which became a NaN QPointF and an undefined polyline
vertex. It now falls back to the axis minimum -- what the loop's own
unreachable `else lo` branch intended. Everything else is asserted bit-identical
against the original loop kept verbatim as an oracle, including that a
single-frame burst in 1200 frames still survives decimation to 460 columns as
the peak, which is the whole reason the envelope is a max and not a mean.

---

## 20. Extraction: the tensor was computing five components nobody read

Measured on the reference footage (`GX010047c2_02_17_26.MP4`, 5312x2988, 11
replicate tiles of ~349x321 at scale 1.0, block 64, grid 76x6, ROI decode),
300-frame window = 10 s of footage, best of 3.

| channel | before | after | speedup |
|---|---|---|---|
| `change` (**the detection default**) | 9.91 s | **2.83 s** | **3.50x** |
| `tensor_speed` | 12.21 s | 9.39 s | 1.30x |
| `appearance` | 18.80 s | 13.59 s | 1.38x |
| `intensity` | 2.85 s | 3.08 s | ~1x (noise; it touches none of this) |

The change-only pass went from **0.8x realtime to 3.5x realtime** — from slower
than the footage to faster than it.

### The win, and why it was invisible

`_stream_channels` built all six tensor components and ran **six full-resolution
Gaussian blurs** per tile per frame regardless of selection. `change` is `tt`
alone: one squared temporal difference. Five of six products and five of six
blurs were computed and discarded every frame — and `tt` is the one component
needing no spatial gradient, so `np.gradient` was pure waste too.

Section 11 measured the channel selection and reported 1.59x for `change`,
correctly noting the blur is the largest span. What it did not spot is that the
blur was *itself* mostly waste. The span table said `tensor_blur 32%` and that
was read as "the blur is expensive", when it meant "the blur is being run six
times for one answer". **A span table names the stage, not the necessity of the
work inside it** — the same trap section 1 records for `block_reduce`.

`tensor_products` now takes a component selection and returns unrequested planes
as **`None`**, not zeros — a partial tensor reaching `flow_from_tensor` or
`spatial_min_eigen` raises instead of silently solving on zeros. Same reasoning
as the zero-length placeholder of section 11.

Spans for `change` after: `preprocess` 33%, `tensor_blur` 23%, `block_reduce`
12%, `tensor_products` 11%, `decode` 5%. Decode is fully hidden by prefetch and
is **no longer the floor** — `intensity`, which does no tensor work at all, is
now only ~10% faster than `change`.

### Two smaller wins, and their exact numerical cost

* **Ragged block-mean as slabs, not cells.** A 349x321 tile at block 64 has 25
  regular cells and **11** ragged ones, so `_block_mean` ran 11 tiny `mean`
  calls per tile per frame — 24 200 in a 200-frame pass, most of its cost.
  Ceiling division guarantees at most one short row and one short column, so the
  edge is three slabs (bottom strip, right strip, corner), not a loop. 1.7-3x.
* **Fused z-score.** Two numpy traversals for mean/std plus four temporaries
  became one `cv2.meanStdDev` and one fused `g*a + b`. 2.3-3x on `_normalize`.
  The trailing `astype(np.float32)` in `Preprocessor.apply` was copying a full
  plane per replicate per frame to change nothing; now `asarray`.

**Attributed by reverting each file in isolation against real footage:**

| change | end-to-end result |
|---|---|
| tensor component selection | **bit-identical** on every channel |
| slab block-mean + fused z-score | max abs delta **1.5e-5** (one float32 ULP at 128) |

The tensor selection — the 3.5x — costs *nothing* numerically; it only stops
computing discarded values. The ULP-level drift comes only from reassociating
two sums, and its direction is toward more accuracy, not less: `cv2.meanStdDev`
accumulates in float64 where `ndarray.std()` accumulated in float32.

One caveat worth carrying: on `tensor_speed` that 1.5e-5 input perturbation
comes out as **3e-3 relative** on the channel. That is not a bug, it is the LK
solve's conditioning — it divides by a spatial determinant that is near zero
exactly where the aperture problem bites. Any future change to preprocessing
should expect the same amplification on flow-derived channels, and it is a
reason to re-check a tuned `tensor_speed` threshold after touching preprocess.

### REJECTED: dropping the blur for `change`, with the measurement

Tempting, and wrong. `change` is `blockmean(gauss(it^2, sigma=2))`, and the
64x64 block mean looks like it should swamp a 6-px Gaussian — the blur is 23% of
the remaining pass, so this is the obvious next cut.

Measured over 47 124 block-samples of real footage: median relative difference
1.2%, but **p95 34.9% and max 327%**, Spearman rank correlation 0.992.

The a-priori argument fails because change energy is spatially **sparse** — a
moving edge is a thin high-value ridge, not a smooth field — so the blur moves a
large fraction of a block's total energy across its boundary. The tail is where
detection happens, so a p95 of 35% is disqualifying however good the median and
the correlation look. **Do not remove this blur**, and do not trust a
correlation coefficient to license a change to a thresholded channel.

## 21. The per-pixel cache never built on real footage: a budget, not a bug

Reported as "changing the block size refetches the extraction — not sure where
that regression came from." It was not a regression. `git log -S` shows
`_pp_budget` and `_perpixel_fits` unchanged since they landed in `90da164`, and
the 10 s window default unchanged since the tab was created. The feature had
never worked on this project's own clips.

The re-reduce path was correct throughout. What failed was the gate in front of
it: `_perpixel_fits` charged the window against a flat **2 GiB**, and at scale
1.0 that buys almost nothing on 5.3K footage with a dozen replicate boxes.
Measured against the real `.rois.json` for each clip:

| clip | reps | scale 1.0 | scale 0.5 | scale 0.25 |
| --- | --- | --- | --- | --- |
| `stab_GX010050c2` 60 fps | 13 | **0.97 s** | 3.89 s | 15.5 s |
| `GX320051c2` 60 fps | 8 | **1.53 s** | 6.12 s | 24.5 s |
| `GX010047c2` 24 fps | 11 | **4.50 s** | 18.0 s | 71.5 s |
| `output.mp4` 120 fps | 2 | **0.97 s** | 3.90 s | 15.6 s |

Against a 10 s default window, every clip missed at scale 1.0, so `_pp` stayed
`None` and every Block change fell through to a full re-extract — the exact cost
the cache exists to remove.

**2 GiB was a magic number.** On the development machine (64 GB installed,
~34 GB available) it is ~3% of the box and bears no relation to it. This is the
same frame-relative-versus-machine-relative error the project has flagged
elsewhere: the budget belongs in units of what the machine actually has.
`core/sysmem.py` now reads available physical memory natively per platform (no
psutil for one number) and `_pp_budget` is a quarter of it, floored at the old
2 GiB so small or unreadable machines keep today's behaviour, capped at 16 GiB
where decode latency rather than memory becomes binding. The quarter is because
the extract holds its own copy alongside the cache, so peak is ~2x the budget.

At 8.5 GiB that clears the 10 s default at scale 0.5 on every clip and at scale
1.0 on three of five. **It does not clear all of them** — `stab_GX010050c2` at
full scale still fits only 4.07 s — which is why the fallback is no longer
silent: `_on_block_changed` now prints what the window needs, what the budget
is, and how many seconds would fit. A silent fallback under a tooltip that
promises a re-reduce is what made this read as a regression for months.

### The test gap that hid it

`test_block_change_re_reduces_cached_pixel_channels` passed the whole time. It
assigns `_pp` and `_pp_key` by hand, so it verifies the *compare* half and never
the *store* half — it cannot fail when nothing populates the cache. The general
shape: a test that constructs the state under test rather than driving the code
that produces it will not notice when production stops producing it.
`test_extract_then_block_change_reuses_the_cache_it_just_stored` runs a real
extract and asserts on the cache that pass actually stored; it fails if the fit
check misses.

---

## 20. Region means measure the tile, not the animal

**Measured 2026-07-20 on `GX010047c2_02_17_26`, 6 tiles, block 64, scale 1.0,
400 frames.** Mean `change` per replicate over time:

| replicate | label | mean `change` |
|---|---|---|
| 0 | rep1-mostly-flying-some-pause | 9.08 |
| 1 | rep2-backlit-flying-whole-time | 15.04 |
| 2 | rep3-intermittent | 9.62 |
| 3 | rep4 | 15.03 |
| 4 | rep5 | 10.94 |
| **5** | **rep6-NO-flying** | **15.22 — the highest of all six** |

**The replicate with no flying has the most change energy.** A region mean over
~81 blocks is dominated by substrate and lighting differences between tiles; the
animal occupies a handful of blocks and is diluted to nothing. Within a replicate
the mean varies only ~1.5x over 400 frames, so it carries almost no temporal
signal either.

**This re-derives the detection-channel design the hard way.** The existing
detector already reads a channel as *in-band block count* and *largest connected
clump*, never as a mean, for exactly this reason. An attempt to validate the new
`occupancy` channel against the marked `Still` span used region means and produced
a change still/moving ratio of **0.988** — an apparent null result that was really
a null *statistic*. Nothing was wrong with the channel; the read was wrong.

**So: any channel validation, comparison or ablation must use the tail statistic
— per-block density, count in band, clump area — not a region mean.** A mean over
a region whose animal occupies a small fraction of the blocks measures the region,
not the behaviour. This is cheap to get wrong because the mean is the obvious
first thing to compute and it returns a plausible number.

**A second thing that attempt exposed: `marks.json` holds one 0.39 s span and no
animal-absent spans at all.** So the corpus cannot currently discriminate
present-but-still from empty, which is the entire claim `occupancy` exists to
support. That is a data gap, not a code gap, and no channel work substitutes for
it (todo.md T30).

### Live-streaming throughput, and why channel choice is now a UI constraint

Same clip and geometry, measured in the same session:

| channel set | throughput | vs 23.98 fps playback |
|---|---|---|
| `intensity` + `change` | ~80 fps | **3.3x realtime** |
| + `appearance` (needs the flow solve) | ~18 fps | **0.75x — falls behind** |

The flow solve alone is ~24% of a full pass. So continuous "process up to the
second" streaming is possible for the cheap channel set and **impossible** for the
full one. Channel selection stopped being only a cost knob and became a
live-performance knob; a streaming surface must show the frontier falling behind
rather than presenting stale plots as current.

### The cone of influence is not cosmetic

Morlet at `w0=6` has e-folding support ~`1.46/f` seconds: **~2.9 s at 0.5 Hz,
~0.07 s at 20 Hz.** On a live trailing window the newest frames are therefore
zero-padding artifact at low frequencies and perfectly good at high ones —
progressive fill is *naturally frequency-dependent*, and a wingbeat band is
trustworthy almost immediately while a 0.5 Hz band needs seconds of history.

Drawing that edge as data would be the same class of failure as §7's withdrawn
detection panel and §4's withdrawn `sig_corr`: an authoritative-looking number
that does not mean what it appears to. Fade or hatch the wedge.

### Two API traps found by review in the same session

Both are in `core/derived_channels.py` / `core/stream_buffer.py` and are fixed;
recorded because both are easy to reintroduce.

- **A default threshold derived from the data it is applied to makes a channel a
  function of its own window.** `intensive()` and `ratio()` originally defaulted
  their floor to a fraction of the median of the array passed in. Under a
  growing-then-sliding ring that recomputes every second, the floor moves every
  tick: a block near it flickers between a finite value and NaN with no input
  having changed, and a threshold tuned on a 10 s window means something else on
  the whole clip. This is §5's `rescale_count_band` trap in a new place. Floors
  are absolute now. **The general rule: a constant that scales with its own input
  is not a constant.**
- **Validate every plane before writing any.** `StreamBuffer.append` originally
  validated as it wrote. On a full ring `index % capacity` is the *oldest
  retained* frame's slot, so a frame missing its second channel clobbered the
  first channel of a live frame and then raised — leaving that frame holding data
  from two points in the clip while `[start, frontier)` still claimed to be
  intact.

---

## 22. Batch P: the extraction loop as a generator

**What changed.** `_stream_channels`'s loop body became
`stream_channel_planes(video_path, plan)`, a generator yielding
`(absolute_index, {channel: (ny, nx)})` per frame. `_stream_channels` is now a
consumer that drains it into the same `(n, ny, nx)` arrays as before. The
windowed API — `extract_channels`, `extract_channels_live`,
`live_channel_source` — is untouched in signature and return shape, and
`CHANNEL_VERSION` did not move, because no arithmetic changed.

**The measurement, which was the point of taking it.** The plan flagged
per-frame yield overhead as "negligible, but measure rather than assume."
Measured on `GX010047c2`, 11 tiles, block 64, scale 1.0, `intensity`+`change`,
400-frame window, back-to-back against the stashed pre-refactor file:

| | best of 3 | fps |
|---|---|---|
| pre-refactor windowed | 5.028 s | 79.5 |
| post-refactor windowed | 5.031 s | 79.5 |

Unchanged. A separate interleaved run (6 reps each) put a bare drain that
*discards* every frame 3.7% ahead of the windowed fill — and the windowed arm
also pays the big-array write, so the generator's own yield-plus-allocate cost is
below that. Per frame the allocation is `len(want)` arrays of `ny*nx` float32
(~9 KB at this grid) against ~12 ms of tensor work.

**Beware quoting absolute fps across sessions.** The same 300-frame window
measured 79.5 fps in one session and 67 fps in another on the same machine, so
run-to-run drift is ~15% and swamps anything this refactor could have cost. Only
the back-to-back paired comparison above is load-bearing; the standalone numbers
are not comparable to §14's or §16's unless taken in the same sitting.

### The clamp had to stop being computed twice

`ChannelPlan` (from `plan_channel_stream`, decode-free) holds the resolved
window, geometry and per-channel gating. It exists because a streaming consumer
must size a ring buffer *before* the pass starts, and the obvious way to do that
— let the caller work out `n` itself — puts a second copy of the
`start`/`n`-vs-`n_frames` clamp next to the pass filling the buffer. That is the
T17/T11 shape again: two pieces of state answering one question, diverging
exactly at the clip end where the window gets truncated.

### Three things review caught, all in the new seam rather than the moved code

- **The empty-channel guard was one layer too high.** `live_channel_source`
  raises on an empty channel set; `plan_channel_stream` is the entry point a
  streaming worker calls *directly*, so the new path routed around it. A pass
  with no channels still decodes and preprocesses every frame and yields frames
  with nothing in them — which `StreamBuffer` cannot even be constructed to
  hold. The check now lives in the plan as well.
- **A negative index does not raise in numpy, it writes the end of the array.**
  The consumer's `out[k][i - plan.start] = plane` is safe by construction, and is
  now checked anyway: the failure mode of that invariant breaking is a frame
  landing silently at the wrong *time*, not a crash. `StreamBuffer.append`
  already refuses a non-contiguous index for this reason, and the two consumers
  of one generator should not disagree about how far to trust it.
- **`ChannelPlan` is comparable but deliberately not hashable.** `tiles` holds
  dicts, so the frozen dataclass's generated `__hash__` raised
  `unhashable type: 'dict'` from inside a tuple — naming neither the class nor
  the caller. `__hash__ = None` makes the error name `ChannelPlan`. Equality is
  the operation a surface actually wants ("did the plan change? then restart").

**`done` is counted before the yield, not after the resume.** Counting on resume
undercounts by one whenever a consumer closes the generator instead of draining
it — and the log line for a cancelled pass, which is exactly when that happens
(a knob edit supersedes the running extraction), is where the lie would land.

---

## 23. Batch Q slice 2: the continuous surface, and four ways a live plot lies

The surface now has a third action, **Live ▶**, that runs `LiveStreamWorker`
forward to the end of the clip and updates the hosted explorer in place at ~1 Hz
instead of extracting a window and blocking. The wiring itself is small; almost
all of the work was in the failure modes, which are all the same shape — **a plot
that renders and does not mean what it appears to** — and all four were reachable
in the first working cut.

**The slice boundary in the plan was wrong, and the symptom was "no user-visible
payload".** Batch Q slice 2 was specced as file-local to
`live_scalogram_surface.py`. It is not: `ScalogramExplorer` derives `T`, the
regions, `freqs`, the cube cache and the detect array **at construction**, and
has no in-place update — `capture_view_state`/`apply_view_state` exist precisely
because rebuilding was the only way to change the data. Cutting the slice at the
file boundary produced a stream worker whose windows nothing could render, i.e. a
progress readout. **This is the fourth time the "· file" locality annotation has
been wrong** (see the note at the end of `todo.md`). The new signal worth
recording: *if a slice has no observable behaviour, the boundary is in the wrong
place* — that is cheaper to notice than re-deriving the file list.

**Trap 1 — a one-frame window is not empty, and renders.** `LiveStreamWorker`
refuses to serve a window of *zero* rows (the "empty is not vacuously valid"
rule). The first tick of a pass routinely lands **one or two** frames, which is
one row above that guard: the explorer then built its entire time axis on it — a
Morlet transform over a single sample, detection window D of 1, a scrub with one
position. All of it drew. The gate is `min(requested, MIN_CAPACITY)`, never a
flat floor, because a request shorter than the floor is a short window the user
asked for. *Known remaining wart:* this lives on the consumer, while the
zero-row guard it extends lives on the producer — slice 3 should move both into
`request_latest`.

**Trap 2 — the cube cache is keyed by `(region, channel)` and carries no span.**
Replacing the channels under a fixed geometry means a cube built over the
previous window is **indistinguishable from a current one**. A `_data_gen`
counter now stamps both the cache and the in-flight worker, and a cube whose
stamp is stale is dropped. Had this been missed the symptom would have been a
scalogram from a span the user had already left — and it would have been blamed
on the transform.

**Trap 3 — and the generation guard alone causes a livelock.** `_ScalogramWorker`
**cannot be cancelled**, and `_request_cube` refuses to launch while one is in
flight. So if a cube takes longer to build than the update interval, every cube
arrives stale, is dropped, and is relaunched only to be dropped again: the
scalogram **never appears at all** rather than appearing late. Cubes run to
"several GB and tens of seconds" on a full-length clip, so this is the normal
case, not the edge. The fix is backpressure, not a faster tick: the surface skips
an update while `explorer.is_building()`, so the display settles to the rate the
transform can actually sustain. Measured on a 1500-frame synthetic pass: one
build plus three in-place updates, each landing between builds.

**Trap 4 — the cursor drifts backwards while appearing to sit still.** The frame
cursor is an index into the *window*, and the window slides. Carrying the index
across an update walks the cursor backwards through the clip at the eviction
rate. It is now carried by **absolute video frame**, clamped when the window has
slid past it — with a `followed` case (cursor on the newest frame keeps riding
it) because that is the live default. `follow_latest()` exists only because a
freshly built explorer sits at frame 0, so without it the live view pins to the
**oldest retained frame** — the exact opposite of live, and it looks stationary
rather than wrong.

**Two lifecycle notes.** A live pass runs to the end of the clip, so unlike the
extract it does not self-terminate in seconds: `hideEvent` now stops it, or a tab
switch leaves the decoder held and `extract`/`process` blocked for minutes. And
the request timer is stopped at the *stop request*, not only at `_end_stream` —
`cancel()` sets a flag the worker notices at its next frame, and the timer would
keep parking requests across that gap.

**`truncated` is not known until the pass ends**, so a live `ChannelData` carries
`False` during the run and the final window is re-rendered once the answer is in.
The status line alone was not enough: anything reading `cd.meta` — detection
included — would have seen a clean window. Same obligation as §15.

## 24. Batch Q slice 5: the continuous-plots split, and the constant it disturbs

The remaining half of the user's ask: the scalogram may refresh at ~1 Hz, but
the selected-channel trace and the detection sweep should be **continuous**.
They cannot simply be ungated — `_show_live_window`'s `is_building()` gate is
what stops every cube arriving stale (§23 trap 3). The shape is a split of
`set_channel_data` into a fast path and the cube rebuild. This section records
the measurement that sized the fast path, because it landed somewhere other than
where the plan (and this author) expected.

### The measured tiers

`F` = 24 scales, fps 24, best-of-N, synthetic `(T, B)` float32. `B` = 377 is
block 16 on a replicate; `B` = 29 is block 64.

| | T=5000, B=377 | T=30000, B=377 |
|---|---|---|
| pooled mean `blocks.mean(axis=1)` | 0.34 ms | 3.2 ms |
| **`np.percentile(blocks, 99)`** | **15.4 ms** | **53.0 ms** |
| pooled Morlet `morlet_power((T,))` | 1.9 ms | 8.1 ms |
| density band sum `cube[i:j].sum(0)` | 5.9 ms | 91.2 ms |
| **per-block cube `morlet_power((T,B))`** | **444 ms** | **6020 ms** |

### The pooled Morlet is NOT the expensive thing, contrary to the obvious read

`_rebuild_selected_views` is docstringed as "cheap per-frame views (no cube
needed)" while visibly calling `morlet_power`, which invites the inference that
the transform is the hidden cost on the fast path. It is not: **8 ms at 30k
frames**, the cheapest non-trivial item measured. It transforms a single `(T,)`
pooled series, so it carries none of the `B` factor that makes the cube 6
seconds. The docstring is accurate and the suspicion was wrong. Recorded because
it is a natural mistake to make twice — `morlet_power` appears in both the fast
path and the slow one, and only the `(T, B)` call site is expensive.

(The pooled-Morlet row is also the noisiest in the table: repeated runs at
identical `T` spread ~2.5x, which is §22's session-drift warning showing up
within a single session. It does not matter here — every reading is under 25 ms
— but do not quote that row to two significant figures.)

### The cost is `np.percentile`, and it re-opens a closed decision

`_ov_scale = max(float(np.percentile(blocks, 99)), EPS)` is **6-45x the pooled
mean** and dominates the fast path, because it is a full sort of `T*B` floats.
It is also the line at `scalogram_explorer.py:1366` carrying the comment that it
is frozen at rebuild time *"so scrubbing (which redraws every frame) never
re-percentiles the array in the hot path"* — i.e. §20's lesson. **Running the
fast path at 10 Hz re-percentiles 10x/sec and reintroduces exactly what that
comment exists to prevent**, one layer up.

So `_ov_scale` stays on the SLOW cadence. It is an overlay colour scale, not
data, and holding it steady is independently the better behaviour: a
normalization that drifts every tick makes the same channel non-comparable by
eye across time — the same argument that refused Batch G's zoom margin (§19).
Same-looking pixels must mean the same value, or the overlay is decoration.

### 10 Hz holds, but only because per-block data stays WINDOWED

Fast path windowed (`T~5000`, minus the percentile): **~8 ms, 10 Hz with wide
headroom.** With the percentile: ~24 ms, still 10 Hz. But at whole-video
`T=30000` the same path is ~155 ms — **6.5 Hz, and the target fails silently.**

This makes the plan's per-frame/per-block split (todo.md, Batch Q REDESIGN) a
**precondition of the cadence, not just of memory**. It was justified there on
footprint alone: per-frame series are ~120 KB at 30k frames while the `(F,T,B)`
cube is several GB. The timing is a second, independent reason for the same
boundary, and the more fragile one — a future change that lets the density
window grow toward clip length would break the frame rate long before it broke
the budget, and would do it without any error.

### The registration hazard, which is what the design call was actually about

todo.md framed the split's risk as "a scalogram from an older span beside traces
from a newer one … must be visibly marked". That understates it. `ScalogramPlot`
and the trace plots map their columns across the **same widget width**, stacked,
and are read down a column — so a cube over `[0, 8000)` beside traces over
`[0, 9500)` puts **frame 7800 directly above frame 8300**. That is an x-axis
misregistration, not staleness, and a badge does not fix it: it yields a
correctly-labelled plot that still invites reading a burst against the wrong
trace value.

Decided with the user: **pad the cube onto the new span's axis and paint the
uncovered tail as unexamined**, reusing slice 4's three-state coverage
vocabulary rather than inventing a staleness marker. Registration is then exact
everywhere, and the lag becomes a geometric fact — the scalogram's frontier
visibly trailing the traces' — instead of a claim in a badge. This is the same
remedy class as slice 4's coverage mask and §10 traps 7/8: make "nobody looked"
a distinct painted state instead of letting it borrow the appearance of a
computed zero.

Cost of that choice, both real: `_sg_cache` is keyed `(region, channel)` with no
span, which is exactly why `set_channel_data` clears it wholesale — keeping a
cube across a span change means storing its span alongside it. And
`set_scalogram` derives its axis from `matrix.shape[1]`, so the pad needs
handling in `_heatmap`'s normalization and in the `matrix.sum(axis=0)` cursor
readout.

### The 10 Hz rate was justified on the wrong half, then measured on the right one

Raising `_WINDOW_REQUEST_HZ` from 1.0 to 10.0 was argued from the table above —
the GUI-side plot cost. But the comment being replaced was not about that. It
said *"a window is a copy of hundreds of block frames"*: a claim about the
PRODUCER, on the worker thread, which is also the decode thread of a pass
already measured at 0.75x realtime once `appearance` is enabled. Answering a
different question than the one the old comment asked is how a regression gets
argued into place, so it was measured too.

`StreamBuffer.window` is a deliberate copy (contiguity for the transform) and
takes the `np.concatenate` seam branch whenever the span wraps the ring. Two
channels per request, filled past capacity so every read wraps:

| trailing window | block grid | MB/request | ms/request | worker duty at 10 Hz |
|---|---|---|---|---|
| 600 | 19x19 | 1.7 | 0.04 | 0.0% |
| 2000 | 19x19 | 5.8 | 1.09 | 1.1% |
| 5000 | 19x19 | 14.4 | 1.81 | **1.8%** |

**Negligible, and the estimate that worried about it was right on volume and
wrong on impact**: ~144 MB/s of copy at the worst case, against a memcpy rate
near 8 GB/s. 10 Hz costs the decode loop under 2% even at the largest window and
finest grid measured. The old comment's instinct was sound and the number is
simply small.

### The hatch's first cut inverted its own purpose, and a dense test could not see it

`_hatch_unexamined` first derived the uncovered columns from `np.unique(col)` —
the columns that actually RECEIVE a frame. That is wrong whenever the covered
span holds fewer frames than it spans pixels: **64 frames over 300 columns leave
236 covered columns frameless**, and every one of them was painted as
unexamined. Real per-block data rendered as "nobody looked" — this mechanism's
own failure mode, inverted, and strictly worse than the state it was built to
prevent, because it destroys signal rather than merely flattering it.

Coverage is a property of the SPAN, so it is now computed from
`[axis_off, axis_off + T)` mapped into column space, with a **ceiling on the
right edge** — flooring it drops the last partial column and leaves a few
columns of real data hatched at the frontier, the same lie made narrower.

**Two things worth carrying forward.** The bug passed the whole suite: both
registration tests used spans far longer than the render width (3000 frames into
~450 columns), where every covered column happens to receive a frame. A test
whose data is denser than the failure requires cannot see it, and "covered" and
"received a frame" are only the same question in that regime. And the reachable
case is not exotic — any live window shorter than the plot is wide (~600 px),
which is every window early in a pass.

## 25. Batch S slice 5: routing, and a rate that would have drifted silently

Slice 4 put each replicate's clip inside its home, so a clip stopped being
something an operator points at and became something that is either there or
not. `core/source_route.py` is the one place that asks, and `resolve_source`
is derived **per call, never recorded** — a pointer file is an existence claim
that outlives the thing it points at, which is §13/§15's rule. The cost is a
head+tail read and a stat per clip, ~1.4 ms, which is what `verify_manifest`
was built cheap for.

**Three outcomes, and the interesting boundary is between the last two.**

| situation | route | why |
|---|---|---|
| no manifest for this video | source, with `reason` | clips are optional; an uncut corpus must still run |
| a manifest that is not this video's | source, with `reason` | for *this* video no manifest exists |
| this video's manifest, not verifying | **raise** | ruled 2026-07-21 |

The middle row is the one that is easy to get wrong. Manifests are named from
the video's basename, so two sources called `GX010047.MP4` in different session
directories map to one manifest path under a shared `--clip-dir`.
`verify_manifest` **cannot** catch it: it validates the manifest against the
source *the manifest names*, which is present and unchanged, so it passes and
one session's pixels get attributed to the other's detections. `cli/pretranscode`
and `core/framecount` had each already discovered this separately; the identity
check is now one function (`framecount.manifest_describes`, made public) rather
than three near-copies. A *discovered* manifest that fails it is somebody else's
file; an **explicitly named** one that fails it is a usage error and raises,
because naming it asserts it applies.

**The rate substitution the streaming path never inherited — found in review,
and it is §3 trap 2 arriving by a new road.** `ClipAtlasSource` seeks by
dividing a frame index by the fps it was given. `live_channel_source` has always
replaced OpenCV's float with the manifest's *rational* rate before extraction
for exactly this reason: 24.0 standing in for 24000/1001 lands 3 frames early by
frame 11000, yielding a window of the right length **from the wrong place**,
silently. The Batch Q streaming path builds its own meta through
`synth_live_meta` and so never received that fix — it did not matter while
nothing streamed from clips, and would have started mattering the moment the new
checkbox was ticked. The route is therefore resolved *before* the plan, because
on a clip route it supplies the rate the plan is built on. **A fix applied at one
of two entry points is not applied**, and the second entry point was written
after the first and looked self-contained.

**The GUI is opt-in, and the checkbox is not persisted.** Below `lossless` a
clip and a live crop are not the same pixels (§10), so which set a session's
numbers came from is a claim the user makes, not one the filesystem makes for
them. The other knobs on that row *are* persisted, and the distinction is worth
stating: a restored downsample is the same measurement at a remembered setting,
while a restored source is a different set of pixels re-armed silently on a
later session. The other knobs answer *how*; this one answers *of what*.

For the same reason a stale manifest **refuses the pass** in the GUI rather than
falling back. The tick is a claim; quietly answering it with the other pixel set
is the exact failure the opt-in exists to make visible.

**The cost model half: `fixed_s` IS the decode floor, and the cut moves it 25x.**
`PassSample` now carries `source_kind` and `channels`, and `CostModel.fit`
**raises** on a mix rather than reading either as scale — which biases the knee
`s* = √(F/M)` toward "downsampling is free", the one direction §6 says this model
must never err in. Three details:

- The token is `clips:<provenance_key>`, not a bare `"clips"`. Two cuts at
  different quality are different pixels and a different floor.
- `channels` was **already recorded** on `ScalePass` for this exact reason (§11,
  1.59x) and was being dropped on the way to `PassSample`, so the axis the code
  had identified was silently mixed anyway. Fixing only the axis the plan named
  would have shipped a guard that refuses one contamination and permits its
  neighbour.
- Empty `channels` mixes with anything: it means *unrecorded*, not "no
  channels". `source_kind` gets no such escape, because its default is a claim
  that was true of every pass ever timed before routing existed.

**Its own trap, in the GUI.** `_cost_samples` was keyed `(scale, block)`. Once
two source kinds exist, re-sweeping the same scales after cutting clips would
*overwrite* the source samples entry by entry, leaving that regime with too few
scales to fit and nothing saying why. The regime axes had to go into the KEY,
not only into the grouping.

**Locality, eighth firing.** The spec named two things — `resolve_source(video)`
and `core/cost_model.py:89`. It took nine files: the new module, `framecount`
(one identity check to share), `shard`, both CLIs, `cost_model`, `scale_sweep`,
`stream_worker`'s caller and the surface. The tell generalizes from §24's: the
spec named a function to ADD and a field to ADD, and adding a field to a value
object is never local — every producer of that object and every consumer that
groups by it moves with it.

---

## 26. Batch S slice 6: retiring a geometry, and the cache that undid it

**The reversal this rests on.** Slice 3 ruled that marks survive a box move —
stale, grayed by `TrackStamp`, but present. The user reversed it: they survive
the *deletion*, not the *move*. The reason is stronger than staleness and it is
what makes a gray-out insufficient. After a move there is no reason to believe
the marks under the box were ever centred on the replicate it now names — a
moved box can land on a different animal, or on nothing. Carrying them forward
as current is **misattribution**, the failure class the per-region track split
(slice 2) exists to prevent, not the milder "computed under old settings" that
the stamp machinery handles. So the fileset is retired *with its rectangle*.

**Home-authoritative geometry made a four-file slice a two-file one.** The
retired rectangle lives at `<home>/old_NNN/geometry.json`, beside the files it
describes, rather than as a `retired: []` list in `rois.json`. Two consequences
fell out that the spec had not predicted:

- **`rois.json`'s schema does not change**, so the per-video box record keeps
  having no generation concept — which is also what keeps the second copy of
  the frac from existing (the duplication T11, T17 and T34 each deleted).
- **The three stores never learn generations exist.** They write to the home
  *root* through `home_path`, and retiring moves that fileset down and out from
  under them. The current generation is deliberately **unnumbered**: it sits at
  the root and takes a number only when superseded.

The generalization: *putting a new concept BELOW the layer that would otherwise
have to know about it is what makes it local.* That was a property of the
ruling, not of the implementation.

**The counter is derived (`max(existing) + 1`), and this departs from `next_id`
on purpose.** `next_id` must be persisted because a deleted box leaves no trace
in `rois.json` — the counter is the only memory of it. A retired generation
leaves a *directory*, which is itself the trace, so the filesystem enforces
monotonicity for free. The hazard is the same one a scale down (a reissued
generation would adopt a dead rectangle's marks), which is why it is read from
the listing and never from `len(generations)`.

**The real defect was in RAM, not on disk, and the retire was undoing itself.**
`LiveScalogramSurface.closeEvent` flushes its `WholeVideoTrack` on the way out,
deliberately — a replicate edit rebuilds the surface, and that flush is how an
accumulated whole-video pass survives the rebuild. After a retire it wrote
old-rectangle band power straight back into the home root the retire had just
emptied, where it now reads as the NEW rectangle's. **`TrackStamp` made the next
load refuse it, so it showed gray rather than lying** — detectable, not silent,
and still not acceptable: a retire that reverses itself one tab switch later is
not a retire.

The fix is `AppState.replicate_retired(int)` → `discard_replicate_track`, which
drops that replicate's in-memory track *without* flushing and re-reads the root.
Three things about it:

- **Per-replicate, not `rois_changed`.** `rois_changed` fires for any box edit,
  and discarding every in-memory track when one box moved would throw away a
  neighbour's accumulated pass — work the move did not invalidate. Non-active
  tracks are already flushed on handover, so the only one at stake is the
  active one, and only if it is the moved replicate's.
- **The restore path emits it too.** A restore is a swap, so it retires the
  current fileset on the way past; the in-memory staleness is identical.
- **`_sync_track` was the wrong reload route.** It reloads only as a side effect
  of pushing a stamp and returns early when there is no stamp yet (no channel
  data, or a surface that has not run a pass), leaving `_track` pointing at the
  object just dropped. `_activate_region` directly, with `_active_region`
  cleared first — which also makes that call skip its own outgoing flush, so
  the two requirements want the same assignment.

**`ghost_boxes` is a separate list on `FrameView`, and that is structural, not
stylistic.** The index carried by `box_grabbed` / `box_clicked` / `box_moved` is
a position in `boxes` that the tab uses to index `self.replicates`. A
non-interactive entry mixed into that list shifts every later index and silently
moves the wrong replicate. Keeping the ghosts apart means `_box_at` cannot see
them at all, so "retired rectangles are not selectable" cannot be regressed by
forgetting a flag.

**Two paint defects, both found in review.** `QColor.hue()` returns **-1 for an
achromatic colour**, so desaturating a ghost through HSV is undefined for a box
sidecar carrying `#ffffff` — replaced with a blend toward mid-gray, which is
total. And the dashed pen was being reused for the label text: a dash pattern
applies to glyph outlines too.

**Locality, ninth firing — the first to fire INWARD.** Predicted four files,
took two, for the ruling reason above. The hazard reappeared in the other
direction: the file the spec did not name held the same state in memory and
would write it again. **When a ruling makes a change local on disk, ask what
holds that state in RAM.** A storage layer can be moved out from under its
writers; it cannot be moved out from under a cache that will write again.

---

## 27. The teardown crash: a QThread outliving its widget, blamed on the machine

**It was ours.** For four sessions a full-suite run intermittently reported all
tests passing and then died at interpreter shutdown with `Windows fatal
exception: access violation` and a faulthandler dump. It was recorded as an
aggressive university endpoint blocker and repeatedly written off. Two real
defects in `ScalogramExplorer` let a cube thread outlive the widget that owned
it, and **Qt destroying a running `QThread` is an access violation, not an
exception** — which is exactly why it appeared as a clean pass followed by a
crash, with no failed test to point at.

**The seam both defects come from.** `_ScalogramWorker.run` ends with
`done.emit(...)`, and `done` is a queued connection. So `_on_cube_ready` runs on
the GUI thread *while the thread that produced the cube is still unwinding* —
`run()` has not returned and `finished` has not fired. Everything below follows
from that overlap.

**Hole 1: `self._worker` is not the set of threads to join.** `_on_cube_ready`
clears `self._worker` and then calls `_request_cube()`, which may launch the
next cube straight into that attribute. `closeEvent` waited on `self._worker`
only — so in that window it waited on the *new* thread and let Qt destroy the
old one, a child `QObject` of the widget, while it was still running. The fix
separates the two questions that had been conflated: `self._worker` still
answers *"is a build in flight"*, and a new `self._threads` answers *"what must
be joined before this widget dies"*. Retirement from `_threads` is connected to
`finished` **before** `deleteLater`, so the ordering is finished → untracked →
freed, never freed → waited on.

**Hole 2: `close()` is not deletion.** A closed widget keeps receiving queued
signals until `deleteLater` gets an event-loop turn. A `done` arriving in that
gap re-entered `_request_cube` and started a thread *after* `closeEvent` had
joined every thread — undoing the join it had just performed. Fixed with a
`_closed` latch set first thing in `closeEvent` and checked in `_request_cube`.
This one is routine rather than exotic: the live surface closes an explorer on
**every** pass restart (`_swap_explorer`), which is the single most travelled
path in the app.

**The debugging lesson, which is the durable part.** Four sessions of
deselect-plus-isolate "proved" the crash was not ours. It proved only that the
crash needs full-suite *context* — which is precisely what a timing-dependent
thread race needs, so that evidence was consistent with the bug the entire time.
"The diff did not touch `gui/stream_worker.py`" was equally true and equally
irrelevant: the leaked threads are `_ScalogramWorker`s, and the `_serve_pending`
frame the dump named belongs to a *different* thread, because faulthandler
prints every thread's stack, not only the faulting one. **An intermittent crash
that survives repeated "unrelated diff" arguments is evidence about
reachability, not about ownership.**

**What actually cracked it: instrument for the PRECONDITION, not the crash.**
The crash could not be reproduced on demand (~35 clean full runs against one
firing). But its precondition — a `QThread` still running when the test that
started it ends — is deterministic, cheap to observe, and was never checked. A
throwaway pytest plugin wrapping `QThread.start` and testing `isRunning()` in
`pytest_runtest_teardown` found **14 tests leaking a running thread on the first
run**. Prefer this whenever a fault is rare but its enabling condition is not.

**How the fix is verified, and the limit on that claim.** Not against the crash,
which will not reproduce. Against the precondition: leak events went 45 → 28,
and — the part that matters — **every remaining entry is a finished-then-deleted
thread, with none still running**. Both fixes carry regression tests
(`tests/test_explorer_threads.py`) confirmed to fail with their own fix reverted
and only their own, driven through the real `closeEvent`, since the bug lives
entirely in that method's ordering against a queued signal.

**A likely aggravator worth recording**: the user asked whether running the GUI
app in the background could cause it. It cannot cause an access violation *in
another process* — separate address spaces — but memory pressure (the cube cache
budget is 6 GB) and decoder contention are exactly the kind of perturbation that
turns a latent thread race from never-firing into occasionally-firing. That
would explain both the original "roughly half of runs" and the ~1-in-35 measured
later, without any of it being environmental in the sense first assumed.
