# Known issues

Open problems and deferred design decisions we intend to revisit. Each entry
lists what you observe, why it happens, what to do in the meantime, and the fix
we expect to make. Resolved items move to the bottom.

---

## 1. CLAHE has a replicate-boundary artifact (active)

**Severity:** high on difficult footage. **Default avoids it** (`normalize` now
defaults to `zscore`).

**Symptom.** With `normalize = clahe`, blocks along a replicate box's edge —
especially the bottom row — report large phantom speeds and threshold crossings
that do not correspond to real motion. On the reference clip, replicate 23 peaked
at **861 px/s with 60 crossings** under CLAHE versus **48 px/s and 0 crossings**
with CLAHE off, while the actual frame translation there was ~0.08 working pixels.

**Cause.** CLAHE runs per replicate box, per frame, on the hard crop
(`core/preprocess.py:_normalize`). Its `tileGridSize=(8,8)` edge tiles have
truncated histograms that `clipLimit` amplifies, and the dense-flow solver is then
handed reflected (mirror) padding at that inflated edge — a mirror is the
worst-case input for a translation estimator. Because the tile grid shifts when
any surrounding context is added, CLAHE also perturbs *every* block in the box,
not only the boundary.

**Workaround.** Use `zscore` (the default): a global per-frame mean/std rescale
with no tiling, so it has no boundary artifact. Do not trust CLAHE caches for
threshold evaluation until the fix lands.

**Planned fix.** Two candidates, to be evaluated on replicate 23:
1. Reduce per-box CLAHE to a single tile (`tileGridSize=(1,1)`), removing the
   truncated-edge-tile mechanism; or
2. Replace the reflected flow-support margin with a real source-pixel halo
   (see issue 2), so the solver sees real gradients at the core boundary.

**References:** `core/preprocess.py:_normalize`, `core/pipeline.py:_pack_flow_atlas`.

---

## 2. Flow-support halo is synthetic, not real source context (active)

**Severity:** medium; the root cause behind issue 1's severity.

**Symptom.** Flow at the core boundary of every replicate is computed against
mirrored edge pixels rather than the real surrounding image, so boundary blocks
are less reliable than interior blocks even with `zscore`.

**Cause.** `_pack_flow_atlas` pads each cropped box with `cv2.BORDER_REFLECT_101`
before solving; the "synthetic halo" contains no real neighbourhood. Supplying
~8 real working pixels of context around the box drops the replicate-23 peak to
~53 px/s with 0 crossings.

**Design tension (why it is not yet done).** A real halo large enough to serve
Farnebäck's support (up to the winsize, ~32 working px ≈ ~130 source px at the
reference scale) reaches into neighbouring replicate boxes, which can be as little
as ~22 source pixels apart. That would let one individual's motion contaminate a
neighbour's cached core through both flow and normalization, breaking the
replicate-isolation contract precisely when animals are adjacent — the case that
matters most. The overlap check in `core/replicates.py:validate_replicates` must
stay.

**Planned fix.** Decode a small real halo (sized to the CLAHE/normalization need,
~one tile / ~8 working px — not the full flow-support size), clipped against
neighbours by a Voronoi rule: a halo may read any background pixel including the
gap between boxes, bounded by the perpendicular bisector to an adjacent box, and
never crosses into another box's interior. Fall back to reflected padding only on
the strip facing a genuinely adjacent neighbour. Cache only the core.

**References:** `core/pipeline.py:_flow_support_pixels`, `_flow_atlas_geometry`,
`_pack_flow_atlas`; `core/replicates.py:validate_replicates`.

---

## 3. Auto-discovery ROI area is still unweighted (active)

**Severity:** low; the manual replicate workflow is already fixed.

**Symptom.** `extract_rois` gates auto-discovered regions on `min_area_blocks`
counting each block as a whole unit, so a thin partial edge block counts the same
as a full block — the same class of bug fixed for `roi_detection`.

**Cause.** `extract_rois` uses `cv2.CC_STAT_AREA` (raw block count) rather than the
valid-area weights now available from `core.replicates.block_weight_plane`.

**Workaround.** None needed for the manual replicate-box workflow, which uses the
weighted `roi_detection` path. Only auto-discovery is affected.

**Planned fix.** Weight component area by `block_weight_plane` in `extract_rois`,
mirroring `roi_detection`.

**References:** `core/roi.py:extract_rois` (`min_area_blocks`).

---

## 4. Cache precision: float16 is the floor (caution / settled decision)

Selecting `dtype = float32` doubles cache size on disk and in RAM versus
`float16`, for precision the `u`/`v`/`speed` arrays do not need — a block flow
estimate carries order 5–10% intrinsic uncertainty, and float16's ~0.05% relative
quantization is already far below it. Band power is stored as `float32` regardless
(it can exceed float16's range). The flow tab shows a red warning when `float32`
is selected. Prefer `float16` unless you have a specific reason.

Going **below** float16 was considered and rejected — do not revisit without new
reasons:
- 8-bit float (E4M3) tops out at 448 and would clip the fastest block speeds
  outright (edge artifacts alone reach ~860 px/s); E5M2 has range but ~25%
  relative error.
- Its ~6–12% relative step is comparable to the real measurement uncertainty, so
  it adds noise where float16 adds none — worst in the low-speed baseline the
  standardized noise references depend on.
- No native numpy float8 in the current stack (`ml_dtypes` is not installed), and
  the cache is already zstd/lz4-compressed, so the real on-disk saving is modest.
- The correct lever for smaller caches is block size (scales storage ~quadratically
  and trades resolution you can reason about), not sub-float16 precision.

**References:** `core/config.py` (`FeatureConfig.dtype`), `gui/tab1_flow.py`
(`OPTION_WARNINGS`); `docs/decisions.md` ("Cache precision stops at float16").

---

## 5. Tensor detection path is not yet accuracy-validated (caution)

**Severity:** medium — it is a usable tuning/triage tool, but not yet a validated
detector.

**Symptom.** The live tensor detection loop (Preprocessing (live) tab → *Process
whole video* → navigation strip) produces detections with no established
precision/recall on marked footage. Its tests cover the math (band power equals a
full-cube slice, windowed mean, region scoping) and the plumbing (window → process
→ navigate), not biological recall.

**Context, not a bug.**
- The whole-video pass is a full streaming structure-tensor solve (comparable to a
  flow pass per frame), so it takes minutes on a long clip. It is the one
  expensive step, paid once after tuning rather than up front. It writes nothing
  to `.cache/`; nothing is persisted between runs yet (see next-steps §11).
- Temporal denoise is forced off for windowed extraction and is unavailable on
  this path — it is stateful from frame zero and cannot be reproduced mid-clip.
  `registration`/`bg_subtract` also do not apply and are flagged approximated.
- Changing the frequency band or channel requires a fresh whole-clip pass; only
  the value band and detection window re-tune instantly (the band power is
  retained, the band sum is not).

**Workaround.** Verify detections by clicking into their windows (the loop is
built for exactly this). For a behavior where a validated ethogram is required
today, use the flow-cache Behavior tab.

**Planned work.** Accuracy validation against marks (next-steps §10), optional
channel sidecar persistence (§11), and the deferred question of the Behavior tab
consuming a tensor sidecar (§12).

**References:** `core/detection.py`, `core/channel_source.py`,
`gui/explorers/live_scalogram_surface.py`; `docs/decisions.md` ("The
tensor/scalogram path runs without a flow cache").

---

## Resolved

- **FFmpeg silently rounded odd-coordinate crops.** The ROI decode did not pass
  `exact=1`, so an odd crop origin shifted the window by up to a full source pixel
  — read by dense flow as real translation. Fixed by adding `exact=1` to the crop
  filter. `core/video.py:ReplicateVideoSource`.
- **Partial edge blocks counted as whole blocks in detection.** Clump size,
  passing-block count and strength treated a one-pixel-tall edge sliver as a full
  block, letting a row of slivers masquerade as a real clump. Fixed with
  valid-area weighting (`core.replicates.block_weight_plane`) in
  `core/roi.py:roi_detection` and the `gui/explorers/speed_explorer.py` clump readout.
