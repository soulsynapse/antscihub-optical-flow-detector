# Handoff: Optical Flow Standardization Additions

## Context

Kendrick's current optical flow pipeline computes $(u, v)$ fields, caches them, and then applies manually-tuned filters at analysis time to isolate behavior. The manual filters don't generalize well across replicates because they use absolute thresholds rather than self-calibrating against per-replicate reference distributions. This handoff lists standardization steps that are widely regarded as no-brainers in the flow literature — they throw away essentially nothing real, and they eliminate large sources of between-replicate variance that currently require re-tuning.

## How to use this handoff

**Add these where they make sense in the existing pipeline. Do not add them mechanically.** Read the current pipeline first. If a proposed addition:

- **Duplicates something already being done** (e.g., there's already an illumination correction step, or the cameras are already locked and don't need CLAHE) — skip it and note that it's already covered.
- **Would silently change results in a way the user can't see** (e.g., baking a filter into cached data with no way to inspect what was filtered, or applying a per-frame normalization that would break temporal comparisons downstream) — push back before adding. Kendrick strongly prefers pipelines where he can see what each step is doing. Opaque "fixes" are worse than no fix.
- **Conflicts with an assumption downstream** (e.g., a spatial median on $(u,v)$ before a step that expects raw vector noise for uncertainty estimation; unit conversion before a step that assumes pixel coordinates) — flag it, don't paper over it.
- **Requires re-running expensive computation the user may not have budgeted for** (specifically: adding CLAHE or backward flow means re-doing flow on the full corpus) — surface this explicitly with an estimate before starting.

Push back is expected. If something in this list would make the pipeline worse or more confusing, say so. Kendrick's preference is "correct me directly" over "quietly comply."

Each item below is tagged with **when** it must run relative to the cache. Respect those tags — several of these are non-negotiable in their ordering.

---

## The additions

### 1. Camera capture settings (upstream of everything)

**When: before capture. Not retrofittable.**

Check that `rpicam-vid` (or the libcamera equivalent) is invoked with fixed exposure, gain, and white balance across the fleet. Auto-anything at capture time introduces temporal intensity changes that flow computation interprets as motion, and nothing downstream can fix it. Flags to verify:

- `--awb off` or fixed AWB gains
- `--exposure off` with fixed `--shutter`
- Fixed `--gain` (analog gain)

If Kendrick already has this locked as part of prior rpicam-vid tuning, confirm and move on. If not, this is upstream of every filter in this document.

### 2. CLAHE preprocessing

**When: before flow computation. Must be at (re-)cache time. Not retrofittable to existing cache.**

Contrast-Limited Adaptive Histogram Equalization normalizes local contrast and removes slow spatial illumination gradients (vignette, uneven IR field). Standard in biological imaging.

- OpenCV: `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`
- Apply identically to every frame within a replicate. Do **not** let it re-fit temporally per frame in any way that would compete with the temporal derivative flow depends on.
- If the existing pipeline already has an illumination correction step (background division, rolling background subtraction, etc.), do not stack CLAHE on top without confirming — it may over-correct.

**Push-back criterion:** if the cameras are already locked and the micronests are illuminated uniformly, CLAHE gains are marginal. Ask before initiating a full re-cache just for CLAHE.

### 3. Backward flow + forward-backward consistency

**When: backward flow at cache time (requires video access); FB consistency check post-cache.**

Compute flow both directions: forward $\mathbf{F}: t \to t+1$ and backward $\mathbf{B}: t+1 \to t$. Store both. At analysis time, reject any pixel where:

$$\|\mathbf{F}(\mathbf{x}) + \mathbf{B}(\mathbf{x} + \mathbf{F}(\mathbf{x}))\| > \tau$$

Reasonable $\tau$: 1 pixel absolute, or a small fraction (5–10%) of $\|\mathbf{F}(\mathbf{x})\|$, whichever is larger.

This is the single highest-value addition. Catches occlusions, illumination artifacts, and matching failures automatically. Roughly doubles cache size and flow compute time.

**Push-back criterion:** if the cache is very large and re-running is a multi-day operation, discuss with Kendrick before kicking it off. It is worth doing, but he should know the cost.

### 4. Structure-tensor / cornerness map

**When: at cache time if possible (needs frame access); can be recomputed later from video if needed.**

For each frame, compute the smaller eigenvalue $\lambda_{\min}$ of the smoothed structure tensor $J = \nabla I \nabla I^\top$. Cache it as a float32 map per frame. Used at analysis time to mask out flow vectors in low-texture regions where the aperture problem makes flow numerically fabricated.

- OpenCV: `cv2.cornerMinEigenVal(gray, blockSize=3, ksize=3)` gives $\lambda_{\min}$ directly.
- At analysis time, mask where $\lambda_{\min}(\mathbf{x}) < q_{\alpha}(\lambda_{\min})$ for some per-frame quantile $\alpha$ (start with $\alpha = 0.25$).

Storage cost: one float per pixel per frame. If storage is tight, `float16` is fine here.

**Push-back criterion:** if the analyzed ROI is always a densely textured colony region (no smooth walls or empty substrate ever in-frame), this may be redundant. Confirm before adding.

### 5. Noise-floor-relative magnitude thresholds

**When: pure post-cache. Just needs the cached $(u,v)$.**

Replace any absolute magnitude threshold (`|v| > 0.5 px/frame`) with a per-replicate calibrated one:

- Identify a quiescent baseline period (or a static ROI outside the colony) and measure the 99th percentile flow magnitude $m_{\text{noise}}$.
- Threshold at $k \cdot m_{\text{noise}}$ with $k \in \{3, 5\}$.
- If no clean quiescent period exists, estimate $m_{\text{noise}}$ from a low quantile (e.g., 25th percentile) of the magnitude distribution *within* the ROI per frame — behavior is spatially sparse, so the bulk of the distribution is noise floor.

Do not bake this into the cache. It's a tunable analysis-time filter.

**Push-back criterion:** none obvious. This is nearly free and self-calibrating. Add it.

### 6. Spatial median filter on $(u, v)$

**When: pure post-cache. Analysis-time filter.**

3×3 or 5×5 median filter applied separately to $u$ and $v$ channels. Removes vector outliers while preserving motion boundaries. Documented as doing much of the work in "sophisticated" variational flow methods (Sun, Roth & Black 2010).

- OpenCV: `cv2.medianBlur(u_f32, 3)` — note: needs uint8 or float32.
- Apply lazily at analysis time, not into the cache. Kendrick may want to compare filtered vs unfiltered.

**Push-back criterion:** if the pipeline has a downstream step that estimates uncertainty from local vector noise (e.g., a Kalman-filter-style variance estimator), a median filter upstream will bias that estimate. Flag it.

### 7. Physical unit conversion

**When: pure post-cache. Analysis-time metadata operation.**

Store per-replicate calibration (px/mm) as metadata alongside the cache, not baked into cached values. Convert to mm/s at analysis time, and optionally further normalize by body length to get body-lengths/s for cross-caste or cross-species comparability.

- Do **not** convert cached values in-place. If a calibration turns out to be wrong for one replicate, in-place conversion loses the original.
- Keep pixel-space cached values as ground truth.

**Push-back criterion:** if the cache metadata schema doesn't currently support per-replicate calibration fields, adding this properly means schema changes. Surface that.

---

## Summary of cache-time vs post-cache

| Item | When | Retrofittable? |
|---|---|---|
| 1. Camera settings | Before capture | No |
| 2. CLAHE | At (re-)cache | No (requires re-flow) |
| 3. Backward flow | At (re-)cache | Yes but expensive |
| 3. FB consistency check | Post-cache | Yes (given backward flow) |
| 4. $\lambda_{\min}$ maps | At cache (preferred) | Yes (needs video) |
| 5. Noise-floor thresholds | Post-cache | Yes |
| 6. Spatial median | Post-cache | Yes |
| 7. Unit conversion | Post-cache (metadata) | Yes |

## General principle for the cache

Cache the most raw, most information-preserving intermediate that's expensive to recompute. Defer everything cheap or tunable. Never cache:

- Thresholded or binarized outputs
- Magnitude-only (direction is information)
- Unit-converted values
- Anything with a parameter Kendrick might want to sweep

## Final note to the agent

If, while reading the existing pipeline, something in this handoff looks like it would conflict, be redundant, or introduce opacity — stop and raise it before implementing. Kendrick would rather have three items added correctly and one item flagged as "this doesn't fit your pipeline because X" than seven items added mechanically.
