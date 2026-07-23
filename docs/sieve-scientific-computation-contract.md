# antscihub-SIEVE Scientific Computation and Extension Contract

**Signal Isolation for Ethological Video Events**

This is the scientific companion to the antscihub-SIEVE system handoff. SIEVE is
not a tensor detector with extension points. It is an extensible system for
asking:

> If video evidence is sampled or transformed across time, value, function,
> resolution, and spatial aggregation, which representation best isolates a
> behavior?

There may be many valid answers, including signals not anticipated when the
system is built. Motion, a brief color display, a texture transition, a spatial
relationship, a learned embedding, or a conjunction of these are peers. The
architecture must make an unanticipated signal implementable without modifying
the executor, CLI parser, artifact store, or optimizer.

This document contains two kinds of contract:

1. **Platform contracts** define typed artifacts, extension registration,
   provenance, validity, composition, and conformance. They apply to every
   SIEVE pipeline.
2. **Reference-profile contracts** define one reproducible implementation,
   `sieve.profile.tensor_morlet.v1`. Its equations are normative only when that
   profile or one of its nodes is selected. It is a supplied method, bootstrap
   implementation, and conformance fixture—not the definition of SIEVE.

Foundational methods motivate reference nodes, but unspecified library defaults
and superficially similar implementations are never silently equivalent.

## Part I: platform and extension contract

## 1. Scientific search space

SIEVE represents an analysis as a serializable, typed directed acyclic graph
(DAG). An experiment may vary any explicit scientific choice, including:

- **Time:** frame selection, temporal decimation, windows, lags, frequency
  bands, event duration, or sequence models.
- **Value:** intensity/color ranges, thresholds, quantization, normalization,
  weights, or learned decision boundaries.
- **Function:** the transform itself—motion, color, texture, morphology,
  spectral power, correlations, embeddings, or a method not yet designed.
- **Resolution:** decode scale, image pyramid level, feature-map resolution, or
  multiscale fusion.
- **Spatial aggregation:** pixels, masks, blocks, regions, components, pooled
  fields, or relationships between regions.

These are search axes, not five hard-coded stages. A graph can branch, combine
signals, retain multiple resolutions, and skip motion computation entirely.

The built-in tensor–Morlet profile is one graph:

```text
decoded samples
  -> canonical grayscale intensity
  -> optional normalization
  -> spatial and temporal derivatives
  -> regularized local structure-tensor solve
  -> block fields
  -> selected scalar channel
  -> complex Morlet power per block
  -> frequency-band power
  -> area-weighted spatial occupancy and components
  -> temporal gate
  -> event intervals
```

An equally valid color-display graph might be:

```text
decoded RGB planes
  -> calibrated color-space conversion
  -> red-feather likelihood mask
  -> morphology and minimum-area filtering
  -> per-region red-area fraction
  -> brief-flash temporal transform
  -> event detector
```

Nothing in that graph is adapted to look like tensor speed. Both graphs use the
same planner, streaming executor, provenance, cache, dataset, and optimization
interfaces.

## 2. Node and artifact model

Every computation is a registered node with named, typed input and output
ports. The minimum built-in node families are:

```text
media source       decoded planes, audio, timestamps, metadata
preprocessor       calibrated or transformed planes/fields
mask producer      selection, validity, or weight masks
feature extractor  pixel, block, region, frame, or sequence features
reducer            spatial or temporal aggregation and resampling
channel            named scalar/vector/tensor observations
transform          spectral, statistical, learned, or other representations
detector           scores, classes, gates, or candidate intervals
postprocessor      event merging, duration rules, confidence calibration
exporter           stable scientific artifacts and reports
```

Families describe interfaces and discovery; they do not impose a linear order.
A node may have multiple inputs and outputs, and graphs may fork and join.
Cycles are invalid. Stateful nodes declare bounded initialization and
finalization requirements so chunks do not change their meaning.

Each artifact descriptor includes:

```text
artifact_type and schema_version
axes and axis coordinates              # e.g. time,y,x or time,block_y,block_x
shape, dtype, and physical unit
value domain and signedness
spatial reference and asset coordinates
timebase and temporal alignment
validity representation
semantic labels for vector/tensor components
producer node id, implementation version, and parameter hash
upstream artifact identities
```

Array shape alone is never a scientific type. For example, a boolean selection
mask and a boolean validity mask are incompatible despite identical storage.

### 2.1 First-class media planes

The decoder exposes only the named planes requested by the graph. Plane types
may include encoded luma, linear-light luminance, RGB, alpha, and declared
native codec planes. Derived spaces such as HSV, HSL, Lab, or opponent-color
coordinates are preprocessing nodes, not hidden decoder effects.

Every color conversion records source primaries, transfer function, matrix,
range, chroma siting, output space, numeric range, backend, and implementation
version. Ambiguous metadata is an error or an explicit policy. A grayscale
motion run need not materialize RGB; a red-feather run must not be forced
through grayscale.

### 2.2 First-class masks and weights

SIEVE distinguishes at least four spatial artifacts:

- `validity_mask`: where a value is known and scientifically usable.
- `selection_mask`: which known samples satisfy a predicate.
- `weight_map`: finite nonnegative contribution weights.
- `scalar_field`: measurements such as color likelihood or change energy.

Nodes declare combination semantics. Validity normally combines by logical AND;
selections use an explicit AND, OR, XOR, or complement; weights use an explicit
multiply, minimum, maximum, or normalized blend. Masking never means silently
writing zero into the underlying signal.

A built-in color predicate should support a declared color space, inclusive or
half-open channel ranges, circular hue intervals that can wrap through zero,
soft or hard boundaries, optional reference-color distance, and versioned
morphology. Its output can feed area fraction, component geometry, persistence,
motion-within-mask, or any later node. Thus “red feather visible” can become a
channel, a mask over another channel, or one input to a multichannel detector.

An illustrative pipeline document is:

```yaml
pipeline_schema: sieve.pipeline.v1
nodes:
  - id: frames
    uses: sieve.media.video_planes.v1
    params: {planes: [rgb], downsample: 0.5}
  - id: hsv
    uses: sieve.color.convert.v1
    inputs: {image: frames.rgb}
    params: {output_space: hsv}
  - id: feather
    uses: sieve.mask.color_predicate.v1
    inputs: {image: hsv.image}
    params:
      hue_intervals_degrees: [[345, 360], [0, 15]]
      saturation: [0.55, 1.0]
      value: [0.20, 1.0]
      boundary: hard
  - id: red_fraction
    uses: sieve.reduce.mask_fraction.v1
    inputs: {selection: feather.selection, validity: feather.validity}
    params: {spatial_partition: {kind: blocks, size: 8, partial: weighted}}
  - id: flash
    uses: sieve.temporal.duration_filter.v1
    inputs: {signal: red_fraction.fraction}
    params: {minimum: 20ms, maximum: 250ms}
  - id: events
    uses: sieve.detect.threshold_events.v1
    inputs: {score: flash.score}
    params: {threshold: 0.12, merge_gap: 40ms}
outputs: [red_fraction.fraction, flash.score, events.intervals]
```

The values above are examples, not biological defaults. An experiment can vary
the color model, hue/value bounds, spatial partition, temporal function, and
detector, or replace any node with a learned alternative.

### 2.3 Channel shapes

A channel may be scalar, vector, tensor, categorical, or an embedding, and may
live per pixel, block, region, frame, interval, or sequence. It may depend on
media planes, masks, metadata, other fields/channels, or immutable model assets.
It declares whether resampling is meaningful and which reducers are permitted.
The optimizer and UI discover these properties rather than assuming every
channel is a scalar block-time array.

## 3. Plugin registration contract

Core and third-party extensions use the same registry. Python packages register
through the `antscihub_sieve.plugins` entry-point group and return one or more
plugin manifests. A manifest contains:

```text
plugin_id                              # globally namespaced
plugin_version
compatible_sieve_api
node registrations
  node_id                              # namespaced, stable
  node_family
  implementation_version
  human description and citations
  input/output port schemas
  parameter JSON Schema and defaults
  scientific versus execution parameters
  axis, unit, validity, and alignment rules
  lookback, lookahead, spatial halo, warm-up, and finalization
  deterministic/seed behavior
  supported backends and resource estimator
  cache-key contribution
```

Node ids are stable scientific identities; display names are not. Semantic
versioning of a package does not substitute for an explicit implementation
version. A result-changing code change requires a new implementation identity.

The execution-facing lifecycle is small and node-family-independent:

```text
resolve(parameters, input_schemas) -> resolved ports, context, resources
open(run_context)                  -> bounded node state
process(input_chunk)               -> zero or more typed output chunks
finish()                           -> final chunks and summaries
close(outcome)                     -> release resources
```

`process` receives absolute axis coordinates and validity with its data. Output
chunks carry coordinates and validity rather than relying on call order.
Stateless vectorized implementations may collapse the lifecycle internally;
model-backed or temporal nodes may retain only the state declared by `resolve`.

Adding a preprocessor, color mask, channel, temporal transform, detector, model,
or exporter requires only a plugin package, manifest, implementation, and
conformance tests. It must not require a switch statement in the CLI, executor,
RunSpec, artifact store, or optimizer.

The canonical discovery and validation surface includes:

```text
sieve plugins list --json
sieve plugins describe PLUGIN_OR_NODE --json
sieve plugins validate PACKAGE_OR_ENV
sieve channels list --json
sieve channels describe NODE_ID --json
sieve pipeline validate PIPELINE.json
sieve pipeline graph PIPELINE.json
sieve pipeline plan PIPELINE.json ASSET
```

Unknown node ids, incompatible port types, unresolved units/axes, invalid
parameters, missing model assets, or insufficient context fail during planning,
before video decoding.

## 4. Pipeline, experiment, and optimization contract

A pipeline document names nodes, connections, parameters, outputs, and declared
search axes. It becomes canonical, hashable JSON after resolution. `sieve
analyze` is a convenience entry point that resolves a named profile or pipeline
document into the same graph execution path.

Plugin parameter schemas may annotate searchable values, bounds, distributions,
conditional dependencies, and cost hints. They describe possibilities; an
experiment manifest chooses the actual search space. The optimizer can vary
nodes and topology as well as numbers—for example, compare a Morlet transform
with a flash-duration filter, or a hard red mask with a learned color
likelihood—while every candidate remains an ordinary executable pipeline.

```text
sieve experiment validate EXPERIMENT.json
sieve experiment plan EXPERIMENT.json --dataset DATASET.json
sieve experiment run EXPERIMENT.json --dataset DATASET.json --out STUDY/
```

Artifact reuse is based on resolved subgraph identity. Common upstream nodes may
be computed once across candidates; changing a downstream threshold must not
reopen video. Train/validation/test boundaries apply to fitted nodes and all
optimization decisions. A model artifact records its training dataset, feature
identities, fitting procedure, software, seed, and held-out evaluation.

Exploration does not weaken reproducibility: each result freezes the resolved
graph, plugin manifests and versions, parameters, assets/models, validity and
coverage, and software environment. Unknown methods can be added later without
making old results ambiguous.

### 4.1 Extension conformance

The conformance kit supplies tiny typed artifacts and a test executor. Every
node is tested for schema agreement, parameter validation, deterministic replay
where claimed, chunk/whole agreement where claimed, validity propagation,
context boundaries, cancellation, artifact identity, and resource-estimate
sanity. Specialized families add tests—for example hue wrap and color metadata
for color nodes, mask-kind preservation for mask nodes, and temporal alignment
for transforms.

At least one core integration fixture loads an external test plugin that adds a
preprocessor and channel, constructs a graph, runs it, caches it, and exposes its
search parameters without any core source change. A synthetic red-flash video
must be detectable through a color-only graph whose plan contains no tensor or
Morlet node.

## Part II: built-in reference profile `sieve.profile.tensor_morlet.v1`

The remainder defines this profile exactly. “Fixed,” “default,” and “requires an
implementation-version change” below are scoped to this profile or named node,
not to all SIEVE pipelines.

## 5. Reference notation and conventions

```text
W, H                  asset dimensions
F                     verified decodable frame count
p/q                   exact rational frames per second
fps = p/q             evaluation of the rational rate
dt = q/p              seconds per frame
s                     requested downsample factor in (0,1]
Ww, Hw                resolved working dimensions
b                     block side in working pixels
R, C                  block-grid rows and columns
I_t[y,x]              working-resolution intensity at absolute frame t
```

Metadata uses `(x,y)`; arrays use `[y,x]`. Boxes, frame spans, and threshold
bands are half-open unless explicitly stated. Stored numerical arrays are
`float32` unless specified otherwise.

### 5.1 Fixed conventions

Changing these requires an implementation-version change:

- Half-open coordinates and bands.
- Canonical intensity interval `[0,1]` before normalization.
- Rational timebase calculations.
- Area downsampling.
- Scharr kernels, signs, and scaling.
- Source-pixel and seconds units.
- Block means over owned pixels.
- Fractional area for partial blocks.
- Unknown-frame and coverage semantics.

### 5.2 Scientific parameters

These belong in `RunSpec` and may later be optimized:

- Downsample, normalization, and normalization parameters.
- Tensor integration scale and ridge terms.
- Block size.
- Channel and channel parameters.
- Morlet carrier, frequency grid, and frequency band.
- Band-power, occupancy, component, and temporal thresholds.
- Detection-window duration/alignment.
- Gap bridging and minimum event duration.

### 5.3 Execution parameters

Chunk sizes, thread/process counts, RAM versus disk placement, FFT batch size,
and progress frequency must not change valid results beyond section 21's
tolerances. If they do, promote them into scientific identity.

## 6. Reference video samples

### 6.1 Time

Frame `t` represents:

```text
time_seconds(t) = t*q/p
```

Duration-to-frame conversion is round-half-up:

```text
frames(D) = max(1, floor(D*p/q + 0.5))
```

A requested start time resolves by floor unless nearest-frame behavior was
explicitly requested. The integer frame range is recorded. Rounded display fps
never enters scientific calculations.

### 6.2 Grayscale node

Each decoded pixel becomes encoded-luma intensity `I_raw` in `[0,1]`.

For full-range integer gray code `v` of bit depth `n`:

```text
I_raw = v/(2^n - 1)
```

Limited-range luma maps its declared legal black/white codes to 0/1 and clips
excursions. Color input resolves color matrix, range, decoder conversion, and
output bit depth in the run plan. Ambiguous color metadata requires an explicit
policy for noninteractive scientific runs.

This quantity is encoded luma, not physical linear-light luminance. A future
linear-light mode is a different preprocessing operation.

Non-finite decoded intensity fails the frame; it is never replaced with zero.

## 7. Reference downsampling node

For `0 < s <= 1`:

```text
Ww = max(1, floor(W*s + 0.5))
Hw = max(1, floor(H*s + 0.5))
sx = Ww/W
sy = Hw/H
```

Each working pixel is the exact area average of its source footprint. Upsampling
is an error. `sx,sy`, not the requested decimal `s`, convert derivatives to
source-pixel units after integer dimension resolution.

## 8. Reference normalization nodes

Normalization is per frame, after downsampling, across only the current asset.

### 8.1 Off

```text
I_t = I_raw,t
```

This is the default.

### 8.2 Z-score

For all `N=Hw*Ww` pixels:

```text
mu = sum(I_raw)/N
variance = sum((I_raw-mu)^2)/N
I_t = (I_raw-mu)/max(sqrt(variance), epsilon)
```

Default `epsilon=1e-6`. Accumulate in `float64`; store `float32`. Population
variance is used. If standard deviation is below epsilon, return an exact zero
frame and record `normalization_degenerate=true`.

### 8.3 CLAHE

CLAHE is optional and has no unnamed defaults. Its parameters are:

```text
tile_grid_x, tile_grid_y, clip_limit, quantization_bits,
implementation_id, implementation_version
```

Clip intensity to `[0,1]`, quantize round-half-up, apply the recorded backend
over the full asset frame, and convert its full-range output back to `[0,1]`.
For OpenCV, record exact `createCLAHE` arguments and version. Backend/version is
scientific identity until equivalence is demonstrated.

## 9. Reference frame pairing and derivatives

Motion at frame `t` describes interval `t-1 -> t` and is indexed at `t`. For a
requested `[a,b)`, decode `a-1` as context when `a>0`. Frame 0 has valid
intensity but invalid motion-related channels.

Use correlation kernels with reflect-101 borders:

```text
Kx = (1/32) * [[-3,  0,  3],
               [-10, 0, 10],
               [-3,  0,  3]]
Ky = transpose(Kx)
```

The derivative of an increasing left-to-right ramp is positive. If a library
implements convolution rather than correlation, reverse the kernels.

For frames `I0=I_{t-1}`, `I1=I_t`:

```text
Ix = 0.5*sx*(correlate(I0,Kx) + correlate(I1,Kx))
Iy = 0.5*sy*(correlate(I0,Ky) + correlate(I1,Ky))
It = (I1-I0)/dt
```

`Ix,Iy` are intensity/source-pixel; `It` is intensity/second. This makes solved
velocity source-pixels/second.

For interior linear ramp `a*x+b*y+c`, pre-source-scale Scharr derivatives must
equal `a,b` within `1e-6`.

## 10. Reference structure tensor

Brightness constancy is:

```text
Ix*u + Iy*v + It = 0
```

### 10.1 Products and integration

```text
Pxx=Ix^2  Pyy=Iy^2  Pxy=Ix*Iy
Pxt=Ix*It Pyt=Iy*It Ptt=It^2
```

Spatially average every product with the same separable normalized Gaussian.
The parameter `tensor_sigma_source_px` defaults to `2.0` and resolves to:

```text
sigma_x_work = tensor_sigma_source_px*sx
sigma_y_work = tensor_sigma_source_px*sy
```

Sample each 1-D Gaussian through `ceil(4*sigma)`, normalize in `float64`, and
use reflect-101 borders. Sigma below `1e-6` is identity. Negative sigma is an
error.

```text
Jxx=G(Pxx) Jyy=G(Pyy) Jxy=G(Pxy)
Jxt=G(Pxt) Jyt=G(Pyt) Jtt=G(Ptt)
```

### 10.2 Regularized solve

```text
trace = Jxx+Jyy
ridge = ridge_abs + ridge_rel*trace
a = Jxx+ridge
b = Jxy
d = Jyy+ridge
det = a*d-b*b

u = -(d*Jxt-b*Jyt)/det
v =  (b*Jxt-a*Jyt)/det
```

Initial values:

```text
ridge_abs=1e-12
ridge_rel=1e-3
det_floor=1e-24
```

Evaluate determinant/numerators in `float64`, store `float32`. If determinant
is non-finite or `<=det_floor`, store NaN velocity and invalidity. `u` is right;
`v` is down.

Regularization stabilizes arithmetic. It is not evidence that textureless flow
is reliable.

### 10.3 Confidence

From the unregularized spatial tensor:

```text
disc = sqrt(max(0,(Jxx-Jyy)^2+4*Jxy^2))
lambda_max = 0.5*(trace+disc)
lambda_min = 0.5*(trace-disc)
texture = max(lambda_min,0)
coherence = (lambda_max-lambda_min)/max(lambda_max+lambda_min,1e-20)
```

Neither texture nor coherence silently masks velocity. A quality gate is an
explicit scientific parameter/channel.

## 11. Reference pixel and block fields

Per-pixel fields:

```text
intensity_pixel = I_t
change_pixel = Jtt
speed_pixel = sqrt(u^2+v^2)
residual = It+Ix*u+Iy*v
appearance_pixel = residual^2
```

Invalid velocity makes residual/appearance invalid. No second spatial smoothing
is applied to appearance: the tensor solve already integrates evidence locally,
and block reduction supplies the next declared aggregation. This also prevents
an invalid velocity sample from contaminating neighboring values through an
implicit NaN-handling policy.

### 11.1 Grid and partial blocks

For block side `b>=1` working pixels:

```text
R=ceil(Hw/b), C=ceil(Ww/b)
y0=r*b; y1=min((r+1)*b,Hw)
x0=c*b; x1=min((c+1)*b,Ww)
area=(y1-y0)*(x1-x0)
block_weight=area/(b*b)
```

No padding pixels enter statistics. A block value is the arithmetic mean of
finite owned pixels using `float64` accumulation. With no finite pixels it is
NaN and effective weight zero for that frame.

### 11.2 Profile channels

```text
intensity    = mean(intensity_pixel)
change       = mean(change_pixel)
appearance   = mean(appearance_pixel)
texture      = mean(texture)
coherence    = mean(coherence)
u            = mean(u_pixel)
v            = mean(v_pixel)
tensor_speed = mean(speed_pixel)
net_speed    = sqrt(u^2+v^2)
```

Mean per-pixel `tensor_speed` and magnitude-of-mean `net_speed` are distinct.

Units with normalization off:

| channel | unit |
|---|---|
| intensity | normalized encoded luma |
| change, appearance | luma^2/s^2 |
| texture | luma^2/source-pixel^2 |
| coherence | dimensionless |
| u,v,tensor_speed,net_speed | source-pixel/s |

Z-score replaces luma with standardized-intensity units. Calibrated mm/s is a
derived channel and never overwrites raw source-pixel/s.

## 12. Reference velocity-gradient channels

Derivatives of block-mean `u,v` use actual block-center source coordinates.
Interior differences span neighboring centers; boundary differences are
one-sided. One-column/row grids have invalid derivatives in that axis. Any NaN
in a stencil yields NaN.

```text
ux=du/dx, uy=du/dy, vx=dv/dx, vy=dv/dy
divergence = ux+vy
vorticity = vx-uy
shear = sqrt((ux-vy)^2+(uy+vx)^2)
```

Units are `s^-1`. Divergence/vorticity are signed; shear is nonnegative. Raw
signed thresholding must state signed, absolute, positive-part, or negative-part
semantics. Spectral power of a signed channel is nonnegative and may be detected.

## 13. Reference-profile registry entries

Each registration declares:

```text
name, display name, implementation version, prerequisites, parameters,
shape, unit, signedness, lookback/lookahead, permitted transforms, validity rule
```

This profile registers intensity, change, appearance, texture, coherence, u, v,
tensor_speed, net_speed, divergence, vorticity, and shear. Uncomputed channels
are absent, not zero or all-NaN placeholders. Missing prerequisites are errors.

## 14. Reference spectral input

The spectral stage does not interpolate missing samples. A coefficient is valid
only when every sample touched by its kernel is finite and channel-valid.

Default detrending is `none`. Optional `constant` or `linear` segment detrending
is scientific identity. Chunked execution may use it only if the trend is fitted
once over the declared segment and shared with every chunk.

## 15. Reference Morlet transform

The transform is defined directly at physical center frequencies; no library
scale-to-frequency default is used.

Parameters and initial values:

```text
frequency_min_hz > 0
frequency_max_hz <= 0.45*fps
voices_per_octave=12
morlet_omega0=6.0
morlet_truncate_sigma=4.0
```

Generate descending centers:

```text
f_k=frequency_max_hz*2^(-k/voices_per_octave)
```

retain centers `>=frequency_min_hz`, then store ascending. The resolved list is
in `RunSpec`.

### 15.1 Kernel

For center `f`:

```text
sigma_time=morlet_omega0/(2*pi*f)
eta=tau/sigma_time
psi(tau)=(exp(i*morlet_omega0*eta)-exp(-morlet_omega0^2/2))*exp(-eta^2/2)
M_f=ceil(morlet_truncate_sigma*sigma_time/dt)
g[n]=psi(n*dt), n=-M_f...M_f
```

Subtract the discrete mean from `g` in `complex128`, giving exact finite-kernel
zero DC response. Form correlation filter:

```text
h[n]=conjugate(g[-n])
response=sum_n h[n]*exp(-i*2*pi*f*n*dt)
h=h/abs(response)
```

Response below `1e-15` or non-finite is invalid. Then:

```text
W_f[t]=sum_n h[n]*x[t-n]
P_f[t]=abs(W_f[t])^2
```

A unit complex sinusoid at center frequency has coefficient magnitude one away
from edges. A real unit cosine has approximately `0.25` power. Kernels and
convolution are `complex128`; stored power is `float32`.

### 15.2 FFT and validity

Direct and FFT convolution are execution alternatives. FFT uses linear padding
of at least `N_signal+N_kernel-1`. Chunked overlap-save/add must match direct
convolution on valid samples.

A coefficient is valid only when all samples `t-M_f...t+M_f` exist and are
valid. Band power is valid only when every frequency with nonzero band weight is
valid. Missing context at asset edges remains unknown. This full-support rule is
conservative and guarantees chunk agreement.

## 16. Reference frequency-band power

The band `[f_low,f_high)` is aggregated by log-frequency cell width, preventing
power from growing merely because more voices per octave were requested.

Let `z_k=log2(f_k)`. Each center owns the Voronoi cell bounded by midpoints to
neighboring centers. At either endpoint, extrapolate by half of that endpoint's
nearest-neighbor spacing. If the grid contains only one center, use the nominal
half-spacing `1/(2*voices_per_octave)` on each side. Clip each cell against
`[log2(f_low),log2(f_high)]`. Let intersection width be `w_k`.

```text
band_power[t,r,c]=sum_k(w_k*P_k[t,r,c])/sum_k(w_k)
```

At least one positive weight is required. Store resolved centers, cell bounds,
and weights. Band power is nonnegative and has squared channel units.

## 17. Reference spatial occupancy

Value band is half-open `[value_low,value_high)`; JSON null means unbounded. NaN
never passes.

```text
passing = valid & (band_power>=low) & (band_power<high)
occupancy = sum(block_weight*passing)
observed_weight = sum(block_weight for valid blocks)
occupancy_fraction = occupancy/observed_weight
```

No observed area yields NaN fraction. Detector thresholding explicitly chooses
equivalent-full-block `occupancy` or `occupancy_fraction`. Initial default is
fraction, which is interpretable across nested spatial crops.

Connected components use explicit 4- or 8-connectivity; default 8. Component
weight is the sum of block weights. Component fraction divides by observed
weight. Report the largest weight/fraction. An optional component gate is
explicit; without it these are diagnostics.

## 18. Reference temporal gate

Convert detection duration to `W` frames by round-half-up.

Trailing indices at `t` are `t-W+1...t`. Centered windows use:

```text
left=floor((W-1)/2)
right=W-1-left
t-left...t+right
```

The even-window asymmetry is intentional. A window is valid only when its full
support exists and every selected occupancy value is valid; it is never shortened
at edges.

```text
q_windowed[t]=sum(q[i])/W
```

Use `float64` prefix sums, store `float32`. Gate interval is half-open
`[count_low,count_high)` in the selected occupancy unit. An optional component
gate aggregates the component quantity over the same temporal window using
explicit `mean`, `minimum`, or `maximum`; default `mean`.

Gate encoding:

```text
0 valid negative
1 valid positive
255 unknown/unprocessed
```

## 19. Reference event intervals

Unknown frames split events and are never bridged. Parameters are
`bridge_gap_seconds>=0` and `minimum_event_seconds>=0`, both default zero.

Within each contiguous valid span:

1. Fill negative gaps no longer than the resolved bridge length when bounded by
   positives on both sides.
2. Remove positive runs shorter than the resolved minimum duration.
3. Export remaining positive runs as absolute half-open frame intervals.

Interval `[a,b)` seconds are exactly `[a*q/p,b*q/p)`.

## 20. Reference pseudocode

```python
def tensor_frame(prev, curr, fps, sx, sy, p):
    ix = .5*sx*(corr(prev,KX)+corr(curr,KX))
    iy = .5*sy*(corr(prev,KY)+corr(curr,KY))
    it = fps*(curr-prev)

    jxx,jyy,jxy = G(ix*ix),G(iy*iy),G(ix*iy)
    jxt,jyt,jtt = G(ix*it),G(iy*it),G(it*it)
    ridge = p.ridge_abs+p.ridge_rel*(jxx+jyy)
    a,b,d = jxx+ridge,jxy,jyy+ridge
    det = a*d-b*b
    valid = finite(det) & (det>p.det_floor)
    u = where(valid,-(d*jxt-b*jyt)/det,nan)
    v = where(valid, (b*jxt-a*jyt)/det,nan)
    appearance = (it+ix*u+iy*v)**2
    return jtt,u,v,appearance,valid
```

```python
def detect(block_series, weights, run):
    power,pvalid = morlet_power(block_series,run.spectral)
    band,bvalid = log_frequency_average(power,pvalid,run.band)
    passing = bvalid & (band>=run.value_lo) & (band<run.value_hi)
    observed = weighted_sum(weights,bvalid)
    occupied = weighted_sum(weights,passing)
    fraction = where(observed>0,occupied/observed,nan)
    largest = largest_component_fraction(passing,bvalid,weights,run.connectivity)
    temporal = strict_window_mean(fraction,run.window)
    gate = encode_unknown(temporal)
    finite_rows = finite(temporal)
    gate[finite_rows] = ((temporal[finite_rows]>=run.count_lo)
                         & (temporal[finite_rows]<run.count_hi))
    return postprocess(gate,run.bridge_gap,run.minimum_event)
```

## 21. Reference numerical precision and tolerances

```text
intensity/derivatives/products       float32
Gaussian and block accumulation      float64 internal, float32 stored
velocity solve                       float64 internal, float32 stored
Morlet kernel/convolution            complex128
wavelet/band power                   float32 stored
temporal prefix sums                 float64 internal
gate                                 uint8 {0,1,255}
```

Masks, coverage, and event intervals match exactly. General finite arrays use
`atol=1e-6, rtol=2e-5`; direct/FFT wavelet arrays use
`atol=2e-6, rtol=5e-5`. Tests avoid thresholds placed within numerical tolerance
of a value. If a backend cannot conform, backend/device becomes scientific
identity.

## 22. Required reference-profile golden tests

All fixtures are analytic or created specifically for SIEVE.

### Intensity and geometry

- Gray8 `[0,127,255] -> [0,127/255,1]`; gray16 endpoints -> 0/1.
- Constant z-score -> exact zeros and degenerate flag.
- Hand-computed four-pixel population z-score.
- A 4x4-to-2x2 area resize equals quadrant means.
- A 5x5 field at block 4 has weights `[[1,1/4],[1/4,1/16]]`.
- Partial blocks exclude nonexistent padding.

### Derivatives and tensors

- Horizontal/vertical ramps verify Scharr sign, scale, and borders.
- Constant frames have zero derivatives.
- Uniform flicker has temporal but no spatial gradient.
- Downsampled ramp converts back to source-pixel units.
- Hand-supplied tensor components verify the 2x2 solve.
- Small translated texture recovers known interior velocity within a declared
  fixture tolerance.
- Uniform flicker gives negligible velocity and high appearance residual.
- Textureless input reports low confidence rather than confident motion.

### Blocks and gradients

- A one-pixel corner block contributes `1/16`, not one.
- Opposing vectors distinguish tensor speed from net speed.
- Translation gives zero velocity gradients.
- Expansion, rotation, and pure shear produce their expected invariants.
- Ragged last-block centers use actual coordinate spacing.

### Morlet and detection

- Each kernel has discrete mean below `1e-14` and unit center response within
  `1e-12`.
- A real center-frequency cosine gives interior power near 0.25.
- Off-frequency response is lower by a fixture-recorded margin.
- Direct, whole-FFT, and overlap-save results agree on valid samples.
- Incomplete-support coefficients remain invalid.
- Increasing voices leaves log-averaged band power stable within tolerance.
- Half-open endpoints, NaN exclusion, weighted occupancy, connectivity, strict
  temporal support, centered-even allocation, gap bridging, and event endpoints
  each have hand-calculable cases.

## 23. Reference-profile optimization axes

Optimization varies scientific parameters without changing their definitions.
Initial axes may include normalization, downsample, tensor sigma/ridge, block
size, channel, frequency grid/band, value band, occupancy quantity/band,
component gate, detection duration/alignment, gap bridge, and minimum duration.

Fixed signs, units, area weights, half-open bands, and unknown semantics are not
search dimensions.

A reusable artifact matches every upstream identity: asset/coverage, decode,
downsample, normalization, tensor, block grid, channel, Morlet grid, band
aggregation, and validity. Threshold trials never mutate upstream artifacts.

An asset and its descendants remain in one train/validation/test group by
default. Comparing whole colonies with nested fractions must not count those
correlated crops as independent samples.

Learned signatures over `channel x spatial scale x temporal scale` use these
exact feature identities. Training-fold normalization and selection never inspect
validation/test labels. A learned score retains the same validity, threshold,
and event rules.

## 24. Reference-profile optimization candidate register

Candidates belong to one of three classes:

- **Equivalent:** must meet section 21 tolerances and exact masks.
- **Reuse:** avoids recomputing an identical upstream artifact.
- **Scientific alternative:** changes values and must be exposed as a new
  parameter/channel/model and evaluated scientifically.

Nothing becomes the default from plausibility alone. Benchmark end-to-end time,
peak memory, temporary storage, and numerical agreement on representative assets.

### 24.1 Equivalent candidates

1. Decode directly to canonical grayscale at resolved working dimensions,
   provided range conversion and exact area-resize tests match the reference
   decode-then-resize path.
2. Bounded decoder prefetch overlapping CPU tensor computation; queue depth is
   memory-budgeted and cancellation-aware.
3. Separable Scharr and Gaussian passes with fused tensor-product generation,
   without materializing unused products.
4. Channel-prerequisite pruning: change-only jobs do not solve velocity;
   velocity-dependent channels share one solve.
5. Reduce each completed pixel field to blocks immediately, releasing pixel
   temporaries before the next field/frame.
6. Reuse the signal FFT for all Morlet filters in a block/time batch; group
   kernels by compatible FFT length.
7. When only band power is requested, accumulate weighted scale power directly
   and discard per-frequency coefficients instead of materializing a frequency
   cube.
8. Choose direct convolution for short kernels and FFT overlap-save for long
   kernels through a measured cost planner.
9. Memory-map block-channel or band-power arrays when RAM placement exceeds the
   budget, preserving the identical decoder/stage path.
10. Vectorized threshold sweeps: sort per-frame block powers once and use
    cumulative area weights to evaluate many value thresholds.
11. Prefix sums for many detection durations/count thresholds.
12. Delay connected-component evaluation until non-spatial criteria pass during
    broad parameter search, then evaluate it for shortlisted candidates. This is
    exact only if rejected trials cannot become positive through the spatial
    criterion; the planner must prove that condition.
13. GPU tensor, FFT, or reduction backends after conformance testing. Backend is
    scientific identity until tolerance equivalence is established.

### 24.2 Reuse candidates

1. Persist selected block-channel series once per upstream `RunSpec` fragment;
   reuse them for spectral and threshold studies without reopening video.
2. Persist band power for dense value/count/window searches.
3. Persist a compact multiband feature basis when many frequency bands will be
   studied; record its approximation/resolution explicitly.
4. Cache resolved Morlet kernels by exact timebase and spectral specification.
5. Share feature artifacts across distributed trials through content-addressed,
   read-only storage.
6. Materialize a feature matrix once for LDA/L1-logistic fitting and retain
   ordered feature identities beside it.

### 24.3 Scientific alternatives

These may be faster but are not equivalent substitutions:

1. Butterworth or FIR band energy instead of Morlet power.
2. Block-level tensor solving instead of per-pixel solve then block reduction.
3. Central temporal differences instead of paired causal differences.
4. Alternative gradient kernels or tensor integration windows.
5. Confidence-gated velocity or robust regression.
6. Multiband/hop-subsampled temporal representations.
7. Lower spatial resolution, larger blocks, or temporal decimation.
8. Learned optical flow or learned temporal features.

Each requires a distinct implementation id, output identity, and labeled-data
comparison. It cannot enter as an invisible performance patch.

## 25. Reference-profile implementation review checklist

- Rational fps only; recorded grayscale/range conversion.
- Tested derivative sign and source-pixel units; temporal derivative per second.
- One Gaussian rule for all tensor products; written 2x2 solve exactly.
- Confidence exposed, not silently thresholded.
- Mean per-pixel speed distinct from net speed.
- Partial blocks retain real area; gradients use actual block centers.
- Morlet kernels use physical frequencies, zero DC, and unit center gain.
- Log-frequency weighting and full-support validity are shared everywhere.
- Half-open value/count bands and strict windows are shared everywhere.
- Chunking changes placement/scheduling, not valid results.
- Every output includes `RunSpec` and coverage.
- Optimizers submit ordinary scientific jobs and cannot bypass the engine.

## 26. Foundational references for the reference profile

The local velocity formulation uses the weighted least-squares principle of
[Lucas and Kanade, 1981](https://idl.uw.edu/living-papers-paper/lucas-kanade/):
spatial intensity gradients constrain a local translation through a small normal
equation.

The time-frequency stage uses a complex Morlet wavelet and treats finite-series
edge support explicitly, informed by [Torrence and Compo,
1998](https://doi.org/10.1175/1520-0477%281998%29079%3C0061%3AAPGTWA%3E2.0.CO%3B2).

SIEVE's discrete unit-gain normalization, full-kernel validity, log-frequency
averaging, area-weighted occupancy, and threshold conventions are definitions in
this contract and should not be attributed to those papers.
