# antscihub-SIEVE

**Signal Isolation for Ethological Video Events**

Implementation handoff for a headless, command-line-first system that searches
video representations for signals capable of isolating behavior.

The command line is the canonical product. Every operation must be available
noninteractively, reproducibly, and without a graphical environment. Desktop or
web interfaces may be added as clients, but they submit the same job documents
to the same executable and consume the same structured outputs.

## 1. Product definition

The system has four responsibilities:

1. Register a video as a self-describing video asset.
2. Derive spatial child assets by cropping regions from any video asset.
3. Analyze one asset at a time with a streaming, memory-bounded computation
   graph.
4. Make new preprocessors, masks, channels, transforms, detectors, and models
   independently installable, reproducible, and searchable by experiments.

Any video may be a root recording, an experimental replicate, or a smaller
spatial child of a replicate. These labels describe lineage and user intent;
they do not select different decoding or analysis implementations.

An analysis always sees one video and treats its full frame as the owned spatial
domain. Neighboring replicates, parent rectangles, and sibling clips never enter
the analysis plan.

The file supplied to `analyze` is the file that is decoded. To analyze a smaller
area, create or select a child asset and analyze that child.

The scientific question is intentionally broader than motion detection:

> If the evidence is sampled or transformed across time, value, function,
> resolution, and spatial aggregation, which representation best isolates a
> behavior?

Tensor motion followed by band-power detection is the first reference profile,
not the product architecture. A momentary red feather, a luminance flicker, a
texture change, motion inside a color mask, a spatial relationship, and a
learned feature are all legitimate signal paths.

## 2. Governing invariants

These are acceptance requirements.

### 2.1 One asset per analysis process

- One analysis job opens exactly one video asset.
- The processing grid covers that asset's full frame.
- No multi-asset atlas or region axis exists in the numerical engine.
- A directory containing other assets cannot change the resolved plan.
- Corpus parallelism launches independent single-asset jobs.

### 2.2 Recursive spatial lineage

- Any asset can be the parent of another spatial crop.
- A root source can produce replicate children.
- A replicate can produce sub-replicates.
- A sub-replicate can be cropped again without a special case.
- Each child records its parent identity and exact coordinate transformation.
- Opening a child reveals that it has a parent even when the parent is not
  presently reachable.
- When the parent is reachable through a supplied search root or catalog, the
  system can resolve and open it.

### 2.3 Location is not identity

- Asset identity never depends on an absolute path.
- Videos and sidecars may live anywhere.
- Moving a video together with its sidecar preserves its identity.
- Parent and child references use ids and content signatures. Paths are optional
  location hints only.
- A rebuildable catalog maps asset ids to discovered paths; it is not
  authoritative scientific state.

### 2.4 CLI authority

- The CLI resolves defaults, validates requests, creates run identities, and
  writes results.
- Other clients do not reimplement this policy.
- Every CLI operation can emit machine-readable JSON.
- Every long operation supports progress, cancellation, failure reporting, and
  deterministic noninteractive execution.
- No CLI module imports a GUI toolkit.

### 2.5 Explicit execution

- No silent decoder fallback.
- No silent source substitution.
- No silent parameter correction beyond documented clamping that appears in the
  resolved plan.
- No result may imply that unprocessed frames were examined.
- A partial or truncated run cannot be mistaken for a complete quiet result.

### 2.6 Open-ended scientific composition

- Analysis is a typed directed acyclic graph, not a fixed sequence of motion
  functions.
- Graphs may branch, join, use multiple spatial/temporal resolutions, and omit
  tensor computation.
- Media planes, preprocessors, masks, weights, fields, channels, temporal
  transforms, detectors, and postprocessors are first-class registered nodes.
- New node types are discovered from plugin manifests; adding one does not
  require edits to the executor, CLI parser, artifact store, or optimizer.
- The graph, node versions, parameters, model assets, and validity rules are
  frozen in every result.
- Optimization may search topology and method choice as well as numeric values.

## 3. Video assets and sidecars

### 3.1 Sidecar convention

For a video named:

```text
D:/footage/colony_07/replicate_12.mkv
```

the default sidecar is:

```text
D:/footage/colony_07/replicate_12.asset.json
```

The video and sidecar form an asset. A package directory is convenient but not
required. Commands accept a video path, sidecar path, asset id resolvable by a
catalog, or a dataset entry.

The sidecar is written atomically and has a versioned JSON Schema.

### 3.2 Asset kinds

`kind` is one of:

- `source`: the user considers this parent footage from which experimental
  units may be derived.
- `replicate`: the user considers the full frame one experimental unit.

The kind is descriptive. Both kinds may be analyzed full-frame and both may
produce child assets.

When a video has no sidecar, an interactive client asks:

> Is this source footage, or is the full video already a replicate?

The CLI expresses the same choice with `asset init --kind source|replicate`.
Noninteractive jobs never prompt; the argument or an asset sidecar is required.

If the user identifies a source but wants to analyze the entire frame as one
unit, either analyze the source directly or create a full-frame replicate child.
The latter is useful when a dataset should contain only assets explicitly named
as experimental replicates.

### 3.3 Sidecar schema

Required top-level fields:

```json
{
  "schema_version": 1,
  "asset_id": "uuid",
  "kind": "source | replicate",
  "label": "human-readable label",
  "media": {},
  "lineage": {},
  "calibration": {},
  "attributes": {}
}
```

`media` contains:

```text
filename                         basename or sidecar-relative path
content_sha256                   stable identity
quick_signature                  optional fast verification hint
size_bytes
width
height
pixel_format
codec
fps_num
fps_den
container_frame_count
decoded_frame_count
duration_seconds
created_utc
```

The exact rational frame rate is authoritative for frame/time conversion.
`decoded_frame_count` is distinct from a container claim. A command that has not
measured decoded length reports that fact rather than presenting it as verified.

`lineage` contains:

```text
parent                           null for a root asset
  asset_id
  content_sha256
  label_snapshot
  kind_snapshot
  location_hints[]               optional, non-authoritative

derivation                      null for an independently registered asset
  operation: crop
  parent_box_xyxy                exact half-open parent-pixel box
  output_width
  output_height
  child_to_parent_transform      exact rational affine mapping
  frame_start                    normally 0
  frame_count
  encoder
  encoder_arguments[]
  created_utc

ancestors[]                      identity snapshots for discoverability
```

The ancestor snapshots let a deeply cropped child explain its history without
opening every ancestor. The immediate parent remains the authoritative lineage
edge.

`calibration` may contain:

```text
pixels_per_mm
body_length_mm
quiescent_intervals_seconds[]
measurement_notes
```

Calibration is interpreted in the asset's pixel coordinates. A crop preserves
pixels/mm when spatial resolution is unchanged. A resized child derives the new
calibration through the recorded rational transform.

`attributes` is for user metadata such as session, colony, treatment, species,
camera, and experimental group. Optimization datasets use these fields for
grouping and leakage control.

### 3.4 Content verification

Commands support verification levels:

- `metadata`: compare dimensions, rational fps, size, and available container
  facts.
- `quick`: metadata plus deterministic head/tail content sampling.
- `full`: whole-file SHA-256.

The sidecar records which facts were measured and when. A mismatch fails with a
structured error. It does not silently generate a new identity.

### 3.5 Catalog

A catalog is an optional SQLite index built by scanning user-supplied roots:

```powershell
sieve catalog build D:/footage E:/archive --out assets.sqlite
sieve catalog locate --catalog assets.sqlite --asset-id <uuid>
sieve lineage show CHILD --catalog assets.sqlite
sieve lineage parent CHILD --catalog assets.sqlite
```

The catalog stores ids, signatures, labels, and paths. Deleting it loses no
scientific information; scanning rebuilds it. Parent resolution verifies the
resolved asset's id and content signature before use.

## 4. Spatial derivation

### 4.1 Derivation behavior

`derive` creates child video assets from exact spatial regions of one parent.
Each output is independently portable and immediately analyzable.

```powershell
sieve derive parent.mp4 `
  --roi 3698,113,4109,541 `
  --label rep27 `
  --kind replicate `
  --out D:/replicates/rep27/
```

Multiple regions can be supplied in a versioned layout document:

```powershell
sieve derive parent.mp4 `
  --layout plate-layout.json `
  --out D:/replicates/
```

The same command works when `parent.mp4` is itself a replicate:

```powershell
sieve derive replicate_12.mkv `
  --roi 80,40,300,260 `
  --label colony-quadrant-a `
  --kind replicate `
  --out D:/subreplicates/quadrant-a/
```

No flag distinguishes a root crop from a nested crop.

### 4.2 Coordinate contract

ROI boxes are integer, half-open parent coordinates `(x0, y0, x1, y1)` with:

```text
0 <= x0 < x1 <= parent_width
0 <= y0 < y1 <= parent_height
```

Cropping is exact. Chroma-subsampled codecs must not round odd origins without
reporting a different resolved box.

If the child is not resized:

```text
x_parent = x_child + x0
y_parent = y_child + y0
```

If resizing is requested, store the exact rational map from child pixel edges to
parent pixel edges. Coordinate composition through an arbitrary ancestor chain
must be deterministic and testable.

Spatial children normally preserve the complete temporal axis. A later temporal
subclip feature may use `frame_start` and `frame_count`; it must compose temporal
lineage with the same rigor.

### 4.3 Encoding

The derivation command exposes named profiles and raw encoder overrides:

```text
lossless
high-quality
compact
copy-compatible, when an exact crop can be represented without re-encoding
```

Profiles resolve to explicit codec, pixel format, color-range, and encoder
arguments recorded in the sidecar. Analysis provenance includes the derived
clip's content hash and encoding details.

Derivation must preserve the evidence required by its declared purpose. A
motion-only child may deliberately store grayscale footage to avoid repeated
color conversion. A child intended for color or unknown future analyses must
retain an appropriate color representation. The profile records codec, bit
depth, range, color primaries, transfer function, matrix, and chroma details so
the information lost or retained is explicit. Derivation never converts to
grayscale merely because the first reference analysis uses luma.

### 4.4 Transactional output

A child is built into a temporary directory. The video is fully closed, probed,
frame-counted, hashed, and sidecar-validated before the completed package is
renamed into place. Cancellation or failure leaves no apparently complete
child.

## 5. Canonical CLI

The canonical executable name is `sieve`. The Python distribution and import
package are named `antscihub-sieve` and `antscihub_sieve`, respectively.

### 5.1 Global behavior

Every command supports:

```text
--log-format text|json
--log-level error|warning|info|debug|trace
--quiet
--no-color
--threads N
--temp-dir PATH
--memory-limit SIZE
--config FILE
--set KEY=VALUE
```

Rules:

- Human-readable status goes to stderr.
- Machine-readable primary output goes to stdout or an explicit output path.
- JSON logs are one object per line.
- Resolved configuration is printable before execution.
- Command-line values override a configuration file explicitly and are recorded
  in the run document.
- Unknown configuration keys are errors.
- Exit code 0 means success, 1 means an execution failure, and 2 means invalid
  usage or configuration.

### 5.2 Asset commands

```powershell
sieve asset inspect VIDEO_OR_SIDECAR --json
sieve asset init VIDEO --kind source --label session-07
sieve asset init VIDEO --kind replicate --parent PARENT_ASSET
sieve asset verify ASSET --level full
sieve asset relink ASSET --parent NEW_PARENT_LOCATION
sieve lineage show ASSET --json
sieve lineage compose ASSET --ancestor <asset-id> --json
```

`asset init` probes media and writes a sidecar. If `--parent` is supplied, the
caller must also provide or confirm the derivation relationship; merely naming a
parent cannot invent crop geometry.

### 5.3 Derivation commands

```powershell
sieve derive PARENT --roi x0,y0,x1,y1 --label NAME --out DIR
sieve derive PARENT --layout FILE --out DIR --profile lossless
sieve derive PARENT --full-frame --kind replicate --out DIR
sieve derive verify CHILD
```

`derive --plan` prints output files, crop filters, expected dimensions, estimated
storage, and encoder settings without decoding.

### 5.4 Analysis command

`analyze` accepts either a named profile, a complete pipeline document, or
convenience flags that resolve to one. The tensor example is:

```powershell
sieve analyze ASSET `
  --profile sieve.profile.tensor_morlet.v1 `
  --channel tensor_speed `
  --downsample 1.0 `
  --block-size 4 `
  --normalize zscore `
  --frequency-band 13.66,25 `
  --value-band 0.0,inf `
  --count-band 1,inf `
  --detection-window 0.25s `
  --window-mode trailing `
  --out result/
```

A color-display analysis can be supplied without adding a new top-level command:

```powershell
sieve analyze ASSET --pipeline red-feather-flash.pipeline.json --out result/
```

Frame selection:

```text
--start FRAME|SECONDS
--stop FRAME|SECONDS
--frames N
--duration SECONDS
```

Only one of the overlapping stop/length forms may be supplied. Parsed values and
the resolved half-open frame range are recorded.

Execution controls:

```text
--resume
--overwrite
--checkpoint-interval SECONDS
--allow-truncated
--save-series
--save-band-power
--keep-intermediates
--stop-after STAGE
--progress-interval SECONDS
--dry-run
```

`--dry-run` performs discovery, validation, planning, memory estimation, and
artifact-key resolution without decoding frames.

### 5.5 Batch and cluster commands

Inputs may be a list of sidecars, a glob, a catalog query, or a dataset manifest.

```powershell
sieve batch "D:/replicates/**/*.asset.json" `
  --params flight-params.json `
  --workers 8 `
  --threads-per-worker 2 `
  --memory-per-worker 8GiB `
  --out D:/results/
```

Deterministic sharding:

```powershell
sieve batch dataset.json --shard 3/32 --params params.json --out results/
```

Cluster planning:

```powershell
sieve jobs plan dataset.json `
  --params params.json `
  --shards 128 `
  --out job-plan/

sieve jobs run job-plan/jobs-003.jsonl
sieve jobs summarize job-plan/ --out summary.json
```

`jobs plan` resolves every asset and run identity before compute begins. Each
JSONL row is a self-contained, noninteractive job. It includes expected memory,
output paths, and a deterministic job id. This form can be submitted through
Slurm, PBS, Kubernetes, or another scheduler without importing project code into
the scheduler wrapper.

Example Slurm array:

```bash
#SBATCH --array=0-127
sieve jobs run "job-plan/jobs-$(printf '%03d' "$SLURM_ARRAY_TASK_ID").jsonl"
```

Per-asset failure does not corrupt other jobs. A summary reports succeeded,
failed, skipped-as-identical, incomplete, and cancelled jobs. Resume compares
complete input and run identity, never output filename alone.

## 6. Run specification

Every analysis resolves to an immutable `RunSpec` serialized as canonical JSON.
The RunSpec stores a resolved graph, not a schema with one permanent slot for
each currently known method.

```text
schema_version
run_id

input
  asset_id
  content_sha256
  width
  height
  fps_num
  fps_den
  decoded_frame_count
  frame_range

pipeline
  pipeline schema version
  named profile, when used
  nodes[]
    instance id
    registered node id and implementation version
    resolved scientific parameters
    input connections
    resolved input/output artifact schemas
  requested outputs
  resolved graph hash

models[]
  model artifact identity and content hash

execution
  chunk sizes
  memory placement policy
  numerical dtype
  deterministic seed where applicable

software
  application version
  dependency versions
  platform summary
```

The familiar preprocessing, block grid, tensor channel, Morlet transform, and
threshold detector appear as nodes when the built-in tensor–Morlet profile is
selected. Other pipelines are not required to contain them.

The `run_id` hashes all scientifically relevant fields. Performance-only choices
may be excluded only after tests prove that they do not change results. If chunk
size, dtype, backend, or device can alter numerical output beyond the declared
tolerance, it belongs in the identity.

## 7. Processing engine

### 7.1 Stage graph

The planner executes an arbitrary validated typed DAG. The following is the
built-in `sieve.profile.tensor_morlet.v1` graph, not a universal stage list:

```text
asset resolver
  -> media probe
  -> requested-plane frame decoder
  -> preprocessing
  -> tensor/base-field computation
  -> block reduction
  -> derived channel evaluation
  -> temporal/spectral transform
  -> threshold and spatial detection
  -> result/checkpoint writer
```

Every node has typed named ports, independent tests, structured timing, and no
knowledge of the CLI parser. Graphs may fork, join, retain parallel resolutions,
and connect non-motion nodes. The orchestration layer constructs the graph from
`RunSpec`; it never selects implementations with hard-coded method branches.

The numerical path never imports UI packages and never receives a list of
replicates.

### 7.2 Streaming and memory

The engine must process videos larger than RAM.

- Decode sequentially inside each scheduled segment.
- Keep only the temporal context required by preprocessing and spectral stages.
- Reduce to block fields as early as mathematical correctness permits.
- Chunk spectral work over block columns and time with explicit overlap.
- Commit only uncontaminated interior samples when a transform needs edge
  context.
- Write fixed-size series through memory-mapped `.npy` arrays or another simple
  chunked backend behind an artifact interface.
- Make large per-block band-power output opt-in.
- Enforce a memory budget before allocation and report the resolved chunk plan.
- Spilling from RAM to disk changes placement, not the stage graph or decoder
  route.

High-resolution input must not create a full-video, full-frame array in memory.

### 7.3 Decoder contract

The decoder yields requested named media planes plus:

```text
absolute frame index
exact presentation time derived from fps_num/fps_den
contiguous plane buffers with semantic plane ids
bit-depth/range/color descriptor
```

Possible outputs include encoded luma, RGB, alpha, or declared native planes.
HSV, Lab, linear-light luminance, and other derived spaces are explicit
preprocessing nodes. The planner requests only planes needed by downstream
nodes: grayscale work pays no mandatory RGB cost, while color analysis retains
the evidence it needs.

A requested backend either starts successfully and remains the recorded backend,
or the job fails. A caller may explicitly request another backend in a new run.

Decoder exhaustion before the verified requested end is truncation. The result
records the final decoded frame and leaves later coverage unexamined.

### 7.4 Preprocessing and masks

Initial built-in nodes include:

- Recorded luma and color interpretation.
- Optional downsampling in `(0, 1]`.
- Normalization: `off`, per-frame z-score, or CLAHE.
- Configurable spatial smoothing used by tensor estimation.
- Color-space conversion and hard or soft color predicates.
- Boolean selection masks, boolean validity masks, and nonnegative weight maps.
- Versioned morphology and connected-component filters.

Masks are typed artifacts, not pixels silently replaced by zero. Combining
masks or weights requires a declared operation. A color predicate specifies its
space and calibration, channel ranges, circular hue wrapping, soft/hard edge,
and morphology. This supports signals such as per-block red-area fraction,
red-component appearance, or motion restricted to a red selection.

No node state is shared between assets. Random-access segments declare their
lookback, lookahead, halo, warm-up, and finalization. A node that cannot
reproduce its state from bounded context declares that limitation so the
planner can reject incompatible random-access execution.

### 7.5 Fields and channels

The first profile supplies measurements such as:

- Mean intensity.
- Temporal change energy.
- Structure-tensor velocity components `u` and `v`.
- Tensor speed.
- Appearance/residual energy.
- Texture or spatial-confidence measures.

Registered channels may depend on media planes, masks, metadata, other fields,
other channels, or immutable models. They can be scalar, vector, tensor,
categorical, or embedding-valued and can live per pixel, block, region, frame,
interval, or sequence. Examples range from divergence over `u,v` to the visible
fraction of pixels satisfying a red-feather predicate.

The planner computes only the requested channel prerequisites. A `channels list
--json` command reports availability, units, signedness, parameters,
prerequisites, axes, dtype, validity, resampling rules, and implementation
versions. Core and third-party channels use the same registry.

```powershell
sieve channels list --json
sieve channels describe tensor_speed --json
```

### 7.6 Transforms and detectors

Temporal/spectral transforms and detectors are registered nodes, not permanent
engine stages. The initial rhythmic profile provides:

1. Per-block temporal band power over a chosen frequency interval.
2. A per-frame count of blocks inside a channel-value band.
3. A trailing or centered mean over a detection duration.
4. A count-band gate.
5. Largest connected in-band spatial component per frame.

Frequency is expressed in Hz, time in seconds, and spatial parameters in asset
pixels or blocks. The resolved frequency grid cannot exceed the usable Nyquist
range.

Windowed and whole-asset runs call the same functions. Segment overlap and edge
validity are explicit in the result coverage.

Other transform nodes may provide flash-duration filters, STFT/filterbanks,
state-space features, cross-channel relationships, or learned embeddings.
Other detectors may use rules, conjunctions, fitted signatures, classifiers, or
sequence models. They converge on stable output artifact types such as scores,
classes, gates, candidate intervals, coverage, and model identity; they need not
share internal mathematics.

### 7.7 Extension discovery

Python extensions register through the `antscihub_sieve.plugins` entry-point
group. Each namespaced node declares typed ports, parameter JSON Schema,
scientific identity, axes/units/validity, context requirements, determinism,
cache contribution, backend support, and a resource estimator.

```powershell
sieve plugins list --json
sieve plugins describe org.example.red_feather --json
sieve plugins validate PACKAGE_OR_ENV
sieve pipeline validate pipeline.json
sieve pipeline graph pipeline.json
sieve pipeline plan pipeline.json ASSET
```

Planning fails before decode on missing nodes/models, incompatible ports, axis or
unit mismatches, invalid parameters, and insufficient context. The detailed
extension and numerical contracts are in
`sieve-scientific-computation-contract.md`.

## 8. Output contract

An output directory contains:

```text
run.json                 canonical resolved RunSpec
summary.json             status, provenance, coverage, timing, intervals
series/
  score.npy              per-frame detector score
  gate.npy               per-frame decision
  count.npy              optional count series
  clump.npy              optional spatial-component series
band_power.npy           optional, potentially large
checkpoints/
logs/
```

Those series names are the tensor–Morlet profile's outputs. The general result
manifest lists arbitrary typed output artifacts by port identity, schema,
axes/coordinates, dtype, units, validity, coverage, and content hash. Clients
render by artifact type rather than assuming `score.npy` is always present.

`summary.json` includes:

```text
job status
run id and input identity
requested frame range
verified available frame range
processed and valid coverage intervals
truncation and cancellation state
detected intervals in frames and seconds
stage timings
achieved frames/second and realtime multiplier
peak resident memory
decoder and execution backend
warnings
paths and hashes for artifacts
```

Outputs are written transactionally. A completed marker is created only after
all required artifacts are closed and verified. Checkpoints are independently
valid, keyed by run id and covered interval, and safe to resume after scheduler
preemption.

## 9. Debuggability and observability

Debugging is a product requirement.

### 9.1 Resolved plan

```powershell
sieve analyze ASSET --params params.json --dry-run --json
```

reports:

- Resolved asset and lineage identity.
- Exact frame range and timebase.
- Stage graph.
- Channel prerequisites.
- Array shapes and dtypes at every boundary.
- Chunk and overlap sizes.
- Estimated peak memory and temporary storage.
- Expected outputs and cache/checkpoint decisions.
- Fully expanded decoder/encoder command arguments.

### 9.2 Stage control

Support:

```text
--stop-after NODE_INSTANCE_ID
--keep-intermediates
--start-from ARTIFACT
--trace-frames START:STOP
--trace-port NODE_INSTANCE_ID.PORT
--trace-coordinate AXIS=VALUE[,AXIS=VALUE...]
--debug-bundle PATH
```

An intermediate artifact contains its producing stage specification and input
hash. `--start-from` validates compatibility before use.

A debug bundle collects the resolved plan, structured logs, environment report,
small selected array slices, media probe output, and exact reproduction command.
It excludes full videos unless explicitly requested.

### 9.3 Structured errors

Errors have stable codes and context, for example:

```text
ASSET_SIDECAR_MISSING
ASSET_CONTENT_MISMATCH
PARENT_NOT_FOUND
LINEAGE_TRANSFORM_INVALID
DECODER_START_FAILED
DECODE_TRUNCATED
MEMORY_PLAN_EXCEEDS_LIMIT
CHANNEL_UNAVAILABLE
SPECTRAL_BAND_INVALID
CHECKPOINT_INCOMPATIBLE
OUTPUT_EXISTS_DIFFERENT_RUN
```

The CLI prints a concise human message and can emit the complete structured
record as JSON. Broad exception handlers may add context but must preserve the
original traceback in debug logs.

### 9.4 Timing

Every stage records wall time, CPU time where available, input/output counts,
bytes processed, and throughput. Progress emission is throttled so logging does
not dominate short or fast jobs.

## 10. Parameter files and composition

All scientific parameters are representable in JSON/YAML without Python code.

```powershell
sieve params validate params.json
sieve params resolve params.json --asset ASSET --json
sieve params diff a.json b.json
sieve params schema --json
```

Parameter and pipeline files may use named includes, but the resolved `RunSpec`
is fully expanded and self-contained. Results store the resolved graph and
values, not only source filenames.

Parameters are divided by stage so future optimization can vary downstream
choices without invalidating reusable upstream artifacts unnecessarily.

## 11. Datasets and annotations

Parameter search requires explicit datasets and labels.

### 11.1 Dataset manifest

A dataset is a versioned manifest of asset ids plus grouping metadata:

```text
dataset_id
assets[]
  asset_id
  sidecar/content identity
  include/exclude intervals
  session
  source recording
  colony
  individual
  treatment
  species
  arbitrary strata
grouping rules
annotation sets
```

Paths are resolved at execution time through explicit roots or a catalog. The
dataset identity hashes asset identities and interval selections, not their
locations.

### 11.2 Annotation format

Annotations are separate, versioned documents keyed to asset id and content
hash. They support:

- Positive behavior intervals.
- Explicit negative intervals.
- Unknown/unlabeled intervals.
- Point events when appropriate.
- Annotator identity and timestamp.
- Label taxonomy and version.
- Confidence and notes.

Unlabeled time is not automatically negative. Metric commands require an
explicit policy.

### 11.3 Dataset commands

```powershell
sieve dataset create assets.jsonl --out colony-study.dataset.json
sieve dataset inspect colony-study.dataset.json --json
sieve dataset verify colony-study.dataset.json --catalog assets.sqlite
sieve annotations validate labels.json --dataset colony-study.dataset.json
sieve annotations summarize labels.json --json
```

## 12. Parameter optimization architecture

Optimization is a planned consumer of the engine, so the initial interfaces must
support it even if the first release implements only manual parameter execution.

### 12.1 Separation of concerns

```text
engine                 evaluates one immutable RunSpec on one asset
dataset runner         evaluates a RunSpec over a dataset
metric evaluator       scores predictions against labeled intervals
study planner          generates candidate parameter sets and folds
study executor         distributes independent trials
study reducer          aggregates trials and selects candidates
model trainer          fits supervised signatures
```

The optimizer never calls numerical internals directly. It generates ordinary
pipeline/run documents and consumes ordinary outputs. A promoted result is
therefore executable by `sieve analyze` without an optimization-only code path.

### 12.2 Search spaces

A versioned search-space document declares:

```text
parameter path
type: categorical | integer | continuous | log-continuous
bounds or choices
conditional activation
node/subgraph affected
scientific description
```

Registered nodes contribute parameter schemas and optional search annotations;
the experiment chooses which ones to vary. A search space may also choose among
node implementations or graph branches. This permits method discovery—not just
tuning one detector—including color versus motion evidence, hard masks versus
soft likelihoods, alternative temporal transforms, single versus multiscale
representations, and rule-based versus learned detectors.

Potential axes include:

- Media plane, color space, value selection, or channel combination.
- Temporal sampling, lag, context, or transform.
- Downsample, pyramid, or spatial scale.
- Block size, masks, regions, or spatial pooling strategy.
- Feature function or learned representation.
- Tensor smoothing parameters.
- Frequency band and temporal scale.
- Channel-value threshold.
- Spatial count/clump threshold.
- Detection-window duration and alignment.
- Normalization mode.
- Detector family, topology, and model hyperparameters.

Search-space validation rejects combinations that violate Nyquist, geometry, or
stage constraints before scheduling compute.

### 12.3 Artifact reuse

The resolved graph exposes immutable, content-addressed artifacts at selected
boundaries. A study may reuse an artifact only when every upstream portion of
the candidate `RunSpec` matches its producer identity.

Examples:

- Threshold changes may reuse compatible band power.
- Detection-window changes may reuse compatible per-frame counts.
- Frequency-band changes may reuse a deliberately persisted multiband or
  transform representation if its schema supports the requested band.
- Channel, preprocessing, or spatial-scale changes require the appropriate
  upstream recomputation.

The planner prints the expected reuse and compute plan for every study. Reuse is
an optimization; disabling it must produce equivalent results within declared
numerical tolerance.

### 12.4 Metrics

Initial supervised metrics should include:

- Frame-level precision, recall, specificity, F1, and precision-recall AUC.
- False-positive and false-negative duration.
- Bout-level precision and recall under a declared overlap rule.
- Bout onset/offset error.
- Time-to-first-detection where relevant.
- Coverage and failure rate.
- Runtime, memory, and artifact storage as separate resource objectives.

Metric aggregation reports per-asset values and distributions before any global
mean. Class imbalance and unknown intervals are handled explicitly.

### 12.5 Data splitting and leakage prevention

Cross-validation groups must be chosen from biological and acquisition units,
not arbitrary frames. A study can group by source recording, session, colony,
individual, plate, experiment, or another dataset attribute.

Frames or nested crops descended from the same ancestor cannot appear on both
sides of a split unless the study explicitly permits it. The lineage graph makes
this enforceable. The study report records the grouping rule and exact fold
membership.

Maintain an immutable final test partition that is not used for parameter
selection.

### 12.6 Study CLI

Threshold and pipeline studies:

```powershell
sieve study plan `
  --dataset colony-study.dataset.json `
  --base-params params.json `
  --space search-space.json `
  --objective bout_f1 `
  --group-by source_asset `
  --folds 5 `
  --method grid `
  --out studies/flight-v1/

sieve study run studies/flight-v1/plan.json --workers 8
sieve study run studies/flight-v1/trials-003.jsonl
sieve study summarize studies/flight-v1/ --out report.json
sieve study best studies/flight-v1/ --out candidate.params.json
sieve study evaluate studies/flight-v1/ `
  --params candidate.params.json `
  --partition test
```

Supported planning methods may grow from grid and random search to Bayesian or
multi-fidelity methods. The method, library version, and random seed are part of
the study identity.

A study plan can be sharded into JSONL trial files for a cluster. Trials are
independent and resumable. Reducers verify dataset, fold, metric, and run
identities before combining results.

### 12.7 Supervised signatures

The long-view supervised path fits interpretable behavior signatures over a
feature basis spanning:

```text
channel x spatial scale x temporal/frequency scale
```

Initial model families:

- Linear discriminant analysis.
- L1-regularized logistic regression.

The sparse weight vector identifies which channels and scales contribute to the
behavior and can be inspected, versioned, and applied without a graphical
session.

Feature extraction is a headless command:

```powershell
sieve features build `
  --dataset colony-study.dataset.json `
  --basis feature-basis.json `
  --out features/flight-v1/

sieve signature fit `
  --features features/flight-v1/ `
  --annotations labels.json `
  --model logistic-l1 `
  --group-by source_asset `
  --folds 5 `
  --out models/flight-v1/

sieve signature evaluate models/flight-v1/model.json `
  --dataset colony-test.dataset.json
```

A model artifact records:

```text
model id and version
training dataset and annotation identities
feature schema and ordering
preprocessing and feature-producing RunSpec fragments
normalization fitted on training folds only
coefficients/intercepts
regularization and hyperparameters
fold assignments and metrics
software versions
```

Applying a signature uses the same engine and result contract:

```powershell
sieve analyze ASSET --model models/flight-v1/model.json --out result/
```

Its output remains per-frame score, gate, detected intervals, coverage, and
provenance. There is no separate production implementation of the fitted model.

### 12.8 Multi-objective and multi-fidelity search

Scientific quality and compute cost are reported separately. The system may
produce Pareto candidates over accuracy, throughput, memory, and storage, but it
does not silently combine them into one score.

Multi-fidelity studies may begin with shorter intervals, fewer folds, or lower
resolution and promote candidates to more expensive evaluations. Every fidelity
level is explicit. A candidate cannot be reported as full-dataset performance
when it was evaluated on a sample.

### 12.9 Unsupervised extension

The feature export contract should permit a later unsupervised behavior map:

- Multi-channel, multi-scale temporal features.
- Optional dimensionality reduction.
- Embedding and density estimation.
- Clustering or watershed-like segmentation into recurring states.
- Mapping state assignments back to asset time.

This is not required for the initial detector. The architectural requirement is
that feature arrays have stable schemas, asset/time identity, and chunked export
so they can be consumed without reopening videos through a separate pipeline.

## 13. Module boundaries

Suggested package structure:

```text
src/antscihub_sieve/
  cli/
    main.py
    asset_commands.py
    derive_commands.py
    analyze_commands.py
    batch_commands.py
    study_commands.py

  domain/
    asset.py
    lineage.py
    coordinates.py
    run_spec.py
    coverage.py
    results.py

  plugins/
    api.py
    manifests.py
    discovery.py
    conformance.py

  pipeline/
    schema.py
    graph.py
    ports.py
    validation.py
    resolution.py

  media/
    probe.py
    decode.py
    encode.py

  engine/
    planner.py
    executor.py
    streaming.py
    resources.py

  builtins/
    media_nodes.py
    color_nodes.py
    mask_nodes.py
    reduction_nodes.py
    tensor_nodes.py
    morlet_nodes.py
    detection_nodes.py
    profiles.py

  artifacts/
    identity.py
    arrays.py
    checkpoints.py
    transactions.py

  datasets/
    manifest.py
    annotations.py
    splits.py
    metrics.py

  optimization/
    spaces.py
    studies.py
    trials.py
    signatures.py

  observability/
    events.py
    timing.py
    debug_bundle.py
```

Dependency direction:

```text
domain <- media
domain <- plugins <- pipeline
domain <- engine <- artifacts
pipeline <- engine
domain <- datasets <- optimization
application orchestration -> all above
CLI -> application orchestration
```

The domain and plugin-API layers import no decoder, numerical backend,
filesystem database, or CLI parser. Built-in methods depend on the same plugin
API exposed to external packages. Optimization consumes public pipeline and
application contracts and does not reach into node-private functions.

## 14. Test strategy

### 14.1 Asset and lineage tests

- Sidecar creation and atomic replacement.
- Video-plus-sidecar relocation.
- Parent resolution by id after paths change.
- Missing parent still exposes recorded lineage.
- Nested crop coordinate composition.
- Full-frame child derivation.
- Odd crop origins and dimensions.
- Parent content mismatch rejection.
- Catalog rebuild from multiple roots.

### 14.2 CLI contract tests

- Every command has stable help and JSON Schema output.
- Configuration precedence is deterministic.
- Unknown keys and incompatible arguments fail before compute.
- `--dry-run` performs no frame decode.
- Structured stdout remains parseable while progress is emitted.
- Exit codes distinguish usage and execution failures.
- Reproduction commands from debug bundles execute successfully.

### 14.3 Numerical tests

- Plugin manifest, port compatibility, and graph-validation fixtures.
- A test-only external plugin adds a preprocessor and channel without core edits.
- Color conversion, circular hue wrap, hard/soft masks, mask composition, and
  weighted area reduction.
- A red-flash synthetic fixture is detected through a color-only pipeline while
  tensor nodes are absent from the plan.
- Synthetic translations with known `u,v` and speed.
- Constant frames produce zero temporal change.
- Derived-channel prerequisite selection.
- Window and full-range overlap agreement.
- Chunk-size and RAM/spill equivalence within declared tolerance.
- Spectral response at known frequencies.
- Count, window, gate, interval, and connected-component behavior.
- Rational frame/time conversion.
- Gray8/gray16 range handling.
- Partial edge-block policy.

### 14.4 Failure and lifecycle tests

- Decoder startup failure.
- Truncation before a verified endpoint.
- Cancellation at every stage.
- Process termination during checkpoint write.
- Resume from the last complete checkpoint.
- Insufficient RAM/disk planning.
- Existing output with a different run id.
- No child or result presented as complete after failure.

### 14.5 Batch and cluster tests

- Stable sorting and sharding.
- One asset failure does not stop unrelated assets unless fail-fast is requested.
- Resume skips only identical completed runs.
- Duplicate asset ids are detected.
- Worker and thread limits are respected.
- Trial reducers reject mixed study or dataset identities.

### 14.6 Optimization tests

- Search spaces generate only valid `RunSpec` candidates.
- Study planning is deterministic for a fixed seed.
- Descendants of the same ancestor remain in the same fold.
- Normalization and feature selection are fit only on training folds.
- Metrics exclude unknown intervals according to declared policy.
- Reused and recomputed artifacts give equivalent trial outputs.
- A promoted parameter file reproduces its recorded candidate.
- A fitted signature produces the same score through `signature evaluate` and
  `analyze --model`.

### 14.7 Performance tests

Maintain representative assets across resolution, duration, bit depth, and
channel cost. Record stage throughput, end-to-end throughput, peak memory, and
temporary storage. Performance tests verify:

- One-asset plans do not grow with the number of sibling assets.
- Progress logging has bounded overhead.
- Memory stays within the resolved budget.
- Batch parallelism improves corpus throughput without changing per-asset
  results.
- A client invoking the CLI cannot cause scientific work to move into the
  client process.

Numeric performance thresholds should be established from the target machines
and representative footage, then stored with the benchmark definitions rather
than embedded as universal constants.

## 15. Delivery sequence

### Milestone 1: asset model and media tools

- Asset schema, probing, verification, and catalog.
- Recursive lineage and coordinate composition.
- Exact spatial derivation with transactional outputs.
- CLI JSON contracts and structured logging.

### Milestone 2: extensible single-asset engine

- Typed DAG, plugin API/entry-point discovery, graph validation, and profile
  resolution.
- Streaming requested-plane decode, including luma and color evidence.
- First-class preprocessing, masks, fields, reducers, channels, transforms, and
  detectors.
- Memory planning, artifacts, checkpoints, and debug bundles.
- `analyze`, `pipeline`, `plugins`, `channels`, and `params` commands.

### Milestone 3: built-in scientific profiles

- Conformant tensor–Morlet reference profile with exact numerical tests.
- Color predicate, mask composition, area/channel reducers, and brief-flash
  reference pipeline.
- Demonstrate that an external test plugin adds a new channel and preprocessor
  without changing core modules.

### Milestone 4: corpus execution

- Dataset manifests.
- Deterministic batch, shard, and job-plan commands.
- Cluster-safe resume and aggregation.
- Resource planning for high-resolution footage.

### Milestone 5: evaluation foundation

- Annotation schema.
- Metrics and grouped dataset splits.
- Feature export with stable schemas.
- Study planning and grid/random threshold search.

### Milestone 6: supervised parameter optimization

- Distributed trials and artifact reuse.
- LDA and L1-logistic signature fitting.
- Cross-validation, held-out evaluation, model artifacts, and promotion.
- `analyze --model` through the standard result path.

### Milestone 7: optional clients and broader modeling

- A graphical client may invoke CLI jobs and render structured progress/results.
- Multi-objective and multi-fidelity study methods.
- Unsupervised feature embedding and state discovery.

No client milestone may change the CLI's scientific result contract. If a client
needs a capability, add it to the headless command or job protocol first.

## 16. Completion criteria

The system is operational when:

- Any video can be registered as source footage or a replicate.
- Any registered asset can be analyzed full-frame or cropped into child assets.
- Nested children retain navigable, location-independent lineage.
- Analysis of a child depends only on that child asset and its sidecar.
- The CLI can process very large videos with bounded RAM and resumable outputs.
- The same noninteractive job documents run locally or through a cluster
  scheduler.
- Results carry exact input, parameter, software, coverage, and artifact
  identity.
- Stage plans, logs, intermediates, checkpoints, and debug bundles make failures
  localizable.
- Dataset and artifact schemas are ready for reproducible parameter studies.
- New scientific nodes can be installed, discovered, planned, cached, debugged,
  and optimized without modifying the core executor or CLI.
- Color-only and motion-only reference pipelines both run through the same
  public contracts.
- Parameter candidates and fitted signatures can be promoted directly into the
  canonical analysis command.

The central abstraction is a video asset, not a project directory and not a
multi-replicate processing session. Spatial derivation creates another asset;
analysis consumes one asset; datasets and optimization orchestrate many such
jobs without changing what any individual job means.
