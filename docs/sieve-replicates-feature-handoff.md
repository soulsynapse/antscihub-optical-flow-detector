# antscihub-SIEVE: Replicates Feature Handoff

**Signal Isolation for Ethological Video Events**

This is the complete implementation scope for the first SIEVE feature. Build a
working command-line asset/derivation system and a working desktop GUI containing
only the Replicates workspace. Stop after this feature is usable and tested by
the user.

Do not implement video analysis, preprocessing, channels, tensors, wavelets,
detection, parameter optimization, datasets, annotations, a plugin framework,
or placeholder tabs. Preserve clean seams for later features, but do not build
their machinery now.

The implementation order is mandatory:

1. Implement and test the headless commands.
2. Implement the Replicates GUI over the same application services and command
   contracts.
3. Present the working feature to the user for hands-on validation.
4. Do not begin another feature until the user accepts this one.

## 1. Outcome

At completion, a user can:

1. Open any video.
2. Declare that its full frame is either source footage or already a replicate.
3. Play, pause, step, and scrub through the video.
4. Draw, stamp, select, rename, move, and remove proposed child regions.
5. Save and reopen that draft layout automatically.
6. Materialize selected regions as independently portable child video assets.
7. Click a created replicate box to open that child as the current asset.
8. See that the child has a parent and navigate back when the parent is
   reachable.
9. Treat that child as a new parent and crop it again using the identical UI and
   command path.

The application does not have a source-versus-crop processing setting. A child
clip is an asset. Opening it means that later analysis will decode that clip
itself.

Rectangles exist only inside the Replicates workspace and its editable layout
document. They are not an input type for preprocessing or analysis. No later
feature receives a box collection, crops a parent on demand, or switches between
two execution routes.

## 2. Core model

### 2.1 Asset kind is descriptive

Every registered video asset has:

```text
kind = source | replicate
```

- `source` means the user currently regards the video as parent footage.
- `replicate` means the user regards the complete frame as an experimental unit.

The kind does not select a different implementation. Both kinds open in the
same Replicates workspace, both can be used whole, and both can produce spatial
children. There is no root-only crop path and no special sub-replicate path.

### 2.2 The selected asset owns the frame

The current asset is always displayed in its own pixel coordinates. The GUI
does not display a parent frame and pretend a child is the active video. If a
child is opened, frame `(0,0)` is the top-left of the child video.

Only one asset is open in a GUI session. Sibling assets and parent geometry do
not enter its decoder.

### 2.3 Draft regions are not child assets

A drawn rectangle is a `draft_region`: an editable instruction for a possible
crop. It has no scientific identity and is not presented as a completed
replicate.

A region becomes a child asset only after `sieve derive` successfully creates,
closes, probes, verifies, hashes, and publishes its video and asset sidecar.

This distinction is visible in the GUI:

- **Draft regions** are editable solid boxes.
- **Created children** are immutable lineage records. They may be shown as
  non-editable dashed boxes in their parent's Replicates workspace and opened
  as assets.

Moving a created child's rectangle does not mutate that child. To try different
geometry, duplicate its rectangle as a new draft, edit the draft, and create a
new child. This avoids carrying future measurements onto different pixels.

### 2.4 Clicking a replicate means opening its asset

For a created child, “zoom into this replicate” is asset navigation, not a
display-only crop of the parent:

```text
click created replicate box
  -> resolve and verify child asset
  -> close parent MediaSession
  -> open child through AssetService
  -> create child MediaSession
  -> display frame decoded from the child video file
```

Opening the child's video or sidecar from **Open…** invokes the exact same
`open_asset(child)` operation. Both routes must produce the same current asset,
decoder, frame count, dimensions, pixel display, layout, and parent breadcrumb.
There is no “opened from box” mode.

This is a correctness and performance boundary. The GUI must not continue
decoding the enclosing parent and cropping it in memory after entering a child.
Later computation will receive only the child asset reference and will likewise
open only the child video.

A draft has no child video to open. Clicking a draft may temporarily focus the
parent display on its rectangle to aid editing, but the UI must label that state
as a **draft preview**. It is never presented as equivalent to opening a child.
Once derivation succeeds, clicking that rectangle opens the resulting child.

### 2.5 Recursive derivation

Every crop is expressed in the immediate parent's coordinates:

```text
parent asset A --crop--> child B --crop--> child C
```

Child C records B as its immediate parent and snapshots the ancestor chain back
to A. The same `derive` implementation performs both crop operations.

### 2.6 Independent regions may overlap

Draft boxes and child boxes may overlap. They are independent candidate assets,
not ownership cells in a shared processing atlas. Touching, nesting, and positive
area overlap are all valid.

The GUI may warn about exact duplicates, but it must not reject overlap merely
because a previous implementation required non-overlapping processing regions.

## 3. Files and identity

### 3.1 Asset sidecar

For:

```text
D:/footage/colony_07.mp4
```

the sidecar is:

```text
D:/footage/colony_07.asset.json
```

Minimum schema:

```json
{
  "schema_version": 1,
  "asset_id": "uuid",
  "kind": "source",
  "label": "colony 07",
  "media": {
    "filename": "colony_07.mp4",
    "content_sha256": "...",
    "size_bytes": 0,
    "width": 0,
    "height": 0,
    "pixel_format": "...",
    "codec": "...",
    "fps_num": 0,
    "fps_den": 1,
    "container_frame_count": null,
    "decoded_frame_count": null,
    "duration_seconds": 0.0
  },
  "lineage": {
    "parent": null,
    "derivation": null,
    "ancestors": []
  },
  "calibration": {},
  "attributes": {}
}
```

Asset identity is the UUID plus verified content identity, never a path. The
media filename is sidecar-relative by default. Moving a video with its sidecar
must preserve identity.

Writes are atomic: write a sibling temporary file, flush and close it, then
replace the destination. A malformed existing sidecar produces a visible error;
it is not overwritten as though the video were new.

### 3.2 Draft layout sidecar

The editable GUI state lives separately at:

```text
D:/footage/colony_07.replicate-layout.json
```

Minimum schema:

```json
{
  "schema_version": 1,
  "layout_id": "uuid",
  "parent": {
    "asset_id": "uuid",
    "content_sha256": "...",
    "width": 4096,
    "height": 2160
  },
  "next_display_number": 4,
  "draft_regions": [
    {
      "region_id": "uuid",
      "label": "rep3",
      "box_xyxy": [200, 100, 700, 650],
      "color": "#ffd24a",
      "created_utc": "...",
      "updated_utc": "..."
    }
  ],
  "created_children": [
    {
      "region_snapshot": {
        "region_id": "uuid",
        "label": "rep1",
        "box_xyxy": [20, 40, 500, 600],
        "color": "#ff5a5a"
      },
      "child": {
        "asset_id": "uuid",
        "content_sha256": "...",
        "location_hints": ["relative/or/previous/path"]
      },
      "created_utc": "..."
    }
  ]
}
```

The layout is an editable work document and convenience index, not the source
of child identity. The child's own asset sidecar is authoritative. A stale or
missing child location hint does not erase the child record.

The layout must match the current parent asset id, content hash, and dimensions
before it is loaded. A layout from another asset can be explicitly imported as
a geometric template only after validating that every box fits the new frame.
Import assigns new region ids and does not copy created-child records.

`next_display_number` only creates convenient default labels such as `rep4`.
Region and asset UUIDs are never recycled after deletion.

### 3.3 Child package

The default batch output is one directory per child:

```text
chosen-output/
  rep1--a1b2c3d4/
    video.mkv
    video.asset.json
  rep2--e5f6a7b8/
    video.mkv
    video.asset.json
```

The human label influences the directory name but is not identity. Sanitization
must be deterministic. The short id suffix prevents duplicate labels from
colliding.

The child asset sidecar records:

```text
kind: replicate
parent.asset_id and parent.content_sha256
parent label/kind snapshots
optional location hints
derivation.operation: crop
derivation.parent_box_xyxy
derivation.output_width/output_height
exact child-to-parent translation
frame_start and frame_count
encoder id and complete arguments
created timestamp
ancestor identity snapshots
```

This milestone always preserves the complete temporal axis and does not resize
the crop. The child-to-parent transform is therefore:

```text
x_parent = x_child + x0
y_parent = y_child + y0
```

## 4. Coordinate rules

Persistent region coordinates are integer, half-open parent pixels:

```text
(x0, y0, x1, y1)
0 <= x0 < x1 <= width
0 <= y0 < y1 <= height
```

The display may use floating-point normalized coordinates during a gesture, but
it resolves all four edges to integer pixels before updating the layout. The
resolved box shown in the inspector is the box sent to `derive`.

Rules:

- Drawing works regardless of display scaling, letterboxing, or focused zoom.
- Clicking on black letterbox margins never creates or moves a box.
- A drag normalizes direction, so any corner can be drawn toward any other.
- Minimum size is at least one source pixel in each dimension.
- Moving clamps the origin at frame boundaries and preserves width and height.
- Moving does not silently resize.
- A fixed-size stamp uses the selected draft's resolved pixel width and height.
- A stamp near an edge shifts inward while preserving its size.
- Selecting or zooming never changes stored coordinates.
- Frame changes never change stored coordinates.

Odd crop origins and dimensions are legal at the domain level. An encoding
profile must either preserve the exact requested crop or fail planning with a
clear explanation. It may not silently round the rectangle for chroma alignment.

## 5. Required headless surface

The executable is `sieve`; the Python distribution is `antscihub-sieve`; the
import package is `antscihub_sieve`.

### 5.1 Asset commands

```powershell
sieve asset inspect VIDEO_OR_SIDECAR --json
sieve asset init VIDEO --kind source --label "colony 07" --json
sieve asset init VIDEO --kind replicate --label "replicate 27" --json
sieve asset verify ASSET --level metadata|quick|full --json
sieve lineage show ASSET --json
sieve lineage parent ASSET --json
sieve lineage parent ASSET --locate PATH --json
```

Requirements:

- `inspect` never changes files.
- `init` fails if an incompatible sidecar already exists.
- Noninteractive `init` never guesses `kind`.
- `lineage parent` can report a known-but-unreachable parent distinctly from a
  root asset.
- `--locate` verifies identity before accepting a parent selected by the user.

### 5.2 Layout commands

```powershell
sieve layout inspect PARENT_OR_LAYOUT --json
sieve layout add PARENT --box x0,y0,x1,y1 --label rep1 --json
sieve layout update PARENT --region-id UUID --box x0,y0,x1,y1 --json
sieve layout rename PARENT --region-id UUID --label NAME --json
sieve layout remove PARENT --region-id UUID --json
sieve layout clear PARENT --json
sieve layout import PARENT TEMPLATE.json --json
sieve layout export PARENT --out TEMPLATE.json --drafts-only --json
sieve layout validate PARENT_OR_LAYOUT --json
```

These commands and the GUI call the same layout application service. GUI code
must not contain a second implementation of coordinate validation or atomic
persistence.

### 5.3 Media-preview commands

```powershell
sieve media probe ASSET --json
sieve media frame ASSET --frame 0 --out frame.png --json
sieve media frame ASSET --time 12.5s --out frame.png --json
```

These commands prove that opening, probing, time/frame resolution, decoding,
and color display exist headlessly. The GUI may keep one decoder session open
through the same media service for responsive playback; it must not launch a new
process for every frame.

Frame rate is stored and calculated as an exact rational. The GUI may display a
rounded value but does not use it for seeking or timestamps.

### 5.4 Derivation commands

Single region:

```powershell
sieve derive PARENT `
  --crop x0,y0,x1,y1 `
  --label rep1 `
  --kind replicate `
  --out OUTPUT_DIR `
  --profile lossless `
  --json
```

Draft layout:

```powershell
sieve derive PARENT `
  --layout PARENT.replicate-layout.json `
  --region-id UUID `
  --region-id UUID `
  --out OUTPUT_DIR `
  --profile lossless `
  --json
```

Planning and verification:

```powershell
sieve derive PARENT --layout FILE --out DIR --profile lossless --plan --json
sieve derive verify CHILD_OR_BATCH_RESULT --json
```

Named profiles in this milestone are `lossless`, `high-quality`, and `compact`.
The GUI defaults to `lossless` so later color and appearance experiments do not
lose evidence accidentally. Profiles resolve to explicit codec, pixel format,
color metadata, and arguments visible in `--plan` and recorded in the child.

Derivation emits structured progress records containing at least:

```text
job id
region id and label
phase: planning | encoding | verifying | publishing | complete | failed
frames completed and expected
fraction complete when known
message and structured error when applicable
completed child identity/path when successful
```

One child is transactional. It is written under a temporary sibling path and is
published only after video closure, probing, frame-count verification, hashing,
and sidecar validation. Cancellation removes incomplete temporary output.
Already completed valid siblings remain completed and are reported honestly.

Re-running the same layout must not silently overwrite children. If the layout
already records a verified child for the identical region snapshot, it is
reported as existing. Changed geometry creates a new child identity.

### 5.5 CLI behavior

- Human status goes to stderr; JSON results go to stdout.
- `--log-format json` uses newline-delimited structured progress on stderr.
- Exit 0 is success, 1 is execution failure, and 2 is invalid usage.
- Unknown keys and invalid coordinates fail before encoding.
- `--plan` performs no encoding and writes no child.
- Ctrl+C requests cooperative cancellation and leaves no incomplete published
  child.
- Every error includes a stable code, human message, and relevant asset/region.

## 6. Application boundaries

Use a small public application layer shared by CLI and GUI:

```text
AssetService
  inspect, initialize, verify

LineageService
  describe, resolve_parent, compose_transform

LayoutService
  load, create, add, update, rename, remove, import_template, save

MediaSession
  probe, read_frame, seek, close

DerivationService
  plan, run, cancel, verify
```

The domain layer knows assets, coordinates, lineage, layouts, and validation. It
imports neither Qt nor a decoder. Media and derivation code import no Qt. CLI
imports no Qt. GUI widgets translate user gestures into application-service
calls and render returned state; they do not own persistence rules.

The GUI can run `derive` as an isolated child process and consume its structured
progress protocol. This is preferred for long encoding because a crash or
cancellation cannot take down the GUI. Frame playback uses one local
`MediaSession`, since process-per-frame playback would be unusable.

There is no global project state hub. The Replicates workspace owns one explicit
`AssetSession` containing the current immutable asset descriptor, current frame,
one decoder handle, and current layout document. Opening another asset closes
and replaces that session.

## 7. Desktop GUI

Use PyQt6 for this milestone unless the new repository has already selected
another desktop toolkit. The entry point is:

```powershell
sieve-gui
```

The window contains one workspace named **Replicates**. Do not create disabled
Preprocessing, Detection, Optimization, or other future tabs.

### 7.1 Empty state

The empty window contains:

- **Open video or asset…**
- A short statement: “Open source footage or an existing replicate.”
- A recent-file list is optional and must not become authoritative state.

Opening accepts a video or `.asset.json` sidecar. Drag-and-drop support is
desirable but not required for acceptance.

### 7.2 First open of an unregistered video

Show one blocking choice after the media has been probed:

> What does this entire video represent?

Two explicit choices:

- **Source footage — I want to create replicates from it**
- **Replicate — the whole video is already one replicate**

Also request an editable label. Cancel leaves the video unregistered and writes
nothing. Confirmation calls the same initialization service as `sieve asset
init`. On later opens, the sidecar answers this question; never prompt again
unless the user deliberately edits asset metadata.

Both choices lead to the same Replicates workspace. A replicate remains
croppable, so the interface never reaches a dead end.

### 7.3 Window structure

Top asset bar:

```text
[Open…]  label  [SOURCE|REPLICATE]  dimensions · fps · duration
Parent: label/status  [Open parent] [Locate parent…]
```

- Root assets say `Parent: none`.
- Reachable children enable **Open parent**.
- Known but unreachable parents show their recorded label/id and enable
  **Locate parent…**.
- Parent resolution verifies id and content hash before opening.

Main area:

```text
+--------------------------------------+------------------------------+
|                                      | Draft regions                |
|             video view               | [region list]                |
|    with draft/child box overlays     | Rename  Delete  Clear        |
|                                      |                              |
|                                      | Created children             |
|                                      | [child list]                 |
|                                      | Open  Locate  Duplicate box  |
+--------------------------------------+------------------------------+
| Play/Pause     scrubber      time/frame                            |
+--------------------------------------------------------------------+
| Draw boxes: ON   Fixed-size stamp   [Create child replicates…]      |
+--------------------------------------------------------------------+
| status / derivation progress / Cancel                               |
+--------------------------------------------------------------------+
```

The video receives most horizontal space. The layout must remain usable on a
normal laptop display and may collapse the inspector below the video at narrow
widths.

### 7.4 Video transport

- Play/Pause button and Space shortcut.
- Left/Right step one frame; Shift+Left/Right step ten frames.
- Clicking the scrubber jumps to that position.
- Dragging the scrubber updates the time label immediately but coalesces random
  decode requests; decode only the latest settled request.
- Display current and total time plus exact frame index in a tooltip or label.
- Playing from the last frame restarts at frame zero.
- Playback stops at the final frame rather than wrapping continuously.
- Opening another asset stops playback and releases the previous decoder.
- Decode/render errors appear in the status area without freezing the GUI.

The displayed image preserves aspect ratio and uses letterboxing. Coordinate
mapping is always based on the actual drawn-image rectangle, not the widget
bounds.

### 7.5 Draft-region interaction

Use a stable visible palette for successive boxes.

- **Draw mode:** left-drag empty space to create a region.
- New regions receive monotonic default labels `rep1`, `rep2`, and so on.
- Drawing selects the new region.
- **Fixed-size stamp:** when enabled, a short click on empty video space creates
  a region with the selected draft's pixel dimensions, centered on the click
  and shifted inside frame edges as necessary.
- Stamp size is derived from the currently selected draft, not separately
  stored state.
- Pressing inside a draft selects it immediately.
- Releasing a draft without a meaningful drag selects it and enters a visibly
  labeled draft-preview focus.
- Dragging repositions it; the preview follows the pointer and the release uses
  the release position, even if intermediate mouse-move events were compressed.
- A movement threshold distinguishes click from drag.
- Right-click during a drag cancels that drag.
- Right-click otherwise exits draft-preview focus or returns from a child to
  its parent when that parent is reachable.
- Delete removes the selected draft after confirmation.
- Rename changes only the label, never the region id.
- Clear removes drafts after confirmation and does not remove created children.
- Selection is synchronized between box and list.
- Clicking a draft-list row selects it and enters draft-preview focus.
- Draft-preview focus uses the exact rectangle without changing stored
  coordinate space.
- Hold Shift to temporarily hide overlays for an unobstructed pixel view;
  hidden boxes are not hit-testable.

Every completed edit autosaves atomically. There is no ordinary Save button to
forget. **Export template…** and **Import template…** are separate explicit
operations.

### 7.6 Created-child interaction

After derivation succeeds:

- Move the region snapshot from the draft list to the created-child list.
- Show label, dimensions, output location, and verified status.
- Draw its parent rectangle as a non-editable dashed overlay.
- Clicking its dashed box, double-clicking its row, or pressing **Open** all call
  the identical `open_asset(child)` operation and replace the current asset
  session with the child's own video.
- **Locate** repairs only a non-authoritative location hint after verifying
  identity.
- **Duplicate box as draft** creates a new editable region with a new region id
  at the same coordinates.

Do not provide a button that edits a created child's crop in place. Do not
delete child video packages as part of draft deletion or Clear. Deleting or
archiving assets is a separate future feature with its own safety contract.

Created-child rectangles remain a Replicates-workspace navigation aid only.
They are not copied into the child's scientific inputs or exposed to future
processing nodes.

The child's `derivation.parent_box_xyxy` is immutable provenance explaining how
the child file was made. It is not an active region instruction. Future
processing may report lineage, but its input contract is exactly:

```text
analyze(asset_reference)
```

It is never:

```text
analyze(parent_asset, replicate_box)
analyze(parent_asset, selected_region_id)
```

The future computation package must not import the layout module. Transitioning
from this workspace to any later feature passes the selected child's asset
reference, just as choosing that child video with **Open…** would.

### 7.7 Create-child dialog and progress

**Create child replicates…** is enabled when at least one draft exists. The
dialog shows:

- Selected drafts, labels, and resolved pixel boxes.
- Output root.
- Encoding profile with `lossless` selected by default.
- Planned output dimensions and paths.
- Estimated storage when available.
- Any collisions or invalid regions before Start is enabled.

Starting launches the headless derivation job. While it runs:

- Disable layout mutations and opening another asset.
- Keep the UI event loop responsive.
- Show the active child, phase, frame progress, and aggregate progress.
- Offer **Cancel**.
- Stream structured errors into a concise status display with an expandable
  detail view.

On completion or cancellation, reload the layout from disk and display exactly
which children completed. Never claim that the entire batch completed if only
some children did.

## 8. Failure behavior

Required stable error classes include:

```text
ASSET_SIDECAR_MISSING
ASSET_SIDECAR_INVALID
ASSET_CONTENT_MISMATCH
MEDIA_PROBE_FAILED
FRAME_DECODE_FAILED
LAYOUT_PARENT_MISMATCH
LAYOUT_COORDINATES_INVALID
PARENT_NOT_FOUND
PARENT_IDENTITY_MISMATCH
OUTPUT_COLLISION
ENCODER_START_FAILED
ENCODE_FAILED
DECODE_TRUNCATED
DERIVATION_VERIFY_FAILED
DERIVATION_CANCELLED
```

Failures must be local and diagnosable:

- A bad layout does not prevent opening its parent video; show the layout error
  and offer to choose another layout or start a new empty one.
- An unreachable parent does not prevent opening a child.
- One failed child does not invalidate verified siblings.
- Temporary output is never displayed as a child.
- The original parent video is opened read-only and never rewritten.
- Tracebacks and exact encoder arguments go to debug logs; dialogs use concise
  language and stable codes.

## 9. Suggested package boundaries

Build only what this feature needs:

```text
src/antscihub_sieve/
  domain/
    asset.py
    geometry.py
    lineage.py
    layout.py

  media/
    probe.py
    session.py
    derive.py
    profiles.py

  persistence/
    json_atomic.py
    identity.py

  application/
    assets.py
    lineage.py
    layouts.py
    derivation.py

  cli/
    main.py
    asset_commands.py
    layout_commands.py
    media_commands.py
    derive_commands.py

  gui/
    main.py
    main_window.py
    replicate_workspace.py
    video_panel.py
    frame_view.py
    derive_job.py
```

Tests mirror these boundaries. Do not create `tensor.py`, `channels.py`,
`detection.py`, empty future packages, or a generic plugin system in this
milestone.

Future computation will accept an asset id/path and operate on that asset's full
frame. The only seam this feature owes it is a stable, verified asset contract.

## 10. Tests required before GUI handoff

### 10.1 Domain and persistence

- Asset sidecar round-trip and atomic replacement.
- Malformed sidecar is reported and not overwritten.
- Video plus sidecar can move together without identity change.
- Half-open coordinate validation at every edge.
- Overlapping and nested draft regions are accepted.
- Region ids and display counters are never recycled.
- Layout parent mismatch is rejected.
- Template import assigns new region ids and drops child records.
- Child-to-parent and grandchild-to-root coordinate composition is exact.

### 10.2 Media and derivation

- Probe records rational fps and dimensions.
- Frame zero, a middle frame, and final frame decode.
- Odd-origin/odd-size crop is exact or planning rejects the selected profile.
- Child dimensions equal `x1-x0` by `y1-y0`.
- First and final child frames correspond to the parent crop.
- Child retains the full verified temporal range.
- Child sidecar records parent and ancestor identity.
- A child can be used as a parent by the identical derive command.
- Ctrl+C/cancellation removes incomplete temporary output.
- Simulated failure before publish leaves no completed child.
- Multiple regions may complete independently and report mixed outcomes.
- Identical rerun recognizes an existing verified child rather than overwriting.

### 10.3 CLI

- Every required command has `--help` and parseable `--json` output.
- Unknown arguments and invalid boxes fail before encoding.
- `--plan` writes no child video.
- Human logs do not corrupt JSON stdout.
- Exit codes distinguish invalid usage and execution failure.
- Commands work without Qt installed or a graphical display.

### 10.4 GUI

Use offscreen Qt for interaction tests where possible.

- Opening an unknown video asks source versus replicate exactly once.
- Canceling registration writes nothing.
- Opening a known asset restores its correct layout only.
- Switching assets closes the old decoder and clears selection/focus.
- Mouse-to-source coordinates remain correct with letterboxing and focus zoom.
- Draw in all directions; minimum and edge boxes resolve correctly.
- Selection synchronization between list and overlay.
- Stamp follows selected draft size and clamps without resizing.
- Click versus move threshold and release-position behavior.
- Right-click cancel and right-click unzoom.
- Delete, rename, clear, autosave, export, and import.
- Overlapping boxes remain editable.
- Created-child boxes cannot be moved.
- Clicking a created-child box and opening its video/sidecar directly resolve to
  the same asset and decode identical displayed frames.
- After clicking a created child, the active decoder reads the child file and
  does not read or crop the parent.
- Derivation progress does not block the event loop.
- Cancel and failure never display temporary output as complete.
- Open child, show parent, open parent, then crop the child again.

## 11. Manual acceptance script

The implementation is ready for the user's validation only when this exact
workflow succeeds:

1. Launch `sieve-gui` in a clean directory.
2. Open an unregistered source video.
3. Choose **Source footage** and confirm its sidecar was created.
4. Scrub to several distant frames and play/pause without a decode backlog.
5. Draw one box, then stamp several copies.
6. Select an earlier box and verify the stamp changes to that selected size.
7. Rename and reposition boxes, including at a frame edge.
8. Create two overlapping boxes and confirm both are accepted.
9. Close and reopen the application; confirm the draft layout returns.
10. Run **Create child replicates…** with the lossless profile.
11. Confirm progress remains responsive and each completed child has a video and
    `.asset.json` sidecar.
12. Open a child. Confirm the entire child frame is displayed and its parent is
    shown.
13. Record several displayed child frames, return to the parent, click that
    child's box, and confirm the same child file and frames are opened.
14. Use **Open parent** to return.
15. Reopen the child, draw a smaller box inside it, and create a grandchild.
16. Move the parent and child packages to new locations, use **Locate parent…**,
    and confirm identity is verified before navigation.
17. Cancel a larger derivation and confirm no partial child appears complete.
18. Perform a representative derivation entirely from the CLI and confirm the
    GUI opens the resulting child identically.

After automated tests pass, stop and ask the user to run this acceptance script.
Do not infer acceptance from tests and do not begin preprocessing work.

## 12. Definition of done

This feature is done when:

- Headless asset, layout, media-frame, lineage, and derivation commands work.
- The GUI implements the complete workflow above without hidden processing
  dependencies.
- Any asset can recursively become a parent.
- Created children are real independent videos with portable lineage sidecars.
- No source-versus-crop processing toggle exists.
- Layout drafts and completed children cannot be confused.
- All mutations are atomic and all long work is cancelable and observable.
- The package contains no analysis engine or speculative future subsystem.
- Automated tests pass.
- The user has personally validated the GUI and accepted the feature.

Only then should a separate handoff be written for the next feature.
