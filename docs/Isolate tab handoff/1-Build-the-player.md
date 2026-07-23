# 1 — Build the Isolate player

Status: implementation handoff for the first Isolate-tab milestone.

This milestone builds the empty Isolate workspace and returns it for user
validation. It does **not** implement channels, preprocessing, signal isolation,
detection, recipes, batch execution, or outputs.

The purpose is to establish and validate the interaction surface before any
scientific processing architecture is placed underneath it.

## 1. Outcome

Add a second top-level workflow tab named **Isolate**.

When an asset is open, the tab contains:

- A player showing the active asset's full video frame.
- A short selected time window that loops during playback.
- Controls for the window start and length.
- A whole-asset timeline at the bottom showing both the selected window and the
  current frame.
- An intentionally empty channel area on the right.

The initial layout is:

```text
+------------------------------------------------+--------------------------+
|                                                | Channels                 |
|                                                |                          |
|                 video player                   | No channels added yet.   |
|                                                |                          |
|                                                |                          |
+------------------------------------------------+--------------------------+
| Play/Pause   window start   window length   current time/frame            |
+---------------------------------------------------------------------------+
| whole-asset timeline                                                   |
| [unselected footage][======= selected looping window =======][.........] |
|                              ^ current frame                              |
+---------------------------------------------------------------------------+
```

The player and timeline are real and interactive. The channel area is only an
empty layout region for the next feature; do not add fake channel cards,
disabled scientific controls, example plots, or a speculative channel API.

## 2. Scope boundary

Implement only:

- The `Isolate` workflow tab.
- Isolate-local player/session state.
- Opening and displaying the active asset.
- Frame seeking and playback.
- Selection of a bounded looping time window.
- The whole-asset timeline.
- An empty right-hand channel area.
- Keyboard interaction and lifecycle cleanup.
- Automated tests for those behaviors.

Do not implement:

- Image preprocessing.
- Downsample, normalization, or block-size controls.
- Channel discovery, registration, or computation.
- Tensor fields, optical flow, scalograms, Morlet transforms, or static signal
  filters.
- Detection thresholds, gates, counts, clumps, marks, or event intervals.
- Whole-video processing.
- Coverage, stale/current results, or processing schedules.
- Recipe/settings files.
- CLI analysis commands.
- HPC execution.
- Validation, rendering, exports, or derived assets.
- Tuning or player-state persistence.
- Placeholder workers, caches, artifact stores, or plugin systems.

Do not begin the next processing milestone after this one passes its automated
tests. Return the working player to the user for visual and interaction
validation.

## 3. Existing application contract

This feature builds on the Replicates/asset milestone rather than replacing it.

### 3.1 Shared active asset

Replicates and Isolate are workflow tabs over the same active asset identity.

Opening an asset by any supported route updates both tabs:

- Opening a source asset in Replicates.
- Opening a derived child from Replicates.
- Opening that child directly from its video or sidecar.
- Returning to a reachable parent.

Isolate must display the video file belonging to the active asset itself. If the
active asset is a child, Isolate decodes the child's video. It must not reopen
the parent and crop it at runtime.

Switching between Replicates and Isolate does not change the active asset.

Use the application's existing active-asset controller/signal if one is already
implemented. Do not create a second competing definition of the active asset,
and do not give Isolate a direct reference to the Replicates widget.

### 3.2 State that is not shared

Only asset identity and immutable asset metadata are shared.

The following are Isolate-local:

- Decoder/media-session handle.
- Current frame.
- Window start.
- Window length.
- Playing/paused state.

Replicates and Isolate do not share a playhead or decoder handle. Switching tabs
must not make one tab reach into the other to discover or mutate its current
frame.

### 3.3 Media service

Reuse the existing headless media probe/session service from the asset milestone.
Do not build another decoder implementation inside the tab.

The GUI may keep one Isolate-local media session open for responsive playback.
The media service owns probing, seeking, frame decoding, and close behavior; Qt
widgets render returned frames and translate user actions into session/controller
requests.

The Isolate GUI must not shell out once per displayed frame.

## 4. Minimal component ownership

Keep the implementation small. A sufficient separation is:

```text
IsolateTab
  layout and empty/loaded presentation

IsolateSession or IsolateController
  active asset descriptor
  current frame
  looping window [start, stop)
  playing state
  one Isolate-local media session

IsolatePlayer
  displayed frame
  aspect-preserving image presentation
  play/pause and frame navigation gestures

IsolateTimeline
  whole-asset scale
  selected-window overlay
  current-frame cursor
  seek/window-placement gestures
```

Equivalent names are fine. The required boundary is behavioral:

- The player does not own scientific processing.
- The timeline does not decode frames.
- The tab does not copy asset-opening policy from Replicates.
- The widgets communicate through the Isolate session/controller, not through
  direct calls into one another.

Do not introduce a general pipeline framework to obtain this separation.

## 5. Time and frame contract

Use exact integer frames internally.

- The asset provides `fps_num` and `fps_den`.
- Convert frames to display seconds using the rational timebase.
- A selected window is the half-open frame interval `[start, stop)`.
- `0 <= start < stop <= decoded_frame_count`.
- A usable window contains at least two frames when the asset contains at least
  two decodable frames.
- The current frame is always inside the selected window while the Isolate
  player is active.

Rounded seconds are presentation only. Seeking, looping, clamping, and timeline
placement use integer frames.

If the media metadata does not provide a trustworthy decodable frame count, use
the existing verified-count behavior from the media/asset layer. Do not silently
invent a duration from a stale container estimate.

### 5.1 Initial window

On opening an asset:

- Start at frame `0`.
- Default to a 10-second window.
- Clamp that window to the available asset duration.
- For an asset shorter than 10 seconds, select the whole asset.
- Set the current frame to the window start.
- Start paused.

Do not restore a window from disk in this milestone.

### 5.2 Window controls

Provide:

- `Window start`, displayed in seconds with the exact frame available in a
  tooltip or adjacent diagnostic.
- `Length`, displayed in seconds.

Use the current v1 interaction bounds as the starting behavior:

- Minimum displayed length: approximately `0.2 s`, but never fewer than two
  frames.
- Maximum selectable length: `60 s` or the complete asset duration, whichever
  is shorter.
- Default: `10 s` or the complete asset duration, whichever is shorter.

Changing start or length must:

1. Resolve the new half-open integer frame interval.
2. Clamp it inside the asset without producing an empty interval.
3. Keep the chosen length when moving near the end whenever possible.
4. Move the current frame into the new interval if it is now outside.
5. Refresh the timeline and displayed frame.

One user edit must produce one settled seek. Spin-box keystrokes or timeline
dragging must not accumulate an unbounded queue of obsolete decode requests.
Decode the latest requested frame.

## 6. Playback behavior

There is one play/pause action on the Isolate tab, and it controls video
playback inside the selected window.

- Pressing **Play** starts at the current frame.
- Playback advances according to the asset timebase.
- On reaching `stop`, playback loops to `start`.
- The final frame displayed before looping is `stop - 1`.
- Pressing **Pause** leaves the current frame where it is.
- Pressing **Play** while parked at `stop - 1` may show that frame and then loop,
  or restart immediately at `start`; choose one behavior and test it
  consistently.
- Playback must not leave the selected interval.
- Changing the window while playing continues playback inside the new window.
- Opening another asset stops playback before closing the old media session.
- Closing the window or application stops timers and closes the media session.

Playback uses bounded backpressure:

- Never queue every missed timer tick.
- If a seek/decode is still pending, coalesce requests to the newest required
  frame.
- A late frame from an old asset or superseded seek must not overwrite the
  current asset's display.

No processing worker exists in this milestone. The word “Play” means video
playback, not “run isolation.”

## 7. Timeline behavior

The bottom timeline spans the complete active asset, not merely the looping
window.

It displays:

- The full asset extent.
- The selected looping window as a clearly visible span.
- The current frame as a distinct cursor.
- An empty/unavailable state when no asset is open.

Required interaction:

- Clicking inside the selected window seeks the current frame to the clicked
  position.
- Clicking outside the selected window moves the window so that the clicked
  frame is inside it, preserving the current window length and clamping at the
  asset edges, then seeks to the clicked frame.
- Dragging/scrubbing may update the visual cursor immediately, but decoding must
  coalesce to the latest requested frame.

Direct dragging of the selected window and resize handles are optional in this
milestone. The start and length controls are the authoritative editing surface.
Do not delay the milestone to build a miniature timeline editor.

The timeline contains no coverage or detection claims yet. Do not paint
unprocessed footage as quiet, stale, negative, or detected. It is simply a
navigator and window display.

## 8. Video presentation

- Preserve the video's aspect ratio.
- Letterbox rather than crop or stretch.
- Base any future pointer-to-frame coordinate mapping on the actual drawn-image
  rectangle, not the widget bounds.
- Show the full active asset frame; there are no replicate rectangles or parent
  coordinates in Isolate.
- Display decode failures in the tab/status area without freezing the GUI.
- Do not retain a frame from the previous asset while the next asset is loading.

The active asset bar or existing global header remains authoritative for asset
label, kind, dimensions, rate, duration, and parent navigation. Avoid repeating
that information inside decorative frames if it is already visible.

## 9. Channel area

Reserve a useful right-hand area labelled **Channels**.

For this milestone it contains only a quiet empty state such as:

> No channels added yet.

The area should be able to receive real widgets in the next milestone without a
layout rewrite. A splitter between the player and the channel area is
appropriate so the user can adjust their relative width.

Do not yet define:

- A channel registry.
- Channel base classes.
- Channel settings schemas.
- Plot interfaces.
- Add/remove/reorder behavior.
- Persistence.
- CLI mappings.

Those decisions should be made when the first real channel is implemented.

## 10. Keyboard and transport controls

When the Isolate tab is active:

- `Space`: play/pause the looping player.
- `Left` / `Right`: step one frame, clamped inside the selected window.
- `Shift+Left` / `Shift+Right`: step approximately one second using the rational
  frame rate, clamped inside the selected window.
- `Home`: seek to the selected window's first frame.
- `End`: seek to the selected window's final frame (`stop - 1`).
- `Ctrl+1`: switch to Replicates.
- `Ctrl+2`: switch to Isolate.

Shortcuts must not fire twice because both the main window and a child player
installed competing handlers. There must be one clear owner for application
shortcuts and one dispatch path to the active workflow tab.

Typing into a numeric control must remain usable. In particular, Space while
editing a control must not produce duplicated playback toggles.

## 11. Empty and failure states

With no active asset:

- Show a concise message directing the user to open footage in Replicates or
  through the global Open action.
- Disable playback and window controls.
- Show an empty timeline.
- Keep the Channels area present but empty.

For an asset with fewer than two decodable frames:

- Display any decodable frame.
- Disable looping playback.
- Explain that a time window requires at least two frames.

For probe or decode failure:

- Keep the application responsive.
- Show a concise error and stable error code supplied by the media layer.
- Stop playback.
- Release the failed media session.
- Do not substitute the parent asset or another decoder silently.

## 12. Suggested implementation files

Adapt this to the v2 repository's existing names and layout rather than creating
parallel infrastructure:

```text
gui/
  isolate_tab.py
  isolate_player.py
  isolate_timeline.py
  isolate_session.py       # or isolate_controller.py
```

Prefer reusing an existing generic video canvas/player component when it already
has the required aspect-ratio, seeking, and lifecycle behavior. Extract a shared
component from Replicates only when that extraction is small and leaves
Replicates behavior unchanged.

Do not move asset registration, lineage, or derivation logic into these files.

## 13. Automated tests

Use the repository's supported offscreen Qt test configuration.

### 13.1 Session/window tests

- Opening an asset produces `[0, min(10 s, asset end))`.
- A short asset selects its complete valid range.
- Window intervals are half-open and never empty for an ordinary video.
- Moving the start near the asset end preserves length by clamping the interval.
- Increasing length near the end moves/clamps the start correctly.
- Shrinking a window around a cursor clamps the cursor into the new interval.
- Frame/second presentation uses the rational timebase.
- Opening a new asset resets the Isolate window and stops playback.

### 13.2 Playback tests

- Space toggles playback exactly once.
- Playback never displays a frame outside `[start, stop)`.
- Reaching `stop` loops to `start`.
- Pause retains the current frame.
- Frame stepping clamps inside the selected window.
- Home and End target the selected window, not the whole asset.
- A late result from a superseded seek is ignored.
- Closing the tab/window leaves no running timer, thread, or open media session.

### 13.3 Timeline tests

- Timeline span represents the complete asset.
- Window overlay maps `[start, stop)` correctly at the beginning, middle, and
  end.
- Cursor mapping is correct at the first and final selected frames.
- Clicking inside the window seeks without moving the window.
- Clicking outside moves the window to include the clicked frame without
  changing its length.
- Rapid scrub requests coalesce to the latest frame.

### 13.4 Asset synchronization tests

- Opening a parent updates Isolate.
- Opening a child through Replicates updates Isolate to the child.
- Opening that child directly produces the same Isolate asset state.
- The Isolate decoder opens the child video and does not decode/crop the parent.
- Switching workflow tabs does not change active asset identity.
- Replicates and Isolate do not share or mutate each other's playhead.
- Isolate imports no Replicates widget module and holds no direct tab reference.

### 13.5 Layout tests

- The video retains aspect ratio under resize.
- The channel area remains visible and usable at the minimum supported window
  size.
- No channel, preprocessing, detection, or processing controls appear.
- Empty and one-frame assets do not crash construction.

## 14. Manual acceptance script

After automated tests pass, stop and ask the user to perform this validation:

1. Launch the v2 GUI and open a normal source asset.
2. Switch to **Isolate**.
3. Confirm the correct asset is visible and the default window covers its first
   10 seconds.
4. Press Space and confirm the footage loops at the selected window boundary.
5. Pause, step frames, and use Home/End.
6. Change the start and length and confirm the selected timeline span updates
   immediately.
7. Click inside the span and confirm it seeks without moving.
8. Click elsewhere on the whole-asset timeline and confirm the same-length
   window moves there.
9. Scrub rapidly and confirm the display lands on the final request rather than
   replaying a backlog.
10. Return to Replicates, open a child asset, and switch back to Isolate.
11. Confirm Isolate displays the child's own complete frame and resets to its
    first window.
12. Switch repeatedly between the tabs and confirm their playheads remain
    independent.
13. Resize the window and adjust the player/channel splitter.
14. Confirm the Channels area is empty and that no processing controls have
    been invented.
15. Close the application while playing and confirm it exits cleanly.

Do not infer user acceptance from automated tests.

## 15. Definition of done

This milestone is complete only when:

- A top-level tab is visibly named **Isolate**.
- It always follows the application's active asset identity.
- It displays and navigates the active asset's own video.
- A bounded window can be positioned and resized.
- Playback loops exactly within that window.
- The whole-asset timeline shows the window and current frame.
- The right-hand Channels area exists and is intentionally empty.
- Decoder, timer, and asset-change lifecycle behavior is tested.
- Replicates and Isolate have no direct widget coupling.
- No scientific processing, recipe, CLI-analysis, output, or plugin
  architecture has been added.
- Automated tests pass.
- The user has personally validated and accepted the interaction.

After reaching this state, stop. The next handoff should add one real channel
end to end; it should not be anticipated by expanding this milestone.
