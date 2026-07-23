# Review — 1 Build the Isolate player

Status: architectural risk review of `1-Build-the-player.md`.

## Verdict

The handoff is safe to use after a few targeted clarifications. It does not make
the mistake of designing the future isolation engine prematurely, and it does
not contain a fatal commitment that would inherently require an expensive
rewrite later.

Its strongest decisions are:

- The milestone ends with a working, user-validated interaction shell.
- The player and timeline do not own scientific computation.
- Isolate follows active asset identity without referencing the Replicates
  widget directly.
- A child asset is decoded from its own video rather than recreated from its
  parent at runtime.
- Playback/window state is kept separate from immutable asset identity.
- The timeline does not yet imply coverage, detections, or quiet footage.
- The channel area remains empty instead of inventing a speculative channel
  framework.
- The implementation stops before processing begins.

Those choices are compatible with later static filtering, Morlet isolation,
headless execution, recipes, validation, rendering, artifact stacking, and
derived assets.

## Important correction 1: do not block the player on a verified frame count

The current handoff defines the window against `decoded_frame_count` and says to
use verified-count behavior when the available metadata is not trustworthy.

The asset schema permits `decoded_frame_count` to be unknown. An implementation
could therefore interpret the handoff as requiring a full decode/count before
showing the first usable player frame. That would make opening long assets slow
and would work against the media-service performance milestone.

Use two concepts:

```text
navigable extent
  The best currently available probed/container estimate used by the player.

verified decodable extent
  The measured range required for authoritative processing and coverage claims.
```

For the player:

- Open and display the first frame without waiting for a full count.
- Permit an explicitly provisional navigable extent when the decoded count is
  unknown.
- Mark an estimated total as estimated where it is shown to the user.
- Clamp or correct the extent when the decoder reaches actual EOF.
- Never present an estimated extent as verified scientific coverage.

Later whole-asset processing may require or produce a verified decodable extent.
That stricter requirement does not need to delay an interactive viewer.

This distinction should live in the session/timeline model rather than being
patched only into the label text.

## Important correction 2: require independent playback state, not necessarily
## an independent decoder architecture

The handoff correctly requires Replicates and Isolate to have independent
playheads. It currently goes further and names the decoder/media-session handle
as Isolate-local.

The Replicates handoff describes a workspace-owned `AssetSession` containing a
decoder. The rewrite may already have a correct media-session ownership model,
and this handoff should not force a second competing decoder architecture before
that implementation is inspected and benchmarked.

Require:

- Isolate owns its current frame, looping window, and playing state.
- Isolate uses the rewrite's existing media-service contract.
- No Isolate widget reaches into Replicates for a decoder or playhead.
- Opening a new active asset invalidates outstanding requests from the old one.
- One requested displayed frame is not accidentally decoded multiple times by
  Isolate consumers.

Leave open:

- Whether the underlying decoder handle is owned by an Isolate controller, a
  promoted active-asset session, or another existing application service.
- Whether Replicates and Isolate keep separate decoder sessions when both
  workspaces remain alive.

Choose decoder ownership from the rewrite's established lifecycle and measured
viewer behavior. Independent UI state is the contract; duplicate decoder handles
are not.

## Important correction 3: 60 seconds is an initial UI choice, not a model limit

The current handoff carries the oracle's 60-second maximum preview length.

That is acceptable as the initial range presented by the first player, but it
must not become a validation rule in the underlying session/window model.
Future signals may require:

- Lower temporal frequencies.
- Long static states.
- Long-duration context.
- Direct navigation over a larger retained result.

The underlying window contract should accept any nonempty, asset-bounded
half-open interval. The GUI may initially offer a convenient 0.2–60 second range
and later expand it without a session-schema or controller rewrite.

## Documentation clarification: precedence

The older Replicates handoff intentionally prohibited placeholder future tabs,
and the older rewrite documents contain a different implementation order. This
new numbered handoff reflects a later decision to rebuild Isolate incrementally,
starting with the player shell.

Add a short precedence note to the implementation handoff:

> This handoff is a later, user-authorized milestone. For the work explicitly in
> scope here, it supersedes older instructions not to add a future tab or not to
> begin this interaction shell before processing. Older asset, media, lineage,
> and scientific correctness contracts remain in force unless this handoff
> explicitly changes them.

This prevents an implementer from treating the documents as mutually blocking
without discarding their still-valid technical contracts.

## Documentation clarification: next milestone

The handoff currently says the next handoff should add one real channel. The next
document is now `2-Media-service-handoff.md`, which benchmarks and improves the
viewer before channel work.

Replace the final prediction with:

> After reaching this state, stop. Continue only with the next numbered,
> user-approved Isolate handoff.

That keeps the stopping rule without guessing the future order.

## Decisions that are safe as written

### Current frame remains inside the looping window

This does not prevent future whole-asset detection navigation. Clicking a result
outside the current window can reposition the window so the requested frame is
inside it. The invariant keeps player behavior understandable.

### The initial timeline is only a navigator

Future coverage, staleness, detections, scores, and event intervals can be added
as render layers or through a later specialized timeline without changing the
active asset or window model.

Do not make the initial timeline own future artifact semantics, but there is no
need to design those layers now.

### No channel API exists yet

This is preferable to inventing a registry around an empty panel. The first real
channel should reveal the smallest useful contract for:

- Its identity and settings.
- Windowed data requests.
- Displayed results.
- Cancellation and supersession.
- Headless equivalence.

The empty right-hand container is enough preparation for this milestone.

### No recipe or CLI analysis surface exists yet

The player already relies on the headless media-service boundary. Scientific
recipe and CLI analysis contracts do not need to be invented to display and
navigate a video.

The important constraint is that later channel computation must not be
implemented inside the player or timeline widgets. The handoff already says so.

### Player-state persistence is deferred

Resetting the first implementation on asset open is reversible. Later
per-asset/session restoration can be added without changing media or scientific
identity, provided the window model is not hard-coded around reset-only
construction.

### Exact rational time is preserved

Keeping integer frames and rational frame rate at the player/session boundary is
the correct long-term choice. Rounded seconds remain display values only.

## Risk classification

### Potentially expensive if left ambiguous

1. Requiring a full verified frame count before the viewer becomes usable.
2. Creating a second decoder/session architecture solely because the handoff
   appears to mandate one.
3. Encoding the 60-second UI cap as a durable domain validation rule.

### Cheap to change later, but worth clarifying now

1. Exact click behavior outside the selected timeline window.
2. Whether Play on `stop - 1` shows that frame before looping.
3. Default window length.
4. Shortcut assignments.
5. Timeline styling.
6. Splitter proportions.
7. Whether window state is later restored per asset.

### Correctly deferred

1. Channel registry and channel-panel interface.
2. Static versus Morlet transforms.
3. Processing workers.
4. Recipe/settings schema.
5. CLI analysis commands and HPC execution.
6. Coverage and stale/current artifacts.
7. Validation, presentation rendering, and derived assets.
8. Detection stacking.

## Recommended disposition

Make the three important corrections and the two documentation clarifications
before giving `1-Build-the-player.md` to an implementation agent.

No broader architectural rewrite of the handoff is needed.
