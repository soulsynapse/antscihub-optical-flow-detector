"""Which pixels a pass will read: the source video, or its pre-transcoded clips.

Batch S made the replicate box the unit of ownership on disk, and slice 4 put the
cut *inside* the home (``<stem>_rep01/<stem>.mkv``). So a clip is no longer
something an operator points at with ``--manifest``; it is something that is
either sitting in the home or is not. This module is the one place that asks.

    route = resolve_source(video, replicates)
    cd = live_channel_source(video, cfg, replicates, **route.extract_kwargs)

**Derived per call, never recorded.** There is no pointer file saying "this video
uses clips", and there must not be one -- that is Batch J's rule arrived at the
hard way (``FINDINGS.md`` sections 13 and 15): *resume is by identity, not
existence*. A recorded pointer is an existence claim that outlives the thing it
points at, so a moved box, a re-stabilized source or a truncated clip would keep
reading as "clips available" until something downstream produced wrong numbers.
Re-deriving costs a ``quick_signature`` head+tail read and a stat per clip --
~1.4 ms, which is why :func:`core.pretranscode.verify_manifest` was built cheap
in the first place, on a live surface that starts a pass per knob edit.

Three outcomes, and the difference between the last two is the whole design
-----------------------------------------------------------------------------

* **No manifest for this video** -> route to the SOURCE, with ``reason`` saying
  so. Not an error: clips are an optional speed-up, and a corpus that was never
  cut must still run.
* **A manifest that is not this video's** -> also the source, and for the same
  reason: manifests are named from the video's basename, so two sources called
  ``GX010047.MP4`` in different session directories map to one manifest path
  under a shared ``--clip-dir``. ``verify_manifest`` cannot catch that -- it
  validates the manifest against the source *the manifest names*, which is
  present and unchanged, so it passes and one session's pixels are attributed to
  the other's detections. Identity is therefore checked against the video in
  hand, exactly as ``cli/pretranscode._existing_state`` and
  ``core.framecount._manifest_describes`` already do. For *this* video no
  manifest exists, which is the case above.
* **This video's manifest, but it does not verify** (boxes moved, source
  changed, a clip missing or truncated) -> **raise**. Ruled 2026-07-21. Falling
  back would silently pay ~25x the decode AND cross a provenance boundary
  (``FINDINGS.md`` section 10) to do it, on footage the operator has evidently
  already cut once. A stale cut is an operator error with a one-line fix
  (re-run the pre-transcode); a silent slow correct run teaches nothing.

Why the route carries a cost-model token
-----------------------------------------
:attr:`SourceRoute.sample_kind` exists because ``CostModel.fixed_s`` *is* the
decode floor, and cutting clips moves it ~25x (``FINDINGS.md`` section 10). A
sweep taken before a cut and one taken after are measurements of two different
machines as far as that model is concerned, and fitting them together reads the
difference as scale -- which moves the knee toward "downsampling is free", the
one direction ``FINDINGS.md`` section 6 says the model must never err in. The
token is the manifest's ``provenance_key`` and not a bare ``"clips"``, because
two cuts at different quality are also different pixels and a different floor.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from core.framecount import manifest_describes
from core.pretranscode import (Manifest, PretranscodeError, manifest_path_for,
                               read_manifest, verify_manifest)

# ``SourceRoute.kind``. Also the cost model's default source kind: every pass
# ever timed before this module existed was a source pass, so the default is a
# fact rather than an assumption.
SOURCE = "source"
CLIPS = "clips"


@dataclass(frozen=True)
class SourceRoute:
    """What a pass over ``video_path`` will actually decode.

    ``reason`` is filled only when clips were possible in principle and are not
    being used; it is the string a caller surfaces as a warning. A route to the
    source is never itself a failure -- see the module docstring for the one
    case that raises instead of routing.
    """
    video_path: str
    kind: str
    manifest: Manifest | None = None
    clip_dir: str | None = None
    manifest_path: str | None = None
    reason: str = ""

    @property
    def uses_clips(self) -> bool:
        return self.kind == CLIPS

    @property
    def sample_kind(self) -> str:
        """The cost-model token for passes taken over this route.

        ``"source"``, or ``"clips:<provenance_key>"``. See the module docstring
        for why the key is in it.
        """
        if not self.uses_clips or self.manifest is None:
            return SOURCE
        return f"{CLIPS}:{self.manifest.provenance_key()}"

    @property
    def extract_kwargs(self) -> dict:
        """``manifest``/``clip_dir`` for ``live_channel_source``.

        Always both keys, both ``None`` on a source route, so a call site reads
        ``**route.extract_kwargs`` unconditionally rather than branching. The
        branch is what would drift: it is the shape that lets one of the two
        extraction paths quietly keep decoding the source.
        """
        if not self.uses_clips:
            return {"manifest": None, "clip_dir": None}
        return {"manifest": self.manifest, "clip_dir": self.clip_dir}

    def describe(self) -> str:
        """One line for a status bar or a log, naming the pixels being read."""
        if self.uses_clips:
            man = self.manifest
            q = f", {man.quality}" if man is not None else ""
            return f"clips ({len(man.clips) if man else 0} replicate{q})"
        return "source video"


def find_manifest_path(video_path: str, clip_dir: str | None) -> str | None:
    """A pre-transcode manifest for this video, if one is there to be found.

    A manifest belongs to exactly one source, so a multi-video run cannot take a
    single ``--manifest`` the way ``cli/detect.py`` does; it discovers one per
    video by the naming convention ``build_pretranscode`` writes. ``clip_dir``
    is searched first so an explicit one wins over a stray manifest beside the
    footage.

    Finding a file here proves nothing about whose it is -- that is
    :func:`resolve_source`'s job and the module docstring's second case.
    """
    seen: set[str] = set()
    for d in ([clip_dir] if clip_dir else []) + [
            os.path.dirname(os.path.abspath(video_path))]:
        cand = manifest_path_for(video_path, d)
        key = os.path.normcase(os.path.abspath(cand))
        if key in seen:
            continue
        seen.add(key)
        if os.path.exists(cand):
            return cand
    return None


def resolve_source(video_path: str, replicates: list[dict] | None = None, *,
                   clip_dir: str | None = None, allow_clips: bool = True,
                   manifest_path: str | None = None) -> SourceRoute:
    """Decide, right now, whether this video's pass reads clips or the source.

    ``replicates`` is passed through to ``verify_manifest`` so a moved box is
    caught here rather than at decode time; ``None`` skips only that check.
    ``manifest_path`` names a manifest explicitly (``cli/detect.py --manifest``)
    instead of discovering one -- an explicitly named manifest that turns out
    not to describe this video is a usage error and raises, where a *discovered*
    one of the same shape is simply somebody else's file.

    Raises :class:`~core.pretranscode.PretranscodeError` when this video's own
    manifest cannot be trusted. See the module docstring for why that is not a
    fallback.
    """
    def src(reason: str = "") -> SourceRoute:
        return SourceRoute(video_path=video_path, kind=SOURCE, reason=reason)

    if not allow_clips:
        return src("clips disabled; decoded the source")

    explicit = manifest_path is not None
    path = manifest_path if explicit else find_manifest_path(video_path, clip_dir)
    if not path:
        where = f" in {clip_dir} or" if clip_dir else ""
        return src(f"no pre-transcode manifest found for this video{where} "
                   f"beside it; decoded the SOURCE instead of ROI clips "
                   f"(slower, and not the same pixels)")

    man = read_manifest(path)
    if not manifest_describes(man, video_path):
        detail = (f"{os.path.basename(path)} was cut from {man.source_path}, "
                  f"not from this video -- their basenames collide, or the "
                  f"source has been re-exported since the cut")
        if explicit:
            raise PretranscodeError(f"--manifest does not describe {video_path}: "
                                    f"{detail}")
        # Discovered, and not ours: for THIS video there is no manifest. Reported
        # rather than merely skipped, because the run then silently pays the full
        # decode the operator thought they had cut away.
        return src(f"{detail}; decoded the SOURCE")

    # From here on the manifest IS this video's, so anything wrong with it is
    # wrong with this job. verify_manifest re-checks the source's identity, the
    # geometry hash, and every clip's presence and size.
    d = clip_dir or os.path.dirname(os.path.abspath(path))
    verify_manifest(man, d, replicates)
    return SourceRoute(video_path=video_path, kind=CLIPS, manifest=man,
                       clip_dir=d, manifest_path=path)
