"""Precompute the structure-tensor temporal channels a feature cache does not
store, so an explorer can read them like any other per-block time series.

The flow pipeline caches motion (u, v, speed) and optionally texture, but not the
temporal-change family the tool-3 explorer needs:

    intensity   mean block intensity            -> amplitude variance (windowed)
    change      <I_t^2> per block               -> fast change energy (J_tt)
    appearance  <r^2>, r = I_t + grad(I).v      -> change no motion explains
    texture     spatial min-eigen               -> read from cache if present
    tensor_speed Lucas-Kanade speed from J       -> independent flow-speed read

``appearance`` uses the flow ALREADY in the cache (block flow, upsampled), not a
fresh solve, so the residual is measured against exactly the motion the pipeline
committed to -- the same "no second flow implementation to mistrust" contract the
other explorers keep.

Preprocessing replays downsampling, grayscale conversion, temporal denoising and
contrast normalization. Denoising is exactly reproducible because extraction
streams the same frames in the same order. Registration, background subtraction
and within-replicate masks need reference/model assets that the cache does not
retain, so those steps are omitted and the returned metadata flags the channels
as approximated when any of them were configured.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import cv2
import numpy as np

from core.config import PipelineConfig
from core.flow import reduce_scalar_to_blocks
from core.preprocess import Preprocessor
from core.structure_tensor import (flow_from_tensor, spatial_min_eigen,
                                   tensor_products)
from core.timing import Timer
from core.video import (ClipAtlasSource, ReplicateVideoSource, VideoSource,
                        prefetch)

CHANNELS = ("intensity", "change", "appearance", "texture", "tensor_speed",
            "u", "v")
CHANNEL_VERSION = 4     # bump when the extraction math changes (sidecar key)

# What each channel costs beyond the frame itself, which is what makes selecting
# one worth doing (FINDINGS.md section 1: tensor_products is ~7% of a pass, the
# downstream solve the bulk of the rest).
#
#   intensity     the preprocessed frame, block-reduced. Free.
#   change        J[2] -- needs the products and the spatial blur, nothing more.
#   texture       min-eigen of J. Needs the blur, not the flow solve.
#   tensor_speed  needs flow_from_tensor on top of the blur.
#   u, v          the flow's signed components, same solve as tensor_speed but
#                 kept as (px/s) vectors rather than collapsed to a magnitude --
#                 the base fields the velocity-gradient derived channels read.
#   appearance    needs the residual, and the flow to form it -- the tensor's own
#                 (so: the solve) unless cached block flow was supplied.
#
# So the detection default, ``change``, skips the solve, the residual and the
# min-eigen entirely.
_NEEDS_TENSOR = frozenset({"change", "appearance", "texture", "tensor_speed",
                           "u", "v"})
_NEEDS_FLOW = frozenset({"tensor_speed", "appearance", "u", "v"})

# Which of the six tensor components each read actually consumes, in COMPONENTS
# order (xx, yy, tt, xy, xt, yt). Forming and blurring all six regardless was
# ~81% of a change-only pass on 5.3K footage (FINDINGS.md section 15) -- `change`
# is tt alone, so five products and five full-resolution Gaussian blurs were
# being computed and thrown away every frame.
_COMPONENTS = {
    "change": frozenset({2}),                       # tt
    "texture": frozenset({0, 1, 3}),                # the spatial 2x2 block
    "flow": frozenset(range(6)),                    # the LK solve reads all six
}

# Shared always-off timer so _reduce can time itself without a per-call branch.
_NO_TIMER = Timer("", enabled=False)


def _tiles_from_meta(meta: dict) -> list[dict]:
    """Replicate tiles as (source_box, atlas_bbox, grid, work dims), with a
    whole-frame fallback so a non-replicate cache still works."""
    ny, nx = map(int, meta["grid"])
    raw = meta.get("replicate_tiles")
    if not raw:
        sw = int(meta.get("src_width", meta.get("work_width", nx)))
        sh = int(meta.get("src_height", meta.get("work_height", ny)))
        return [{"id": None, "source_box": (0, 0, sw, sh),
                 "atlas_bbox": (0, 0, ny, nx),
                 "work_width": nx * int(meta["block_size"]),
                 "work_height": ny * int(meta["block_size"])}]
    tiles = []
    for i, t in enumerate(raw):
        y0, x0, y1, x1 = map(int, t["atlas_bbox"])
        tiles.append({
            "id": t.get("id", i),
            "source_box": tuple(map(int, t["source_box"])),
            "atlas_bbox": (y0, x0, y1, x1),
            "work_width": int(t.get("work_width", (x1 - x0) * int(meta["block_size"]))),
            "work_height": int(t.get("work_height", (y1 - y0) * int(meta["block_size"]))),
        })
    return tiles


class _MetaTile:
    """The four attributes ``ReplicateVideoSource`` reads off a tile.

    ``_tiles_from_meta`` yields dicts (it also has to serve a whole-frame
    fallback that no real ``ReplicateLayout`` can express), so the ROI decoder
    gets this adapter rather than a reconstructed ``ReplicateTile`` -- there is
    no layout to rebuild here, only geometry that meta already carries.
    """

    __slots__ = ("replicate_id", "source_box", "work_width", "work_height")

    def __init__(self, key: int, t: dict):
        self.replicate_id = key
        self.source_box = t["source_box"]
        self.work_width = t["work_width"]
        self.work_height = t["work_height"]


class _MetaLayout:
    def __init__(self, tiles: list[_MetaTile]):
        self.tiles = tiles


def _roi_layout(tiles: list[dict]) -> _MetaLayout:
    """Adapt meta tiles for the ROI decoder, keyed by list position.

    Position, not ``t["id"]``: the whole-frame fallback tile has id ``None``,
    and ``crop`` keys with ``int(...)``. The caller pairs a tile with its atlas
    slot by position anyway, so this cannot drift.
    """
    return _MetaLayout([_MetaTile(i, t) for i, t in enumerate(tiles)])


def _open_roi(video_path: str, tiles: list[dict], first: int, count: int,
              fps: float):
    """``(decoder, frame_iter)`` for the FFmpeg ROI path, or ``(None, None)``.

    The first frame is pulled here rather than in the loop so a decoder that
    builds fine but fails on contact -- a filter graph FFmpeg rejects, a codec it
    cannot seek -- falls back to full-frame decode instead of killing the pass.
    Only construction is guarded in the pipeline's copy of this; the live surface
    starts a pass on every knob edit, so a hard failure there is far more costly.
    """
    try:
        roi = ReplicateVideoSource(video_path, _roi_layout(tiles), count,
                                   start=first, fps=fps)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return None, None

    inner = roi.iter_frames()
    try:
        head = next(inner)
    except (StopIteration, OSError, RuntimeError, ValueError):
        # StopIteration included deliberately: a decoder that yields nothing is
        # as useless as one that raises, and both should fall back rather than
        # hand the caller an empty pass.
        roi.release()
        return None, None

    def frames():
        # A generator, not itertools.chain, so close() propagates through the
        # yield from and shuts the FFmpeg reader down -- prefetch relies on that.
        yield head
        yield from inner

    return roi, frames()


def _open_clips(clip_paths: list[str], tiles: list[dict], first: int,
                count: int, fps: float):
    """``(decoder, frame_iter)`` for pre-transcoded clips. **Never falls back.**

    The deliberate difference from :func:`_open_roi`: a clip decoder that fails
    raises instead of quietly reverting to whole-source decode. The fallback is
    right for the ROI path, where both routes read the same source pixels and
    only speed differs. It is wrong here. Below ``lossless`` a clip's pixels are
    *not* the source's (``FINDINGS.md`` section 10), and the caller has already
    folded the clips' provenance key into whatever it caches -- so a silent
    fallback would file source-decoded numbers under a clip-decoded identity,
    which is exactly the confusion ``provenance_key`` exists to prevent.
    """
    # verify_sizes=False: callers reach this through
    # ``channel_source.resolve_clip_paths``, which has already checked every
    # clip's recorded size against its replicate box using the manifest, for
    # free. Probing the files again would cost ~42 ms per pass on 6 clips --
    # against a ~160 ms decode, on a surface that starts a pass per knob edit.
    clips = ClipAtlasSource(clip_paths, _roi_layout(tiles), count, start=first,
                            fps=fps, verify_sizes=False)
    inner = clips.iter_frames()
    try:
        head = next(inner)
    except StopIteration:
        clips.release()
        raise RuntimeError("clip decode yielded no frames")
    except BaseException:
        clips.release()
        raise

    def frames():
        yield head
        yield from inner

    return clips, frames()


@dataclass(frozen=True)
class ChannelPlan:
    """Everything about a pass that is fixed before the first frame is decoded.

    Split out of the extraction loop so the two consumers agree by construction
    rather than by both getting the clamping right: :func:`_stream_channels`
    allocates its ``(n, ny, nx)`` arrays from this, and a streaming consumer
    sizes its ring buffer from the same object. The clamp of ``start``/``n``
    against the video length lives here and nowhere else -- a second copy of it
    is how a buffer and the pass filling it end up disagreeing about which frame
    is which.

    ``channels_computed`` is the authoritative list of which keys hold
    measurements; gate on it, not on key presence.

    **Comparable, not hashable.** ``==`` works and is the useful operation -- a
    streaming surface asks "did the plan change?" to decide whether to restart.
    Hashing does not, because ``tiles`` holds dicts, and a frozen dataclass
    generates a ``__hash__`` that would raise ``unhashable type: 'dict'`` from
    inside the tuple -- an error naming neither this class nor the caller. So it
    is disabled explicitly below, which at least names ``ChannelPlan`` in the
    traceback. Making the tiles hashable instead was considered and dropped: a
    plan is a description rather than an identity, and keying anything by one
    would reintroduce exactly the second copy of window state this class exists
    to collapse.
    """

    __hash__ = None     # see the note above; keeps ``==`` and drops ``hash()``

    fps: float
    block: int
    ny: int
    nx: int
    scale: float
    start: int              # absolute index of the first STORED frame
    n: int                  # stored frames, already clamped to the video
    want: frozenset
    approximated: bool
    # -- how the work is gated; internal to the extraction loop ---------------
    tiles: tuple
    pre_cfg: object
    comps: frozenset
    need_tensor: bool
    need_flow: bool
    need_texture: bool
    need_appearance: bool

    @property
    def channels_computed(self) -> list:
        return sorted(self.want)

    @property
    def stop(self) -> int:
        """Absolute index one past the last stored frame."""
        return self.start + self.n


def plan_channel_stream(meta: dict, *, start: int = 0, n: int | None = None,
                        want: frozenset | None = None,
                        cached_uv: tuple[np.ndarray, np.ndarray] | None = None,
                        cached_texture: np.ndarray | None = None,
                        denoise: str | None = None) -> ChannelPlan:
    """Resolve geometry, window and per-channel gating for a pass.

    Cheap and decode-free, so a caller can build the plan, size a buffer from it
    and only then start the (expensive) stream. ``cached_uv`` and
    ``cached_texture`` are inspected for presence only -- they change what work
    is needed, and are passed again to the stream itself, which reads them.
    """
    fps = float(meta["fps"])
    block = int(meta["block_size"])
    ny, nx = map(int, meta["grid"])
    total = int(meta["n_frames"])
    start = max(0, min(int(start), max(0, total - 1)))
    n = (total - start) if n is None else max(0, min(int(n), total - start))
    scale = float(meta.get("downsample", 1.0))
    tiles = _tiles_from_meta(meta)

    base_cfg = PipelineConfig.from_dict(meta.get("config", {})).preprocess
    denoise = base_cfg.denoise if denoise is None else denoise
    # Replay every step that can be reconstructed exactly by a sequential pass.
    # Registration/background models need fitted assets the cache does not retain,
    # so they are dropped and the result is flagged approximated when they were on.
    pre_cfg = replace(base_cfg, downsample=scale, mask_path=None,
                      registration="off", bg_subtract="off", denoise=denoise)
    approximated = (base_cfg.mask_path is not None or
                    base_cfg.registration != "off" or
                    base_cfg.bg_subtract != "off" or
                    denoise != base_cfg.denoise)

    want = frozenset(CHANNELS) if want is None else frozenset(want) & frozenset(CHANNELS)
    if not want:
        # ``live_channel_source`` already rejects this, but that guard is one
        # layer up and this function is the entry point a streaming consumer
        # calls directly -- so the check has to live here too or the new path
        # routes around it. A pass with nothing to measure still decodes and
        # preprocesses every frame, so it is always a caller bug rather than a
        # cheap no-op: it costs a full pass and returns frames with no channels
        # in them, which a ring buffer cannot even be constructed to hold.
        raise ValueError(f"no channels requested; pick from {sorted(CHANNELS)}")
    # What the selection actually buys, resolved once. ``appearance`` needs the
    # flow solve only when no cached block flow was handed in to form the
    # residual against; with cached_uv it rides the supplied field.
    #
    # Read _NEEDS_FLOW rather than restating its members here, so adding a
    # flow-dependent channel to the table above is enough to make it work.
    # ``appearance`` is the one exception the table cannot express: it needs the
    # solve only to form its own flow, so a supplied cached field excuses it.
    need_flow = bool((want & _NEEDS_FLOW) -
                     ({"appearance"} if cached_uv is not None else set()))
    need_texture = "texture" in want and cached_texture is None

    # The components actually consumed, so the products and the blur can skip the
    # rest. Note what is NOT here: ``appearance`` against a supplied ``cached_uv``
    # reads no tensor component at all -- it forms its own spatial gradient and
    # rides the cached field -- so the cache-backed path now builds no tensor for
    # it. Against the tensor's own flow it needs the solve, and so all six, which
    # ``need_flow`` already covers.
    comps: set[int] = set()
    if "change" in want:
        comps |= _COMPONENTS["change"]
    if need_texture:
        comps |= _COMPONENTS["texture"]
    if need_flow:
        comps |= _COMPONENTS["flow"]
    return ChannelPlan(
        fps=fps, block=block, ny=ny, nx=nx, scale=scale, start=start, n=n,
        want=want, approximated=approximated, tiles=tuple(tiles),
        pre_cfg=pre_cfg, comps=frozenset(comps),
        # Derived from ``comps`` rather than from ``want``: the two disagree
        # exactly in the appearance-with-cached-flow case above, and ``comps`` is
        # the one that reflects the work.
        need_tensor=bool(comps),
        need_flow=need_flow, need_texture=need_texture,
        # Appearance is gated separately from the tensor for the same reason.
        need_appearance="appearance" in want)


def stream_channel_planes(video_path: str, plan: ChannelPlan, *,
                          sigma: float = 2.0,
                          cached_uv: tuple[np.ndarray, np.ndarray] | None = None,
                          cached_texture: np.ndarray | None = None,
                          clip_paths: list[str] | None = None, progress=None):
    """Yield ``(absolute_index, {channel: (ny, nx)})`` as each frame is measured.

    This is the extraction loop itself; :func:`_stream_channels` is a consumer
    that happens to fill one big array with it. The point of the generator form
    is that the partial result already exists in memory during a pass and used to
    be unreachable until the pass ended -- a live surface can now render frames
    as they arrive instead of extracting a window and blocking on it.

    **Absolute indices**, matching ``core.stream_buffer.StreamBuffer.append``,
    which is the intended consumer and for which the ring position is private.
    The first yielded index is ``plan.start``; indices are contiguous.

    Every key in ``plan.want`` is present in every yielded dict, so a consumer
    never has to branch on which channels a given frame happens to carry.
    Channels that need a previous frame (everything but ``intensity`` and
    read-through ``texture``) are zero on the first frame of a window starting at
    frame zero -- there is no motion to measure across a boundary that has no
    other side.

    The generator RETURNS the pass metadata (``StopIteration.value``): frame
    count, ``truncated``, and the timing spans. It is only available at the end
    because ``truncated`` and the spans are only known then. A consumer that
    stops early gets no metadata and should not invent any -- close the generator
    and the timing still lands in the log, which is the honest record of a
    partial pass.
    """
    fps, block, ny, nx = plan.fps, plan.block, plan.ny, plan.nx
    start, n, want = plan.start, plan.n, plan.want
    tiles, comps = plan.tiles, plan.comps
    need_tensor, need_flow = plan.need_tensor, plan.need_flow
    need_texture, need_appearance = plan.need_texture, plan.need_appearance

    base_meta = {"fps": fps, "block": block, "grid": (ny, nx),
                 "channel_version": CHANNEL_VERSION,
                 "approximated": plan.approximated, "sigma": sigma,
                 "window_start": start,
                 "channels_computed": plan.channels_computed}
    if n == 0:
        # Present on both exits: an empty window was asked for and delivered,
        # which is not the same as a decode ending early, and a consumer reading
        # meta["truncated"] should not have to know which return path produced
        # its dict.
        return {**base_meta, "n_frames": 0, "truncated": False}

    pres = {t["id"]: Preprocessor(plan.pre_cfg,
                                  t["source_box"][2] - t["source_box"][0],
                                  t["source_box"][3] - t["source_box"][1])
            for t in tiles}
    prev_g: dict = {t["id"]: None for t in tiles}

    # For a window that does not start at frame zero, read one preceding frame to
    # seed prev_g, so the first stored frame carries motion. This is only valid
    # because denoise is forced off in the windowed path -- a stateful denoise
    # would need every frame from zero to reach the correct state here.
    seed = 1 if start > 0 else 0
    first = start - seed
    count = n + seed

    tm = Timer("extract_channels")
    done = 0            # stored frames, so a cancelled pass can report how far it got
    # Pre-transcoded clips when the caller has them: the decoder then reads files
    # that are already only replicate pixels, which is the one thing that moves
    # the decode floor (~25x, FINDINGS.md section 10). Otherwise ROI decode:
    # FFmpeg crops, greyscales and downsamples each replicate box in its own
    # process, so the ~92% of a 5.3K frame no replicate owns never crosses into
    # Python -- though the decoder still paid for it. Falls back to full-frame
    # OpenCV when even that is unavailable.
    if clip_paths is not None:
        roi, frames = _open_clips(clip_paths, tiles, first, count, fps)
    else:
        roi, frames = _open_roi(video_path, tiles, first, count, fps)
    src = None if roi is not None else VideoSource(video_path)
    try:
        # The decode is driven by hand rather than with a plain ``for`` so the
        # generator's own work (seek + read + colour convert) lands in its own span
        # instead of being invisibly folded into the loop body. Behind ``prefetch``
        # that decode runs on its own thread, so the span now measures how long
        # this loop *waits* for a frame -- near zero when the overlap is working,
        # and still the honest number when decode is the bottleneck.
        frames = prefetch(frames if roi is not None
                          else src.iter_frames(first, count))
        while True:
            with tm.span("decode"):
                nxt = next(frames, None)
            if nxt is None:
                break
            i, frame = nxt
            oi = i - start                     # <0 for the seed frame (not stored)
            # One (ny, nx) plane per wanted channel, allocated fresh each frame
            # because it is handed to the consumer and outlives this iteration.
            # Zero-filled so the no-previous-frame case (and any tile the atlas
            # does not cover) reads as the same zero it did when this loop wrote
            # into a preallocated array -- the yielded dict is the whole frame,
            # not a sparse update.
            planes = ({k: np.zeros((ny, nx), np.float32) for k in want}
                      if oi >= 0 else {})
            for ti, t in enumerate(tiles):
                rid = t["id"]
                x0, y0, x1, y1 = t["source_box"]
                ay0, ax0, ay1, ax1 = t["atlas_bbox"]
                with tm.span("preprocess"):
                    # On the ROI path the crop is already gray and already at the
                    # tile's work size, so Preprocessor's downsample/grayscale
                    # steps collapse to no-ops and the remaining steps (normalize,
                    # mask, ...) run on identical input. Same geometry either way,
                    # because both derive it from the same source_box and scale.
                    owned = (roi.crop(frame, ti) if roi is not None
                             else frame[y0:y1, x0:x1])
                    g = pres[rid].apply(owned)
                th, tw = ay1 - ay0, ax1 - ax0

                gp = prev_g[rid]
                if oi >= 0:
                    if "intensity" in want:
                        planes["intensity"][ay0:ay1, ax0:ax1] = \
                            _reduce(g, block, th, tw, tm)
                    if "texture" in want and cached_texture is not None:
                        planes["texture"][ay0:ay1, ax0:ax1] = \
                            cached_texture[i, ay0:ay1, ax0:ax1]
                    if gp is not None and (need_tensor or need_appearance):
                        # Form and spatially smooth the tensor components this
                        # selection reads, at a small scale, solve LK per pixel,
                        # then reduce speed to blocks. Solving once per block
                        # would couple the aperture problem to the user's
                        # display/block size.
                        J: list = [None] * 6
                        if need_tensor:
                            with tm.span("tensor_products"):
                                prods = tensor_products(gp, g, comps)
                            with tm.span("tensor_blur"):
                                # Only the requested planes; the rest stay None,
                                # so a read that needs them raises rather than
                                # quietly working on zeros.
                                for k in comps:
                                    J[k] = cv2.GaussianBlur(prods[k], (0, 0), sigma)
                        uv = None
                        if need_flow:
                            with tm.span("flow_solve"):
                                uv = flow_from_tensor(J)         # px/frame
                        if "tensor_speed" in want:
                            speed = np.hypot(uv[..., 0], uv[..., 1]) * fps
                            planes["tensor_speed"][ay0:ay1, ax0:ax1] = \
                                _reduce(speed, block, th, tw, tm)
                        # Signed flow components in px/s (the same *fps scaling as
                        # tensor_speed), block-reduced. These are the base fields
                        # the velocity-gradient derived channels take spatial
                        # gradients of; kept as vectors rather than a magnitude so
                        # divergence/shear/vorticity are recoverable.
                        if "u" in want:
                            planes["u"][ay0:ay1, ax0:ax1] = \
                                _reduce(uv[..., 0] * fps, block, th, tw, tm)
                        if "v" in want:
                            planes["v"][ay0:ay1, ax0:ax1] = \
                                _reduce(uv[..., 1] * fps, block, th, tw, tm)
                        if "change" in want:
                            planes["change"][ay0:ay1, ax0:ax1] = \
                                _reduce(J[2], block, th, tw, tm)
                        # Appearance residual r = I_t + grad(I).v. Against cached
                        # block flow (px/s -> px/frame, upsampled) when given, else
                        # against the tensor's own per-pixel flow (already px/frame).
                        if "appearance" in want:
                            with tm.span("appearance"):
                                if cached_uv is not None:
                                    U, V = cached_uv
                                    ub = cv2.resize(U[i, ay0:ay1, ax0:ax1] / fps,
                                                    (g.shape[1], g.shape[0]),
                                                    interpolation=cv2.INTER_NEAREST)
                                    vb = cv2.resize(V[i, ay0:ay1, ax0:ax1] / fps,
                                                    (g.shape[1], g.shape[0]),
                                                    interpolation=cv2.INTER_NEAREST)
                                else:
                                    ub, vb = uv[..., 0], uv[..., 1]
                                iy, ix = np.gradient(g)
                                r = (g - gp) + ix * ub + iy * vb
                            planes["appearance"][ay0:ay1, ax0:ax1] = \
                                _reduce(r * r, block, th, tw, tm)
                        if need_texture:
                            with tm.span("texture"):
                                mineig = spatial_min_eigen(J)
                            planes["texture"][ay0:ay1, ax0:ax1] = \
                                _reduce(mineig, block, th, tw, tm)
                prev_g[rid] = g
            if oi >= 0:
                # Counted as produced BEFORE it is handed over, which keeps
                # ``done`` meaning what it meant when this loop wrote into an
                # array: frames this pass measured. Counting on resume instead
                # would silently undercount by one whenever a consumer closes the
                # generator rather than draining it -- and the log line for a
                # cancelled pass is exactly where that lie would land.
                done = oi + 1
                yield i, planes
            if progress and (oi >= 0) and (oi % 20 == 0):
                # The progress hook re-enters the GUI thread and can supersede or
                # cancel the pass, so it is timed apart from the actual work.
                with tm.span("progress_cb"):
                    progress(oi + 1, n)
    finally:
        try:
            # Close before releasing: the decode thread holds the VideoCapture,
            # and closing joins it. Releasing first would free a capture still in
            # use by a cancelled pass's producer. Nested so that a close() that
            # raises cannot skip the release and leak the handle -- the live
            # surface starts a pass on every knob edit, so a leak per pass adds up.
            if frames is not None:
                frames.close()
        finally:
            (src if src is not None else roi).release()
        # Logged from the finally so a pass cancelled mid-stream still reports its
        # spans. A knob edit supersedes the running extraction by raising inside
        # the progress callback, so during tuning -- exactly when these numbers are
        # wanted -- most passes leave by the exception path. ``done`` distinguishes
        # a partial line from a complete one.
        tm.log(frames=n, done=done, tiles=len(tiles), grid=f"{ny}x{nx}",
               block=block, scale=f"{plan.scale:.3f}",
               src="clips" if clip_paths is not None else
                   ("roi" if roi is not None else "full"))

    # A decoder that ended early yields FEWER frames than the window claimed, and
    # the metadata says so rather than letting a consumer pad. A short window is
    # self-describing, whereas zeros are indistinguishable from "measured, and
    # nothing happened" -- i.e. a silent false negative, on the detection default
    # (``change``) above all. Reachable in practice: a clip truncated by a crash
    # or a full disk during the cut passes ``verify_manifest`` (which checks a
    # clip exists, not how long it is), and a 20-of-64-frame clip was measured
    # yielding 36 all-zero frames of 40 reported as real. The ROI and full-frame
    # paths get the same treatment; they can hit it too, just far less often.
    short = done < n
    if progress:
        progress(done, done)
    return {**base_meta, "n_frames": done, "truncated": short,
            # The spans are already measured for the log line; returning them too
            # is what lets the cost model be built from passes the user actually
            # ran instead of from asserted constants. Only on the complete path:
            # a cancelled pass unwinds through the finally above and never
            # reaches here, so a partial pass can never be mistaken for a timing
            # sample.
            "timing": {"wall": tm.elapsed, "spans": dict(tm.totals),
                       "frames": done, "scale": plan.scale, "block": block}}


def _stream_channels(video_path: str, meta: dict, *, sigma: float = 2.0,
                     start: int = 0, n: int | None = None,
                     cached_uv: tuple[np.ndarray, np.ndarray] | None = None,
                     cached_texture: np.ndarray | None = None,
                     clip_paths: list[str] | None = None,
                     want: frozenset | None = None,
                     denoise: str | None = None, progress=None) -> dict:
    """Stream a frame window [start, start+n) and return per-block channel arrays.

    The windowed contract: drain :func:`stream_channel_planes` into one array per
    channel. Everything about *how* a frame is measured lives there; this is the
    consumer that waits for all of them.

    Geometry comes from ``meta`` (the same shape a feature cache carries); the
    video is streamed directly, so this needs no cache -- it is the seam that lets
    the tensor/scalogram path run on a bare video.

    ``cached_uv`` is the pipeline's block flow (u, v) in px/s, indexed by absolute
    frame, used for the appearance residual; pass ``None`` to measure the residual
    against the tensor's OWN per-pixel flow instead (px/frame, no cache needed).
    ``cached_texture`` is read straight through when present, else computed.
    ``denoise`` overrides the config's temporal-denoise mode (the live/windowed
    path forces it off, since denoise is stateful from frame zero and cannot be
    reproduced starting mid-clip).

    ``want`` restricts which channels are computed; ``None`` means all of them.
    Every ``CHANNELS`` key is still returned -- this dict's shape is fixed -- but
    an unwanted channel's array is a **zero-length** ``(0, ny, nx)`` placeholder.
    Length
    zero, not NaN-filled and not zero-filled: a full-length array of either
    still costs the memory the selection exists to save (~88 MB per channel per
    hour of 30 fps footage at a 41x5 grid), and a zero-FILLED one is worse than
    that, being indistinguishable from "measured, and nothing happened" -- the
    same silent false negative the truncation trim below exists to prevent (its
    rationale now lives on ``stream_channel_planes``, which is what detects the
    short decode; this function only acts on the ``truncated`` flag).
    A zero-length array cannot be mistaken for data or silently broadcast
    against a real one.

    ``meta['channels_computed']`` is the authoritative list of which keys hold
    measurements; gate on it, not on key presence. (``live_channel_source``
    hands out a ``ChannelData`` that drops the placeholders entirely, so at THAT
    layer -- and only there -- key presence is equivalent.)

    Returns a dict of the ``CHANNELS`` arrays (length = clamped window n) plus
    ``meta`` describing how they were built.
    """
    plan = plan_channel_stream(meta, start=start, n=n, want=want,
                               cached_uv=cached_uv,
                               cached_texture=cached_texture, denoise=denoise)
    out = {k: np.zeros((plan.n if k in plan.want else 0, plan.ny, plan.nx),
                       np.float32) for k in CHANNELS}
    planes = stream_channel_planes(video_path, plan, sigma=sigma,
                                   cached_uv=cached_uv,
                                   cached_texture=cached_texture,
                                   clip_paths=clip_paths, progress=progress)
    # Driven by hand rather than with a ``for``, because the pass metadata is the
    # generator's RETURN value and a ``for`` loop discards it.
    while True:
        try:
            i, frame = next(planes)
        except StopIteration as stop:
            chan_meta = stop.value
            break
        oi = i - plan.start
        if not 0 <= oi < plan.n:
            # Cannot happen through the generator, which only yields indices in
            # the window -- and that is exactly why it is checked. A negative
            # ``oi`` does not raise in numpy, it writes to the END of the array,
            # so the failure mode of this invariant breaking is a frame silently
            # landing at the wrong time. ``StreamBuffer.append`` refuses a
            # non-contiguous index for the same reason; the two consumers of this
            # generator should not disagree about how much they trust it.
            raise AssertionError(
                f"frame {i} is outside the planned window "
                f"[{plan.start}, {plan.stop})")
        for k, plane in frame.items():
            out[k][oi] = plane

    # Trim to what was actually measured; see the truncation note above.
    done = int(chan_meta["n_frames"])
    if chan_meta["truncated"]:
        for k in CHANNELS:
            out[k] = out[k][:done]
    out["meta"] = chan_meta
    return out


def extract_channels_live(video_path: str, meta: dict, *, start: int = 0,
                          n: int | None = None, sigma: float = 2.0,
                          clip_paths: list[str] | None = None,
                          channels: Iterable[str] | None = None,
                          progress=None) -> dict:
    """Cacheless windowed extraction: geometry from ``meta``, appearance against
    the tensor's own flow, temporal denoise forced off (stateful, can't reproduce
    mid-clip). This is what feeds the live scalogram surface.

    ``clip_paths`` (aligned with ``meta['replicate_tiles']`` by position) reads
    pre-transcoded per-replicate clips instead of the source. The math is
    untouched -- only where the pixels come from changes. Callers should go
    through ``channel_source.live_channel_source``, which resolves the paths from
    a manifest and verifies the geometry still matches; passing raw paths here
    skips those checks.

    ``channels`` restricts the pass to the channels asked for (default: all).
    The whole-video commit detects on ONE channel, and computing the other four
    was most of its wall time -- on the detection default ``change`` the skipped
    work is the flow solve, the appearance residual and the min-eigen. Unwanted
    channels come back as zero-LENGTH placeholders (never zero-filled ones);
    read ``meta['channels_computed']``. See ``_stream_channels``."""
    return _stream_channels(video_path, meta, sigma=sigma, start=start, n=n,
                            cached_uv=None, cached_texture=None,
                            clip_paths=clip_paths, want=channels, denoise="off",
                            progress=progress)


def _reduce(field: np.ndarray, block: int, th: int, tw: int,
            tm: Timer = _NO_TIMER) -> np.ndarray:
    """Block-mean a tile field and force it to the tile's (th, tw) atlas shape.

    Timing lives inside rather than at the five call sites, so every reduction in
    a pass lands in one ``block_reduce`` span without wrapping each caller."""
    with tm.span("block_reduce"):
        r = reduce_scalar_to_blocks(field, block, "mean", include_partial=True)
    if r.shape != (th, tw):
        # Geometry drift between preprocess output and cached grid: crop/pad so the
        # channel aligns with the cached flow rather than silently misindexing.
        out = np.zeros((th, tw), np.float32)
        h, w = min(th, r.shape[0]), min(tw, r.shape[1])
        out[:h, :w] = r[:h, :w]
        return out
    return r.astype(np.float32)
