"""Configuration objects for the flow pipeline.

Every config here is a frozen dataclass so it can be hashed. The feature cache
on disk is keyed by (video_hash, replicate_geometry, preprocess_config,
flow_config, feature_config)
so that retuning downstream histogram filters never triggers recomputation, but
changing anything upstream does.

Time is in SECONDS and frequency is in HZ throughout. Frame indices appear only
at the video-decode boundary and in tooltips.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict, replace
from typing import Literal

CONFIG_VERSION = 3

# Blocks are measured in WORKING pixels, but localization is a property of the
# SOURCE image, so the default block size is denominated in source pixels and the
# working value is derived from the scale (see FlowConfig.resolve_block_size).
# 64 reproduces the block grid the tool used at the old 1300px target width
# (64 * 0.245 ~ 16 working px), so switching to scale 1.0 keeps both today's
# localization and today's cache size -- only the per-pixel stages get finer
# input. See todo.md Batch K for the measured table this comes from.
BASE_BLOCK_SOURCE_PX = 64


@dataclass(frozen=True)
class Band:
    """A frequency band for band-power, in Hz."""
    lo_hz: float
    hi_hz: float

    def label(self) -> str:
        return f"bandpower_{self.lo_hz:g}-{self.hi_hz:g}Hz"


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing applied to frames before flow is computed.

    Defaults are all-off except downsampling and normalization, so a first-time
    user can hit "Run test" immediately and get a result. Normalization defaults
    to z-score rather than off because raw inter-frame brightness drift otherwise
    shows up as spurious global flow; z-score is chosen over CLAHE because CLAHE's
    per-tile equalization has a known replicate-edge artifact (see KNOWN_ISSUES).
    """
    # Optional mask within replicate-owned rectangles (white = keep). This does
    # not change ROI-first decode/cache geometry; None keeps every owned pixel.
    mask_path: str | None = None

    # Frame registration for camera motion.
    registration: Literal["off", "phase", "orb"] = "off"
    registration_ref_time_s: float = 0.0

    # Temporal denoising across frames.
    denoise: Literal["off", "median", "gaussian"] = "off"
    denoise_window: int = 3

    # Background subtraction.
    bg_subtract: Literal["off", "median", "mog2"] = "off"
    bg_median_samples: int = 25

    # Downsampling. None means NO downsampling (1.0), which is the default.
    #
    # This is deliberately not automatic. A pipeline that silently downsamples
    # has already decided which behaviours are detectable -- the tool would be
    # defining the data collected rather than the other way around. Coarser
    # resolution may well be sufficient for a given behaviour, but that has to be
    # demonstrated per behaviour and per species, never assumed. Downsampling is
    # therefore an explicit, user-wielded lever for reducing compute, not a
    # default. See todo.md Batch K.
    #
    # It also makes the scale physically meaningful: at 1.0 a pre-cropped video
    # and an equivalent uncropped replicate box resolve identically, because
    # neither is rescaled. The old width-derived default did not have that
    # property, so those two workflows were not comparable.
    downsample: float | None = None

    # Contrast / illumination normalization. z-score (global per-frame mean/std)
    # is the default; CLAHE is available but carries a known replicate-boundary
    # artifact -- see KNOWN_ISSUES.md before selecting it.
    normalize: Literal["off", "clahe", "zscore"] = "zscore"

    def resolve_downsample(self, src_width: int = 0) -> float:
        """The working scale. ``None`` means no downsampling.

        ``src_width`` is accepted and ignored: the scale is no longer derived
        from the frame width. The parameter is kept because callers pass it and
        because a future organism-relative mode (pixels per body length) would
        need geometry again -- but it would need the *replicate* geometry, not
        the source width, so this signature is not the one it will use.
        """
        if self.downsample is None:
            return 1.0
        return float(self.downsample)


@dataclass(frozen=True)
class FlowConfig:
    """Optical flow backend selection and block reduction."""
    backend: Literal["farneback", "dis", "raft"] = "farneback"

    # Block size in WORKING pixels. None means "track the scale", i.e. hold the
    # block grid fixed in source pixels at BASE_BLOCK_SOURCE_PX. An explicit int
    # overrides that and is taken as a working-pixel count directly.
    #
    # Tracking is what keeps the two levers separable. `downsample` buys compute
    # and `block_size` buys storage; if the grid moved with the scale, turning
    # the compute knob would also coarsen localization, so a user could not
    # attribute a change in detection output to either one. Detection reads
    # per-block band power and thresholds on per-region block counts, so a grid
    # that holds still under a scale change is what makes the scale knob a
    # single-variable experiment.
    block_size: int | None = None

    # Farneback params (ignored by other backends).
    fb_pyr_scale: float = 0.5
    fb_levels: int = 3
    fb_winsize: int = 15
    fb_iterations: int = 3
    fb_poly_n: int = 5
    fb_poly_sigma: float = 1.2

    # DIS preset: 0=ultrafast 1=fast 2=medium
    dis_preset: int = 1

    # RAFT
    raft_iters: int = 12

    def resolve_block_size(self, scale: float) -> int:
        """Working-pixel block size at a given scale.

        Tracking rounds ``BASE_BLOCK_SOURCE_PX * scale`` and clamps at 1, so an
        aggressive scale degrades to per-pixel blocks rather than to zero. The
        grid it produces is then within rounding of scale-invariant.
        """
        if self.block_size is not None:
            return int(self.block_size)
        return max(1, round(BASE_BLOCK_SOURCE_PX * float(scale)))


@dataclass(frozen=True)
class FeatureConfig:
    """What gets written to disk.

    The core arrays (u, v, speed) are always cached: they are fundamental and
    everything else in the registry derives from them cheaply. Band-power is
    cached because recomputing an STFT over the full clip on every histogram
    drag would not be interactive.
    """
    bands: tuple[Band, ...] = (Band(12.0, 25.0),)

    # STFT used for band-power, in seconds. window_s sets frequency resolution
    # (df = 1/window_s); hop_s sets the time resolution of the band-power track.
    window_s: float = 1.0
    hop_s: float = 0.25

    # Optional cached expansions. Each one costs disk; the UI shows the cost.
    cache_coherence: bool = False
    cache_divergence_curl: bool = False
    cache_spectral_flatness: bool = False
    cache_direction_oscillation: bool = False

    # Standardization diagnostics that genuinely require frame access. Both are
    # continuous, inspectable fields: analysis-time thresholds remain tunable.
    # Forward/backward error roughly doubles flow compute, while texture adds one
    # cheap structure-tensor pass and one block-grid plane to the cache.
    cache_fb_error: bool = False
    cache_texture: bool = False

    dtype: Literal["float16", "float32"] = "float16"
    compression: Literal["zstd", "lz4", "none"] = "zstd"
    compression_level: int = 5

    def nyquist_hz(self, fps: float) -> float:
        return fps / 2.0

    def validate_bands(self, fps: float) -> list[str]:
        """Return human-readable warnings for bands that the fps can't support.

        A band whose upper edge approaches Nyquist cannot be trusted: real motion
        above Nyquist aliases and folds back down into the band as a false
        signal. We warn above 0.8*Nyquist and reject above Nyquist.
        """
        warnings: list[str] = []
        nyq = self.nyquist_hz(fps)
        for b in self.bands:
            if b.hi_hz > nyq:
                warnings.append(
                    f"Band {b.lo_hz:g}-{b.hi_hz:g} Hz exceeds the Nyquist limit "
                    f"({nyq:.1f} Hz at {fps:.2f} fps). Content above {nyq:.1f} Hz "
                    f"aliases into this band and cannot be distinguished from real "
                    f"signal. Lower the band or use higher-fps footage."
                )
            elif b.hi_hz > 0.8 * nyq:
                warnings.append(
                    f"Band {b.lo_hz:g}-{b.hi_hz:g} Hz reaches {b.hi_hz / nyq:.0%} of "
                    f"Nyquist ({nyq:.1f} Hz). Only ~{fps / b.hi_hz:.1f} samples per "
                    f"cycle at the top edge; power there is unreliable."
                )
        return warnings

    def suggest_band(self, fps: float) -> Band:
        """Propose a default band that the given fps can actually resolve."""
        nyq = self.nyquist_hz(fps)
        return Band(lo_hz=round(min(12.0, 0.4 * nyq), 1),
                    hi_hz=round(min(25.0, 0.8 * nyq), 1))


@dataclass(frozen=True)
class PipelineConfig:
    """The full pipeline config. This is what gets hashed into the cache key."""
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    version: int = CONFIG_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        d = {k: v for k, v in d.items() if not k.startswith("_")}
        bands = tuple(Band(**b) for b in d.get("features", {}).get("bands", []))
        feat = {**d.get("features", {}), "bands": bands}
        return cls(
            preprocess=PreprocessConfig(**d.get("preprocess", {})),
            flow=FlowConfig(**d.get("flow", {})),
            features=FeatureConfig(**feat),
            version=d.get("version", CONFIG_VERSION),
        )

    def cache_key(self, video_hash: str,
                  replicate_geometry_hash: str | None = None,
                  provenance_key: str | None = None) -> str:
        """Stable hash over video, processing settings, and ROI geometry.

        ``provenance_key`` is ``Manifest.provenance_key()`` when the pass read
        pre-transcoded clips. It is a third provenance axis on top of the two
        this key already misses -- the decoder and its bit depth (``FINDINGS.md``
        section 3 trap 3) -- and without it a result computed from a live crop
        and one computed from a clip cut at a different quality compare as equal,
        which they are not: below ``lossless`` the clip's pixels differ from the
        source's, and ``change`` measures exactly the frame-to-frame quantity
        lossy inter-frame coding perturbs.

        Omitted from the blob entirely when absent, rather than hashed as
        ``None``: absence unambiguously means "read from the source", so every
        cache built before clips existed keeps its key and stays valid.

        **Opt-in, and currently opted into by nobody.** No caller passes this
        today, because nothing caches a clip-derived result yet -- the flow cache
        is not the detection path on this branch and ``run_pipeline`` takes no
        manifest. So this is an available guard, not an active one, and the
        obligation lands on whoever first caches something read from clips: the
        provenance key travels in the channel meta as ``clip_provenance``
        (``channel_source.live_channel_source``) and must be threaded to here.
        Miss it and a source-derived result and a clip-derived one collide
        silently in the same cache entry. Recorded in ``todo.md`` under Standing
        decisions, which also carries the recommended fix: make this parameter
        required and keyword-only, so a caller must claim ``provenance_key=None``
        rather than forget it.
        """
        d = {"video": video_hash,
             "replicate_geometry": replicate_geometry_hash,
             **self.to_dict()}
        if provenance_key is not None:
            d["clip_provenance"] = provenance_key
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:16]

    def with_band(self, band: Band) -> "PipelineConfig":
        return replace(self, features=replace(self.features, bands=(band,)))
