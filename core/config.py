"""Configuration objects for the processing pipeline.

Every config here is a frozen dataclass so it can be hashed. That hashability
existed for the feature cache's identity key, which is **gone** along with the
cache -- ``PipelineConfig.cache_key`` was deleted once it had no consumers. It is
kept because the same property makes a config safe to use as a dict key and to
compare for staleness, which the live path's ``TrackStamp`` does.

If something here ever caches a clip-derived result again, its key MUST fold in
``meta["clip_provenance"]``: below ``lossless``, a clip-derived and a
source-derived result are different measurements (FINDINGS.md section 10).

Time is in SECONDS and frequency is in HZ throughout. Frame indices appear only
at the video-decode boundary and in tooltips.
"""
from __future__ import annotations

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
    """Band and diagnostic settings, largely inert since the flow cache went.

    This described what the feature cache wrote to disk. Nothing writes to disk
    now, so most fields below are retained only for round-tripping serialized
    configs (see the flags' own comments). Measured, not assumed: the only member
    with a production consumer is ``suggest_band`` (and ``nyquist_hz`` beneath
    it), called once from ``gui/state.py`` to seed the band picker. ``bands``,
    ``window_s``, ``hop_s``, ``dtype``, ``compression`` and both ``cache_*``
    diagnostics have no callers outside this file.
    """
    bands: tuple[Band, ...] = (Band(12.0, 25.0),)

    # STFT used for band-power, in seconds. window_s sets frequency resolution
    # (df = 1/window_s); hop_s sets the time resolution of the band-power track.
    window_s: float = 1.0
    hop_s: float = 0.25

    # Inert flow-cache expansion flags. The flow cache they configured is gone
    # (its derived-feature compute lived in the deleted core.features); these are
    # retained only so a config dict serialized with them -- e.g. a marks-file
    # provenance block -- still round-trips through from_dict without error.
    cache_coherence: bool = False
    cache_divergence_curl: bool = False
    cache_spectral_flatness: bool = False
    cache_direction_oscillation: bool = False

    # Standardization diagnostics that genuinely require frame access. Also inert
    # now -- they gated cache writes, and nothing reads them. Retained for the
    # same round-trip reason as the flags above. Note that ``texture`` survives as
    # a live channel in core.tensor_channels; only this write flag is dead.
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

        **UNCALLED.** No production code invokes this, and it reads ``self.bands``
        which only ``gui/state.py`` writes and nothing reads. The live aliasing
        guard is structural instead: ``core.wavelet.default_freqs`` caps the
        frequency bank at 0.45*fps, so an above-Nyquist band is unreachable on
        the axis rather than warned about. Retained because the prose is the
        clearest statement of why the limit matters -- but wire it up before
        treating it as a check that runs.
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
    """The full pipeline config.

    Was hashed into the feature cache's identity key; ``cache_key`` is deleted
    and this is now just the settings bundle the live pass and the headless CLI
    are configured from.
    """
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

    def with_band(self, band: Band) -> "PipelineConfig":
        return replace(self, features=replace(self.features, bands=(band,)))
