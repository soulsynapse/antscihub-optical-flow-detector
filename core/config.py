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

# The working resolution the downsample factor targets by default. Chosen so a
# 5.3K GoPro frame lands near 1328px wide (factor 0.25) and a 1080p frame lands
# near 960px (factor 0.5), matching the handoff's lean-cache intent without
# hardcoding a factor that only makes sense at one input resolution.
DEFAULT_TARGET_WIDTH = 1300


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

    Defaults are all-off except downsampling, so a first-time user can hit
    "Run test" immediately and get a result.
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

    # Downsampling. None means "derive from DEFAULT_TARGET_WIDTH at load time",
    # which is the default. An explicit float overrides that.
    downsample: float | None = None

    # Contrast / illumination normalization.
    normalize: Literal["off", "clahe", "zscore"] = "off"

    def resolve_downsample(self, src_width: int) -> float:
        """Pick the downsample factor for a given source width."""
        if self.downsample is not None:
            return float(self.downsample)
        if src_width <= 0:
            return 1.0
        return min(1.0, DEFAULT_TARGET_WIDTH / src_width)


@dataclass(frozen=True)
class FlowConfig:
    """Optical flow backend selection and block reduction."""
    backend: Literal["farneback", "dis", "raft"] = "farneback"
    block_size: int = 16

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

    def to_json(self, video_hash: str | None = None) -> str:
        payload = self.to_dict()
        payload["_video_hash"] = video_hash
        return json.dumps(payload, indent=2, sort_keys=True)

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
                  replicate_geometry_hash: str | None = None) -> str:
        """Stable hash over video, processing settings, and ROI geometry."""
        blob = json.dumps(
            {"video": video_hash,
             "replicate_geometry": replicate_geometry_hash,
             **self.to_dict()}, sort_keys=True
        ).encode()
        return hashlib.sha1(blob).hexdigest()[:16]

    def with_band(self, band: Band) -> "PipelineConfig":
        return replace(self, features=replace(self.features, bands=(band,)))
