"""Preprocessing pipeline: everything that happens to a frame before flow.

Each step is independently toggleable and defaults to off (except downsampling),
so a first run works without configuration. Steps are stateful across frames
(registration needs a reference, denoising and background subtraction need a
history), so this is a class, not a pure function.

Order matters and is fixed:
    downsample -> mask -> registration -> denoise -> bg subtract -> normalize

Downsampling first is the single biggest compute win and everything downstream
is cheaper for it. Registration must precede any temporal operation, because a
rolling median over unregistered frames smears camera motion into the estimate.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace

import cv2
import numpy as np

from core.config import PreprocessConfig


class Preprocessor:
    def __init__(self, cfg: PreprocessConfig, src_width: int, src_height: int,
                 mask_image: np.ndarray | None = None):
        self.cfg = cfg
        self.scale = cfg.resolve_downsample(src_width)
        self.width = max(1, int(round(src_width * self.scale)))
        self.height = max(1, int(round(src_height * self.scale)))

        self._mask: np.ndarray | None = None
        if mask_image is not None:
            m = np.asarray(mask_image)
            if m.ndim == 3:
                m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
            self._mask = (cv2.resize(m, (self.width, self.height),
                                     interpolation=cv2.INTER_NEAREST) > 127)
        elif cfg.mask_path:
            m = cv2.imread(cfg.mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise IOError(f"Could not read mask: {cfg.mask_path}")
            self._mask = (cv2.resize(m, (self.width, self.height),
                                     interpolation=cv2.INTER_NEAREST) > 127)

        self._ref_gray: np.ndarray | None = None
        self._orb = None
        self._matcher = None
        if cfg.registration == "orb":
            self._orb = cv2.ORB_create(nfeatures=1000)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self._ref_kp = None
            self._ref_desc = None

        self._history: deque[np.ndarray] = deque(maxlen=max(1, cfg.denoise_window))
        self._bg: np.ndarray | None = None
        self._mog2 = None
        if cfg.bg_subtract == "mog2":
            self._mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

        self._clahe = None
        if cfg.normalize == "clahe":
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # -- reference frame for registration / background model ------------------

    def set_reference(self, bgr: np.ndarray) -> None:
        """Give the preprocessor a reference frame (for registration)."""
        gray = self._to_gray(self._downsample(bgr))
        self._ref_gray = gray
        if self._orb is not None:
            self._ref_kp, self._ref_desc = self._orb.detectAndCompute(gray, None)

    def fit_background(self, frames: list[np.ndarray]) -> None:
        """Build a temporal-median background from sampled frames."""
        if self.cfg.bg_subtract != "median" or not frames:
            return
        stack = np.stack([self._to_gray(self._downsample(f)) for f in frames])
        self._bg = np.median(stack, axis=0).astype(np.float32)

    # -- the pipeline ---------------------------------------------------------

    def apply(self, bgr: np.ndarray) -> np.ndarray:
        """Return a float32 grayscale image ready for optical flow."""
        small = self._downsample(bgr)
        gray = self._to_gray(small)

        if self.cfg.registration != "off":
            gray = self._register(gray)

        if self._mask is not None:
            gray = np.where(self._mask, gray, 0).astype(np.float32)

        if self.cfg.denoise != "off":
            gray = self._denoise(gray)

        if self.cfg.bg_subtract != "off":
            gray = self._subtract_bg(gray)

        if self.cfg.normalize != "off":
            gray = self._normalize(gray)

        return gray.astype(np.float32)

    @property
    def mask(self) -> np.ndarray | None:
        return self._mask

    # -- steps ----------------------------------------------------------------

    def _downsample(self, bgr: np.ndarray) -> np.ndarray:
        if self.scale >= 1.0:
            return bgr
        if bgr.shape[:2] == (self.height, self.width):
            # Already at the target size: the ROI decoder scales inside FFmpeg,
            # so this would be a resize to the size it already is. INTER_AREA at
            # scale 1 is bit-identical to its input (verified), so skipping is
            # free of any numerical change -- and it is not free to run: 610 us
            # per call on a 760x730 tile, once per replicate per frame.
            return bgr
        # INTER_AREA is the correct choice for shrinking: it area-averages, which
        # low-pass filters before decimating. INTER_LINEAR would alias high
        # spatial frequencies into the flow field.
        return cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _to_gray(bgr: np.ndarray) -> np.ndarray:
        if bgr.ndim == 2:
            return bgr.astype(np.float32)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def _register(self, gray: np.ndarray) -> np.ndarray:
        if self._ref_gray is None:
            self.set_reference(gray)
            return gray

        if self.cfg.registration == "phase":
            # Phase correlation recovers translation only. Fast, and sufficient
            # for tripod-mounted footage with slight drift or vibration.
            shift, _ = cv2.phaseCorrelate(
                np.float32(self._ref_gray), np.float32(gray)
            )
            M = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
            return cv2.warpAffine(gray, M, (self.width, self.height),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

        # ORB + homography handles rotation, scale and parallax-free pan.
        kp, desc = self._orb.detectAndCompute(gray.astype(np.uint8), None)
        if desc is None or self._ref_desc is None or len(kp) < 8:
            return gray
        matches = self._matcher.match(self._ref_desc, desc)
        if len(matches) < 8:
            return gray
        src = np.float32([self._ref_kp[m.queryIdx].pt for m in matches])
        dst = np.float32([kp[m.trainIdx].pt for m in matches])
        H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 3.0)
        if H is None:
            return gray
        return cv2.warpPerspective(gray, H, (self.width, self.height),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        self._history.append(gray)
        if len(self._history) < self._history.maxlen:
            return gray
        stack = np.stack(self._history)
        if self.cfg.denoise == "median":
            return np.median(stack, axis=0).astype(np.float32)
        return stack.mean(axis=0).astype(np.float32)

    def _subtract_bg(self, gray: np.ndarray) -> np.ndarray:
        if self.cfg.bg_subtract == "mog2":
            fg = self._mog2.apply(gray.astype(np.uint8))
            return (gray * (fg > 0)).astype(np.float32)
        if self._bg is None:
            return gray
        return np.abs(gray - self._bg).astype(np.float32)

    def _normalize(self, gray: np.ndarray) -> np.ndarray:
        if self.cfg.normalize == "clahe":
            # KNOWN ISSUE (see KNOWN_ISSUES.md): CLAHE runs per replicate box, per
            # frame, on the hard crop. Its edge tiles have truncated histograms
            # that clipLimit amplifies, and because the tile grid shifts with any
            # added context it perturbs every block, not just the boundary. On
            # difficult footage this produces large phantom edge speeds (replicate
            # 23: 861 px/s, 60 crossings vs 48 px/s / 0 with CLAHE off). Prefer
            # z-score until the halo/global-normalization rework lands.
            g = np.clip(gray, 0, 255).astype(np.uint8)
            return self._clahe.apply(g).astype(np.float32)
        std = float(gray.std())
        if std < 1e-6:
            return gray - float(gray.mean())
        return (gray - float(gray.mean())) / std * 32.0 + 128.0


def sample_frames_for_background(source, n: int) -> list[np.ndarray]:
    """Evenly sample n frames across the clip for the median background model."""
    total = source.info.frame_count
    idxs = np.linspace(0, max(0, total - 1), num=min(n, total), dtype=int)
    return [f for f in (source.frame_at(int(i)) for i in idxs) if f is not None]


def flow_input_preview(bgr: np.ndarray, work_size: tuple[int, int],
                       cfg: PreprocessConfig) -> np.ndarray:
    """Displayable BGR preview at the spatial resolution supplied to flow.

    Downsampling, grayscale conversion and contrast normalization are replayed
    exactly. Registration, temporal denoising, background subtraction and masks
    need run history/reference assets that are not stored in the feature cache,
    so this diagnostic intentionally omits those stateful/contextual steps.
    """
    work_w, work_h = map(int, work_size)
    h, w = bgr.shape[:2]
    scale = work_w / max(1, w)
    preview_cfg = replace(
        cfg,
        downsample=scale,
        mask_path=None,
        registration="off",
        denoise="off",
        bg_subtract="off",
    )
    gray = Preprocessor(preview_cfg, w, h).apply(bgr)
    if gray.shape != (work_h, work_w):
        gray = cv2.resize(gray, (work_w, work_h), interpolation=cv2.INTER_AREA)
    gray8 = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
