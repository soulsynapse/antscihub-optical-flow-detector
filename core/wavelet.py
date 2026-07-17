"""Morlet continuous wavelet transform for per-block channel time series.

The scalogram is the multi-scale generalization of the pipeline's fixed band
power: band power is this scalogram summed over one frequency band. It is the
time-frequency representation the ZeChat/Berman unsupervised-behavior recipe runs
on (see docs/expanded_cache_plan.md).

FFT-based on purpose: pywt is not a dependency and scipy.signal.cwt was removed
in scipy 1.15+. This is the standard Torrence & Compo (1998) construction with
w0=6, normalized so power is comparable across scales.
"""
from __future__ import annotations

import numpy as np

W0 = 6.0    # Morlet nondimensional frequency


def morlet_scales(freqs_hz: np.ndarray) -> np.ndarray:
    """Wavelet scale s for each desired Fourier frequency (w0=6 Morlet)."""
    f = np.asarray(freqs_hz, float)
    return (W0 + np.sqrt(2.0 + W0 * W0)) / (4.0 * np.pi * f)


def default_freqs(fps: float, fmin: float = 0.5, fmax: float = 25.0,
                  n: int = 24) -> np.ndarray:
    """Log-spaced frequency bank, capped below Nyquist."""
    return np.geomspace(fmin, min(fmax, 0.45 * fps), n)


def morlet_power(x: np.ndarray, fs: float, freqs_hz: np.ndarray) -> np.ndarray:
    """Morlet scalogram power. ``x`` (T,) or (T,B) -> (F,T) or (F,T,B) float32.

    Loops frequencies to bound memory; each is one FFT-domain multiply plus an
    inverse FFT along the time axis.
    """
    x = np.asarray(x, np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    T = x.shape[0]
    dt = 1.0 / fs
    Xf = np.fft.fft(x, axis=0)
    omega = 2.0 * np.pi * np.fft.fftfreq(T, d=dt)
    heavi = (omega > 0).astype(np.float64)
    scales = morlet_scales(freqs_hz)
    out = np.empty((len(scales), *x.shape), np.float32)
    for i, s in enumerate(scales):
        norm = np.sqrt(2.0 * np.pi * s / dt) * np.pi ** -0.25
        daughter = norm * heavi * np.exp(-0.5 * (s * omega - W0) ** 2)
        w = np.fft.ifft(Xf * daughter[:, None], axis=0)
        out[i] = (np.abs(w) ** 2).astype(np.float32)
    return out[:, :, 0] if squeeze else out
