"""Morlet continuous wavelet transform for per-block channel time series.

The scalogram is the multi-scale generalization of the pipeline's fixed band
power: band power is this scalogram summed over one frequency band. It is the
time-frequency representation the ZeChat/Berman unsupervised-behavior recipe runs
on (see docs/expanded_cache_plan.md).

FFT-based on purpose: pywt is not a dependency and scipy.signal.cwt was removed
in scipy 1.15+. This is the standard Torrence & Compo (1998) construction with
w0=6, normalized so power is comparable across scales.

Uses scipy.fft rather than numpy.fft: it keeps float32 in single precision
(complex64, half the memory traffic), pads to a fast composite length (a prime
T would otherwise fall back to Bluestein), and threads across block columns
(workers=-1). Together ~10x on the explorer's per-block (T, B) workload.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as _fft

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
    x = np.asarray(x, np.float32)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    T = x.shape[0]
    dt = 1.0 / fs
    scales = morlet_scales(freqs_hz)
    # Zero-pad past the largest wavelet's e-folding support, then round up to a
    # fast composite length. This is the Torrence & Compo zero-padding: the
    # ends see zeros instead of circularly wrapping onto the other end of the
    # record (and a prime T would force the FFT into a Bluestein fallback).
    support = int(np.ceil(np.sqrt(2.0) * scales.max() / dt))
    n = _fft.next_fast_len(T + support)
    Xf = _fft.fft(x, n=n, axis=0, workers=-1)          # complex64
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)
    heavi = omega > 0
    out = np.empty((len(scales), *x.shape), np.float32)
    buf = np.empty_like(Xf)                # reused scratch: Xf * daughter
    for i, s in enumerate(scales):
        norm = np.sqrt(2.0 * np.pi * s / dt) * np.pi ** -0.25
        daughter = (norm * heavi * np.exp(-0.5 * (s * omega - W0) ** 2)) \
            .astype(np.complex64)
        np.multiply(Xf, daughter[:, None], out=buf)
        w = _fft.ifft(buf, axis=0, workers=-1, overwrite_x=True)[:T]
        out[i] = w.real ** 2 + w.imag ** 2
    return out[:, :, 0] if squeeze else out
