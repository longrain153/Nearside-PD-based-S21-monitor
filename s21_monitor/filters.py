"""FIR filter design helpers and convolution utilities (NumPy only)."""

from __future__ import annotations

import numpy as np


def lowpass_fir(num_taps: int, cutoff: float, window: str = "blackman") -> np.ndarray:
    """Windowed-sinc low-pass FIR filter.

    Parameters
    ----------
    num_taps : filter length (odd recommended, linear phase).
    cutoff : cutoff frequency normalized to Nyquist (0 < cutoff < 1).
    window : "blackman" or "hamming".
    """
    if not 0.0 < cutoff < 1.0:
        raise ValueError("cutoff must be in (0, 1) (normalized to Nyquist)")
    n = np.arange(num_taps) - (num_taps - 1) / 2.0
    h = cutoff * np.sinc(cutoff * n)
    if window == "blackman":
        h *= np.blackman(num_taps)
    elif window == "hamming":
        h *= np.hamming(num_taps)
    else:
        raise ValueError(f"unknown window: {window}")
    return h / h.sum()


def fractional_delay_fir(num_taps: int, delay: float) -> np.ndarray:
    """Windowed-sinc fractional-delay filter.

    Total delay is (num_taps - 1) / 2 + delay samples; ``delay`` is the
    fractional offset from the linear-phase center and should satisfy
    |delay| << num_taps / 2.
    """
    n = np.arange(num_taps) - (num_taps - 1) / 2.0
    h = np.sinc(n - delay) * np.blackman(num_taps)
    return h / h.sum()


def rrc_fir(num_taps: int, rolloff: float, sps: float) -> np.ndarray:
    """Root-raised-cosine pulse-shaping filter.

    Parameters
    ----------
    num_taps : filter length.
    rolloff : roll-off factor in (0, 1].
    sps : samples per symbol.
    """
    t = (np.arange(num_taps) - (num_taps - 1) / 2.0) / sps
    h = np.empty(num_taps)
    b = rolloff
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - b + 4.0 * b / np.pi
        elif b > 0 and abs(abs(ti) - 1.0 / (4.0 * b)) < 1e-12:
            h[i] = (b / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * b))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * b))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - b)) + 4.0 * b * ti * np.cos(
                np.pi * ti * (1.0 + b)
            )
            den = np.pi * ti * (1.0 - (4.0 * b * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h**2))


def freq_response(h: np.ndarray, nfft: int = 4096, fs: float = 1.0):
    """One-sided frequency response of an FIR filter.

    Returns (f, H) where f is in the units of ``fs``.
    """
    H = np.fft.rfft(h, nfft)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return f, H


def fft_conv_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full linear convolution via FFT (equivalent to np.convolve(a, b))."""
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    out = np.fft.irfft(np.fft.rfft(a, nfft) * np.fft.rfft(b, nfft), nfft)
    return out[:n]


def causal_conv(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Causal FIR filtering y[n] = sum_l h[l] x[n-l], truncated to len(x)."""
    return fft_conv_full(h, x)[: len(x)]
