"""Extraction of linear transmitter imperfections from an estimated COI.

With the field defined as y = yI + j yQ, the complex response seen by
each drive is

    H_I(w) = L_II(w) + j L_QI(w)          (xI -> field)
    H_Q(w) = (L_IQ(w) + j L_QQ(w)) / j    (xQ -> field, ideal is j)
           = L_QQ(w) - j L_IQ(w)

The ratio R(w) = H_Q(w) / H_I(w) is invariant to the scale, delay and
rotation ambiguities of the square-law monitor and carries the IQ
imperfections: |R| is the amplitude imbalance, angle(R) at DC the
quadrature phase error, and the slope of angle(R) over frequency the
I/Q skew.
"""

from __future__ import annotations

import numpy as np


def branch_responses(L: np.ndarray, nfft: int = 4096, fs: float = 1.0):
    """Complex drive->field responses (f, H_I, H_Q) of a 2x2 COI."""
    L = np.asarray(L, float)
    F = np.fft.rfft(L, nfft, axis=-1)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    H_I = F[0, 0] + 1j * F[1, 0]
    H_Q = F[1, 1] - 1j * F[0, 1]
    return f, H_I, H_Q


def analyze_iq(
    L: np.ndarray,
    nfft: int = 4096,
    fs: float = 1.0,
    band_frac: float = 0.5,
) -> dict:
    """Estimate IQ amplitude imbalance, phase error and skew from a COI.

    Parameters
    ----------
    L : 2x2xL_len widely linear filter.
    fs : sample rate (skew is returned both in samples and in 1/fs units).
    band_frac : fraction of Nyquist used for the weighted fits.

    Returns a dict with keys ``amp_imbalance_db``, ``phase_error_deg``,
    ``skew_samples`` and ``skew_seconds``.
    """
    f, H_I, H_Q = branch_responses(L, nfft=nfft, fs=fs)
    R = H_Q / np.where(np.abs(H_I) > 1e-12, H_I, 1e-12)
    band = f <= band_frac * (fs / 2.0)
    w = np.abs(H_I[band]) ** 2  # weight by in-band confidence

    amp_db = 20.0 * np.log10(np.abs(R[0]))

    # Weighted linear fit of the unwrapped ratio phase over normalized
    # frequency x = f / (fs/2):  ph(x) ~ phi0 - pi * tau_samples * x
    x = f[band] / (fs / 2.0)
    ph = np.unwrap(np.angle(R[band]))
    slope, phi0 = np.polyfit(x, ph, 1, w=w)
    tau_samples = -slope / np.pi

    return {
        "amp_imbalance_db": float(amp_db),
        "phase_error_deg": float(np.rad2deg(np.angle(np.exp(1j * phi0)))),
        "skew_samples": float(tau_samples),
        "skew_seconds": float(tau_samples / fs),
    }
