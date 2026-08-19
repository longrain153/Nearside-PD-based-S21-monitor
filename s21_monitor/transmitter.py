"""Widely linear (2x2 real FIR) model of a coherent transmitter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filters import causal_conv, fractional_delay_fir, fft_conv_full, lowpass_fir


@dataclass
class WidelyLinearTransmitter:
    """Coherent transmitter modeled as a 2x2 widely linear FIR filter.

    ``L`` has shape (2, 2, L_len); L[i, j] maps drive j onto field
    quadrature i, with (I, Q) indexed as (0, 1):

        yI = L[0,0] * xI + L[0,1] * xQ
        yQ = L[1,0] * xI + L[1,1] * xQ
    """

    L: np.ndarray

    def __post_init__(self):
        self.L = np.asarray(self.L, dtype=float)
        if self.L.shape[:2] != (2, 2):
            raise ValueError("L must have shape (2, 2, L_len)")

    def apply(self, xI: np.ndarray, xQ: np.ndarray):
        """Return the field quadratures (yI, yQ), truncated to len(xI)."""
        x = (xI, xQ)
        yI = sum(causal_conv(self.L[0, j], x[j]) for j in range(2))
        yQ = sum(causal_conv(self.L[1, j], x[j]) for j in range(2))
        return yI, yQ


def make_transmitter(
    branch_taps: int = 41,
    bw_i: float = 0.55,
    bw_q: float = 0.50,
    gain_q: float = 1.0,
    phase_error_deg: float = 0.0,
    skew_samples: float = 0.0,
    delay_taps: int = 15,
) -> WidelyLinearTransmitter:
    """Build a transmitter with typical linear imperfections.

    Parameters
    ----------
    branch_taps : length of the per-branch low-pass responses.
    bw_i, bw_q : I/Q branch bandwidths, normalized to Nyquist.
    gain_q : Q-branch amplitude relative to I (amplitude imbalance).
    phase_error_deg : quadrature phase error phi. The modulator produces
        E = hI*xI(t) cos(wt) - g_Q hQ*xQ(t - skew) sin(wt + phi), so at
        baseband  yI = hI*xI - sin(phi) g_Q hQ*xQ',
                  yQ = cos(phi) g_Q hQ*xQ'.
    skew_samples : Q-branch delay relative to I, in samples (fractional OK).
    delay_taps : length of the fractional-delay filter used for the skew.
    """
    hI = fft_conv_full(lowpass_fir(branch_taps, bw_i), fractional_delay_fir(delay_taps, 0.0))
    hQ = fft_conv_full(
        lowpass_fir(branch_taps, bw_q), fractional_delay_fir(delay_taps, skew_samples)
    )
    phi = np.deg2rad(phase_error_deg)
    L_len = branch_taps + delay_taps - 1
    L = np.zeros((2, 2, L_len))
    L[0, 0] = hI
    L[0, 1] = -np.sin(phi) * gain_q * hQ
    L[1, 0] = 0.0
    L[1, 1] = np.cos(phi) * gain_q * hQ
    return WidelyLinearTransmitter(L)
