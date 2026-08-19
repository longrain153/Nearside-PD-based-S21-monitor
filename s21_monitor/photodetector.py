"""Low-bandwidth photodetector model: square law + low-pass FIR + noise."""

from __future__ import annotations

import numpy as np

from .filters import causal_conv


def photodetect(
    yI: np.ndarray,
    yQ: np.ndarray,
    M: np.ndarray,
    snr_db: float | None = None,
    rng: np.random.Generator | None = None,
    decimate: int = 1,
) -> np.ndarray:
    """Detected intensity waveform z[n] = M * (yI^2 + yQ^2) (+ noise).

    ``M`` is the cascaded impulse response of the PD front end and the
    ADC (the observation channel). When ``snr_db`` is given, white
    Gaussian noise is added at that SNR relative to the AC power of z.

    ``decimate`` models a low-rate monitoring ADC: only every
    ``decimate``-th sample of z is returned (the PD response M is the
    anti-alias filter, so choose the ADC rate fs/decimate above twice
    the PD bandwidth). Pass the same factor to ``fit_monitor`` as
    ``sample_every``.
    """
    p = yI**2 + yQ**2
    z = causal_conv(np.asarray(M, dtype=float), p)
    if decimate > 1:
        z = z[::decimate].copy()
    if snr_db is not None:
        rng = rng or np.random.default_rng()
        ac_power = np.var(z)
        noise_rms = np.sqrt(ac_power * 10.0 ** (-snr_db / 10.0))
        z = z + noise_rms * rng.standard_normal(len(z))
    return z
