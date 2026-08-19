"""End-to-end convergence test on a scaled-down scenario.

The transmitter has ~10x the PD bandwidth so the test exercises the key
property of the method: recovering the full-bandwidth COI through a
narrow, unknown OC. The scenario is smaller than the demo to keep the
runtime reasonable.
"""

import numpy as np
import pytest

from s21_monitor import (
    analyze_iq,
    fit_monitor,
    fit_monitor_hybrid,
    lowpass_fir,
    make_transmitter,
    photodetect,
    rrc_fir,
)
from s21_monitor.filters import causal_conv
from s21_monitor.metrics import branch_responses


def _qam_drive(rng, n_samples, sps=2):
    levels = np.array([-3.0, -1.0, 1.0, 3.0])
    h = rrc_fir(65, 0.1, sps)
    out = []
    for _ in range(2):
        up = np.zeros(n_samples)
        up[::sps] = rng.choice(levels, n_samples // sps)
        x = causal_conv(h, up)
        out.append(x / np.std(x))
    return out


def test_plain_gradient_descent_converges():
    """The paper's LMS (plain gradient descent) path also converges."""
    rng = np.random.default_rng(3)
    xI, xQ = _qam_drive(rng, 4096)
    tx = make_transmitter(
        branch_taps=15, bw_i=0.55, bw_q=0.50, gain_q=0.95,
        phase_error_deg=3.0, skew_samples=0.3, delay_taps=7,
    )
    yI, yQ = tx.apply(xI, xQ)
    z = photodetect(yI, yQ, lowpass_fir(41, 0.08), snr_db=40, rng=rng)
    res = fit_monitor(xI, xQ, z, L_len=21, M_len=41, n_iter=800, optimizer="gd")
    assert np.isfinite(res.loss[-1])
    assert res.loss[-1] < 0.05 * res.loss[0]


def test_monitor_recovers_coi_through_narrow_oc():
    rng = np.random.default_rng(7)
    n = 1 << 13
    xI, xQ = _qam_drive(rng, n)

    tx = make_transmitter(
        branch_taps=21,
        bw_i=0.55,
        bw_q=0.50,
        gain_q=10 ** (-0.8 / 20),
        phase_error_deg=5.0,
        skew_samples=0.6,
        delay_taps=9,
    )
    yI, yQ = tx.apply(xI, xQ)
    M_true = lowpass_fir(81, 0.05)  # OC ~10x narrower than the signal
    z = photodetect(yI, yQ, M_true, snr_db=35, rng=rng)

    res = fit_monitor_hybrid(
        xI, xQ, z,
        L_len=29, M_len=81,
        n_iter_freq=5000, n_iter_tap=5000,
    )

    assert res.nmse_db < -25.0

    truth = analyze_iq(tx.L)
    est = analyze_iq(res.L)
    assert est["amp_imbalance_db"] == pytest.approx(
        truth["amp_imbalance_db"], abs=0.15
    )
    assert est["phase_error_deg"] == pytest.approx(
        truth["phase_error_deg"], abs=0.75
    )
    assert est["skew_samples"] == pytest.approx(truth["skew_samples"], abs=0.1)

    # Full-band branch magnitude responses recovered through the narrow OC:
    # tight accuracy across 90% of the excited band (RRC-0.1 band edge is
    # at 0.55x Nyquist), i.e. up to ~18x the OC bandwidth.
    f, HI_t, HQ_t = branch_responses(tx.L, nfft=1024)
    _, HI_e, HQ_e = branch_responses(res.L, nfft=1024)
    band = f <= 0.45 * 0.5
    for Ht, He in ((HI_t, HI_e), (HQ_t, HQ_e)):
        mag_t = np.abs(Ht[band]) / np.abs(Ht[0])
        mag_e = np.abs(He[band]) / np.abs(He[0])
        err_db = 20 * np.log10(mag_e / mag_t)
        assert np.max(np.abs(err_db)) < 0.5
