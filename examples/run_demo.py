"""End-to-end demo: monitor a 100 GBaud-class coherent transmitter with a
5 GHz photodetector.

A 16QAM signal at 100 GBaud (2 samples/symbol, fs = 200 GSa/s) drives a
transmitter with realistic linear imperfections (branch bandwidth
limitations, IQ amplitude/phase imbalance, IQ skew). The optical
intensity is detected by a PD modeled as a square law followed by a
5 GHz low-pass filter, sampled with 30 dB SNR. The monitor then jointly
learns the 2x2 transmitter response L_ij[l] and the PD/ADC response
M[m] from (xI, xQ, z) alone, and the linear imperfections are extracted
from the estimate.

Run from the repository root:

    python3 examples/run_demo.py

Takes a few minutes (full-batch training on a 32k-sample snapshot);
figures and a summary are written to examples/output/.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s21_monitor import (
    analyze_iq,
    fit_monitor_hybrid,
    lowpass_fir,
    make_transmitter,
    photodetect,
    rrc_fir,
)
from s21_monitor.filters import causal_conv, freq_response
from s21_monitor.metrics import branch_responses

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def make_qam_drive(rng, n_samples, sps=2, rolloff=0.1):
    """Random 16QAM drive signals, RRC pulse-shaped, unit variance."""
    levels = np.array([-3.0, -1.0, 1.0, 3.0])
    h = rrc_fir(65, rolloff, sps)
    out = []
    for _ in range(2):
        up = np.zeros(n_samples)
        up[::sps] = rng.choice(levels, n_samples // sps)
        x = causal_conv(h, up)
        out.append(x / np.std(x))
    return out


def main():
    rng = np.random.default_rng(42)
    fs = 200e9  # 200 GSa/s -> 2 samples/symbol at 100 GBaud
    n_samples = 1 << 15

    # --- known digital drive (in-service traffic) ------------------------
    xI, xQ = make_qam_drive(rng, n_samples)

    # --- transmitter under test (unknown to the monitor) ------------------
    tx = make_transmitter(
        branch_taps=41,
        bw_i=0.55,  # 55 GHz I-branch bandwidth
        bw_q=0.50,  # 50 GHz Q-branch bandwidth
        gain_q=10 ** (-0.8 / 20),  # -0.8 dB IQ amplitude imbalance
        phase_error_deg=5.0,  # 5 deg quadrature phase error
        skew_samples=0.6,  # 3 ps IQ skew at 200 GSa/s
        delay_taps=15,
    )
    yI, yQ = tx.apply(xI, xQ)

    # --- 5 GHz photodetector (also unknown to the monitor) ---------------
    M_true = lowpass_fir(161, 5e9 / (fs / 2))
    z = photodetect(yI, yQ, M_true, snr_db=30, rng=rng)

    # --- joint learning of COI and OC -------------------------------------
    # Two stages: frequency-domain Adam equalizes convergence across the
    # band (accuracy at weakly excited band-edge frequencies), then a
    # tap-domain polish restores full precision in the strongly excited
    # region and in the derived IQ metrics.
    print("Fitting the monitor (8000 freq-domain + 8000 tap-domain iterations)...")
    t0 = time.time()
    res = fit_monitor_hybrid(
        xI, xQ, z,
        L_len=55, M_len=161,
        n_iter_freq=8000, n_iter_tap=8000,
        verbose_every=2000,
    )
    print(f"done in {time.time() - t0:.0f} s, fit NMSE {res.nmse_db:.1f} dB")

    # --- extracted imperfections ------------------------------------------
    truth = analyze_iq(tx.L, fs=fs)
    est = analyze_iq(res.L, fs=fs)
    print(f"\n{'metric':24s} {'true':>12s} {'estimated':>12s}")
    rows = [
        ("amp imbalance (dB)", "amp_imbalance_db", 1.0),
        ("phase error (deg)", "phase_error_deg", 1.0),
        ("IQ skew (ps)", "skew_seconds", 1e12),
    ]
    lines = []
    for label, key, scale in rows:
        line = f"{label:24s} {truth[key]*scale:12.3f} {est[key]*scale:12.3f}"
        print(line)
        lines.append(line)

    # --- figures -----------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    nfft = 4096
    f, HI_t, HQ_t = branch_responses(tx.L, nfft=nfft, fs=fs)
    _, HI_e, HQ_e = branch_responses(res.L, nfft=nfft, fs=fs)
    fghz = f / 1e9
    # The COI is only identifiable where the drive has energy: the RRC-0.1
    # 100 GBaud signal rolls off at 55 GHz.
    band = fghz <= 55

    def norm_db(H):
        return 20 * np.log10(np.maximum(np.abs(H) / np.abs(H[0]), 1e-6))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(fghz[band], norm_db(HI_t)[band], "k-", label="true I branch")
    ax[0].plot(fghz[band], norm_db(HI_e)[band], "r--", label="estimated I branch")
    ax[0].plot(fghz[band], norm_db(HQ_t)[band], "b-", label="true Q branch")
    ax[0].plot(fghz[band], norm_db(HQ_e)[band], "g--", label="estimated Q branch")
    ax[0].set_xlabel("frequency (GHz)")
    ax[0].set_ylabel("normalized |S21| (dB)")
    ax[0].set_ylim(-40, 5)
    ax[0].set_title("Transmitter branch responses (COI), signal band")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    fM, HM_t = freq_response(M_true, nfft=nfft, fs=fs)
    _, HM_e = freq_response(res.M, nfft=nfft, fs=fs)
    bm = fM / 1e9 <= 20
    ax[1].plot(fM[bm] / 1e9, norm_db(HM_t)[bm], "k-", label="true PD (5 GHz)")
    ax[1].plot(fM[bm] / 1e9, norm_db(HM_e)[bm], "r--", label="estimated PD")
    ax[1].set_xlabel("frequency (GHz)")
    ax[1].set_ylabel("normalized |M| (dB)")
    ax[1].set_ylim(-60, 5)
    ax[1].set_title("Photodetector/ADC response (OC)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    for name, Ht, He, color in (("I", HI_t, HI_e, "r"), ("Q", HQ_t, HQ_e, "g")):
        err_db = norm_db(He)[band] - norm_db(Ht)[band]
        ax[2].plot(fghz[band], err_db, color, label=f"{name} branch")
    ax[2].set_xlabel("frequency (GHz)")
    ax[2].set_ylabel("|S21| estimation error (dB)")
    ax[2].set_ylim(-1, 1)
    ax[2].set_title("COI magnitude error, signal band")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.suptitle("100 GBaud-class transmitter monitored through a 5 GHz PD")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "responses.png"), dpi=150)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.semilogy(res.loss)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("MSE of z fit")
    ax2.set_title("Learning curve")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "learning_curve.png"), dpi=150)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as fh:
        fh.write(f"fit NMSE: {res.nmse_db:.2f} dB\n")
        fh.write(f"{'metric':24s} {'true':>12s} {'estimated':>12s}\n")
        fh.write("\n".join(lines) + "\n")

    print(f"\nFigures and summary written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
