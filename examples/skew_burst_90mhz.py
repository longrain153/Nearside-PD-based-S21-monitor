"""Skew via simultaneous dual-frequency full-branch AM with frequency swap.

Record type 1: I branch AM at nu1, Q branch AM at nu2 (same record).
Record type 2: swapped (I at nu2, Q at nu1).
M-free skew readouts (arg M cancels exactly across the swap):
    psi_Q1 - psi_I2 = 2*pi*nu2*(tauQ - tauI)
    psi_Q2 - psi_I1 = 2*pi*nu1*(tauQ - tauI)
phi = 0, eps = 0.1 per branch (0.5% power ripple each).
"""
import sys, time
import numpy as np

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from s21_monitor import make_transmitter, photodetect, lowpass_fir, rrc_fir
from s21_monitor.filters import causal_conv

t_start = time.time()
fs = 200e9
N = 1 << 18
D_adc = 1024
Nd = N // D_adc
bin_hz = fs / N
m1, m2 = 118, 118              # single nu = 90.03 MHz (in-band ideal ADC)
nu1, nu2 = m1 * bin_hz, m2 * bin_hz
eps = 0.1
snr_db = 30
R_avg = 2000

tx = make_transmitter(branch_taps=41, bw_i=0.55, bw_q=0.50,
                      gain_q=10**(-0.8/20), phase_error_deg=0.0,
                      skew_samples=0.6, delay_taps=15)   # skew = 3 ps, phi = 0
M_true = lowpass_fir(8001, 100e6 / (fs / 2))

hI_t = tx.L[0, 0]; hQ_t = tx.L[1, 1]
freqs = np.fft.rfftfreq(N, 1 / fs)
HII = np.exp(-2j * np.pi * np.outer(freqs, np.arange(len(hI_t))) / fs) @ hI_t
HQQ = np.exp(-2j * np.pi * np.outer(freqs, np.arange(len(hQ_t))) / fs) @ hQ_t

bins = np.arange(m1 - 5, m2 + 5)          # covers both lines + noise bins
w_win = np.hanning(Nd)
W = np.zeros((len(bins), N), complex)
for k in range(Nd):
    lo = max(0, k * D_adc - len(M_true) + 1)
    hi = min(N, k * D_adc + 1)
    seg = M_true[k * D_adc - np.arange(lo, hi)]
    W[:, lo:hi] += w_win[k] * np.exp(-2j * np.pi * bins[:, None] * k / Nd) * seg[None, :]
i1 = list(bins).index(m1); i2 = list(bins).index(m2)
E_noise = np.exp(-2j * np.pi * np.outer(bins, np.arange(Nd)) / Nd) * w_win[None, :]

rrc = rrc_fir(65, 0.1, 2)
levels = np.array([-3.0, -1.0, 1.0, 3.0])
rng = np.random.default_rng(31)
c1 = np.cos(2 * np.pi * nu1 * np.arange(N) / fs)
c2 = np.cos(2 * np.pi * nu2 * np.arange(N) / fs)

def make_data():
    out = []
    for _ in range(2):
        up = np.zeros(N); up[::2] = rng.choice(levels, N // 2)
        x = causal_conv(rrc, up); out.append(x / np.std(x))
    return out

_x = make_data()
_y = (np.fft.irfft(np.fft.rfft(_x[0]) * HII, N),
      np.fft.irfft(np.fft.rfft(_x[1]) * HQQ, N))
noise_std = np.sqrt(np.var(photodetect(_y[0], _y[1], M_true, decimate=D_adc))
                    * 10.0 ** (-snr_db / 10))

# lines[type, which, r]: type 0 = (I@nu1, Q@nu2), type 1 = swapped
# which 0 = line at m1, which 1 = line at m2
lines = np.zeros((2, 2, R_avg), complex)
for r in range(R_avg):
    xI, xQ = make_data()
    ydI = np.fft.irfft(np.fft.rfft(xI) * HII, N)
    ydQ = np.fft.irfft(np.fft.rfft(xQ) * HQQ, N)
    for typ in range(2):                     # 0: I-branch record, 1: Q-branch
        if typ == 0:
            dy = np.fft.irfft(np.fft.rfft(eps * xI * c1) * HII, N)
            p_diff = 2 * ydI * dy + dy * dy
        else:
            dy = np.fft.irfft(np.fft.rfft(eps * xQ * c1) * HQQ, N)
            p_diff = 2 * ydQ * dy + dy * dy
        v = W @ p_diff
        v += E_noise @ (noise_std * rng.standard_normal(Nd))
        lines[typ, 0, r] = v[i1]
        lines[typ, 1, r] = v[i1]
    if r % 500 == 0:
        print(f"r={r} elapsed {time.time()-t_start:.0f}s", flush=True)

# truth: band-averaged tau difference at each aperture
RRC = np.abs(np.exp(-2j * np.pi * np.outer(freqs, np.arange(len(rrc))) / fs) @ rrc) ** 2
def true_skew(m):
    TI = np.sum(RRC * np.roll(HII, -m) * np.conj(HII))
    TQ = np.sum(RRC * np.roll(HQQ, -m) * np.conj(HQQ))
    return -np.angle(TQ * np.conj(TI)) / (2 * np.pi * m * bin_hz)
sk_t = true_skew(m1)

def skew_from(sl):
    B_I = lines[0, 0, sl].mean()
    B_Q = lines[1, 0, sl].mean()
    return -np.angle(B_Q * np.conj(B_I)) / (2 * np.pi * nu1)   # same nu: M cancels

np.savez("skew01_lines.npz", lines=lines, nu1=nu1, nu2=nu2, sk_t=sk_t)
print(f"\ndual-frequency swap, eps={eps}/branch, phi=0, "
      f"total {time.time()-t_start:.0f}s")
for blk_size in (R_avg // 16, R_avg // 4, R_avg):
    n_blk = R_avg // blk_size
    ests = np.array([skew_from(slice(i * blk_size, (i + 1) * blk_size))
                     for i in range(n_blk)])
    T_ms = blk_size * N / fs * 1e3
    spread = (f"{np.std(ests)*1e12:.3f} ps over {n_blk} blocks"
              if n_blk > 1 else "single estimate")
    print(f"T = {T_ms:7.2f} ms/branch: skew {np.mean(ests)*1e12:+.3f} ps "
          f"(true {sk_t*1e12:+.3f}), spread {spread}")
snrI = np.abs(lines[0, 0].mean()) / (np.std(lines[0, 0]) / np.sqrt(R_avg))
print(f"I line SNR at full integration: {snrI:.0f}x")
