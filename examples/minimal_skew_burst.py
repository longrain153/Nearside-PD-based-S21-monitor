"""Scheme A: TX-side pre-cancellation of the data self-beat.

No monitor-side data knowledge. Shallow full-branch dither eps=0.08
(~-25 dB, EVM-neutral). The TX DSP predicts the readout-band component
of the data intensity from its OWN data and its current H estimate,
and injects a tiny common-mode gain g(t) = -d_hat/(2*Pbar) that cancels
it in the optical intensity. The monitor just reads the line at nu.

Three arms:
  0: no cancellation           (baseline penalty)
  1: cancel with coarse H-hat  (+-10% magnitude, +-0.1 rad ripple, 1 ps skew err)
  2: cancel with exact H-hat   (converged limit)
Skew burst config: full-branch AM, nu = 90 MHz, I/Q alternating, phi=0.
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
m_beat = 118                  # nu = 90.03 MHz
nu = m_beat * bin_hz
eps = 0.08                    # SHALLOW dither: 0.32% power ripple
snr_db = 30
R_avg = int(sys.argv[2])
CANCEL_BINS = np.arange(m_beat - 4, m_beat + 5)   # cancellation band

tx = make_transmitter(branch_taps=41, bw_i=0.55, bw_q=0.50,
                      gain_q=10**(-0.8/20), phase_error_deg=0.0,
                      skew_samples=0.6, delay_taps=15)   # 3 ps, phi=0
M_true = lowpass_fir(8001, 100e6 / (fs / 2))
hI_t = tx.L[0, 0]; hQ_t = tx.L[1, 1]
freqs = np.fft.rfftfreq(N, 1 / fs)
HII = np.exp(-2j * np.pi * np.outer(freqs, np.arange(len(hI_t))) / fs) @ hI_t
HQQ = np.exp(-2j * np.pi * np.outer(freqs, np.arange(len(hQ_t))) / fs) @ hQ_t

# coarse H-hat: smooth +-10% magnitude and +-0.1 rad phase ripple + 1 ps skew err
rng0 = np.random.default_rng(7)
def rough(H, seed, extra_delay_ps):
    r = np.random.default_rng(seed)
    nk = 6
    a = r.uniform(-0.10, 0.10, nk); b = r.uniform(-0.10, 0.10, nk)
    x = freqs / 55e9
    rip_a = sum(a[i] * np.cos((i + 1) * np.pi * x) for i in range(nk))
    rip_p = sum(b[i] * np.cos((i + 1) * np.pi * x) for i in range(nk))
    return H * (1 + rip_a) * np.exp(1j * (rip_p - 2 * np.pi * freqs * extra_delay_ps * 1e-12))
HII_coarse = rough(HII, 11, 0.0)
HQQ_coarse = rough(HQQ, 12, 1.0)     # 1 ps skew error in the estimate

bins = np.arange(m_beat - 9, m_beat + 9)
w_win = np.hanning(Nd)
W = np.zeros((len(bins), N), complex)
for k in range(Nd):
    lo = max(0, k * D_adc - len(M_true) + 1)
    hi = min(N, k * D_adc + 1)
    seg = M_true[k * D_adc - np.arange(lo, hi)]
    W[:, lo:hi] += w_win[k] * np.exp(-2j * np.pi * bins[:, None] * k / Nd) * seg[None, :]
i_beat = list(bins).index(m_beat)
E_noise = np.exp(-2j * np.pi * np.outer(bins, np.arange(Nd)) / Nd) * w_win[None, :]

rrc = rrc_fir(65, 0.1, 2)
levels = np.array([-3.0, -1.0, 1.0, 3.0])
rng = np.random.default_rng(int(sys.argv[1]))
c_nu = np.cos(2 * np.pi * nu * np.arange(N) / fs)
band_mask = np.zeros(N // 2 + 1)
band_mask[CANCEL_BINS] = 1.0

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

def g_from(Hi_hat, Hq_hat, XI, XQ):
    """DSP predictor: readout-band data self-beat -> common-mode gain."""
    yi = np.fft.irfft(XI * Hi_hat, N)
    yq = np.fft.irfft(XQ * Hq_hat, N)
    ph = yi * yi + yq * yq
    d_hat = np.fft.irfft(np.fft.rfft(ph) * band_mask, N)
    return -d_hat / (2 * np.mean(ph))

lines = np.zeros((2, R_avg), complex)
for r in range(R_avg):
    xI, xQ = make_data()
    for b in range(2):                        # 0: I-dither record, 1: Q-dither
        mI = 1 + (eps * c_nu if b == 0 else 0)
        mQ = 1 + (eps * c_nu if b == 1 else 0)
        yI = np.fft.irfft(np.fft.rfft(xI * mI) * HII, N)
        yQ = np.fft.irfft(np.fft.rfft(xQ * mQ) * HQQ, N)
        p = yI * yI + yQ * yQ
        v = W @ (p - p.mean())
        v += E_noise @ (noise_std * rng.standard_normal(Nd))
        lines[b, r] = v[i_beat]
    if r % 2000 == 0:
        print(f"[{sys.argv[3]}] r={r} elapsed {time.time()-t_start:.0f}s", flush=True)

np.savez(sys.argv[3], lines=lines, nu=nu)
print(f"done {sys.argv[3]} {time.time()-t_start:.0f}s", flush=True)
