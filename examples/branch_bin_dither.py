"""Single-bin (multiplicative, data-carried) per-branch dither calibration,
with quadrature error phi = 0 (worst case for cross-line methods).

Per grid frequency f_k and branch X in {I, Q}: the branch drive's
spectral slice around f_k (width B_slice) is amplitude-modulated by
(1 + eps*cos(2*pi*nu*t)). The slice's own data acts as the carrier: its
beat with its dither sidebands puts a coherent line at nu whose complex
amplitude is  eps * S_slice * M(nu) * H_X(f+nu) H_X*(f)  -- data phase
cancels (|X|^2). Cross-branch beats average to zero (independent data),
so this measures each branch separately:

  |B_X|                     -> per-branch |S21| (after known-PSD norm)
  sqrt-ratio                -> amplitude imbalance
  tau_X = -d(arg B_X)/(2*pi*nu), Delta tau = tau_Q - tau_I -> skew
     (arg M(nu) is common to both chains at the same nu and cancels;
      works at ANY phi, including exactly 0)
"""
import sys, time
import numpy as np

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s21_monitor import make_transmitter, photodetect, lowpass_fir, rrc_fir
from s21_monitor.filters import causal_conv

t_start = time.time()
fs = 200e9
N = 1 << 18
D_adc = 2048
Nd = N // D_adc
bin_hz = fs / N
m_beat = 60                    # nu = 39.67 MHz
nu = m_beat * bin_hz
eps = 1.0                      # AM depth on the slice (calibration burst)
B_slice = 4e9                  # dithered slice width (one shaping bin group)
snr_db = 30
R_avg = 1500

PHI_DEG = 0.0                  # quadrature error: exactly zero (worst case)
tx = make_transmitter(branch_taps=41, bw_i=0.55, bw_q=0.50,
                      gain_q=10**(-0.8/20), phase_error_deg=PHI_DEG,
                      skew_samples=0.6, delay_taps=15)   # skew = 3 ps
M_true = lowpass_fir(8001, 100e6 / (fs / 2))

def Hf(h, f):
    return np.exp(-2j * np.pi * np.outer(np.atleast_1d(f), np.arange(len(h))) / fs) @ h

hI_t = tx.L[0, 0]
hQ_t = tx.L[1, 1]              # gQ*hQ (phi=0)

f_grid = np.round(np.arange(4e9, 48.1e9, 4e9) / bin_hz).astype(int) * bin_hz
K = len(f_grid)
fit_band = f_grid <= 40e9      # skew fit away from the band edges

# ---- Hann-windowed projection: M filter -> ADC decimation -> DFT bins ----
bins = np.arange(m_beat - 5, m_beat + 7)
w_win = np.hanning(Nd)
W = np.zeros((len(bins), N), complex)
for k in range(Nd):
    lo = max(0, k * D_adc - len(M_true) + 1)
    hi = min(N, k * D_adc + 1)
    seg = M_true[k * D_adc - np.arange(lo, hi)]
    W[:, lo:hi] += w_win[k] * np.exp(-2j * np.pi * bins[:, None] * k / Nd) * seg[None, :]
i_beat = 5
nb_rows = [i for i in range(len(bins)) if abs(bins[i] - m_beat) > 2]
E_noise = np.exp(-2j * np.pi * np.outer(bins, np.arange(Nd)) / Nd) * w_win[None, :]

# DFT sign convention: a component cos(2*pi*nu*t - psi) reads as ~e^{-j*psi}
t_dec = np.arange(Nd) * D_adc / fs
conv_test = (np.cos(2 * np.pi * nu * t_dec - 0.7) * w_win) @ \
    np.exp(-2j * np.pi * m_beat * np.arange(Nd) / Nd)
assert abs(np.angle(conv_test) + 0.7) < 1e-3   # angle(B) = -psi
# so psi = 2*pi*nu*tau - arg M  =>  tau = -angle(B)/(2*pi*nu) + const(M)

# ---- frequency-domain machinery for the slice dither ----------------------
freqs = np.fft.rfftfreq(N, 1 / fs)
masks = [np.abs(freqs - fk) <= B_slice / 2 for fk in f_grid]
mshift = m_beat                # nu in units of full-record bins
HII = np.exp(-2j * np.pi * np.outer(freqs, np.arange(len(hI_t))) / fs) @ hI_t
HQQ = np.exp(-2j * np.pi * np.outer(freqs, np.arange(len(hQ_t))) / fs) @ hQ_t

rrc = rrc_fir(65, 0.1, 2)
levels = np.array([-3.0, -1.0, 1.0, 3.0])
rng = np.random.default_rng(5)

def make_data():
    out = []
    for _ in range(2):
        up = np.zeros(N); up[::2] = rng.choice(levels, N // 2)
        x = causal_conv(rrc, up); out.append(x / np.std(x))
    return out  # (xI, xQ) drive signals

# ADC noise std referenced to the actual PD output variance
_xI0, _xQ0 = make_data()
_yd = (np.fft.irfft(np.fft.rfft(_xI0) * HII, N),
       np.fft.irfft(np.fft.rfft(_xQ0) * HQQ, N))
z0 = photodetect(_yd[0], _yd[1], M_true, decimate=D_adc)
noise_std = np.sqrt(np.var(z0) * 10.0 ** (-snr_db / 10))

acc = np.zeros((2 * K, len(bins)), complex)
for r in range(R_avg):
    xI, xQ = make_data()
    XI = np.fft.rfft(xI); XQ = np.fft.rfft(xQ)
    ydI = np.fft.irfft(XI * HII, N)                # data through TX (phi=0)
    ydQ = np.fft.irfft(XQ * HQQ, N)
    for k in range(K):
        for b, (Xbr, Hbr, yd_same) in enumerate(
                ((XI, HII, ydI), (XQ, HQQ, ydQ))):
            # spectrum of eps * slice(x) * cos(2 pi nu t): masked spectrum
            # shifted by +-nu (nu = m_beat full-record bins)
            SM = Xbr * masks[k]
            DX = eps * 0.5 * (np.roll(SM, m_beat) + np.roll(SM, -m_beat))
            dy = np.fft.irfft(DX * Hbr, N)
            p_diff = 2 * yd_same * dy + dy * dy
            v = W @ p_diff
            v += E_noise @ (noise_std * rng.standard_normal(Nd))
            acc[2 * k + b] += v
    if r % 100 == 0:
        print(f"r={r} elapsed {time.time()-t_start:.0f}s", flush=True)
spec = acc / R_avg

B = spec[:, i_beat]
noise_p = np.mean(np.abs(spec[:, nb_rows]) ** 2, axis=1)
A = np.sqrt(np.maximum(np.abs(B) ** 2 - noise_p, 1e-30))
snr_line = np.abs(B) / np.sqrt(noise_p + 1e-30)
B_I, B_Q = B[0::2], B[1::2]
A_I, A_Q = A[0::2], A[1::2]
snr_I, snr_Q = snr_line[0::2], snr_line[1::2]

# ---- recoveries -------------------------------------------------------------
# magnitude: |B_X| ~ eps * P_slice(f) * |M(nu)| * |H_X(f)|^2; P_slice from
# the known pulse spectrum (DSP knows its own PSD)
RRC = np.abs(np.exp(-2j * np.pi * np.outer(freqs, np.arange(len(rrc))) / fs) @ rrc) ** 2
p_slice = np.array([np.sum(RRC * m) for m in masks])
magI = np.sqrt(A_I / p_slice); magI /= magI[0]
magQ = np.sqrt(A_Q / p_slice); magQ /= magQ[0]
imb_db = 10 * np.log10(A_Q / A_I)          # 20log10 of |hQ/hI| ratio
d_tau = -np.angle(B_Q * np.conj(B_I)) / (2 * np.pi * nu)   # M(nu) cancels
w_fit = (snr_I * snr_Q)[fit_band]
skew_est = np.average(d_tau[fit_band], weights=w_fit)
skew_std = np.sqrt(np.average((d_tau[fit_band] - skew_est) ** 2, weights=w_fit)
                   / max(fit_band.sum() - 1, 1))

# ---- truth ------------------------------------------------------------------
f2 = f_grid + nu
HI1 = Hf(hI_t, f_grid); HI2 = Hf(hI_t, f2)
HQ1 = Hf(hQ_t, f_grid); HQ2 = Hf(hQ_t, f2)
magI_t = np.sqrt(np.abs(HI1 * HI2)); magI_t /= magI_t[0]
magQ_t = np.sqrt(np.abs(HQ1 * HQ2)); magQ_t /= magQ_t[0]
imb_t = 10 * np.log10(np.abs(HQ1 * HQ2) / np.abs(HI1 * HI2))
tauI_t = -np.angle(HI2 * np.conj(HI1)) / (2 * np.pi * nu)
tauQ_t = -np.angle(HQ2 * np.conj(HQ1)) / (2 * np.pi * nu)
d_tau_t = tauQ_t - tauI_t

print(f"\nphi = {PHI_DEG} deg (cross-line methods give NO line here)")
print(f"K={K} points x 2 branches x {R_avg} avg "
      f"({R_avg*N/fs*1e3:.2f} ms/point), total {time.time()-t_start:.0f}s")
print(f"line SNR: I median {np.median(snr_I):.0f}x, Q median {np.median(snr_Q):.0f}x")
print(f"max | |H_I| err |: {np.max(np.abs(20*np.log10(magI/magI_t))):.3f} dB")
print(f"max | |H_Q| err |: {np.max(np.abs(20*np.log10(magQ/magQ_t))):.3f} dB")
print(f"max |imbalance err|: {np.max(np.abs(imb_db-imb_t)):.3f} dB")
print(f"IQ skew: true 3.000 ps, estimated {skew_est*1e12:.3f} "
      f"+- {skew_std*1e12:.3f} ps")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
fghz = f_grid / 1e9
ax[0].plot(fghz, 20*np.log10(magI_t), "k-", label="true |H_I|")
ax[0].plot(fghz, 20*np.log10(magI), "ro", ms=5, label="est |H_I|")
ax[0].plot(fghz, 20*np.log10(magQ_t), "b-", label="true |H_Q|")
ax[0].plot(fghz, 20*np.log10(magQ), "g^", ms=5, label="est |H_Q|")
ax[0].set_xlabel("frequency (GHz)"); ax[0].set_ylabel("normalized |S21| (dB)")
ax[0].set_title("Per-branch magnitude (bin-dither)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].plot(fghz, imb_t, "k-", label="true")
ax[1].plot(fghz, imb_db, "ro", ms=5, label="measured")
ax[1].set_xlabel("frequency (GHz)"); ax[1].set_ylabel("$|h_Q/h_I|$ (dB)")
ax[1].set_title("Amplitude imbalance"); ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)

ax[2].plot(fghz, d_tau_t*1e12, "k-", label="true $\\tau_Q-\\tau_I$")
ax[2].plot(fghz, d_tau*1e12, "ro", ms=5, label="measured")
ax[2].axhline(skew_est*1e12, color="r", ls="--",
              label=f"weighted mean {skew_est*1e12:.2f} ps")
ax[2].set_xlabel("frequency (GHz)"); ax[2].set_ylabel("$\\Delta\\tau$ (ps)")
ax[2].set_title(f"Skew at $\\varphi=0$: {skew_est*1e12:.2f}"
                f"$\\pm${skew_std*1e12:.2f} ps (true 3.00)")
ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

fig.suptitle("Per-branch single-bin multiplicative dither, $\\varphi = 0$ -- "
             "100 MHz PD + 97.7 MS/s ADC, live 16QAM\n"
             f"slice width {B_slice/1e9:.0f} GHz, AM depth {eps}, "
             f"{R_avg*N/fs*1e3:.1f} ms per point, arg M($\\nu$) cancels in "
             "$\\tau_Q-\\tau_I$")
fig.tight_layout()
fig.savefig("bin_dither.png", dpi=150)
print("figure saved: bin_dither.png")
