"""Final swept calibration, simplified scenario per spec:
- fixed (frequency-flat) group delay: I/Q skew = 20 ps, no ripple
- phase response reported directly in radians (absolute, M known)
- skew reported as one frequency-independent constant

Swept per-branch slice-AM dither, ADC ideal in-band (M known), no mixer,
nu = 76 MHz, phi = 0, live 16QAM.
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
D_adc = 1024                   # ADC at 195.3 MS/s
Nd = N // D_adc
bin_hz = fs / N
m_beat = 100                   # nu = 76.29 MHz
nu = m_beat * bin_hz
eps = 0.7
B_slice = 2e9
snr_db = 30
R_avg = 600

SKEW_PS = 3.0                  # I/Q differ by 3 ps, frequency-flat
QUAD_RAD, F_EDGE = 6.0, 50e9   # common quadratic phase, 6 rad at 50 GHz
from s21_monitor.filters import fft_conv_full
from s21_monitor.transmitter import WidelyLinearTransmitter
nfft_ap = 8192
f_ap = np.fft.rfftfreq(nfft_ap, 1 / fs)
th_add = -QUAD_RAD * (f_ap / F_EDGE) ** 2
ap = np.fft.irfft(np.exp(1j * th_add), nfft_ap)
ap = np.roll(ap, 200)[:401] * np.hamming(401)
tx0 = make_transmitter(branch_taps=41, bw_i=0.55, bw_q=0.50,
                       gain_q=10**(-0.8/20), phase_error_deg=0.0,
                       skew_samples=SKEW_PS * 1e-12 * fs,  # 0.6 samples
                       delay_taps=15)
L = np.zeros((2, 2, tx0.L.shape[2] + 400))
for i in range(2):
    for j in range(2):
        L[i, j] = fft_conv_full(tx0.L[i, j], ap)   # SAME quadratic on I and Q
tx = WidelyLinearTransmitter(L)
hI_t = tx.L[0, 0]; hQ_t = tx.L[1, 1]
M_true = lowpass_fir(8001, 100e6 / (fs / 2))

def Hf(h, f):
    return np.exp(-2j * np.pi * np.outer(np.atleast_1d(f), np.arange(len(h))) / fs) @ h

M_nu = Hf(M_true, nu)[0]       # known observation response
freqs = np.fft.rfftfreq(N, 1 / fs)
HII = np.exp(-2j * np.pi * np.outer(freqs, np.arange(len(hI_t))) / fs) @ hI_t
HQQ = np.exp(-2j * np.pi * np.outer(freqs, np.arange(len(hQ_t))) / fs) @ hQ_t

f_grid = np.round(np.arange(2e9, 50.1e9, 2e9) / bin_hz).astype(int) * bin_hz
K = len(f_grid)
masks = [np.abs(freqs - fk) <= B_slice / 2 for fk in f_grid]

bins = np.arange(m_beat - 6, m_beat + 8)
w_win = np.hanning(Nd)
W = np.zeros((len(bins), N), complex)
for k in range(Nd):
    lo = max(0, k * D_adc - len(M_true) + 1)
    hi = min(N, k * D_adc + 1)
    seg = M_true[k * D_adc - np.arange(lo, hi)]
    W[:, lo:hi] += w_win[k] * np.exp(-2j * np.pi * bins[:, None] * k / Nd) * seg[None, :]
i_beat = 6
nb_rows = [i for i in range(len(bins)) if abs(bins[i] - m_beat) > 2]
E_noise = np.exp(-2j * np.pi * np.outer(bins, np.arange(Nd)) / Nd) * w_win[None, :]

rrc = rrc_fir(65, 0.1, 2)
levels = np.array([-3.0, -1.0, 1.0, 3.0])
rng = np.random.default_rng(29)

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

acc = np.zeros((2 * K, len(bins)), complex)
for r in range(R_avg):
    xI, xQ = make_data()
    XI = np.fft.rfft(xI); XQ = np.fft.rfft(xQ)
    ydI = np.fft.irfft(XI * HII, N)
    ydQ = np.fft.irfft(XQ * HQQ, N)
    for k in range(K):
        for b, (Xbr, Hbr, yd) in enumerate(((XI, HII, ydI), (XQ, HQQ, ydQ))):
            SM = Xbr * masks[k]
            DX = eps * 0.5 * (np.roll(SM, m_beat) + np.roll(SM, -m_beat))
            dy = np.fft.irfft(DX * Hbr, N)
            p_diff = 2 * yd * dy + dy * dy
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

# ---- extraction ------------------------------------------------------------
RRC = np.abs(np.exp(-2j * np.pi * np.outer(freqs, np.arange(len(rrc))) / fs) @ rrc) ** 2
p_slice = np.array([np.sum(RRC * m) for m in masks])
magI = np.sqrt(A_I / (p_slice * abs(M_nu))); magI /= magI[0]
magQ = np.sqrt(A_Q / (p_slice * abs(M_nu))); magQ /= magQ[0]
tau_I = (np.angle(M_nu) - np.angle(B_I)) / (2 * np.pi * nu)   # absolute
tau_Q = (np.angle(M_nu) - np.angle(B_Q)) / (2 * np.pi * nu)

# phase response in rad: theta(f) = -2*pi*cumtrapz(tau), anchored by
# linear extrapolation to DC (theta(0) = 0 by definition of real filters)
def phase_curve(tau):
    th = -2 * np.pi * np.concatenate(([0], np.cumsum(
        0.5 * (tau[1:] + tau[:-1]) * np.diff(f_grid))))
    return th - 2 * np.pi * f_grid[0] * tau[0]   # anchor: DC..f0 at tau(f0)
th_I = phase_curve(tau_I)
th_Q = phase_curve(tau_Q)

# skew: frequency-flat by spec -> plain mean of Delta tau
d_tau = tau_Q - tau_I
skew_est = np.mean(d_tau)
skew_sem = np.std(d_tau, ddof=1) / np.sqrt(K)

# ---- truth ------------------------------------------------------------------
def truth_line(Hbr):
    return np.array([np.sum(RRC * m * np.roll(Hbr, -m_beat) * np.conj(Hbr))
                     for m in masks])
TI = truth_line(HII); TQ = truth_line(HQQ)
magI_t = np.sqrt(np.abs(TI) / p_slice); magI_t /= magI_t[0]
magQ_t = np.sqrt(np.abs(TQ) / p_slice); magQ_t /= magQ_t[0]
# true phase on the grid, unwrapped on the fine rfft grid then sampled
idx = [np.argmin(np.abs(freqs - fk)) for fk in f_grid]
th_I_t = np.unwrap(np.angle(HII))[idx]
th_Q_t = np.unwrap(np.angle(HQQ))[idx]
# reference linear phase (nominal bulk delay, known by design) so the
# quadratic component is visible; identical reference for est and truth
tau_ref = tauI_t[0] if False else None
tauI_t = -np.angle(TI) / (2 * np.pi * nu)
tauQ_t = -np.angle(TQ) / (2 * np.pi * nu)
skew_t = np.mean(tauQ_t - tauI_t)

if np.dot(tau_I, tauI_t) < 0:
    raise RuntimeError("sign convention flipped")

# subtract the same nominal linear phase (design bulk delay at low f)
tau_ref = tauI_t[0]
th_I_r = th_I + 2 * np.pi * f_grid * tau_ref
th_Q_r = th_Q + 2 * np.pi * f_grid * tau_ref
th_I_tr = th_I_t + 2 * np.pi * f_grid * tau_ref
th_Q_tr = th_Q_t + 2 * np.pi * f_grid * tau_ref

print(f"\nswept final: skew {SKEW_PS:.0f} ps flat, K={K} pts, {R_avg} avg "
      f"({R_avg*N/fs*1e3:.2f} ms/point), total {time.time()-t_start:.0f}s")
print(f"line SNR median {np.median(snr_line):.0f}x")
print(f"|H_I| max err {np.max(np.abs(20*np.log10(magI/magI_t))):.3f} dB, "
      f"|H_Q| max err {np.max(np.abs(20*np.log10(magQ/magQ_t))):.3f} dB")
print(f"phase response RMS err: I {np.std(th_I-th_I_t):.4f} rad, "
      f"Q {np.std(th_Q-th_Q_t):.4f} rad "
      f"(max |err| {max(np.max(np.abs(th_I-th_I_t)), np.max(np.abs(th_Q-th_Q_t))):.4f} rad)")
print(f"skew: est {skew_est*1e12:+.3f} +- {skew_sem*1e12:.3f} ps "
      f"(true {skew_t*1e12:+.3f} ps)")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
fghz = f_grid / 1e9
ax[0].plot(fghz, 20*np.log10(magI_t), "k-", label="true |H_I|")
ax[0].plot(fghz, 20*np.log10(magI), "ro", ms=4, label="est |H_I|")
ax[0].plot(fghz, 20*np.log10(magQ_t), "b-", label="true |H_Q|")
ax[0].plot(fghz, 20*np.log10(magQ), "g^", ms=4, label="est |H_Q|")
ax[0].set_xlabel("frequency (GHz)"); ax[0].set_ylabel("normalized |S21| (dB)")
ax[0].set_title("Magnitude response"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].plot(fghz, th_I_tr, "k-", label="true $\\theta_I$")
ax[1].plot(fghz, th_I_r, "ro", ms=4, label="measured $\\theta_I$")
ax[1].plot(fghz, th_Q_tr, "b-", label="true $\\theta_Q$")
ax[1].plot(fghz, th_Q_r, "g^", ms=4, label="measured $\\theta_Q$")
ax[1].set_xlabel("frequency (GHz)"); ax[1].set_ylabel("phase (rad)")
ax[1].set_title("Phase response (rad), common quadratic $-6(f/50G)^2$")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

dth = th_Q - th_I; dth_t = th_Q_t - th_I_t
ax[2].plot(fghz, dth_t, "k-", label="true $\\theta_Q-\\theta_I$")
ax[2].plot(fghz, dth, "ro", ms=4, label="measured")
ax[2].set_xlabel("frequency (GHz)")
ax[2].set_ylabel("$\\theta_Q-\\theta_I$ (rad)")
ax[2].set_title(f"Skew (flat): {skew_est*1e12:.2f}$\\pm${skew_sem*1e12:.2f} ps "
                f"(true {skew_t*1e12:.2f} ps)")
ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

fig.suptitle("Swept dither calibration: common quadratic phase (6 rad) "
             "+ 3 ps skew\nADC ideal in-band, no mixer, "
             f"$\\nu$={nu/1e6:.0f} MHz, $\\varphi$=0, live 16QAM, "
             f"{R_avg*N/fs*1e3:.1f} ms/point")
fig.tight_layout()
fig.savefig("swept_final2.png", dpi=150)
np.savez("swept_final2.npz", B_I=B_I, B_Q=B_Q, tau_I=tau_I, tau_Q=tau_Q,
         th_I=th_I, th_Q=th_Q, f_grid=f_grid)
print("figure saved: swept_final.png")
