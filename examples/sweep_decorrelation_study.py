"""Sweep with DECORRELATED data across frequency points.

The previous sweep reused one data realization for all K points (a
simulation shortcut). The un-cancelled data self-beat residual was then
common-mode across points -> a common tau offset -> a phase error ramp
growing with f. A real swept calibration visits points sequentially in
time, so each point sees different live data and the residual is
independent per point. This run models that: fresh data for every
(point, branch, record).
"""
import sys, time
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from s21_monitor import make_transmitter, lowpass_fir, rrc_fir
from s21_monitor.filters import fft_conv_full

t0 = time.time()
fs, N = 200e9, 1 << 18
D_adc = 1024; Nd = N // D_adc; bin_hz = fs / N
m_beat = 100; nu = m_beat * bin_hz
eps = 0.4; B_slice = 2e9; snr_db = 30
R = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SHARED = len(sys.argv) > 2 and sys.argv[2] == "shared"
CANCEL = len(sys.argv) > 2 and sys.argv[2] == "cancel"

QUAD_RAD, F_EDGE = 6.0, 50e9
f_ap = np.fft.rfftfreq(8192, 1 / fs)
ap = np.fft.irfft(np.exp(1j * (-QUAD_RAD * (f_ap / F_EDGE) ** 2)), 8192)
ap = np.roll(ap, 200)[:401] * np.hamming(401)
tx0 = make_transmitter(branch_taps=41, bw_i=0.55, bw_q=0.50, gain_q=10**(-0.8/20),
                       phase_error_deg=0.0, skew_samples=3.0e-12*fs, delay_taps=15)
Lc = np.zeros((2, 2, tx0.L.shape[2] + 400))
for i in range(2):
    for j in range(2):
        Lc[i, j] = fft_conv_full(tx0.L[i, j], ap)
M_true = lowpass_fir(8001, 100e6 / (fs / 2))
def Hf(h, f):
    return np.exp(-2j*np.pi*np.outer(np.atleast_1d(f), np.arange(len(h)))/fs) @ h
M_nu = Hf(M_true, nu)[0]
freqs = np.fft.rfftfreq(N, 1 / fs)
HII = np.exp(-2j*np.pi*np.outer(freqs, np.arange(Lc.shape[2]))/fs) @ Lc[0, 0]
HQQ = np.exp(-2j*np.pi*np.outer(freqs, np.arange(Lc.shape[2]))/fs) @ Lc[1, 1]
f_grid = np.round(np.arange(2e9, 50.1e9, 2e9) / bin_hz).astype(int) * bin_hz
K = len(f_grid)
masks = [np.abs(freqs - fk) <= B_slice / 2 for fk in f_grid]

bins = np.arange(m_beat - 6, m_beat + 8); w_win = np.hanning(Nd)
W = np.zeros((len(bins), N), complex)
for k in range(Nd):
    lo = max(0, k*D_adc - len(M_true) + 1); hi = min(N, k*D_adc + 1)
    W[:, lo:hi] += w_win[k]*np.exp(-2j*np.pi*bins[:, None]*k/Nd) * M_true[k*D_adc - np.arange(lo, hi)][None, :]
i_beat = 6
E_noise = np.exp(-2j*np.pi*np.outer(bins, np.arange(Nd))/Nd) * w_win[None, :]
Wb = W[i_beat]                      # only the beat row is needed

RRC_h = rrc_fir(65, 0.1, 2)
RRCf = np.fft.rfft(RRC_h, N)
levels = np.array([-3., -1., 1., 3.])
rng = np.random.default_rng(4242)

def data_spectra():
    """One fresh (XI, XQ) pair, RRC-shaped, unit variance (freq domain)."""
    out = []
    for _ in range(2):
        up = np.zeros(N); up[::2] = rng.choice(levels, N // 2)
        X = np.fft.rfft(up) * RRCf
        x = np.fft.irfft(X, N)
        s = np.std(x)
        out.append(X / s)
    return out

# noise scale reference
XI0, XQ0 = data_spectra()
p0 = np.fft.irfft(XI0*HII, N)**2 + np.fft.irfft(XQ0*HQQ, N)**2
from s21_monitor import photodetect
noise_std = np.sqrt(np.var(photodetect(np.fft.irfft(XI0*HII, N),
                                       np.fft.irfft(XQ0*HQQ, N),
                                       M_true, decimate=D_adc)) * 10**(-snr_db/10))

acc = np.zeros(2*K, complex)
for r in range(R):
    if SHARED:
        XI, XQ = data_spectra()
    for k in range(K):
        for b in range(2):
            if not SHARED:
                XI, XQ = data_spectra()      # fresh data per point & branch
            ydI = np.fft.irfft(XI*HII, N); ydQ = np.fft.irfft(XQ*HQQ, N)
            Xbr, Hbr, yd, other = (XI, HII, ydI, ydQ) if b == 0 else (XQ, HQQ, ydQ, ydI)
            SM = Xbr * masks[k]
            DX = eps*0.5*(np.roll(SM, m_beat) + np.roll(SM, -m_beat))
            dy = np.fft.irfft(DX*Hbr, N)
            p = (yd + dy)**2 + other**2
            if CANCEL:                       # TX-side perfect pre-cancellation
                p = p - (yd**2 + other**2)
            v = Wb @ (p - p.mean())
            v += (E_noise @ (noise_std*rng.standard_normal(Nd)))[i_beat]
            acc[2*k + b] += v
    if r % 25 == 0:
        print(f"r={r} elapsed {time.time()-t0:.0f}s", flush=True)
B = acc / R
tau = (np.angle(M_nu) - np.angle(B)) / (2*np.pi*nu)
tau_I, tau_Q = tau[0::2], tau[1::2]

def slice_tau(H):
    RRCp = np.abs(np.fft.rfft(RRC_h, N))**2
    return np.array([-np.angle(np.sum(RRCp*m*np.roll(H, -m_beat)*np.conj(H)))/(2*np.pi*nu)
                     for m in masks])
tIs, tQs = slice_tau(HII), slice_tau(HQQ)
def integ(t):
    return -2*np.pi*np.concatenate(([0], np.cumsum(0.5*(t[1:]+t[:-1])*np.diff(f_grid))))
eI = integ(tau_I) - integ(tIs); eQ = integ(tau_Q) - integ(tQs)

tag = ("SHARED data (old)" if SHARED else
       "DECORRELATED + pre-cancel" if CANCEL else "DECORRELATED (realistic)")
print(f"\n{tag}, R={R}/point ({R*N/fs*1e3:.2f} ms/point), {time.time()-t0:.0f}s")
print(f"tau bias:  I {np.mean(tau_I-tIs)*1e12:+.3f} ps   Q {np.mean(tau_Q-tQs)*1e12:+.3f} ps")
print(f"tau scatter: I {np.std(tau_I-tIs)*1e12:.3f} ps  Q {np.std(tau_Q-tQs)*1e12:.3f} ps")
print(f"phase err: RMS I {np.std(eI):.3f} Q {np.std(eQ):.3f} rad | "
      f"endpoint I {eI[-1]:+.3f} Q {eQ[-1]:+.3f} rad")
np.savez(f"decorr_{'shared' if SHARED else 'cancel' if CANCEL else 'indep'}2.npz",
         tau_I=tau_I, tau_Q=tau_Q, tIs=tIs, tQs=tQs, f_grid=f_grid, eI=eI, eQ=eQ)
