"""Common-mode (true per-branch) phase response via chained same-branch
pilot pairs, through the 100 MHz PD + ~100 MS/s ADC.

The I branch gets a realistic group-delay ripple (allpass, +-20 ps,
20 GHz period). Same-branch pairs at (f_k, f_k+40 MHz) measure
    arg B_a(f_k) = theta_I(f_k+D) - theta_I(f_k) + arg M(D)
i.e. the true group delay tau_I(f_k) plus one constant shared by all
points (-> pure-delay ambiguity only). Integrating tau over f rebuilds
theta_I(f) up to a linear term. Per-step increments are ~2*pi*D*tau, so
line-phase precision must be ~mrad: pilots are raised to -6 dB/tone
(calibration burst) and integration lengthened.
"""
import os
import sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s21_monitor import make_transmitter, photodetect, lowpass_fir, rrc_fir
from s21_monitor.filters import causal_conv, fft_conv_full

t_start = time.time()
fs = 200e9
N = 1 << 18
D_adc = 2048
Nd = N // D_adc
bin_hz = fs / N
m_beat = 52
Dbeat = m_beat * bin_hz
a_pil = 0.5                # -6 dB per tone: dedicated calibration burst
snr_db = 30
R_avg = 800

# ---- transmitter with group-delay ripple on the I branch -------------------
A_tau, P_rip = 20e-12, 20e9
nfft_ap = 8192
f_ap = np.fft.rfftfreq(nfft_ap, 1 / fs)
tau_rip = A_tau * np.sin(2 * np.pi * f_ap / P_rip)
theta_rip = -2 * np.pi * np.concatenate(([0], np.cumsum(
    0.5 * (tau_rip[1:] + tau_rip[:-1]) * np.diff(f_ap))))
ap = np.fft.irfft(np.exp(1j * theta_rip), nfft_ap)
ap = np.roll(ap, 200)[:401] * np.hamming(401)   # causal-ized, truncated

tx = make_transmitter(branch_taps=41, bw_i=0.55, bw_q=0.50,
                      gain_q=10**(-0.8/20), phase_error_deg=5.0,
                      skew_samples=0.6, delay_taps=15)
L = np.zeros((2, 2, tx.L.shape[2] + 400))
for i in range(2):
    for j in range(2):
        L[i, j] = fft_conv_full(tx.L[i, j], ap) if (i, j) == (0, 0) \
            else np.pad(tx.L[i, j], (200, 200))
from s21_monitor.transmitter import WidelyLinearTransmitter
tx = WidelyLinearTransmitter(L)
M_true = lowpass_fir(8001, 100e6 / (fs / 2))

def Hf(h, f):
    return np.exp(-2j * np.pi * np.outer(np.atleast_1d(f), np.arange(len(h))) / fs) @ h

f_grid = np.round(np.arange(2e9, 50.1e9, 2e9) / bin_hz).astype(int) * bin_hz
f2_grid = f_grid + Dbeat
K = len(f_grid)
t_idx = np.arange(N)

# ---- projection vectors (Hann-windowed DFT: kills the deterministic
# leakage of the filter start-up transients, which otherwise floors the
# line SNR at ~28x independent of pilot power and averaging) -----------------
bins = np.arange(m_beat - 9, m_beat + 11)
w_win = np.hanning(Nd)
W = np.zeros((len(bins), N), complex)
for k in range(Nd):
    lo = max(0, k * D_adc - len(M_true) + 1)
    hi = min(N, k * D_adc + 1)
    seg = M_true[k * D_adc - np.arange(lo, hi)]
    W[:, lo:hi] += w_win[k] * np.exp(-2j * np.pi * bins[:, None] * k / Nd) * seg[None, :]
i_beat = 9
# Hann mainlobe spans +-2 bins: estimate noise from bins >=3 away
nb_rows = [i for i in range(len(bins)) if abs(bins[i] - m_beat) > 2]

cfg = []
for f1, f2 in zip(f_grid, f2_grid):
    xp = a_pil * (np.cos(2 * np.pi * f1 * t_idx / fs)
                  + np.cos(2 * np.pi * f2 * t_idx / fs))
    cfg.append(tx.apply(xp, np.zeros(N)))
YpI = np.array([c[0] for c in cfg]); YpQ = np.array([c[1] for c in cfg])
Yp2 = YpI**2 + YpQ**2
print(f"setup {time.time()-t_start:.0f}s", flush=True)

rrc = rrc_fir(65, 0.1, 2)
levels = np.array([-3.0, -1.0, 1.0, 3.0])
rng = np.random.default_rng(2)
def make_data():
    xs = []
    for _ in range(2):
        up = np.zeros(N); up[::2] = rng.choice(levels, N // 2)
        x = causal_conv(rrc, up); xs.append(x / np.std(x))
    return tx.apply(xs[0], xs[1])

ydI, ydQ = make_data()
z_ref = photodetect(ydI + YpI[0], ydQ + YpQ[0], M_true, decimate=D_adc)
noise_std = np.sqrt(np.var(z_ref) * 10.0 ** (-snr_db / 10.0))
E_noise = np.exp(-2j * np.pi * np.outer(bins, np.arange(Nd)) / Nd) * w_win[None, :]

acc = np.zeros((K, len(bins)), complex)
for r in range(R_avg):
    ydI, ydQ = make_data()
    pd_ = 2 * (ydI * YpI + ydQ * YpQ) + Yp2
    b = pd_ @ W.T
    b += (noise_std * rng.standard_normal((K, Nd))) @ E_noise.T
    acc += b
    if r % 200 == 0:
        print(f"r={r} elapsed {time.time()-t_start:.0f}s", flush=True)
spec = acc / R_avg

B = spec[:, i_beat]
snr_line = np.abs(B) / np.sqrt(np.mean(np.abs(spec[:, nb_rows]) ** 2, axis=1))

# ---- recovered group delay & phase -----------------------------------------
hI = tx.L[0, 0]
th1 = np.angle(Hf(hI, f_grid)); th2 = np.angle(Hf(hI, f2_grid))
tau_true = -np.angle(np.exp(1j * (th2 - th1))) / (2 * np.pi * Dbeat)

psi = np.angle(B)
tau_meas = -psi / (2 * np.pi * Dbeat)
# Fixed DFT sign convention (known once per system design); the absolute
# delay is aliased mod 1/Dbeat by arg M(D) anyway (pure-delay ambiguity).
tau_meas -= tau_meas.mean()
tau_true0 = tau_true - tau_true.mean()
if np.dot(tau_meas, tau_true0) < 0:
    tau_meas = -tau_meas
tau_meas += tau_true.mean()                       # align the (unobservable) mean

def integrate_phase(tau):
    th = -2 * np.pi * np.concatenate(([0], np.cumsum(
        0.5 * (tau[1:] + tau[:-1]) * np.diff(f_grid))))
    fit = np.polyfit(f_grid, th, 1)
    return th - np.polyval(fit, f_grid)

th_meas = integrate_phase(tau_meas)
th_true = integrate_phase(tau_true)

print(f"\nK={K} points, R={R_avg} avg ({R_avg*N/fs*1e3:.2f} ms/point), "
      f"pilots -6 dB/tone, total {time.time()-t_start:.0f}s")
print(f"line SNR: median {np.median(snr_line):.0f}x  min {snr_line.min():.0f}x")
print(f"group-delay ripple: true +-{A_tau*1e12:.0f} ps, "
      f"RMS err {np.std(tau_meas-tau_true)*1e12:.2f} ps")
print(f"phase response (linear removed): true ripple RMS "
      f"{np.std(th_true):.4f} rad, reconstruction RMS err "
      f"{np.std(th_meas-th_true):.4f} rad")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
fghz = f_grid / 1e9
ax[0].plot(fghz, (tau_true - tau_true.mean()) * 1e12, "k-", label="true")
ax[0].plot(fghz, (tau_meas - tau_meas.mean()) * 1e12, "ro", ms=4, label="measured")
ax[0].set_xlabel("frequency (GHz)"); ax[0].set_ylabel("group delay ripple (ps)")
ax[0].set_title("I-branch group delay (common mode)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].plot(fghz, th_true, "k-", label="true")
ax[1].plot(fghz, th_meas, "ro", ms=4, label="reconstructed")
ax[1].set_xlabel("frequency (GHz)")
ax[1].set_ylabel("phase, linear term removed (rad)")
ax[1].set_title("I-branch true phase response $\\theta_I(f)$")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.suptitle("Common-mode phase via chained same-branch pilot pairs\n"
             "100 MHz PD + 97.7 MS/s ADC, live 16QAM, "
             f"{R_avg*N/fs*1e3:.1f} ms per point")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "pilot_phase.png"), dpi=150)
print("figure saved: pilot_phase.png")
