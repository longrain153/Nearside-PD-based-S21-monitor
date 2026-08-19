"""Quick simulation: pilot-pair (dither) calibration of a coherent TX
through a 100 MHz PD and a ~100 MS/s ADC, with live data as interference.

Per grid frequency f_k, three pilot configurations:
  (a) I-pair : tones at (f_k, f_k+D) on the I drive
  (b) Q-pair : same tones on the Q drive
  (c) cross  : tone f_k on I, tone f_k+D on Q
Each record: 16QAM data + pilots -> TX -> square law -> 100 MHz PD ->
97.7 MS/s ADC (30 dB SNR). Extraction = one DFT bin at the beat D, with
(i) decision-aided cancellation: the DSP knows its own data, so the
    predicted data-only intensity is subtracted from z, and
(ii) coherent averaging over R records (pilot phase deterministic,
     residual data terms random).

Constant-free recoveries:
  |B_a| ~ |hI(f)|^2, |B_b| ~ (gQ|hQ(f)|)^2      -> magnitude responses
  sqrt(|B_b|/|B_a|)                              -> amp imbalance vs f
  |B_c|/sqrt(|B_a||B_b|) = sin(phi)              -> quadrature error
  arg B_c(f) = theta_Q(f)-theta_I(f)+const       -> relative phase resp.
     slope -> IQ skew (O(1) observable, no stitching)
"""
import os
import sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s21_monitor import make_transmitter, photodetect, lowpass_fir, rrc_fir
from s21_monitor.filters import causal_conv

t_start = time.time()
fs = 200e9
N = 1 << 18                # 1.31 us per record
D_adc = 2048               # ADC at 97.66 MS/s
Nd = N // D_adc            # 128 ADC samples per record
bin_hz = fs / N            # 762.9 kHz
m_beat = 52                # beat at 39.67 MHz (< ADC Nyquist 48.8 MHz)
Dbeat = m_beat * bin_hz
a_pil = 0.2                # -17 dB per tone vs unit data power
snr_db = 30
R_avg = 16                 # coherent averages per config point
R_cross = 96               # extra averaging for the weak cross line (~sin phi)

tx = make_transmitter(branch_taps=41, bw_i=0.55, bw_q=0.50,
                      gain_q=10**(-0.8/20), phase_error_deg=5.0,
                      skew_samples=0.6, delay_taps=15)
M_true = lowpass_fir(8001, 100e6 / (fs / 2))

def Hf(h, f):
    return np.exp(-2j * np.pi * np.outer(np.atleast_1d(f), np.arange(len(h))) / fs) @ h

hI_t = tx.L[0, 0]
hQg_t = tx.L[1, 1] / np.cos(np.deg2rad(5.0))   # gQ*hQ

f_grid = np.round(np.arange(2e9, 50.1e9, 4e9) / bin_hz).astype(int) * bin_hz
f2_grid = f_grid + Dbeat
K = len(f_grid)

# --- precompute R shared data realizations and their predictions ----------
rrc = rrc_fir(65, 0.1, 2)
levels = np.array([-3.0, -1.0, 1.0, 3.0])
rng = np.random.default_rng(1)
t_idx = np.arange(N)

y_data, z_pred = [], []
for _ in range(R_cross):
    xs = []
    for _ in range(2):
        up = np.zeros(N); up[::2] = rng.choice(levels, N // 2)
        x = causal_conv(rrc, up); xs.append(x / np.std(x))
    yI, yQ = tx.apply(xs[0], xs[1])
    y_data.append((yI, yQ))
    # decision-aided prediction of the data-only intensity (known data)
    z_pred.append(photodetect(yI, yQ, M_true, decimate=D_adc))

def measure(fI_tones, fQ_tones, n_avg):
    """Averaged beat-bin readout for one pilot configuration.

    Returns the coherently averaged bin, its noise-debiased magnitude
    (noise power estimated from neighboring bins of the averaged
    spectrum), and the line SNR.
    """
    xpI = sum(a_pil * np.cos(2 * np.pi * f * t_idx / fs) for f in fI_tones) \
        if fI_tones else np.zeros(N)
    xpQ = sum(a_pil * np.cos(2 * np.pi * f * t_idx / fs) for f in fQ_tones) \
        if fQ_tones else np.zeros(N)
    ypI, ypQ = tx.apply(xpI, xpQ)
    spec_acc = np.zeros(Nd, complex)
    for r in range(n_avg):
        yI = y_data[r][0] + ypI
        yQ = y_data[r][1] + ypQ
        z = photodetect(yI, yQ, M_true, snr_db=snr_db, rng=rng, decimate=D_adc)
        resid = z - z_pred[r]                      # decision-aided cancel
        spec_acc += np.fft.fft(resid - resid.mean())
    spec = spec_acc / n_avg
    nb = np.r_[spec[m_beat - 6:m_beat - 1], spec[m_beat + 2:m_beat + 7]]
    noise_p = np.mean(np.abs(nb) ** 2)
    mag = np.sqrt(max(np.abs(spec[m_beat]) ** 2 - noise_p, 1e-30))
    snr = np.abs(spec[m_beat]) / max(np.sqrt(noise_p), 1e-30)
    return spec[m_beat], mag, snr

B_a = np.zeros(K, complex); B_b = np.zeros(K, complex); B_c = np.zeros(K, complex)
A_a = np.zeros(K); A_b = np.zeros(K); A_c = np.zeros(K)
snr_line = np.zeros((3, K))
for k, (f1, f2) in enumerate(zip(f_grid, f2_grid)):
    B_a[k], A_a[k], snr_line[0, k] = measure([f1, f2], [], R_avg)
    B_b[k], A_b[k], snr_line[1, k] = measure([], [f1, f2], R_avg)
    B_c[k], A_c[k], snr_line[2, k] = measure([f1], [f2], R_cross)

# ---------------- recovered quantities ------------------------------------
fghz = f_grid / 1e9
magI_n = A_a ** 0.5; magI_n /= magI_n[0]
magQ_n = A_b ** 0.5; magQ_n /= magQ_n[0]
imb_db = 20 * np.log10(np.sqrt(A_b / A_a))
sin_phi = A_c / np.sqrt(A_a * A_b)
w_phi = snr_line[2] ** 2
phi_est = np.rad2deg(np.arcsin(np.average(sin_phi, weights=w_phi)))
psi_c = np.unwrap(np.angle(B_c))
slope, _ = np.polyfit(f_grid, psi_c, 1, w=np.abs(B_c))
skew_est_ps = slope / (2 * np.pi) * 1e12

# ---------------- truth ----------------------------------------------------
HI1 = Hf(hI_t, f_grid); HI2 = Hf(hI_t, f2_grid)
HQ1 = Hf(hQg_t, f_grid); HQ2 = Hf(hQg_t, f2_grid)
magI_true = np.sqrt(np.abs(HI1 * HI2)); magI_true /= magI_true[0]
magQ_true = np.sqrt(np.abs(HQ1 * HQ2)); magQ_true /= magQ_true[0]
imb_true = 20 * np.log10(np.sqrt(np.abs(HQ1 * HQ2) / np.abs(HI1 * HI2)))
psi_true = np.unwrap(np.angle(HQ2) - np.angle(HI1))
slope_t, _ = np.polyfit(f_grid, psi_true, 1)
skew_true_ps = slope_t / (2 * np.pi) * 1e12

sgn = -1.0 if skew_est_ps * skew_true_ps < 0 else 1.0   # DFT sign convention
skew_est_ps *= sgn

print(f"\n{K} points x 3 configs x {R_avg} averages, {time.time()-t_start:.0f}s")
print("pilot line SNR after decision-aided cancellation + averaging: "
      f"{snr_line.min():.0f}x .. {snr_line.max():.0f}x (median {np.median(snr_line):.0f}x)")
print(f"{'metric':30s} {'true':>10s} {'estimated':>10s}")
print(f"{'IQ amp imbalance @2GHz (dB)':30s} {imb_true[0]:10.3f} {imb_db[0]:10.3f}")
print(f"{'quadrature error (deg)':30s} {5.0:10.3f} {phi_est:10.3f}")
print(f"{'IQ skew (ps)':30s} {abs(skew_true_ps):10.3f} {abs(skew_est_ps):10.3f}")
print(f"max | |H_I| err |: {np.max(np.abs(20*np.log10(magI_n/magI_true))):.3f} dB")
print(f"max | |H_Q| err |: {np.max(np.abs(20*np.log10(magQ_n/magQ_true))):.3f} dB")
print(f"max |imbalance err|: {np.max(np.abs(imb_db-imb_true)):.3f} dB")

# ---------------- figure ----------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].plot(fghz, 20*np.log10(magI_true), "k-", label="true |H_I|")
ax[0, 0].plot(fghz, 20*np.log10(magI_n), "ro", ms=5, label="pilot est |H_I|")
ax[0, 0].plot(fghz, 20*np.log10(magQ_true), "b-", label="true |H_Q|")
ax[0, 0].plot(fghz, 20*np.log10(magQ_n), "g^", ms=5, label="pilot est |H_Q|")
ax[0, 0].set_xlabel("frequency (GHz)"); ax[0, 0].set_ylabel("normalized |S21| (dB)")
ax[0, 0].set_title("S21 magnitude response"); ax[0, 0].legend(fontsize=8)
ax[0, 0].grid(alpha=0.3)

_, ic_t = np.polyfit(f_grid, psi_true, 1)
sl_m, ic_m = np.polyfit(f_grid, sgn * psi_c, 1, w=np.abs(B_c))
p0 = psi_true - ic_t
pm = sgn * psi_c - ic_m
ax[0, 1].plot(fghz, p0, "k-", label="true  $\\theta_Q-\\theta_I$")
ax[0, 1].plot(fghz, pm, "ro", ms=5, label="measured (arg $B_c$)")
ax[0, 1].set_xlabel("frequency (GHz)"); ax[0, 1].set_ylabel("relative phase (rad)")
ax[0, 1].set_title(f"I/Q relative phase response -> skew "
                   f"{abs(skew_est_ps):.2f} ps (true {abs(skew_true_ps):.2f} ps)")
ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

ax[1, 0].plot(fghz, imb_true, "k-", label="true")
ax[1, 0].plot(fghz, imb_db, "ro", ms=5, label="measured")
ax[1, 0].set_xlabel("frequency (GHz)")
ax[1, 0].set_ylabel("$g_Q|h_Q/h_I|$ (dB)")
ax[1, 0].set_title("IQ amplitude imbalance vs frequency")
ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

ax[1, 1].plot(fghz, np.rad2deg(np.arcsin(np.clip(sin_phi, 0, 1))), "ro", ms=5,
              label="per-point estimate")
ax[1, 1].axhline(5.0, color="k", label="true 5$^\\circ$")
ax[1, 1].axhline(phi_est, color="r", ls="--",
                 label=f"mean {phi_est:.2f}$^\\circ$")
ax[1, 1].set_xlabel("frequency (GHz)"); ax[1, 1].set_ylabel("quadrature error (deg)")
ax[1, 1].set_title("Quadrature (IQ phase) error, $|B_c|/\\sqrt{|B_a||B_b|}$")
ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

fig.suptitle("Pilot-dither S21 calibration: 100 MHz PD + 97.7 MS/s ADC, live 16QAM\n"
             f"pilots $-17$ dB/tone, {R_avg}x coherent avg + decision-aided data "
             f"cancellation ({R_avg*N/fs*1e6:.1f} $\\mu$s per point)")
fig.tight_layout()
import os
out = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "pilot_calibration.png"), dpi=150)
print("figure saved: pilot_calibration.png")
