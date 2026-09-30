"""Literal end-to-end demo of the multiplicative-dither monitor.

Pilot (dither) frequency nu = 90 MHz, ADC 200 MSa/s, PD 100 MHz low-pass.
Every 1-us record is simulated sample by sample at 200 GS/s:

    TX DSP   : x_X(t) -> x_X(t) * [1 + eps * cos(2*pi*nu*t)]     (one branch,
               or one contiguous frequency slice of one branch)
    TX chain : y_I = h_I * x_I,  y_Q = h_Q * x_Q   (common quadratic phase
               -6 (f/50 GHz)^2 rad on both, Q delayed by 3 ps = skew)
    PD       : p = y_I^2 + y_Q^2   -> M (100 MHz low-pass) -> ADC 200 MS/s
    monitor  : lock-in  B = sum_n w[n] z[n] exp(-j 2 pi nu n / fs_adc)

The TX NCO runs continuously (90 cycles per 1-us record), so records add
coherently.  Fresh 16QAM data every record, no data reuse anywhere.

Part A (skew burst)   : full-band dither, I and Q records interleaved.
                        skew = -angle(B_Q conj(B_I)) / (2 pi nu)
Part B (phase sweep)  : K contiguous slices of width Delta covering 0..50 GHz,
                        per slice tau_g = (arg M(nu) - arg B) / (2 pi nu),
                        phase accumulated slice by slice (telescoping sum):
                        theta(f_{k+1}) = theta(f_k) - 2 pi Delta tau_g(k)

Usage: python dither_demo_90mhz.py [--eps 0.08] [--t-skew-ms 30]
                                   [--t-slice-ms 3] [--slices 10]
                                   [--eps-slice auto] [--procs 4]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from s21_monitor import make_transmitter, lowpass_fir, rrc_fir
from s21_monitor.filters import causal_conv, fft_conv_full

# ----------------------------------------------------------------------------
# fixed physical setup
# ----------------------------------------------------------------------------
fs = 200e9                 # TX DSP / waveform rate
N = 200_000                # 1 us per record
D_adc = 1000               # 200 GS/s / 1000 = 200 MS/s
fs_adc = fs / D_adc
Nd = N // D_adc            # 200 ADC samples per record
bin_hz = fs / N            # 1 MHz record bin
m_beat = 90                # nu = 90 MHz = 90 cycles per record (NCO continuous)
nu = m_beat * bin_hz
t_rec = np.arange(N) / fs
dither_cos = np.cos(2 * np.pi * nu * t_rec)

SKEW_PS = 3.0
QUAD_RAD, F_EDGE = 6.0, 50e9
snr_db_adc = 30.0

# transmitter: common quadratic phase on both branches + 3 ps skew
from s21_monitor.transmitter import WidelyLinearTransmitter  # noqa: E402
_nap = 8192
_fap = np.fft.rfftfreq(_nap, 1 / fs)
_ap = np.fft.irfft(np.exp(-1j * QUAD_RAD * (_fap / F_EDGE) ** 2), _nap)
_ap = np.roll(_ap, 200)[:401] * np.hamming(401)
_tx0 = make_transmitter(branch_taps=41, bw_i=0.55, bw_q=0.50,
                        gain_q=10 ** (-0.8 / 20), phase_error_deg=0.0,
                        skew_samples=SKEW_PS * 1e-12 * fs, delay_taps=15)
_L = np.zeros((2, 2, _tx0.L.shape[2] + 400))
for _i in range(2):
    for _j in range(2):
        _L[_i, _j] = fft_conv_full(_tx0.L[_i, _j], _ap)
tx = WidelyLinearTransmitter(_L)
hI, hQ = tx.L[0, 0], tx.L[1, 1]
M_true = lowpass_fir(8001, 100e6 / (fs / 2))       # PD + ADC front end

freqs = np.fft.rfftfreq(N, 1 / fs)
_E = lambda h: np.fft.rfft(h, N)  # noqa: E731  (zero-padded, causal taps)
HII, HQQ = _E(hI), _E(hQ)
M_f = _E(M_true)
M_nu = M_f[m_beat]                                  # known: ADC ideal in band

rrc = rrc_fir(65, 0.1, 2)
levels = np.array([-3.0, -1.0, 1.0, 3.0])
w_adc = np.hanning(Nd)
lockin = w_adc * np.exp(-2j * np.pi * m_beat * np.arange(Nd) / Nd)
S_rrc = np.abs(_E(rrc)) ** 2                        # data PSD shape


def make_data(rng):
    out = []
    for _ in range(2):
        up = np.zeros(N)
        up[::2] = rng.choice(levels, N // 2)
        x = causal_conv(rrc, up)
        out.append(x / np.std(x))
    return out


def one_record(rng, branch, mask, eps, noise_std):
    """Simulate one 1-us record; return lock-in output and ADC spectrum."""
    xI, xQ = make_data(rng)
    X = [np.fft.rfft(xI), np.fft.rfft(xQ)]
    H = [HII, HQQ]
    # multiplicative dither on `branch`, restricted to `mask` (None = full band)
    if mask is None:
        xs = xI if branch == 0 else xQ
    else:
        xs = np.fft.irfft(X[branch] * mask, N)
    d = eps * xs * dither_cos
    X[branch] = X[branch] + np.fft.rfft(d)
    yI = np.fft.irfft(X[0] * H[0], N)
    yQ = np.fft.irfft(X[1] * H[1], N)
    p = yI ** 2 + yQ ** 2
    z_full = np.fft.irfft(np.fft.rfft(p) * M_f, N)
    z = z_full[::D_adc] + noise_std * rng.standard_normal(Nd)
    z = z - z.mean()
    B = np.sum(lockin * z)
    spec = np.abs(np.fft.rfft(w_adc * z)) ** 2
    return B, spec


def worker(args):
    seed, start, n_rec, mode, K, masks, eps, noise_std, chunk = args
    rng = np.random.default_rng(seed)
    n_slots = 2 if mode == "skew" else 2 * K
    acc = np.zeros((n_slots, (n_rec + chunk - 1) // chunk), complex)
    cnt = np.zeros_like(acc, dtype=int)
    spec_on = np.zeros(Nd // 2 + 1)
    n_on = 0
    for i in range(n_rec):
        r = start + i
        b = r % 2
        if mode == "skew":
            slot, mask = b, None
        else:
            k = (r // 2) % K
            slot, mask = 2 * k + b, masks[k]
        B, spec = one_record(rng, b, mask, eps, noise_std)
        acc[slot, i // chunk] += B
        cnt[slot, i // chunk] += 1
        if mode == "skew" and b == 0:
            spec_on += spec
            n_on += 1
    return acc, cnt, spec_on, n_on


def run(mode, n_total, K, masks, eps, noise_std, procs, chunk, seed0):
    per = (n_total + procs - 1) // procs
    jobs = [(seed0 + j, j * per, min(per, n_total - j * per), mode, K, masks,
             eps, noise_std, chunk) for j in range(procs)]
    t0 = time.time()
    with Pool(procs) as pool:
        outs = pool.map(worker, jobs)
    acc = np.concatenate([o[0] for o in outs], axis=1)
    cnt = np.concatenate([o[1] for o in outs], axis=1)
    spec = sum(o[2] for o in outs) / max(1, sum(o[3] for o in outs))
    print(f"[{mode}] {n_total} records ({n_total * N / fs * 1e3:.2f} ms) "
          f"in {time.time() - t0:.0f} s", flush=True)
    return acc, cnt, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.08)
    ap.add_argument("--t-skew-ms", type=float, default=30.0)
    ap.add_argument("--t-slice-ms", type=float, default=3.0)
    ap.add_argument("--slices", type=int, default=10)
    ap.add_argument("--eps-slice", type=float, default=None)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="dither_demo_90mhz")
    ap.add_argument("--replot", action="store_true", help="only redraw from <out>.npz")
    a = ap.parse_args()
    if a.replot:
        make_figure(a.out)
        return

    K = a.slices
    Delta = F_EDGE / K
    edges = np.arange(K + 1) * Delta
    masks = [(freqs >= edges[k]) & (freqs < edges[k + 1]) for k in range(K)]
    # same total dither power as full-band eps -> EVM-neutral by construction
    eps_s = a.eps_slice if a.eps_slice else a.eps * np.sqrt(K)

    rng = np.random.default_rng(1)
    xI, xQ = make_data(rng)
    yI = np.fft.irfft(np.fft.rfft(xI) * HII, N)
    yQ = np.fft.irfft(np.fft.rfft(xQ) * HQQ, N)
    z0 = np.fft.irfft(np.fft.rfft(yI ** 2 + yQ ** 2) * M_f, N)[::D_adc]
    noise_std = np.sqrt(np.var(z0) * 10 ** (-snr_db_adc / 10))
    _, spec_off = one_record(rng, 0, None, 0.0, noise_std)
    spec_off = np.zeros_like(spec_off)
    for _ in range(200):
        spec_off += one_record(rng, 0, None, 0.0, noise_std)[1] / 200

    print(f"nu={nu / 1e6:.0f} MHz, ADC {fs_adc / 1e6:.0f} MS/s, {Nd} samples/record, "
          f"eps={a.eps} (full band), eps_slice={eps_s:.3f} x {K} slices of "
          f"{Delta / 1e9:.1f} GHz, dither power {a.eps ** 2 / 2 * 100:.2f}% of data",
          flush=True)

    # ---------------- Part A: skew burst -----------------------------------
    n_skew = int(round(2 * a.t_skew_ms * 1e-3 * fs / N))
    chunk = 500
    accA, cntA, spec_on = run("skew", n_skew, K, masks, a.eps, noise_std,
                              a.procs, chunk, seed0=100)
    BI_c, BQ_c = np.cumsum(accA[0]), np.cumsum(accA[1])
    nI_c = np.cumsum(cntA[0])
    t_axis = nI_c * N / fs                            # seconds per branch
    skew_run = -np.angle(BQ_c * np.conj(BI_c)) / (2 * np.pi * nu)
    skew_A = skew_run[-1]

    # truth of the full-band line phase (weighted mean group delay)
    def line(Hb, m):
        return np.sum(S_rrc * m * np.roll(Hb, -m_beat) * np.conj(Hb))
    full = np.ones_like(freqs, bool)
    tau_t = [-np.angle(line(H, full)) / (2 * np.pi * nu) for H in (HII, HQQ)]
    skew_true = tau_t[1] - tau_t[0]

    # ---------------- Part B: contiguous slice sweep ----------------------
    n_sweep = int(round(2 * K * a.t_slice_ms * 1e-3 * fs / N))
    accB, cntB, _ = run("sweep", n_sweep, K, masks, eps_s, noise_std,
                        a.procs, chunk=10 ** 9, seed0=900)
    B = accB[:, 0] / np.maximum(cntB[:, 0], 1)
    B_I, B_Q = B[0::2], B[1::2]
    tau_I = (np.angle(M_nu) - np.angle(B_I)) / (2 * np.pi * nu)
    tau_Q = (np.angle(M_nu) - np.angle(B_Q)) / (2 * np.pi * nu)
    # telescoping accumulation on the slice edges
    th_I = -2 * np.pi * np.concatenate(([0.0], np.cumsum(tau_I * Delta)))
    th_Q = -2 * np.pi * np.concatenate(([0.0], np.cumsum(tau_Q * Delta)))
    # per-slice truth: same weighted line, and exact phase on the edges
    TI = np.array([line(HII, m) for m in masks])
    TQ = np.array([line(HQQ, m) for m in masks])
    tauI_t = -np.angle(TI) / (2 * np.pi * nu)
    tauQ_t = -np.angle(TQ) / (2 * np.pi * nu)
    idx = [int(round(e / bin_hz)) for e in edges]
    thI_t = np.unwrap(np.angle(HII))[idx]
    thQ_t = np.unwrap(np.angle(HQQ))[idx]
    thI_t -= thI_t[0]
    thQ_t -= thQ_t[0]
    # remove the SAME nominal bulk delay (true low-f delay) from all curves
    tau_ref = tauI_t[0]
    lin = 2 * np.pi * edges * tau_ref
    skew_B = np.mean(tau_Q - tau_I)
    skew_B_sem = np.std(tau_Q - tau_I, ddof=1) / np.sqrt(K)
    # smooth model: quadratic phase -> linear group delay, 2-parameter fit
    fc = edges[:-1] + Delta / 2
    pI = np.polyfit(fc, tau_I, 1)
    pQ = np.polyfit(fc, tau_Q, 1)
    fit_th = lambda p: -2 * np.pi * (p[0] * edges ** 2 / 2 + p[1] * edges)  # noqa: E731

    print("\n=== Part A: full-band skew burst ===")
    print(f"  {a.t_skew_ms:.1f} ms per branch, eps={a.eps}")
    print(f"  skew est {skew_A * 1e12:+.3f} ps   (true {skew_true * 1e12:+.3f} ps, "
          f"nominal {SKEW_PS:.1f} ps)")
    print("=== Part B: contiguous slice sweep ===")
    print(f"  {K} slices x {Delta / 1e9:.1f} GHz, {a.t_slice_ms:.1f} ms per slice per "
          f"branch, eps_slice={eps_s:.3f}")
    print(f"  tau_g RMS err: I {np.std(tau_I - tauI_t) * 1e12:.2f} ps, "
          f"Q {np.std(tau_Q - tauQ_t) * 1e12:.2f} ps")
    print(f"  accumulated phase err at 50 GHz: I {th_I[-1] - thI_t[-1]:+.3f} rad, "
          f"Q {th_Q[-1] - thQ_t[-1]:+.3f} rad;  RMS over edges "
          f"I {np.std(th_I - thI_t):.3f}, Q {np.std(th_Q - thQ_t):.3f} rad")
    print(f"  linear-fit phase err at 50 GHz: I {fit_th(pI)[-1] - thI_t[-1]:+.3f} rad, "
          f"Q {fit_th(pQ)[-1] - thQ_t[-1]:+.3f} rad")
    print(f"  skew from sweep {skew_B * 1e12:+.3f} +- {skew_B_sem * 1e12:.3f} ps")

    np.savez(a.out + ".npz", accA=accA, cntA=cntA, spec_on=spec_on, spec_off=spec_off,
             B_I=B_I, B_Q=B_Q, tau_I=tau_I, tau_Q=tau_Q, tauI_t=tauI_t, tauQ_t=tauQ_t,
             th_I=th_I, th_Q=th_Q, thI_t=thI_t, thQ_t=thQ_t, edges=edges,
             skew_run=skew_run, t_axis=t_axis, skew_true=skew_true,
             eps=a.eps, eps_s=eps_s, nu=nu, t_skew_ms=a.t_skew_ms,
             t_slice_ms=a.t_slice_ms, skew_A=skew_A, skew_B=skew_B,
             skew_B_sem=skew_B_sem, tau_ref=tau_ref, pI=pI, pQ=pQ, K=K, Delta=Delta)
    make_figure(a.out)


def make_figure(out):
    d = dict(np.load(out + ".npz"))
    for k, v in d.items():
        if v.ndim == 0:
            d[k] = v.item()
    (spec_on, spec_off, tau_I, tau_Q, tauI_t, tauQ_t, th_I, th_Q, thI_t, thQ_t,
     edges, skew_run, t_axis, skew_true, eps_s, skew_A, skew_B, skew_B_sem,
     tau_ref, pI, pQ, K, Delta) = (d[k] for k in (
        "spec_on", "spec_off", "tau_I", "tau_Q", "tauI_t", "tauQ_t", "th_I", "th_Q",
        "thI_t", "thQ_t", "edges", "skew_run", "t_axis", "skew_true", "eps_s", "skew_A",
        "skew_B", "skew_B_sem", "tau_ref", "pI", "pQ", "K", "Delta"))

    class A:  # noqa: D401 - small namespace for the plot code below
        pass
    a = A(); a.eps = d["eps"]; a.t_slice_ms = d["t_slice_ms"]; a.out = out
    lin = 2 * np.pi * edges * tau_ref
    fc = edges[:-1] + Delta / 2
    fit_th = lambda p: -2 * np.pi * (p[0] * edges ** 2 / 2 + p[1] * edges)  # noqa: E731

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    f_adc = np.fft.rfftfreq(Nd, 1 / fs_adc) / 1e6
    ax[0, 0].semilogy(f_adc, spec_on / spec_on.max(), "r-", label="dither ON (I), 1 record avg")
    ax[0, 0].semilogy(f_adc, spec_off / spec_on.max(), "k--", alpha=0.6, label="dither OFF")
    ax[0, 0].axvline(nu / 1e6, color="b", ls=":", label="$\\nu$ = 90 MHz")
    ax[0, 0].set_xlabel("ADC frequency (MHz)"); ax[0, 0].set_ylabel("PSD (norm.)")
    ax[0, 0].set_title("What the 200 MS/s ADC sees (Hann, 1 us records)")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    tt = t_axis * 1e3
    ax[0, 1].plot(tt, skew_run * 1e12, "r-", label="running estimate")
    ax[0, 1].axhline(skew_true * 1e12, color="k", ls="--", label=f"true {skew_true * 1e12:.2f} ps")
    sig = np.sqrt(2) / (2 * np.pi * nu * a.eps * np.sqrt(55e9 * t_axis)) * 1e12
    ax[0, 1].fill_between(tt, (skew_true) * 1e12 - sig, (skew_true) * 1e12 + sig,
                          color="b", alpha=0.15, label="design law $\\pm\\sigma$")
    ax[0, 1].set_xlabel("integration time per branch (ms)"); ax[0, 1].set_ylabel("skew (ps)")
    ax[0, 1].set_ylim(skew_true * 1e12 - 6, skew_true * 1e12 + 6)
    ax[0, 1].set_title(f"Part A: skew burst, $\\epsilon$={a.eps}, I/Q interleaved")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

    fg = freqs / 1e9
    ax[0, 2].plot(fg, np.unwrap(np.angle(HII)) + 2 * np.pi * freqs * tau_ref, "k-",
                  label="true $\\theta_I(f)$ (bulk delay removed)")
    ax[0, 2].plot(fg, np.unwrap(np.angle(HQQ)) + 2 * np.pi * freqs * tau_ref, "b-",
                  label="true $\\theta_Q(f)$")
    for e in edges:
        ax[0, 2].axvline(e / 1e9, color="gray", lw=0.5, alpha=0.5)
    ax[0, 2].set_xlim(0, 55); ax[0, 2].set_ylim(-8, 1)
    ax[0, 2].set_xlabel("frequency (GHz)"); ax[0, 2].set_ylabel("phase (rad)")
    ax[0, 2].set_title(f"Scenario: quadratic $-6(f/50G)^2$, {K} contiguous slices")
    ax[0, 2].legend(fontsize=8); ax[0, 2].grid(alpha=0.3)

    fcg = fc / 1e9
    ax[1, 0].plot(fcg, (tauI_t - tau_ref) * 1e12, "k-", label="true $\\tau_{g,I}$")
    ax[1, 0].plot(fcg, (tau_I - tau_ref) * 1e12, "ro", label="measured $\\tau_{g,I}$")
    ax[1, 0].plot(fcg, (tauQ_t - tau_ref) * 1e12, "b-", label="true $\\tau_{g,Q}$")
    ax[1, 0].plot(fcg, (tau_Q - tau_ref) * 1e12, "g^", label="measured $\\tau_{g,Q}$")
    ax[1, 0].set_xlabel("slice centre (GHz)"); ax[1, 0].set_ylabel("group delay (ps)")
    ax[1, 0].set_title(f"Part B: per-slice group delay from arg B(k), "
                       f"$\\epsilon_s$={eps_s:.2f}, {a.t_slice_ms:.0f} ms/slice")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    eg = edges / 1e9
    ax[1, 1].plot(eg, thI_t + lin, "k-", label="true $\\theta_I$")
    ax[1, 1].plot(eg, th_I + lin, "ro-", ms=5, label="accumulated (telescoping)")
    ax[1, 1].plot(eg, fit_th(pI) + lin, "r:", label="2-param fit")
    ax[1, 1].plot(eg, thQ_t + lin, "b-", label="true $\\theta_Q$")
    ax[1, 1].plot(eg, th_Q + lin, "g^-", ms=5, label="accumulated")
    ax[1, 1].plot(eg, fit_th(pQ) + lin, "g:", label="2-param fit")
    ax[1, 1].set_xlabel("frequency (GHz)"); ax[1, 1].set_ylabel("phase (rad)")
    ax[1, 1].set_title("Phase accumulated slice by slice (bulk delay removed)")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

    ax[1, 2].plot(eg, thQ_t - thI_t, "k-", label="true $\\theta_Q-\\theta_I$")
    ax[1, 2].plot(eg, th_Q - th_I, "ro-", ms=5, label="measured")
    ax[1, 2].plot(eg, -2 * np.pi * edges * skew_A, "b:", label=f"Part A skew {skew_A * 1e12:.2f} ps")
    ax[1, 2].set_xlabel("frequency (GHz)"); ax[1, 2].set_ylabel("$\\theta_Q-\\theta_I$ (rad)")
    ax[1, 2].set_title(f"I/Q differential phase; sweep skew {skew_B * 1e12:.2f}$\\pm$"
                       f"{skew_B_sem * 1e12:.2f} ps")
    ax[1, 2].legend(fontsize=8); ax[1, 2].grid(alpha=0.3)

    fig.suptitle(f"Multiplicative dither at $\\nu$=90 MHz, ADC 200 MS/s, PD 100 MHz, "
                 f"live 16QAM 100 GBaud, fresh data every record, no cancellation",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=140)
    print("figure:", a.out + ".png")


if __name__ == "__main__":
    main()
