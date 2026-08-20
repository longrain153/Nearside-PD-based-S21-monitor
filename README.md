# Nearside-PD-based-S21-monitor

In-service monitoring of a high-baud-rate (100 GBaud-class) coherent
transmitter using only a **low-bandwidth photodetector** (e.g. 5 GHz),
implemented after Zhang et al., *"In-Service Monitoring of a 100
GBaud-Class Coherent Transmitter Using a 5 GHz Photodetector"*.

## Principle

The coherent transmitter is modeled as a 2×2 **widely linear FIR
filter** `L_ij[l]` (`i, j ∈ {I, Q}`) — the *channel of interest* (COI) —
acting on the known digital drive signals `xI[n]`, `xQ[n]`:

```
[yI]   [L_II  L_IQ]   [xI]
[  ] = [          ] ⊗ [  ]
[yQ]   [L_QI  L_QQ]   [xQ]
```

The nearside photodetector is modeled as an ideal **square-law**
detector followed by an unknown low-pass FIR filter `M[m]` — the
*observation channel* (OC), which also absorbs the ADC response:

```
z[n] = M[m] ⊗ (yI[n]² + yQ[n]²)  (+ noise)
```

Through a purely linear observation, `L` and `M` could never be
separated — only their cascade would be identifiable. The **square law
is what makes the problem solvable**: spectral components of the field
separated by less than the OC bandwidth beat against each other, so a
term like `M(Δω)·L(ω₁)·L(ω₁+Δω)·cos(Δωt)` survives the low-pass filter
and carries full-bandwidth information about `L` into the narrow
observation band. The full-bandwidth response of `L_ij[l]` can
therefore be recovered even though `M[m]` has ~10× less bandwidth than
the signal.

## Learning algorithm

The monitor builds the digital model

```
ŷ_i[n] = Σ_j Σ_l L̂_ij[l] x_j[n−l]
ẑ[n]  = Σ_m M̂[m] (ŷI[n−m]² + ŷQ[n−m]²)
e[n]   = (ẑ[n] − z[n])²
```

and minimizes `e[n]` over both `L̂` and `M̂` with gradient descent and
error backpropagation (the chain rule), matching Eqs. (4)–(5) of the
paper:

```
∂E/∂M̂[m]     = 2 Σ_n err[n] · p[n−m]                 (p = ŷI² + ŷQ²)
q[u]          = Σ_n err[n] · M̂[n−u]                   (backprop through OC)
∂E/∂L̂_ij[l] = 4 Σ_u q[u] · ŷ_i[u] · x_j[u−l]
```

`fit_monitor` implements this either as plain batch gradient descent
(`optimizer="gd"`, i.e. the paper's LMS on snapshot data) or with a
full-batch Adam accelerator (`optimizer="adam"`, default). All
convolutions/correlations are FFT-based; the analytic gradients are
verified against finite differences in `tests/test_gradients.py`.

### Band-edge accuracy: two-stage optimization

The MSE objective is badly conditioned across frequency: at frequencies
where the drive spectrum or the transmitter response is weak (band
edges), the gradient of `L̂(ω)` scales with the *square* of the
excitation, so tap-domain descent leaves those bins essentially
unconverged — more data does not help, since the problem is
conditioning, not noise. `fit_monitor(domain="freq")` therefore
parameterizes `L̂` by its DFT bins, where Adam's per-parameter step
adaptation automatically equalizes convergence rates across the band
(the gradient of a real filter is Hermitian, so the taps stay real).

`fit_monitor_hybrid` (used by the demo) combines both: a
frequency-domain stage levels all excited bins, then a tap-domain
polish, warm-started from it, restores full precision in the strongly
excited region and in the derived IQ metrics while barely moving the
band-edge bins. In the demo scenario this reduces the worst-case
|S21| error at 40–50 GHz from ≈1 dB to ≈0.03 dB.

### Inherent ambiguities

The intensity-only observation leaves a few quantities unidentifiable:
a scale exchange (`L → αL`, `M → M/α²`), a common delay exchange
between `L` and `M`, and a static orthogonal rotation of the field
(`(yI, yQ) → Q(yI, yQ)`). The extracted imperfection metrics are built
from the ratio `R(ω) = H_Q(ω)/H_I(ω)` of the complex drive→field
responses, which is **invariant to all of these**. Initializing `L̂` as
an identity (delta) response makes the optimizer converge to the
solution nearest the unrotated one.

## Extracted imperfections

`analyze_iq` reports, from an estimated COI:

- **IQ amplitude imbalance** — `20·log10|R(0)|`
- **Quadrature phase error** — intercept of the fitted `∠R(ω)`
- **IQ skew** — slope of `∠R(ω)` (group-delay difference)

and `branch_responses` returns the full complex S21-style responses
`H_I(ω) = L_II + jL_QI` and `H_Q(ω) = L_QQ − jL_IQ` of both branches.

## Layout

```
s21_monitor/
  filters.py        FIR design (windowed sinc, fractional delay, RRC) + FFT convolution
  transmitter.py    2x2 widely linear transmitter model with imperfection builder
  photodetector.py  square law + low-pass OC + noise
  monitor.py        joint COI/OC estimation (LMS / Adam, tap- or freq-domain, hybrid)
  metrics.py        imperfection extraction from the estimated COI
examples/run_demo.py   end-to-end 100 GBaud / 5 GHz PD demo (figures in examples/output/)
tests/                 gradient finite-difference check + end-to-end convergence test
```

## Quick start

```bash
pip install -r requirements.txt
python3 examples/run_demo.py   # takes a few minutes
python3 -m pytest tests/       # ~40 s
```

The demo drives the transmitter with 100 GBaud RRC-shaped 16QAM at
200 GSa/s, applies a 55/50 GHz branch-bandwidth mismatch, −0.8 dB
amplitude imbalance, 5° quadrature phase error and 3 ps IQ skew,
detects through a 5 GHz PD at 30 dB SNR, and recovers all of them
(branch |S21| to within ≈0.03 dB up to 50 GHz):

| metric              | true  | estimated |
|---------------------|-------|-----------|
| amp imbalance (dB)  | −0.80 | −0.81     |
| phase error (deg)   | 5.00  | 5.01      |
| IQ skew (ps)        | 3.00  | 2.97      |

As expected physically, the COI is recovered over the excitation
bandwidth (the drive signal's ~55 GHz occupied band); outside it the
response is unobservable and the estimate is unconstrained.

Because transmitter imperfections drift slowly, the learning can run
offline on captured snapshots and sleep between updates.

## Dither-based swept calibration (active scheme)

When the TX DSP may inject MHz-rate multiplicative dither on per-branch
spectral slices, calibration becomes a lock-in measurement instead of
blind identification. See `docs/dither_s21_calibration_report.pdf`
(Chinese technical report: theory, simulation validation, and the
0.1 ps skew feasibility analysis) and the examples:

- `examples/swept_dither_calibration.py` - final swept scheme: per-branch
  |S21|, absolute phase response (common quadratic recovered), and skew,
  through a 100 MHz PD + ~195 MS/s in-band-ideal ADC, no mixer
- `examples/skew_burst_90mhz.py` - dedicated skew burst (full-branch AM,
  nu = 90 MHz, I/Q alternating): 0.35 ps at 0.66 ms/branch, ~61 ms/branch
  extrapolates to 0.1 ps
- `examples/dual_freq_skew_scaling.py` - sigma proportional to 1/sqrt(T)
  scaling-law verification (earlier dual-frequency + swap configuration)
- `examples/branch_bin_dither.py` - per-branch single-bin dither at
  phi = 0: magnitudes, imbalance, and skew via same-nu chain difference
- `examples/pilot_dither_calibration.py`, `examples/pilot_phase_response.py`
  - earlier additive pilot-pair variants

### Minimal scheme (final recommended form)

Under the tightest constraints - no DSP data at the monitor (no
decision-aided cancellation), shallow dither only (total <= -25 dB,
EVM-neutral), free-running low-rate ADC - the full calibration cycle
still delivers everything (see report section 5):

- `examples/minimal_sweep.py` - swept slice AM (eps=0.4), no
  cancellation: |S21| 0.010/0.031 dB, common quadratic phase RMS
  0.16/0.23 rad, in ~0.13 s
- `examples/minimal_skew_burst.py` - full-branch AM eps=0.08 skew burst
  worker: sigma = 0.122 +- 0.009 ps at 62.9 ms/branch (48000-record
  block statistics), 1/sqrt(T) verified 0.25-63 ms; 0.1 ps at ~0.19 s
- `examples/precancel_scheme_a.py` - three-arm experiment showing the
  data self-beat is NOT the dominant noise in this configuration
  (exact-H pre-cancellation gains only ~5 dB), which is why the
  no-cancellation minimal scheme works
