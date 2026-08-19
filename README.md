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
  monitor.py        joint COI/OC estimation (LMS / Adam with backpropagation)
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
detects through a 5 GHz PD at 30 dB SNR, and recovers all of them:

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
