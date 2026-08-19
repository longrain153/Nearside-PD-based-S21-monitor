"""Joint estimation of the transmitter filters L_ij[l] (COI) and the
photodetector/ADC filter M[m] (OC) from (xI, xQ, z).

Digital model (all sequences causal, outputs truncated to N = len(xI)):

    yhat_i[n] = sum_j sum_l Lhat[i,j,l] x_j[n - l]          (i in {I, Q})
    p[n]      = yhat_I[n]^2 + yhat_Q[n]^2
    zhat[n]   = sum_m Mhat[m] p[n - m]
    e[n]      = (zhat[n] - z[n])^2

Gradients (error backpropagation / chain rule), matching Eqs. (4)-(5)
of Zhang et al., "In-Service Monitoring of a 100 GBaud-Class Coherent
Transmitter Using a 5 GHz Photodetector":

    dE/dMhat[m]     = 2 sum_n err[n] p[n - m]
    q[u]            = sum_n err[n] Mhat[n - u]         (backprop through OC)
    dE/dLhat[i,j,l] = 4 sum_u q[u] yhat_i[u] x_j[u - l]

The square law creates a scale ambiguity (L -> a*L, M -> M/a^2) and a
common delay/orthogonal-rotation ambiguity between L and M; relative
quantities (normalized responses, IQ imbalance, phase error, skew) are
unaffected. Initializing Lhat near an identity (delta) response makes
gradient descent converge to the solution closest to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .filters import causal_conv, fft_conv_full


@dataclass
class MonitorResult:
    L: np.ndarray  # estimated COI, shape (2, 2, L_len)
    M: np.ndarray  # estimated OC, shape (M_len,)
    loss: list = field(default_factory=list)  # MSE per iteration
    nmse_db: float = np.nan  # final fit NMSE of zhat vs z


def _forward(L, M, xI, xQ, N):
    x = (xI, xQ)
    y = np.empty((2, N))
    for i in range(2):
        y[i] = sum(causal_conv(L[i, j], x[j]) for j in range(2))
    p = y[0] ** 2 + y[1] ** 2
    zhat = causal_conv(M, p)
    return y, p, zhat


def _gradients(L, M, xI, xQ, err, y, p, N):
    M_len = len(M)
    L_len = L.shape[2]
    grad_M = 2.0 * fft_conv_full(err, p[::-1])[N - 1 : N - 1 + M_len]
    # Backpropagate the error through the OC: q[u] = sum_n err[n] M[n-u]
    q = fft_conv_full(err, M[::-1])[M_len - 1 : M_len - 1 + N]
    grad_L = np.empty_like(L)
    x = (xI, xQ)
    for i in range(2):
        s = q * y[i]
        for j in range(2):
            grad_L[i, j] = 4.0 * fft_conv_full(s, x[j][::-1])[N - 1 : N - 1 + L_len]
    return grad_L, grad_M


def fit_monitor(
    xI: np.ndarray,
    xQ: np.ndarray,
    z: np.ndarray,
    L_len: int,
    M_len: int,
    n_iter: int = 2000,
    lr: float = 5e-3,
    lr_final: float | None = None,
    optimizer: str = "adam",
    lr_L: float | None = None,
    lr_M: float | None = None,
    init_L: np.ndarray | None = None,
    init_M: np.ndarray | None = None,
    verbose_every: int = 0,
) -> MonitorResult:
    """Jointly estimate the COI Lhat[i,j,l] and the OC Mhat[m].

    Parameters
    ----------
    xI, xQ : known digital drive signals of the transmitter.
    z : sampled PD output (same rate and alignment as xI/xQ).
    L_len, M_len : estimated filter lengths (>= the true supports).
    n_iter : number of full-batch iterations over the snapshot.
    lr : Adam step size (used when optimizer="adam").
    lr_final : if given, the Adam step is cosine-annealed from ``lr``
        down to ``lr_final`` over the iterations.
    optimizer : "adam" (default, fast batch convergence) or "gd"
        (plain gradient descent, i.e. batch LMS as in the paper; use
        lr_L / lr_M as the step sizes beta_L / beta_M).
    init_L, init_M : optional initial filters. By default Lhat starts as
        a centered identity (delta) response and Mhat as a centered
        delta scaled by a least-squares gain fit.
    """
    xI = np.asarray(xI, float)
    xQ = np.asarray(xQ, float)
    z = np.asarray(z, float)
    N = len(xI)
    if len(xQ) != N or len(z) != N:
        raise ValueError("xI, xQ and z must have equal length")

    # --- initialization ---------------------------------------------------
    if init_L is not None:
        L = np.array(init_L, float)
    else:
        L = np.zeros((2, 2, L_len))
        L[0, 0, (L_len - 1) // 2] = 1.0
        L[1, 1, (L_len - 1) // 2] = 1.0
    if init_M is not None:
        M = np.array(init_M, float)
    else:
        M = np.zeros(M_len)
        M[(M_len - 1) // 2] = 1.0
        _, p0, z0 = _forward(L, M, xI, xQ, N)
        g = float(np.dot(z0, z) / max(np.dot(z0, z0), 1e-30))
        M *= g

    # Normalize the loss scale so step sizes are data-independent.
    inv_n = 1.0 / N

    if optimizer == "adam":
        mL = np.zeros_like(L); vL = np.zeros_like(L)
        mM = np.zeros_like(M); vM = np.zeros_like(M)
        b1, b2, eps = 0.9, 0.999, 1e-12
    elif optimizer == "gd":
        if lr_L is None or lr_M is None:
            # Stable defaults: normalize by signal power entering each filter.
            px = np.mean(xI**2 + xQ**2)
            _, p0, _ = _forward(L, M, xI, xQ, N)
            pp = np.mean(p0**2)
            if lr_M is None:
                lr_M = 0.01 / max(pp, 1e-30)
            if lr_L is None:
                lr_L = 0.005 / max(px * np.sum(M**2) * pp, 1e-30)
    else:
        raise ValueError("optimizer must be 'adam' or 'gd'")

    result = MonitorResult(L=L, M=M)
    prev_loss = np.inf
    prev_L = prev_M = None
    for k in range(n_iter):
        y, p, zhat = _forward(L, M, xI, xQ, N)
        err = zhat - z
        loss = float(np.mean(err**2))

        if optimizer == "gd":
            # Bold-driver adaptation keeps plain gradient descent stable
            # regardless of the data scale: revert and shrink the steps
            # when the loss increases, grow them gently otherwise.
            if (not np.isfinite(loss) or loss > prev_loss) and prev_L is not None:
                L, M = prev_L, prev_M
                lr_L *= 0.5
                lr_M *= 0.5
                y, p, zhat = _forward(L, M, xI, xQ, N)
                err = zhat - z
                loss = prev_loss
            else:
                lr_L *= 1.02
                lr_M *= 1.02
                prev_L, prev_M, prev_loss = L, M, loss

        result.loss.append(loss)
        grad_L, grad_M = _gradients(L, M, xI, xQ, err, y, p, N)
        grad_L *= inv_n
        grad_M *= inv_n

        if optimizer == "adam":
            t = k + 1
            if lr_final is not None and n_iter > 1:
                step = lr_final + 0.5 * (lr - lr_final) * (
                    1.0 + np.cos(np.pi * k / (n_iter - 1))
                )
            else:
                step = lr
            mL = b1 * mL + (1 - b1) * grad_L
            vL = b2 * vL + (1 - b2) * grad_L**2
            mM = b1 * mM + (1 - b1) * grad_M
            vM = b2 * vM + (1 - b2) * grad_M**2
            mLh = mL / (1 - b1**t); vLh = vL / (1 - b2**t)
            mMh = mM / (1 - b1**t); vMh = vM / (1 - b2**t)
            L = L - step * mLh / (np.sqrt(vLh) + eps)
            M = M - step * mMh / (np.sqrt(vMh) + eps)
        else:
            L = L - lr_L * grad_L * N  # paper's beta_L acts on the sum, not mean
            M = M - lr_M * grad_M * N

        if verbose_every and (k % verbose_every == 0 or k == n_iter - 1):
            print(f"iter {k:5d}  mse {loss:.6e}")

    result.L, result.M = L, M
    _, _, zhat = _forward(L, M, xI, xQ, N)
    resid = zhat - z
    result.nmse_db = float(10.0 * np.log10(np.mean(resid**2) / np.var(z)))
    return result
