"""Validate the backpropagation gradients against finite differences."""

import numpy as np
import pytest

from s21_monitor.monitor import _forward, _gradients


def _loss(L, M, xI, xQ, z, N):
    _, _, zhat = _forward(L, M, xI, xQ, N)
    return np.sum((zhat - z) ** 2)


@pytest.mark.parametrize("seed", [0, 1])
def test_gradients_match_finite_differences(seed):
    rng = np.random.default_rng(seed)
    N, L_len, M_len = 64, 3, 5
    xI = rng.standard_normal(N)
    xQ = rng.standard_normal(N)
    z = rng.standard_normal(N)
    L = rng.standard_normal((2, 2, L_len)) * 0.5
    M = rng.standard_normal(M_len) * 0.5

    y, p, zhat = _forward(L, M, xI, xQ, N)
    err = zhat - z
    grad_L, grad_M = _gradients(L, M, xI, xQ, err, y, p, N)

    eps = 1e-6
    for m in range(M_len):
        Mp, Mm = M.copy(), M.copy()
        Mp[m] += eps
        Mm[m] -= eps
        fd = (_loss(L, Mp, xI, xQ, z, N) - _loss(L, Mm, xI, xQ, z, N)) / (2 * eps)
        assert grad_M[m] == pytest.approx(fd, rel=1e-4, abs=1e-6)

    for i in range(2):
        for j in range(2):
            for l in range(L_len):
                Lp, Lm = L.copy(), L.copy()
                Lp[i, j, l] += eps
                Lm[i, j, l] -= eps
                fd = (_loss(Lp, M, xI, xQ, z, N) - _loss(Lm, M, xI, xQ, z, N)) / (
                    2 * eps
                )
                assert grad_L[i, j, l] == pytest.approx(fd, rel=1e-4, abs=1e-6)
