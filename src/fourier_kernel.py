"""Truncated Fourier kernel on [0,1] for the second-order ODE example
(paper Sec. 4.1, p.17; kernel construction in Example 2.6, p.4-5).

The kernel is

    k_L(x, y) = 1 + sum_{n=1}^{L} (1/n^4) cos(pi n (x - y)),

each cosine mode cos(pi n (x-y)) being an eigenfunction of the differential
operator A = d^2/dx^2 + c (second derivative plus a constant), with
eigenvalue c - (pi n)^2, and of the identity operator, with eigenvalue 1.
Because A_1 = d^2/dx^2 + zeta_n and A_2 is the restriction (identity), the
cross-kernels k^{l,m} = A_l A_m k admit closed-form termwise expressions
(implementation note, paper Sec. 4.1, p.17): no automatic differentiation is
required.

Custom implementation (paper Sec. 4.1, Example 2.6, Eq. 28): the truncated
Fourier/cosine kernel and the operator eigenvalue bookkeeping below are
specific to this paper's ODE example and not provided by any library.
"""

from __future__ import annotations

import numpy as np


def mode_coefficients(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Mode indices n = 0, ..., L and kernel coefficients c_n.

    c_0 = 1 (constant term) and c_n = 1/n^4 for n = 1, ..., L, matching
    k_L(x,y) = 1 + sum_{n=1}^L (1/n^4) cos(pi n (x-y)).
    """
    n = np.arange(0, L + 1)
    c = np.ones_like(n, dtype=float)
    c[1:] = 1.0 / n[1:] ** 4
    return n, c


def second_derivative_plus_const_multiplier(n: np.ndarray, const: float) -> np.ndarray:
    """Eigenvalue of A = d^2/dx^2 + const on cos(pi n x): const - (pi n)^2."""
    return const - (np.pi * n) ** 2


def identity_multiplier(n: np.ndarray) -> np.ndarray:
    """Eigenvalue of the identity (restriction) operator: 1 for every mode."""
    return np.ones_like(n, dtype=float)


def cross_kernel_matrix(x: np.ndarray, y: np.ndarray,
                         n: np.ndarray, c: np.ndarray,
                         mult_l: np.ndarray, mult_m: np.ndarray) -> np.ndarray:
    """k^{l,m}(x_i, y_j) = sum_n c_n * mult_l(n) * mult_m(n) * cos(pi n (x_i-y_j))."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    diff = x[:, None] - y[None, :]
    weight = c * mult_l * mult_m
    angles = np.pi * n[None, None, :] * diff[:, :, None]
    return np.tensordot(np.cos(angles), weight, axes=([2], [0]))


def kernel_matrix(x: np.ndarray, y: np.ndarray, n: np.ndarray, c: np.ndarray) -> np.ndarray:
    """k_L(x_i, y_j) = sum_n c_n cos(pi n (x_i - y_j))."""
    identity = identity_multiplier(n)
    return cross_kernel_matrix(x, y, n, c, identity, identity)
