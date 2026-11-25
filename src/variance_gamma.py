"""Variance Gamma model pricing PIDE via Gauss-Laguerre quadrature (paper
Sec. 4.2.5, p.23-25, Eq. 63-64).

The log-price follows, up to a deterministic drift, an asymmetric VG process
with Levy measure (Eq. 63)

    mu(dz) = exp(-zeta(z)|z|) / (nu|z|) dz,
    zeta(z) = (sqrt(theta^2*nu^2/4 + sigma^2*nu/2) + theta*nu*sign(z)/2)^{-1}.

The pricing PIDE (Eq. 64) is

    d_t u + (r+omega) d_x u + int (u(t,x+z) - u(t,x)) mu(dz) - r*u = 0,
    u(T,x) = g(x),

with convexity correction omega = (1/nu) ln(1 - theta*nu - sigma^2*nu/2).
Unlike the Bates model, the jump integral applied to Gaussian kernels has no
closed form here (Sec. 4.2.5: "the integral operator applied to Gaussian
kernels has no closed-form expression"), so it is approximated by
Gauss-Laguerre quadrature on each half-line, with no truncation of small
jumps (the integrand (f(x+z)-f(x))/|z| is bounded as z -> 0 since f is
smooth).

Custom implementation (paper Sec. 4.2.5, Eq. 63-64): the change of variables
mapping the VG Levy measure to a standard Gauss-Laguerre quadrature on
(0, infinity) is specific to this Levy measure and not provided by any
library in this environment; ``numpy.polynomial.laguerre.laggauss`` supplies
only the standard nodes/weights for the weight exp(-z), which this module
rescales to the VG Levy density on each half-line.

Because this jump operator is defined generically as an
``src.jax_utils.Operator`` callable (evaluating ``f`` at quadrature-shifted
points, with no assumption that ``f`` is Gaussian), it plugs directly into
the existing ``src.jax_utils.combine_operators`` / ``cross_kernel`` /
``representer`` machinery used for the ODE and PDE experiments, without any
additional special-casing.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.polynomial.laguerre import laggauss

from src import jax_utils as ju


def vg_zeta(theta: float, sigma: float, nu: float, sign: float) -> float:
    """zeta(z) for z of the given sign (Eq. 63)."""
    core = np.sqrt(theta ** 2 * nu ** 2 / 4 + sigma ** 2 * nu / 2)
    return 1.0 / (core + theta * nu * sign / 2)


def vg_omega(theta: float, sigma: float, nu: float) -> float:
    """Convexity correction omega = (1/nu) ln(1 - theta*nu - sigma^2*nu/2)."""
    arg = 1.0 - theta * nu - 0.5 * sigma ** 2 * nu
    assert arg > 0, "theta*nu + sigma^2*nu/2 must be < 1 (paper Sec. 4.2.5)"
    return np.log(arg) / nu


def vg_jump_operator(theta: float, sigma: float, nu: float, n_nodes: int) -> ju.Operator:
    """Gauss-Laguerre approximation of Au(t,x) = int (u(t,x+z)-u(t,x)) mu(dz).

    The jump acts on coordinate x = z[1] only (z = (t, x)). On each
    half-line the measure exp(-zeta_pm|z|)/(nu|z|) dz is mapped to the
    standard Gauss-Laguerre weight exp(-xi) via z = +/- xi/zeta_pm, giving

        int_0^inf (f(x+/-xi/zeta_pm) - f(x)) * exp(-xi) / (nu*xi) dxi
        ~= sum_j w_j * (f(x+/-xi_j/zeta_pm) - f(x)) / (nu*xi_j).
    """
    zeta_pos = vg_zeta(theta, sigma, nu, sign=+1.0)
    zeta_neg = vg_zeta(theta, sigma, nu, sign=-1.0)
    nodes, weights = laggauss(n_nodes)  # standard nodes/weights for int_0^inf e^{-xi} f(xi) dxi

    def op(f: Callable, z: np.ndarray) -> float:
        total = 0.0
        for xi, w in zip(nodes, weights):
            step_pos = xi / zeta_pos
            step_neg = xi / zeta_neg
            z_pos = z.at[1].add(step_pos) if hasattr(z, "at") else _shift(z, step_pos)
            z_neg = z.at[1].add(-step_neg) if hasattr(z, "at") else _shift(z, -step_neg)
            f0 = f(z)
            total = total + w * ((f(z_pos) - f0) + (f(z_neg) - f0)) / (nu * xi)
        return total

    return op


def _shift(z: np.ndarray, delta: float) -> np.ndarray:
    z = np.array(z, dtype=float, copy=True)
    z[1] += delta
    return z


def vg_pde_operator(theta: float, sigma: float, nu: float, r: float, n_nodes: int) -> ju.Operator:
    """Full interior operator A1 = d_t + (r+omega) d_x + jump - r*Id (Eq. 64)."""
    omega = vg_omega(theta, sigma, nu)
    dt = ju.partial_operator(0)
    dx = ju.partial_operator(1)
    jump = vg_jump_operator(theta, sigma, nu, n_nodes)
    return ju.combine_operators(
        (1.0, dt),
        (r + omega, dx),
        (1.0, jump),
        (-r, ju.identity_operator),
    )


def vg_characteristic_function(u: np.ndarray, T: float, X0: float, r: float,
                                theta: float, sigma: float, nu: float) -> np.ndarray:
    """E[exp(i*u*X_T)] under the VG model with drift r+omega (Eq. 64).

    VG log-return characteristic function: (1 - i*u*theta*nu + 0.5*sigma^2*nu*u^2)^(-T/nu),
    Madan, Carr and Chang (1998).
    """
    u = np.asarray(u, dtype=complex)
    omega = vg_omega(theta, sigma, nu)
    vg_cf = (1.0 - 1j * u * theta * nu + 0.5 * sigma ** 2 * nu * u ** 2) ** (-T / nu)
    return np.exp(1j * u * (X0 + (r + omega) * T)) * vg_cf
