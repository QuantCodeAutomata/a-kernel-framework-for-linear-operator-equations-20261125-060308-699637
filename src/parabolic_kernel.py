"""Parabolic kernel with time-dependent spatial scaling (paper Eq. 53-55,
Sec. 4.2.1-4.2.3, p.18-20).

    k((t,x), (s,y)) = exp{ -gamma0/2 |t-s|^2 - gamma1/2 ||psi(t,x)-psi(s,y)||^2 },
    psi(t,x) = x / sqrt(t + eps)           (forward warp, Eq. 54, heat equation)
    psi(t,x) = x / sqrt(T - t + eps)       (backward warp, Eq. 55, pricing PDEs)

The warp eps > 0 avoids the singularity at t = 0 (forward) or t = T
(backward). The differential operators used in experiments 3-5 (heat
equation, Bachelier PDE, multi-asset Black-Scholes PDE) require up to
fourth-order mixed derivatives of this kernel (because A_1 contains a second
spatial derivative and the cross-kernel k^{1,1} = A_1 A_1 k differentiates
both arguments). These are obtained by automatic differentiation through
``src.jax_utils``, following the paper's remark (Sec. 3.3, p.14) that
"operator-induced kernels are computed by automatic differentiation with the
JAX library" whenever no closed form is used.

Custom implementation (paper Eq. 53-55): the parabolic warp and the use of
nested JAX derivatives for its cross-kernels are specific to this paper.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np

from src import jax_utils as ju


def make_forward_parabolic_kernel(gamma0: float, gamma1: float, eps: float) -> Callable:
    """k((t,x),(s,y)) with forward warp psi(t,x)=x/sqrt(t+eps) (Eq. 53-54).

    ``z = (t, x_1, ..., x_d)`` is a (d+1)-vector; z[0] is time, z[1:] space.
    """

    def k(z: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        t, x = z[0], z[1:]
        s, y = w[0], w[1:]
        psi_z = x / jnp.sqrt(t + eps)
        psi_w = y / jnp.sqrt(s + eps)
        return jnp.exp(-gamma0 / 2 * (t - s) ** 2
                        - gamma1 / 2 * jnp.sum((psi_z - psi_w) ** 2))

    return k


def make_backward_parabolic_kernel(gamma0: float, gamma1: float, eps: float, T: float) -> Callable:
    """k((t,x),(s,y)) with backward warp psi(t,x)=x/sqrt(T-t+eps) (Eq. 55)."""

    def k(z: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        t, x = z[0], z[1:]
        s, y = w[0], w[1:]
        psi_z = x / jnp.sqrt(T - t + eps)
        psi_w = y / jnp.sqrt(T - s + eps)
        return jnp.exp(-gamma0 / 2 * (t - s) ** 2
                        - gamma1 / 2 * jnp.sum((psi_z - psi_w) ** 2))

    return k


def make_backward_parabolic_kernel_scaled(gamma0: float, gamma1: float, eps: float, T: float,
                                           shift: float, scale: float) -> Callable:
    """Backward-warped parabolic kernel with an affine spatial rescaling,

        psi(t,x) = (x - shift*1) / (scale * sqrt(T - t + eps))     (Sec. 4.2.3, Eq. 53),

    used for the multi-asset Black-Scholes max-call (paper Sec. 4.2.3, p.19-20,
    "psi(t,x) = (x-S0*1)/(S0*sigma*sqrt(T-t+eps))"), with ``shift=S0`` and
    ``scale=S0*sigma``. ``x`` may be a vector of any dimension d.
    """

    def k(z: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        t, x = z[0], z[1:]
        s, y = w[0], w[1:]
        psi_z = (x - shift) / (scale * jnp.sqrt(T - t + eps))
        psi_w = (y - shift) / (scale * jnp.sqrt(T - s + eps))
        return jnp.exp(-gamma0 / 2 * (t - s) ** 2
                        - gamma1 / 2 * jnp.sum((psi_z - psi_w) ** 2))

    return k


def make_anisotropic_gaussian_kernel(gammas: np.ndarray) -> Callable:
    """k(z,w) = exp(-1/2 sum_i gamma_i (z_i-w_i)^2), no time-dependent warp.

    Used for the Bates model kernel (paper Sec. 4.2.4, p.22), where the
    coordinates are (t, x=ln S, v) and no rescaling is applied.
    """
    gammas = jnp.asarray(gammas, dtype=jnp.float64)

    def k(z: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        diff = z - w
        return jnp.exp(-0.5 * jnp.sum(gammas * diff ** 2))

    return k


def time_derivative_operator() -> ju.Operator:
    """A = d/dt, first coordinate of z."""
    return ju.partial_operator(0)


def laplacian_operator(d: int, coeffs: np.ndarray | None = None) -> ju.Operator:
    """A = sum_i coeff_i d^2/dx_i^2 over the spatial coordinates (indices 1..d)."""
    if coeffs is None:
        coeffs = np.ones(d)
    terms = [(float(coeffs[i]), ju.second_partial_operator(i + 1, i + 1)) for i in range(d)]
    return ju.combine_operators(*terms)


def cross_diffusion_operator(d: int, cov: np.ndarray) -> ju.Operator:
    """A = sum_{i,j} 0.5*cov_ij * x_i*x_j* d^2/(dx_i dx_j) style operators are
    built by the caller via ``ju.combine_operators``; this helper returns the
    plain sum_{i,j} cov_ij d^2/(dx_i dx_j) part (coefficients supplied by the
    caller, e.g. multiplied by asset prices for Black-Scholes)."""
    terms = []
    for i in range(d):
        for j in range(d):
            terms.append((float(cov[i, j]), ju.second_partial_operator(i + 1, j + 1)))
    return ju.combine_operators(*terms)
