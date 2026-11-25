"""Bates model pricing PIDE with a closed-form jump-integral kernel block
(paper Sec. 4.2.4, p.21-23, Eq. 57-62).

Dynamics (Eq. 57-58), zero interest rate and dividends:

    dX_t = -(V_t/2 + zeta*gammaQ) dt + sqrt(V_t) dW_t + d(sum_i J_i),
    dV_t = kappa*(Vbar - V_t) dt + eta*sqrt(V_t) dZ_t,   corr(W,Z) = rho,

J_i ~ N(alpha, beta) i.i.d. jump sizes, gammaQ = E^Q[e^{J_1}] - 1. The
pricing PIDE (Eq. 59) is d_t u + Gu + Iu - zeta*u = 0, u(T,x,v) = g(x), with

    Gu = -(v/2+zeta*gammaQ) d_x u + kappa(Vbar-v) d_v u
         + v/2 d_xx u + rho*eta*v d_xv u + eta^2*v/2 d_vv u,     (Eq. 60)
    Iu(t,x,v) = zeta * int u(t,x+z,v) phi_{alpha,beta}(z) dz.     (Eq. 61)

G is a state-dependent (coefficients depend on v) differential operator, so
it is implemented as a plain ``src.jax_utils.Operator`` callable combining
``jax.grad``/``jax.hessian`` terms, like ``src.black_scholes``.

I acting on a Gaussian kernel has the closed form (Eq. 62, p.22):

    k_x^I(y) = zeta*c*phi_{x+alpha, beta+1/gamma1}(y),  c = sqrt(2*pi/gamma1),

i.e. convolving the Gaussian kernel section with the jump density leaves it
Gaussian, with the mean shifted by +/-alpha and the variance increased by
beta. Writing the base log-price kernel section as a Gaussian in
u = x - y with (amplitude, mean, variance) = (1, 0, 1/gamma1), a single jump
convolution on the "x" (first) argument maps (amp, mean, var) to
(amp*sqrt(var/(var+beta)), mean-alpha, var+beta); on the "y" (second)
argument, to (amp*sqrt(var/(var+beta)), mean+alpha, var+beta) (derived from
Eq. 62 by expanding phi_{x+alpha,beta+1/gamma1}(y) in u = x-y). This module
builds the resulting closed-form representer blocks instead of applying
quadrature, eliminating the need for numerical quadrature of the jump term
for this model.

Custom implementation (paper Sec. 4.2.4, Eq. 57-62): no option-pricing
library in this environment exposes the Bates PIDE operator or the
closed-form Gaussian-kernel/jump-convolution identity used by the kernel
solver; both are implemented here directly from the paper's formulas.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from src import jax_utils as ju
from src.parabolic_kernel import make_anisotropic_gaussian_kernel

GaussianTerm = tuple[float, float, float]  # (amplitude, mean, variance) of amp*exp(-(u-mean)^2/(2*var))


def _shift_x(term: GaussianTerm, alpha: float, beta: float) -> GaussianTerm:
    """Jump convolution applied to the first ("x") argument (Eq. 62)."""
    amp, mean, var = term
    return amp * jnp.sqrt(var / (var + beta)), mean - alpha, var + beta


def _shift_y(term: GaussianTerm, alpha: float, beta: float) -> GaussianTerm:
    """Jump convolution applied to the second ("y") argument."""
    amp, mean, var = term
    return amp * jnp.sqrt(var / (var + beta)), mean + alpha, var + beta


def _term_value(term: GaussianTerm, u: jnp.ndarray) -> jnp.ndarray:
    amp, mean, var = term
    return amp * jnp.exp(-(u - mean) ** 2 / (2 * var))


def bates_generator(kappa: float, vbar: float, eta: float, rho: float,
                     zeta: float, alpha: float, beta: float) -> ju.Operator:
    """A1 = d_t + G (Eq. 60), without the jump part I - zeta*Id.

    z = (t, x, v): z[0]=t (time), z[1]=x (log-price), z[2]=v (variance).
    """
    gamma_q = jnp.exp(alpha + 0.5 * beta) - 1.0  # E^Q[e^J] - 1

    def op(f: Callable, z: jnp.ndarray) -> jnp.ndarray:
        v = z[2]
        g = jax.grad(f)(z)
        h = jax.hessian(f)(z)
        dt, dx, dv = g[0], g[1], g[2]
        dxx, dvv, dxv = h[1, 1], h[2, 2], h[1, 2]
        return (dt
                - (v / 2 + zeta * gamma_q) * dx
                + kappa * (vbar - v) * dv
                + 0.5 * v * dxx
                + rho * eta * v * dxv
                + 0.5 * eta ** 2 * v * dvv)

    return op


def build_bates_kernel_blocks(gamma0: float, gamma1: float, gamma2: float,
                               kappa: float, vbar: float, eta: float, rho: float,
                               zeta: float, alpha: float, beta: float):
    """Return the four pairwise kernel functions k^{l,m}(z, w), l, m in {1, 2}.

    A1 = d_t + G + (I - zeta*Id) on the interior domain X1 (Eq. 59); A2 =
    restriction (identity) on the terminal slice X2 = {T} x R x (0,inf).
    """
    base_kernel = make_anisotropic_gaussian_kernel(np.array([gamma0, gamma1, gamma2]))
    op_dg = bates_generator(kappa, vbar, eta, rho, zeta, alpha, beta)

    def k_t(t: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-0.5 * gamma0 * (t - s) ** 2)

    def k_v(v: jnp.ndarray, vp: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-0.5 * gamma2 * (v - vp) ** 2)

    base_x_term: GaussianTerm = (1.0, 0.0, 1.0 / gamma1)

    def k_jump_first(z: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        """(I - zeta*Id) applied to the first (training) argument z."""
        u = z[1] - w[1]
        shifted = _shift_x(base_x_term, alpha, beta)
        x_part = zeta * _term_value(shifted, u) - zeta * _term_value(base_x_term, u)
        return x_part * k_t(z[0], w[0]) * k_v(z[2], w[2])

    def k_jump_second(z: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        """(I - zeta*Id) applied to the second (query) argument w."""
        u = z[1] - w[1]
        shifted = _shift_y(base_x_term, alpha, beta)
        x_part = zeta * _term_value(shifted, u) - zeta * _term_value(base_x_term, u)
        return x_part * k_t(z[0], w[0]) * k_v(z[2], w[2])

    def k_jump_both(z: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        """(I - zeta*Id) applied to both arguments (expanded, see module docstring)."""
        u = z[1] - w[1]
        term_y = _shift_y(base_x_term, alpha, beta)
        term_xy = _shift_x(term_y, alpha, beta)
        term_x = _shift_x(base_x_term, alpha, beta)
        combo = (zeta ** 2 * _term_value(term_xy, u)
                 - zeta ** 2 * _term_value(term_x, u)
                 - zeta ** 2 * _term_value(term_y, u)
                 + zeta ** 2 * _term_value(base_x_term, u))
        return combo * k_t(z[0], w[0]) * k_v(z[2], w[2])

    def k11(z: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        return (ju.cross_kernel(base_kernel, op_dg, op_dg)(z, w)
                + op_dg(lambda zz: k_jump_second(zz, w), z)
                + op_dg(lambda ww: k_jump_first(z, ww), w)
                + k_jump_both(z, w))

    def k12(z: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        return ju.cross_kernel(base_kernel, op_dg, ju.identity_operator)(z, w) + k_jump_first(z, w)

    def k21(z: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        return ju.cross_kernel(base_kernel, ju.identity_operator, op_dg)(z, w) + k_jump_second(z, w)

    def k22(z: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        return base_kernel(z, w)

    return k11, k12, k21, k22


def bates_characteristic_function(u: np.ndarray, T: float, X0: float, V0: float,
                                   kappa: float, vbar: float, eta: float, rho: float,
                                   zeta: float, alpha: float, beta: float) -> np.ndarray:
    """E[exp(i*u*X_T)] under the Bates model (Eq. 57-58), r = q = 0.

    Heston diffusion component in the numerically stable "little trap" form
    (Albrecher, Mayer, Schoutens, Tistaert 2007); jump component is the
    standard compensated compound-Poisson characteristic function.
    """
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    d = np.sqrt((rho * eta * iu - kappa) ** 2 + eta ** 2 * (iu + u ** 2))
    g = (kappa - rho * eta * iu - d) / (kappa - rho * eta * iu + d)
    exp_dT = np.exp(-d * T)
    C = (kappa * vbar / eta ** 2) * ((kappa - rho * eta * iu - d) * T
                                      - 2 * np.log((1 - g * exp_dT) / (1 - g)))
    D = ((kappa - rho * eta * iu - d) / eta ** 2) * (1 - exp_dT) / (1 - g * exp_dT)
    heston_cf = np.exp(C + D * V0)

    gamma_q = np.exp(alpha + 0.5 * beta) - 1.0
    jump_cf = np.exp(zeta * T * (np.exp(iu * alpha - 0.5 * u ** 2 * beta) - 1.0 - iu * gamma_q))

    return np.exp(iu * X0) * heston_cf * jump_cf
