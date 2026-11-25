"""Multi-asset Black-Scholes max-call pricing (paper Sec. 4.2.3, p.19-20).

Symmetric, positively correlated assets dS_t^i = sigma*S_t^i dW_t^i (Eq. in
Sec. 4.2.3). The price function u(t,x) = E^Q[g(S_T) | S_t = x] solves

    d_t u + (sigma^2/2) sum_{i,j} rho_ij x_i x_j d_{ij} u = 0, on (0,T)x(0,inf)^d,
    u(T,.) = g,                                                on {T}x(0,inf)^d,

with the max-call payoff g(x) = (max_i x_i - K)^+, smoothed by the
log-sum-exp function g_eps(x) = eps*ln(1 + sum_i exp((x_i-K)/eps)).

Custom implementation (paper Sec. 4.2.3): the state-dependent generator
(coefficients x_i x_j depend on the collocation point itself, not a fixed
kernel bandwidth) and the log-sum-exp payoff smoothing are specific to this
paper's example; they are implemented directly rather than imported from a
pricing library, since the kernel solver (src/kernel_solver.py) needs the
operator as a plain callable consistent with src/jax_utils.Operator.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np

from src import jax_utils as ju


def smoothed_max_call_payoff(x: jnp.ndarray, K: float, eps: float) -> jnp.ndarray:
    """g_eps(x) = eps * ln(1 + sum_i exp((x_i - K)/eps)) (Sec. 4.2.3, p.19)."""
    return eps * jnp.log(1.0 + jnp.sum(jnp.exp((x - K) / eps)))


def max_call_payoff(x: np.ndarray, K: float) -> np.ndarray:
    """Exact (non-smoothed) payoff g(x) = (max_i x_i - K)^+."""
    return np.maximum(np.max(x, axis=-1) - K, 0.0)


def black_scholes_generator(sigma: float, rho: np.ndarray, d: int) -> ju.Operator:
    """A1 (without the d_t part) = (sigma^2/2) sum_ij rho_ij x_i x_j d_ij,
    acting on the spatial coordinates z[1:1+d] of z=(t,x_1,...,x_d).

    The coefficients x_i x_j are evaluated AT the point z the operator acts
    on (state-dependent, unlike the fixed-bandwidth kernel operators used in
    Sections 4.1-4.2.2), so this cannot be expressed with
    ``src.jax_utils.combine_operators`` alone and is implemented as its own
    ``Operator`` callable.
    """
    rho = jnp.asarray(rho, dtype=jnp.float64)

    def op(f: Callable, z: jnp.ndarray) -> jnp.ndarray:
        x = z[1:1 + d]
        hess = ju.jax.hessian(f)(z)[1:1 + d, 1:1 + d]
        quad = jnp.sum(rho * jnp.outer(x, x) * hess)
        return 0.5 * sigma ** 2 * quad

    return op


def bs_pde_operator(sigma: float, rho: np.ndarray, d: int) -> ju.Operator:
    """Full interior operator A1 = d_t + (sigma^2/2) sum rho_ij x_i x_j d_ij."""
    return ju.combine_operators(
        (1.0, ju.partial_operator(0)),
        (1.0, black_scholes_generator(sigma, rho, d)),
    )


def correlation_matrix(d: int, rho: float) -> np.ndarray:
    """d x d correlation matrix with unit diagonal and off-diagonal rho."""
    mat = np.full((d, d), rho)
    np.fill_diagonal(mat, 1.0)
    return mat


def mc_max_call_price(S0: float, K: float, T: float, sigma: float, rho: np.ndarray,
                       d: int, n_paths: int, seed: int) -> tuple[float, float]:
    """Monte Carlo benchmark price (and standard error) of the max-call option.

    Uses the exact terminal distribution S_T = S0*exp(-sigma^2*T/2 +
    sigma*sqrt(T)*Z), Z ~ N(0, rho) correlated, with antithetic variates
    (Z and -Z), matching the paper's "Monte Carlo benchmark: 2^20 paths with
    antithetic variates" (Sec. 4.2.3, data requirements).
    """
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    chol = np.linalg.cholesky(rho)
    z = rng.standard_normal(size=(half, d)) @ chol.T
    drift = -0.5 * sigma ** 2 * T
    diffusion = sigma * np.sqrt(T)
    s_t_plus = S0 * np.exp(drift + diffusion * z)
    s_t_minus = S0 * np.exp(drift - diffusion * z)
    payoff = np.concatenate([max_call_payoff(s_t_plus, K), max_call_payoff(s_t_minus, K)])
    return float(np.mean(payoff)), float(np.std(payoff, ddof=1) / np.sqrt(payoff.size))
