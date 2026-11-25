"""Synthetic collocation-point generators for the kernel-solver experiments.

No market data is used anywhere in this repository: every experiment solves
a linear operator equation (an ODE, a PDE, or a PIDE) defined analytically in
Filipovic & Tissot-Daguette, "A Kernel Framework for Linear Operator
Equations". Collocation points are drawn from the probability measures the
paper specifies for each experiment (uniform, equally spaced, or scrambled
Sobol), never estimated from historical prices.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc


def uniform_points(n: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    """n points drawn uniformly from (low, high)."""
    return rng.uniform(low, high, size=n)


def equally_spaced_points(n: int, low: float, high: float) -> np.ndarray:
    """n equally spaced points in [low, high]."""
    return np.linspace(low, high, n)


def uniform_box_points(n: int, low: np.ndarray, high: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """n points drawn uniformly from the box [low, high]^d (d = len(low))."""
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    d = low.shape[0]
    u = rng.uniform(size=(n, d))
    return low + u * (high - low)


def sobol_uniform_points(n: int, d: int, seed: int) -> np.ndarray:
    """n scrambled Sobol points in (0,1)^d, clipped away from the boundary.

    ``n`` is rounded up internally to a power of two as required by
    ``scipy.stats.qmc.Sobol``, and the first ``n`` points of the resulting
    sequence are returned.
    """
    m = int(np.ceil(np.log2(max(n, 1))))
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    points = sampler.random_base2(m=m)[:n]
    eps = 1e-6
    return np.clip(points, eps, 1 - eps)


def sobol_time_space_points(n: int, t_low: float, t_high: float,
                             x_low: np.ndarray, x_high: np.ndarray,
                             seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Scrambled Sobol points mapped to a box [t_low,t_high] x [x_low,x_high]^d.

    Returns (t, x) with t of shape (n,) and x of shape (n, d).
    """
    x_low = np.atleast_1d(np.asarray(x_low, dtype=float))
    x_high = np.atleast_1d(np.asarray(x_high, dtype=float))
    d = x_low.shape[0]
    u = sobol_uniform_points(n, d + 1, seed)
    t = t_low + u[:, 0] * (t_high - t_low)
    x = x_low + u[:, 1:] * (x_high - x_low)
    return t, x


def sobol_time_lognormal_points(n: int, T: float, mean_log: np.ndarray, cov_log: np.ndarray,
                                 seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Scrambled Sobol points with time uniform on (0,T) (first coordinate)
    and space mapped to a correlated lognormal distribution (paper Sec.
    4.2.3: "map to uniform on (0,T) in time, and to correlated lognormal
    distribution S0*exp(sigma*sqrt(T)*Z) ... in space").

    Returns (t, x) with t of shape (n,) and x of shape (n, d).
    """
    from scipy.stats import norm

    d = mean_log.shape[0]
    u = sobol_uniform_points(n, d + 1, seed)
    t = u[:, 0] * T
    z = norm.ppf(u[:, 1:])
    chol = np.linalg.cholesky(cov_log)
    log_s = mean_log + z @ chol.T
    return t, np.exp(log_s)


def sobol_lognormal_points(n: int, d: int, mean_log: np.ndarray, cov_log: np.ndarray,
                            seed: int) -> np.ndarray:
    """n scrambled Sobol points mapped to a correlated multivariate normal
    in log-space via the inverse normal CDF, then exponentiated.

    Used for multi-asset Black-Scholes collocation points (paper Sec.
    4.2.3): prices S = exp(mean_log + L @ Z) with Z standard normal obtained
    from Sobol points via the inverse CDF, and L the Cholesky factor of
    ``cov_log``.
    """
    from scipy.stats import norm

    u = sobol_uniform_points(n, d, seed)
    z = norm.ppf(u)
    chol = np.linalg.cholesky(cov_log)
    log_s = mean_log + z @ chol.T
    return np.exp(log_s)
