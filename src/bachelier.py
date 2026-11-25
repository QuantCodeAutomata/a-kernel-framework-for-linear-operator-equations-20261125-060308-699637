"""Exact Bachelier (normal) model formulas for the call price and Greeks
(paper Sec. 4.2.2, p.18-19).

Under the Bachelier model X_t = X_0 + sigma W_t, the time-t price of a call
struck at x0 maturing at T is

    u(t,x) = (x - x0) * Phi(d) + sigma*sqrt(T-t) * phi(d),   d = (x-x0)/(sigma*sqrt(T-t))

with Delta(t,x) = Phi(d) and Gamma(t,x) = phi(d) / (sigma*sqrt(T-t)), Phi/phi
the standard normal CDF/PDF (implementation note, paper Sec. 4.2.2).

Custom implementation (paper Sec. 4.2.2): the closed-form Bachelier price and
Greeks are standard option-pricing formulas, reproduced here from the paper's
statement rather than imported from a pricing library, since they serve only
as the reference values compared with the kernel solution.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d(t: np.ndarray, x: np.ndarray, x0: float, sigma: float, T: float) -> np.ndarray:
    tau = np.maximum(T - t, 1e-12)
    return (x - x0) / (sigma * np.sqrt(tau))


def price(t: np.ndarray, x: np.ndarray, x0: float, sigma: float, T: float) -> np.ndarray:
    """Exact Bachelier call price u(t,x)."""
    tau = np.maximum(T - t, 1e-12)
    d = _d(t, x, x0, sigma, T)
    return (x - x0) * norm.cdf(d) + sigma * np.sqrt(tau) * norm.pdf(d)


def delta(t: np.ndarray, x: np.ndarray, x0: float, sigma: float, T: float) -> np.ndarray:
    """Exact Delta = d u/d x = Phi(d)."""
    d = _d(t, x, x0, sigma, T)
    return norm.cdf(d)


def gamma(t: np.ndarray, x: np.ndarray, x0: float, sigma: float, T: float) -> np.ndarray:
    """Exact Gamma = d^2 u/d x^2 = phi(d) / (sigma*sqrt(T-t))."""
    tau = np.maximum(T - t, 1e-12)
    d = _d(t, x, x0, sigma, T)
    return norm.pdf(d) / (sigma * np.sqrt(tau))
