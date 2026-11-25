"""Carr-Madan FFT benchmark for European call prices (paper refs [7], used as
the independent benchmark in Sec. 4.2.4 and 4.2.5).

Custom implementation (Carr, P. and Madan, D. B., "Option valuation using the
fast Fourier transform", Journal of Computational Finance, 1999): no
quantitative-finance library in this environment exposes a generic
Carr-Madan FFT pricer parametrized by an arbitrary characteristic function,
so the damped-call transform and its FFT evaluation are implemented here
directly from the original formula.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def carr_madan_call_prices(char_func: Callable[[np.ndarray], np.ndarray],
                            S0: float,
                            strikes: np.ndarray,
                            T: float,
                            r: float,
                            alpha: float = 1.5,
                            n: int = 2 ** 16,
                            eta: float = 0.05) -> np.ndarray:
    """European call prices at ``strikes`` via the Carr-Madan FFT method.

    ``char_func(u)`` must return the characteristic function E[exp(i*u*X_T)]
    of the risk-neutral log-price X_T = ln(S0) + (risk-neutral log-return),
    evaluated at complex argument ``u - i*(alpha+1)``; discounting by
    exp(-r*T) is applied separately here, so ``char_func`` must represent an
    undiscounted expectation of exp(i*u*X_T).
    """
    lambda_ = 2 * np.pi / (n * eta)       # log-strike grid spacing
    b = n * lambda_ / 2                     # half-width of the log-strike grid
    u = np.arange(n) * eta
    # The characteristic function already carries ln(S0) in its phase
    # (it is E[exp(i*u*ln(S_T))]), so the FFT log-strike grid must span
    # [-b, b) in absolute log-strike, not be re-centered on ln(S0).
    log_strike_grid = -b + lambda_ * np.arange(n)

    j = np.arange(1, n + 1)
    simpson = (3 + (-1) ** j - (j == 1).astype(float)) / 3  # Simpson weights

    freq = u - 1j * (alpha + 1)
    numerator = np.exp(-r * T) * char_func(freq)
    denom = alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u
    psi = numerator / denom

    integrand = np.exp(1j * b * u) * psi * eta * simpson
    fft_values = np.fft.fft(integrand).real
    call_prices = np.exp(-alpha * log_strike_grid) / np.pi * fft_values

    return np.interp(np.log(strikes), log_strike_grid, call_prices)
