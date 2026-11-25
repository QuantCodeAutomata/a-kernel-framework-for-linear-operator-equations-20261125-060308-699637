"""Experiment 6: Bates model pricing PIDE with closed-form jump integral
(paper Sec. 4.2.4, p.21-23).

Prices a one-year European call in the Bates stochastic-volatility-with-jumps
model by solving the pricing PIDE (Eq. 59-61) with the kernel solver, using
the closed-form jump-integral kernel blocks of ``src.bates`` (Eq. 62) so no
numerical quadrature of the jump term is required. The kernel price curve at
t=0, v=EXP6_V0 is compared with an independent Carr-Madan FFT benchmark
(``src.fft_pricing``) built from the Bates characteristic function
(``src.bates.bates_characteristic_function``).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import sobol_uniform_points
from src import bates, config, fft_pricing, jax_utils as ju, kernel_solver as ks

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_collocation(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Scrambled Sobol collocation points in the box ln S in EXP6_X_BOX,
    v in EXP6_V_BOX (paper Sec. 4.2.4, data requirements): interior points
    in (0,T) x box, terminal points on {T} x box."""
    t_low, t_high = 0.0, config.EXP6_T
    x_low, x_high = config.EXP6_X_BOX
    v_low, v_high = config.EXP6_V_BOX

    u1 = sobol_uniform_points(n, 3, seed)
    t1 = u1[:, 0] * (t_high - t_low) + t_low
    x1 = u1[:, 1] * (x_high - x_low) + x_low
    v1 = u1[:, 2] * (v_high - v_low) + v_low
    z1 = np.column_stack([t1, x1, v1])

    u2 = sobol_uniform_points(n, 2, seed + 1)
    x2 = u2[:, 0] * (x_high - x_low) + x_low
    v2 = u2[:, 1] * (v_high - v_low) + v_low
    z2 = np.column_stack([np.full(n, t_high), x2, v2])
    return z1, z2


def run(seed: int = config.RANDOM_SEED, full: bool = False) -> dict:
    n = config.EXP6_N_PAPER if full else config.EXP6_N_RUN

    z1, z2 = build_collocation(n, seed)
    f1 = np.zeros(n)
    f2 = np.maximum(np.exp(z2[:, 1]) - config.EXP6_K, 0.0)
    w1, w2 = float(n), float(n)  # Sec. 4.2.4: w_i = n_i so that lambda*G = lambda*I_N

    k11, k12, k21, k22 = bates.build_bates_kernel_blocks(
        config.EXP6_GAMMA0, config.EXP6_GAMMA1, config.EXP6_GAMMA2,
        config.EXP6_KAPPA, config.EXP6_VBAR, config.EXP6_ETA, config.EXP6_RHO,
        config.EXP6_ZETA, config.EXP6_ALPHA, config.EXP6_BETA2)

    k_blocks = {
        (0, 0): ju.build_matrix(k11, z1, z1),
        (0, 1): ju.build_matrix(k12, z1, z2),
        (1, 0): ju.build_matrix(k21, z2, z1),
        (1, 1): ju.build_matrix(k22, z2, z2),
    }
    solution = ks.solve(k_blocks, [f1, f2], [w1, w2], config.EXP6_LAMBDA)
    objective = ks.objective_value(k_blocks, [f1, f2], [w1, w2], config.EXP6_LAMBDA, solution)

    spots = np.linspace(80.0, 120.0, config.EXP6_N_PRICE_POINTS)
    x_query = np.log(spots)
    zq = np.column_stack([np.zeros_like(x_query), x_query, np.full_like(x_query, config.EXP6_V0)])

    # Eq. 30: A_m acts on the *training* argument only; the query argument is
    # evaluated plainly (identity). That is exactly the functional form of the
    # (1,2) and (2,2) blocks above (A1 on the first/training argument,
    # identity on the second/query argument) -- *not* k11/k21, which apply
    # the operator to both arguments and are only valid for the Gram matrix.
    kq1 = ju.build_matrix(k12, z1, zq)
    kq2 = ju.build_matrix(k22, z2, zq)
    kernel_price = solution.predict([kq1, kq2])

    # Figure 9 plots the price as a function of the *current spot* S at fixed
    # strike K=EXP6_K, t=0, v=EXP6_V0 -- the same quantity the kernel solution
    # evaluates along its x=ln(S) query dimension. The Carr-Madan transform
    # instead prices a fixed spot across a *strike* grid, so a single FFT
    # call with X0=ln(EXP6_S0) would price Call(S0 fixed, K=spots varying),
    # not Call(S=spots varying, K=EXP6_K fixed). Under Bates' log-price
    # dynamics the characteristic function of ln(S_T/S_0) does not depend on
    # S_0, so Call(S, K, T) = S * c(K/S, T) for a model-only function c; one
    # FFT per spot, each centered at X0=ln(spot) and read off at the single
    # strike K=EXP6_K, reproduces the spot curve exactly.
    fft_prices = np.empty_like(spots)
    for i, spot in enumerate(spots):
        cf = lambda u, s=spot: bates.bates_characteristic_function(
            u, config.EXP6_T, np.log(s), config.EXP6_V0,
            config.EXP6_KAPPA, config.EXP6_VBAR, config.EXP6_ETA, config.EXP6_RHO,
            config.EXP6_ZETA, config.EXP6_ALPHA, config.EXP6_BETA2)
        fft_prices[i] = fft_pricing.carr_madan_call_prices(
            cf, spot, np.array([config.EXP6_K]), config.EXP6_T, config.EXP6_R)[0]

    rel_l2_error = float(np.linalg.norm(kernel_price - fft_prices) / np.linalg.norm(fft_prices))
    max_abs_error = float(np.max(np.abs(kernel_price - fft_prices)))

    atm_idx = int(np.argmin(np.abs(spots - config.EXP6_S0)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(spots, fft_prices, label="FFT benchmark (Carr-Madan)", lw=2)
    ax.plot(spots, kernel_price, "--", label="kernel solution", lw=2)
    ax.set_xlabel("S")
    ax.set_ylabel("call price")
    ax.set_title("Exp. 6: Bates model, t=0, v=0.04")
    ax.legend()
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / "exp6_price_comparison.png", dpi=150)
    plt.close(fig)

    metrics = {
        "experiment": "exp6_bates_closed_form_jump",
        "n": n, "full": full,
        "lambda": config.EXP6_LAMBDA,
        "relative_l2_error_vs_fft": rel_l2_error,
        "max_abs_error_vs_fft": max_abs_error,
        "atm_kernel_price": float(kernel_price[atm_idx]),
        "atm_fft_price": float(fft_prices[atm_idx]),
        "objective_value": objective,
        "random_seed": seed,
    }
    with open(RESULTS_DIR / "exp6_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


if __name__ == "__main__":
    import sys

    print(json.dumps(run(full="--full" in sys.argv), indent=2))
