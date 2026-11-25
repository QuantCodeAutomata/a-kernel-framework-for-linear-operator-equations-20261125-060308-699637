"""Experiment 7: Variance Gamma model pricing PIDE with numerical quadrature
(paper Sec. 4.2.5, p.23-25, Eq. 63-64).

Prices a one-year at-the-money European call in the pure-jump Variance Gamma
model by solving the pricing PIDE with the kernel solver. Unlike the Bates
model of experiment 6, the jump integral operator applied to a plain
(non-parabolic) Gaussian kernel has no closed form (Sec. 4.2.5), so it is
approximated by Gauss-Laguerre quadrature (``src.variance_gamma.vg_jump_operator``).
The study (1) tracks convergence of the kernel price as the number of
collocation points n grows, and (2) reports a 99% confidence interval for the
mean price over independent randomized Sobol runs at a fixed n, both compared
with an independent Carr-Madan FFT benchmark built from the VG characteristic
function.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from data.generate_synthetic import sobol_uniform_points
from src import config, fft_pricing, jax_utils as ju, kernel_solver as ks, variance_gamma as vg
from src.parabolic_kernel import make_anisotropic_gaussian_kernel

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_collocation(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Scrambled Sobol collocation points in [0,T] x log-strike box (paper
    Sec. 4.2.5, data requirements): interior points in (0,T) x box, terminal
    points on {T} x box."""
    x_low = np.log(config.EXP7_S0) - 5.0 * config.EXP7_SIGMA * np.sqrt(config.EXP7_T)
    x_high = np.log(config.EXP7_S0) + 5.0 * config.EXP7_SIGMA * np.sqrt(config.EXP7_T)

    u1 = sobol_uniform_points(n, 2, seed)
    t1 = u1[:, 0] * config.EXP7_T
    x1 = u1[:, 1] * (x_high - x_low) + x_low
    z1 = np.column_stack([t1, x1])

    u2 = sobol_uniform_points(n, 1, seed + 1)
    x2 = u2[:, 0] * (x_high - x_low) + x_low
    z2 = np.column_stack([np.full(n, config.EXP7_T), x2])
    return z1, z2


def build_kernel_blocks():
    """The four pairwise kernel functions k^{l,m} for the VG system (Eq. 63-64):
    A1 = d_t + (r+omega) d_x + jump - r*Id on the interior; A2 = restriction
    (identity) at maturity (paper Sec. 4.2.5)."""
    base_kernel = make_anisotropic_gaussian_kernel(np.array([config.EXP7_GAMMA0, config.EXP7_GAMMA1]))
    op_a1 = vg.vg_pde_operator(config.EXP7_THETA, config.EXP7_SIGMA, config.EXP7_NU,
                                config.EXP7_R, config.EXP7_N_GAUSS_LAGUERRE)
    k11 = ju.cross_kernel(base_kernel, op_a1, op_a1)
    k12 = ju.cross_kernel(base_kernel, op_a1, ju.identity_operator)
    k21 = ju.cross_kernel(base_kernel, ju.identity_operator, op_a1)
    k22 = base_kernel
    return k11, k12, k21, k22


def price_once(n: int, seed: int) -> tuple[float, np.ndarray]:
    """Solve the kernel system once at the given collocation size and seed;
    return the ATM call price and the solution's beta sizes (for the
    objective)."""
    z1, z2 = build_collocation(n, seed)
    f1 = np.zeros(n)
    f2 = np.maximum(np.exp(z2[:, 1]) - config.EXP7_K, 0.0)
    w1 = w2 = float(n)  # Sec. 4.2.5: w_i = n so that lambda*G = lambda*I_N

    k11, k12, k21, k22 = build_kernel_blocks()
    k_blocks = {
        (0, 0): ju.build_matrix(k11, z1, z1),
        (0, 1): ju.build_matrix(k12, z1, z2),
        (1, 0): ju.build_matrix(k21, z2, z1),
        (1, 1): ju.build_matrix(k22, z2, z2),
    }
    solution = ks.solve(k_blocks, [f1, f2], [w1, w2], config.EXP7_LAMBDA)

    zq = np.array([[0.0, np.log(config.EXP7_S0)]])
    kq1 = ju.build_matrix(k12, z1, zq)
    kq2 = ju.build_matrix(k22, z2, zq)
    price = float(solution.predict([kq1, kq2])[0])
    return price, k_blocks, solution, f1, f2, w1, w2


def fft_benchmark() -> float:
    cf = lambda u: vg.vg_characteristic_function(
        u, config.EXP7_T, np.log(config.EXP7_S0), config.EXP7_R,
        config.EXP7_THETA, config.EXP7_SIGMA, config.EXP7_NU)
    price = fft_pricing.carr_madan_call_prices(
        cf, config.EXP7_S0, np.array([config.EXP7_K]), config.EXP7_T, config.EXP7_R)[0]
    return float(price)


def run(seed: int = config.RANDOM_SEED, full: bool = False) -> dict:
    n_values = config.EXP7_N_CONVERGENCE_PAPER if full else config.EXP7_N_CONVERGENCE_RUN
    n_ci = config.EXP7_N_CI_PAPER if full else min(config.EXP7_N_CI_PAPER, n_values[-1])
    n_ci_runs = config.EXP7_N_CI_RUNS_RUN

    fft_price = fft_benchmark()

    convergence = []
    for n in n_values:
        price, k_blocks, solution, f1, f2, w1, w2 = price_once(n, seed)
        objective = ks.objective_value(k_blocks, [f1, f2], [w1, w2], config.EXP7_LAMBDA, solution)
        convergence.append({"n": int(n), "kernel_price": price, "objective_value": objective})

    ci_prices = []
    for run_idx in range(n_ci_runs):
        price, *_ = price_once(n_ci, seed + 1000 * (run_idx + 1))
        ci_prices.append(price)
    ci_prices = np.array(ci_prices)
    mean_price = float(np.mean(ci_prices))
    sem = float(stats.sem(ci_prices)) if len(ci_prices) > 1 else 0.0
    t_crit = float(stats.t.ppf(0.995, df=max(len(ci_prices) - 1, 1)))
    ci_low = mean_price - t_crit * sem
    ci_high = mean_price + t_crit * sem

    fig, ax = plt.subplots(figsize=(6, 4))
    ns = [c["n"] for c in convergence]
    prices = [c["kernel_price"] for c in convergence]
    ax.plot(ns, prices, "o-", label="kernel price")
    ax.axhline(fft_price, color="k", ls="--", label=f"FFT benchmark = {fft_price:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("number of collocation points n")
    ax.set_ylabel("ATM call price")
    ax.set_title("Exp. 7: Variance Gamma convergence")
    ax.legend()
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / "exp7_convergence.png", dpi=150)
    plt.close(fig)

    metrics = {
        "experiment": "exp7_variance_gamma_quadrature",
        "full": full,
        "lambda": config.EXP7_LAMBDA,
        "fft_benchmark_price": fft_price,
        "fft_benchmark_price_paper": config.EXP7_FFT_BENCHMARK_PRICE,
        "convergence": convergence,
        "n_ci": int(n_ci),
        "n_ci_runs": n_ci_runs,
        "ci_mean_price": mean_price,
        "ci_99_low": ci_low,
        "ci_99_high": ci_high,
        "random_seed": seed,
    }
    with open(RESULTS_DIR / "exp7_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


if __name__ == "__main__":
    import sys

    print(json.dumps(run(full="--full" in sys.argv), indent=2))
