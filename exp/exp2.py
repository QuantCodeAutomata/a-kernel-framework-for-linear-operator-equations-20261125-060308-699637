"""Experiment 2: second-order ODE with truncated Fourier kernel (paper
Sec. 4.1, p.17).

Solves u'' + zeta_n u = 0 on (0,1), u(0)=0, u(1)=1, for n = 1, 2, 5 using the
closed-form termwise cross-kernels of the truncated Fourier kernel k_L
(Example 2.6), and compares with the exact solution u_n(x) = sin(sqrt(zeta_n) x).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import equally_spaced_points, uniform_points
from src import config, fourier_kernel as fk, kernel_solver as ks

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def zeta(n: int) -> float:
    """zeta_n = (2n + 1/2)^2 pi^2, paper Sec. 4.1."""
    return (2 * n + 0.5) ** 2 * np.pi ** 2


def exact_solution(x: np.ndarray, n: int) -> np.ndarray:
    return np.sin(np.sqrt(zeta(n)) * x)


def solve_one(n_freq: int, x1: np.ndarray, x2: np.ndarray, modes, coeffs) -> ks.KernelSolution:
    z = zeta(n_freq)
    mult_a1 = fk.second_derivative_plus_const_multiplier(modes, z)  # A1 = d^2/dx^2 + zeta_n
    mult_a2 = fk.identity_multiplier(modes)                          # A2 = restriction

    k11 = fk.cross_kernel_matrix(x1, x1, modes, coeffs, mult_a1, mult_a1)
    k12 = fk.cross_kernel_matrix(x1, x2, modes, coeffs, mult_a1, mult_a2)
    k21 = fk.cross_kernel_matrix(x2, x1, modes, coeffs, mult_a2, mult_a1)
    k22 = fk.cross_kernel_matrix(x2, x2, modes, coeffs, mult_a2, mult_a2)
    k_blocks = {(0, 0): k11, (0, 1): k12, (1, 0): k21, (1, 1): k22}

    f1 = np.zeros_like(x1)
    f2 = np.array([0.0, 1.0])  # u(0)=0, u(1)=1 (Sec. 4.1: boundary data used in this replication, see RESULTS.md)

    return ks.solve(k_blocks, [f1, f2], [config.EXP2_W1, config.EXP2_W2], config.EXP2_LAMBDA), (f1, f2), mult_a1, mult_a2


def predict(solution: ks.KernelSolution, x1: np.ndarray, x2: np.ndarray, x_query: np.ndarray,
            modes, coeffs, mult_a1, mult_a2) -> np.ndarray:
    """Query representers k_{x_{m,n}}^{A_m}(x_query) = A_m acting on the first
    (training) argument of the Fourier kernel, identity on the query argument
    (Algorithm 1 prediction step), using the same termwise closed form."""
    identity = fk.identity_multiplier(modes)
    kq1 = fk.cross_kernel_matrix(x1, x_query, modes, coeffs, mult_a1, identity)
    kq2 = fk.cross_kernel_matrix(x2, x_query, modes, coeffs, mult_a2, identity)
    return solution.predict([kq1, kq2])


def run(seed: int = config.RANDOM_SEED) -> dict:
    rng = np.random.default_rng(seed)
    x1 = equally_spaced_points(config.EXP2_N1, 0.01, 0.99)
    x2 = np.array([0.0, 1.0])
    x_query = equally_spaced_points(config.EXP2_N_QUERY, 0.0, 1.0)

    modes, coeffs = fk.mode_coefficients(config.EXP2_L)

    metrics = {"experiment": "exp2_second_order_ode_fourier_kernel", "lambda": config.EXP2_LAMBDA,
               "L": config.EXP2_L, "n1": config.EXP2_N1, "n2": config.EXP2_N2, "per_n": {}}

    fig, axes = plt.subplots(1, len(config.EXP2_N_VALUES), figsize=(4 * len(config.EXP2_N_VALUES), 4), sharey=True)
    for ax, n_freq in zip(np.atleast_1d(axes), config.EXP2_N_VALUES):
        solution, (f1, f2), mult_a1, mult_a2 = solve_one(n_freq, x1, x2, modes, coeffs)
        h_hat = predict(solution, x1, x2, x_query, modes, coeffs, mult_a1, mult_a2)
        u_exact = exact_solution(x_query, n_freq)

        rel_l2 = float(np.linalg.norm(h_hat - u_exact) / np.linalg.norm(u_exact))
        max_abs = float(np.max(np.abs(h_hat - u_exact)))
        metrics["per_n"][str(n_freq)] = {
            "zeta_n": float(zeta(n_freq)),
            "relative_l2_error": rel_l2,
            "max_abs_error": max_abs,
        }

        ax.plot(x_query, u_exact, label="exact", lw=2)
        ax.plot(x_query, h_hat, "--", label="kernel solution", lw=2)
        ax.set_title(f"n = {n_freq}")
        ax.set_xlabel("x")
        ax.legend()
    axes_list = np.atleast_1d(axes)
    axes_list[0].set_ylabel("u(x)")
    fig.suptitle("Exp. 2: second-order ODE, truncated Fourier kernel")
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / "exp2_solution.png", dpi=150)
    plt.close(fig)

    with open(RESULTS_DIR / "exp2_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
