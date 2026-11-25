"""Experiment 4: Bachelier option pricing PDE with sensitivities (paper
Sec. 4.2.2, p.18-19).

Solves the backward heat equation d/dt u + 1/2 sigma^2 d^2/dx^2 u = 0 on
(0,T) x R with terminal condition u(T,x) = (x - x0)^+ (Bachelier call),
using the backward-warped parabolic kernel of ``src.parabolic_kernel``.
Delta and Gamma are obtained by differentiating the kernel representers
with respect to the query point (Sec. 4.2.2): since A_m already acts on the
training argument of the representer, Delta/Gamma require the cross-kernel
of A_m (training argument) with a first/second derivative operator on the
query argument, both produced generically by ``src.jax_utils.cross_kernel``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import uniform_box_points
from src import bachelier, config, jax_utils as ju, kernel_solver as ks, parabolic_kernel as pk

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_operators(sigma: float):
    """A1 = d/dt + 1/2 sigma^2 d^2/dx^2 (Bachelier pricing operator), A2 = restriction."""
    dt = ju.partial_operator(0)
    dxx = ju.second_partial_operator(1, 1)
    a1 = ju.combine_operators((1.0, dt), (0.5 * sigma ** 2, dxx))
    a2 = ju.identity_operator
    return a1, a2


def run(seed: int = config.RANDOM_SEED) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    T, sigma, x0 = config.EXP4_T, config.EXP4_SIGMA, config.EXP4_X0
    kernel = pk.make_backward_parabolic_kernel(config.EXP4_GAMMA0, config.EXP4_GAMMA1, config.EXP4_EPS_WARP, T)
    a1, a2 = build_operators(sigma)
    dx_query = ju.partial_operator(1)
    dxx_query = ju.second_partial_operator(1, 1)

    z1 = uniform_box_points(config.EXP4_N1, np.array([0.0, config.EXP4_X_MIN]),
                             np.array([T, config.EXP4_X_MAX]), rng)
    x2_space = rng.uniform(config.EXP4_X_MIN, config.EXP4_X_MAX, size=config.EXP4_N2)
    z2 = np.column_stack([np.full_like(x2_space, T), x2_space])

    k11 = ju.build_matrix(ju.cross_kernel(kernel, a1, a1), z1, z1)
    k12 = ju.build_matrix(ju.cross_kernel(kernel, a1, a2), z1, z2)
    k21 = ju.build_matrix(ju.cross_kernel(kernel, a2, a1), z2, z1)
    k22 = ju.build_matrix(ju.cross_kernel(kernel, a2, a2), z2, z2)
    k_blocks = {(0, 0): k11, (0, 1): k12, (1, 0): k21, (1, 1): k22}

    f1 = np.zeros(config.EXP4_N1)
    f2 = np.maximum(x2_space - x0, 0.0)
    w = [config.EXP4_W1, config.EXP4_W2]
    solution = ks.solve(k_blocks, [f1, f2], w, config.EXP4_LAMBDA)
    objective = ks.objective_value(k_blocks, [f1, f2], w, config.EXP4_LAMBDA, solution)

    t_slices = np.linspace(0.0, T, config.EXP4_GRID_NT)
    x_grid = np.linspace(config.EXP4_X_MIN, config.EXP4_X_MAX, config.EXP4_GRID_NX)
    metrics = {
        "experiment": "exp4_bachelier_greeks",
        "lambda": config.EXP4_LAMBDA, "n1": config.EXP4_N1, "n2": config.EXP4_N2,
        "gamma0": config.EXP4_GAMMA0, "gamma1": config.EXP4_GAMMA1, "eps_warp": config.EXP4_EPS_WARP,
        "objective_value": objective, "per_time_slice": [],
    }

    fig, axes = plt.subplots(3, 1, figsize=(7, 11), sharex=True)
    labels = ["price", "Delta", "Gamma"]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(t_slices)))
    for t_val, color in zip(t_slices, colors):
        if np.isclose(t_val, T):
            t_val = T - 1e-4  # Bachelier formulas are singular exactly at maturity
        z_query = np.column_stack([np.full_like(x_grid, t_val), x_grid])

        kq1 = ju.build_matrix(ju.representer(kernel, a1), z1, z_query)
        kq2 = ju.build_matrix(ju.representer(kernel, a2), z2, z_query)
        price_hat = solution.predict([kq1, kq2])

        dq1 = ju.build_matrix(ju.cross_kernel(kernel, a1, dx_query), z1, z_query)
        dq2 = ju.build_matrix(ju.cross_kernel(kernel, a2, dx_query), z2, z_query)
        delta_hat = solution.predict([dq1, dq2])

        gq1 = ju.build_matrix(ju.cross_kernel(kernel, a1, dxx_query), z1, z_query)
        gq2 = ju.build_matrix(ju.cross_kernel(kernel, a2, dxx_query), z2, z_query)
        gamma_hat = solution.predict([gq1, gq2])

        price_exact = bachelier.price(t_val, x_grid, x0, sigma, T)
        delta_exact = bachelier.delta(t_val, x_grid, x0, sigma, T)
        gamma_exact = bachelier.gamma(t_val, x_grid, x0, sigma, T)

        metrics["per_time_slice"].append({
            "t": float(t_val),
            "price_rel_l2_error": float(np.linalg.norm(price_hat - price_exact) / np.linalg.norm(price_exact)),
            "delta_max_abs_error": float(np.max(np.abs(delta_hat - delta_exact))),
            "gamma_max_abs_error": float(np.max(np.abs(gamma_hat - gamma_exact))),
        })

        for ax, hat, exact in zip(axes, (price_hat, delta_hat, gamma_hat), (price_exact, delta_exact, gamma_exact)):
            ax.plot(x_grid, exact, color=color, lw=1.5)
            ax.plot(x_grid, hat, "--", color=color, lw=1.0)

    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
    axes[-1].set_xlabel("x")
    fig.suptitle("Exp. 4: Bachelier price, Delta, Gamma (solid = exact, dashed = kernel,\ncolor = time to maturity)")
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / "exp4_greeks.png", dpi=150)
    plt.close(fig)

    metrics["runtime_seconds"] = time.time() - t0
    with open(RESULTS_DIR / "exp4_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
