"""Experiment 3: one-dimensional heat equation with a parabolic kernel
(paper Sec. 4.2.1, p.18).

Solves (d/dt - 1/2 d^2/dx^2) u = 0 on (0,T) x R with initial condition
u(0,x) = phi_{1/2}(x) (the N(0, 1/2) density), using the forward-warped
parabolic kernel of ``src.parabolic_kernel`` and operator-induced kernels
computed by automatic differentiation (``src.jax_utils``, following Sec. 3.3).
Compared with the exact solution u(t,x) = phi_{1/2+t}(x).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from data.generate_synthetic import equally_spaced_points, uniform_box_points
from src import config, jax_utils as ju, kernel_solver as ks, parabolic_kernel as pk

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def phi(t_var: np.ndarray, x: np.ndarray) -> np.ndarray:
    """phi_t(x): N(0, t) density, paper Sec. 4.2.1."""
    return norm.pdf(x, loc=0.0, scale=np.sqrt(t_var))


def exact_solution(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """u(t,x) = phi_{1/2+t}(x), paper Sec. 4.2.1."""
    return phi(config.EXP3_SIGMA0 ** 2 + t, x)


def build_operators():
    """A1 = d/dt - 1/2 d^2/dx^2 (heat operator), A2 = restriction (identity)."""
    dt = ju.partial_operator(0)
    dxx = ju.second_partial_operator(1, 1)
    a1 = ju.combine_operators((1.0, dt), (-0.5, dxx))
    a2 = ju.identity_operator
    return a1, a2


def run(seed: int = config.RANDOM_SEED) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    kernel = pk.make_forward_parabolic_kernel(config.EXP3_GAMMA0, config.EXP3_GAMMA1, config.EXP3_EPS_WARP)
    a1, a2 = build_operators()

    z1 = uniform_box_points(config.EXP3_N1, np.array([0.0, config.EXP3_X_MIN]),
                             np.array([config.EXP3_T, config.EXP3_X_MAX]), rng)
    x2_space = equally_spaced_points(config.EXP3_N2, config.EXP3_X_MIN, config.EXP3_X_MAX)
    z2 = np.column_stack([np.zeros_like(x2_space), x2_space])

    k11 = ju.build_matrix(ju.cross_kernel(kernel, a1, a1), z1, z1)
    k12 = ju.build_matrix(ju.cross_kernel(kernel, a1, a2), z1, z2)
    k21 = ju.build_matrix(ju.cross_kernel(kernel, a2, a1), z2, z1)
    k22 = ju.build_matrix(ju.cross_kernel(kernel, a2, a2), z2, z2)
    k_blocks = {(0, 0): k11, (0, 1): k12, (1, 0): k21, (1, 1): k22}

    f1 = np.zeros(config.EXP3_N1)
    f2 = phi(config.EXP3_SIGMA0 ** 2, x2_space)

    solution = ks.solve(k_blocks, [f1, f2], [config.EXP3_W1, config.EXP3_W2], config.EXP3_LAMBDA)
    objective = ks.objective_value(k_blocks, [f1, f2], [config.EXP3_W1, config.EXP3_W2], config.EXP3_LAMBDA, solution)

    t_grid = np.linspace(0.0, config.EXP3_T, config.EXP3_GRID_NT)
    x_grid = np.linspace(config.EXP3_X_MIN, config.EXP3_X_MAX, config.EXP3_GRID_NX)
    tt, xx = np.meshgrid(t_grid, x_grid, indexing="ij")
    z_query = np.column_stack([tt.ravel(), xx.ravel()])

    kq1 = ju.build_matrix(ju.representer(kernel, a1), z1, z_query)
    kq2 = ju.build_matrix(ju.representer(kernel, a2), z2, z_query)
    h_hat = solution.predict([kq1, kq2]).reshape(tt.shape)
    u_exact = exact_solution(tt, xx)

    rel_l2 = float(np.linalg.norm(h_hat - u_exact) / np.linalg.norm(u_exact))
    max_abs = float(np.max(np.abs(h_hat - u_exact)))

    metrics = {
        "experiment": "exp3_heat_equation_parabolic_kernel",
        "lambda": config.EXP3_LAMBDA, "n1": config.EXP3_N1, "n2": config.EXP3_N2,
        "gamma0": config.EXP3_GAMMA0, "gamma1": config.EXP3_GAMMA1, "eps_warp": config.EXP3_EPS_WARP,
        "relative_l2_error": rel_l2, "max_abs_error": max_abs, "objective_value": objective,
        "runtime_seconds": time.time() - t0,
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].pcolormesh(x_grid, t_grid, u_exact, shading="auto")
    axes[0].set_title("exact u(t,x)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].pcolormesh(x_grid, t_grid, h_hat, shading="auto")
    axes[1].set_title("kernel solution")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("t")
    fig.colorbar(im1, ax=axes[1])
    axes[2].plot(x_grid, u_exact[-1], label="exact, t=T", lw=2)
    axes[2].plot(x_grid, h_hat[-1], "--", label="kernel, t=T", lw=2)
    axes[2].plot(x_grid, u_exact[0], label="exact, t=0", lw=1, alpha=0.6)
    axes[2].plot(x_grid, h_hat[0], "--", label="kernel, t=0", lw=1, alpha=0.6)
    axes[2].set_xlabel("x"); axes[2].set_ylabel("u"); axes[2].legend()
    fig.suptitle("Exp. 3: 1D heat equation, parabolic kernel")
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / "exp3_solution.png", dpi=150)
    plt.close(fig)

    with open(RESULTS_DIR / "exp3_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
