"""Experiment 1: first-order ODE with Gaussian kernel (paper Sec. 4.1, p.16).

Solves u'(x) = -pi sin(pi x) on (-1,1) with the overdetermined boundary
condition u(-1) = u(1) = -1 using the kernel solver of Algorithm 1, and
compares the kernel solution with the exact solution u(x) = cos(pi x).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import equally_spaced_points, uniform_points
from src import config, jax_utils as ju, kernel_solver as ks

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def gaussian_kernel(gamma: float):
    def k(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        d = x - y
        return jnp.exp(-gamma / 2.0 * jnp.sum(d * d))

    return k


def build_blocks(x1: np.ndarray, x2: np.ndarray, gamma: float):
    """Pairwise operator-induced kernel blocks for A1=d/dx, A2=identity.

    k^{1,1}(x,y) = d_x d_y k(x,y) (Eq. 25, 28); verified against the paper's
    Example 2.5 formula k_x^{d11}(y) = (gamma^2(x-y)^2 - gamma) k(x,y) (Eq.
    10) up to sign: for a difference kernel k(x,y)=g(x-y), d_x d_y g(x-y) =
    -g''(x-y), so k^{1,1} = -k_x^{d11} = (gamma - gamma^2(x-y)^2) k(x,y),
    which matches a direct finite-difference check of the mixed derivative.
    """
    k = gaussian_kernel(gamma)
    deriv = ju.partial_operator(0)
    ident = ju.identity_operator

    k11 = ju.cross_kernel(k, deriv, deriv)
    k12 = ju.cross_kernel(k, deriv, ident)
    k21 = ju.cross_kernel(k, ident, deriv)
    k22 = ju.cross_kernel(k, ident, ident)

    x1v, x2v = ju.as_points(x1), ju.as_points(x2)
    return {
        (0, 0): ju.build_matrix(k11, x1v, x1v),
        (0, 1): ju.build_matrix(k12, x1v, x2v),
        (1, 0): ju.build_matrix(k21, x2v, x1v),
        (1, 1): ju.build_matrix(k22, x2v, x2v),
    }, (k11, k12, k21, k22)


def exact_solution(x: np.ndarray) -> np.ndarray:
    return np.cos(np.pi * x)


def run(seed: int = config.RANDOM_SEED) -> dict:
    rng = np.random.default_rng(seed)
    x1 = uniform_points(config.EXP1_N1, -1.0, 1.0, rng)
    x2 = np.array([-1.0, 1.0])

    f1 = -np.pi * np.sin(np.pi * x1)
    f2 = np.full(config.EXP1_N2, -1.0)

    k_blocks, (k11, k12, k21, k22) = build_blocks(x1, x2, config.EXP1_GAMMA)
    solution = ks.solve(
        k_blocks, [f1, f2], [config.EXP1_W1, config.EXP1_W2], config.EXP1_LAMBDA
    )

    x_query = equally_spaced_points(config.EXP1_N_QUERY, -1.0, 1.0)
    xq = ju.as_points(x_query)
    x1v, x2v = ju.as_points(x1), ju.as_points(x2)
    # Query evaluation uses k_{x_{m,n}}^{A_m}(x_query) (Eq. 30): the operator
    # A_m acts on the *training* argument x_{m,n}, and the representer is
    # evaluated, un-differentiated, at the query point.
    k = gaussian_kernel(config.EXP1_GAMMA)
    kq_block1 = ju.build_matrix(ju.representer(k, ju.partial_operator(0)), x1v, xq)
    kq_block2 = ju.build_matrix(ju.representer(k, ju.identity_operator), x2v, xq)
    h_hat = solution.predict([kq_block1, kq_block2])

    u_exact = exact_solution(x_query)
    rel_l2_error = float(np.linalg.norm(h_hat - u_exact) / np.linalg.norm(u_exact))
    max_abs_error = float(np.max(np.abs(h_hat - u_exact)))

    objective = ks.objective_value(
        k_blocks, [f1, f2], [config.EXP1_W1, config.EXP1_W2], config.EXP1_LAMBDA, solution
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_query, u_exact, label="exact: cos(pi x)", lw=2)
    ax.plot(x_query, h_hat, "--", label="kernel solution", lw=2)
    ax.scatter(x1, f1 * 0 + ax.get_ylim()[0], marker="|", s=10, alpha=0.3,
               label="interior collocation points")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title("Exp. 1: first-order ODE, Gaussian kernel")
    ax.legend()
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / "exp1_solution.png", dpi=150)
    plt.close(fig)

    metrics = {
        "experiment": "exp1_first_order_ode_gaussian_kernel",
        "gamma": config.EXP1_GAMMA,
        "lambda": config.EXP1_LAMBDA,
        "n1": config.EXP1_N1,
        "n2": config.EXP1_N2,
        "relative_l2_error": rel_l2_error,
        "max_abs_error": max_abs_error,
        "objective_value": objective,
        "random_seed": seed,
    }
    with open(RESULTS_DIR / "exp1_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
