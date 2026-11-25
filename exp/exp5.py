"""Experiment 5: multi-asset Black-Scholes max-call with Nystrom approximation
(paper Sec. 4.2.3, p.19-20).

For each number of underlying assets d in {2, 3, 5}, solves the
multi-dimensional Black-Scholes pricing PDE for a max-call option with
``src.black_scholes`` operators and the scaled backward parabolic kernel of
``src.parabolic_kernel``, using the Nystrom-reduced normal equation of
``src.kernel_solver`` (Eq. 39) so that only O(n*r) kernel evaluations are
needed instead of the full O(n^2) block matrix. For each Nystrom size r the
reduced solution is compared with a Monte Carlo benchmark
(``src.black_scholes.mc_max_call_price``) and the regularized objective
Phi_lambda(h_{lambda,r}) is recorded to check its O(r^-1) decay (Sec. 3.5).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import sobol_lognormal_points, sobol_time_lognormal_points
from src import black_scholes as bs, config, jax_utils as ju, kernel_solver as ks, parabolic_kernel as pk


def build_collocation(d: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interior and terminal collocation points (Sec. 4.2.3, data requirements):

    "map to uniform on (0,T) in time, and to correlated lognormal
    distribution S0*exp(sigma*sqrt(T)*Z) ... in space", Z ~ N(0, rho).
    """
    T, S0, sigma = config.EXP5_T, config.EXP5_S0, config.EXP5_SIGMA
    rho = bs.correlation_matrix(d, config.EXP5_RHO)
    mean_log = np.full(d, np.log(S0))
    cov_log = sigma ** 2 * T * rho
    t1, x1 = sobol_time_lognormal_points(n, T, mean_log, cov_log, seed)
    z1 = np.column_stack([t1, x1])
    x2 = sobol_lognormal_points(n, d, mean_log, cov_log, seed + 1)
    z2 = np.column_stack([np.full(n, T), x2])
    return z1, z2, x2


def run_dimension(d: int, n_run: int, r_grid: tuple[int, ...], mc_paths: int,
                   seed: int) -> dict:
    """Run the Nystrom-reduced kernel solver for a fixed dimension d, across
    every Nystrom size in ``r_grid``, and compare with the Monte Carlo
    benchmark (Sec. 4.2.3)."""
    rng = np.random.default_rng(seed)
    T, S0, K, sigma = config.EXP5_T, config.EXP5_S0, config.EXP5_K, config.EXP5_SIGMA
    rho = bs.correlation_matrix(d, config.EXP5_RHO)

    kernel = pk.make_backward_parabolic_kernel_scaled(
        config.EXP5_GAMMA0, config.EXP5_GAMMA1, config.EXP5_EPS_WARP, T,
        shift=S0, scale=S0 * sigma)
    a1 = bs.bs_pde_operator(sigma, rho, d)
    a2 = ju.identity_operator
    k_fns = {
        (0, 0): ju.cross_kernel(kernel, a1, a1),
        (0, 1): ju.cross_kernel(kernel, a1, a2),
        (1, 0): ju.cross_kernel(kernel, a2, a1),
        (1, 1): ju.cross_kernel(kernel, a2, a2),
    }

    z1, z2, x2 = build_collocation(d, n_run, seed)
    f1 = np.zeros(n_run)
    payoff_fn = jax_vmapped_payoff(K, config.EXP5_EPS_PAYOFF)
    f2 = np.asarray(payoff_fn(x2))
    w = [config.EXP5_W1, config.EXP5_W2]

    mc_price, mc_se = bs.mc_max_call_price(S0, K, T, sigma, rho, d, mc_paths, seed + 1000)

    results = {"d": d, "mc_price": mc_price, "mc_std_error": mc_se, "per_r": []}
    for r in r_grid:
        r2 = int(np.floor(config.EXP5_R_SELECTED_FRACTION_TERMINAL * r))
        r1 = r - r2
        idx_list = ks.select_nystrom_indices([n_run, n_run], [r1, r2], rng)
        z_blocks, k_r = ks.build_nystrom_matrices(k_fns, [z1, z2], idx_list, ju.build_matrix)
        solution = ks.solve_nystrom(z_blocks, k_r, [f1, f2], w, config.EXP5_LAMBDA)
        objective = ks.nystrom_objective_value(z_blocks, k_r, [f1, f2], w, config.EXP5_LAMBDA, solution)

        rep_points = [z1[idx_list[0]], z2[idx_list[1]]]
        query = np.concatenate([[0.0], np.full(d, S0)]).reshape(1, -1)
        phi_q1 = ju.build_matrix(ju.representer(kernel, a1), rep_points[0], query)
        phi_q2 = ju.build_matrix(ju.representer(kernel, a2), rep_points[1], query)
        phi_query = np.concatenate([phi_q1, phi_q2], axis=0)
        price_hat = float(solution.predict(phi_query)[0])

        results["per_r"].append({
            "r": r, "r1": r1, "r2": r2,
            "objective": objective,
            "price_hat_atm": price_hat,
            "abs_error_vs_mc": abs(price_hat - mc_price),
        })
    return results


def jax_vmapped_payoff(K: float, eps: float):
    """Vectorized smoothed max-call payoff g_eps(x) over a batch of points x."""
    import jax

    def single(x):
        return bs.smoothed_max_call_payoff(x, K, eps)

    return jax.vmap(single)


def run(seed: int = config.RANDOM_SEED, full: bool = False) -> dict:
    t0 = time.time()
    n_run = config.EXP5_N_PAPER if full else config.EXP5_N_RUN
    r_grid = config.EXP5_R_GRID_PAPER if full else config.EXP5_R_GRID_RUN
    mc_paths = config.EXP5_MC_PATHS_PAPER if full else config.EXP5_MC_PATHS_RUN

    metrics = {
        "experiment": "exp5_max_call_nystrom",
        "n_run": n_run, "r_grid": list(r_grid), "mc_paths": mc_paths, "full": full,
        "lambda": config.EXP5_LAMBDA, "dims": [],
    }
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for d in config.EXP5_DIMS:
        dim_result = run_dimension(d, n_run, r_grid, mc_paths, seed)
        metrics["dims"].append(dim_result)
        rs = [p["r"] for p in dim_result["per_r"]]
        objs = [p["objective"] for p in dim_result["per_r"]]
        ax.loglog(rs, objs, "o-", label=f"d={d}")

    r_ref = np.array([metrics["dims"][0]["per_r"][0]["r"], metrics["dims"][0]["per_r"][-1]["r"]], dtype=float)
    ref_level = metrics["dims"][0]["per_r"][0]["objective"] * (r_ref[0] / r_ref)
    ax.loglog(r_ref, ref_level, "k--", label=r"$O(r^{-1})$ reference")
    ax.set_xlabel("Nystrom size r")
    ax.set_ylabel(r"$\Phi_\lambda(\hat h_{\lambda,r})$")
    ax.set_title("Exp. 5: Nystrom objective convergence, max-call option")
    ax.legend()
    fig.tight_layout()

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    fig.savefig(results_dir / "exp5_nystrom_convergence.png", dpi=150)
    plt.close(fig)

    metrics["runtime_seconds"] = time.time() - t0
    with open(results_dir / "exp5_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


if __name__ == "__main__":
    import sys

    print(json.dumps(run(full="--full" in sys.argv), indent=2))
