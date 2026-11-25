"""Generic kernel solver for systems of linear operator equations.

Implements the training/prediction steps of Algorithm 1 and the Nystrom
reduction of Section 3.5 in Filipovic & Tissot-Daguette, "A Kernel Framework
for Linear Operator Equations". This module is agnostic to the kernel and
the operators: callers supply the pairwise operator-induced kernel blocks
(Eq. 28) and forcing vectors, and this module assembles and solves the
block linear system (Eq. 29) or its Nystrom-reduced counterpart (Eq. 39).

Custom implementation (paper Algorithm 1, Eq. 28-30, 37-39): no library
implements this regularized operator-equation collocation scheme; portfolio,
backtesting and statistical-learning libraries available in this environment
(skfolio, vectorbt, arch, hmmlearn, ruptures, copulae, statsmodels) do not
cover linear-operator-equation collocation with Riesz representers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class KernelSolution:
    """Coefficients of the kernel solution (Eq. 29-30)."""

    betas: list[np.ndarray]          # beta_m, m = 1..M
    n: list[int]                      # n_m, number of collocation points per block
    w: list[float]                    # w_m, objective weights per block

    def predict(self, k_query_blocks: Sequence[np.ndarray]) -> np.ndarray:
        """Evaluate h_lambda(x) = sum_m sum_n beta_{m,n} k_{x_{m,n}}^{A_m}(x).

        ``k_query_blocks[m]`` has shape (n_m, n_query): entry (n, q) is
        ``A_m k(x_{m,n}, x_query_q)`` (Eq. 30).
        """
        assert len(k_query_blocks) == len(self.betas)
        total = None
        for beta_m, k_m in zip(self.betas, k_query_blocks):
            contribution = beta_m @ k_m
            total = contribution if total is None else total + contribution
        return total


def assemble_block_matrix(k_blocks: dict[tuple[int, int], np.ndarray],
                           n: Sequence[int]) -> np.ndarray:
    """Assemble the full (N x N) matrix AA* from pairwise blocks (Eq. 28).

    ``k_blocks[(l, m)]`` is the (n_l x n_m) matrix of ``k^{l,m}`` evaluations.
    """
    M = len(n)
    rows = []
    for l in range(M):
        row_blocks = [k_blocks[(l, m)] for m in range(M)]
        rows.append(np.concatenate(row_blocks, axis=1))
    full = np.concatenate(rows, axis=0)
    N = sum(n)
    assert full.shape == (N, N), f"expected ({N},{N}), got {full.shape}"
    return full


def solve(k_blocks: dict[tuple[int, int], np.ndarray],
          y: Sequence[np.ndarray],
          w: Sequence[float],
          lam: float) -> KernelSolution:
    """Solve the regularized least-squares normal equation (Eq. 29).

    (AA* + lambda * G) beta = y,  G = diag(n_1/w_1 I, ..., n_M/w_M I).
    """
    n = [len(y_m) for y_m in y]
    N = sum(n)
    AAstar = assemble_block_matrix(k_blocks, n)
    g_diag = np.concatenate([np.full(n_m, n_m / w_m) for n_m, w_m in zip(n, w)])
    system = AAstar + lam * np.diag(g_diag)
    y_full = np.concatenate([np.asarray(y_m, dtype=float) for y_m in y])
    beta_full = np.linalg.solve(system, y_full)
    betas = []
    offset = 0
    for n_m in n:
        betas.append(beta_full[offset:offset + n_m])
        offset += n_m
    return KernelSolution(betas=betas, n=n, w=list(w))


def objective_value(k_blocks: dict[tuple[int, int], np.ndarray],
                     y: Sequence[np.ndarray],
                     w: Sequence[float],
                     lam: float,
                     solution: KernelSolution) -> float:
    """Empirical regularized objective Phi_lambda(h) at the kernel solution.

    Phi_lambda(h) = sum_m (w_m/n_m) ||A_m h - f_m||_2^2 + lambda ||h||_H^2,
    with ||h||_H^2 = beta^T (AA*) beta for h = sum beta_{m,n} k^{A_m}_{x_{m,n}}.
    """
    n = solution.n
    AAstar = assemble_block_matrix(k_blocks, n)
    beta_full = np.concatenate(solution.betas)
    residual_term = 0.0
    offset = 0
    pred_full = AAstar @ beta_full
    for n_m, w_m, y_m in zip(n, w, y):
        pred_m = pred_full[offset:offset + n_m]
        residual_term += (w_m / n_m) * np.sum((pred_m - np.asarray(y_m)) ** 2)
        offset += n_m
    norm_term = beta_full @ AAstar @ beta_full
    return float(residual_term + lam * norm_term)


@dataclass
class NystromSolution:
    """Coefficients of the Nystrom-reduced solution (Eq. 38-39)."""

    alpha: np.ndarray

    def predict(self, phi_query: np.ndarray) -> np.ndarray:
        """Evaluate h_{lambda,r}(x) = sum_j alpha_j varphi_j(x).

        ``phi_query`` has shape (r, n_query): entry (j, q) is
        ``A_{m(j)} k(x_{m(j), n(j)}, x_query_q)`` for the j-th selected
        representer.
        """
        return self.alpha @ phi_query


def solve_nystrom(z_blocks: Sequence[np.ndarray],
                   k_r: np.ndarray,
                   y: Sequence[np.ndarray],
                   w: Sequence[float],
                   lam: float) -> NystromSolution:
    """Solve the Nystrom-reduced normal equation (Eq. 39).

    (sum_m (w_m/n_m) Z_m^T Z_m + lambda K_r) alpha
        = sum_m (w_m/n_m) Z_m^T y_m.

    ``z_blocks[m]`` has shape (n_m, r): entries ``A_m varphi_j(x_{m,n})``.
    ``k_r`` has shape (r, r): the Gram matrix of the selected representers.
    """
    r = k_r.shape[0]
    lhs = lam * k_r.copy()
    rhs = np.zeros(r)
    for z_m, w_m, y_m in zip(z_blocks, w, y):
        n_m = z_m.shape[0]
        lhs += (w_m / n_m) * (z_m.T @ z_m)
        rhs += (w_m / n_m) * (z_m.T @ np.asarray(y_m))
    alpha, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return NystromSolution(alpha=alpha)


def nystrom_objective_value(z_blocks: Sequence[np.ndarray],
                             k_r: np.ndarray,
                             y: Sequence[np.ndarray],
                             w: Sequence[float],
                             lam: float,
                             solution: NystromSolution) -> float:
    """Empirical regularized objective Phi_lambda(h_{lambda,r}) (Eq. 38, 40)."""
    alpha = solution.alpha
    residual_term = 0.0
    for z_m, w_m, y_m in zip(z_blocks, w, y):
        n_m = z_m.shape[0]
        pred_m = z_m @ alpha
        residual_term += (w_m / n_m) * np.sum((pred_m - np.asarray(y_m)) ** 2)
    norm_term = alpha @ k_r @ alpha
    return float(residual_term + lam * norm_term)


def select_nystrom_indices(n_list: Sequence[int],
                            r_list: Sequence[int],
                            rng: np.random.Generator) -> list[np.ndarray]:
    """Draw nested index sets J_m of size r_m from {0,...,n_m-1} (Sec. 3.5).

    One random permutation is drawn per block and reused as nested prefixes
    across different r_m values, so that index sets are nested in r as the
    paper specifies ("nested in r, prefixes of one random permutation").
    """
    perms = [rng.permutation(n_m) for n_m in n_list]
    return [perm[:r_m] for perm, r_m in zip(perms, r_list)]


def build_nystrom_matrices(k_fns: dict[tuple[int, int], Callable],
                            z_list: Sequence[np.ndarray],
                            idx_list: Sequence[np.ndarray],
                            build_matrix_fn: Callable) -> tuple[list[np.ndarray], np.ndarray]:
    """Assemble the Nystrom matrices Z_m (Eq. 37) and K_r (Eq. 39) from the
    pairwise operator-induced kernel functions ``k_fns[(l, m)](x, y)``.

    ``z_list[m]`` holds all n_m collocation points of block m; ``idx_list[m]``
    the indices of the selected representers within block m (Sec. 3.5). The
    selected representers are ordered by block then index. ``build_matrix_fn``
    vectorizes a pairwise kernel function over point arrays, matching the
    signature of ``src.jax_utils.build_matrix``.

    Returns ``z_blocks`` with ``z_blocks[m]`` of shape (n_m, r), entry (n, j)
    = A_m varphi_j(x_{m,n}), and ``k_r`` of shape (r, r), the Gram matrix of
    the selected representers.
    """
    M = len(z_list)
    rep_points = [z_list[m][idx_list[m]] for m in range(M)]
    r_sizes = [len(idx) for idx in idx_list]

    k_r_rows = []
    for l in range(M):
        row_blocks = [build_matrix_fn(k_fns[(l, m)], rep_points[l], rep_points[m]) for m in range(M)]
        k_r_rows.append(np.concatenate(row_blocks, axis=1))
    k_r = np.concatenate(k_r_rows, axis=0)
    r = sum(r_sizes)
    assert k_r.shape == (r, r), f"expected ({r},{r}), got {k_r.shape}"

    z_blocks = []
    for m in range(M):
        col_blocks = [build_matrix_fn(k_fns[(m, mp)], z_list[m], rep_points[mp]) for mp in range(M)]
        z_blocks.append(np.concatenate(col_blocks, axis=1))
    return z_blocks, k_r
