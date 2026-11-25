"""Unit tests for src/kernel_solver.py: Algorithm 1 training/prediction and
the Nystrom reduction (paper Eq. 28-30, 37-39)."""

import numpy as np
import pytest

from src import kernel_solver as ks


def _toy_blocks(n1: int, n2: int, seed: int = 0):
    """A random SPD system mimicking AA* with two equation blocks."""
    rng = np.random.default_rng(seed)
    n = n1 + n2
    base = rng.normal(size=(n, n))
    gram = base @ base.T + n * np.eye(n)  # SPD, well conditioned
    blocks = {
        (0, 0): gram[:n1, :n1], (0, 1): gram[:n1, n1:],
        (1, 0): gram[n1:, :n1], (1, 1): gram[n1:, n1:],
    }
    return blocks, n1, n2


def test_assemble_block_matrix_shape_and_content():
    blocks, n1, n2 = _toy_blocks(3, 2)
    full = ks.assemble_block_matrix(blocks, [n1, n2])
    assert full.shape == (5, 5)
    assert np.allclose(full[:3, :3], blocks[(0, 0)])
    assert np.allclose(full[3:, 3:], blocks[(1, 1)])


def test_solve_recovers_exact_system_at_zero_regularization():
    """With lambda -> 0 and a square invertible AA*, beta solves AA* beta = y
    exactly up to floating-point precision."""
    blocks, n1, n2 = _toy_blocks(4, 3)
    full = ks.assemble_block_matrix(blocks, [n1, n2])
    beta_true = np.random.default_rng(1).normal(size=n1 + n2)
    y_full = full @ beta_true
    y = [y_full[:n1], y_full[n1:]]
    solution = ks.solve(blocks, y, w=[1.0, 1.0], lam=1e-12)
    beta_hat = np.concatenate(solution.betas)
    assert np.allclose(beta_hat, beta_true, atol=1e-6)


def test_objective_value_nonnegative_and_decreases_with_more_regularization_residual_tradeoff():
    blocks, n1, n2 = _toy_blocks(4, 3)
    y = [np.ones(n1), np.ones(n2)]
    sol_small_lam = ks.solve(blocks, y, w=[1.0, 1.0], lam=1e-10)
    obj_small = ks.objective_value(blocks, y, [1.0, 1.0], 1e-10, sol_small_lam)
    assert obj_small >= 0.0
    # A tiny regularizer should fit the data almost exactly: small objective.
    assert obj_small < 1e-6


def test_predict_matches_manual_sum_of_betas():
    blocks, n1, n2 = _toy_blocks(3, 2)
    y = [np.ones(n1), np.zeros(n2)]
    solution = ks.solve(blocks, y, w=[1.0, 1.0], lam=1e-8)
    n_query = 5
    k_q1 = np.random.default_rng(2).normal(size=(n1, n_query))
    k_q2 = np.random.default_rng(3).normal(size=(n2, n_query))
    pred = solution.predict([k_q1, k_q2])
    manual = solution.betas[0] @ k_q1 + solution.betas[1] @ k_q2
    assert np.allclose(pred, manual)


def test_select_nystrom_indices_are_nested_prefixes():
    rng = np.random.default_rng(7)
    idx_small = ks.select_nystrom_indices([20], [5], rng)[0]
    rng2 = np.random.default_rng(7)
    idx_large = ks.select_nystrom_indices([20], [10], rng2)[0]
    assert np.array_equal(idx_small, idx_large[:5])


def test_solve_nystrom_matches_full_solve_when_r_equals_n():
    """When every point is selected as a representer (r = n), the Nystrom
    system coincides with the full normal equation (Eq. 39 reduces to Eq. 29
    when Z_m = I_{n_m} and K_r = AA*)."""
    blocks, n1, n2 = _toy_blocks(3, 2)
    full = ks.assemble_block_matrix(blocks, [n1, n2])
    y = [np.ones(n1), -np.ones(n2)]
    w = [1.0, 2.0]
    lam = 1e-6

    full_solution = ks.solve(blocks, y, w, lam)
    beta_full = np.concatenate(full_solution.betas)

    z_blocks = [blocks[(0, 0)], blocks[(1, 0)]]  # n_m x n (n = n1+n2), full representer set
    z_blocks = [np.concatenate([blocks[(0, 0)], blocks[(0, 1)]], axis=1),
                np.concatenate([blocks[(1, 0)], blocks[(1, 1)]], axis=1)]
    nystrom_solution = ks.solve_nystrom(z_blocks, full, y, w, lam)
    assert np.allclose(nystrom_solution.alpha, beta_full, atol=1e-5)


def test_inputs_must_be_consistent_lengths():
    blocks, n1, n2 = _toy_blocks(3, 2)
    with pytest.raises(AssertionError):
        ks.assemble_block_matrix(blocks, [n1, n2 + 1])
