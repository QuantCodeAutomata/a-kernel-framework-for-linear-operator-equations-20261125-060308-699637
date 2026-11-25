"""Unit tests for src/fourier_kernel.py (truncated Fourier kernel, Example 2.6)."""

import numpy as np

from src import fourier_kernel as fk


def test_mode_coefficients_values():
    n, c = fk.mode_coefficients(3)
    assert np.array_equal(n, [0, 1, 2, 3])
    assert np.isclose(c[0], 1.0)
    assert np.allclose(c[1:], [1.0, 1.0 / 16.0, 1.0 / 81.0])


def test_kernel_matrix_symmetric_and_diagonal_equals_sum_of_coeffs():
    n, c = fk.mode_coefficients(5)
    x = np.linspace(0, 1, 10)
    k = fk.kernel_matrix(x, x, n, c)
    assert k.shape == (10, 10)
    assert np.allclose(k, k.T, atol=1e-12)
    assert np.allclose(np.diag(k), np.sum(c))  # cos(0) = 1 for every mode


def test_second_derivative_plus_const_multiplier_eigenvalue():
    n = np.array([0, 1, 2])
    mult = fk.second_derivative_plus_const_multiplier(n, const=5.0)
    assert np.allclose(mult, [5.0, 5.0 - np.pi ** 2, 5.0 - 4 * np.pi ** 2])


def test_identity_multiplier_is_all_ones():
    n = np.array([0, 1, 2, 7])
    assert np.array_equal(fk.identity_multiplier(n), np.ones(4))


def test_cross_kernel_matches_direct_termwise_second_derivative():
    """The second-derivative-plus-constant operator is diagonal on the cosine
    basis: cos(pi n (x-y)) is an eigenfunction of d^2/dx^2 with eigenvalue
    -(pi n)^2, so k^{A,I}(x,y) = sum_n c_n * (const - (pi n)^2) * cos(pi n (x-y))
    must equal the direct termwise sum built from the stated eigenvalue."""
    n, c = fk.mode_coefficients(4)
    x = np.array([0.37])
    y = np.array([0.12])
    identity = fk.identity_multiplier(n)
    const = 5.0
    mult_a = fk.second_derivative_plus_const_multiplier(n, const)
    kd = fk.cross_kernel_matrix(x, y, n, c, mult_a, identity)
    direct = np.sum(c * mult_a * np.cos(np.pi * n * (x[0] - y[0])))
    assert np.isclose(kd[0, 0], direct, atol=1e-10)
