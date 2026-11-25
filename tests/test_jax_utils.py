"""Unit tests for the generic operator-induced kernel machinery (src/jax_utils.py).

Validates the sign and value of the Gaussian-kernel derivative representers
against the paper's closed-form Example 2.5 (Eq. 9-10) and against a
central finite-difference check, and checks the query-representer
convention used by Algorithm 1's prediction step.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from src import jax_utils as ju


def gaussian(gamma: float):
    def k(x, y):
        d = x - y
        return jnp.exp(-gamma / 2.0 * jnp.sum(d * d))
    return k


def test_representer_matches_paper_closed_form_eq9_eq10():
    gamma = 100.0
    k = gaussian(gamma)
    x = jnp.array([0.3])
    y = jnp.array([-0.2])

    k_x = float(k(x, y))
    expected_d1 = -gamma * (float(x[0]) - float(y[0])) * k_x          # Eq. 9
    expected_d11 = (gamma ** 2 * (float(x[0]) - float(y[0])) ** 2 - gamma) * k_x  # Eq. 10

    r1 = ju.representer(k, ju.partial_operator(0))
    r11 = ju.representer(k, ju.second_partial_operator(0, 0))

    assert np.isclose(float(r1(x, y)), expected_d1, atol=1e-10)
    assert np.isclose(float(r11(x, y)), expected_d11, atol=1e-10)


def test_representer_matches_finite_difference():
    gamma = 100.0
    k = gaussian(gamma)
    x0, y = 0.3, -0.2
    h = 1e-6

    def k_scalar(xx):
        return float(k(jnp.array([xx]), jnp.array([y])))

    fd = (k_scalar(x0 + h) - k_scalar(x0 - h)) / (2 * h)
    r1 = ju.representer(k, ju.partial_operator(0))
    analytic = float(r1(jnp.array([x0]), jnp.array([y])))
    assert np.isclose(analytic, fd, atol=1e-5)


def test_cross_kernel_matches_eq28_first_and_second_argument_convention():
    """k^{1,1}(x,y) = d_x d_y k(x,y): for a difference kernel k(x,y)=g(x-y),
    d_x d_y g(x-y) = -g''(x-y), matched against Eq. 10's d_xx k via sign."""
    gamma = 100.0
    k = gaussian(gamma)
    deriv = ju.partial_operator(0)
    k11 = ju.cross_kernel(k, deriv, deriv)
    x = jnp.array([0.3])
    y = jnp.array([-0.2])
    d = float(x[0]) - float(y[0])
    k_x = float(k(x, y))
    expected = -(gamma ** 2 * d ** 2 - gamma) * k_x  # -d_xx k(x,y) for a difference kernel
    assert np.isclose(float(k11(x, y)), expected, atol=1e-8)


def test_build_matrix_shape_and_symmetry():
    gamma = 10.0
    k = gaussian(gamma)
    x = np.linspace(-1, 1, 7)
    mat = ju.build_matrix(k, ju.as_points(x), ju.as_points(x))
    assert mat.shape == (7, 7)
    assert np.allclose(mat, mat.T, atol=1e-10)
    assert np.allclose(np.diag(mat), 1.0)  # k(x,x) = exp(0) = 1


def test_combine_operators_linearity():
    gamma = 10.0
    k = gaussian(gamma)
    op = ju.combine_operators((2.0, ju.partial_operator(0)), (-1.0, ju.identity_operator))
    x = jnp.array([0.5])
    y = jnp.array([0.1])
    val = op(lambda z: k(z, y), x)
    expected = 2.0 * ju.partial_operator(0)(lambda z: k(z, y), x) - ju.identity_operator(lambda z: k(z, y), x)
    assert np.isclose(float(val), float(expected))


@pytest.mark.parametrize("n", [1, 2, 4])
def test_second_partial_operator_on_quadratic(n):
    """d^2/dx_i^2 of a pure quadratic should recover twice its coefficient."""
    def f(z):
        return 3.0 * jnp.sum(z ** 2)

    z0 = jnp.zeros(n)
    op = ju.second_partial_operator(0, 0)
    assert np.isclose(float(op(f, z0)), 6.0)
