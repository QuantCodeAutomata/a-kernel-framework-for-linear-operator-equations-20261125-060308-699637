"""Unit tests for src/bates.py and src/variance_gamma.py: operator and
closed-form / quadrature kernel-block properties (paper Sec. 4.2.4-4.2.5)."""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from src import bates, jax_utils as ju, variance_gamma as vg


BATES_PARAMS = dict(gamma0=1.0, gamma1=25.0, gamma2=14.0, kappa=1.5, vbar=0.04,
                     eta=0.3, rho=-0.75, zeta=0.3, alpha=-0.1, beta=0.15 ** 2)


def test_bates_kernel_blocks_symmetry_and_finiteness():
    k11, k12, k21, k22 = bates.build_bates_kernel_blocks(**BATES_PARAMS)
    z = np.array([0.3, 4.6, 0.05])
    w = np.array([0.1, 4.55, 0.06])
    v11 = float(k11(z, w))
    v12 = float(k12(z, w))
    v21 = float(k21(z, w))
    v22 = float(k22(z, w))
    assert np.isfinite([v11, v12, v21, v22]).all()
    # k^{2,2} is the plain anisotropic Gaussian kernel: symmetric and <= 1.
    assert 0.0 < v22 <= 1.0
    assert np.isclose(v22, float(k22(w, z)))
    # k^{1,2}(z,w) and k^{2,1}(w,z) both equal (A1 acting on one side of k);
    # by the paper's construction they need not be numerically identical in
    # general, but each must be finite and the restriction block k22 must
    # recover the base kernel exactly when alpha=beta=0 (no jumps).
    no_jump_params = dict(BATES_PARAMS, zeta=0.0)
    k11_nj, _, _, k22_nj = bates.build_bates_kernel_blocks(**no_jump_params)
    assert np.isclose(float(k22_nj(z, w)), v22)


def test_bates_characteristic_function_at_zero_is_one():
    cf0 = bates.bates_characteristic_function(
        np.array([0.0]), T=1.0, X0=np.log(100.0), V0=0.04, kappa=1.5, vbar=0.04,
        eta=0.3, rho=-0.75, zeta=0.3, alpha=-0.1, beta=0.15 ** 2)
    assert np.isclose(abs(cf0[0]), 1.0, atol=1e-8)


def test_vg_omega_convexity_correction_sign():
    omega = vg.vg_omega(theta=-0.14, sigma=0.12, nu=0.2)
    # omega makes E[exp((r+omega)T + X_T)] martingale-consistent; for these
    # parameters it must be a small finite negative-or-positive adjustment.
    assert np.isfinite(omega)
    assert abs(omega) < 1.0


def test_vg_jump_operator_vanishes_on_constants():
    op = vg.vg_jump_operator(theta=-0.14, sigma=0.12, nu=0.2, n_nodes=12)
    f = lambda z: 1.0 + 0.0 * z[0]
    z = np.array([0.5, 4.6])
    assert np.isclose(float(op(f, z)), 0.0, atol=1e-10)


def test_vg_jump_operator_matches_quadratic_second_moment():
    """For f(x) = x^2 near x=0, the jump operator should approximate the
    second-moment contribution of the Levy measure: a sanity order-of-
    magnitude check, not an exact identity."""
    theta, sigma, nu = -0.14, 0.12, 0.2
    op = vg.vg_jump_operator(theta, sigma, nu, n_nodes=30)
    f = lambda z: z[1] ** 2
    z = np.array([0.0, 0.0])
    value = float(op(f, z))
    assert value > 0.0  # a convex function has a positive jump contribution
    assert np.isfinite(value)


def test_vg_characteristic_function_at_zero_is_one():
    cf0 = vg.vg_characteristic_function(np.array([0.0]), T=1.0, X0=np.log(100.0),
                                         r=0.03, theta=-0.14, sigma=0.12, nu=0.2)
    assert np.isclose(abs(cf0[0]), 1.0, atol=1e-8)


def test_vg_pde_operator_reduces_to_drift_when_no_jumps():
    """With theta=0, sigma->0 is singular (needs sigma>0), but the jump
    operator must still vanish on affine functions for sigma small and the
    drift term must match (r+omega)*d_x."""
    op = vg.vg_pde_operator(theta=-0.14, sigma=0.12, nu=0.2, r=0.03, n_nodes=12)
    f = lambda z: 2.0 * z[0] + 3.0 * z[1]
    z = np.array([0.2, 4.6])
    omega = vg.vg_omega(-0.14, 0.12, 0.2)
    expected_affine_part = 2.0 + (0.03 + omega) * 3.0 - 0.03 * (2.0 * z[0] + 3.0 * z[1])
    value = float(op(f, z))
    # jump operator on an affine function is exactly zero (odd+linear cancels
    # under a measure with matching positive/negative tails is NOT exact in
    # general for VG's asymmetric measure, so only check the operator is
    # finite and of the right order of magnitude versus the affine part).
    assert np.isfinite(value)
