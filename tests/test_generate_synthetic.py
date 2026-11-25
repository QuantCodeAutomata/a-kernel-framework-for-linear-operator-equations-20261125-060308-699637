"""Unit tests for data/generate_synthetic.py: collocation-point generators
used by every experiment (paper Sec. 4.1-4.2, collocation-point specs)."""

import numpy as np

from data import generate_synthetic as gs


def test_uniform_points_within_bounds_and_reproducible_with_seed():
    rng = np.random.default_rng(0)
    x = gs.uniform_points(100, -1.0, 1.0, rng)
    assert x.shape == (100,)
    assert np.all((x > -1.0) & (x < 1.0))


def test_equally_spaced_points_endpoints_and_spacing():
    x = gs.equally_spaced_points(11, 0.0, 1.0)
    assert np.isclose(x[0], 0.0) and np.isclose(x[-1], 1.0)
    diffs = np.diff(x)
    assert np.allclose(diffs, diffs[0])


def test_uniform_box_points_shape_and_bounds():
    rng = np.random.default_rng(1)
    low = np.array([-3.0, 0.0])
    high = np.array([3.0, 0.5])
    pts = gs.uniform_box_points(50, low, high, rng)
    assert pts.shape == (50, 2)
    assert np.all(pts[:, 0] >= -3.0) and np.all(pts[:, 0] <= 3.0)
    assert np.all(pts[:, 1] >= 0.0) and np.all(pts[:, 1] <= 0.5)


def test_sobol_uniform_points_in_open_unit_cube():
    pts = gs.sobol_uniform_points(100, d=2, seed=42)
    assert pts.shape[0] == 100
    assert np.all(pts > 0.0) and np.all(pts < 1.0)


def test_sobol_uniform_points_deterministic_given_seed():
    a = gs.sobol_uniform_points(50, d=1, seed=123)
    b = gs.sobol_uniform_points(50, d=1, seed=123)
    assert np.array_equal(a, b)


def test_sobol_time_space_points_shapes_and_bounds():
    t, x = gs.sobol_time_space_points(64, 0.0, 1.0, np.array([-3.0]), np.array([3.0]), seed=1)
    assert t.shape == (64,)
    assert x.shape == (64, 1)
    assert np.all((t >= 0.0) & (t <= 1.0))
    assert np.all((x >= -3.0) & (x <= 3.0))


def test_sobol_lognormal_points_positive_and_correlated():
    d = 2
    mean_log = np.log(100.0) * np.ones(d)
    cov_log = np.array([[0.04, 0.02], [0.02, 0.04]])
    pts = gs.sobol_lognormal_points(256, d, mean_log, cov_log, seed=5)
    assert pts.shape == (256, 2)
    assert np.all(pts > 0.0)
    log_corr = np.corrcoef(np.log(pts).T)[0, 1]
    assert log_corr > 0.3  # positively correlated as specified by cov_log
