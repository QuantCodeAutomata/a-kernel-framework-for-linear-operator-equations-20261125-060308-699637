"""Generic machinery for computing operator-induced kernels by automatic
differentiation, as prescribed in Algorithm 1 / Section 3.3 of Filipovic &
Tissot-Daguette, "A Kernel Framework for Linear Operator Equations":
"When A_m is a differential operator, we compute the operator-induced kernel
k^{A_m} by automatic differentiation with the JAX library."

Custom implementation (paper Sec. 3.3, Eq. 25, 28): JAX provides the
automatic-differentiation primitives (jax.grad, jax.hessian, jax.vmap) used
here, but the construction of operator-induced cross-kernels from a scalar
kernel and a linear differential operator is the paper's own method, not a
feature of any quantitative-finance library in this environment.

Linear operators are represented as plain Python callables

    operator(f, z) -> float

where ``f`` is a scalar JAX-traceable function of a single point ``z`` and
``operator`` returns ``(A f)(z)`` for the differential operator ``A``. This
makes operators composable with ordinary Python code and lets JAX trace
through nested applications for the cross-kernels k^{l,m} = A_l A_m k.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

Operator = Callable[[Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray], jnp.ndarray]


def identity_operator(f: Callable, z: jnp.ndarray) -> jnp.ndarray:
    """Restriction/identity operator: (I f)(z) = f(z)."""
    return f(z)


def partial_operator(index: int) -> Operator:
    """First partial derivative operator w.r.t. coordinate ``index``."""

    def op(f: Callable, z: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(f)(z)[index]

    return op


def second_partial_operator(i: int, j: int) -> Operator:
    """Second partial derivative operator d^2f/(dz_i dz_j)."""

    def op(f: Callable, z: jnp.ndarray) -> jnp.ndarray:
        return jax.hessian(f)(z)[i, j]

    return op


def combine_operators(*weighted_ops: tuple[float, Operator]) -> Operator:
    """Linear combination of operators: sum_k coeff_k * op_k."""

    def op(f: Callable, z: jnp.ndarray) -> jnp.ndarray:
        return sum(coeff * sub_op(f, z) for coeff, sub_op in weighted_ops)

    return op


def apply_operator_to_second_argument(k: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                                       operator: Operator,
                                       x: jnp.ndarray,
                                       y: jnp.ndarray) -> jnp.ndarray:
    """Evaluate (A k(x, .))(y), i.e. the representer A_m k_x evaluated at y."""
    return operator(lambda yy: k(x, yy), y)


def operator_induced_kernel(k: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                             operator: Operator) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Return k_x^{A}(y) := (A k(x, .))(y) as a function of (x, y)."""

    def k_a(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return apply_operator_to_second_argument(k, operator, x, y)

    return k_a


def cross_kernel(k: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  op_l: Operator,
                  op_m: Operator) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Return k^{l,m}(x, y) = (A_l A_m k)(x, y) (Eq. 25, 28).

    Per Eq. 28, "A_l and A_m act on the first and second arguments of k,
    respectively": A_m is applied to the second argument of k (treating the
    first argument as a fixed parameter), producing a function of (x, y);
    A_l is then applied to the first argument of that function (treating y
    as fixed). Since x and y are independent variables, the two
    differentiations commute, matching Eq. 28's block entries
    (A_l A_m^*)_{n,n'} = k^{l,m}(x_{l,n}, x_{m,n'}).
    """

    def k_lm(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        def phi(xx: jnp.ndarray) -> jnp.ndarray:
            return operator_induced_kernel(k, op_m)(xx, y)

        return op_l(phi, x)

    return k_lm


def representer(k: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                operator: Operator) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Return the per-point representer k_x^{A}(y) := (A k_y)(x) (Eq. 4, 7).

    The operator A acts on the FIRST argument of k (the training/collocation
    point x, before substitution), while y (the free/query argument) is held
    fixed. This is the function evaluated by Algorithm 1's prediction step,
    "Evaluate k_{x_{m,n}}^{A_m}(x) = A_m k(x_{m,n}, x)" (Sec. 3.3, p.10):
    here x_{m,n} plays the role of the first argument acted on by A_m, and
    the query point plays the role of the free second argument. This is
    equivalent to ``cross_kernel(k, operator, identity_operator)``.
    """
    return cross_kernel(k, operator, identity_operator)


def build_matrix(pairwise_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  X: np.ndarray,
                  Y: np.ndarray) -> np.ndarray:
    """Vectorize ``pairwise_fn`` over all pairs (X[i], Y[j]) into a matrix.

    ``X`` has shape (n, d) or (n,); ``Y`` has shape (m, d) or (m,). Scalar
    inputs are promoted to shape (n, 1) so a single ``operator`` definition
    works for both scalar and vector-valued collocation points.
    """
    X = jnp.atleast_2d(jnp.asarray(X, dtype=jnp.float64))
    Y = jnp.atleast_2d(jnp.asarray(Y, dtype=jnp.float64))
    if X.shape[1] == 1 and np.asarray(X).ndim == 1:
        pass  # already (n,1)
    vectorized = jax.vmap(jax.vmap(pairwise_fn, in_axes=(None, 0)), in_axes=(0, None))
    return np.asarray(vectorized(X, Y))


def as_points(x: np.ndarray) -> np.ndarray:
    """Reshape a 1D array of scalar collocation points to shape (n, 1)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x
