"""Utility functions: Cholesky parameterisation, jitter, solves."""

import jax
import jax.numpy as jnp
from jax import Array


def cholesky_to_spd(L: Array) -> Array:
    """L @ L^T  ->  symmetric positive definite matrix."""
    return L @ L.T


def safe_cholesky(L_raw: Array) -> Array:
    """Ensure a lower-triangular matrix with positive diagonal.

    L_raw is an unconstrained lower-triangular matrix.  We apply
    softplus to the diagonal to guarantee positivity.
    """
    diag_idx = jnp.diag_indices_from(L_raw)
    L = L_raw.at[diag_idx].set(jax.nn.softplus(L_raw[diag_idx]))
    return jnp.tril(L)


def spd_log_det(L: Array) -> Array:
    """Log-determinant of SPD matrix given its Cholesky factor L.

    log det(A) = 2 * sum(log(diag(L)))  where A = L L^T.
    """
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))


def add_jitter(K: Array, jitter: float = 1e-6) -> Array:
    """K + jitter * I."""
    return K + jitter * jnp.eye(K.shape[0])


def solve_cholesky(L: Array, b: Array) -> Array:
    """Solve L L^T x = b  via two triangular solves.

    Returns x = (L L^T)^{-1} b.
    """
    y = jax.scipy.linalg.solve_triangular(L, b, lower=True)
    x = jax.scipy.linalg.solve_triangular(L.T, y, lower=False)
    return x


def inv_from_cholesky(L: Array) -> Array:
    """Compute (L L^T)^{-1} via Cholesky solve with identity."""
    I = jnp.eye(L.shape[0])
    return solve_cholesky(L, I)
