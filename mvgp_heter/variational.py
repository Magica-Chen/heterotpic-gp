"""Matrix-normal variational posterior q(U) = MN(M, C, D) for W=1.

Parameters (all unconstrained):
  M_raw: (M, D)       variational mean
  L_C:   (M, M) lower Cholesky of C (row cov, inducing locations)
  L_D:   (D, D) lower Cholesky of D (col cov, outputs)
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import NamedTuple

from mvgp_heter.utils import safe_cholesky, cholesky_to_spd


class VariationalParams(NamedTuple):
    """Variational parameters for W=1 matrix-normal posterior."""
    M_var: Array       # (M, D) variational mean
    L_C_raw: Array     # (M, M) unconstrained Cholesky of row cov
    L_D_raw: Array     # (D, D) unconstrained Cholesky of col cov


def init_variational_params(key: Array, M: int, D: int) -> VariationalParams:
    """Initialise variational parameters.

    M_var near zero, C and D near identity.
    """
    key1, key2, key3 = jax.random.split(key, 3)

    M_var = 0.01 * jax.random.normal(key1, (M, D))

    # Cholesky of row cov: near identity
    L_C_raw = jnp.eye(M) * 0.5 + 0.01 * jax.random.normal(key2, (M, M))
    L_C_raw = jnp.tril(L_C_raw)

    # Cholesky of col cov: near identity
    L_D_raw = jnp.eye(D) * 0.5 + 0.01 * jax.random.normal(key3, (D, D))
    L_D_raw = jnp.tril(L_D_raw)

    return VariationalParams(M_var=M_var, L_C_raw=L_C_raw, L_D_raw=L_D_raw)


def get_C(params: VariationalParams) -> Array:
    """Row covariance C = L_C @ L_C^T."""
    L_C = safe_cholesky(params.L_C_raw)
    return cholesky_to_spd(L_C)


def get_D_cov(params: VariationalParams) -> Array:
    """Column covariance D = L_D @ L_D^T."""
    L_D = safe_cholesky(params.L_D_raw)
    return cholesky_to_spd(L_D)


def get_L_C(params: VariationalParams) -> Array:
    """Cholesky factor of C with positive diagonal."""
    return safe_cholesky(params.L_C_raw)


def get_L_D(params: VariationalParams) -> Array:
    """Cholesky factor of D with positive diagonal."""
    return safe_cholesky(params.L_D_raw)
