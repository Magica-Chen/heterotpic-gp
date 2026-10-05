"""MV-GP prior model: kernel params, Lambda, inducing points Z."""

import jax
import jax.numpy as jnp
from jax import Array
from typing import NamedTuple

from mvgp_heter.kernels import (
    init_rbf_params, init_matern32_params, init_matern52_params,
    kernel_matrix, kernel_diag,
    RBFParams, Matern32Params, Matern52Params,
)
from mvgp_heter.utils import safe_cholesky, cholesky_to_spd, add_jitter, solve_cholesky


class MVGPParams(NamedTuple):
    """All learnable parameters of the MV-GP prior."""
    kernel_params: RBFParams | Matern32Params | Matern52Params
    L_Lambda_raw: Array                          # (D, D) unconstrained Cholesky of Lambda
    Z: Array                                     # (M, T) inducing inputs


def init_model_params(
    key: Array,
    D: int,
    M: int,
    T: int,
    kernel_name: str,
    x_min: Array,
    x_max: Array,
) -> MVGPParams:
    """Initialise model parameters.

    Z is initialised on a uniform grid between x_min and x_max.
    Lambda is initialised near identity.
    """
    key1, key2 = jax.random.split(key)

    # Kernel params
    if kernel_name == "rbf":
        kp = init_rbf_params()
    elif kernel_name == "matern32":
        kp = init_matern32_params()
    elif kernel_name == "matern52":
        kp = init_matern52_params()
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    # Lambda Cholesky: initialise as identity + small noise
    L_Lambda_raw = jnp.eye(D) + 0.01 * jax.random.normal(key1, (D, D))
    L_Lambda_raw = jnp.tril(L_Lambda_raw)

    # Inducing inputs: uniform grid
    x_min = jnp.atleast_1d(x_min).squeeze()
    x_max = jnp.atleast_1d(x_max).squeeze()
    if T == 1:
        Z = jnp.linspace(float(x_min), float(x_max), M).reshape(-1, 1)
    else:
        # For multi-dimensional, use random uniform
        Z = x_min + (x_max - x_min) * jax.random.uniform(key2, (M, T))

    return MVGPParams(kernel_params=kp, L_Lambda_raw=L_Lambda_raw, Z=Z)


def get_Lambda(params: MVGPParams) -> Array:
    """Reconstruct Lambda = L L^T from unconstrained Cholesky."""
    L = safe_cholesky(params.L_Lambda_raw)
    return cholesky_to_spd(L)


def get_L_Lambda(params: MVGPParams) -> Array:
    """Get the Cholesky factor of Lambda with positive diagonal."""
    return safe_cholesky(params.L_Lambda_raw)


def compute_K_ZZ(params: MVGPParams, kernel_name: str, jitter: float = 1e-6) -> Array:
    """K_ZZ with jitter.  Shape (M, M)."""
    K = kernel_matrix(kernel_name, params.kernel_params, params.Z, params.Z)
    return add_jitter(K, jitter)


def compute_K_xZ(params: MVGPParams, kernel_name: str, X: Array) -> Array:
    """K_xZ.  Shape (B, M)."""
    return kernel_matrix(kernel_name, params.kernel_params, X, params.Z)


def compute_a_vec(K_xZ: Array, L_ZZ: Array) -> Array:
    """a(x) = K_xZ @ K_ZZ^{-1}.  Shape (B, M).

    Uses Cholesky factor L_ZZ of K_ZZ.
    """
    # Solve K_ZZ @ a^T = K_xZ^T  =>  a^T = K_ZZ^{-1} K_xZ^T
    # So a = (K_ZZ^{-1} K_xZ^T)^T
    return solve_cholesky(L_ZZ, K_xZ.T).T


def compute_r_scalar(
    kernel_name: str,
    kernel_params,
    X: Array,
    K_xZ: Array,
    L_ZZ: Array,
) -> Array:
    """r(x) = k(x,x) - K_xZ K_ZZ^{-1} K_Zx.  Shape (B,).

    Diagonal residual variance.
    """
    k_diag = kernel_diag(kernel_name, kernel_params, X)
    # K_xZ @ K_ZZ^{-1} @ K_Zx  = sum over rows of (L_ZZ^{-1} K_xZ^T)^2
    v = jax.scipy.linalg.solve_triangular(L_ZZ, K_xZ.T, lower=True)  # (M, B)
    quad = jnp.sum(v**2, axis=0)  # (B,)
    return jnp.maximum(k_diag - quad, 0.0)  # clip to non-negative
