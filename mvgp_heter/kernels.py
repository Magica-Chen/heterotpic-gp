"""Kernel functions: pure JAX implementations of RBF and Matern kernels."""

import jax.numpy as jnp
from jax import Array
from typing import NamedTuple


class RBFParams(NamedTuple):
    log_variance: Array    # log(sigma_f^2)
    log_lengthscale: Array # log(ell)


class Matern52Params(NamedTuple):
    log_variance: Array
    log_lengthscale: Array


class Matern32Params(NamedTuple):
    log_variance: Array
    log_lengthscale: Array


def init_rbf_params() -> RBFParams:
    return RBFParams(
        log_variance=jnp.array(0.0),
        log_lengthscale=jnp.array(0.0),
    )


def init_matern52_params() -> Matern52Params:
    return Matern52Params(
        log_variance=jnp.array(0.0),
        log_lengthscale=jnp.array(0.0),
    )


def init_matern32_params() -> Matern32Params:
    return Matern32Params(
        log_variance=jnp.array(0.0),
        log_lengthscale=jnp.array(0.0),
    )


def _sq_distances(X1: Array, X2: Array) -> Array:
    """Squared Euclidean distances.  X1: (N1,T), X2: (N2,T) -> (N1,N2)."""
    return jnp.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)


def rbf_kernel(params: RBFParams, X1: Array, X2: Array) -> Array:
    """RBF (squared exponential) kernel matrix.

    k(x1, x2) = sigma_f^2 * exp(-||x1 - x2||^2 / (2 * ell^2))
    """
    variance = jnp.exp(params.log_variance)
    lengthscale = jnp.exp(params.log_lengthscale)
    sq_dist = _sq_distances(X1, X2)
    return variance * jnp.exp(-0.5 * sq_dist / lengthscale**2)


def matern52_kernel(params: Matern52Params, X1: Array, X2: Array) -> Array:
    """Matern-5/2 kernel matrix.

    k(x1, x2) = sigma_f^2 * (1 + sqrt(5)*r/ell + 5*r^2/(3*ell^2))
                 * exp(-sqrt(5)*r/ell)
    """
    variance = jnp.exp(params.log_variance)
    lengthscale = jnp.exp(params.log_lengthscale)
    sq_dist = _sq_distances(X1, X2)
    r = jnp.sqrt(jnp.maximum(sq_dist, 1e-36))
    scaled = jnp.sqrt(5.0) * r / lengthscale
    return variance * (1.0 + scaled + scaled**2 / 3.0) * jnp.exp(-scaled)


def matern32_kernel(params: Matern32Params, X1: Array, X2: Array) -> Array:
    """Matern-3/2 kernel matrix.

    k(x1, x2) = sigma_f^2 * (1 + sqrt(3)*r/ell) * exp(-sqrt(3)*r/ell)
    """
    variance = jnp.exp(params.log_variance)
    lengthscale = jnp.exp(params.log_lengthscale)
    sq_dist = _sq_distances(X1, X2)
    r = jnp.sqrt(jnp.maximum(sq_dist, 1e-36))
    scaled = jnp.sqrt(3.0) * r / lengthscale
    return variance * (1.0 + scaled) * jnp.exp(-scaled)


def kernel_matrix(kernel_name: str, params, X1: Array, X2: Array) -> Array:
    """Dispatch to the appropriate kernel."""
    if kernel_name == "rbf":
        return rbf_kernel(params, X1, X2)
    elif kernel_name == "matern32":
        return matern32_kernel(params, X1, X2)
    elif kernel_name == "matern52":
        return matern52_kernel(params, X1, X2)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")


def kernel_diag(kernel_name: str, params, X: Array) -> Array:
    """Diagonal of the kernel matrix: [k(x_i, x_i)]."""
    if kernel_name in ("rbf", "matern32", "matern52"):
        return jnp.full(X.shape[0], jnp.exp(params.log_variance))
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")
