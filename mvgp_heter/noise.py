"""Heteroscedastic noise models.

Three noise types:
1. "constant" — one learnable log-variance per output (homoscedastic).
2. "basis"    — RBF basis expansion: g_j(x) = gamma_j^T phi(x) + bias_j.
3. "gp"       — sparse GP prior on each log-noise function g^(j)(x).

noise_var_j(x) = exp(g_j(x))   in all cases.
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import NamedTuple

from mvgp_heter.utils import safe_cholesky, cholesky_to_spd, add_jitter, solve_cholesky, spd_log_det
from mvgp_heter.kernels import kernel_matrix, kernel_diag, init_rbf_params, init_matern52_params


# ── Basis-expansion noise ──────────────────────────────────────────────────

class NoiseParams(NamedTuple):
    """Learnable noise parameters."""
    gamma: Array       # (D, n_basis) basis coefficients
    bias: Array        # (D,) per-output bias
    log_width: Array   # scalar, shared width for RBF bases


class ConstantNoiseParams(NamedTuple):
    """Learnable constant (homoscedastic) noise."""
    log_variance: Array  # (D,) per-output log-variance


# ── GP-noise parameters ───────────────────────────────────────────────────

class GPNoiseParams(NamedTuple):
    """Sparse-GP noise model: one independent GP per output.

    For output j, q(u_g^(j)) = N(m_g_j, S_g_j) where S_g_j = L_S_j L_S_j^T.

    Fields
    ------
    Z_g : Array, (M_g, T) — shared inducing inputs for all noise GPs.
    m_g : Array, (D, M_g) — variational means.
    L_S_raw : Array, (D, M_g, M_g) — unconstrained lower-Cholesky of
              variational covariances (one per output).
    kernel_params : kernel hyperparameters for the noise kernel (shared).
    """
    Z_g: Array                  # (M_g, T)
    m_g: Array                  # (D, M_g)
    L_S_raw: Array              # (D, M_g, M_g)
    noise_kernel_params: object # RBFParams or Matern52Params (shared)


# ── Initialisation ─────────────────────────────────────────────────────────

def init_noise_params(
    key: Array,
    D: int,
    n_basis: int,
    centres: Array,
    noise_type: str = "basis",
    M_noise: int = 15,
    T: int = 1,
    x_min: float = 0.0,
    x_max: float = 10.0,
    noise_kernel: str = "rbf",
) -> NoiseParams | ConstantNoiseParams | GPNoiseParams:
    """Initialise noise parameters.

    Parameters
    ----------
    centres : Array, shape (n_basis, T)
        Fixed basis function centres (used for noise_type="basis").
    noise_type : "basis", "constant", or "gp"
    M_noise : number of inducing points for GP noise (noise_type="gp")
    """
    if noise_type == "constant":
        return ConstantNoiseParams(log_variance=jnp.full(D, -2.3))

    if noise_type == "gp":
        return _init_gp_noise_params(key, D, M_noise, T, x_min, x_max, noise_kernel)

    # Basis expansion noise
    gamma = 0.01 * jax.random.normal(key, (D, n_basis))
    bias = jnp.full(D, -2.0)  # exp(-2) ~ 0.135
    log_width = jnp.array(0.0)  # width = 1.0

    return NoiseParams(gamma=gamma, bias=bias, log_width=log_width)


def _init_gp_noise_params(key, D, M_g, T, x_min, x_max, noise_kernel):
    """Initialise GP-noise variational parameters."""
    key, k1, k2 = jax.random.split(key, 3)

    # Shared inducing inputs on uniform grid
    if T == 1:
        Z_g = jnp.linspace(x_min, x_max, M_g).reshape(-1, 1)
    else:
        Z_g = jnp.linspace(x_min, x_max, M_g).reshape(-1, 1).repeat(T, axis=1)

    # Variational means: initialise near -2 (exp(-2) ≈ 0.135 noise variance)
    m_g = jnp.full((D, M_g), -2.0) + 0.01 * jax.random.normal(k1, (D, M_g))

    # Variational Cholesky factors: initialise near 0.5*I
    L_S_raw = jnp.tile(jnp.eye(M_g) * 0.5, (D, 1, 1))
    L_S_raw = L_S_raw + 0.01 * jax.random.normal(k2, (D, M_g, M_g))
    # Enforce lower-triangular
    mask = jnp.tril(jnp.ones((M_g, M_g)))
    L_S_raw = L_S_raw * mask[None, :, :]

    # Noise kernel params
    if noise_kernel == "rbf":
        nkp = init_rbf_params()
    elif noise_kernel == "matern52":
        nkp = init_matern52_params()
    else:
        raise ValueError(f"Unknown noise kernel: {noise_kernel}")

    return GPNoiseParams(Z_g=Z_g, m_g=m_g, L_S_raw=L_S_raw, noise_kernel_params=nkp)


def make_basis_centres(x_min: float, x_max: float, n_basis: int, T: int = 1) -> Array:
    """Create evenly spaced RBF basis centres."""
    if T == 1:
        return jnp.linspace(x_min, x_max, n_basis).reshape(-1, 1)
    else:
        return jnp.linspace(x_min, x_max, n_basis).reshape(-1, 1).repeat(T, axis=1)


# ── Basis-expansion evaluation ─────────────────────────────────────────────

def _rbf_basis(X: Array, centres: Array, log_width: Array) -> Array:
    """Evaluate RBF basis functions.

    Parameters
    ----------
    X : (B, T)
    centres : (K, T)
    log_width : scalar

    Returns
    -------
    phi : (B, K)
    """
    width = jnp.exp(log_width)
    sq_dist = jnp.sum((X[:, None, :] - centres[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-0.5 * sq_dist / width**2)


# ── GP-noise marginal computation ──────────────────────────────────────────

def _gp_noise_marginals(gp_noise_params: GPNoiseParams, X: Array, j: int,
                        jitter: float = 1e-6):
    """Compute marginal mean and variance of g^(j)(x) under q(u_g^(j)).

    m_g(x) = a_g(x) @ m_g_j
    v_g(x) = r_g(x) + a_g(x) @ S_g_j @ a_g(x)^T

    where a_g(x) = K_xZ_g @ K_ZZ_g^{-1}, r_g(x) = k(x,x) - K_xZ_g K_ZZ_g^{-1} K_Z_gx.

    Parameters
    ----------
    X : (B, T)
    j : output index

    Returns
    -------
    m_g : (B,) marginal mean
    v_g : (B,) marginal variance
    """
    nkp = gp_noise_params.noise_kernel_params
    Z_g = gp_noise_params.Z_g  # (M_g, T)
    nk_name = "rbf" if hasattr(nkp, 'log_variance') and not hasattr(nkp, '__matern__') else "rbf"
    # Determine kernel name from param type
    from mvgp_heter.kernels import RBFParams, Matern52Params
    if isinstance(nkp, Matern52Params):
        nk_name = "matern52"
    else:
        nk_name = "rbf"

    # Kernel matrices
    K_ZZ = kernel_matrix(nk_name, nkp, Z_g, Z_g)
    K_ZZ = add_jitter(K_ZZ, jitter)
    L_ZZ = jnp.linalg.cholesky(K_ZZ)

    K_xZ = kernel_matrix(nk_name, nkp, X, Z_g)  # (B, M_g)
    k_diag = kernel_diag(nk_name, nkp, X)  # (B,)

    # a_g(x) = K_xZ @ K_ZZ^{-1}
    a_g = solve_cholesky(L_ZZ, K_xZ.T).T  # (B, M_g)

    # r_g(x) = k(x,x) - a_g K_Zx = k_diag - sum of (L^{-1} K_xZ^T)^2
    v_tmp = jax.scipy.linalg.solve_triangular(L_ZZ, K_xZ.T, lower=True)
    r_g = jnp.maximum(k_diag - jnp.sum(v_tmp**2, axis=0), 0.0)  # (B,)

    # Variational parameters for output j
    m_j = gp_noise_params.m_g[j]  # (M_g,)
    L_S_j = safe_cholesky(gp_noise_params.L_S_raw[j])  # (M_g, M_g)
    S_j = cholesky_to_spd(L_S_j)  # (M_g, M_g)

    # Marginal mean
    m_g = a_g @ m_j  # (B,)

    # Marginal variance: r_g + a_g S_j a_g^T (diagonal only)
    v_g = r_g + jnp.sum(a_g @ S_j * a_g, axis=1)  # (B,)

    return m_g, v_g


def gp_noise_kl(gp_noise_params: GPNoiseParams, D: int, jitter: float = 1e-6):
    """KL(q(u_g) || p(u_g)) summed over all D output noise GPs.

    Each q(u_g^(j)) = N(m_j, S_j), p(u_g^(j)) = N(0, K_ZZ_g).
    """
    nkp = gp_noise_params.noise_kernel_params
    Z_g = gp_noise_params.Z_g
    from mvgp_heter.kernels import RBFParams, Matern52Params
    nk_name = "matern52" if isinstance(nkp, Matern52Params) else "rbf"

    K_ZZ = kernel_matrix(nk_name, nkp, Z_g, Z_g)
    K_ZZ = add_jitter(K_ZZ, jitter)
    L_ZZ = jnp.linalg.cholesky(K_ZZ)
    M_g = Z_g.shape[0]

    total_kl = 0.0
    for j in range(D):
        m_j = gp_noise_params.m_g[j]
        L_S_j = safe_cholesky(gp_noise_params.L_S_raw[j])
        S_j = cholesky_to_spd(L_S_j)

        # KL(N(m,S) || N(0,K)) = 0.5 [tr(K^{-1}S) + m^T K^{-1} m - M_g + log|K| - log|S|]
        K_inv_S = solve_cholesky(L_ZZ, S_j)
        K_inv_m = solve_cholesky(L_ZZ, m_j)

        kl_j = 0.5 * (
            jnp.trace(K_inv_S)
            + m_j @ K_inv_m
            - M_g
            + spd_log_det(L_ZZ) - spd_log_det(L_S_j)
        )
        total_kl = total_kl + kl_j

    return total_kl


# ── Unified interface ──────────────────────────────────────────────────────

def noise_log_variance(
    noise_params: NoiseParams | ConstantNoiseParams | GPNoiseParams,
    X: Array,
    j: int,
    centres: Array | None = None,
    clip: tuple = (-10.0, 5.0),
) -> Array:
    """Compute log noise variance for output j at locations X.

    For GP noise, returns the marginal mean m_g(x) (point estimate).
    For the full GP-noise ELL, use gp_noise_marginals() directly.

    Parameters
    ----------
    X : (B, T)
    j : output index
    centres : (K, T) basis centres (required for "basis" type)

    Returns
    -------
    log_var : (B,)
    """
    if isinstance(noise_params, ConstantNoiseParams):
        return jnp.full(X.shape[0], noise_params.log_variance[j])

    if isinstance(noise_params, GPNoiseParams):
        m_g, _ = _gp_noise_marginals(noise_params, X, j)
        return jnp.clip(m_g, clip[0], clip[1])

    # Basis expansion
    phi = _rbf_basis(X, centres, noise_params.log_width)  # (B, K)
    g = phi @ noise_params.gamma[j] + noise_params.bias[j]  # (B,)
    return jnp.clip(g, clip[0], clip[1])


def noise_variance(
    noise_params,
    X: Array,
    j: int,
    centres: Array | None = None,
    clip: tuple = (-10.0, 5.0),
) -> Array:
    """exp(log_variance)."""
    return jnp.exp(noise_log_variance(noise_params, X, j, centres, clip))
