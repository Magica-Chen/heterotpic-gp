"""Posterior predictive mean and variance for output j at test points."""

import jax.numpy as jnp
from jax import Array

from mvgp_heter.model import (
    MVGPParams, get_Lambda,
    compute_K_ZZ, compute_K_xZ, compute_a_vec, compute_r_scalar,
)
from mvgp_heter.variational import VariationalParams, get_C, get_D_cov
from mvgp_heter.noise import (
    noise_variance, ConstantNoiseParams, GPNoiseParams, _gp_noise_marginals,
)


def predict(
    model_params: MVGPParams,
    var_params: VariationalParams,
    noise_params,
    kernel_name: str,
    x_star: Array,
    output_j: int,
    jitter: float = 1e-6,
    noise_centres: Array | None = None,
    log_var_clip: tuple = (-10.0, 5.0),
    include_noise: bool = True,
) -> tuple[Array, Array]:
    """Posterior predictive for output j at test points x_star.

    Parameters
    ----------
    x_star : (B, T)
    output_j : int, which output

    Returns
    -------
    mean : (B,)
    variance : (B,)  — includes observation noise if include_noise=True
    """
    # Kernel computations
    K_ZZ = compute_K_ZZ(model_params, kernel_name, jitter)
    L_ZZ = jnp.linalg.cholesky(K_ZZ)
    K_xZ = compute_K_xZ(model_params, kernel_name, x_star)
    a = compute_a_vec(K_xZ, L_ZZ)  # (B, M)
    r = compute_r_scalar(
        kernel_name, model_params.kernel_params, x_star, K_xZ, L_ZZ
    )  # (B,)

    # Variational parameters
    M_var = var_params.M_var  # (M, D)
    C = get_C(var_params)     # (M, M)
    D_cov = get_D_cov(var_params)  # (D, D)
    Lambda = get_Lambda(model_params)

    # mean_j(x) = a(x) @ M[:,j]
    mean = a @ M_var[:, output_j]  # (B,)

    # var_j(x) = Lambda[j,j] * r(x) + (a C a^T) * D[j,j]
    aCa = jnp.sum(a @ C * a, axis=1)  # (B,)
    var = Lambda[output_j, output_j] * r + aCa * D_cov[output_j, output_j]

    if include_noise:
        if isinstance(noise_params, GPNoiseParams):
            # For GP noise: predictive noise variance = exp(m_g + v_g/2)
            # This is E[exp(g)] when g ~ N(m_g, v_g)
            m_g, v_g = _gp_noise_marginals(noise_params, x_star, output_j, jitter)
            m_g = jnp.clip(m_g, log_var_clip[0], log_var_clip[1])
            nv = jnp.exp(m_g + 0.5 * v_g)
        else:
            nv = noise_variance(
                noise_params, x_star, output_j, noise_centres, log_var_clip
            )
        var = var + nv

    return mean, var
