"""ELBO computation: KL divergence + expected log-likelihood.

For W=1:
  q(U) = MN(M, C, D),  p(U) = MN(0, K_ZZ, Lambda)

KL(q || p) = 0.5 * [
    tr(K_ZZ^{-1} C) * tr(Lambda^{-1} D)
  + tr(Lambda^{-1} M^T K_ZZ^{-1} M)
  - M*D
  + M * log det(Lambda) - M * log det(D)
  + D * log det(K_ZZ) - D * log det(C)
]

ELL uses univariate-marginal reduction:
  m_j(x) = a(x) @ M[:,j]
  v_j(x) = Lambda[j,j] * r(x) + (a(x) C a(x)^T) * D[j,j]

Supports three likelihood × noise combinations:
  - Gaussian + constant/basis noise  (original)
  - Gaussian + GP noise              (closed-form local ELL, Corollary 4)
  - Student-t + any noise            (MC local ELL, Remark 4.8)
"""

import jax
import jax.numpy as jnp
from jax import Array

from mvgp_heter.utils import (
    safe_cholesky, spd_log_det, solve_cholesky, add_jitter,
)
from mvgp_heter.model import (
    MVGPParams, get_Lambda, get_L_Lambda,
    compute_K_ZZ, compute_K_xZ, compute_a_vec, compute_r_scalar,
)
from mvgp_heter.variational import (
    VariationalParams, get_C, get_D_cov, get_L_C, get_L_D,
)
from mvgp_heter.noise import (
    noise_log_variance, ConstantNoiseParams, GPNoiseParams,
    _gp_noise_marginals, gp_noise_kl,
)


def kl_divergence(
    var_params: VariationalParams,
    model_params: MVGPParams,
    kernel_name: str,
    jitter: float = 1e-6,
) -> Array:
    """Exact KL(q(U) || p(U)) for W=1 matrix-normal.

    Returns a scalar.
    """
    M_ind = var_params.M_var.shape[0]  # number of inducing points
    D_out = var_params.M_var.shape[1]  # number of outputs

    # Cholesky factors
    L_C = get_L_C(var_params)
    L_D = get_L_D(var_params)
    L_Lambda = get_L_Lambda(model_params)

    # Kernel matrix
    K_ZZ = compute_K_ZZ(model_params, kernel_name, jitter)
    L_ZZ = jnp.linalg.cholesky(K_ZZ)

    # Covariance matrices
    C = get_C(var_params)             # (M, M)
    D_cov = get_D_cov(var_params)     # (D, D)
    Lambda = get_Lambda(model_params) # (D, D)

    # Inverses via Cholesky solves
    K_ZZ_inv_C = solve_cholesky(L_ZZ, C)                     # K_ZZ^{-1} C
    Lambda_inv_D = solve_cholesky(L_Lambda, D_cov)            # Lambda^{-1} D
    M_var = var_params.M_var                                   # (M, D)

    # tr(K_ZZ^{-1} C) * tr(Lambda^{-1} D)
    trace_term = jnp.trace(K_ZZ_inv_C) * jnp.trace(Lambda_inv_D)

    # tr(Lambda^{-1} M^T K_ZZ^{-1} M)
    K_ZZ_inv_M = solve_cholesky(L_ZZ, M_var)                 # (M, D)
    mean_term = jnp.trace(solve_cholesky(L_Lambda, (M_var.T @ K_ZZ_inv_M)))

    # Log-determinant terms
    log_det_Lambda = spd_log_det(L_Lambda)
    log_det_D = spd_log_det(L_D)
    log_det_K_ZZ = spd_log_det(L_ZZ)
    log_det_C = spd_log_det(L_C)

    kl = 0.5 * (
        trace_term
        + mean_term
        - M_ind * D_out
        + M_ind * log_det_Lambda - M_ind * log_det_D
        + D_out * log_det_K_ZZ - D_out * log_det_C
    )
    return kl


# ── Signal univariate marginals ───────────────────────────────────────────

def _signal_marginals(var_params, model_params, kernel_name, batch_x, batch_j, jitter):
    """Compute univariate marginal mean and variance of q(f_ij).

    Returns m (B,), v (B,), plus precomputed kernel quantities.
    """
    K_ZZ = compute_K_ZZ(model_params, kernel_name, jitter)
    L_ZZ = jnp.linalg.cholesky(K_ZZ)
    K_xZ = compute_K_xZ(model_params, kernel_name, batch_x)
    a = compute_a_vec(K_xZ, L_ZZ)
    r = compute_r_scalar(kernel_name, model_params.kernel_params, batch_x, K_xZ, L_ZZ)

    M_var = var_params.M_var
    C = get_C(var_params)
    D_cov = get_D_cov(var_params)
    Lambda = get_Lambda(model_params)

    m = jnp.sum(a * M_var[:, batch_j].T, axis=1)
    Lambda_jj = Lambda[batch_j, batch_j]
    D_jj = D_cov[batch_j, batch_j]
    aCa = jnp.sum(a @ C * a, axis=1)
    v = Lambda_jj * r + aCa * D_jj

    return m, v


# ── Gaussian ELL with constant/basis noise ─────────────────────────────────

def _ell_gaussian_parametric(
    key, m, v, noise_params, batch_x, batch_j, batch_y, n_obs, n_mc,
    noise_centres, log_var_clip,
):
    """Gaussian likelihood with constant or basis noise — MC reparameterised."""
    B = batch_x.shape[0]

    noise_log_var = jax.vmap(
        lambda xi, ji: noise_log_variance(
            noise_params, xi.reshape(1, -1), ji, noise_centres, log_var_clip
        ).squeeze()
    )(batch_x, batch_j)
    noise_var = jnp.exp(noise_log_var)

    eps = jax.random.normal(key, (n_mc, B))
    f_samples = m[None, :] + jnp.sqrt(jnp.maximum(v, 1e-10))[None, :] * eps

    residuals = batch_y[None, :] - f_samples
    log_lik = -0.5 * (
        jnp.log(2.0 * jnp.pi) + noise_log_var[None, :] + residuals**2 / noise_var[None, :]
    )

    ell = (n_obs / B) * jnp.sum(jnp.mean(log_lik, axis=0))
    return ell


def _ell_gaussian_parametric_exact(m, v, noise_params, batch_x, batch_j,
                                   batch_y, n_obs, noise_centres, log_var_clip):
    """Exact Gaussian expectation for deterministic constant/basis noise."""
    noise_log_var = jax.vmap(
        lambda xi, ji: noise_log_variance(
            noise_params, xi.reshape(1, -1), ji, noise_centres, log_var_clip
        ).squeeze()
    )(batch_x, batch_j)
    terms = jnp.log(2.0 * jnp.pi) + noise_log_var + (
        (batch_y - m)**2 + v) * jnp.exp(-noise_log_var)
    return -0.5 * (n_obs / batch_x.shape[0]) * jnp.sum(terms)


# ── Gaussian ELL with GP noise (closed form, Corollary 4) ──────────────────

def _gp_noise_batch_marginals(gp_noise_params, batch_x, batch_j, D, jitter):
    """Compute GP-noise marginals for a batch in a JIT-compatible way.

    Precomputes marginals for all D outputs and selects per-element.
    """
    B = batch_x.shape[0]

    # Compute marginals for all outputs (each output uses all batch_x)
    all_m_g = jnp.zeros((D, B))
    all_v_g = jnp.zeros((D, B))
    for j_val in range(D):
        mg_j, vg_j = _gp_noise_marginals(gp_noise_params, batch_x, j_val, jitter)
        all_m_g = all_m_g.at[j_val].set(mg_j)
        all_v_g = all_v_g.at[j_val].set(vg_j)

    # Select the correct output for each batch element
    m_g = all_m_g[batch_j, jnp.arange(B)]
    v_g = all_v_g[batch_j, jnp.arange(B)]
    return m_g, v_g


def _ell_gaussian_gp_noise(m, v, gp_noise_params, batch_x, batch_j, batch_y,
                           n_obs, log_var_clip, jitter):
    """Closed-form Gaussian ELL under independent q(f) and q(g).

    E[log p(y|f,g)] = -0.5 log(2π) - 0.5 m_g
                      - 0.5 exp(-m_g + v_g/2) * ((y - m_f)^2 + v_f)

    No MC samples needed — fully analytic.
    """
    D = gp_noise_params.m_g.shape[0]
    m_g, v_g = _gp_noise_batch_marginals(gp_noise_params, batch_x, batch_j, D, jitter)

    # Clip for numerical stability
    m_g = jnp.clip(m_g, log_var_clip[0], log_var_clip[1])

    # Second moment of (y - f)^2 under q(f)
    mu2 = (batch_y - m) ** 2 + v  # E[(y - f)^2] = (y - m_f)^2 + v_f

    # Closed-form ELL
    B = batch_x.shape[0]
    local_ell = -0.5 * (
        jnp.log(2.0 * jnp.pi)
        + m_g
        + jnp.exp(-m_g + 0.5 * v_g) * mu2
    )

    ell = (n_obs / B) * jnp.sum(local_ell)
    return ell


# ── Student-t ELL (MC, any noise type) ─────────────────────────────────────

def _student_t_logpdf(y, loc, scale, nu):
    """Log-pdf of Student-t distribution."""
    z = (y - loc) / scale
    log_norm = (
        jax.lax.lgamma(0.5 * (nu + 1.0))
        - jax.lax.lgamma(0.5 * nu)
        - 0.5 * jnp.log(nu * jnp.pi)
        - jnp.log(scale)
    )
    return log_norm - 0.5 * (nu + 1.0) * jnp.log1p(z**2 / nu)


def _ell_student_t(
    key, m, v, noise_params, batch_x, batch_j, batch_y, n_obs, n_mc,
    nu, noise_centres, log_var_clip, jitter,
):
    """Student-t likelihood ELL via 2D reparameterised MC.

    Sample f ~ q(f_ij), g ~ q(g_ij), evaluate log t_nu(y; f, exp(g/2)).
    """
    B = batch_x.shape[0]

    # Sample f
    key, k_f, k_g = jax.random.split(key, 3)
    eps_f = jax.random.normal(k_f, (n_mc, B))
    f_samples = m[None, :] + jnp.sqrt(jnp.maximum(v, 1e-10))[None, :] * eps_f  # (S, B)

    # Sample g (or compute deterministic noise)
    if isinstance(noise_params, GPNoiseParams):
        D = noise_params.m_g.shape[0]
        m_g, v_g = _gp_noise_batch_marginals(noise_params, batch_x, batch_j, D, jitter)
        m_g = jnp.clip(m_g, log_var_clip[0], log_var_clip[1])
        eps_g = jax.random.normal(k_g, (n_mc, B))
        g_samples = m_g[None, :] + jnp.sqrt(jnp.maximum(v_g, 1e-10))[None, :] * eps_g
    else:
        # Deterministic noise from basis/constant
        g_det = jax.vmap(
            lambda xi, ji: noise_log_variance(
                noise_params, xi.reshape(1, -1), ji, noise_centres, log_var_clip
            ).squeeze()
        )(batch_x, batch_j)  # (B,)
        g_samples = jnp.broadcast_to(g_det[None, :], (n_mc, B))

    # scale = exp(g/2)
    scale = jnp.exp(0.5 * g_samples)  # (S, B)

    log_lik = _student_t_logpdf(batch_y[None, :], f_samples, scale, nu)  # (S, B)

    ell = (n_obs / B) * jnp.sum(jnp.mean(log_lik, axis=0))
    return ell


# ── Main ELL dispatcher ───────────────────────────────────────────────────

def expected_log_likelihood(
    key: Array,
    var_params: VariationalParams,
    model_params: MVGPParams,
    noise_params,
    kernel_name: str,
    batch_x: Array,
    batch_j: Array,
    batch_y: Array,
    n_obs: int,
    n_mc: int = 10,
    jitter: float = 1e-6,
    noise_centres: Array | None = None,
    log_var_clip: tuple = (-10.0, 5.0),
    likelihood: str = "gaussian",
    nu: float = 5.0,
    gaussian_ell: str = "analytic",
) -> Array:
    """Expected log-likelihood via univariate-marginal reduction.

    Dispatches to the appropriate local ELL based on likelihood and noise type.
    """
    m, v = _signal_marginals(var_params, model_params, kernel_name, batch_x, batch_j, jitter)

    if likelihood == "student_t":
        return _ell_student_t(
            key, m, v, noise_params, batch_x, batch_j, batch_y,
            n_obs, n_mc, nu, noise_centres, log_var_clip, jitter,
        )

    # Gaussian likelihood
    if isinstance(noise_params, GPNoiseParams):
        return _ell_gaussian_gp_noise(
            m, v, noise_params, batch_x, batch_j, batch_y,
            n_obs, log_var_clip, jitter,
        )

    if gaussian_ell == "analytic":
        return _ell_gaussian_parametric_exact(
            m, v, noise_params, batch_x, batch_j, batch_y,
            n_obs, noise_centres, log_var_clip,
        )
    if gaussian_ell != "mc":
        raise ValueError("gaussian_ell must be 'analytic' or 'mc'.")
    return _ell_gaussian_parametric(
        key, m, v, noise_params, batch_x, batch_j, batch_y,
        n_obs, n_mc, noise_centres, log_var_clip,
    )


# ── ELBO ──────────────────────────────────────────────────────────────────

def elbo(
    key: Array,
    var_params: VariationalParams,
    model_params: MVGPParams,
    noise_params,
    kernel_name: str,
    batch_x: Array,
    batch_j: Array,
    batch_y: Array,
    n_obs: int,
    n_mc: int = 10,
    jitter: float = 1e-6,
    noise_centres: Array | None = None,
    log_var_clip: tuple = (-10.0, 5.0),
    likelihood: str = "gaussian",
    nu: float = 5.0,
    gaussian_ell: str = "analytic",
) -> Array:
    """ELBO = -KL_signal + ELL - KL_noise (if GP noise)."""
    kl = kl_divergence(var_params, model_params, kernel_name, jitter)
    ell = expected_log_likelihood(
        key, var_params, model_params, noise_params, kernel_name,
        batch_x, batch_j, batch_y, n_obs, n_mc, jitter,
        noise_centres, log_var_clip, likelihood, nu, gaussian_ell,
    )

    total = ell - kl

    # Add KL for GP-noise variational posteriors
    if isinstance(noise_params, GPNoiseParams):
        D = var_params.M_var.shape[1]
        kl_noise = gp_noise_kl(noise_params, D, jitter)
        total = total - kl_noise

    return total
