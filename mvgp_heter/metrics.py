"""Evaluation metrics: RMSE, NLPD, coverage, interval score, Delta_Lambda."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm


def rmse(pred_mean: Array, true_f: Array) -> float:
    """Root mean squared error on latent function."""
    return float(jnp.sqrt(jnp.mean((pred_mean - true_f) ** 2)))


def nlpd(pred_mean: Array, pred_var: Array, true_y: Array,
         likelihood: str = "gaussian", nu: float = 5.0) -> float:
    """Negative log-predictive density.

    NLPD = -mean[ log p(y_i | mu_i, sigma_i^2) ]

    Supports "gaussian" and "student_t" likelihoods.
    """
    if likelihood == "student_t":
        scale = jnp.sqrt(pred_var)
        z = (true_y - pred_mean) / scale
        log_norm = (
            jax.lax.lgamma(0.5 * (nu + 1.0))
            - jax.lax.lgamma(0.5 * nu)
            - 0.5 * jnp.log(nu * jnp.pi)
            - jnp.log(scale)
        )
        log_lik = log_norm - 0.5 * (nu + 1.0) * jnp.log1p(z**2 / nu)
        return float(-jnp.mean(log_lik))

    log_lik = norm.logpdf(true_y, loc=pred_mean, scale=jnp.sqrt(pred_var))
    return float(-jnp.mean(log_lik))


def coverage(
    pred_mean: Array,
    pred_var: Array,
    true_f: Array,
    level: float = 0.95,
) -> float:
    """Empirical coverage of prediction interval at given level."""
    z = norm.ppf((1.0 + level) / 2.0)
    std = jnp.sqrt(pred_var)
    lower = pred_mean - z * std
    upper = pred_mean + z * std
    in_interval = (true_f >= lower) & (true_f <= upper)
    return float(jnp.mean(in_interval))


def interval_score(
    pred_mean: Array,
    pred_var: Array,
    true_f: Array,
    level: float = 0.95,
) -> float:
    """Interval score (Gneiting & Raftery 2007).

    IS = (u - l) + (2/alpha) * (l - y) * 1{y < l} + (2/alpha) * (y - u) * 1{y > u}
    where alpha = 1 - level.
    """
    alpha = 1.0 - level
    z = norm.ppf((1.0 + level) / 2.0)
    std = jnp.sqrt(pred_var)
    lower = pred_mean - z * std
    upper = pred_mean + z * std

    width = upper - lower
    below = jnp.maximum(lower - true_f, 0.0)
    above = jnp.maximum(true_f - upper, 0.0)

    score = width + (2.0 / alpha) * below + (2.0 / alpha) * above
    return float(jnp.mean(score))


def delta_lambda(Lambda_hat: Array, Lambda_star: Array) -> float:
    """Relative Frobenius error in Lambda estimation.

    ||Lambda_hat - Lambda_star||_F / ||Lambda_star||_F
    """
    diff_norm = jnp.linalg.norm(Lambda_hat - Lambda_star, ord="fro")
    star_norm = jnp.linalg.norm(Lambda_star, ord="fro")
    return float(diff_norm / star_norm)


def normalize_lambda(Lambda: Array) -> Array:
    """Normalize Lambda to correlation matrix R_Lambda.

    R_Lambda = diag(Lambda)^{-1/2} Lambda diag(Lambda)^{-1/2}
    """
    d = jnp.sqrt(jnp.diag(Lambda))
    d_inv = 1.0 / jnp.maximum(d, 1e-12)
    return Lambda * jnp.outer(d_inv, d_inv)


def delta_r_lambda(Lambda_hat: Array, Lambda_star: Array) -> float:
    """Relative Frobenius error on normalized correlation matrices.

    ||R_hat - R_star||_F / ||R_star||_F
    """
    R_hat = normalize_lambda(Lambda_hat)
    R_star = normalize_lambda(Lambda_star)
    diff_norm = jnp.linalg.norm(R_hat - R_star, ord="fro")
    star_norm = jnp.linalg.norm(R_star, ord="fro")
    return float(diff_norm / star_norm)


def sign_recovery_score(Lambda_hat: Array, Lambda_star: Array) -> float:
    """Fraction of off-diagonal pairs where sign of R_Lambda matches.

    Returns a value in [0, 1]; 1.0 means perfect sign recovery.
    """
    R_hat = normalize_lambda(Lambda_hat)
    R_star = normalize_lambda(Lambda_star)
    D = R_star.shape[0]
    # Extract strict upper-triangle entries
    idx = jnp.triu_indices(D, k=1)
    signs_hat = jnp.sign(R_hat[idx])
    signs_star = jnp.sign(R_star[idx])
    return float(jnp.mean(signs_hat == signs_star))
