"""Diagnostics for the multivariate kriging experiments."""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

def radial_kernel(x, y, lengthscale, family='rbf'):
    x, y = (np.atleast_2d(x), np.atleast_2d(y))
    ell = np.broadcast_to(np.asarray(lengthscale, float), (x.shape[1],))
    if np.any(~np.isfinite(ell)) or np.any(ell <= 0):
        raise ValueError('Lengthscales must be finite and positive.')
    r = np.sqrt(np.sum(((x[:, None, :] - y[None, :, :]) / ell) ** 2, axis=-1))
    if family == 'rbf':
        return np.exp(-r * r / 2)
    if family == 'matern32':
        z = np.sqrt(3) * r
        return (1 + z) * np.exp(-z)
    if family == 'matern52':
        z = np.sqrt(5) * r
        return (1 + z + z * z / 3) * np.exp(-z)
    raise ValueError(f'Unknown kernel family: {family}')

def coverage_mass_bounds(x, y, lengthscale, radius, family='rbf'):
    """Coverage/proximity certificates on the supplied sites, in kernel units."""
    if radius <= 0:
        raise ValueError('Radius must be positive.')
    x, y = (np.atleast_2d(x), np.atleast_2d(y))
    ell = np.broadcast_to(np.asarray(lengthscale, float), (x.shape[1],))
    k = radial_kernel(x, y, ell, family)
    d = np.sqrt(np.sum(((x[:, None, :] - y[None, :, :]) / ell) ** 2, axis=-1))
    dp, dq = (d.min(axis=1), d.min(axis=0))
    tail = radial_kernel(np.array([[0.0]]), np.array([[radius]]), 1.0, family)[0, 0] ** 2
    return {'mass': float(np.sum(k * k)), 'average_mass': float(np.mean(k * k)), 'coverage_lower': float(max(np.sum(dp <= radius), np.sum(dq <= radius)) * tail), 'proximity_lower': float(max(len(x) * max(1 - dp.mean() / radius, 0), len(y) * max(1 - dq.mean() / radius, 0)) * tail)}

def expected_information(covariance, derivatives):
    """Full-sample expected Gaussian covariance information, in given coordinates."""
    factor = cho_factor(covariance, lower=True)
    whitened = [cho_solve(factor, d) for d in derivatives]
    return np.array([[0.5 * np.trace(a @ b) for b in whitened] for a in whitened])

def correlation_information(kpq, sigma_pp, sigma_qq, signal_variance_p, signal_variance_q):
    lp, lq = (np.linalg.cholesky(sigma_pp), np.linalg.cholesky(sigma_qq))
    a = solve_triangular(lp, kpq, lower=True)
    a = solve_triangular(lq, a.T, lower=True).T
    return float(signal_variance_p * signal_variance_q * np.sum(a * a))

def gaussian_gain(covariance, target_covariance, target_indices):
    """Schur-channel and direct variance-difference gain, for any Gaussian law."""
    sigma, c = (np.asarray(covariance), np.asarray(target_covariance))
    j = np.asarray(target_indices)
    aux = np.setdiff1d(np.arange(len(sigma)), j)
    marginal = cho_factor(sigma[np.ix_(j, j)], lower=True)
    projection = cho_solve(marginal, sigma[np.ix_(j, aux)])
    conditional = sigma[np.ix_(aux, aux)] - sigma[np.ix_(aux, j)] @ projection
    residual = c[aux] - c[j] @ projection
    schur_gain = residual @ cho_solve(cho_factor(conditional, lower=True), residual)
    direct_gain = c @ cho_solve(cho_factor(sigma, lower=True), c) - c[j] @ cho_solve(marginal, c[j])
    return (float(schur_gain), float(direct_gain))

def block_penalty(information, gradient, dependence_count):
    """Return full quadratic form and its correct adjusted-gradient decomposition."""
    p = dependence_count
    i, a = (np.asarray(information), np.asarray(gradient))
    chol = cho_factor(i, lower=True)
    inu = cho_factor(i[p:, p:], lower=True)
    adjusted = a[:p] - i[:p, p:] @ cho_solve(inu, a[p:])
    schur = i[:p, :p] - i[:p, p:] @ cho_solve(inu, i[p:, :p])
    nuisance = a[p:] @ cho_solve(inu, a[p:])
    dependence = adjusted @ cho_solve(cho_factor(schur, lower=True), adjusted)
    return (float(a @ cho_solve(chol, a)), float(nuisance + dependence))
