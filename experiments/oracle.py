"""Oracle for the multivariate kriging experiments."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import numpy as np
from experiments.geometry import _to_2d

def kernel_matrix(name: str, x1: np.ndarray, x2: np.ndarray, variance: float, lengthscale: float) -> np.ndarray:
    x1 = _to_2d(x1)
    x2 = _to_2d(x2)
    sq_dist = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    if name == 'rbf':
        return variance * np.exp(-0.5 * sq_dist / lengthscale ** 2)
    r = np.sqrt(np.maximum(sq_dist, 1e-36))
    if name == 'matern32':
        scaled = np.sqrt(3.0) * r / lengthscale
        return variance * (1.0 + scaled) * np.exp(-scaled)
    if name == 'matern52':
        scaled = np.sqrt(5.0) * r / lengthscale
        return variance * (1.0 + scaled + scaled ** 2 / 3.0) * np.exp(-scaled)
    raise ValueError(f'Unknown kernel {name}')

@dataclass
class OracleParams:
    kernel: str
    variance: float
    lengthscale: float
    Lambda: np.ndarray
    noise: np.ndarray

def build_obs_cov(xs: list[np.ndarray], params: OracleParams) -> np.ndarray:
    d = len(xs)
    blocks = []
    for p in range(d):
        row = []
        for q in range(d):
            base = kernel_matrix(params.kernel, xs[p], xs[q], params.variance, params.lengthscale)
            block = params.Lambda[p, q] * base
            if p == q:
                block = block + params.noise[p] * np.eye(len(xs[p]))
            row.append(block)
        blocks.append(row)
    return np.block(blocks)

def stack_outputs(ys: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(y, dtype=float) for y in ys])

def cross_cov_vector(x_star: np.ndarray, xs: list[np.ndarray], params: OracleParams, target_j: int) -> list[np.ndarray]:
    out = []
    for q, x_q in enumerate(xs):
        base = kernel_matrix(params.kernel, x_star, x_q, params.variance, params.lengthscale)[0]
        out.append(params.Lambda[target_j, q] * base)
    return out

def predictive_variance(x_star: np.ndarray, xs: list[np.ndarray], params: OracleParams, target_j: int, joint: bool=True) -> float:
    x_star = _to_2d(x_star)
    prior_var = params.Lambda[target_j, target_j] * params.variance
    if joint:
        Sigma = build_obs_cov(xs, params)
        c = np.concatenate(cross_cov_vector(x_star, xs, params, target_j))
        return float(prior_var - c @ np.linalg.solve(Sigma, c))
    Sigma_j = params.Lambda[target_j, target_j] * kernel_matrix(params.kernel, xs[target_j], xs[target_j], params.variance, params.lengthscale) + params.noise[target_j] * np.eye(len(xs[target_j]))
    c_j = cross_cov_vector(x_star, xs, params, target_j)[target_j]
    return float(prior_var - c_j @ np.linalg.solve(Sigma_j, c_j))

def oracle_gain(x_star: np.ndarray, xs: list[np.ndarray], params: OracleParams, target_j: int) -> float:
    return predictive_variance(x_star, xs, params, target_j, joint=False) - predictive_variance(x_star, xs, params, target_j, joint=True)

def average_oracle_gain(eval_grid: np.ndarray, xs: list[np.ndarray], params: OracleParams, target_j: int) -> float:
    eval_grid = _to_2d(eval_grid)
    return float(np.mean([oracle_gain(x[None, :], xs, params, target_j) for x in eval_grid]))

def W_matrix(xs: list[np.ndarray], params: OracleParams) -> np.ndarray:
    d = len(xs)
    out = np.zeros((d, d), dtype=float)
    for p, q in product(range(d), range(d)):
        base = kernel_matrix(params.kernel, xs[p], xs[q], params.variance, params.lengthscale)
        out[p, q] = float(np.sum(base ** 2))
    return out

def sample_latent_and_observed(rng: np.random.Generator, xs: list[np.ndarray], params: OracleParams) -> tuple[list[np.ndarray], list[np.ndarray]]:
    Sigma_latent = build_obs_cov(xs, OracleParams(kernel=params.kernel, variance=params.variance, lengthscale=params.lengthscale, Lambda=params.Lambda, noise=np.zeros_like(params.noise)))
    latent = rng.multivariate_normal(np.zeros(Sigma_latent.shape[0]), Sigma_latent + 1e-08 * np.eye(Sigma_latent.shape[0]))
    ys = []
    fs = []
    offset = 0
    for j, x_j in enumerate(xs):
        n_j = len(x_j)
        f_j = latent[offset:offset + n_j]
        y_j = f_j + rng.normal(scale=np.sqrt(params.noise[j]), size=n_j)
        fs.append(f_j)
        ys.append(y_j)
        offset += n_j
    return (fs, ys)

def sample_train_test_outputs(rng: np.random.Generator, train_xs: list[np.ndarray], test_xs: list[np.ndarray], params: OracleParams) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    combo = [np.vstack([train_xs[j], test_xs[j]]) for j in range(len(train_xs))]
    fs_all, ys_all = sample_latent_and_observed(rng, combo, params)
    train_f, train_y, test_f, test_y = ([], [], [], [])
    for j, xj in enumerate(train_xs):
        n_train = len(xj)
        train_f.append(fs_all[j][:n_train])
        train_y.append(ys_all[j][:n_train])
        test_f.append(fs_all[j][n_train:])
        test_y.append(ys_all[j][n_train:])
    return (train_f, train_y, test_f, test_y)

def residual_channel_terms(x_star: np.ndarray, x1: np.ndarray, x2: np.ndarray, params: OracleParams) -> dict[str, float]:
    x_star = _to_2d(x_star)
    k11 = kernel_matrix(params.kernel, x1, x1, params.variance, params.lengthscale)
    k21 = kernel_matrix(params.kernel, x2, x1, params.variance, params.lengthscale)
    k1s = kernel_matrix(params.kernel, x_star, x1, params.variance, params.lengthscale)[0]
    k2s = kernel_matrix(params.kernel, x_star, x2, params.variance, params.lengthscale)[0]
    sigma1 = params.noise[0] / params.Lambda[0, 0]
    resid = k2s - k21 @ np.linalg.solve(k11 + sigma1 * np.eye(len(x1)), k1s)
    delta = oracle_gain(x_star, [x1, x2], params, 0)
    raw_norm = float(np.sum(k2s ** 2))
    resid_norm = float(np.sum(resid ** 2))
    nearest = int(np.argmin(np.linalg.norm(_to_2d(x2) - x_star, axis=1)))
    lower = params.Lambda[0, 1] ** 2 / (params.Lambda[1, 1] + params.noise[1]) * float(resid[nearest] ** 2)
    upper = params.Lambda[0, 1] ** 2 / params.noise[1] * resid_norm
    return {'delta': float(delta), 'raw_norm': raw_norm, 'resid_norm': resid_norm, 'lower_bound': float(lower), 'upper_bound': float(upper), 'nearest_residual_sq': float(resid[nearest] ** 2)}

def latent_coupling_score(Lambda: np.ndarray, j: int) -> float:
    idx = [q for q in range(Lambda.shape[0]) if q != j]
    sub = Lambda[np.ix_(idx, idx)]
    vec = Lambda[j, idx]
    return float(vec @ np.linalg.solve(sub, vec) / Lambda[j, j])
