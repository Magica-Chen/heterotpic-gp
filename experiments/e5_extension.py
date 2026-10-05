"""E5 extension for the multivariate kriging experiments."""
from functools import lru_cache
import gzip
import json
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from experiments.oracle import kernel_matrix
from experiments.exact_gp import FitConfig
from experiments.paired_models import fit_models
from experiments.common import ROOT, canonical
from experiments.synthetic import seed_for

def configurations():
    return [{'id': family + '_' + g, 'family': family, 'geometry': g} for family in ('separable', 'original_lmc') for g in ('isotopic', 'interleaved', 'separated')]

@lru_cache(maxsize=6)
def design(scenario_json):
    s = json.loads(scenario_json)
    x = np.linspace(0, 1, 16)[:, None]
    second = x.copy() if s['geometry'] == 'isotopic' else (np.linspace(0, 1, 16, endpoint=False) + 0.5 / 16)[:, None] if s['geometry'] == 'interleaved' else np.linspace(1.25, 2.25, 16)[:, None]
    xs = [x, second]
    evaluation = np.linspace(0, 1, 120)[:, None]
    combo = [np.vstack([v, evaluation]) for v in xs]
    if s['family'] == 'separable':
        components = [(np.array([[1, 0.7], [0.7, 1]]), 0.16)]
    else:
        loadings = np.array([[0.78, 0.44], [0.58, 0.28]])
        components = [(np.outer(loadings[:, j], loadings[:, j]), ell) for j, ell in enumerate((0.09, 0.34))]
    latent = np.block([[sum((c[p, q] * kernel_matrix('matern32', combo[p], combo[q], 1, ell) for c, ell in components)) for q in range(2)] for p in range(2)])
    latent += 1e-08 * np.eye(len(latent))
    train = np.r_[np.arange(16), 136 + np.arange(16)]
    test = 16 + np.arange(120)
    covariance = latent[np.ix_(train, train)] + 0.01 * np.eye(32)
    cross = latent[np.ix_(test, train)]
    weights = cho_solve(cho_factor(covariance, lower=True), cross.T).T
    iw = cho_solve(cho_factor(covariance[:16, :16], lower=True), cross[:, :16].T).T
    diagonal = np.diag(latent)[test]
    variance = diagonal - np.sum(weights * cross, axis=1)
    iv = diagonal - np.sum(iw * cross[:, :16], axis=1)
    return (xs, evaluation, latent, train, test, weights, variance, iw, iv)

def draw(s, seed):
    xs, evaluation, covariance, train, test, weights, variance, iw, iv = design(canonical(s))
    rng = np.random.default_rng(seed)
    latent = rng.multivariate_normal(np.zeros(len(covariance)), covariance)
    observed = latent + np.concatenate([rng.normal(scale=0.1, size=136) for _ in range(2)])
    ys = [observed[train[:16]], observed[train[16:]]]
    return (xs, evaluation, ys, latent[test], observed[test], weights @ observed[train], variance, iw @ ys[0], iv)

def run_dataset(task):
    manifest, s, rep = task
    phase = manifest['phase']
    path = ROOT / 'runs/E5_extension_v1' / phase / s['id'] / f'dataset_{rep:04d}.json.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    started = perf_counter()
    seed = seed_for('E5_extension', phase, s['id'], rep)
    xs, evaluation, ys, latent, observed, om, ov, oim, oiv = draw(s, seed)
    config = FitConfig(**manifest['fit_config'])
    fits = {}
    predictions = {}
    for mode in ('correlation_only', 'full'):
        known = {'signal_variance': [1, 1], 'noise_variance': [0.01, 0.01], 'lengthscale': 0.16} if mode == 'correlation_only' else None
        marginal, joint = fit_models(xs, ys, config, known)
        fits[mode] = {'marginals': [v.record() for v in marginal], 'joint': joint.record()}
        predictions[mode + '_independent'] = np.array(marginal[0].predict(evaluation))
        predictions[mode + '_joint'] = np.array(joint.predict(evaluation)) if joint.success else predictions[mode + '_independent'].copy()
    assessment = {}
    for name, p in predictions.items():
        mean, _, variance = p
        assessment[name] = {'risk': float(np.mean(ov + (om - mean) ** 2)), 'latent_mse': float(np.mean((latent - mean) ** 2)), 'rmse': float(np.sqrt(np.mean((latent - mean) ** 2))), 'observed_mse': float(np.mean((observed - mean) ** 2)), 'mlpd': float(np.mean(-0.5 * (np.log(2 * np.pi * variance) + (observed - mean) ** 2 / variance))), 'coverage95': float(np.mean(abs(observed - mean) <= 1.959963984540054 * np.sqrt(variance))), 'width95': float(np.mean(2 * 1.959963984540054 * np.sqrt(variance)))}
    arrays = path.with_suffix('').with_suffix('.npz')
    np.savez_compressed(arrays, train_x=np.vstack(xs), train_y=np.concatenate(ys), output_ids=np.repeat(np.arange(2), 16), evaluation=evaluation, latent_eval=latent, observed_eval=observed, oracle_mean=om, oracle_variance=ov, oracle_independent_mean=oim, oracle_independent_variance=oiv, method_names=np.array(list(predictions)), predictions=np.array(list(predictions.values())))
    record = {'phase': phase, 'scenario': s, 'replication': rep, 'dataset_seed': seed, 'fits': fits, 'assessment': assessment, 'wall_seconds': perf_counter() - started}
    with gzip.open(path, 'wt') as stream:
        stream.write(canonical(record))
