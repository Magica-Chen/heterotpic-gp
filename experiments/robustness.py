"""Robustness for the multivariate kriging experiments."""
from __future__ import annotations
from functools import lru_cache
import gzip
from itertools import product
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from experiments.exact_gp import FitConfig, fit_gp
from experiments.paired_models import paired_traces
from experiments.rich_gp import fit_rich
from experiments.common import canonical
from experiments.screening import ScreenConfig, site_folds
from experiments.synthetic import GaussianDGP, pair_design, seed_for
FAMILIES = ('lmc_positive', 'lmc_cancelling', 'ar_fidelity', 'trend_confounded', 'heteroscedastic', 'heavy_tailed')

def scenarios():
    return [{'id': f'B{i + 1:02d}', 'family': family, 'geometry': geometry, 'n1': n, 'n2': n, 'working_kernel': 'matern32', 'dimension': 1, 'replicates': 200, 'conditional_reference': family != 'heavy_tailed'} for i, (family, geometry, n) in enumerate(product(FAMILIES, ('isotopic', 'interleaved', 'separated'), (20, 40)))]

@lru_cache(maxsize=8)
def generating_model(scenario_json):
    s = json.loads(scenario_json)
    xs = pair_design(s['geometry'], s['n1'])
    evaluation = ((np.arange(201) + 0.5) / 201)[:, None]
    family = s['family']
    noise = [0.05, 0.05]
    means = evalmean = None
    if family.startswith('lmc'):
        loading = np.array([[1.0, 0.25], [0.2, 1.0]]) if family == 'lmc_positive' else np.array([[np.sqrt(0.8), np.sqrt(0.2)], [np.sqrt(0.2), -np.sqrt(0.8)]])
        loading = loading / np.linalg.norm(loading, axis=1)[:, None]
        components = [(np.outer(loading[:, i], loading[:, i]), ell, 'matern32') for i, ell in enumerate((0.06, 0.35))]
    elif family == 'ar_fidelity':
        components = [(np.array([[0.49, 0.7], [0.7, 1.0]]), 0.3, 'matern32'), (np.diag([0.51, 0.0]), 0.06, 'matern32')]
    elif family == 'trend_confounded':
        components = [(np.eye(2), 0.15, 'matern32')]
        means = [2 * xs[0][:, 0], -1.5 * xs[1][:, 0]]
        evalmean = 2 * evaluation[:, 0]
    else:
        components = [(np.array([[1.0, 0.7], [0.7, 1.0]]), 0.15, 'matern32')]
        if family == 'heteroscedastic':
            noise = [0.01 + 0.19 * xs[0][:, 0] ** 2, 0.02]
        if family == 'heavy_tailed':
            noise = [0.0, 0.0]
    dgp = GaussianDGP(xs, evaluation, components, noise, means, evalmean)
    return dgp

def draw_dataset(scenario, phase, rep):
    dgp = generating_model(canonical(scenario))
    seed = seed_for('E8', phase, scenario['id'], rep)
    ys, latent, oj, oi = dgp.draw(seed)
    noise_rng = np.random.default_rng(seed_for('E8', phase, scenario['id'], rep, 'observation_noise'))
    if scenario['family'] == 'heavy_tailed':
        ys = [y + np.sqrt(0.05 * 3 / 5) * noise_rng.standard_t(5, size=len(y)) for y in ys]
        observed = latent + np.sqrt(0.05 * 3 / 5) * noise_rng.standard_t(5, size=len(latent))
        oj = oi = None
    else:
        test_noise = 0.01 + 0.19 * dgp.evaluation[:, 0] ** 2 if scenario['family'] == 'heteroscedastic' else 0.05
        observed = latent + np.sqrt(test_noise) * noise_rng.standard_normal(len(latent))
    return (dgp, ys, latent, observed, oj, oi, seed)

def richer_fits(xs, ys, family, config):
    if family.startswith('lmc'):
        marginals = [fit_rich([x], [y], 'mixture', config) for x, y in zip(xs, ys)]
        joint = fit_rich(xs, ys, 'lmc', config)
    elif family == 'ar_fidelity':
        marginals = [fit_rich([xs[0]], [ys[0]], 'mixture', config), fit_gp([xs[1]], [ys[1]], config)]
        joint = fit_rich(xs, ys, 'ar', config)
    elif family == 'trend_confounded':
        marginals = [fit_rich([x], [y], 'separable', config, linear_mean=True) for x, y in zip(xs, ys)]
        joint = fit_rich(xs, ys, 'separable', config, linear_mean=True)
    elif family == 'heteroscedastic':
        marginals = [fit_rich([x], [y], 'separable', config, noise_shape=[shape]) for x, y, shape in zip(xs, ys, ('quadratic', 'constant'))]
        joint = fit_rich(xs, ys, 'separable', config, noise_shape=['quadratic', 'constant'])
    else:
        raise ValueError('A richer comparator is defined for the five Gaussian families.')
    return (marginals, joint)

def richer_trace(xs, ys, evaluation, fold_seed, family, config):
    start = perf_counter()
    marginals, joint = richer_fits(xs, ys, family, config)
    independent = marginals[0].predict(evaluation)
    combined = joint.predict(evaluation) if joint.success else independent
    membership = site_folds(xs, fold_seed, 4)
    folds = []
    for k in range(4):
        masks = [m != k for m in membership]
        x = [v[m] for v, m in zip(xs, masks)]
        y = [v[m] for v, m in zip(ys, masks)]
        xv, yv = (xs[0][~masks[0]], ys[0][~masks[0]])
        marginal, paired = richer_fits(x, y, family, config)
        am = marginal[0].predict(xv)[0]
        bm = paired.predict(xv)[0] if paired.success else am
        folds.append({'fold': k, 'train_indices': [np.flatnonzero(m).tolist() for m in masks], 'validation_target_indices': np.flatnonzero(~masks[0]).tolist(), 'count': len(yv), 'independent_sse': float(np.sum((am - yv) ** 2)), 'joint_sse': float(np.sum((bm - yv) ** 2)), 'marginals': [f.record() for f in marginal], 'joint': paired.record()})
    return ({'marginals': [f.record() for f in marginals], 'joint': joint.record(), 'folds': folds, 'runtime_seconds': perf_counter() - start}, independent, combined)

def run_dataset(task):
    directory, manifest, scenario, rep = task
    phase = manifest['phase']
    folder = Path(directory) / phase / scenario['id']
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'dataset_{rep:04d}.json.gz'
    if path.exists():
        return
    started = perf_counter()
    dgp, ys, latent, observed, oj, oi, seed = draw_dataset(scenario, phase, rep)
    xs, evaluation = (dgp.xs, dgp.evaluation)
    fold_seed = seed_for('E8', phase, scenario['id'], rep, 'folds')
    config, screen = (FitConfig(**manifest['fit_config']), ScreenConfig(**manifest['screen_config']))
    marginal, joint, scores, _, folds = paired_traces(xs, ys, evaluation, fold_seed, config, screen)
    independent = marginal[0].predict(evaluation)
    combined = joint.predict(evaluation) if joint.success else independent
    predictions = {'independent': independent, 'joint': combined}
    rich = None
    if scenario['family'] != 'heavy_tailed':
        rich, rich_ind, rich_joint = richer_trace(xs, ys, evaluation, fold_seed, scenario['family'], config)
        predictions.update(rich_independent=rich_ind, rich_joint=rich_joint)
    assessment = {}
    for name, (mean, var, obsvar) in predictions.items():
        mse = float(np.mean((latent - mean) ** 2))
        risk = float(np.mean(dgp.joint_variance + (oj - mean) ** 2)) if oj is not None else mse
        assessment[name] = {'risk': risk, 'risk_kind': 'conditional_Gaussian' if oj is not None else 'realised_latent_loss', 'latent_mse': mse, 'observed_mse': float(np.mean((observed - mean) ** 2)), 'mlpd': float(np.mean(-0.5 * (np.log(2 * np.pi * obsvar) + (observed - mean) ** 2 / obsvar))), 'coverage95': float(np.mean(np.abs(observed - mean) <= 1.959963984540054 * np.sqrt(obsvar))), 'width95': float(np.mean(2 * 1.959963984540054 * np.sqrt(obsvar)))}
    record = {'experiment_id': 'E8', 'phase': phase, 'scenario': scenario, 'replication': rep, 'dataset_seed': seed, 'fold_seed': fold_seed, 'working': {'marginals': [f.record() for f in marginal], 'joint': joint.record(), 'scores': scores, 'folds': folds}, 'rich': rich, 'assessment': assessment, 'research_runtime_seconds': perf_counter() - started}
    keys = list(predictions)
    arrays = path.with_suffix('').with_suffix('.npz')
    temporary = arrays.with_name(arrays.name + f'.tmp.{os.getpid()}')
    with temporary.open('wb') as f:
        np.savez_compressed(f, train_x=np.vstack(xs), train_y=np.concatenate(ys), output_ids=np.repeat(np.arange(2), [len(x) for x in xs]), evaluation=evaluation, latent_eval=latent, observed_eval=observed, oracle_joint=oj if oj is not None else np.array([]), oracle_independent=oi if oi is not None else np.array([]), oracle_joint_variance=dgp.joint_variance if oj is not None else np.array([]), method_names=np.array(keys), prediction_mean=np.array([predictions[k][0] for k in keys]), prediction_observed_variance=np.array([predictions[k][2] for k in keys]))
    temporary.replace(arrays)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with gzip.open(temporary, 'wt') as f:
        f.write(canonical(record))
    temporary.replace(path)
