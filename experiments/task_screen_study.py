"""Task screen study for the multivariate kriging experiments."""
from __future__ import annotations
from experiments.common import ROOT as PACKAGE_ROOT
from functools import lru_cache
import gzip
from itertools import product
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from experiments.exact_gp import FitConfig, fit_gp
from experiments.common import canonical
from experiments.screening import ScreenConfig, fit_marginals, pair_scores
from experiments.synthetic import GaussianDGP, seed_for
from experiments.shared_scale_gp import fit_shared_scale, shared_marginals
from experiments.task_screening import TaskScreenConfig, coupled_fit, fold_data, opportunity_scores, target_folds
ROOT = PACKAGE_ROOT
METHOD = ROOT / 'configuration/E6_task_v2'

def task_scenarios():
    return [dict(id=f'T{i + 1:02d}', geometry=geometry, n1=n, n2=4 * n, rho=rho, noise_variance=[noise, 0.01], signal_variance=[1.0, 1.0], lengthscale=0.15, kernel='matern32') for i, (geometry, n, rho, noise) in enumerate(product(('colocated', 'infill', 'partial', 'remote'), (12, 24), (0.0, 0.4, 0.8), (0.01, 0.1)))]

def task_design(scenario):
    n, n2 = (scenario['n1'], scenario['n2'])
    x = np.linspace(0, 1, n)[:, None]
    if scenario['geometry'] == 'colocated':
        y = np.repeat(x, 4, axis=0)
    else:
        offset = {'infill': 0.0, 'partial': 0.5, 'remote': 2.0}[scenario['geometry']]
        y = (offset + (np.arange(n2) + 0.5) / n2)[:, None]
    return [x, y]

@lru_cache(maxsize=4)
def task_dgp(scenario_json):
    s = json.loads(scenario_json)
    lam = np.sqrt(np.outer(s['signal_variance'], s['signal_variance'])) * np.array([[1.0, s['rho']], [s['rho'], 1.0]])
    return GaussianDGP(task_design(s), ((np.arange(201) + 0.5) / 201)[:, None], [(lam, s['lengthscale'], s['kernel'])], s['noise_variance'])

def timed_prediction(fit, x):
    started = perf_counter()
    prediction = fit.predict(x)
    return (prediction, perf_counter() - started)

def run_dataset(task):
    run_dir, manifest, scenario, rep = task
    out = Path(run_dir) / manifest['phase'] / scenario['id'] / f'dataset_{rep:04d}.json.gz'
    arrays = out.with_suffix('').with_suffix('.npz')
    if out.exists():
        return {'skipped': True}
    started = perf_counter()
    phase = manifest['phase']
    fit_config, screen = (FitConfig(**manifest['fit_config']), TaskScreenConfig(**manifest['screen_config']))
    seed = seed_for('E6_task_v2', phase, scenario['id'], rep)
    fold_seed = seed_for('E6_task_v2', phase, scenario['id'], rep, 'folds')
    dgp = task_dgp(canonical(scenario))
    ys, latent, oracle, oracle_ind = dgp.draw(seed)
    xs, evaluation = (dgp.xs, dgp.evaluation)
    full = fit_marginals(xs, ys, fit_config)
    diagonal = fit_shared_scale(xs, ys, full, fit_config)
    scores = opportunity_scores(shared_marginals(diagonal), evaluation, screen)
    legacy_scores = pair_scores(full, evaluation, ScreenConfig())
    joint = coupled_fit(xs, ys, full, diagonal, fit_config)
    own_prediction, own_prediction_time = timed_prediction(full[0], evaluation)
    diagonal_prediction, diagonal_prediction_time = timed_prediction(diagonal if diagonal.success else full[0], evaluation)
    joint_prediction, joint_prediction_time = timed_prediction(joint if joint.success else diagonal if diagonal.success else full[0], evaluation)
    membership = target_folds(xs[0], fold_seed, screen.folds)
    folds = []
    for index in range(screen.folds):
        tx, ty, vx, vy = fold_data(xs, ys, membership, index)
        target = fit_gp([tx[0]], [ty[0]], fit_config)
        pilots = [target, full[1]]
        diagonal_fold = fit_shared_scale(tx, ty, pilots, fit_config)
        score = opportunity_scores(shared_marginals(diagonal_fold), evaluation, screen)
        legacy_score = pair_scores(pilots, evaluation, ScreenConfig())
        coupled = coupled_fit(tx, ty, pilots, diagonal_fold, fit_config)
        own, tp = timed_prediction(target, vx)
        diag_pred, td = timed_prediction(diagonal_fold if diagonal_fold.success else target, vx)
        pred, tj = timed_prediction(coupled if coupled.success else diagonal_fold if diagonal_fold.success else target, vx)
        folds.append({'fold': index, 'target_train_indices': np.flatnonzero(membership != index).tolist(), 'target_validation_indices': np.flatnonzero(membership == index).tolist(), 'auxiliary_train_indices': list(range(len(xs[1]))), 'target_marginal': target.record(), 'diagonal': diagonal_fold.record(), 'joint': coupled.record(), 'scores': score, 'legacy_scores': legacy_score, 'independent_sse': float(np.sum((vy - own[0]) ** 2)), 'diagonal_sse': float(np.sum((vy - diag_pred[0]) ** 2)), 'joint_sse': float(np.sum((vy - pred[0]) ** 2)), 'independent_prediction_seconds': tp, 'diagonal_prediction_seconds': td, 'joint_prediction_seconds': tj})
    observed = latent + np.random.default_rng(seed_for('E6_task_v2', phase, scenario['id'], rep, 'test_noise')).normal(scale=np.sqrt(scenario['noise_variance'][0]), size=len(latent))

    def metrics(prediction):
        mean, variance, observed_variance = prediction
        return {'conditional_risk': float(np.mean(dgp.joint_variance + (oracle - mean) ** 2)), 'latent_mse': float(np.mean((latent - mean) ** 2)), 'observed_mse': float(np.mean((observed - mean) ** 2)), 'coverage95': float(np.mean(np.abs(observed - mean) <= 1.959963984540054 * np.sqrt(observed_variance))), 'width95': float(np.mean(2 * 1.959963984540054 * np.sqrt(observed_variance))), 'mlpd': float(np.mean(-0.5 * (np.log(2 * np.pi * observed_variance) + (observed - mean) ** 2 / observed_variance)))}
    row = {'scenario': scenario, 'replication': rep, 'phase': phase, 'dataset_seed': seed, 'fold_seed': fold_seed, 'marginals': [m.record() for m in full], 'diagonal': diagonal.record(), 'joint': joint.record(), 'scores': scores, 'legacy_scores': legacy_scores, 'folds': folds, 'independent_prediction_seconds': own_prediction_time, 'joint_prediction_seconds': joint_prediction_time, 'diagonal_prediction_seconds': diagonal_prediction_time, 'assessment': {'independent': metrics(own_prediction), 'diagonal': metrics(diagonal_prediction), 'joint': metrics(joint_prediction), 'oracle_independent_risk': float(dgp.ind_variance.mean()), 'oracle_joint_risk': float(dgp.joint_variance.mean()), 'independent_parameter_penalty': float(np.mean((own_prediction[0] - oracle_ind) ** 2)), 'joint_parameter_penalty': float(np.mean((joint_prediction[0] - oracle) ** 2))}, 'research_runtime_seconds': perf_counter() - started}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_arrays = arrays.with_name(arrays.name + f'.tmp.{os.getpid()}')
    with tmp_arrays.open('wb') as stream:
        np.savez_compressed(stream, train_x=np.vstack(xs), train_y=np.concatenate(ys), output_ids=np.repeat(np.arange(2), list(map(len, xs))), evaluation=evaluation, latent_eval=latent, observed_eval=observed, oracle_mean=oracle, oracle_variance=dgp.joint_variance, independent_mean=own_prediction[0], joint_mean=joint_prediction[0], independent_variance=own_prediction[1], diagonal_mean=diagonal_prediction[0], diagonal_variance=diagonal_prediction[1], diagonal_observed_variance=diagonal_prediction[2], joint_variance=joint_prediction[1], independent_observed_variance=own_prediction[2], joint_observed_variance=joint_prediction[2])
    tmp_arrays.replace(arrays)
    tmp = out.with_name(out.name + f'.tmp.{os.getpid()}')
    with gzip.open(tmp, 'wt') as stream:
        stream.write(canonical(row))
    tmp.replace(out)
    return {'skipped': False, 'runtime_seconds': row['research_runtime_seconds']}
