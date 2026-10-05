"""Penalty study for the multivariate kriging experiments."""
from __future__ import annotations
from functools import lru_cache
import gzip
from itertools import product
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from experiments.exact_gp import FitConfig, kernel_and_logscale_derivatives
from experiments.paired_models import fit_models
from experiments.penalties import information_penalties, fitted_oracle_gain, bootstrap_penalties
from experiments.common import canonical
from experiments.screening import site_folds
from experiments.synthetic import pair_design, GaussianDGP, seed_for

def scenarios_for(study):
    scenarios = []
    if study == 'E7':
        for i, (geometry, rho) in enumerate(product(('isotopic', 'interleaved', 'separated'), (0.0, 0.7))):
            for mode in ('correlation_only', 'full'):
                scenarios.append({'id': f'P{i + 1:02d}_{mode}', 'design_id': f'P{i + 1:02d}', 'geometry': geometry, 'rho': rho, 'n1': 20, 'n2': 20, 'dimension': 1, 'lengthscale': 0.15, 'signal_variance': [1.0, 1.0], 'noise_variance': [0.05, 0.05], 'correlation': [[1.0, rho], [rho, 1.0]], 'kernel': 'matern32', 'fit_mode': mode, 'evaluation_replicates': 500, 'bootstrap_outer_indices': list(range(30))})
    else:
        for i, (geometry, ell) in enumerate(product(('rich', 'moderate', 'poor'), (0.12, 0.08, 0.2))):
            for mode in ('correlation_only', 'full'):
                scenarios.append({'id': f'K{i + 1:02d}_{mode}', 'design_id': f'K{i + 1:02d}', 'geometry': geometry, 'n1': 12, 'n2': 12, 'n3': 11, 'dimension': 1, 'lengthscale': ell, 'signal_variance': [1.0, 1.0, 1.0], 'noise_variance': [0.1, 0.03, 0.03], 'correlation': [[1.0, 0.85, 0.45], [0.85, 1.0, 0.25], [0.45, 0.25, 1.0]], 'kernel': 'matern32', 'fit_mode': mode, 'evaluation_replicates': 500 if mode == 'correlation_only' else 300, 'scale_role': 'recovered_original' if ell == 0.12 else 'declared_sensitivity', 'bootstrap_outer_indices': []})
    return scenarios

@lru_cache(maxsize=8)
def dgp_for(study, scenario_json):
    scenario = json.loads(scenario_json)
    if study == 'E7':
        xs = pair_design(scenario['geometry'], 20)
        evaluation = ((np.arange(201) + 0.5) / 201)[:, None]
    else:
        x1 = np.linspace(0.0, 0.5, 12)[:, None]
        endpoints = {'rich': (0.0, 0.48), 'moderate': (0.5, 0.9), 'poor': (0.9, 1.0)}[scenario['geometry']]
        x2 = np.linspace(*endpoints, 12)[:, None]
        xs = [x1, x2, np.linspace(0.03, 0.47, 11)[:, None]]
        evaluation = np.linspace(0.0, 0.5, 120)[:, None]
    lam = np.array(scenario['correlation'])
    return GaussianDGP(xs, evaluation, [(lam, scenario['lengthscale'], scenario['kernel'])], scenario['noise_variance'])

def validate(xs, ys, fold_seed, config, known):
    membership = site_folds(xs, fold_seed, 4)
    folds = []
    start = perf_counter()
    for k in range(4):
        mask = [m != k for m in membership]
        x = [a[m] for a, m in zip(xs, mask)]
        y = [a[m] for a, m in zip(ys, mask)]
        xv, yv = (xs[0][~mask[0]], ys[0][~mask[0]])
        marginals, joint = fit_models(x, y, config, known)
        am = marginals[0].predict(xv)[0]
        bm = joint.predict(xv)[0] if joint.success else am
        folds.append({'fold': k, 'train_indices': [np.flatnonzero(m).tolist() for m in mask], 'validation_target_indices': np.flatnonzero(~mask[0]).tolist(), 'count': len(yv), 'independent_sse': float(np.sum((am - yv) ** 2)), 'joint_sse': float(np.sum((bm - yv) ** 2)), 'marginals': [m.record() for m in marginals], 'joint': joint.record()})
    return (folds, perf_counter() - start)

def run_dataset(task):
    directory, manifest, scenario, rep = task
    folder = Path(directory) / manifest['phase'] / scenario['id']
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'dataset_{rep:04d}.json.gz'
    if path.exists():
        return {'skipped': True}
    started = perf_counter()
    study, phase = (manifest['experiment_id'], manifest['phase'])
    seed = seed_for(study, phase, scenario['design_id'], rep)
    fold_seed = seed_for(study, phase, scenario['design_id'], rep, 'folds')
    dgp = dgp_for(study, canonical(scenario))
    ys, latent, oj, oi = dgp.draw(seed)
    xs, evaluation = (dgp.xs, dgp.evaluation)
    config = FitConfig(**manifest['fit_config'])
    known = {k: scenario[k] for k in ('signal_variance', 'noise_variance', 'lengthscale')} if scenario['fit_mode'] == 'correlation_only' else None
    fit_start = perf_counter()
    marginals, joint = fit_models(xs, ys, config, known)
    am, av, ao = marginals[0].predict(evaluation)
    bm, bv, bo = joint.predict(evaluation) if joint.success else (am, av, ao)
    full_fit_prediction_seconds = perf_counter() - fit_start
    folds, validation_seconds = validate(xs, ys, fold_seed, config, known)
    penalties = {'independent': information_penalties(marginals[0], evaluation), 'joint': information_penalties(joint, evaluation)}
    gain = fitted_oracle_gain(joint, evaluation)
    bootstrap = None
    if scenario['bootstrap_outer_indices'] and (phase == 'pilot' or rep in scenario['bootstrap_outer_indices']):
        count = manifest['pilot_bootstrap_draws'] if phase == 'pilot' else manifest['bootstrap_draws']
        bootstrap = bootstrap_penalties(joint, evaluation, scenario['id'], rep, phase, count)
    noise_rng = np.random.default_rng(seed_for(study, phase, scenario['design_id'], rep, 'test_noise'))
    observed = latent + noise_rng.normal(scale=np.sqrt(scenario['noise_variance'][0]), size=len(latent))

    def metrics(mean, var):
        return {'latent_mse': float(np.mean((latent - mean) ** 2)), 'observed_mse': float(np.mean((observed - mean) ** 2)), 'mlpd': float(np.mean(-0.5 * (np.log(2 * np.pi * var) + (observed - mean) ** 2 / var))), 'coverage95': float(np.mean(np.abs(observed - mean) <= 1.959963984540054 * np.sqrt(var))), 'width95': float(np.mean(2 * 1.959963984540054 * np.sqrt(var)))}
    ri, rj = (float(np.mean(dgp.joint_variance + (oj - m) ** 2)) for m in (am, bm))
    oracle_gain = float(np.mean(dgp.ind_variance - dgp.joint_variance))
    hi, hj = (float(np.mean((m - o) ** 2)) for m, o in ((am, oi), (bm, oj)))
    mass = {f'W1{j + 1}': float(np.sum(kernel_and_logscale_derivatives(xs[0], xs[j], np.array([scenario['lengthscale']]), scenario['kernel'], False)[0] ** 2)) for j in range(1, len(xs))}
    record = {'experiment_id': study, 'phase': phase, 'scenario': scenario, 'replication': rep, 'dataset_seed': seed, 'fold_seed': fold_seed, 'optimizer_starts': 'fixed_complete_vectors_recorded_per_fit', 'marginals': [f.record() for f in marginals], 'joint': joint.record(), 'folds': folds, 'penalties': penalties, 'fitted_oracle_gain': gain, 'bootstrap': bootstrap, 'assessment': {'risk_independent': ri, 'risk_joint': rj, 'oracle_gain': oracle_gain, 'oracle_independent_variance': float(np.mean(dgp.ind_variance)), 'oracle_joint_variance': float(np.mean(dgp.joint_variance)), 'independent_penalty_draw': hi, 'joint_penalty_draw': hj, 'risk_difference_draw': ri - rj, 'accounting_rhs_draw': oracle_gain - (hj - hi), 'accounting_cross_term_draw': ri - rj - (oracle_gain - (hj - hi)), 'independent_metrics': metrics(am, ao), 'joint_metrics': metrics(bm, bo), 'true_scale_mass': mass}, 'runtime': {'full_fit_and_prediction': full_fit_prediction_seconds, 'four_fold_validation': validation_seconds, 'research_total': perf_counter() - started}}
    arrays = path.with_suffix('').with_suffix('.npz')
    temporary = arrays.with_name(arrays.name + f'.tmp.{os.getpid()}')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, train_x=np.vstack(xs), train_y=np.concatenate(ys), output_ids=np.repeat(np.arange(len(xs)), [len(x) for x in xs]), evaluation=evaluation, latent_eval=latent, observed_eval=observed, oracle_joint=oj, oracle_independent=oi, oracle_joint_variance=dgp.joint_variance, oracle_independent_variance=dgp.ind_variance, independent_mean=am, joint_mean=bm, independent_observed_variance=ao, joint_observed_variance=bo)
    temporary.replace(arrays)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with gzip.open(temporary, 'wt') as stream:
        stream.write(canonical(record))
    temporary.replace(path)
    return {'skipped': False, 'seconds': record['runtime']['research_total']}
