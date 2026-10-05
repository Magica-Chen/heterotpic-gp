"""Section7 transfer for the multivariate kriging experiments."""
from experiments.common import ROOT as PACKAGE_ROOT
from dataclasses import replace
from functools import lru_cache
import gzip
import json
import os
from time import perf_counter
import numpy as np
from experiments.exact_gp import FitConfig
from experiments.common import canonical
from experiments.synthetic import GaussianDGP, seed_for
from experiments.task_screen_study import task_scenarios
from experiments import robustness
from experiments import scale_design as scale_study
from experiments.pipeline import execute, coordinate_sensitivity
ROOT = PACKAGE_ROOT
ID = 'section7_alignment_v1'
METHOD = ROOT / 'configuration' / ID
RUN = ROOT / 'runs' / ID
STUDIES = ('balanced', 'robustness', 'scales', 'q1', 'q2')
REPS = dict(balanced=200, robustness=200, scales=100, q1=200, q2=200)

def scenarios(study):
    if study == 'balanced':
        return [{**s, 'n2': s['n1']} for s in task_scenarios()]
    if study == 'robustness':
        return robustness.scenarios()
    if study == 'scales':
        return scale_study.scenarios()
    if study == 'q1':
        return [dict(id=g, geometry=g, n1=14, n2=14) for g in ('isotopic', 'interleaved', 'separated')]
    return [dict(id=f'Q{i + 1:02d}', n1=n, n2=60, geometry=g) for i, (n, g) in enumerate(((n, g) for n in (8, 12, 20) for g in ('distributed', 'lower_half', 'outside')))]

def branches(study):
    return ('ard', 'isotropic') if study == 'scales' else ('centred', 'linear') if study in ('q1', 'q2') else ('primary',)

def fit_config(study, branch):
    config = FitConfig(ard=branch == 'ard')
    if study in ('q1', 'q2'):
        config = replace(config, noise_variance_bounds=(1e-08, 2.0))
    if study == 'q1':
        config = replace(config, lengthscale_bounds=(0.005, 1.5))
    return config

@lru_cache(maxsize=8)
def balanced_dgp(scenario_json):
    s = json.loads(scenario_json)
    n, n2 = (s['n1'], s['n2'])
    x = np.linspace(0, 1, n)[:, None]
    y = x.copy() if s['geometry'] == 'colocated' else ({'infill': 0.0, 'partial': 0.5, 'remote': 2.0}[s['geometry']] + (np.arange(n2) + 0.5) / n2)[:, None]
    return GaussianDGP([x, y], ((np.arange(201) + 0.5) / 201)[:, None], [(np.array([[1.0, s['rho']], [s['rho'], 1.0]]), 0.15, 'matern32')], s['noise_variance'])

def queue_path(study, s, rep):
    if study == 'q1':
        return ROOT / 'data/queue_inputs/q1' / s['id'] / f'dataset_{rep:04d}.npz'
    return ROOT / 'data/queue_inputs/q2' / s['id'] / f'dataset_{rep:04d}.json.gz'

def load_data(study, s, phase, rep):
    seed = seed_for(ID, phase, study + '_' + s['id'], rep)
    if study in ('q1', 'q2'):
        path = queue_path(study, s, rep)
        if study == 'q1':
            with np.load(path) as a:
                ids = a['output_ids']
                xs = [a['train_x'][ids == i] for i in range(2)]
                ys = [a['train_y'][ids == i] for i in range(2)]
                nv = [a['simulation_variance'][ids == i] for i in range(2)]
                evaluation, truth, observed = (a['evaluation'], a['analytic_target'], a['observed_eval'])
            membership = None
        else:
            with gzip.open(path, 'rt') as stream:
                record = json.load(stream)
            xs, ys, nv = [[np.asarray(v) for v in record[k]] for k in ('train_x', 'train_y', 'simulation_variance')]
            evaluation, truth = (np.asarray(record['evaluation']), np.asarray(record['reference_mean']))
            observed = None
            membership = np.empty(len(xs[0]), int)
            for f in record['folds']:
                if f['branch'] == 'linear':
                    membership[f['validation_target_indices']] = f['fold']
        fold_seed = seed_for(ID, 'queue_target_folds', f"{study}_n{s['n1']}", rep)
        return dict(xs=xs, ys=ys, noise=nv, evaluation=evaluation, latent=truth, observed=observed, oracle=truth, oracle_variance=np.zeros(len(truth)), seed=None, fold_seed=fold_seed, membership=membership, risk_kind='analytic_mean_mse' if study == 'q1' else 'independent_simulation_reference_mse')
    dgp = balanced_dgp(canonical(s)) if study == 'balanced' else robustness.generating_model(canonical(s)) if study == 'robustness' else scale_study.generating_model(canonical(s))
    ys, latent, oracle, _ = dgp.draw(seed)
    rng = np.random.default_rng(seed_for(ID, phase, study + '_' + s['id'], rep, 'observation_noise'))
    heavy = study == 'robustness' and s['family'] == 'heavy_tailed'
    if heavy:
        ys = [y + np.sqrt(0.05 * 3 / 5) * rng.standard_t(5, size=len(y)) for y in ys]
        observed = latent + np.sqrt(0.05 * 3 / 5) * rng.standard_t(5, size=len(latent))
        oracle, oracle_variance = (latent, np.zeros(len(latent)))
    else:
        noise = s['noise_variance'][0] if study == 'balanced' else 0.05
        if study == 'robustness' and s['family'] == 'heteroscedastic':
            noise = 0.01 + 0.19 * dgp.evaluation[:, 0] ** 2
        observed = latent + np.sqrt(noise) * rng.standard_normal(len(latent))
        oracle_variance = dgp.joint_variance
    return dict(xs=dgp.xs, ys=ys, noise=None, evaluation=dgp.evaluation, latent=latent, observed=observed, oracle=oracle, oracle_variance=oracle_variance, seed=seed, fold_seed=seed_for(ID, phase, study + '_' + s['id'], rep, 'folds'), membership=None, risk_kind='realised_latent_mse' if heavy else 'conditional_latent_risk')

def metrics(prediction, data):
    mean, latent_var, obs_var = prediction
    truth, observed = (data['latent'], data['observed'])
    result = dict(conditional_risk=float(np.mean(data['oracle_variance'] + (data['oracle'] - mean) ** 2)), latent_mse=float(np.mean((truth - mean) ** 2)), latent_coverage95=float(np.mean(np.abs(truth - mean) <= 1.959963984540054 * np.sqrt(latent_var))), latent_width95=float(np.mean(2 * 1.959963984540054 * np.sqrt(latent_var))))
    if observed is not None:
        result.update(observed_mse=float(np.mean((observed - mean) ** 2)), coverage95=float(np.mean(np.abs(observed - mean) <= 1.959963984540054 * np.sqrt(obs_var))), width95=float(np.mean(2 * 1.959963984540054 * np.sqrt(obs_var))), mlpd=float(np.mean(-0.5 * (np.log(2 * np.pi * obs_var) + (observed - mean) ** 2 / obs_var))))
    return result

def write_record(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with gzip.open(temp, 'wt') as stream:
        stream.write(canonical(row))
    temp.replace(path)

def run_one(task):
    study, s, phase, rep, manifest = task
    out = RUN / phase / study / s['id'] / f'dataset_{rep:04d}.json.gz'
    if out.exists():
        return dict(skipped=True, seconds=0.0)
    started = perf_counter()
    data = load_data(study, s, phase, rep)
    records, predictions = ({}, {})
    sensitivity = None
    for branch in branches(study):
        row, pred, diagonal = execute(data['xs'], data['ys'], data['evaluation'], data['fold_seed'], config=fit_config(study, branch), noise=data['noise'], linear=branch == 'linear', membership=data['membership'], future_noise=study == 'q1', return_diagonal=True)
        row.update(scenario=s, replication=rep, phase=phase, assessment={name: metrics(p, data) for name, p in pred.items()})
        records[branch], predictions[branch] = (row, pred)
        if study == 'scales' and branch == 'ard' and (rep < 10):
            sensitivity = coordinate_sensitivity(diagonal, data['evaluation'], seed_for(ID, phase, s['id'], rep, 'scale_box'), dense=s['dimension'] == 2 and rep < 2)
    row = dict(study=study, scenario=s, phase=phase, replication=rep, dataset_seed=data['seed'], fold_seed=data['fold_seed'], risk_kind=data['risk_kind'], branches=records, predictions=predictions, coordinate_sensitivity=sensitivity, data={k: data[k] for k in ('xs', 'ys', 'noise', 'evaluation', 'latent', 'observed', 'oracle', 'oracle_variance')}, runtime_seconds=perf_counter() - started)
    write_record(out, row)
    return dict(skipped=False, seconds=row['runtime_seconds'])
