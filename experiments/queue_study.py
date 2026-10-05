"""Queue study for the multivariate kriging experiments."""
from __future__ import annotations
from experiments.common import ROOT as PACKAGE_ROOT
from dataclasses import replace
import gzip
import json
from time import perf_counter, process_time
import numpy as np
from scipy.stats import qmc
from experiments.queues import tandem_run
from experiments.queue_simulation import simulate
from experiments.pipeline import execute, SCREEN
from experiments.transfer_study import fit_config, metrics, write_record
from experiments.synthetic import seed_for
ROOT = PACKAGE_ROOT
ID = 'Q2_calibrated_v3'
METHOD = ROOT / 'configuration' / ID
RUN = ROOT / 'runs' / ID
OUT = ROOT / 'results' / ID
BRANCHES = ('centred', 'linear')
REPS = dict(pilot=1, calibration=100, evaluation=200)
THRESHOLDS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)

def cases():
    return [dict(id=f'Q{i + 1:02d}', n1=n, n2=60, geometry=g) for i, (n, g) in enumerate(((n, g) for n in (8, 12, 20) for g in ('distributed', 'lower_half', 'outside')))]

def design(case, phase, replication):
    target = qmc.LatinHypercube(2, seed=seed_for(ID, phase, f"target_n{case['n1']}", replication, 'design')).random(case['n1'])
    auxiliary = qmc.LatinHypercube(2, seed=seed_for(ID, phase, 'auxiliary', replication, 'design')).random(case['n2'])
    if case['geometry'] == 'lower_half':
        auxiliary[:, 0] *= 0.5
    elif case['geometry'] == 'outside':
        auxiliary[:, 0] = -0.75 + 0.5 * auxiliary[:, 0]
    return [target, auxiliary]

def folds(case, phase, replication):
    seed = seed_for(ID, phase, f"target_n{case['n1']}", replication, 'folds')
    membership = np.empty(case['n1'], int)
    order = np.random.default_rng(seed).permutation(case['n1'])
    membership[order] = np.arange(case['n1']) % 4
    return (seed, membership)

def ref_path(phase):
    return ROOT / 'data/queue_reference' / f'reference_{phase}.json'

def reference_point(task):
    phase, index, point = task
    physical = np.array(point) * np.array([0.4, 1.25]) + np.array([0.35, 0.25])
    seeds = [seed_for(ID, 'reference_' + phase, str(index), i) for i in range(128)]
    start = process_time()
    values = [tandem_run(*physical, np.random.default_rng(seed), 100000, 10000, True) for seed in seeds]
    return dict(index=index, normalized=point, physical=physical, seeds=seeds, run_values=values, runs=128, customers=100000, warmup=10000, mean=float(np.mean(values)), mcse=float(np.std(values, ddof=1) / np.sqrt(128)), cpu_seconds=process_time() - start)

def primary_config():
    path = METHOD / 'selection.json'
    result = json.loads(path.read_text())
    return result

def execute_application(data, branch, threshold, mode='screen'):
    mode = 'direct' if mode == 'screen' and threshold == 0.0 else mode
    return execute(data['xs'], data['ys'], data['evaluation'], data['fold_seed'], config=fit_config('q2', branch), screen=replace(SCREEN, threshold=threshold), noise=data['noise'], linear=branch == 'linear', mode=mode, membership=data['membership'])

def run_one(task):
    phase, case, rep, manifest, reference, selection = task
    out = RUN / phase / case['id'] / f'dataset_{rep:04d}.json.gz'
    if out.exists():
        return dict(skipped=True)
    started = perf_counter()
    xs = design(case, phase, rep)
    simulation_phase = f"{ID}_{phase}_n{case['n1']}"
    runs, seeds, costs = simulate(xs, simulation_phase, case['geometry'], rep)
    ys = [r.mean(axis=1) for r in runs]
    nv = [r.var(axis=1, ddof=1) / r.shape[1] for r in runs]
    evaluation = np.array([p['normalized'] for p in reference['points']])
    truth = np.array([p['mean'] for p in reference['points']])
    seed, membership = folds(case, phase, rep)
    data = dict(xs=xs, ys=ys, noise=nv, evaluation=evaluation, latent=truth, oracle=truth, oracle_variance=np.zeros(len(truth)), observed=None, fold_seed=seed, membership=membership)
    records, predictions = ({}, {})
    for branch in BRANCHES:
        row, pred = execute_application(data, branch, 0.2, mode='shadow')
        row.update(scenario=case, replication=rep, phase=phase, assessment={k: metrics(p, data) for k, p in pred.items()})
        records[branch], predictions[branch] = (row, pred)
    raw = dict(study='q2', experiment_id=ID, phase=phase, scenario=case, replication=rep, data=data, train_run_values=runs, simulation_seeds=seeds, simulation_costs=costs, branches=records, predictions=predictions, runtime_seconds=perf_counter() - started)
    write_record(out, raw)
    return dict(skipped=False)

def load_reference(phase):
    path = ref_path('calibration' if phase == 'pilot' else phase)
    result = json.loads(path.read_text())
    return result
