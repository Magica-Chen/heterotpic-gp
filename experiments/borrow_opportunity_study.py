"""Borrow opportunity study for the multivariate kriging experiments."""
from __future__ import annotations
from dataclasses import replace
import gzip
from itertools import product
import json
import os
from time import perf_counter, process_time
import numpy as np
from scipy.stats import qmc
from experiments.opportunity_design import e8_dgp
from experiments.exact_gp import FitConfig, fit_gp
from experiments.noise_aware_gp import fit_noise_gp
from experiments.queue_simulation import simulate
from experiments.queues import tandem_run
from experiments.rich_gp import fit_rich
from experiments.common import ROOT, canonical, clean_json
from experiments.synthetic import seed_for
IDS = {'e8': 'E8_opportunity_v2', 'q2': 'Q2_opportunity_v2'}
MODELS = ('independent', 'mixture_independent', 'separable', 'rich_joint')
BASELINES = dict(independent='independent', mixture_independent='mixture_independent', separable='independent', rich_joint='mixture_independent')

def scenarios(study):
    if study == 'e8':
        return [dict(id=f'B{i:02d}', family=f, n1=n, n2=n * r, geometry=g) for i, (f, n, r, g) in enumerate(product(('lmc_positive', 'ar_fidelity'), (8, 16), (1, 4), ('interleaved', 'separated')), 1)]
    return [dict(id=f'Q{i:02d}', n1=n, n2=60, geometry=g) for i, (n, g) in enumerate(product((8, 12, 20), ('distributed', 'lower_half', 'outside')), 1)]

def q2_design(case, phase, replication):
    target = qmc.LatinHypercube(2, seed=seed_for(IDS['q2'], phase, f"target_n{case['n1']}", replication, 'design')).random(case['n1'])
    auxiliary = qmc.LatinHypercube(2, seed=seed_for(IDS['q2'], phase, 'auxiliary', replication, 'design')).random(case['n2'])
    if case['geometry'] == 'lower_half':
        auxiliary[:, 0] *= 0.5
    elif case['geometry'] == 'outside':
        auxiliary[:, 0] = -0.75 + 0.5 * auxiliary[:, 0]
    return [target, auxiliary]

def target_folds(case, phase, replication):
    membership = np.empty(case['n1'], dtype=int)
    order = np.random.default_rng(seed_for(IDS['q2'], phase, f"target_n{case['n1']}", replication, 'folds')).permutation(case['n1'])
    membership[order] = np.arange(case['n1']) % 4
    return membership

def choose(sse):
    baseline = 'mixture_independent' if sse['mixture_independent'] < 0.98 * sse['independent'] else 'independent'
    coupled = min(('separable', 'rich_joint'), key=lambda k: (sse[k], MODELS.index(k)))
    return coupled if sse[coupled] < 0.98 * sse[baseline] else baseline

def q2_stage(xs, ys, nv, evaluation, linear):
    centres = np.array([v.mean() for v in ys])
    scales = np.array([max(v.std(ddof=1), 1e-08) for v in ys])
    z = [(y - c) / s for y, c, s in zip(ys, centres, scales)]
    noise = [v / s ** 2 for v, s in zip(nv, scales)]
    config = replace(FitConfig(), noise_variance_bounds=(1e-08, 2.0))
    settings = dict(config=config, linear_mean=linear)
    fits = dict(independent=fit_noise_gp([xs[0]], [z[0]], [noise[0]], **settings), mixture_independent=fit_noise_gp([xs[0]], [z[0]], [noise[0]], kind='mixture', **settings), separable=fit_noise_gp(xs, z, noise, **settings), rich_joint=fit_noise_gp(xs, z, noise, kind='ar', **settings))
    predictions = {}
    for name, fit in fits.items():
        actual = fit if fit.success else fits[BASELINES[name]]
        mean, variance, observed = actual.predict(evaluation)
        predictions[name] = np.array([mean * scales[0] + centres[0], variance * scales[0] ** 2, observed * scales[0] ** 2])
    record = dict(centres=centres, scales=scales, linear_mean=linear, fits={k: v.record() for k, v in fits.items()}, predictions=predictions)
    return record

def e8_dataset(case, phase, replication):
    dgp = e8_dgp(case)
    seed = seed_for(IDS['e8'], phase, case['id'], replication)
    ys, latent, oj, oi = dgp.draw(seed)
    xs, evaluation = (dgp.xs, dgp.evaluation)
    config = FitConfig()
    fits = dict(independent=fit_gp([xs[0]], [ys[0]], config), mixture_independent=fit_rich([xs[0]], [ys[0]], 'mixture', config), separable=fit_gp(xs, ys, config), rich_joint=fit_rich(xs, ys, 'lmc' if case['family'] == 'lmc_positive' else 'ar', config))
    predictions, assessment = ({}, {})
    for name, fit in fits.items():
        prediction = (fit if fit.success else fits[BASELINES[name]]).predict(evaluation)
        mean, variance, observed = prediction
        predictions[name] = prediction
        own_oracle = oi if name == BASELINES[name] else oj
        assessment[name] = dict(risk=float(np.mean(dgp.joint_variance + (oj - mean) ** 2)), latent_mse=float(np.mean((latent - mean) ** 2)), own_oracle_penalty=float(np.mean((own_oracle - mean) ** 2)), baseline=BASELINES[name])
    return dict(seed=seed, train_x=xs, train_y=ys, evaluation=evaluation, latent=latent, oracle_joint=oj, oracle_independent=oi, oracle_joint_variance=dgp.joint_variance, oracle_independent_variance=dgp.ind_variance, fits={k: v.record() for k, v in fits.items()}, predictions=predictions, assessment=assessment)

def q2_dataset(case, phase, replication, reference):
    xs = q2_design(case, phase, replication)
    simulation_phase = f"{IDS['q2']}_{phase}_n{case['n1']}"
    runs, seeds, costs = simulate(xs, simulation_phase, case['geometry'], replication)
    ys = [v.mean(axis=1) for v in runs]
    nv = [v.var(axis=1, ddof=1) / v.shape[1] for v in runs]
    evaluation = np.array([r['normalized'] for r in reference['points']])
    reference_mean = np.array([r['mean'] for r in reference['points']])
    membership = target_folds(case, phase, replication)
    full, folds, predictions, assessment = ({}, [], {}, {})
    for branch, linear in (('linear', True), ('centred', False)):
        full[branch] = q2_stage(xs, ys, nv, evaluation, linear)
        sse = dict.fromkeys(MODELS, 0.0)
        for fold in range(4):
            keep = membership != fold
            xx, yy, vv = ([xs[0][keep], xs[1]], [ys[0][keep], ys[1]], [nv[0][keep], nv[1]])
            stage = q2_stage(xx, yy, vv, xs[0][~keep], linear)
            local_sse = {k: float(np.sum((stage['predictions'][k][0] - ys[0][~keep]) ** 2)) for k in MODELS}
            for k in MODELS:
                sse[k] += local_sse[k]
            folds.append(dict(branch=branch, fold=fold, train_indices=[np.flatnonzero(keep), np.arange(len(xs[1]))], validation_target_indices=np.flatnonzero(~keep), sse=local_sse, stage=stage))
        selected = choose(sse)
        full[branch].update(validation_sse=sse, selected=selected)
        for name in (*MODELS, 'selected'):
            fitted_name = selected if name == 'selected' else name
            prediction = full[branch]['predictions'][fitted_name]
            key = f'{branch}_{name}'
            baseline = 'independent' if name == 'selected' else BASELINES[name]
            predictions[key] = prediction
            assessment[key] = dict(risk=float(np.mean((prediction[0] - reference_mean) ** 2)), baseline=f'{branch}_{baseline}', fitted_method=fitted_name)
    return dict(train_x=xs, train_y=ys, train_run_values=runs, simulation_variance=nv, simulation_seeds=seeds, simulation_costs=costs, evaluation=evaluation, reference_mean=reference_mean, full=full, folds=folds, predictions=predictions, assessment=assessment)

def reference_point(task):
    index, point = task
    physical = np.array(point) * np.array([0.4, 1.25]) + np.array([0.35, 0.25])
    seeds = [seed_for(IDS['q2'], 'reference', str(index), i) for i in range(128)]
    start = process_time()
    values = [tandem_run(*physical, np.random.default_rng(seed), 100000, 10000, True) for seed in seeds]
    return dict(index=index, normalized=point, physical=physical, seeds=seeds, run_values=values, runs=128, customers=100000, warmup=10000, mean=float(np.mean(values)), mcse=float(np.std(values, ddof=1) / np.sqrt(128)), cpu_seconds=process_time() - start)

def run_one(task):
    study, phase, case, replication, manifest, reference = task
    directory = ROOT / 'runs' / IDS[study] / phase / case['id']
    path = directory / f'dataset_{replication:04d}.json.gz'
    if path.exists():
        with gzip.open(path, 'rt') as stream:
            record = json.load(stream)
        return
    start = perf_counter()
    record = e8_dataset(case, phase, replication) if study == 'e8' else q2_dataset(case, phase, replication, reference)
    record.update(experiment_id=IDS[study], phase=phase, case=case, replication=replication, elapsed_seconds=perf_counter() - start)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with gzip.open(temporary, 'wt') as stream:
        stream.write(canonical(clean_json(record)))
    temporary.replace(path)
