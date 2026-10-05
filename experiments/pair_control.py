"""Analyze pair control for the multivariate kriging experiments."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from experiments.analyze_task_screen import bootstrap_means, interval, load_records, replay
from experiments.exact_gp import FitConfig, fit_gp
from experiments.screening import fit_pair
from experiments.task_screening import TaskScreenConfig, fold_data, run_pipeline, target_folds
from experiments.task_screen_study import ROOT, METHOD

def pair_validate(xs, ys, evaluation, fold_seed, fit_config=FitConfig(), trace=False):
    started = perf_counter()
    auxiliary = fit_gp([xs[1]], [ys[1]], fit_config)
    membership = target_folds(xs[0], fold_seed)
    folds = []
    for index in range(4):
        tx, ty, vx, vy = fold_data(xs, ys, membership, index)
        own = fit_gp([tx[0]], [ty[0]], fit_config)
        joint = fit_pair(tx, ty, [own, auxiliary], fit_config)
        own_prediction = own.predict(vx)[0]
        joint_prediction = (joint if joint.success else own).predict(vx)[0]
        folds.append({'independent_sse': float(np.sum((vy - own_prediction) ** 2)), 'joint_sse': float(np.sum((vy - joint_prediction) ** 2)), 'target_marginal': own.record() if trace else None, 'joint': joint.record() if trace else None})
    selected = sum((f['joint_sse'] for f in folds)) < 0.98 * sum((f['independent_sse'] for f in folds))
    target = fit_gp([xs[0]], [ys[0]], fit_config)
    joint = fit_pair(xs, ys, [target, auxiliary], fit_config) if selected else None
    use_joint = bool(joint is not None and joint.success)
    prediction = (joint if use_joint else target).predict(evaluation)
    return {'selected_joint': use_joint, 'prediction': prediction, 'runtime_seconds': perf_counter() - started, 'folds': folds if trace else None, 'marginals': [target.record(), auxiliary.record()] if trace else None, 'joint': joint.record() if trace and joint is not None else None}

def source_data(run_dir, scenario, rep):
    path = Path(run_dir) / 'evaluation' / scenario['id'] / f'dataset_{rep:04d}.json.gz'
    with gzip.open(path, 'rt') as stream:
        row = json.load(stream)
    data_path = path.with_suffix('').with_suffix('.npz')
    data = np.load(data_path)
    xs = [data['train_x'][data['output_ids'] == i] for i in range(2)]
    ys = [data['train_y'][data['output_ids'] == i] for i in range(2)]
    return (row, data, xs, ys, None)

def run_one(task):
    run_dir, output_dir, scenario, rep, manifest = task
    out = Path(output_dir) / scenario['id'] / f'dataset_{rep:04d}.json.gz'
    arrays = out.with_suffix('').with_suffix('.npz')
    row, data, xs, ys, unused = source_data(run_dir, scenario, rep)
    if out.exists():
        return
    result = pair_validate(xs, ys, data['evaluation'], row['fold_seed'], trace=True)
    prediction = result.pop('prediction')
    mean, variance, observed_variance = prediction
    result.update(scenario=scenario, replication=rep, assessment={'conditional_risk': float(np.mean(data['oracle_variance'] + (data['oracle_mean'] - mean) ** 2)), 'latent_mse': float(np.mean((data['latent_eval'] - mean) ** 2)), 'observed_mse': float(np.mean((data['observed_eval'] - mean) ** 2)), 'coverage95': float(np.mean(np.abs(data['observed_eval'] - mean) <= 1.959963984540054 * np.sqrt(observed_variance))), 'width95': float(np.mean(2 * 1.959963984540054 * np.sqrt(observed_variance))), 'mlpd': float(np.mean(-0.5 * (np.log(2 * np.pi * observed_variance) + (data['observed_eval'] - mean) ** 2 / observed_variance)))})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = arrays.with_name(arrays.name + f'.tmp.{os.getpid()}')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, mean=mean, variance=variance, observed_variance=observed_variance)
    tmp.replace(arrays)
    tmp = out.with_name(out.name + f'.tmp.{os.getpid()}')
    with gzip.open(tmp, 'wt') as stream:
        json.dump(result, stream)
    tmp.replace(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'analyze', 'time'))
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/E6_task_v2')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/E6_task_pair_control_v1')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    settings_path = METHOD / 'pair_control.json'
    method = json.loads((METHOD / 'method.json').read_text())
    manifest = json.loads(settings_path.read_text())
    if args.action == 'run':
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        tasks = [(str(args.run_dir), str(args.output_dir), s, r, manifest) for s in method['scenarios'] for r in range(200)]
        done, exceptions, started = (0, [], perf_counter())
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(run_one, task): task for task in tasks}
            for future in as_completed(pending):
                task = pending[future]
                try:
                    future.result()
                    done += 1
                except Exception as error:
                    exceptions.append({'scenario': task[2]['id'], 'replication': task[3], 'error': repr(error)})
                if (done + len(exceptions)) % 240 == 0:
                    print(f'Pair validation: {done}/9600; {len(exceptions)} exceptions; {perf_counter() - started:.1f}s', flush=True)
        completion = {'completed': done, 'attempted': len(tasks), 'task_exceptions': exceptions, 'wall_seconds': perf_counter() - started}
        (args.output_dir / 'completion.json').write_text(json.dumps(completion, indent=2) + '\n')
        if exceptions:
            raise RuntimeError('Resolve comparator task exceptions before analysis.')
        return
    thresholds = json.loads((METHOD / 'thresholds.json').read_text())
    config = TaskScreenConfig(threshold=thresholds['chosen']['screen']['threshold'])
    result_dir = ROOT / 'results/E6_task_v2/pair_control'
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.action == 'analyze':
        completion = json.loads((args.output_dir / 'completion.json').read_text())
        _, rows, _ = load_records(args.run_dir, 'evaluation')
        controls, _ = ([], None)
        for row in rows:
            path = args.output_dir / row['scenario']['id'] / f"dataset_{row['replication']:04d}.json.gz"
            with gzip.open(path, 'rt') as stream:
                control = json.load(stream)
            controls.append(control)

        def summary(subset):
            rr, cc = ([rows[i] for i in subset], [controls[i] for i in subset])
            values = np.array([[r['assessment']['independent']['conditional_risk'], c['assessment']['conditional_risk'], r['assessment'][replay(r, 'screen', config.threshold)['model']]['conditional_risk'], c['assessment']['coverage95'], c['assessment']['width95'], c['assessment']['mlpd']] for r, c in zip(rr, cc)])
            b, mu = (bootstrap_means(values, rr, 'evaluation'), values.mean(axis=0))
            return {'risk': interval(mu[1], b[:, 1]), 'risk_reduction_percent': interval(100 * (1 - mu[1] / mu[0]), 100 * (1 - b[:, 1] / b[:, 0])), 'screen_risk_excess_percent': interval(100 * (mu[2] - mu[1]) / mu[0], 100 * (b[:, 2] - b[:, 1]) / b[:, 0]), 'coverage95': interval(100 * mu[3], 100 * b[:, 3]), 'width95': interval(mu[4], b[:, 4]), 'mlpd': interval(mu[5], b[:, 5]), 'datasets': len(rr), 'selected_joint': sum((c['selected_joint'] for c in cc))}
        report = {'pooled': summary(range(len(rows))), 'by_geometry': {}, 'by_correlation': {}}
        for field, name in [('geometry', 'by_geometry'), ('rho', 'by_correlation')]:
            for value in sorted(set((r['scenario'][field] for r in rows))):
                report[name][str(value)] = summary([i for i, r in enumerate(rows) if r['scenario'][field] == value])
        (result_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report['pooled'], indent=2))
    else:
        if any((os.getenv(v) != '1' for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'))):
            raise RuntimeError('Timing requires one BLAS/OMP thread.')
        records, rows = ([], [])
        for scenario in method['scenarios']:
            for rep in range(5):
                row, data, xs, ys, unused = source_data(args.run_dir, scenario, rep)
                path = args.output_dir / scenario['id'] / f'dataset_{rep:04d}.npz'
                control_json = path.with_suffix('.json.gz')
                with gzip.open(control_json, 'rt') as stream:
                    control = json.load(stream)
                out = result_dir / f"timing_{scenario['id']}_{rep:04d}.json"
                if out.exists():
                    record = json.loads(out.read_text())
                else:
                    order = ['pair', 'screen'] if len(records) % 2 == 0 else ['screen', 'pair']
                    record = {'scenario_id': scenario['id'], 'replication': rep, 'order': order, 'measurements': {}}
                    for name in order:
                        if name == 'pair':
                            actual = pair_validate(xs, ys, data['evaluation'], row['fold_seed'])
                        else:
                            actual = run_pipeline(xs, ys, data['evaluation'], row['fold_seed'], config=config)
                        record['measurements'][name] = {'wall_seconds': actual['runtime_seconds']}
                    out.write_text(json.dumps(record, indent=2) + '\n')
                records.append(record)
                rows.append(row)
                if len(records) % 24 == 0:
                    print(f'Timed {len(records)}/240 pair-control comparisons', flush=True)
        value = np.array([[r['measurements'][name]['wall_seconds'] for name in ('pair', 'screen')] for r in records])
        b, mu = (bootstrap_means(value, rows, 'timing_pair_control'), value.mean(axis=0))
        report = {'pairs': len(rows), 'pair_seconds': interval(mu[0], b[:, 0]), 'screen_seconds': interval(mu[1], b[:, 1]), 'screen_saving_percent': interval(100 * (1 - mu[1] / mu[0]), 100 * (1 - b[:, 1] / b[:, 0])), 'timing_repeats': 5}
        (result_dir / 'timing-summary.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
