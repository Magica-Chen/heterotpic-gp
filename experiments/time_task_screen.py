"""Time task screen for the multivariate kriging experiments."""
from __future__ import annotations
import argparse
import gzip
import json
import os
from pathlib import Path
import platform
import numpy as np
from experiments.analyze_task_screen import bootstrap_means, interval
from experiments.exact_gp import FitConfig
from experiments.task_screening import TaskScreenConfig, run_pipeline
from experiments.task_screen_study import ROOT, METHOD

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/E6_task_v2')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/E6_task_v2/timing')
    args = parser.parse_args()
    if any((os.getenv(v) != '1' for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'))):
        raise RuntimeError('Timing requires one BLAS/OMP thread.')
    method = json.loads((METHOD / 'method.json').read_text())
    thresholds = json.loads((METHOD / 'thresholds.json').read_text())
    config = TaskScreenConfig(**{**method['screen_config'], 'threshold': thresholds['chosen']['screen']['threshold']})
    fit_config = FitConfig(**method['fit_config'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, rows = ([], [])
    for s in method['scenarios']:
        for rep in method['timing']['replication_indices']:
            path = args.run_dir / 'evaluation' / s['id'] / f'dataset_{rep:04d}.json.gz'
            with gzip.open(path, 'rt') as stream:
                row = json.load(stream)
            data_path = path.with_suffix('').with_suffix('.npz')
            out = args.output_dir / f"{s['id']}_{rep:04d}.json"
            if out.exists():
                record = json.loads(out.read_text())
            else:
                data = np.load(data_path)
                xs = [data['train_x'][data['output_ids'] == i] for i in range(2)]
                ys = [data['train_y'][data['output_ids'] == i] for i in range(2)]
                order = ['direct', 'screen'] if len(results) % 2 == 0 else ['screen', 'direct']
                record = {'scenario_id': s['id'], 'replication': rep, 'order': order, 'measurements': {}}
                for name in order:
                    actual = run_pipeline(xs, ys, data['evaluation'], row['fold_seed'], fit_config, config, direct=name == 'direct')
                    record['measurements'][name] = {'wall_seconds': actual['runtime_seconds'], 'selected_model': actual['selected_model'], 'counts': actual['counts']}
                out.write_text(json.dumps(record, indent=2) + '\n')
            results.append(record)
            rows.append(row)
            if len(results) % 24 == 0:
                print(f'Timed {len(results)}/240 pairs', flush=True)
    values = np.array([[r['measurements'][name]['wall_seconds'] for name in ('direct', 'screen')] for r in results])
    boot = bootstrap_means(values, rows, 'timing')
    summary = {'pairs': len(results), 'direct_seconds': interval(values[:, 0].mean(), boot[:, 0]), 'screen_seconds': interval(values[:, 1].mean(), boot[:, 1]), 'saving_percent': interval(100 * (1 - values[:, 1].mean() / values[:, 0].mean()), 100 * (1 - boot[:, 1] / boot[:, 0])), 'paired_difference_seconds': interval(np.mean(values[:, 1] - values[:, 0]), boot[:, 1] - boot[:, 0]), 'hardware': {'platform': platform.platform(), 'cpu': next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown'), 'workers': 1, 'OPENBLAS_NUM_THREADS': os.getenv('OPENBLAS_NUM_THREADS'), 'OMP_NUM_THREADS': os.getenv('OMP_NUM_THREADS')}, 'scope': method['timing']['includes'], 'by_geometry': {}}
    for geometry in ('colocated', 'infill', 'partial', 'remote'):
        mask = np.array([r['scenario']['geometry'] == geometry for r in rows])
        b = bootstrap_means(values[mask], [r for r, keep in zip(rows, mask) if keep], 'timing')
        mean = values[mask].mean(axis=0)
        summary['by_geometry'][geometry] = interval(100 * (1 - mean[1] / mean[0]), 100 * (1 - b[:, 1] / b[:, 0]))
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
if __name__ == '__main__':
    main()
