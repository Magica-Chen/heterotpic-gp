"""Analyze borrow opportunity for the multivariate kriging experiments."""
import argparse
from collections import Counter
import gzip
import json
import numpy as np
from experiments.borrow_opportunity_study import IDS, BASELINES, MODELS
from experiments.common import ROOT, clean_json
from experiments.synthetic import seed_for

def interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()

def weights(size, key, replicates=2000):
    rng = np.random.default_rng(seed_for('borrow_opportunity_analysis', 'bootstrap', key, 0))
    return rng.multinomial(size, np.full(size, 1 / size), size=replicates) / size

def reference_bootstrap(reference, replicates=2000):
    rng = np.random.default_rng(seed_for(IDS['q2'], 'bootstrap', 'reference', 0))
    values = []
    for point in reference['points']:
        runs = np.asarray(point['run_values'])
        draws = rng.integers(0, len(runs), size=(replicates, len(runs)))
        values.append(runs[draws].mean(axis=1))
    return np.array(values).T

def queue_bootstrap_mse(predictions, sampled_weights, sampled_reference):
    means = np.einsum('bn,nmg->bmg', sampled_weights, predictions, optimize=True)
    second = sampled_weights @ np.mean(predictions ** 2, axis=2)
    cross = np.mean(means * sampled_reference[:, None, :], axis=2)
    return second - 2 * cross + np.mean(sampled_reference ** 2, axis=1)[:, None]

def count_fits(records, study):
    count = Counter()
    for record in records:
        if study == 'e8':
            fits = list(record['fits'].values())
        else:
            stages = [*record['full'].values(), *[f['stage'] for f in record['folds']]]
            fits = [fit for stage in stages for fit in stage['fits'].values()]
        for fit in fits:
            count['attempted'] += 1
            count['successful'] += int(fit['success'])
            count['active_bound'] += int(bool(fit['active_bounds']))
            gradient = fit.get('projected_gradient_norm')
            count['above_gradient_tolerance'] += int(gradient is not None and gradient > 1e-05)
            count['failed_starts'] += sum((not s['success'] for s in fit['starts']))
            jitter = fit.get('jitter')
            count['jitter_escalations'] += int(jitter is not None and jitter > 1.00001e-10)
    return dict(count)

def analyze_e8(case, records):
    names = list(MODELS)
    risk = np.array([[r['assessment'][m]['risk'] for m in names] for r in records])
    penalty = np.array([[r['assessment'][m]['own_oracle_penalty'] for m in names] for r in records])
    w = weights(len(records), f"e8_{case['id']}")
    boot = w @ risk
    oracle_ind = np.mean(records[0]['oracle_independent_variance'])
    oracle_joint = np.mean(records[0]['oracle_joint_variance'])
    for record in records:
        for method in names:
            mean = np.array(record['predictions'][method][0])
    rows = []
    for i, method in enumerate(names):
        j = names.index(BASELINES[method])
        mean = risk.mean(axis=0)
        relative = 100 * (1 - mean[i] / mean[j])
        relative_boot = 100 * (1 - boot[:, i] / boot[:, j])
        rows.append(dict(method=method, baseline=BASELINES[method], risk=float(mean[i]), risk_ci95=interval(boot[:, i]), risk_reduction_percent=float(relative), risk_reduction_ci95=interval(relative_boot), oracle_deviation=float(penalty[:, i].mean()), excess_oracle_deviation=float(np.mean(penalty[:, i] - penalty[:, j])), mean_accounting_residual=float(np.mean(risk[:, j] - risk[:, i]) - (oracle_ind - oracle_joint - np.mean(penalty[:, i] - penalty[:, j]))) if i != j else None))
    return dict(case=case, datasets=len(records), results=rows, oracle_independent_risk=float(oracle_ind), oracle_joint_risk=float(oracle_joint), oracle_gain_percent=float(100 * (1 - oracle_joint / oracle_ind)), fit_diagnostics=count_fits(records, 'e8'))

def analyze_q2(case, records, reference, reference_draws, w):
    names = list(records[0]['predictions'])
    raw = np.array([[r['predictions'][m][0] for m in names] for r in records])
    simple_names = names.copy()
    for branch in ('linear', 'centred'):
        independent_selected = []
        for record in records:
            sse = record['full'][branch]['validation_sse']
            selected = 'mixture_independent' if sse['mixture_independent'] < 0.98 * sse['independent'] else 'independent'
            independent_selected.append(record['predictions'][f'{branch}_{selected}'][0])
        raw = np.concatenate([raw, np.array(independent_selected)[:, None, :]], axis=1)
        names.append(f'{branch}_independent_selected')
    truth = np.array([p['mean'] for p in reference['points']])
    risks = np.mean((raw - truth[None, None, :]) ** 2, axis=2)
    boot = queue_bootstrap_mse(raw, w, reference_draws)
    mean = risks.mean(axis=0)
    rows = []
    for i, method in enumerate(names):
        branch = method.split('_', 1)[0]
        if method in simple_names:
            baseline = records[0]['assessment'][method]['baseline']
            j = names.index(baseline)
        else:
            baseline = f'{branch}_independent'
            j = names.index(baseline)
        row = dict(method=method, baseline=baseline, mse=float(mean[i]), mse_ci95=interval(boot[:, i]), rmse=float(np.sqrt(mean[i])), rmse_ci95=interval(np.sqrt(np.maximum(boot[:, i], 0))), risk_reduction_percent=float(100 * (1 - mean[i] / mean[j])), risk_reduction_ci95=interval(100 * (1 - boot[:, i] / boot[:, j])))
        if method == f'{branch}_selected':
            j = names.index(f'{branch}_independent_selected')
            row.update(coupling_advantage_over_independent_selection_percent=float(100 * (1 - mean[i] / mean[j])), coupling_advantage_ci95=interval(100 * (1 - boot[:, i] / boot[:, j])), selected_counts=dict(Counter((r['full'][branch]['selected'] for r in records))))
        rows.append(row)
    return dict(case=case, datasets=len(records), results=rows, fit_diagnostics=count_fits(records, 'q2'), mean_simulation_cpu_seconds=[float(np.mean([r['simulation_costs'][j]['cpu_seconds'] for r in records])) for j in (0, 1)], mean_full_fit_seconds={branch: {name: float(np.mean([r['full'][branch]['fits'][name]['runtime_seconds'] for r in records])) for name in MODELS} for branch in ('linear', 'centred')})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', choices=tuple(IDS), required=True)
    args = parser.parse_args()
    directory = ROOT / 'runs' / IDS[args.study]
    manifest_path = directory / 'evaluation_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    spec = manifest['spec']
    expected_count = spec['replicates'] * len(spec['cases'])
    reference = None
    if args.study == 'q2':
        rp = directory / 'reference.json'
        reference = json.loads(rp.read_text())
        rb = reference_bootstrap(reference)
        w = weights(spec['replicates'], 'q2_macro_datasets')
    total = Counter()
    cases = []
    for case in spec['cases']:
        files = sorted((directory / 'evaluation' / case['id']).glob('dataset_*.json.gz'))
        records = []
        for index, path in enumerate(files):
            with gzip.open(path, 'rt') as stream:
                record = json.load(stream)
            records.append(record)
        result = analyze_e8(case, records) if args.study == 'e8' else analyze_q2(case, records, reference, rb, w)
        total.update(result['fit_diagnostics'])
        cases.append(result)
        del records
        print(args.study, case['id'], [(r['method'], round(r['risk_reduction_percent'], 2), [round(v, 2) for v in r['risk_reduction_ci95']]) for r in result['results'] if r['method'] != r['baseline']], flush=True)
    output = ROOT / 'results' / IDS[args.study]
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(experiment_id=IDS[args.study], datasets=expected_count, cases=cases, fit_diagnostics=dict(total), uncertainty='2000 paired dataset resamples within each E8 cell; Q2 macro weights shared across all cells plus shared independent reference-run resamples. Intervals are pointwise.', timing_scope='Recorded fitting/simulator components on concurrently running workers; no deployed wall-time speedup inferred.')
    if reference:
        summary['reference'] = dict(points=len(reference['points']), runs_per_point=128, mean_mcse=float(np.mean([p['mcse'] for p in reference['points']])), maximum_mcse=float(max((p['mcse'] for p in reference['points']))), cpu_seconds=float(sum((p['cpu_seconds'] for p in reference['points']))))
    (output / 'summary.json').write_text(json.dumps(clean_json(summary), indent=2) + '\n')
if __name__ == '__main__':
    main()
