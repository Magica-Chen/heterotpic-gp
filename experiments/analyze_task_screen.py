"""Analyze task screen for the multivariate kriging experiments."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import numpy as np
from experiments.screening import gate
from experiments.synthetic import seed_for
from experiments.task_screening import exclusion
from experiments.task_screen_study import ROOT, METHOD
THRESHOLDS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
BOOTSTRAPS = 2000
METHODS = ('independent', 'diagonal', 'joint', 'direct', 'screen', 'retain_all', 'loose', 'boundary', 'legacy', 'no_shortcut')

def load_records(run_dir, phase):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / f'{phase}_manifest.json').read_text())
    rows, _ = ([], None)
    for scenario in manifest['scenarios']:
        for rep in range(manifest['replicates']):
            path = run_dir / phase / scenario['id'] / f'dataset_{rep:04d}.json.gz'
            with gzip.open(path, 'rt') as stream:
                row = json.load(stream)
            rows.append(row)
    return (manifest, rows, None)

def independent_choice(row, fraction=0.02):
    return 'diagonal' if sum((f['diagonal_sse'] for f in row['folds'])) < (1 - fraction) * sum((f['independent_sse'] for f in row['folds'])) else 'independent'

def replay(row, method, threshold=0.0, fraction=0.02):
    full, folds = (row, row['folds'])
    base_cost = sum((m['runtime_seconds'] for m in full['marginals'])) + full['diagonal']['runtime_seconds']
    if method in ('independent', 'diagonal', 'joint'):
        name = method
        if method == 'independent':
            cost = full['marginals'][0]['runtime_seconds']
        else:
            cost = base_cost + (full['joint']['runtime_seconds'] if method == 'joint' else 0)
        if name == 'joint' and (not full['joint']['success']):
            name = 'diagonal'
        if name == 'diagonal' and (not full['diagonal']['success']):
            name = 'independent'
        return dict(model=name, full_excluded=False, fold_excluded=0, unresolved=False, component_seconds=cost + full[f'{name}_prediction_seconds'], joint_fits=int(method == 'joint'), marginal_fits=1 if method == 'independent' else 2, diagonal_fits=int(method != 'independent'), score_blocks=0)
    is_direct = method == 'direct' or (threshold == 0 and method in ('screen', 'loose', 'boundary', 'no_shortcut'))
    threshold = 0.0 if method == 'retain_all' else threshold
    legacy, boundary, loose = (method == 'legacy', method == 'boundary', method == 'loose')

    def exclude(record):
        if is_direct:
            return False
        if legacy:
            return gate(record['legacy_scores'], 0.01, 64.0) == 'exclude'
        return exclusion(record['scores'], threshold, boundary, loose)
    score_name = 'legacy_scores' if legacy else 'scores'
    full_excluded = exclude(full)
    shortcut = full_excluded and method != 'no_shortcut'
    baseline = independent_choice(row, fraction)
    own_sse = sum((f[f'{baseline}_sse'] for f in folds))
    cost = base_cost + (0 if is_direct else full[score_name]['runtime_seconds'])
    coupled_sse, joint_count, fold_excluded = (0.0, 0, 0)
    for fold in folds:
        skip = shortcut or exclude(fold)
        cost += fold['target_marginal']['runtime_seconds'] + fold['diagonal']['runtime_seconds'] + fold['independent_prediction_seconds'] + fold['diagonal_prediction_seconds']
        if not is_direct and (not shortcut):
            cost += fold[score_name]['runtime_seconds']
        if skip:
            fold_excluded += 1
            coupled_sse += fold['diagonal_sse']
        else:
            cost += fold['joint']['runtime_seconds'] + fold['joint_prediction_seconds']
            coupled_sse += fold['joint_sse']
            joint_count += 1
    wanted = not full_excluded and coupled_sse < (1 - fraction) * own_sse
    name = 'joint' if wanted and full['joint']['success'] else baseline
    if name == 'diagonal' and (not full['diagonal']['success']):
        name = 'independent'
    if wanted:
        cost += full['joint']['runtime_seconds']
        joint_count += 1
    cost += full[f'{name}_prediction_seconds']
    unresolved = False
    if not is_direct:
        score = full[score_name]
        unresolved = bool(score['flags']) or score['opportunity'] is None or (not np.isfinite(score['opportunity']))
        if boundary:
            unresolved |= any(('active_bound' in w for w in score['warnings']))
    return dict(model=name, full_excluded=full_excluded, fold_excluded=fold_excluded, unresolved=bool(unresolved), component_seconds=cost, joint_fits=joint_count, marginal_fits=2 + len(folds), diagonal_fits=1 + len(folds), score_blocks=0 if is_direct else 1 + (0 if shortcut else len(folds)))

def bootstrap_means(values, rows, namespace, replicates=BOOTSTRAPS):
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    rng = np.random.default_rng(seed_for('E6_task_v2', namespace, 'all', 0, 'stratified_bootstrap'))
    result = np.zeros((replicates, values.shape[1]))
    ids = np.array([r['scenario']['id'] for r in rows])
    for sid in sorted(set(ids)):
        indices = np.flatnonzero(ids == sid)
        draw = rng.choice(indices, size=(replicates, len(indices)), replace=True)
        result += values[draw].sum(axis=1) / len(rows)
    return result

def calibrate(rows):
    if any((r['phase'] != 'calibration' for r in rows)):
        raise ValueError('Only independent calibration datasets may set thresholds.')
    ri = np.array([r['assessment']['independent']['conditional_risk'] for r in rows])
    rd = np.array([r['assessment']['diagonal']['conditional_risk'] for r in rows])
    rj = np.array([r['assessment']['joint']['conditional_risk'] for r in rows])
    direct = [replay(r, 'direct') for r in rows]
    rv = np.array([r['assessment'][d['model']]['conditional_risk'] for r, d in zip(rows, direct)])
    candidates, selections = ([], {})
    for method in ('screen', 'loose'):
        for threshold in THRESHOLDS:
            decisions = [replay(r, method, threshold) for r in rows]
            risk = np.array([r['assessment'][d['model']]['conditional_risk'] for r, d in zip(rows, decisions)])
            excluded = np.array([d['full_excluded'] for d in decisions])
            positive_regret = excluded * np.maximum(np.minimum(ri, rd) - rj, 0)
            values = np.c_[ri, risk - rv, positive_regret]
            boot = bootstrap_means(values, rows, 'calibration')
            upper_excess = float(np.quantile(boot[:, 1] / boot[:, 0], 0.95))
            upper_regret = float(np.quantile(boot[:, 2] / boot[:, 0], 0.95))
            geometry_upper = {}
            for geometry in ('colocated', 'infill', 'partial', 'remote'):
                mask = np.array([r['scenario']['geometry'] == geometry for r in rows])
                group = [r for r, keep in zip(rows, mask) if keep]
                b = bootstrap_means(values[mask], group, 'calibration')
                geometry_upper[geometry] = float(np.quantile(b[:, 1] / b[:, 0], 0.95))
            admissible = upper_excess <= 0.01 and upper_regret <= 0.01 and (max(geometry_upper.values()) <= 0.02)
            candidate = dict(method=method, threshold=threshold, admissible=bool(admissible), mean_component_seconds=float(np.mean([d['component_seconds'] for d in decisions])), risk_excess_fraction=float((risk - rv).mean() / ri.mean()), upper_risk_excess_fraction=upper_excess, upper_positive_regret_fraction=upper_regret, geometry_upper_risk_excess_fraction=geometry_upper, excluded=int(excluded.sum()), false_exclusions=int(np.sum(excluded & (rj < 0.98 * np.minimum(ri, rd)))))
            candidates.append(candidate)
        eligible = [c for c in candidates if c['method'] == method and c['admissible']]
        if not eligible:
            raise RuntimeError('The retain-all rule must remain admissible.')
        selections[method] = min(eligible, key=lambda c: (c['mean_component_seconds'], c['threshold']))
    return {'chosen': selections, 'all_candidates': candidates, 'calibration_datasets': len(rows), 'direct_mean_component_seconds': float(np.mean([d['component_seconds'] for d in direct])), 'bootstrap_replicates': BOOTSTRAPS, 'constraints': 'Upper one-sided pointwise bootstrap percentiles: 1% pooled excess risk versus direct, 1% positive exclusion regret, 2% excess risk within each geometry; fractions of separate marginal risk.', 'scope': 'Empirical calibration constraints; independent evaluation assesses the selected thresholds.'}

def corrected_calibration_view(frozen_summary):
    """Return score-free zero-threshold costs without changing frozen records.

    Historical zero-threshold rows charged the retain-all ablation's scores.
    Their decisions and risks equal direct validation, whose saved component
    cost supplies the correction. Positive-threshold rows remain unchanged.
    """
    direct_cost = frozen_summary['direct_mean_component_seconds']
    if not np.isfinite(direct_cost) or direct_cost < 0:
        raise ValueError('A finite nonnegative direct-validation cost is required.')
    result = deepcopy(frozen_summary)
    for method in ('screen', 'loose'):
        candidates = [c for c in result['all_candidates'] if c['method'] == method]
        zero = [c for c in candidates if c['threshold'] == 0]
        if len(zero) != 1:
            raise ValueError(f'Expected exactly one zero-threshold row for {method}.')
        zero = zero[0]
        if zero['excluded'] != 0 or zero['false_exclusions'] != 0 or zero['risk_excess_fraction'] != 0 or (zero['upper_risk_excess_fraction'] != 0) or (zero['upper_positive_regret_fraction'] != 0) or any(zero['geometry_upper_risk_excess_fraction'].values()):
            raise ValueError('Zero-threshold risk and exclusion records must match direct validation.')
        zero['mean_component_seconds'] = direct_cost
        eligible = [c for c in candidates if c['admissible']]
        if not eligible:
            raise ValueError(f'No admissible calibration candidate for {method}.')
        result['chosen'][method] = min(eligible, key=lambda c: (c['mean_component_seconds'], c['threshold']))
    return result

def interval(estimate, samples):
    return {'estimate': float(estimate), 'ci95': np.quantile(samples, [0.025, 0.975]).tolist()}

def summarise(rows, threshold, loose_threshold):
    ri = np.array([r['assessment']['independent']['conditional_risk'] for r in rows])
    rd = np.array([r['assessment']['diagonal']['conditional_risk'] for r in rows])
    rj = np.array([r['assessment']['joint']['conditional_risk'] for r in rows])
    beneficial = rj < 0.98 * np.minimum(ri, rd)
    best = np.minimum.reduce([ri, rd, rj])
    results, records = ({}, {})
    for method in METHODS:
        decisions = [replay(r, method, loose_threshold if method == 'loose' else threshold) for r in rows]
        metrics = [r['assessment'][d['model']] for r, d in zip(rows, decisions)]
        selected = np.array([d['model'] == 'joint' for d in decisions])
        excluded = np.array([d['full_excluded'] for d in decisions])
        values = np.column_stack([ri, [m['conditional_risk'] for m in metrics], [m['latent_mse'] for m in metrics], [m['coverage95'] for m in metrics], [m['width95'] for m in metrics], [m['mlpd'] for m in metrics], beneficial, ~beneficial, selected & ~beneficial, ~selected & beneficial, excluded & beneficial, excluded, [d['component_seconds'] for d in decisions], [m['conditional_risk'] - b for m, b in zip(metrics, best)], [d['joint_fits'] for d in decisions]])
        b = bootstrap_means(values, rows, 'evaluation')
        mu = values.mean(axis=0)

        def ratio(numerator, denominator, scale=100):
            if mu[denominator] == 0:
                return None
            samples = np.divide(b[:, numerator], b[:, denominator], out=np.full(len(b), np.nan), where=b[:, denominator] > 0) * scale
            return {'estimate': float(mu[numerator] / mu[denominator] * scale), 'ci95': np.nanquantile(samples, [0.025, 0.975]).tolist()}
        results[method] = {'risk': interval(mu[1], b[:, 1]), 'risk_reduction_percent': interval(100 * (1 - mu[1] / mu[0]), 100 * (1 - b[:, 1] / b[:, 0])), 'latent_mse': interval(mu[2], b[:, 2]), 'coverage95': interval(100 * mu[3], 100 * b[:, 3]), 'width95': interval(mu[4], b[:, 4]), 'mlpd': interval(mu[5], b[:, 5]), 'final_fpr_percent': ratio(8, 7), 'final_fnr_percent': ratio(9, 6), 'screen_false_exclusion_percent': ratio(10, 6), 'exclusion_percent': interval(100 * mu[11], 100 * b[:, 11]), 'component_seconds': interval(mu[12], b[:, 12]), 'excess_risk_over_best_percent': ratio(13, 0), 'joint_fits_mean': interval(mu[14], b[:, 14]), 'counts': {'datasets': len(rows), 'beneficial_coupled': int(beneficial.sum()), 'nonbeneficial_coupled': int((~beneficial).sum()), 'excluded': int(excluded.sum()), 'false_exclusions': int(np.sum(excluded & beneficial)), 'final_false_positive': int(np.sum(selected & ~beneficial)), 'final_false_negative': int(np.sum(~selected & beneficial)), 'unresolved': sum((d['unresolved'] for d in decisions)), 'selections': {name: sum((d['model'] == name for d in decisions)) for name in ('independent', 'diagonal', 'joint')}}}
        records[method] = (values, decisions)
    direct = records['direct'][0]
    for method, (value, _) in records.items():
        diffs = np.c_[ri, value[:, 1] - direct[:, 1], value[:, 12], direct[:, 12]]
        boot = bootstrap_means(diffs, rows, 'evaluation')
        means = diffs.mean(axis=0)
        results[method]['risk_excess_vs_direct_percent'] = interval(100 * means[1] / means[0], 100 * boot[:, 1] / boot[:, 0])
        results[method]['component_saving_percent'] = interval(100 * (1 - means[2] / means[3]), 100 * (1 - boot[:, 2] / boot[:, 3]))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('calibrate', 'evaluate'))
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/E6_task_v2')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/E6_task_v2')
    args = parser.parse_args()
    phase = {'calibrate': 'calibration', 'evaluate': 'evaluation', 'pilot': 'pilot'}[args.action]
    manifest, rows, digest = load_records(args.run_dir, phase)
    if args.action == 'calibrate':
        output = METHOD / 'thresholds.json'
        result = calibrate(rows)
        result.update()
        output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result['chosen'], indent=2))
    else:
        thresholds = json.loads((METHOD / 'thresholds.json').read_text()) if args.action == 'evaluate' else None
        threshold = thresholds['chosen']['screen']['threshold'] if thresholds else 0.02
        loose_threshold = thresholds['chosen']['loose']['threshold'] if thresholds else 0.02
        results = {'phase': phase, 'threshold': threshold, 'loose_threshold': loose_threshold, 'pooled': summarise(rows, threshold, loose_threshold), 'by_geometry': {}, 'by_correlation': {}, 'scenarios': {}}
        for field, name in [('geometry', 'by_geometry'), ('rho', 'by_correlation'), ('id', 'scenarios')]:
            for key in sorted(set((r['scenario'][field] for r in rows))):
                subset = [r for r in rows if r['scenario'][field] == key]
                results[name][str(key)] = summarise(subset, threshold, loose_threshold)
        fits = [fit for r in rows for fit in [*r['marginals'], r['diagonal'], r['joint'], *(f[name] for f in r['folds'] for name in ('target_marginal', 'diagonal', 'joint'))]]
        results['fit_diagnostics'] = {'fits': len(fits), 'failed': sum((not f['success'] for f in fits)), 'active_bounds': sum((bool(f['active_bounds']) for f in fits)), 'failed_starts': sum((not start['success'] for f in fits for start in f['starts'])), 'gradient_above_tolerance': sum((f['projected_gradient_norm'] > 1e-05 for f in fits)), 'jitter_escalations': sum((f['jitter'] > 1e-10 for f in fits))}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({k: {name: v[name] for name in ['risk_reduction_percent', 'component_saving_percent', 'counts']} for k, v in results['pooled'].items()}, indent=2))
if __name__ == '__main__':
    main()
