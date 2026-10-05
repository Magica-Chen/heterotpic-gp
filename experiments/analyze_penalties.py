"""Analyze penalties for the multivariate kriging experiments."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from experiments.synthetic import seed_for

def choose(record, method, epsilon):
    ri = record['assessment']['risk_independent']
    rj = record['assessment']['risk_joint']
    positive = ri - rj > epsilon * ri
    success = record['joint']['success']
    cv = sum((f['joint_sse'] for f in record['folds'])) < (1 - epsilon) * sum((f['independent_sse'] for f in record['folds']))
    fallback = False
    if method == 'independent':
        chosen = False
    elif method == 'joint':
        chosen = success
    elif method == 'validate':
        chosen = success and cv
    else:
        gain = record['fitted_oracle_gain']
        if method == 'bootstrap':
            boot = record['bootstrap']
            if boot is None:
                return None
            available = boot['available']
            summary = boot.get('successful_draw_summary', {})
            hi, hj = (summary.get('independent_penalty'), summary.get('joint_penalty'))
        else:
            values = [record['penalties'][m].get(method, {}) for m in ('independent', 'joint')]
            available = all((v.get('available', False) for v in values))
            hi, hj = [v.get('penalty') for v in values]
        fallback = not (available and gain is not None and success)
        chosen = cv and success if fallback else gain['gain'] - (hj - hi) > epsilon * (gain['marginal_variance'] + hi)
    risk = rj if chosen else ri
    return {'selected_joint': int(chosen), 'beneficial': int(positive), 'false_positive': int(chosen and (not positive)), 'false_negative': int(not chosen and positive), 'fallback': int(fallback), 'risk': risk, 'baseline_risk': ri, 'risk_reduction': ri - risk, 'relative_regret': (risk - min(ri, rj)) / max(ri, 1e-10)}

def interval(values, weights):
    values = np.asarray(values, float)
    return np.quantile(weights @ values, [0.025, 0.975]).tolist()

def mean_summary(values, weights):
    a = np.asarray(values, float)
    return {'mean': float(a.mean()), 'ci95': interval(a, weights), 'n': len(a)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / 'evaluation_manifest.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    decisions = []
    calibration = []
    numerical = []
    pooled_parts = {}
    fit_diagnostics = {'outer_datasets': 0, 'outer_fit_failures': 0, 'outer_active_bound_fits': 0, 'outer_fits': 0, 'bootstrap_outer_datasets': 0, 'bootstrap_draws': 0, 'bootstrap_refit_failures': 0, 'bootstrap_active_bound_draws': 0}
    with gzip.open(args.output_dir / 'penalty_draws.csv.gz', 'wt', newline='') as stream:
        writer = None
        for scenario in manifest['scenarios']:
            records = []
            for rep in range(scenario['evaluation_replicates']):
                path = args.run_dir / 'evaluation' / scenario['id'] / f'dataset_{rep:04d}.json.gz'
                with gzip.open(path, 'rt') as f:
                    r = json.load(f)
                records.append(r)
                fit_diagnostics['outer_datasets'] += 1
                fits = r['marginals'] + [r['joint']] + [f for k in r['folds'] for f in k['marginals'] + [k['joint']]]
                fit_diagnostics['outer_fits'] += len(fits)
                fit_diagnostics['outer_fit_failures'] += sum((not f['success'] for f in fits))
                fit_diagnostics['outer_active_bound_fits'] += sum((bool(f['active_bounds']) for f in fits))
                boot = r['bootstrap']
                if boot is not None:
                    fit_diagnostics['bootstrap_outer_datasets'] += 1
                    fit_diagnostics['bootstrap_draws'] += boot.get('attempted_draws', 0)
                    fit_diagnostics['bootstrap_refit_failures'] += boot.get('attempted_draws', 0) - boot.get('successful_draws', 0)
                    fit_diagnostics['bootstrap_active_bound_draws'] += boot.get('active_bound_draws', 0)
                row = {'scenario_id': scenario['id'], 'replication': rep, 'fit_mode': scenario['fit_mode'], **{k: r['assessment'][k] for k in ('oracle_gain', 'independent_penalty_draw', 'joint_penalty_draw', 'risk_independent', 'risk_joint', 'risk_difference_draw', 'accounting_rhs_draw', 'accounting_cross_term_draw')}}
                for method in ('expected', 'observed', 'bootstrap'):
                    for model in ('independent', 'joint'):
                        row[f'{method}_{model}'] = (boot.get('successful_draw_summary', {}).get(model + '_penalty') if boot else None) if method == 'bootstrap' else r['penalties'][model].get(method, {}).get('penalty')
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
            n = len(records)
            rng = np.random.default_rng(seed_for(manifest['experiment_id'], 'analysis', scenario['id'], 0, 'bootstrap'))
            weights = rng.multinomial(n, np.full(n, 1 / n), size=args.bootstrap) / n
            scalar_names = ('oracle_gain', 'independent_penalty_draw', 'joint_penalty_draw', 'risk_independent', 'risk_joint', 'risk_difference_draw', 'accounting_rhs_draw', 'accounting_cross_term_draw')
            summary = {'scenario': scenario, 'n': n, **{k: mean_summary([r['assessment'][k] for r in records], weights) for k in scalar_names}}
            summary['relative_mean_risk_reduction_percent'] = 100 * summary['risk_difference_draw']['mean'] / summary['risk_independent']['mean']
            dr = weights @ np.array([r['assessment']['risk_difference_draw'] for r in records])
            ri = weights @ np.array([r['assessment']['risk_independent'] for r in records])
            summary['relative_mean_risk_reduction_percent_ci95'] = np.quantile(100 * dr / ri, [0.025, 0.975]).tolist()
            summary['mass'] = records[0]['assessment']['true_scale_mass']
            true_corr = np.array(scenario['correlation'])
            d = len(true_corr)
            for j in range(1, d):
                values = np.array([r['joint']['correlation'][0][j] for r in records])
                summary[f'rho1{j + 1}_estimate'] = mean_summary(values, weights)
                summary[f'rho1{j + 1}_squared_error'] = mean_summary((values - true_corr[0, j]) ** 2, weights)
            summaries.append(summary)
            for estimator in ('expected', 'observed', 'bootstrap'):
                for model in ('independent', 'joint'):
                    eligible = [r for r in records if estimator != 'bootstrap' or r['bootstrap'] is not None]
                    for stratum in ('all', 'interior_gradient_converged', 'boundary_or_gradient_flag'):
                        subset = []
                        for r in eligible:
                            interior = r['penalties'][model].get('regular_interior', False)
                            if stratum == 'interior_gradient_converged' and (not interior):
                                continue
                            if stratum == 'boundary_or_gradient_flag' and interior:
                                continue
                            subset.append(r)
                        pairs = []
                        for r in subset:
                            if estimator == 'bootstrap':
                                item = r['bootstrap']
                                available = item['available']
                                estimate = item.get('successful_draw_summary', {}).get(model + '_penalty')
                            else:
                                item = r['penalties'][model].get(estimator, {})
                                available = item.get('available', False)
                                estimate = item.get('penalty')
                            if available and estimate is not None:
                                pairs.append([estimate, r['assessment'][model + '_penalty_draw']])
                        row = {'scenario_id': scenario['id'], 'estimator': estimator, 'model': model, 'stratum': stratum, 'eligible': len(subset), 'available': len(pairs), 'unavailable': len(subset) - len(pairs)}
                        if pairs:
                            a = np.array(pairs)
                            local = len(a)
                            w = rng.multinomial(local, np.full(local, 1 / local), size=args.bootstrap) / local
                            error = a[:, 0] - a[:, 1]
                            row.update(estimated=mean_summary(a[:, 0], w), matched_actual=mean_summary(a[:, 1], w), signed_error=mean_summary(error, w), absolute_draw_error=mean_summary(abs(error), w), rmse_draw_error=float(np.sqrt(np.mean(error ** 2))), relative_error_of_means=float(error.mean() / a[:, 1].mean()) if a[:, 1].mean() > 1e-08 else None, caveat='Draw-wise error includes realised predictor-deviation variation; ensemble mean calibration is the primary target.')
                        calibration.append(row)
            for tier in ('all', 'bootstrap_outer'):
                subset = records if tier == 'all' else [r for r in records if r['bootstrap'] is not None]
                if not subset:
                    continue
                local = len(subset)
                w = weights if tier == 'all' else rng.multinomial(local, np.full(local, 1 / local), size=args.bootstrap) / local
                for epsilon in manifest['material_fractions']:
                    for method in ('independent', 'joint', 'validate', 'expected', 'observed', 'bootstrap'):
                        if method == 'bootstrap' and tier == 'all':
                            continue
                        values = [choose(r, method, epsilon) for r in subset]
                        if any((v is None for v in values)):
                            continue
                        a = {k: np.array([v[k] for v in values]) for k in values[0]}
                        pooled_parts.setdefault((scenario['fit_mode'], tier, epsilon, method), []).append(a)
                        row = {'scenario_id': scenario['id'], 'tier': tier, 'epsilon': epsilon, 'method': method, 'datasets': local, 'selected_joint': int(a['selected_joint'].sum()), 'fallback_count': int(a['fallback'].sum()), 'risk': mean_summary(a['risk'], w), 'relative_regret_percent': mean_summary(100 * a['relative_regret'], w), 'relative_mean_risk_reduction_percent': float(100 * a['risk_reduction'].mean() / a['baseline_risk'].mean()), 'relative_mean_risk_reduction_percent_ci95': np.quantile(100 * (w @ a['risk_reduction']) / (w @ a['baseline_risk']), [0.025, 0.975]).tolist()}
                        for error, den in (('false_positive', 1 - a['beneficial']), ('false_negative', a['beneficial'])):
                            num = int(a[error].sum())
                            denominator = int(den.sum())
                            bootden = w @ den
                            rates = np.divide(w @ a[error], bootden, out=np.full(args.bootstrap, np.nan), where=bootden > 0)
                            row[error] = {'numerator': num, 'denominator': denominator, 'rate': num / denominator if denominator else None, 'ci95': np.nanquantile(rates, [0.025, 0.975]).tolist() if np.any(bootden > 0) else None}
                        decisions.append(row)
            numerical.append({'scenario_id': scenario['id'], 'expected_unavailable_joint': sum((not r['penalties']['joint'].get('expected', {}).get('available', False) for r in records)), 'observed_unavailable_joint': sum((not r['penalties']['joint'].get('observed', {}).get('available', False) for r in records)), 'bootstrap_unavailable': sum((r['bootstrap'] is not None and (not r['bootstrap']['available']) for r in records)), 'mean_information_seconds': float(np.mean([sum((v['runtime_seconds'] for v in r['penalties'].values())) for r in records])), 'mean_bootstrap_seconds': float(np.mean([r['bootstrap']['runtime_seconds'] for r in records if r['bootstrap']])) if any((r['bootstrap'] for r in records)) else None, 'mean_full_fit_prediction_seconds': float(np.mean([r['runtime']['full_fit_and_prediction'] for r in records])), 'mean_validation_seconds': float(np.mean([r['runtime']['four_fold_validation'] for r in records]))})
            print(f"Analysed {scenario['id']}: {n} datasets", flush=True)
    pooled = []
    pooled_samples = {}
    for key, parts in pooled_parts.items():
        mode, tier, epsilon, method = key
        n = sum((len(p['risk']) for p in parts))
        names = list(parts[0])
        means = {k: sum((p[k].sum() for p in parts)) / n for k in names}
        resamples = {k: np.zeros(args.bootstrap) for k in names}
        rng = np.random.default_rng(seed_for(manifest['experiment_id'], 'pooled_analysis', str(key[:3]), 0, 'bootstrap'))
        for part in parts:
            count = len(part['risk'])
            w = rng.multinomial(count, np.full(count, 1 / count), size=args.bootstrap) / n
            for name in names:
                resamples[name] += w @ part[name]
        row = {'fit_mode': mode, 'tier': tier, 'epsilon': epsilon, 'method': method, 'datasets': n, 'selected_joint': int(round(means['selected_joint'] * n)), 'fallback_count': int(round(means['fallback'] * n)), 'relative_mean_risk_reduction_percent': 100 * means['risk_reduction'] / means['baseline_risk'], 'relative_mean_risk_reduction_percent_ci95': np.quantile(100 * resamples['risk_reduction'] / resamples['baseline_risk'], [0.025, 0.975]).tolist(), 'relative_regret_percent': 100 * means['relative_regret'], 'relative_regret_percent_ci95': np.quantile(100 * resamples['relative_regret'], [0.025, 0.975]).tolist()}
        for error, positive in (('false_positive', False), ('false_negative', True)):
            denominator = means['beneficial'] if positive else 1 - means['beneficial']
            bootden = resamples['beneficial'] if positive else 1 - resamples['beneficial']
            rates = np.divide(resamples[error], bootden, out=np.full(args.bootstrap, np.nan), where=bootden > 0)
            row[error] = {'numerator': int(round(means[error] * n)), 'denominator': int(round(denominator * n)), 'rate': means[error] / denominator if denominator else None, 'ci95': np.nanquantile(rates, [0.025, 0.975]).tolist() if np.any(bootden > 0) else None}
        pooled.append(row)
        pooled_samples[key] = (means, resamples)
    contrasts = []
    for key, (mean, samples) in pooled_samples.items():
        mode, tier, epsilon, method = key
        if method not in ('expected', 'observed', 'bootstrap'):
            continue
        reference = pooled_samples[mode, tier, epsilon, 'validate']
        point = 100 * (mean['risk_reduction'] - reference[0]['risk_reduction']) / mean['baseline_risk']
        paired = 100 * (samples['risk_reduction'] - reference[1]['risk_reduction']) / samples['baseline_risk']
        contrasts.append({'fit_mode': mode, 'tier': tier, 'epsilon': epsilon, 'method': method, 'reference': 'validate', 'risk_reduction_advantage_percentage_points': point, 'paired_ci95': np.quantile(paired, [0.025, 0.975]).tolist()})
    result = {'experiment_id': manifest['experiment_id'], 'bootstrap_replicates': args.bootstrap, 'uncertainty': 'independent datasets within each fixed scenario/mode; paired method comparisons on identical datasets', 'fit_diagnostics': fit_diagnostics, 'scenarios': summaries, 'penalty_calibration': calibration, 'decisions': decisions, 'pooled_decisions': pooled, 'paired_contrasts': contrasts, 'numerical': numerical}
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(fit_diagnostics), flush=True)
if __name__ == '__main__':
    main()
