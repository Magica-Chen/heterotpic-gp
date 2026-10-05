"""Analyze robustness for the multivariate kriging experiments."""
from dataclasses import replace
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from experiments.screening import ScreenConfig, gate, validation_selection
from experiments.synthetic import seed_for

def decision_rows(record, epsilon):
    working = record['working']
    rich = record['rich']
    assessment = record['assessment']
    screen = replace(ScreenConfig(**working['scores']['settings']), material_fraction=epsilon)
    status = gate(working['scores'], screen.opportunity_threshold, screen.information_threshold)
    cv = validation_selection(working['folds'], screen, unscreened=True)
    adaptive = validation_selection(working['folds'], screen)
    success = working['joint']['success']
    choices = {'always_independent': 'independent', 'always_joint': 'joint' if success else 'independent', 'validate': 'joint' if cv and success else 'independent', 'screen_validate': 'joint' if adaptive and status != 'exclude' and success else 'independent'}
    if rich is not None:
        selected = sum((f['joint_sse'] for f in rich['folds'])) < (1 - epsilon) * sum((f['independent_sse'] for f in rich['folds'])) and rich['joint']['success']
        choices.update(rich_independent='rich_independent', rich_joint='rich_joint' if rich['joint']['success'] else 'rich_independent', rich_validate='rich_joint' if selected else 'rich_independent')
    conditional = record['scenario']['conditional_reference']
    baseline = assessment['independent']['risk']
    combined = assessment['joint']['risk']
    best = min((v['risk'] for v in assessment.values()))
    beneficial = baseline - combined > epsilon * baseline if conditional else None
    rows = []
    for method, chosen in choices.items():
        metrics = assessment[chosen]
        risk = metrics['risk']
        rich_reference = assessment['rich_independent']['risk'] if rich is not None and method.startswith('rich') else None
        row = {'method': method, 'epsilon': epsilon, 'chosen': chosen, 'risk': risk, 'baseline_risk': baseline, 'risk_reduction': baseline - risk, 'borrowed': int(chosen in ('joint', 'rich_joint')), 'conditional_reference': int(conditional), 'catalogue_relative_regret': (risk - best) / max(baseline, 1e-10) if conditional else None, 'working_beneficial': int(beneficial) if conditional else None, 'gate_excluded': int(status == 'exclude'), 'gate_unresolved': int(status == 'unresolved'), 'false_exclusion': int(status == 'exclude' and beneficial) if conditional else None, 'working_false_positive': int(chosen == 'joint' and (not beneficial)) if conditional and (not method.startswith('rich')) else None, 'working_false_negative': int(chosen != 'joint' and beneficial) if conditional and (not method.startswith('rich')) else None, 'rich_baseline_risk': rich_reference, 'rich_risk_reduction': rich_reference - risk if rich_reference is not None else None, **{k: metrics[k] for k in ('latent_mse', 'observed_mse', 'mlpd', 'coverage95', 'width95')}}
        rows.append(row)
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args()
    manifest = json.loads((args.run_dir / 'evaluation_manifest.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    costs = {}
    fit_diagnostics = {'datasets': 0, 'fits': 0, 'fit_failures': 0, 'active_bound_fits': 0, 'above_gradient_tolerance': 0, 'failed_starts': 0, 'jitter_escalations': 0}
    with gzip.open(args.output_dir / 'decisions.csv.gz', 'wt', newline='') as stream:
        writer = None
        for scenario in manifest['scenarios']:
            for rep in range(manifest['replicates_per_scenario']):
                path = args.run_dir / 'evaluation' / scenario['id'] / f'dataset_{rep:04d}.json.gz'
                with gzip.open(path, 'rt') as f:
                    r = json.load(f)
                fit_diagnostics['datasets'] += 1
                for branch_name in ('working', 'rich'):
                    branch = r[branch_name]
                    if branch is None:
                        continue
                    fits = branch['marginals'] + [branch['joint']] + [f for k in branch['folds'] for f in k['marginals'] + [k['joint']]]
                    fit_diagnostics['fits'] += len(fits)
                    fit_diagnostics['fit_failures'] += sum((not f['success'] for f in fits))
                    fit_diagnostics['active_bound_fits'] += sum((bool(f['active_bounds']) for f in fits))
                    fit_diagnostics['above_gradient_tolerance'] += sum((f['projected_gradient_norm'] is not None and f['projected_gradient_norm'] > f['config']['gtol'] for f in fits))
                    fit_diagnostics['failed_starts'] += sum((not a['success'] for f in fits for a in f['starts']))
                    fit_diagnostics['jitter_escalations'] += sum((f['jitter'] > f['config']['jitter_ladder'][0] * f['config']['jitter_scale'] for f in fits))
                    costs.setdefault((scenario['family'], branch_name), []).append({'fit_seconds': sum((f['runtime_seconds'] for f in fits)), 'diagnostic_seconds': sum((f['scores']['runtime_seconds'] for f in [branch, *branch['folds']])) if branch_name == 'working' else 0.0, 'fits': len(fits), 'failed_fits': sum((not f['success'] for f in fits)), 'active_bound_fits': sum((bool(f['active_bounds']) for f in fits))})
                for epsilon in manifest['material_fractions']:
                    for row in decision_rows(r, epsilon):
                        output = {'scenario_id': scenario['id'], 'replication': rep, **row}
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(output))
                            writer.writeheader()
                        writer.writerow(output)
                        groups.setdefault((scenario['id'], row['method'], epsilon), []).append(row)
            print(f"Processed {scenario['id']}", flush=True)
    summaries = []
    family_parts = {}
    scenario_map = {s['id']: s for s in manifest['scenarios']}
    for key, values in groups.items():
        scenario, method, epsilon = key
        s = scenario_map[scenario]
        n = len(values)
        names = [k for k, v in values[0].items() if isinstance(v, (int, float))]
        a = {k: np.array([v[k] for v in values]) for k in names}
        rng = np.random.default_rng(seed_for('E8', 'analysis', scenario, 0, str(epsilon)))
        w = rng.multinomial(n, np.full(n, 1 / n), size=args.bootstrap) / n
        mean = {k: float(v.mean()) for k, v in a.items()}
        boot = {k: w @ v for k, v in a.items()}
        row = {'scenario': s, 'method': method, 'epsilon': epsilon, 'datasets': n, 'relative_mean_risk_reduction_percent': 100 * mean['risk_reduction'] / mean['baseline_risk'], 'relative_mean_risk_reduction_percent_ci95': np.quantile(100 * boot['risk_reduction'] / boot['baseline_risk'], [0.025, 0.975]).tolist(), 'mean_risk': mean['risk'], 'mean_baseline_risk': mean['baseline_risk'], 'mean_latent_mse': mean['latent_mse'], 'mean_mlpd': mean['mlpd'], 'mean_coverage95': mean['coverage95'], 'gate_exclusions': int(round(n * mean['gate_excluded'])), 'gate_unresolved': int(round(n * mean['gate_unresolved'])), 'borrowed_count': int(round(n * mean['borrowed']))}
        if s['conditional_reference']:
            row['catalogue_relative_regret_percent'] = 100 * mean['catalogue_relative_regret']
            row['catalogue_relative_regret_percent_ci95'] = np.quantile(100 * boot['catalogue_relative_regret'], [0.025, 0.975]).tolist()
            numerator = int(round(n * mean['false_exclusion']))
            denominator = int(round(n * mean['working_beneficial']))
            row['false_exclusion'] = {'numerator': numerator, 'denominator': denominator, 'rate': numerator / denominator if denominator else None}
            if not method.startswith('rich'):
                row['working_false_positive'] = {'numerator': int(round(n * mean['working_false_positive'])), 'denominator': n - denominator}
                row['working_false_negative'] = {'numerator': int(round(n * mean['working_false_negative'])), 'denominator': denominator}
        else:
            row.update(catalogue_relative_regret_percent=None, false_exclusion=None, reference_note='Repeated-sampling latent loss; exact conditional labels and regret are not reported.')
        if method.startswith('rich'):
            row['relative_mean_risk_reduction_vs_rich_independent_percent'] = 100 * mean['rich_risk_reduction'] / mean['rich_baseline_risk']
            row['relative_mean_risk_reduction_vs_rich_independent_percent_ci95'] = np.quantile(100 * boot['rich_risk_reduction'] / boot['rich_baseline_risk'], [0.025, 0.975]).tolist()
        summaries.append(row)
        family_parts.setdefault((s['family'], method, epsilon), []).append(a)
    pooled = []
    samples = {}
    for key, parts in family_parts.items():
        family, method, epsilon = key
        n = sum((len(p['risk']) for p in parts))
        names = list(parts[0])
        means = {k: sum((p[k].sum() for p in parts)) / n for k in names}
        boot = {k: np.zeros(args.bootstrap) for k in names}
        rng = np.random.default_rng(seed_for('E8', 'family_analysis', family, 0, str(epsilon)))
        for a in parts:
            count = len(a['risk'])
            w = rng.multinomial(count, np.full(count, 1 / count), size=args.bootstrap) / n
            for k in names:
                boot[k] += w @ a[k]
        row = {'family': family, 'method': method, 'epsilon': epsilon, 'datasets': n, 'relative_mean_risk_reduction_percent': 100 * means['risk_reduction'] / means['baseline_risk'], 'relative_mean_risk_reduction_percent_ci95': np.quantile(100 * boot['risk_reduction'] / boot['baseline_risk'], [0.025, 0.975]).tolist(), 'mean_risk': means['risk'], 'mean_mlpd': means['mlpd'], 'mean_coverage95': means['coverage95'], 'gate_exclusions': int(round(n * means['gate_excluded'])), 'gate_unresolved': int(round(n * means['gate_unresolved'])), 'borrowed_count': int(round(n * means['borrowed']))}
        if 'catalogue_relative_regret' in means:
            row['catalogue_relative_regret_percent'] = 100 * means['catalogue_relative_regret']
            row['catalogue_relative_regret_percent_ci95'] = np.quantile(100 * boot['catalogue_relative_regret'], [0.025, 0.975]).tolist()
            row['false_exclusion'] = {'numerator': int(round(n * means['false_exclusion'])), 'denominator': int(round(n * means['working_beneficial']))}
        else:
            row.update(catalogue_relative_regret_percent=None, false_exclusion=None)
        if method.startswith('rich'):
            row['relative_mean_risk_reduction_vs_rich_independent_percent'] = 100 * means['rich_risk_reduction'] / means['rich_baseline_risk']
            row['relative_mean_risk_reduction_vs_rich_independent_percent_ci95'] = np.quantile(100 * boot['rich_risk_reduction'] / boot['rich_baseline_risk'], [0.025, 0.975]).tolist()
        pooled.append(row)
        samples[key] = (means, boot)
    contrasts = []
    for key, (means, boot) in samples.items():
        family, method, epsilon = key
        if method not in ('screen_validate', 'rich_validate', 'rich_joint', 'rich_independent'):
            continue
        for comparison in ('validate',) if method == 'screen_validate' else ('always_joint', 'validate'):
            ref_mean, ref_boot = samples[family, comparison, epsilon]
            contrasts.append({'family': family, 'method': method, 'reference': comparison, 'epsilon': epsilon, 'risk_reduction_advantage_percentage_points': 100 * (means['risk_reduction'] - ref_mean['risk_reduction']) / means['baseline_risk'], 'paired_ci95': np.quantile(100 * (boot['risk_reduction'] - ref_boot['risk_reduction']) / boot['baseline_risk'], [0.025, 0.975]).tolist()})
    result = {'bootstrap_replicates': args.bootstrap, 'fit_diagnostics': fit_diagnostics, 'uncertainty': 'datasets within scenario; family summaries stratify over six design/sample-size configurations', 'heavy_tail_scope': 'Unconditional repeated-sampling latent risk; no exact conditional error labels or regret.', 'cost_scope': 'Sum of recorded full/fold fitting and working diagnostic times, including research-only shadow joint fits; not measured deployed pipeline speedup.', 'costs': [{'family': key[0], 'branch': key[1], 'datasets': len(values), **{'mean_' + k: float(np.mean([v[k] for v in values])) for k in values[0]}} for key, values in costs.items()], 'scenarios': summaries, 'families': pooled, 'paired_contrasts': contrasts}
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(fit_diagnostics), flush=True)
if __name__ == '__main__':
    main()
