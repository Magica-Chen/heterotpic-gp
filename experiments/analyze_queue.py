"""Analyze queue for the multivariate kriging experiments."""
from __future__ import annotations
import argparse
from collections import defaultdict
from functools import lru_cache
import gzip
import json
import numpy as np
from experiments.queue_study import ID, METHOD, RUN, OUT, BRANCHES, REPS, THRESHOLDS, cases, load_reference, primary_config, execute_application
from experiments.analyze_task_screen import replay
from experiments.common import clean_json
from experiments.synthetic import seed_for
METHODS = ('independent', 'diagonal', 'joint', 'direct', 'transfer', 'calibrated')
BOOTSTRAPS = 2000

def decision(row, method, threshold):
    if method == 'transfer':
        return replay(row, 'screen', 0.2)
    if method == 'calibrated':
        return replay(row, 'direct' if threshold == 0.0 else 'screen', threshold)
    return replay(row, method)

def load_rows(phase):
    result, _ = ([], None)
    for case in cases():
        for rep in range(REPS[phase]):
            path = RUN / phase / case['id'] / f'dataset_{rep:04d}.json.gz'
            with gzip.open(path, 'rt') as f:
                raw = json.load(f)
            result.append(raw)
    return (result, None)

def interval(value, samples):
    return dict(estimate=float(value), ci95=np.quantile(samples, [0.025, 0.975]).tolist())

def bootstrap(values, rows, namespace):
    rng = np.random.default_rng(seed_for(ID, 'analysis', namespace, 0, 'bootstrap'))
    strata = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        strata[row['scenario']['n1']][row['replication']].append(i)
    result = np.zeros((BOOTSTRAPS, values.shape[1]))
    for n in sorted(strata):
        units = np.array([values[indices].sum(axis=0) for _, indices in sorted(strata[n].items())])
        weights = rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=BOOTSTRAPS)
        result += weights @ units / len(rows)
    return result

@lru_cache(maxsize=128)
def reference_delta(phase, namespace):
    reference = load_reference(phase)
    rng = np.random.default_rng(seed_for(ID, 'analysis', namespace, 0, 'reference'))
    result = []
    for point in reference['points']:
        values = np.array(point['run_values'])
        weights = rng.multinomial(len(values), np.full(len(values), 1 / len(values)), size=BOOTSTRAPS)
        result.append(weights @ values / len(values) - values.mean())
    return np.array(result).T

def summarise(rows, branch, threshold, phase, namespace):
    risk, errors, excluded, selected, costs, counts, useful, metrics, fits = ([], [], [], [], [], [], [], [], [])
    for raw in rows:
        row = raw['branches'][branch]
        ds = [decision(row, method, threshold) for method in METHODS]
        loss = [row['assessment'][d['model']]['conditional_risk'] for d in ds]
        risk.append(loss)
        errors.append([np.array(raw['predictions'][branch][d['model']][0]) - np.array(raw['data']['oracle']) for d in ds])
        excluded.append([d['full_excluded'] for d in ds])
        selected.append([d['model'] for d in ds])
        costs.append([d['component_seconds'] for d in ds])
        counts.append([d['joint_fits'] for d in ds])
        useful.append(loss[2] < 0.98 * min(loss[:2]))
        metrics.append([row['assessment'][d['model']]['latent_coverage95'] for d in ds])
        fits.extend(row['marginals'] + [row['diagonal'], row['joint']] + [f[k] for f in row['folds'] for k in ('target_marginal', 'diagonal', 'joint')])
    risk, errors, excluded = (np.array(risk), np.array(errors), np.array(excluded))
    useful = np.array(useful)
    chosen = np.array(selected)
    fields = dict(risk=risk, errors=errors.reshape(len(rows), -1), excluded=excluded, false_excluded=excluded * useful[:, None], useful=useful[:, None], seconds=np.array(costs), joint_fits=np.array(counts), latent_coverage95=np.array(metrics), positive_regret=excluded * np.maximum(np.minimum(risk[:, 0], risk[:, 1]) - risk[:, 2], 0)[:, None])
    parts, offsets, start = ([], {}, 0)
    for key, value in fields.items():
        offsets[key] = slice(start, start + value.shape[1])
        start += value.shape[1]
        parts.append(value)
    boot = bootstrap(np.column_stack(parts), rows, namespace)
    b = {k: boot[:, s] for k, s in offsets.items()}
    delta = reference_delta(phase, namespace)
    err = b['errors'].reshape(BOOTSTRAPS, len(METHODS), -1)
    b['risk'] = b['risk'] - 2 * np.mean(err * delta[:, None, :], axis=2) + np.mean(delta ** 2, axis=1)[:, None]
    rng = np.random.default_rng(seed_for(ID, 'analysis', namespace, 0, 'regret_bootstrap'))
    strata = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        strata[row['scenario']['n1']][row['replication']].append(i)
    reg_boot = np.zeros((BOOTSTRAPS, len(METHODS)))
    for first in range(0, BOOTSTRAPS, 100):
        d = delta[first:first + 100]
        adjusted = risk[None, :, :3] - 2 * np.einsum('ngp,bp->bng', errors[:, :3], d) / d.shape[1] + np.mean(d * d, axis=1)[:, None, None]
        positive = np.maximum(np.minimum(adjusted[:, :, 0], adjusted[:, :, 1]) - adjusted[:, :, 2], 0)
        for n in sorted(strata):
            units = [indices for _, indices in sorted(strata[n].items())]
            weights = rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=len(d))
            for u, indices in enumerate(units):
                reg_boot[first:first + len(d)] += weights[:, u, None] * np.einsum('bn,nm->bm', positive[:, indices], excluded[indices]) / len(rows)
    b['positive_regret'] = reg_boot
    means = {k: v.mean(axis=0) for k, v in fields.items()}
    result = {}
    for j, name in enumerate(METHODS):
        entry = dict(risk=interval(means['risk'][j], b['risk'][:, j]), risk_reduction_percent=interval(100 * (1 - means['risk'][j] / means['risk'][0]), 100 * (1 - b['risk'][:, j] / b['risk'][:, 0])), excess_over_direct_percent=interval(100 * (means['risk'][j] - means['risk'][3]) / means['risk'][0], 100 * (b['risk'][:, j] - b['risk'][:, 3]) / b['risk'][:, 0]), exclusion_percent=interval(100 * means['excluded'][j], 100 * b['excluded'][:, j]), positive_exclusion_regret_percent=interval(100 * means['positive_regret'][j] / means['risk'][0], 100 * b['positive_regret'][:, j] / b['risk'][:, 0]), component_seconds=interval(means['seconds'][j], b['seconds'][:, j]), joint_fits=interval(means['joint_fits'][j], b['joint_fits'][:, j]), latent_coverage95=interval(100 * means['latent_coverage95'][j], 100 * b['latent_coverage95'][:, j]), selected_counts={k: int(np.sum(chosen[:, j] == k)) for k in METHODS[:3]})
        valid = b['useful'][:, 0] > 0
        entry['false_exclusion_percent'] = None if not useful.any() else interval(100 * means['false_excluded'][j] / means['useful'][0], 100 * b['false_excluded'][valid, j] / b['useful'][valid, 0])
        result[name] = entry
    return dict(datasets=len(rows), beneficial_joint=int(useful.sum()), methods=result, threshold=threshold, fit_diagnostics=dict(fits=len(fits), failed=sum((not f['success'] for f in fits)), boundary=sum((bool(f['active_bounds']) for f in fits)), above_gtol=sum((f['projected_gradient_norm'] is None or f['projected_gradient_norm'] > 1e-05 for f in fits))), calibration_limits=dict(upper_excess=float(np.quantile((b['risk'][:, 5] - b['risk'][:, 3]) / b['risk'][:, 0], 0.95)), upper_regret=float(np.quantile(b['positive_regret'][:, 5] / b['risk'][:, 0], 0.95))))

def calibrate(rows, digest):
    if any((r['phase'] != 'calibration' for r in rows)):
        raise ValueError('Only independent calibration data may choose the application settings.')
    candidates, chosen, mean_risks = ({}, {}, {})
    for branch in BRANCHES:
        candidates[branch] = []
        for threshold in THRESHOLDS:
            overall = summarise(rows, branch, threshold, 'calibration', f'calibration_{branch}_overall')
            limits = overall['calibration_limits']
            cell_limits = {}
            for case in cases():
                group = [r for r in rows if r['scenario']['id'] == case['id']]
                report = summarise(group, branch, threshold, 'calibration', f"calibration_{branch}_{case['id']}")
                cell_limits[case['id']] = report['calibration_limits']['upper_excess']
            admissible = limits['upper_excess'] <= 0.01 and limits['upper_regret'] <= 0.01 and (max(cell_limits.values()) <= 0.02)
            candidate = dict(threshold=threshold, admissible=bool(admissible), pooled_limits=limits, cell_upper_excess=cell_limits, mean_component_seconds=overall['methods']['calibrated']['component_seconds']['estimate'], summary=overall)
            candidates[branch].append(candidate)
            mean_risks[branch] = overall['methods']['direct']['risk']['estimate']
            print('calibration', branch, threshold, 'eligible', admissible, flush=True)
        eligible = [r for r in candidates[branch] if r['admissible']]
        chosen[branch] = min(eligible, key=lambda r: (r['mean_component_seconds'], r['threshold']))
    primary = min(BRANCHES, key=lambda b: (mean_risks[b], BRANCHES.index(b)))
    return dict(experiment_id=ID, primary_mean=primary, chosen=chosen, all_candidates=candidates, mean_choice_direct_physical_mse=mean_risks, calibration_datasets=len(rows), bootstrap_replicates=BOOTSTRAPS)

def timing(rows, selection):
    branch = selection['primary_mean']
    threshold = selection['chosen'][branch]['threshold']
    result = []
    for i, raw in enumerate((r for r in rows if r['replication'] < 5)):
        data = {k: [np.array(v) for v in val] if k in ('xs', 'ys', 'noise') else np.array(val) if k in ('evaluation', 'membership') else val for k, val in raw['data'].items()}
        methods = ('direct', 'transfer', 'calibrated')
        order = methods[i % 3:] + methods[:i % 3]
        record = dict(scenario=raw['scenario'], replication=raw['replication'], order=order, methods={})
        for name in order:
            t = 0.2 if name == 'transfer' else threshold
            mode = 'direct' if name == 'direct' else 'screen'
            row, pred = execute_application(data, branch, t, mode)
            record['methods'][name] = dict(seconds=row['runtime_seconds'], counts=row['counts'], selected_model=row['selected_model'])
        result.append(record)
        print('timing', raw['scenario']['id'], raw['replication'], 'complete', flush=True)
    values = np.array([[r['methods'][m]['seconds'] for m in ('direct', 'transfer', 'calibrated')] for r in result])
    b = bootstrap(values, result, 'paired_timing')
    summaries = {m: dict(seconds=interval(values[:, j].mean(), b[:, j]), saving_vs_direct_percent=interval(100 * (1 - values[:, j].mean() / values[:, 0].mean()), 100 * (1 - b[:, j] / b[:, 0]))) for j, m in enumerate(('direct', 'transfer', 'calibrated'))}
    return dict(primary_mean=branch, threshold=threshold, datasets=len(result), records=result, methods=summaries)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('calibration', 'evaluation', 'timing'), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    phase = 'calibration' if args.phase == 'calibration' else 'evaluation'
    rows, digest = load_rows(phase)
    if args.phase == 'calibration':
        path = METHOD / 'selection.json'
        result = calibrate(rows, digest)
        path.write_text(json.dumps(clean_json(result), indent=2) + '\n')
        print('Selected primary mean:', result['primary_mean'], 'thresholds:', {b: r['threshold'] for b, r in result['chosen'].items()}, flush=True)
    elif args.phase == 'timing':
        result = timing(rows, primary_config())
        (OUT / 'timing.json').write_text(json.dumps(clean_json(result), indent=2) + '\n')
    else:
        selection = primary_config()
        result = dict(experiment_id=ID, primary_mean=selection['primary_mean'], branches={})
        for branch in BRANCHES:
            threshold = selection['chosen'][branch]['threshold']
            groups = dict(overall=rows)
            groups.update({f"cell={c['id']}": [r for r in rows if r['scenario']['id'] == c['id']] for c in cases()})
            groups.update({f'geometry={g}': [r for r in rows if r['scenario']['geometry'] == g] for g in ('distributed', 'lower_half', 'outside')})
            result['branches'][branch] = {name: summarise(group, branch, threshold, phase, f'evaluation_{branch}_{name}') for name, group in groups.items()}
            print('Analysed', branch, flush=True)
        (OUT / 'evaluation.json').write_text(json.dumps(clean_json(result), indent=2) + '\n')
if __name__ == '__main__':
    main()
