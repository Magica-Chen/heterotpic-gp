"""Analyze transfer for the multivariate kriging experiments."""
from experiments.common import ROOT as PACKAGE_ROOT
import argparse
from collections import defaultdict
import gzip
import json
import numpy as np
ROOT = PACKAGE_ROOT
from experiments.analyze_task_screen import replay
from experiments.common import clean_json
from experiments.synthetic import seed_for
from experiments.transfer_study import ID, RUN, METHOD, REPS, STUDIES, branches, scenarios
METHODS = ('independent', 'diagonal', 'joint', 'direct', 'screen', 'pair')
BOOTSTRAPS = 2000
OUT = ROOT / 'results' / ID

def decisions(row):
    result = [replay(row, method, 0.2) for method in METHODS[:-1]]
    selected = sum((f['joint_sse'] for f in row['folds'])) < 0.98 * sum((f['independent_sse'] for f in row['folds']))
    name = 'joint' if selected and row['joint']['success'] else 'independent'
    result.append(dict(model=name, full_excluded=False, joint_fits=4 + int(selected), component_seconds=None))
    return result

def compact(raw, branch):
    row = raw['branches'][branch]
    ds = decisions(row)
    metrics = [row['assessment'][d['model']] for d in ds]
    r = np.array([row['assessment'][k]['conditional_risk'] for k in METHODS[:3]])
    fits = row['marginals'] + [row['diagonal'], row['joint']] + [f[k] for f in row['folds'] for k in ('target_marginal', 'diagonal', 'joint')]
    answer = dict(scenario=raw['scenario'], replication=raw['replication'], risk=np.array([m['conditional_risk'] for m in metrics]), selected_joint=np.array([d['model'] == 'joint' for d in ds]), selected_model=[d['model'] for d in ds], excluded=np.array([d['full_excluded'] for d in ds]), component_seconds=np.array([d['component_seconds'] or 0.0 for d in ds]), joint_fits=np.array([d['joint_fits'] for d in ds]), beneficial=bool(r[2] < 0.98 * min(r[:2])), best=float(r.min()), unresolved=bool(row['scores']['flags']), score=row['scores']['opportunity'], fit_diagnostics=dict(fits=len(fits), failed=sum((not f['success'] for f in fits)), boundary=sum((bool(f['active_bounds']) for f in fits)), above_gtol=sum((f['projected_gradient_norm'] is None or f['projected_gradient_norm'] > 1e-05 for f in fits))), metrics={k: np.array([m[k] for m in metrics]) for k in metrics[0] if k != 'conditional_risk'})
    if raw['study'] == 'q2':
        answer['mean_error'] = np.array([raw['predictions'][branch][d['model']][0] for d in ds]) - np.array(raw['data']['oracle'])
    return answer

def bootstrap_means(values, rows, study, namespace):
    """Cluster geometry arms by macro-replication where simulations are paired."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed_for(ID, 'analysis', study + '_' + namespace, 0, 'bootstrap'))
    strata = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        s = row['scenario']
        key = s['id'] if study not in ('q1', 'q2') else 'all' if study == 'q1' else str(s['n1'])
        strata[key][row['replication']].append(i)
    boot = np.zeros((BOOTSTRAPS, values.shape[1]))
    for key in sorted(strata):
        units = np.array([values[indices].sum(axis=0) for _, indices in sorted(strata[key].items())])
        weights = rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=BOOTSTRAPS)
        boot += weights @ units / len(rows)
    return boot

def interval(value, samples):
    return dict(estimate=float(value), ci95=np.quantile(samples, [0.025, 0.975]).tolist())

def reference_errors(namespace):
    ref = json.loads((ROOT / 'data/queue_inputs/q2_reference.json').read_text())
    rng = np.random.default_rng(seed_for(ID, 'analysis', namespace, 0, 'reference_bootstrap'))
    result = []
    for point in ref['points']:
        v = np.asarray(point['run_values'])
        weights = rng.multinomial(len(v), np.full(len(v), 1 / len(v)), size=BOOTSTRAPS)
        result.append(weights @ v / len(v) - np.mean(v))
    return np.asarray(result).T

def summarise(rows, study, namespace):
    risk = np.array([r['risk'] for r in rows])
    benefit = np.array([r['beneficial'] for r in rows])
    selected = np.array([r['selected_joint'] for r in rows])
    excluded = np.array([r['excluded'] for r in rows])
    fields = dict(risk=risk, component_seconds=np.array([r['component_seconds'] for r in rows]), joint_fits=np.array([r['joint_fits'] for r in rows]), excluded=excluded, false_exclusion=excluded * benefit[:, None], false_positive=selected * ~benefit[:, None], false_negative=~selected * benefit[:, None], benefit=benefit[:, None], nonbenefit=~benefit[:, None], regret=risk - np.array([r['best'] for r in rows])[:, None])
    for key in rows[0]['metrics']:
        fields[key] = np.array([r['metrics'][key] for r in rows])
    if study == 'q2':
        fields['mean_error'] = np.array([r['mean_error'].ravel() for r in rows])
    offsets, pieces, cursor = ({}, [], 0)
    for key, value in fields.items():
        offsets[key] = slice(cursor, cursor + value.shape[1])
        cursor += value.shape[1]
        pieces.append(value)
    boot = bootstrap_means(np.column_stack(pieces), rows, study, namespace)
    b = {key: boot[:, offset] for key, offset in offsets.items()}
    if study == 'q2':
        delta = reference_errors(namespace)
        err = b['mean_error'].reshape(BOOTSTRAPS, len(METHODS), -1)
        b['risk'] = b['risk'] - 2 * np.mean(err * delta[:, None, :], axis=2) + np.mean(delta ** 2, axis=1)[:, None]
    means = {key: value.mean(axis=0) for key, value in fields.items()}
    results = {}
    for j, name in enumerate(METHODS):
        entry = dict(risk=interval(means['risk'][j], b['risk'][:, j]), risk_reduction_percent=interval(100 * (1 - means['risk'][j] / means['risk'][0]), 100 * (1 - b['risk'][:, j] / b['risk'][:, 0])), excess_over_direct_percent=interval(100 * (means['risk'][j] - means['risk'][3]) / means['risk'][0], 100 * (b['risk'][:, j] - b['risk'][:, 3]) / b['risk'][:, 0]), exclusion_percent=interval(100 * means['excluded'][j], 100 * b['excluded'][:, j]), regret_over_best_percent=interval(100 * means['regret'][j] / means['risk'][0], 100 * b['regret'][:, j] / b['risk'][:, 0]), joint_fits_mean=interval(means['joint_fits'][j], b['joint_fits'][:, j]), selected_counts={m: sum((r['selected_model'][j] == m for r in rows)) for m in METHODS[:3]})
        if name != 'pair':
            entry['component_seconds'] = interval(means['component_seconds'][j], b['component_seconds'][:, j])
            entry['component_saving_vs_direct_percent'] = interval(100 * (1 - means['component_seconds'][j] / means['component_seconds'][3]), 100 * (1 - b['component_seconds'][:, j] / b['component_seconds'][:, 3]))
        for numerator, denominator, label in (('false_exclusion', 'benefit', 'false_exclusion_percent'), ('false_negative', 'benefit', 'final_fnr_percent'), ('false_positive', 'nonbenefit', 'final_fpr_percent')):
            denominator_value = means[denominator][0]
            if denominator_value == 0:
                entry[label] = None
            else:
                valid = b[denominator][:, 0] > 0
                entry[label] = interval(100 * means[numerator][j] / denominator_value, 100 * b[numerator][valid, j] / b[denominator][valid, 0])
        for key in rows[0]['metrics']:
            entry[key] = interval(means[key][j], b[key][:, j])
        results[name] = entry
    return dict(datasets=len(rows), beneficial_joint=int(benefit.sum()), nonbeneficial_joint=int((~benefit).sum()), unresolved=sum((r['unresolved'] for r in rows)), score_quantiles=np.quantile([r['score'] for r in rows if r['score'] is not None], [0, 0.25, 0.5, 0.75, 1]).tolist(), fit_diagnostics={k: sum((r['fit_diagnostics'][k] for r in rows)) for k in rows[0]['fit_diagnostics']}, methods=results)

def load_study(study, phase='evaluation'):
    result, sensitivity, _ = ({b: [] for b in branches(study)}, [], None)
    for s in scenarios(study):
        for rep in range(REPS[study] if phase == 'evaluation' else 1):
            path = RUN / phase / study / s['id'] / f'dataset_{rep:04d}.json.gz'
            with gzip.open(path, 'rt') as f:
                raw = json.load(f)
            for b in result:
                result[b].append(compact(raw, b))
            if raw['coordinate_sensitivity'] is not None:
                sensitivity.append(dict(scenario=s, replication=rep, **raw['coordinate_sensitivity']))
    return (result, sensitivity, None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--studies', nargs='+', choices=STUDIES, default=STUDIES)
    parser.add_argument('--phase', choices=('pilot', 'evaluation'), default='evaluation')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for study in args.studies:
        rows, sensitivity, digest = load_study(study, args.phase)
        report = dict(study=study, phase=args.phase, bootstraps=BOOTSTRAPS, branches={})
        group_keys = {'balanced': ('geometry',), 'robustness': ('family', 'geometry'), 'scales': ('dimension', 'geometry', 'anisotropy_ratio'), 'q1': ('geometry',), 'q2': ('geometry', 'n1')}[study]
        for branch, values in rows.items():
            groups = dict(overall=values)
            for key in group_keys:
                for value in sorted(set((r['scenario'][key] for r in values)), key=str):
                    groups[f'{key}={value}'] = [r for r in values if r['scenario'][key] == value]
            for s in scenarios(study):
                groups[f"cell={s['id']}"] = [r for r in values if r['scenario']['id'] == s['id']]
            report['branches'][branch] = {name: summarise(group, study, branch + '_' + name) for name, group in groups.items()}
        report['coordinate_sensitivity'] = sensitivity
        (OUT / f'{args.phase}_{study}.json').write_text(json.dumps(clean_json(report), indent=2) + '\n')
        print(study, 'analysed', sum((len(v) for v in rows.values())), 'branch datasets', flush=True)
if __name__ == '__main__':
    main()
