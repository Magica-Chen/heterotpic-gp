"""Analyze aqs lmc for the multivariate kriging experiments."""
from collections import Counter, defaultdict
import gzip
import json
import numpy as np
import pandas as pd
from experiments.common import ROOT, clean_json
from experiments.aqs_lmc_study import MODELS
from experiments.synthetic import seed_for

def main():
    base = ROOT / 'runs/AQS_LMC_v1'
    out = ROOT / 'results/AQS_LMC_v1'
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / 'data/aqs/analysis.csv')
    folds = json.loads((ROOT / 'data/aqs/folds.json').read_text())
    lookup = {(r['state'], r['site_id']): r['block'] for r in folds['bootstrap_blocks']}
    frames = []
    counts = Counter()
    costs = []
    for path in sorted((base / 'evaluation').glob('*.json.gz')):
        with gzip.open(path, 'rt') as stream:
            r = json.load(stream)
        f = r['fold']
        ap = path.with_suffix('').with_suffix('.npz')
        with np.load(ap) as a:
            pred = a['predictions']
            y = a['test_y']
            indices = a['test_indices']
            parameters = a['test_parameters']
        sites = data.loc[indices, 'site_id'].to_numpy()
        block = [lookup[f['state'], s] for s in sites]
        for i, method in enumerate(MODELS):
            mean, latent, var = pred[i]
            error = mean - y
            frames.append(pd.DataFrame({'state': f['state'], 'task': f['task'], 'repeat': f['repeat'], 'fold': f['fold'], 'analysis_row': indices, 'parameter': parameters, 'site_id': sites, 'block': block, 'method': method, 'prediction': mean, 'observed': y, 'latent_variance': latent, 'observed_variance': var, 'squared_error': error ** 2, 'absolute_error': abs(error), 'mlpd': -0.5 * (np.log(2 * np.pi * var) + error ** 2 / var), 'coverage95': (abs(error) <= 1.959963984540054 * np.sqrt(var)).astype(float), 'width95': 2 * 1.959963984540054 * np.sqrt(var)}))
        for fit in [*r['mixture_marginals'], r['lmc']]:
            counts['fits'] += 1
            counts['failed'] += not fit['success']
            counts['bound'] += bool(fit['active_bounds'])
            counts['above_gradient_tolerance'] += (fit['projected_gradient_norm'] or 0) > 1e-05
            counts['failed_starts'] += sum((not s['success'] for s in fit['starts']))
        costs.append({'state': f['state'], 'task': f['task'], 'fold_index': r['fold_index'], 'research_wall_seconds': r['research_wall_seconds'], 'sum_fit_seconds': sum((f['runtime_seconds'] for f in [*r['mixture_marginals'], r['lmc']]))})
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(out / 'out_of_fold.csv.gz', index=False, compression='gzip')
    groups = ['state', 'task', 'parameter']
    metrics = ['squared_error', 'absolute_error', 'mlpd', 'coverage95', 'width95']
    site = full.groupby(groups + ['method', 'site_id', 'block'], as_index=False)[metrics].mean()
    weights = {state: np.random.default_rng(seed_for('AQS', 'analysis', str(state), 0, 'spatial_block_bootstrap')).multinomial(10, np.full(10, 0.1), size=2000) for state in (6, 8, 48)}
    results = []
    pooled = defaultdict(list)

    def ci(a):
        return np.quantile(a[np.isfinite(a)], [0.025, 0.975]).tolist()
    for key, local in site.groupby(groups, sort=True):
        state, task, parameter = key
        values = []
        boot = []
        for method in MODELS:
            a = local[local.method == method].sort_values('site_id')
            value = a[metrics].to_numpy()
            blocks = a.block.to_numpy()
            count = np.bincount(blocks, minlength=10)
            sums = np.array([value[blocks == b].sum(axis=0) for b in range(10)])
            den = weights[state] @ count
            draw = np.divide(weights[state] @ sums, den[:, None], out=np.full((2000, len(metrics)), np.nan), where=den[:, None] > 0)
            values.append(value.mean(axis=0))
            boot.append(draw)
        mean = np.array(values)
        boot = np.array(boot)
        for i, method in enumerate(MODELS):
            row = {'state': state, 'task': task, 'parameter': parameter, 'method': method, 'sites': len(a), 'occupied_spatial_blocks': len(set(blocks)), 'valid_bootstrap_draws': int(np.sum(den > 0)), 'mse': mean[i, 0], 'mse_ci95': ci(boot[i, :, 0]), 'rmse': np.sqrt(mean[i, 0]), 'rmse_ci95': ci(np.sqrt(boot[i, :, 0]))}
            for j, name in enumerate(metrics[1:], 1):
                row[name] = mean[i, j]
                row[name + '_ci95'] = ci(boot[i, :, j])
            for index, reference in [(0, 'primary_independent'), (1, 'mixture_independent'), (2, 'primary_joint')]:
                row['risk_reduction_vs_' + reference + '_percent'] = 100 * (mean[index, 0] - mean[i, 0]) / mean[index, 0]
                row['risk_reduction_vs_' + reference + '_ci95'] = ci(100 * (boot[index, :, 0] - boot[i, :, 0]) / boot[index, :, 0])
                pooled[state, task, method, reference].append((mean[i, 0] / mean[index, 0], boot[i, :, 0] / boot[index, :, 0]))
            results.append(row)
    output_weighted = []
    for (state, task, method, reference), values in pooled.items():
        point = np.mean([v[0] for v in values])
        draw = np.mean([v[1] for v in values], axis=0)
        output_weighted.append({'state': state, 'task': task, 'method': method, 'reference': reference, 'equal_output_risk_reduction_percent': 100 * (1 - point), 'equal_output_risk_reduction_ci95': ci(100 * (1 - draw))})
    result = {'training_splits': 650, 'fit_counts': dict(counts), 'results': results, 'output_weighted': output_weighted, 'costs': costs, 'out_of_fold_method_records': len(full), 'scope': 'All primary-instrument unbuffered A/B/C splits. Same fixed-prediction2000-spatial-block resamples as primary analysis; paired marginal-mixture comparator distinguishes marginal flexibility from borrowing. Intervals omit refitting and may be unstable with few occupied blocks.'}
    (out / 'summary.json').write_text(json.dumps(clean_json(result), indent=2) + '\n')
    print(json.dumps(dict(counts)), flush=True)
    for row in results:
        if row['state'] == 48 and row['method'] == 'lmc':
            print(json.dumps(clean_json({k: row[k] for k in ('task', 'parameter', 'risk_reduction_vs_primary_independent_percent', 'risk_reduction_vs_mixture_independent_percent', 'risk_reduction_vs_mixture_independent_ci95')})), flush=True)
if __name__ == '__main__':
    main()
