"""Analyze aqs for the multivariate kriging experiments."""
from collections import Counter, defaultdict
import gzip
import json
import numpy as np
import pandas as pd
from experiments.common import ROOT, clean_json
from experiments.aqs_study import MODELS
from experiments.synthetic import seed_for

def main():
    base = ROOT / 'runs/AQS_task_v1'
    out = ROOT / 'results/AQS_task_v1'
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / 'data/aqs/analysis.csv')
    folds = json.loads((ROOT / 'data/aqs/folds.json').read_text())
    block_lookup = {(b['state'], b['site_id']): b['block'] for b in folds['bootstrap_blocks']}
    frames = []
    fit_counts = Counter()
    costs = []
    scale_km = {int(state): float((local.x_km.max() - local.x_km.min()) / (local.x_model.max() - local.x_model.min())) for state, local in data.groupby('State Code')}
    for path in sorted((base / 'evaluation').glob('*/*.json.gz')):
        with gzip.open(path, 'rt') as stream:
            r = json.load(stream)
        f = r['fold']
        ap = path.with_suffix('').with_suffix('.npz')
        with np.load(ap) as arrays:
            indices = arrays['test_indices']
            prediction = arrays['predictions']
            y = arrays['test_y']
            parameters = arrays['test_parameters']
        sites = data.loc[indices, 'site_id'].to_numpy()
        blocks = np.array([block_lookup[f['state'], site] for site in sites])
        distance = {v['analysis_row']: v for v in r['test_distances']}
        metadata = {'state': f['state'], 'task': f['task'], 'buffer_multiplier': f['buffer_multiplier'], 'variant': r['variant'], 'repeat': f['repeat'], 'fold': f['fold'], 'analysis_row': indices, 'site_id': sites, 'block': blocks, 'parameter': parameters, 'observed': y, 'target_distance_km': [distance[i]['target_nearest_model_distance'] * scale_km[f['state']] for i in indices], 'auxiliary_distance_km': [distance[i]['auxiliary_nearest_model_distance'] * scale_km[f['state']] for i in indices]}
        for mi, method in enumerate(MODELS):
            mean, latent, observed = prediction[mi]
            error = mean - y
            frame = pd.DataFrame({**metadata, 'method': method, 'prediction': mean, 'latent_variance': latent, 'observed_variance': observed, 'squared_error': error ** 2, 'absolute_error': np.abs(error)})
            if mi >= 4:
                frame['mlpd'] = -0.5 * (np.log(2 * np.pi * observed) + error ** 2 / observed)
                frame['coverage95'] = (np.abs(error) <= 1.959963984540054 * np.sqrt(observed)).astype(float)
                frame['width95'] = 2 * 1.959963984540054 * np.sqrt(observed)
            else:
                frame['mlpd'] = np.nan
                frame['coverage95'] = np.nan
                frame['width95'] = np.nan
            frames.append(frame)
        fits = [*r['marginals'], r['shared'], r['joint']]
        for fit in fits:
            fit_counts['fits'] += 1
            fit_counts['failed'] += not fit['success']
            fit_counts['bound'] += bool(fit['active_bounds'])
            fit_counts['above_gradient_tolerance'] += (fit['projected_gradient_norm'] or 0) > 1e-05
            fit_counts['failed_starts'] += sum((not s['success'] for s in fit['starts']))
        costs.append({'state': f['state'], 'task': f['task'], 'buffer_multiplier': f['buffer_multiplier'], 'variant': r['variant'], 'repeat': f['repeat'], 'fold': f['fold'], 'train_counts': {p: c['train'] for p, c in f['target_counts'].items()}, 'test_counts': {p: c['test'] for p, c in f['target_counts'].items()}, 'buffer_km': f['buffer_km'], 'research_wall_seconds': r['research_wall_seconds'], 'sum_fit_seconds': sum((v['runtime_seconds'] for v in fits))})
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(out / 'out_of_fold.csv.gz', index=False, compression='gzip')
    group = ['state', 'task', 'buffer_multiplier', 'variant', 'parameter']
    metrics = ['squared_error', 'absolute_error', 'mlpd', 'coverage95', 'width95']
    site = full.groupby(group + ['method', 'site_id', 'block'], as_index=False)[metrics].mean()
    site.to_csv(out / 'site_losses.csv.gz', index=False, compression='gzip')
    weights = {state: np.random.default_rng(seed_for('AQS', 'analysis', str(state), 0, 'spatial_block_bootstrap')).multinomial(10, np.full(10, 0.1), size=2000) for state in (6, 8, 48)}
    results = []
    pooled = defaultdict(list)
    partition = []
    distance_summary = []

    def ci(value):
        valid = np.isfinite(value)
        return np.quantile(value[valid], [0.025, 0.975]).tolist() if np.any(valid) else None
    for key, local in site.groupby(group, sort=True):
        state, task, buffer, variant, parameter = key
        ident = dict(zip(group, key))
        n = local.site_id.nunique()
        values = []
        blocks = None
        for method in MODELS:
            subset = local[local.method == method].sort_values('site_id')
            values.append(subset[metrics].to_numpy())
            blocks = subset.block.to_numpy() if blocks is None else blocks
        values = np.array(values)
        block_counts = np.bincount(blocks, minlength=10)
        denominator = weights[state] @ block_counts
        boot = []
        for value in values:
            block_sums = np.array([value[blocks == b].sum(axis=0) if np.any(blocks == b) else np.zeros(len(metrics)) for b in range(10)])
            numer = weights[state] @ block_sums
            boot.append(np.divide(numer, denominator[:, None], out=np.full_like(numer, np.nan), where=denominator[:, None] > 0))
        boot = np.array(boot)
        mean = values.mean(axis=1)
        baseline = MODELS.index('independent')
        valid = int(np.sum(denominator > 0))
        for mi, method in enumerate(MODELS):
            row = {**ident, 'method': method, 'sites': n, 'repeat_predictions': n * 10, 'occupied_spatial_blocks': int(np.sum(block_counts > 0)), 'bootstrap_draws': 2000, 'valid_bootstrap_draws': valid, 'mse': mean[mi, 0], 'mse_ci95': ci(boot[mi, :, 0]), 'rmse': np.sqrt(mean[mi, 0]), 'rmse_ci95': ci(np.sqrt(boot[mi, :, 0])), 'mae': mean[mi, 1], 'mae_ci95': ci(boot[mi, :, 1]), 'paired_rmse_difference_independent': np.sqrt(mean[mi, 0]) - np.sqrt(mean[baseline, 0]), 'paired_rmse_difference_independent_ci95': ci(np.sqrt(boot[mi, :, 0]) - np.sqrt(boot[baseline, :, 0])), 'risk_reduction_vs_independent_percent': 100 * (mean[baseline, 0] - mean[mi, 0]) / mean[baseline, 0], 'risk_reduction_vs_independent_ci95': ci(100 * (boot[baseline, :, 0] - boot[mi, :, 0]) / boot[baseline, :, 0])}
            for j, metric in enumerate(metrics[2:], 2):
                row[metric] = mean[mi, j]
                row[metric + '_ci95'] = ci(boot[mi, :, j])
            results.append(row)
            pooled[state, task, buffer, variant, method].append((parameter, mean[mi, 0] / mean[baseline, 0], boot[mi, :, 0] / boot[baseline, :, 0]))
    output_weighted = []
    for (state, task, buffer, variant, method), values in pooled.items():
        means = np.array([v[1] for v in values])
        draws = np.mean([v[2] for v in values], axis=0)
        output_weighted.append({'state': state, 'task': task, 'buffer_multiplier': buffer, 'variant': variant, 'method': method, 'equal_output_risk_reduction_percent': 100 * (1 - means.mean()), 'equal_output_risk_reduction_ci95': ci(100 * (1 - draws)), 'normalization': 'Equal mean of pollutant-specific MSE ratios to IND-SEPARATE; never pool squared physical units.', 'valid_bootstrap_draws': int(np.sum(np.isfinite(draws)))})
    for key, local in full.groupby(group + ['method'], sort=True):
        by_repeat = local.groupby('repeat').squared_error.mean().to_numpy()
        partition.append({**dict(zip(group + ['method'], key)), 'repeat_mse_minimum': float(by_repeat.min()), 'repeat_mse_median': float(np.median(by_repeat)), 'repeat_mse_maximum': float(by_repeat.max()), 'repeat_mse_std_descriptive': float(by_repeat.std(ddof=1))})
    bands = [0.0, 25.0, 50.0, 100.0, 250.0, float('inf')]
    primary = full[(full.variant == 'primary') & (full.buffer_multiplier == 0) & full.method.isin(['independent', 'joint'])]
    for key, local in primary.groupby(['state', 'task', 'parameter'], sort=True):
        a = local[local.method == 'independent'].sort_values(['repeat', 'analysis_row'])
        b = local[local.method == 'joint'].sort_values(['repeat', 'analysis_row'])
        for variable in ('target_distance_km', 'auxiliary_distance_km'):
            values = a[variable].to_numpy()
            for lo, hi in zip(bands[:-1], bands[1:]):
                mask = (values >= lo) & (values < hi)
                if not np.any(mask):
                    continue
                independent = a.squared_error.to_numpy()[mask].mean()
                joint = b.squared_error.to_numpy()[mask].mean()
                distance_summary.append({**dict(zip(['state', 'task', 'parameter'], key)), 'distance': variable, 'lower_km': lo, 'upper_km': hi if np.isfinite(hi) else None, 'repeat_records': int(mask.sum()), 'unique_sites': int(a.loc[mask, 'site_id'].nunique()), 'independent_mse': independent, 'joint_mse': joint, 'risk_reduction_percent': 100 * (independent - joint) / independent})
    result = {'training_splits': 1820, 'fits': dict(fit_counts), 'out_of_fold_method_records': len(full), 'results': results, 'output_weighted': output_weighted, 'partition_variation': partition, 'distance_summaries': distance_summary, 'costs': costs, 'uncertainty': '2000 paired resamples of10 coordinate-defined blocks per state, all10 repeated OOF losses for a site averaged before resampling. Use identical block multiplicities across models, tasks, buffers and instrument variants. Discard only empty-denominator bootstrap draws and report their number. RMSE recomputed from resampled MSE.', 'scope': 'Physical annual-summary prediction error on the observed network, conditional on fixed OOF predictions. Spatial bootstrap does not propagate refitting uncertainty, and intervals with few occupied blocks may be unstable. Simple baselines have no asserted predictive density; their probability metrics are null.'}
    (out / 'summary.json').write_text(json.dumps(clean_json(result), indent=2) + '\n')
    print(json.dumps({'fits': dict(fit_counts), 'method_records': len(full), 'summary_rows': len(results)}), flush=True)
    for row in results:
        if row['variant'] == 'primary' and row['state'] == 48 and (row['buffer_multiplier'] == 0) and (row['method'] == 'joint'):
            print(json.dumps(clean_json({k: row[k] for k in ('state', 'task', 'parameter', 'sites', 'risk_reduction_vs_independent_percent', 'risk_reduction_vs_independent_ci95', 'rmse', 'mlpd', 'coverage95', 'occupied_spatial_blocks')})), flush=True)
if __name__ == '__main__':
    main()
