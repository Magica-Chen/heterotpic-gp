"""Analyze e5 extension for the multivariate kriging experiments."""
import gzip
import json
import numpy as np
import pandas as pd
from experiments.common import ROOT
from experiments.synthetic import seed_for

def interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()

def main():
    base = ROOT / 'runs/E5_extension_v1'
    out = ROOT / 'results/E5_extension_v1'
    manifest = json.loads((base / 'evaluation_manifest.json').read_text())
    names = ('correlation_only_independent', 'correlation_only_joint', 'full_independent', 'full_joint')
    keys = ('risk', 'latent_mse', 'rmse', 'observed_mse', 'mlpd', 'coverage95', 'width95')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    repetitions = manifest.get('replicates', 300)
    for s in manifest['scenarios']:
        data = []
        diagnostics = []
        for rep in range(repetitions):
            path = base / 'evaluation' / s['id'] / f'dataset_{rep:04d}.json.gz'
            with gzip.open(path, 'rt') as stream:
                r = json.load(stream)
            data.append([[r['assessment'][name][k] for k in keys] for name in names])
            diagnostics.append(r['fits'])
        data = np.array(data)
        weights = np.random.default_rng(seed_for('E5_extension', 'analysis', s['id'], 0)).multinomial(repetitions, np.full(repetitions, 1 / repetitions), size=2000) / repetitions
        mean = data.mean(axis=0)
        boot = (weights @ data.reshape(repetitions, -1)).reshape(2000, len(names), len(keys))
        for i, name in enumerate(names):
            mode = name.rsplit('_', 1)[0]
            own = 0 if i < 2 else 2
            joint = own + 1
            result = {'family': s['family'], 'geometry': s['geometry'], 'method': name, 'datasets': repetitions}
            for j, key in enumerate(keys):
                result[key] = float(mean[i, j])
                result[key + '_ci95'] = interval(boot[:, i, j])
            result.update(risk_reduction_vs_own_independent_percent=float(100 * (1 - mean[i, 0] / mean[own, 0])), risk_reduction_vs_own_independent_ci95=interval(100 * (1 - boot[:, i, 0] / boot[:, own, 0])), risk_reduction_vs_full_independent_percent=float(100 * (1 - mean[i, 0] / mean[2, 0])), risk_reduction_vs_full_independent_ci95=interval(100 * (1 - boot[:, i, 0] / boot[:, 2, 0])), paired_rmse_improvement=float(mean[own, 2] - mean[joint, 2]), paired_rmse_improvement_ci95=interval(boot[:, own, 2] - boot[:, joint, 2]), paired_mlpd_improvement=float(mean[joint, 4] - mean[own, 4]), paired_mlpd_improvement_ci95=interval(boot[:, joint, 4] - boot[:, own, 4]))
            fits = [r[mode]['joint'] if name.endswith('_joint') else r[mode]['marginals'][0] for r in diagnostics]
            result['active_bound_count'] = sum((bool(f['active_bounds']) for f in fits))
            result['above_gradient_tolerance_count'] = sum((f['projected_gradient_norm'] > f['config']['gtol'] for f in fits))
            result['failed_fit_count'] = sum((not f['success'] for f in fits))
            rows.append(result)
    original = []
    for family, filename in [('separable', 's5_kriging_revisited.csv'), ('original_lmc', 's5_lmc_misspec.csv')]:
        path = ROOT / 'data/original_simulations' / filename
        frame = pd.read_csv(path)
        for geometry in ('isotopic', 'interleaved', 'separated'):
            subset = frame[frame.regime == geometry].sort_values('rep')
            weights = np.random.default_rng(seed_for('E5_extension', 'original_intervals', family + '_' + geometry, 0)).multinomial(80, np.full(80, 1 / 80), size=2000) / 80
            result = {'family': family, 'geometry': geometry, 'datasets': 80}
            for key in ('rmse_diff', 'mlpd_diff'):
                x = subset[key].to_numpy()
                result[key] = float(x.mean())
                result[key + '_ci95'] = interval(weights @ x)
            original.append(result)
    result = {'datasets': repetitions * len(manifest['scenarios']), 'results': rows, 'original80_intervals': original, 'scope': '2000 dataset-paired resamples separately within each fixed family/geometry; matched modes share weights. Conditional-risk ratio is ratio of means, RMSE difference is mean of per-dataset RMSE differences to match original convention. Original80-dataset intervals use preserved CSV records, separate from fresh300; no CI on deterministic oracle quantities.'}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for r in rows:
        if r['method'].endswith('_joint'):
            print(r['family'], r['geometry'], r['method'], 'own risk', r['risk_reduction_vs_own_independent_percent'], r['risk_reduction_vs_own_independent_ci95'], flush=True)
if __name__ == '__main__':
    main()
