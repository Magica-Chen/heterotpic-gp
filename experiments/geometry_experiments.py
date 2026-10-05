"""Recompute the deterministic quantities and figures in Experiments I-A–I-C."""
import json
import numpy as np
import pandas as pd
from experiments.geometry import global_bpi, summarise_pair_geometry
from experiments.oracle import OracleParams, W_matrix, average_oracle_gain, latent_coupling_score, predictive_variance
from experiments import original_simulations as original
from experiments.common import ROOT

def main():
    root = ROOT
    figures = root / 'figures'
    tables = root / 'tables'
    output = root / 'results/paper_geometry'
    for directory in (figures, tables, output):
        directory.mkdir(parents=True, exist_ok=True)
    original.FIGS_DIR = figures
    grid = np.linspace(0.0, 1.0, 200)[:, None]
    params = OracleParams('matern32', 1.0, 0.15, np.array([[1.0, 0.7], [0.7, 1.0]]), np.array([0.08, 0.08]))
    rows, variances = ([], [])
    for regime in ('interleaved', 'moderate', 'separated'):
        xs = original._design_zero_overlap(regime)
        geometry = summarise_pair_geometry(*xs, np.linspace(0.0, 0.3, 60))
        original._plot_s1_design(xs, regime)
        original._plot_s1_coverage(geometry, regime)
        for target in (0, 1):
            marginal = np.mean([predictive_variance(x[None], xs, params, target, False) for x in grid])
            joint = np.mean([predictive_variance(x[None], xs, params, target, True) for x in grid])
            rows.append(dict(regime=regime, target=target + 1, omega=geometry.omega, tpi_forward=geometry.tpi_pq if target == 0 else geometry.tpi_qp, W12=W_matrix(xs, params)[0, 1], avg_oracle_gain=marginal - joint))
            variances.append(dict(geometry=regime, target=target + 1, independent_variance=marginal, joint_variance=joint, oracle_gain=marginal - joint, relative_gain_percent=100 * (1 - joint / marginal)))
    s1 = pd.DataFrame(rows)
    s2 = original.run_s2()
    original.write_tables(s1=s1, s2=s2)
    xs = [np.linspace(0.0, 1.0, 35)[:, None], np.linspace(0.0, 0.55, 15)[:, None], np.linspace(0.82, 1.0, 12)[:, None]]
    params = OracleParams('matern32', 1.0, 0.18, np.array([[1.0, 0.85, 0.65], [0.85, 1.0, 0.8], [0.65, 0.8, 1.0]]), np.array([0.02, 0.03, 0.03]))
    bpi = global_bpi(xs, grid)
    s3 = pd.DataFrame([dict(target=j + 1, B_j=bpi[j], coupling=latent_coupling_score(params.Lambda, j), avg_oracle_gain=average_oracle_gain(grid, xs, params, j)) for j in range(3)])
    lines = ['\\begin{tabular}{rrrr}', '\\hline', 'Output & $B_j$ & $R_j^2$ & Avg. gain\\\\', '\\hline']
    for row in s3.to_dict('records'):
        lines.append(f"{row['target']} & {row['B_j']:.3f} & {row['coupling']:.3f} & {row['avg_oracle_gain']:.3f}" + '\\\\')
    lines += ['\\hline', '\\end{tabular}']
    (figures / 'table_s3_summary.tex').write_text('\n'.join(lines) + '\n')
    lines = ['\\begin{tabular}{llrrrr}', '\\toprule', 'Geometry & Target & Marginal variance & Joint variance & Oracle gain & Relative gain (\\%) \\\\', '\\midrule']
    labels = {'interleaved': 'Interleaved', 'moderate': 'Partial coverage', 'separated': 'Remote'}
    for row in variances:
        lines.append(f"{labels[row['geometry']]} & {row['target']} & {row['independent_variance']:.5f} & {row['joint_variance']:.5f} & {row['oracle_gain']:.5f} & {row['relative_gain_percent']:.2f}" + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (tables / 'e1_relative_oracle_table.tex').write_text('\n'.join(lines) + '\n')
    for name, frame in [('I-A', s1), ('I-B', s2), ('I-C', s3)]:
        frame.to_csv(output / (name + '.csv'), index=False)
    (output / 'I-A-variances.json').write_text(json.dumps(variances, indent=2) + '\n')
    print('Computed geometry, oracle gains and figures for I-A/I-B/I-C.')
