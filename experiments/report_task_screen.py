"""Report task screen for the multivariate kriging experiments."""
from __future__ import annotations
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from experiments.analyze_task_screen import bootstrap_means, corrected_calibration_view, interval, load_records, replay, summarise
from experiments.task_screen_study import ROOT, METHOD
RESULT = ROOT / 'results/E6_task_v2'
FIGS = ROOT / 'figures'
LABELS = {'independent': 'Always SEP', 'diagonal': 'Always DIAG', 'joint': 'Always JOINT', 'direct': 'Direct validation (three)', 'screen': 'Screened validation (three)', 'retain_all': 'Retain all', 'loose': 'Loose bound', 'boundary': 'Retain boundaries', 'legacy': 'Earlier gate', 'no_shortcut': 'Without shortcut'}
GEOMETRIES = ('colocated', 'infill', 'partial', 'remote')

def ci(statistic, digits=2):
    if statistic is None:
        return '--'
    value, bounds = (statistic['estimate'], statistic['ci95'])
    return f'{value:.{digits}f} [{bounds[0]:.{digits}f}, {bounds[1]:.{digits}f}]'

def table(header, lines, columns):
    return '\n'.join(['\\begin{tabular}{@{}' + columns + '@{}}', '\\toprule', header + ' \\\\', '\\midrule', *[line + ' \\\\' for line in lines], '\\bottomrule', '\\end{tabular}', ''])

def calibration_table(thresholds):
    """Render operational costs for the calibrated thresholds."""
    thresholds = corrected_calibration_view(thresholds)
    lines = []
    for candidate in thresholds['all_candidates']:
        if candidate['method'] == 'screen':
            lines.append(f"{candidate['threshold']:g} & {candidate['excluded']} & {candidate['false_exclusions']} & {100 * candidate['upper_risk_excess_fraction']:.3f} & {100 * candidate['upper_positive_regret_fraction']:.3f} & {candidate['mean_component_seconds']:.3f} & {('Yes' if candidate['admissible'] else 'No')}")
    return table('$t_U$ & Excluded & False excl. & Risk upper (\\%) & Regret upper (\\%) & Seconds & Admissible', lines, 'rrrrrrc')

def main():
    summary = json.loads((RESULT / 'summary.json').read_text())
    pair = json.loads((RESULT / 'pair_control/summary.json').read_text())
    timing = json.loads((RESULT / 'timing/summary.json').read_text())
    pair_timing = json.loads((RESULT / 'pair_control/timing-summary.json').read_text())
    thresholds = corrected_calibration_view(json.loads((METHOD / 'thresholds.json').read_text()))
    rows = load_records(ROOT / 'runs/E6_task_v2', 'evaluation')[1] if (ROOT / 'runs/E6_task_v2/evaluation').exists() else []
    pooled = summary['pooled']
    outputs = {}
    time_values = {'TaskScreenSaving': timing['saving_percent']['estimate'], 'TaskScreenSavingLow': timing['saving_percent']['ci95'][0], 'TaskScreenSavingHigh': timing['saving_percent']['ci95'][1], 'TaskPairExtraTime': -pair_timing['screen_saving_percent']['estimate'], 'TaskPairExtraTimeLow': -pair_timing['screen_saving_percent']['ci95'][1], 'TaskPairExtraTimeHigh': -pair_timing['screen_saving_percent']['ci95'][0]}
    number_path = ROOT / 'tables/task_screen_numbers.tex'
    number_path.write_text('' + ''.join(('\\newcommand{\\' + name + '}{' + f'{value:.2f}' + '}\n' for name, value in time_values.items())))
    core = []
    for method in ('independent', 'diagonal', 'joint', 'pair', 'direct', 'screen'):
        p = pair['pooled'] if method == 'pair' else pooled[method]
        label = 'Direct validation (two)' if method == 'pair' else LABELS[method]
        selected = p['selected_joint'] if method == 'pair' else p['counts']['selections']['joint']
        core.append(f"{label} & {p['risk']['estimate']:.4f} & {ci(p['risk_reduction_percent'])} & {100 * selected / 9600:.1f}")
    outputs['table_e6_task_core.tex'] = table('Procedure & Mean risk & Reduction (\\%) [95\\% CI] & JOINT (\\%)', core, 'lrrr')
    errors = []
    for method in ('direct', 'screen', 'retain_all', 'loose', 'boundary', 'legacy', 'no_shortcut'):
        p = pooled[method]
        errors.append(f"{LABELS[method]} & {ci(p['exclusion_percent'])} & {ci(p['screen_false_exclusion_percent'])} & {ci(p['final_fpr_percent'])} & {ci(p['final_fnr_percent'])}")
    outputs['table_e6_task_errors.tex'] = table('Rule & Exclude (\\%) & False exclude (\\%) & FPR (\\%) & FNR (\\%)', errors, 'lrrrr')
    controls = []
    for method in ('direct', 'screen', 'retain_all', 'loose', 'boundary', 'legacy', 'no_shortcut'):
        p = pooled[method]
        controls.append(f"{LABELS[method]} & {ci(p['risk_excess_vs_direct_percent'])} & {ci(p['component_saving_percent'])} & {p['joint_fits_mean']['estimate']:.2f} & {ci(p['excess_risk_over_best_percent'])}")
    outputs['table_e6_task_ablations.tex'] = table('Rule & Excess risk (\\%) & Component saving (\\%) & Joint fits & Regret (\\%)', controls, 'lrrrr')
    outputs['table_e6_task_calibration.tex'] = calibration_table(thresholds)
    calibration_metrics = []
    secondary = {} if rows else json.loads((RESULT / 'secondary-assessment.json').read_text())
    for method in ('independent', 'diagonal', 'joint', 'pair', 'direct', 'screen'):
        p = pair['pooled'] if method == 'pair' else pooled[method]
        label = 'Direct validation (two)' if method == 'pair' else LABELS[method]
        calibration_metrics.append(f"{label} & {ci(p['coverage95'])} & {ci(p['width95'])} & {ci(p['mlpd'])}")
        if method != 'pair' and rows:
            metrics = [r['assessment'][replay(r, method, summary['threshold'])['model']] for r in rows]
            values = np.array([[m['latent_mse'], m['observed_mse']] for m in metrics])
            b = bootstrap_means(values, rows, 'evaluation')
            secondary[method] = {name: interval(values[:, i].mean(), b[:, i]) for i, name in enumerate(('latent_mse', 'observed_mse'))}
    outputs['table_e6_task_predictive_calibration.tex'] = table('Procedure & Coverage (\\%) & Width & Mean log score', calibration_metrics, 'lrrr')
    mse_rows = [f"{LABELS[method]} & {ci(s['latent_mse'], 4)} & {ci(s['observed_mse'], 4)}" for method, s in secondary.items() if method in LABELS]
    outputs['table_e6_task_test_mse.tex'] = table('Procedure & Realised latent MSE & Realised response MSE', mse_rows, 'lrr')
    if rows:
        independent = np.array([r['assessment']['independent']['conditional_risk'] for r in rows])
        regret = np.array([replay(r, 'screen', summary['threshold'])['full_excluded'] * max(min((r['assessment'][m]['conditional_risk'] for m in ('independent', 'diagonal'))) - r['assessment']['joint']['conditional_risk'], 0) for r in rows])
        b = bootstrap_means(np.c_[independent, regret], rows, 'evaluation')
        secondary['positive_exclusion_regret_percent'] = interval(100 * regret.mean() / independent.mean(), 100 * b[:, 1] / b[:, 0])
        (RESULT / 'secondary-assessment.json').write_text(json.dumps(secondary, indent=2) + '\n')
    timing_rows = [f"Three-candidate direct & {ci(timing['direct_seconds'], 3)} & {ci(timing['screen_seconds'], 3)} & {ci(timing['saving_percent'])}", f"Two-candidate direct & {ci(pair_timing['pair_seconds'], 3)} & {ci(pair_timing['screen_seconds'], 3)} & {ci(pair_timing['screen_saving_percent'])}"]
    outputs['table_e6_task_timing.tex'] = table('Comparator & Direct seconds & Screen seconds & Screen saving (\\%)', timing_rows, 'lrrr')
    cells = {} if rows else json.loads((RESULT / 'geometry-correlation.json').read_text())
    grid_rows = []
    heatmaps = [np.zeros((4, 3)) for _ in range(3)]
    for g, geometry in enumerate(GEOMETRIES):
        for j, rho in enumerate((0.0, 0.4, 0.8)):
            subset = [r for r in rows if r['scenario']['geometry'] == geometry and r['scenario']['rho'] == rho]
            s = summarise(subset, summary['threshold'], summary['loose_threshold']) if rows else cells[f'{geometry}_{rho:g}']
            cells[f'{geometry}_{rho:g}'] = s
            d, c = (s['diagonal']['risk']['estimate'], s['joint']['risk']['estimate'])
            direct_fits, screened_fits = (s['direct']['joint_fits_mean']['estimate'], s['screen']['joint_fits_mean']['estimate'])
            heatmaps[0][g, j] = 100 * (1 - c / d)
            heatmaps[1][g, j] = s['screen']['risk_reduction_percent']['estimate']
            heatmaps[2][g, j] = 100 * (1 - screened_fits / direct_fits)
            grid_rows.append(f"{geometry.capitalize()} & {rho:g} & {ci(s['diagonal']['risk_reduction_percent'])} & {ci(s['joint']['risk_reduction_percent'])} & {ci(s['screen']['risk_reduction_percent'])}")
    outputs['table_e6_task_geometry.tex'] = table('Geometry & $\\rho$ & DIAG vs SEP (\\%) & JOINT vs SEP (\\%) & Screen vs SEP (\\%)', grid_rows, 'lrrrr')
    (RESULT / 'geometry-correlation.json').write_text(json.dumps(cells, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.75), layout='constrained')
    titles = ['(a) Coupling: JOINT vs DIAG', '(b) Screened prediction vs SEP', '(c) Coupled fits avoided']
    for index, (axis, values) in enumerate(zip(axes, heatmaps)):
        if index < 2:
            limit = max(5, float(np.ceil(np.max(np.abs(values)) / 5) * 5))
            plot = axis.imshow(values, cmap='RdBu', norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit), aspect='auto')
        else:
            plot = axis.imshow(values, cmap='Blues', vmin=0, vmax=100, aspect='auto')
        axis.set_title(titles[index], fontsize=8, pad=7)
        axis.set_xticks(range(3), ['0', '0.4', '0.8'])
        axis.set_xlabel('Generating correlation $\\rho$')
        axis.set_yticks(range(4), ['Colocated', 'Infill', 'Partial', 'Remote'] if index == 0 else [''] * 4)
        axis.tick_params(length=0)
        for i in range(4):
            for j in range(3):
                rgba = plot.cmap(plot.norm(values[i, j]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                shown = 0.0 if abs(values[i, j]) < 0.05 else values[i, j]
                axis.text(j, i, f'{shown:.1f}', ha='center', va='center', color='white' if luminance < 0.5 else 'black', fontsize=8)
        bar = fig.colorbar(plot, ax=axis, location='bottom', shrink=0.9, pad=0.08, aspect=20)
        bar.ax.tick_params(labelsize=7)
        bar.set_label('Risk reduction (%)' if index < 2 else 'Fit reduction (%)', fontsize=7)
    figure = FIGS / 'e6_task_geometry.pdf'
    fig.savefig(figure, bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(FIGS / 'e6_task_geometry.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    for name, value in outputs.items():
        (FIGS / name).write_text('' + value)
if __name__ == '__main__':
    main()
