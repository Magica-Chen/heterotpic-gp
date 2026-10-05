"""Report penalties for the multivariate kriging experiments."""
from experiments.common import ROOT as PACKAGE_ROOT
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = PACKAGE_ROOT
FIGS = ROOT / 'figures'
TEX = ROOT / 'tables'

def number(value, scale=1):
    x = scale * value
    return f'{(0.0 if abs(x) < 0.0005 else x):.3f}'

def ci(value, scale=1):
    lo, hi = value['ci95']
    return f"{number(value['mean'], scale)} [{number(lo, scale)}, {number(hi, scale)}]"

def risk_table(rows, filename, include_scale=False):
    columns = 'lrlrrrr' if include_scale else 'lr l rrrr'
    head = 'Geometry & $\\ell$ & Mode & $W_{12}$ & $G_0$ & $H_{\\rm mv}-H_{\\rm ind}$ & Risk difference [95\\% CI] \\\\' if include_scale else 'Geometry & $\\rho$ & Mode & $G_0$ & $H_{\\rm mv}$ & $H_{\\rm ind}$ & Risk difference [95\\% CI] \\\\'
    lines = ['\\begin{tabular}{' + columns + '}', '\\toprule', head, '\\midrule']
    for row in rows:
        s = row['scenario']
        mode = 'C' if s['fit_mode'] == 'correlation_only' else 'F'
        geometry = {'rich': 'Within-region', 'moderate': 'Adjacent', 'poor': 'Remote'}[s['geometry']] if include_scale else s['geometry'].capitalize()
        common = f"{geometry} & {(s['lengthscale'] if include_scale else s.get('rho', 0)):.2f} & {mode} & "
        if include_scale:
            common += f"{row['mass']['W12']:.2f} & {number(row['oracle_gain']['mean'], 1000)} & "
            common += number(row['joint_penalty_draw']['mean'] - row['independent_penalty_draw']['mean'], 1000)
        else:
            common += ' & '.join((number(row[k]['mean'], 1000) for k in ('oracle_gain', 'joint_penalty_draw', 'independent_penalty_draw')))
        lines.append(common + ' & ' + ci(row['risk_difference_draw'], 1000) + ' \\\\')
    lines.extend(['\\bottomrule', '\\end{tabular}'])
    (FIGS / filename).write_text('\n'.join(lines) + '\n')

def error_and_decision_plots(summary):
    """Protocol E5 views; use saved estimates and their existing intervals."""
    scenarios = [row['scenario'] for row in summary['scenarios']]
    keys = [s['id'] for s in scenarios]
    labels = [f"{s['geometry'][:3]} {s['rho']:.1f} " + ('C' if s['fit_mode'] == 'correlation_only' else 'F') for s in scenarios]
    rows = {(r['scenario_id'], r['estimator']): r for r in summary['penalty_calibration'] if r['model'] == 'joint' and r['stratum'] == 'all'}
    styles = [('expected', '#245982', 'o'), ('observed', '#bd7430', 's'), ('bootstrap', '#477f60', '^')]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.5), sharex=True, layout='constrained')
    for offset, (method, color, marker) in enumerate(styles):
        x = np.arange(len(keys)) + (offset - 1) * 0.17
        for ax, field in zip(axes[:2], ('absolute_draw_error', 'signed_error')):
            values = np.array([1000 * rows[k, method][field]['mean'] for k in keys])
            intervals = np.array([rows[k, method][field]['ci95'] for k in keys]) * 1000
            ax.vlines(x, intervals[:, 0], intervals[:, 1], color=color, lw=0.7)
            ax.scatter(x, values, color=color, marker=marker, s=14, label=method.capitalize())
        values = np.array([rows[k, method]['relative_error_of_means'] if rows[k, method]['relative_error_of_means'] is not None else np.nan for k in keys])
        axes[2].scatter(x, values, color=color, marker=marker, s=14)
    axes[0].set(yscale='log', ylabel='Mean absolute draw error\n(×1000)')
    axes[1].set_yscale('symlog', linthresh=0.1)
    axes[1].set_ylabel('Mean signed error\n(×1000)')
    axes[2].set_yscale('symlog', linthresh=1)
    axes[2].set_ylabel('Relative error of means')
    absolute_intervals = np.array([rows[k, m]['absolute_draw_error']['ci95'] for k in keys for m, _, _ in styles]) * 1000
    signed_intervals = np.array([rows[k, m]['signed_error']['ci95'] for k in keys for m, _, _ in styles]) * 1000
    relative_values = [rows[k, m]['relative_error_of_means'] for k in keys for m, _, _ in styles if rows[k, m]['relative_error_of_means'] is not None]
    axes[0].set_ylim(absolute_intervals[absolute_intervals > 0].min() / 1.5, absolute_intervals.max() * 1.5)
    axes[1].set_ylim(min(0, signed_intervals.min()) * 1.5, max(0.1, signed_intervals.max()) * 1.5)
    axes[2].set_ylim(min(-0.1, min(relative_values)) * 1.5, max(0.1, max(relative_values)) * 1.5)
    for ax in axes[1:]:
        ax.axhline(0, color='.4', lw=0.7)
    for ax in axes:
        ax.grid(axis='y', alpha=0.15)
    axes[0].legend(ncol=3, fontsize=8)
    axes[2].set_xticks(range(len(keys)), labels, rotation=40, ha='right', fontsize=8)
    fig.savefig(FIGS / 'e7_penalty_errors.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True, layout='constrained')
    methods = ('validate', 'expected', 'observed', 'bootstrap')
    for ax, mode in zip(axes, ('correlation_only', 'full')):
        rows = {r['method']: r for r in summary['pooled_decisions'] if r['fit_mode'] == mode and r['tier'] == 'bootstrap_outer' and (r['epsilon'] == 0.02)}
        for shift, field, label, color in [(-0.16, 'false_positive', 'False positive', '#bd7430'), (0.16, 'false_negative', 'False negative', '#245982')]:
            x = np.arange(4) + shift
            values = [100 * rows[m][field]['rate'] for m in methods]
            limits = np.array([rows[m][field]['ci95'] for m in methods]) * 100
            ax.bar(x, values, width=0.3, color=color, alpha=0.8, label=label)
            ax.vlines(x, limits[:, 0], limits[:, 1], color='black', lw=0.8)
        ax.set(xticks=range(4), xticklabels=[m.capitalize() for m in methods], ylim=(0, 100), title='Known nuisance' if mode == 'correlation_only' else 'Full fitting')
        ax.tick_params(axis='x', labelsize=8, rotation=25)
        ax.grid(axis='y', alpha=0.15)
    axes[0].set_ylabel('Final decision error rate (%)')
    axes[1].legend(fontsize=8)
    fig.savefig(FIGS / 'e7_decision_errors.pdf')
    plt.close(fig)

def main():
    summary_paths = [ROOT / f'results/{s}_v1/summary.json' for s in ('E7', 'E4C')]
    e7, e4 = [json.loads(p.read_text()) for p in summary_paths]
    risk_table(e7['scenarios'], 'table_e7_accounting.tex')
    risk_table(e4['scenarios'], 'table_e4c_all_scales.tex', include_scale=True)
    matched = [s for s in e4['scenarios'] if s['scenario']['lengthscale'] == 0.12]
    risk_table(matched, 'table_e4c_matched.tex', include_scale=True)
    lines = ['\\begin{longtable}{llrrrr}', '\\caption{{E7 joint-penalty calibration. Penalties and errors are multiplied by $10^3$. The realised comparison uses exactly the datasets on which the estimate is available. Information rows use up to 500 independent outer datasets per scenario; bootstrap rows use the first 30, each with 200 refit draws. Penalty is squared deviation from the joint oracle predictor, averaged over the target grid and outer datasets; signed error is estimated minus realised penalty. Paired outer-dataset resampling supplies uncertainty. Unavailable information is counted and omitted only from its paired calibration comparison; no fit fails. C fits dependence at fixed nuisance, while F also estimates scale and signal/noise variances; zero means are fixed.}}\\label{tab:e7_calibration}\\\\', '\\toprule Scenario & Estimator & Available & Estimate & Matched realised & Signed error [95\\% CI] \\\\', '\\midrule\\endfirsthead', '\\toprule Scenario & Estimator & Available & Estimate & Matched realised & Signed error [95\\% CI] \\\\', '\\midrule\\endhead']
    lookup = {s['scenario']['id']: s['scenario'] for s in e7['scenarios']}
    for row in e7['penalty_calibration']:
        if row['model'] != 'joint' or row['stratum'] != 'all':
            continue
        s = lookup[row['scenario_id']]
        geometry = 'Remote' if s['geometry'] == 'separated' else s['geometry'][:3].capitalize()
        label = f"{geometry}, {s['rho']:.1f}, {('C' if s['fit_mode'] == 'correlation_only' else 'F')}"
        lines.append(f"{label} & {row['estimator'].capitalize()} & {row['available']}/{row['eligible']} & {number(row['estimated']['mean'], 1000)} & {number(row['matched_actual']['mean'], 1000)} & " + ci(row['signed_error'], 1000) + ' \\\\')
    lines.extend(['\\bottomrule', '\\end{longtable}'])
    (FIGS / 'table_e7_calibration.tex').write_text('\n'.join(lines) + '\n')
    lines = ['\\begin{tabular}{llrrrr}', '\\toprule', 'Mode/subset & Selection & Reduction (\\%) [95\\% CI] & FPR (\\%) & FNR (\\%) & Fallback \\\\', '\\midrule']
    for row in e7['pooled_decisions']:
        if row['epsilon'] != 0.02 or row['method'] not in ('validate', 'expected', 'observed', 'bootstrap'):
            continue
        label = ('C' if row['fit_mode'] == 'correlation_only' else 'F') + ('/500' if row['tier'] == 'all' else '/30')
        lo, hi = row['relative_mean_risk_reduction_percent_ci95']
        lines.append(f"{label} & {row['method'].capitalize()} & {row['relative_mean_risk_reduction_percent']:.2f} [{lo:.2f}, {hi:.2f}] & {100 * row['false_positive']['rate']:.2f} & {100 * row['false_negative']['rate']:.2f} & {row['fallback_count']}/{row['datasets']} \\\\")
    lines.extend(['\\bottomrule', '\\end{tabular}'])
    (FIGS / 'table_e7_decisions.tex').write_text('\n'.join(lines) + '\n')
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), layout='constrained')
    for ax, estimator in zip(axes, ('expected', 'observed', 'bootstrap')):
        for mode, marker, color in (('correlation_only', 'o', '#2563a6'), ('full', 's', '#b45b20')):
            rows = [r for r in e7['penalty_calibration'] if r['model'] == 'joint' and r['stratum'] == 'all' and (r['estimator'] == estimator) and (lookup[r['scenario_id']]['fit_mode'] == mode)]
            x = [r['matched_actual']['mean'] for r in rows]
            y = [r['estimated']['mean'] for r in rows]
            ax.scatter(x, y, marker=marker, color=color, label='Known nuisance' if mode == 'correlation_only' else 'Full fit', s=24)
        ax.plot([1e-07, 0.03], [1e-07, 0.03], color='.5', lw=0.8)
        ax.set(xscale='log', yscale='log', xlim=(1e-07, 0.03), ylim=(1e-07, 0.03), title=estimator.capitalize(), xlabel='Matched realised mean penalty')
        ax.grid(alpha=0.15)
    axes[0].set_ylabel('Mean estimated penalty')
    axes[0].legend(fontsize=7, loc='upper left')
    fig.savefig(FIGS / 'e7_penalty_calibration.pdf')
    plt.close(fig)
if __name__ == '__main__':
    main()
