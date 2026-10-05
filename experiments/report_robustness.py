"""Report robustness for the multivariate kriging experiments."""
from experiments.common import ROOT as PACKAGE_ROOT
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = PACKAGE_ROOT
FIGS = ROOT / 'figures'
TEX = ROOT / 'tables'
FAMILIES = ['lmc_positive', 'lmc_cancelling', 'ar_fidelity', 'trend_confounded', 'heteroscedastic', 'heavy_tailed']
LABELS = ['LMC positive', 'LMC cancelling', 'AR fidelity', 'Linear trends', 'Varying noise', 'Student-t noise']
COLOUR = ''

def interval(row, own=False):
    key = 'relative_mean_risk_reduction_vs_rich_independent_percent' if own else 'relative_mean_risk_reduction_percent'
    lo, hi = row[key + '_ci95']
    return f'{row[key]:.2f} [{lo:.2f}, {hi:.2f}]'

def main():
    source = ROOT / 'results/E8_v1/summary.json'
    result = json.loads(source.read_text())
    rows = {(r['family'], r['method']): r for r in result['families'] if r['epsilon'] == 0.02}
    methods = ('always_joint', 'validate', 'screen_validate', 'rich_independent', 'rich_validate')
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), layout='constrained', gridspec_kw={'width_ratios': [1.2, 1]})
    for method, offset, colour, marker, label in [('validate', -0.2, '#28669d', 'o', 'Working validation'), ('screen_validate', 0.0, '#b65026', 's', 'Screen + validation'), ('rich_validate', 0.2, '#397840', '^', 'Rich validation')]:
        for i, family in enumerate(FAMILIES):
            if (family, method) not in rows:
                continue
            r = rows[family, method]
            mean = r['relative_mean_risk_reduction_percent']
            lo, hi = r['relative_mean_risk_reduction_percent_ci95']
            axes[0].errorbar(mean, i + offset, xerr=[[mean - lo], [hi - mean]], fmt=marker, color=colour, ms=3.5, lw=0.9, label=label if i == 0 else None)
    axes[0].set(yticks=np.arange(6), yticklabels=LABELS, xlabel='Risk reduction vs working independent (%)')
    axes[0].legend(loc='lower left', fontsize=7, frameon=False)
    for i, family in enumerate(FAMILIES[:-1]):
        r = rows[family, 'rich_validate']
        mean = r['relative_mean_risk_reduction_vs_rich_independent_percent']
        lo, hi = r['relative_mean_risk_reduction_vs_rich_independent_percent_ci95']
        axes[1].errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt='^', color='#397840', ms=4, lw=0.9)
    axes[1].text(0, 5, 'Not fitted', ha='center', va='center', color='.4', fontsize=8)
    axes[1].set(yticks=np.arange(6), yticklabels=[], xlabel='Rich validation vs rich independent (%)')
    for ax in axes:
        ax.axvline(0, color='.55', ls=':', lw=0.8)
        ax.set_ylim(5.5, -0.6)
        ax.grid(axis='x', alpha=0.15)
    figure = FIGS / 'e8_robustness.pdf'
    fig.savefig(figure)
    fig.savefig(figure.with_suffix('.png'), dpi=180)
    plt.close(fig)
    lines = [COLOUR, '\\begin{tabular}{llrr}', '\\toprule', 'Family & Procedure & Reduction (\\%) [95\\% CI] & Versus rich independent \\\\', '\\midrule']
    for family, label in zip(FAMILIES, LABELS):
        for method in methods:
            if (family, method) not in rows:
                continue
            r = rows[family, method]
            name = {'always_joint': 'MV-SEP', 'validate': 'Validate', 'screen_validate': 'Screen + validate', 'rich_independent': 'Rich independent', 'rich_validate': 'Rich validate'}[method]
            lines.append(f'{label} & {name} & ' + interval(r) + ' & ' + (interval(r, True) if method.startswith('rich') else '--') + ' \\\\')
    lines.extend(['\\bottomrule', '\\end{tabular}'])
    family_table = FIGS / 'table_e8_families.tex'
    family_table.write_text('\n'.join(lines) + '\n')
    configs = {r['scenario']['id']: r['scenario'] for r in result['scenarios']}
    detail = {(r['scenario']['id'], r['method']): r for r in result['scenarios'] if r['epsilon'] == 0.02}
    lines = [COLOUR, '\\begin{longtable}{llrrrrr}', '\\caption{{E8 full grid: risk reduction (\\%) [95\\% paired interval] versus the working independent fit, at 2\\% materiality. IDs B01--B06 are LMC positive, B07--B12 cancelling, B13--B18 AR fidelity, B19--B24 trends, B25--B30 varying noise, and B31--B36 Student-t noise. Each row has 200 datasets. Positive reduction favours selection; intervals use 2,000 paired within-configuration dataset resamples. Gaussian families use conditional latent risk and Student-t uses realised latent loss. Covariance parameters are fitted; linear means and varying noise are fitted only in the corresponding richer branches. }}\\label{tab:e8_grid}\\\\', '\\toprule ID & Geometry & $n$ & MV-SEP & Validate & Screen + validate & Rich validate \\\\', '\\midrule\\endfirsthead', '\\toprule ID & Geometry & $n$ & MV-SEP & Validate & Screen + validate & Rich validate \\\\', '\\midrule\\endhead']
    for sid, s in configs.items():
        values = [interval(detail[sid, m]) if (sid, m) in detail else '--' for m in ('always_joint', 'validate', 'screen_validate', 'rich_validate')]
        lines.append(f"{sid} & {s['geometry'][:3].capitalize()} & {s['n1']} & " + ' & '.join(values) + ' \\\\')
    lines.extend(['\\bottomrule', '\\end{longtable}'])
    grid_table = FIGS / 'table_e8_grid.tex'
    grid_table.write_text('\n'.join(lines) + '\n')
    lines = [COLOUR, '\\begin{tabular}{lrrrrr}', '\\toprule', 'Family & Excluded & Unresolved & False exclusion & Regret (\\%) & Coverage (\\%) \\\\', '\\midrule']
    for family, label in zip(FAMILIES, LABELS):
        r = rows[family, 'screen_validate']
        fe = r['false_exclusion']
        regret = r['catalogue_relative_regret_percent']
        lines.append(f"{label} & {r['gate_exclusions']}/1200 & {r['gate_unresolved']}/1200 & " + (f"{fe['numerator']}/{fe['denominator']}" if fe else '--') + ' & ' + (f'{regret:.2f}' if regret is not None else '--') + f" & {100 * r['mean_coverage95']:.1f}" + ' \\\\')
    lines.extend(['\\bottomrule', '\\end{tabular}'])
    gate_table = FIGS / 'table_e8_gate.tex'
    gate_table.write_text('\n'.join(lines) + '\n')
    lines = [COLOUR, '\\begin{tabular}{llrrrr}', '\\toprule', 'Family & Branch & Fits (s) & Scores (s) & Bound fits & Failed fits \\\\', '\\midrule']
    for r in result['costs']:
        label = LABELS[FAMILIES.index(r['family'])]
        lines.append(f"{label} & {r['branch'].capitalize()} & {r['mean_fit_seconds']:.3f} & {r['mean_diagnostic_seconds']:.3f} & {r['mean_active_bound_fits']:.2f}/15 & {r['mean_failed_fits']:.0f}/15" + ' \\\\')
    lines.extend(['\\bottomrule', '\\end{tabular}'])
    cost_table = FIGS / 'table_e8_cost.tex'
    cost_table.write_text('\n'.join(lines) + '\n')
if __name__ == '__main__':
    main()
