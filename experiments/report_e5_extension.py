"""Report e5 extension for the multivariate kriging experiments."""
import json
from experiments.common import ROOT

def estimate(value, ci, digits=2):

    def number(x):
        if 0 < abs(x) < 0.5 * 10 ** (-digits):
            mantissa, exponent = f'{x:.1e}'.split('e')
            return f'${mantissa}\\times10^{{{int(exponent)}}}$'
        return f'{x:.{digits}f}'
    return f'{number(value)} [{number(ci[0])}, {number(ci[1])}]'

def main():
    s = json.loads((ROOT / 'results/E5_extension_v1/summary.json').read_text())
    rows = {(r['family'], r['geometry'], r['method']): r for r in s['results']}
    outputs = []

    def write(path, text):
        path.write_text(text)
        outputs.append(path)
    families = {'separable': 'Separable', 'original_lmc': 'Original LMC'}
    lines = ['', '\\begin{tabular}{llrr}', '\\toprule', 'Generator & Geometry & Correlation-only risk reduction & Full-fit risk reduction \\\\', '\\midrule']
    for family in families:
        for geometry in ('isotopic', 'interleaved', 'separated'):
            values = []
            for mode in ('correlation_only', 'full'):
                r = rows[family, geometry, mode + '_joint']
                values.append(estimate(r['risk_reduction_vs_own_independent_percent'], r['risk_reduction_vs_own_independent_ci95'], 3))
            lines.append(f'{families[family]} & {geometry.capitalize()} & ' + ' & '.join(values) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write(ROOT / 'figures/table_e5_extension_risk.tex', '\n'.join(lines) + '\n')
    lines = ['', '\\begin{tabular}{lllrr}', '\\toprule', 'Generator & Geometry & Fitting & $\\Delta$RMSE & $\\Delta$MLPD \\\\', '\\midrule']
    for family in families:
        for geometry in ('isotopic', 'interleaved', 'separated'):
            for mode, label in (('correlation_only', 'Correlation'), ('full', 'Full')):
                r = rows[family, geometry, mode + '_joint']
                values = [estimate(r['paired_' + k + '_improvement'], r['paired_' + k + '_improvement_ci95'], 4) for k in ('rmse', 'mlpd')]
                lines.append(f'{families[family]} & {geometry.capitalize()} & {label} & ' + ' & '.join(values) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write(ROOT / 'figures/table_e5_extension_metrics.tex', '\n'.join(lines) + '\n')
    lines = ['', '\\begin{tabular}{llrr}', '\\toprule', 'Generator & Geometry & Original $\\Delta$RMSE [95\\% CI] & Original $\\Delta$MLPD [95\\% CI] \\\\', '\\midrule']
    for r in s['original80_intervals']:
        values = [estimate(r[k], r[k + '_ci95'], 4) for k in ('rmse_diff', 'mlpd_diff')]
        lines.append(f"{families[r['family']]} & {r['geometry'].capitalize()} & " + ' & '.join(values) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write(ROOT / 'figures/table_e5_original_intervals.tex', '\n'.join(lines) + '\n')
if __name__ == '__main__':
    main()
