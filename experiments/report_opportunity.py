"""Report opportunity for the multivariate kriging experiments."""
from experiments.common import ROOT as PACKAGE_ROOT
import json
ROOT = PACKAGE_ROOT
source = ROOT / 'results/E8_opportunity_v2/summary.json'
data = json.loads(source.read_text())
cases = data['cases']

def fitted(case, method):
    row = next((r for r in case['results'] if r['method'] == method))
    v = row['risk_reduction_percent']
    low, high = row['risk_reduction_ci95']
    return f'{v:.2f} [{low:.2f}, {high:.2f}]'
main = ['', '\\begin{tabular}{lrrrrrr}', '\\toprule', ' & & & \\multicolumn{2}{c}{Interleaved} & \\multicolumn{2}{c}{Separated} \\\\', '\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}', 'Family & $N_1$ & $N_q$ & Oracle & Fitted [95\\% interval] & Oracle & Fitted [95\\% interval] \\\\', '\\midrule']
details = ['', '\\begin{tabular}{lrrlrrr}', '\\toprule', 'Family & $N_1$ & $N_q$ & Geometry & Oracle (\\%) & Separable / own & Rich / own \\\\', '\\midrule']
for family, label in (('lmc_positive', 'LMC'), ('ar_fidelity', 'AR')):
    for n in (8, 16):
        for ratio in (1, 4):
            pair = [next((c for c in cases if c['case']['family'] == family and c['case']['n1'] == n and (c['case']['n2'] == ratio * n) and (c['case']['geometry'] == geometry))) for geometry in ('interleaved', 'separated')]
            main.append(f"{label} & {n} & {ratio * n} & {pair[0]['oracle_gain_percent']:.3f} & {fitted(pair[0], 'rich_joint')} & {pair[1]['oracle_gain_percent']:.3f} & {fitted(pair[1], 'rich_joint')} \\\\")
            for case, geometry in zip(pair, ('Interleaved', 'Remote')):
                details.append(f"{label} & {n} & {ratio * n} & {geometry} & {case['oracle_gain_percent']:.3f} & {fitted(case, 'separable')} & {fitted(case, 'rich_joint')} \\\\")
outputs = []
for name, lines in (('e8_opportunity_main_table.tex', main), ('e8_opportunity_full_table.tex', details)):
    path = ROOT / 'tables' / name
    path.write_text('\n'.join([*lines, '\\bottomrule', '\\end{tabular}']) + '\n')
    outputs.append(path)
