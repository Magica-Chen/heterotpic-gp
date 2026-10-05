"""Report queue for the multivariate kriging experiments."""
from experiments.common import ROOT as PACKAGE_ROOT
import json
ROOT = PACKAGE_ROOT
METHOD = ROOT / 'configuration/Q2_calibrated_v3'
OUT = ROOT / 'results/Q2_calibrated_v3'
TEX = ROOT / 'tables'

def cases():
    return json.loads((METHOD / 'method.json').read_text())['cases']

def number(x):
    return f'{(0.0 if abs(x) < 0.005 else x):.2f}'

def estimate(x):
    return number(x['estimate'])

def ci(x):
    return f"{number(x['estimate'])} [{number(x['ci95'][0])}, {number(x['ci95'][1])}]"

def table(name, columns, header, rows):
    text = ['', '', '\\begin{tabular}{@{}' + columns + '@{}}', '\\toprule', ' & '.join(header) + ' \\\\', '\\midrule']
    text += [' & '.join(row) + ' \\\\' for row in rows]
    text += ['\\bottomrule', '\\end{tabular}']
    (TEX / name).write_text('\n'.join(text) + '\n')

def add_candidate_panel(name, branch_results, geometry):
    lines = ['\\par\\smallskip', '\\begin{tabular}{@{}rlrrrrr@{}}', '\\toprule', '\\multicolumn{2}{l}{(b) Candidates and final selection} & \\multicolumn{2}{c}{MSE reduction vs SEP (\\%)} & \\multicolumn{3}{c}{Calibrated selection (\\%)} \\\\', '\\cmidrule(lr){3-4}\\cmidrule(l){5-7}', '$N_1$ & Auxiliary region & DIAG & JOINT & SEP & DIAG & JOINT \\\\', '\\midrule']
    for case in cases():
        group = branch_results['cell=' + case['id']]
        methods = group['methods']
        counts = methods['calibrated']['selected_counts']
        row = [str(case['n1']), geometry[case['geometry']], ci(methods['diagonal']['risk_reduction_percent']), ci(methods['joint']['risk_reduction_percent'])]
        row += [f"{100 * counts[key] / group['datasets']:.1f}" for key in ('independent', 'diagonal', 'joint')]
        lines.append(' & '.join(row) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    path = TEX / name
    first_panel = path.read_text().replace('\\toprule', '\\toprule' + '\n' + '\\multicolumn{6}{l}{(a) Prediction procedures} \\\\' + '\n', 1)
    path.write_text(first_panel + '\n'.join(lines) + '\n')

def main():
    result = json.loads((OUT / 'evaluation.json').read_text())
    selection = json.loads((METHOD / 'selection.json').read_text())
    primary = selection['primary_mean']
    geometry_report = json.loads((ROOT / 'results/application_geometry_v1/summary.json').read_text())['tqa']
    geometry_rows = {r['scenario']['id']: r for r in geometry_report['rows']}
    geometry = dict(distributed='Within-region', lower_half='Half-region', outside='Remote')
    for branch in ('centred', 'linear'):
        rows = []
        for case in cases():
            m = result['branches'][branch]['cell=' + case['id']]['methods']
            rows.append([str(case['n1']), geometry[case['geometry']], f"{m['independent']['risk']['estimate'] ** 0.5:.4f}", ci(m['direct']['risk_reduction_percent']), ci(m['transfer']['risk_reduction_percent']), ci(m['calibrated']['risk_reduction_percent'])])
        table('queue_v3_' + branch + '.tex', 'rlrrrr', ['$N_1$', 'Auxiliary region', 'SEP RMSE', 'Direct', 'Fixed $t_U=0.20$', 'Calibrated'], rows)
        add_candidate_panel('queue_v3_' + branch + '.tex', result['branches'][branch], geometry)
        if branch == primary:
            main_rows = []
            for case, row in zip(cases(), rows):
                g = geometry_rows[case['id']]
                main_rows.append([*row[:2], f"{g['conditional_opportunity']['median']:.3f}", f"{g['exclusion_percent']:.1f}", f"{g['joint_selected_percent']:.1f}", row[5]])
            table('queue_v3_main.tex', 'rlrrrr', ['$N_1$', 'Auxiliary region', '$U_{\\rm cond}$', 'Exclude (\\%)', 'JOINT (\\%)', 'MSE reduction (\\%)'], main_rows)
    rows = []
    for branch in ('centred', 'linear'):
        m = result['branches'][branch]['overall']['methods']
        for key, label in (('direct', 'Direct'), ('transfer', 'Fixed 0.20'), ('calibrated', 'Calibrated')):
            r = m[key]
            rows.append([branch.capitalize() + ' / ' + label, f"{r['risk']['estimate']:.6f}", ci(r['risk_reduction_percent']), ci(r['excess_over_direct_percent']), estimate(r['exclusion_percent']), '--' if r['false_exclusion_percent'] is None else ci(r['false_exclusion_percent'])])
    table('queue_v3_decisions.tex', 'lrrrrr', ['Mean / procedure', 'MSE', 'Reduction vs SEP (\\%)', 'Excess vs direct (pp)', 'Exclude (\\%)', 'False exclusion (\\%)'], rows)
    rows = []
    for branch in ('centred', 'linear'):
        for c in selection['all_candidates'][branch]:
            rows.append([branch.capitalize(), f"{c['threshold']:g}", 'Yes' if c['admissible'] else 'No', number(100 * c['pooled_limits']['upper_excess']), number(100 * c['pooled_limits']['upper_regret']), number(100 * max(c['cell_upper_excess'].values()))])
    table('queue_v3_calibration.tex', 'lrlrrr', ['Mean', '$t_U$', 'Eligible', 'Pooled excess', 'Exclusion regret', 'Max. cell excess'], rows)
    path = OUT / 'timing.json'
    if path.exists():
        timing = json.loads(path.read_text())
        primary_method = result['branches'][primary]['overall']['methods']['calibrated']
        gain = primary_method['risk_reduction_percent']
        direct_gain = result['branches'][primary]['overall']['methods']['direct']['risk_reduction_percent']
        difference = primary_method['excess_over_direct_percent']
        saving = timing['methods']['calibrated']['saving_vs_direct_percent']
        (TEX / 'queue_v3_main_cost.tex').write_text(f"Across all nine configurations, direct validation reduces SEP risk by {estimate(direct_gain)}\\%;\nscreening gives {estimate(gain)}\\% ({number(gain['ci95'][0])}--{number(gain['ci95'][1])}\\%)\n(Table~\\ref{{si-tab:q2_calibrated_decisions}})." + f"\nRelative to direct validation, screening changes risk by \\({estimate(difference)}\\)\npercentage points of SEP risk (95\\% interval \\({number(difference['ci95'][0])}\\) to\n\\({number(difference['ci95'][1])}\\)) and reduces execution time by {estimate(saving)}\\%\n({number(saving['ci95'][0])}--{number(saving['ci95'][1])}\\%).\n")
        table('queue_v3_timing.tex', 'lrr', ['Procedure', 'Time (s)', 'Saving vs direct (\\%)'], [[label, f"{timing['methods'][key]['seconds']['estimate']:.4f}", ci(timing['methods'][key]['saving_vs_direct_percent'])] for key, label in (('direct', 'Direct validation'), ('transfer', 'Fixed 0.20'), ('calibrated', 'Calibrated screen'))])
    print('Generated Q2 calibrated tables; primary:', primary)
if __name__ == '__main__':
    main()
