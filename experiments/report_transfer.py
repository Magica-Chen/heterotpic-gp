"""Report transfer for the multivariate kriging experiments."""
from experiments.common import ROOT as PACKAGE_ROOT
import json
import numpy as np
ROOT = PACKAGE_ROOT
RESULTS = ROOT / 'results/section7_alignment_v1'
TEX = ROOT / 'tables'
FAMILY = dict(lmc_positive='Positive LMC', lmc_cancelling='Cancelling LMC', ar_fidelity='AR fidelity', trend_confounded='Linear trends', heteroscedastic='Varying noise', heavy_tailed='Student-$t$')
GEOMETRY = dict(colocated='Isotopic', infill='Infill', partial='Partial coverage', remote='Remote', isotopic='Isotopic', interleaved='Interleaved', separated='Remote', distributed='Within-region', half='Half-region', lower_half='Half-region', outside='Remote')

def load(study):
    return json.loads((RESULTS / f'evaluation_{study}.json').read_text())

def ci(value, digits=2):
    a, b = value['ci95']
    return f"{value['estimate']:.{digits}f} [{a:.{digits}f}, {b:.{digits}f}]"

def risk(group, model):
    return ci(group['methods'][model]['risk_reduction_percent'])

def table(name, headers, rows):
    text = ''
    text += '\\begin{tabular}{@{}l' + 'r' * (len(headers) - 1) + '@{}}\n\\toprule\n'
    text += ' & '.join(headers) + ' \\\\\n\\midrule\n'
    text += ''.join((' & '.join(map(str, row)) + ' \\\\\n' for row in rows))
    text += '\\bottomrule\n\\end{tabular}\n'
    (TEX / (name + '.tex')).write_text(text)

def grouped_tables(study, branch, key, labels, prefix):
    groups = load(study)['branches'][branch]
    selected = [(label, groups[f'{key}={value}']) for value, label in labels.items()]
    table(prefix + '_risk', [key.title(), 'DIAG', 'JOINT', 'Screen'], [[label, *[risk(g, m) for m in ('diagonal', 'joint', 'screen')]] for label, g in selected])
    table(prefix + '_decisions', [key.title(), 'Direct', 'Exclude (\\%)', 'False exclude (\\%)'], [[label, risk(g, 'direct'), f"{g['methods']['screen']['exclusion_percent']['estimate']:.2f}", ci(g['methods']['screen']['false_exclusion_percent']) if g['methods']['screen']['false_exclusion_percent'] else '--'] for label, g in selected])

def numerical_table():
    rows = []
    for study in ('balanced', 'robustness', 'scales', 'q1', 'q2'):
        for branch, groups in load(study)['branches'].items():
            label = {'balanced': 'III equal counts', 'robustness': 'V-A', 'scales': 'V-B', 'q1': '$M/M/1$', 'q2': 'Tandem'}[study]
            if branch != 'primary':
                label += ' / ' + branch.upper() if branch == 'ard' else ' / ' + branch
            d = groups['overall']['fit_diagnostics']
            rows.append([label, *[f'{d[k]:,}' for k in ('fits', 'failed', 'boundary', 'above_gtol')], groups['overall']['unresolved']])
    table('section7_transfer_numerics_table', ['Study / specification', 'Fits', 'Failed', 'Bound', '$g>10^{-5}$', 'Unresolved'], rows)

def detailed_cells():
    output = ['study\tbranch\tcell\tmethod\tSEP_risk\trisk\trisk_reduction_percent\tci95_low\tci95_high\texcluded_percent\tfalse_exclusion_percent']
    for study in ('balanced', 'robustness', 'scales', 'q1', 'q2'):
        for branch, groups in load(study)['branches'].items():
            for key, group in groups.items():
                if not key.startswith('cell='):
                    continue
                for method, m in group['methods'].items():
                    rr = m['risk_reduction_percent']
                    output.append('\t'.join(map(str, [study, branch, key[5:], method, group['methods']['independent']['risk']['estimate'], m['risk']['estimate'], rr['estimate'], *rr['ci95'], m['exclusion_percent']['estimate'], None if m['false_exclusion_percent'] is None else m['false_exclusion_percent']['estimate']])))
    (RESULTS / 'all_cells.tsv').write_text('\n'.join(output) + '\n')

def queue_tables():
    q1 = load('q1')['branches']
    rows, uncertainty = ([], [])
    for branch in ('centred', 'linear'):
        for geometry in ('isotopic', 'interleaved', 'separated'):
            group = q1[branch]['geometry=' + geometry]
            m = group['methods']['screen']
            label = branch.title() + ' / ' + GEOMETRY[geometry]
            rows.append([label, f"{np.sqrt(group['methods']['independent']['risk']['estimate']):.4f}", f"{np.sqrt(m['risk']['estimate']):.4f}", ci(m['risk_reduction_percent']), f"{m['exclusion_percent']['estimate']:.2f}"])
            uncertainty.append([label, f"{100 * m['coverage95']['estimate']:.2f}", f"{m['width95']['estimate']:.3f}", f"{m['mlpd']['estimate']:.3f}", f"{100 * m['latent_coverage95']['estimate']:.2f}"])
    table('section7_q1_risk', ['Mean / geometry', 'SEP RMSE', 'Screen RMSE', 'Reduction (\\%)', 'Exclude (\\%)'], rows)
    table('section7_q1_coverage', ['Mean / geometry', 'Obs. coverage', 'Width', 'MLPD', 'Latent coverage'], uncertainty)
    q2 = load('q2')['branches']
    for branch in ('linear', 'centred'):
        rows = []
        for i, (n, geometry) in enumerate(((n, g) for n in (8, 12, 20) for g in ('distributed', 'lower_half', 'outside'))):
            group = q2[branch][f'cell=Q{i + 1:02d}']
            rows.append([f'{n} / ' + GEOMETRY[geometry], risk(group, 'direct'), risk(group, 'screen'), f"{group['methods']['screen']['exclusion_percent']['estimate']:.1f}"])
        table('section7_q2_' + branch, ['$N_1$ / auxiliary region', 'Direct', 'Screen', 'Exclude (\\%)'], rows)

def scale_tables():
    report = load('scales')
    rows = []
    for branch in ('ard', 'isotropic'):
        for dim in (2, 5, 10):
            g = report['branches'][branch][f'dimension={dim}']
            rows.append([f"{(branch.upper() if branch == 'ard' else 'Isotropic')} / {dim}", *[risk(g, m) for m in ('diagonal', 'joint', 'screen')]])
    table('section7_ix_risk', ['Fit / dimension', 'DIAG', 'JOINT', 'Screen'], rows)
    rows, detail = ([], [])
    for dim in (2, 5, 10):
        group = [s for s in report['coordinate_sensitivity'] if s['scenario']['dimension'] == dim]
        changes = sum((s['pointwise_exclusion_changes'] for s in group))
        reversals = 0
        global_values, box_values = ([], [])
        for s in group:
            vals = [r['opportunity'] for r in s['scores']['scenarios']]
            global_value, box = (max(vals[:3]), max(vals))
            global_values.append(global_value)
            box_values.append(box)
            reversals += int(global_value < 0.2 <= box)
        dense = [s for s in group if 'dense' in s]
        dense_reversals = sum(((s['scores']['opportunity'] < 0.2) != (s['dense']['opportunity'] < 0.2) for s in dense))
        gaps = [s['dense']['opportunity'] - s['scores']['opportunity'] for s in dense]
        item = dict(dimension=dim, datasets=len(group), scenario_decision_changes=changes, global_to_box_retention=reversals, max_box_minus_global=float(np.max(np.array(box_values) - global_values)), dense_datasets=len(dense), dense_gate_disagreements=dense_reversals, dense_minus_box_min=min(gaps) if gaps else None, dense_minus_box_max=max(gaps) if gaps else None)
        detail.append(item)
        rows.append([dim, len(group), changes, reversals, f"{item['max_box_minus_global']:.3f}"])
    table('section7_ix_scales', ['$T$', 'Datasets', 'Point decisions vary', 'Global exclude / box retain', 'Max score increase'], rows)
    (RESULTS / 'coordinate_summary.json').write_text(json.dumps(detail, indent=2) + '\n')

def main():
    grouped_tables('balanced', 'primary', 'geometry', {g: GEOMETRY[g] for g in ('colocated', 'infill', 'partial', 'remote')}, 'section7_balanced')
    grouped_tables('robustness', 'primary', 'family', FAMILY, 'section7_robustness')
    queue_tables()
    scale_tables()
    numerical_table()
    detailed_cells()
    timing = json.loads((RESULTS / 'timing_summary.json').read_text())
    table('section7_transfer_timing_table', ['Study / specification', 'Pairs', 'Screen (s)', 'Direct (s)', 'Saving (\\%)'], [[key, value['pairs'], f"{value['screen_seconds']:.3f}", f"{value['direct_seconds']:.3f}", ci(value['saving_percent'])] for key, value in timing.items()])
if __name__ == '__main__':
    main()
