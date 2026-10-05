"""Report aqs matched for the multivariate kriging experiments."""
from experiments.common import ROOT as PACKAGE_ROOT
import json
ROOT = PACKAGE_ROOT
primary_path = ROOT / 'results/AQS_task_v1/summary.json'
rich_path = ROOT / 'results/AQS_LMC_v1/summary.json'
geometry_path = ROOT / 'results/application_geometry_v1/summary.json'
geometry = {(r['state'], r['task']): r for r in json.loads(geometry_path.read_text())['aqs']['state_task']}
primary = json.loads(primary_path.read_text())['output_weighted']
rich = json.loads(rich_path.read_text())['output_weighted']
primary = {(r['state'], r['task']): r for r in primary if r['method'] == 'joint' and r['variant'] == 'primary' and (r['buffer_multiplier'] == 0)}
rich = {(r['state'], r['task']): r for r in rich if r['method'] == 'lmc' and r['reference'] == 'mixture_independent'}

def value(row):
    point = row['equal_output_risk_reduction_percent']
    low, high = row['equal_output_risk_reduction_ci95']
    return f'{point:.2f} [{low:.2f}, {high:.2f}]'
lines = ['', '\\begin{tabular}{@{}llrr@{}}', '\\toprule', 'State & Task & Separable & LMC \\\\', '\\midrule']
distance_lines = ['', '\\begin{tabular}{@{}lrrr@{}}', '\\toprule', 'State & Task A & Task B & Task C \\\\', '\\midrule']
rows = []
for state, label in ((6, 'CA'), (8, 'CO'), (48, 'TX')):
    distance_pairs = []
    for task in 'ABC':
        a, b = (primary[state, task], rich[state, task])
        g = geometry[state, task]
        distance_pairs.append(f"({g['target_distance_km']:.1f}, {g['auxiliary_distance_km']:.1f})")
        lines.append(f'{label} & {task} & {value(a)} & {value(b)} \\\\')
        rows.append(dict(state=state, task=task, geometry=g, separable=a, lmc=b))
    distance_lines.append(' & '.join([label, *distance_pairs]) + ' \\\\')
lines.extend(['\\bottomrule', '\\end{tabular}'])
distance_lines.extend(['\\bottomrule', '\\end{tabular}'])
output = ROOT / 'tables/aqs_matched_comparison.tex'
output.write_text('\n'.join(lines) + '\n')
distance_output = ROOT / 'tables/aqs_distance_summary.tex'
distance_output.write_text('\n'.join(distance_lines) + '\n')
