"""Report aqs consolidated for the multivariate kriging experiments."""
from experiments.common import ROOT as PACKAGE_ROOT
import json
ROOT = PACKAGE_ROOT
COLOUR = ''
STATES = {6: 'CA', 8: 'CO', 48: 'TX'}
POLLUTANTS = {88101: 'PM$_{2.5}$', 44201: 'O$_3$', 42602: 'NO$_2$'}
BASE_METHODS = ('mean', 'nearest', 'idw5', 'linear_trend', 'independent', 'shared', 'joint')
MIXTURE_METHODS = ('mixture_independent', 'lmc')
METHODS = BASE_METHODS + MIXTURE_METHODS
GP_METHODS = ('independent', 'shared', 'joint') + MIXTURE_METHODS

def baseline_lines(rows):
    lines = [COLOUR, '\\begin{tabular}{@{}lllrrrrrrrrr@{}}', '\\toprule', 'State & Task & Pollutant & Mean & Nearest & IDW & Trend & \\texttt{SEP} & \\texttt{DIAG} & \\texttt{JOINT} & \\texttt{MIX} & LMC \\\\', '\\midrule']
    for state, abbreviation in STATES.items():
        for task in 'ABC':
            for parameter, label in POLLUTANTS.items():
                values = [rows[state, task, parameter, method]['rmse'] for method in METHODS]
                digits = 5 if parameter == 44201 else 3
                lines.append(f'{abbreviation} & {task} & {label} & ' + ' & '.join((f'{value:.{digits}f}' for value in values)) + ' \\\\')
            if task != 'C':
                lines.append('\\addlinespace[2pt]')
        if state != 48:
            lines.append('\\midrule')
    return lines + ['\\bottomrule', '\\end{tabular}']

def calibration_lines(rows):
    lines = ['\\begin{table}[htbp]\\centering\\scriptsize\\setlength{\\tabcolsep}{1pt}', '\\caption{{AQS response calibration for separable and two-component families: primary instruments, zero buffer. Cells give 95\\% response-interval coverage (\\%) / mean width / MLPD in each pollutant\\textquotesingle s physical units. Scores average ten out-of-fold predictions per site, then sites. All transforms and covariance parameters are fitted within training folds; these are point summaries. \\texttt{MIX} is the matching independent two-kernel mixture for LMC. CA: California; CO: Colorado; TX: Texas.}}', '\\label{tab:aqs_calibration_all}', COLOUR, '\\begin{tabular}{@{}lllrrrrr@{}}', '\\toprule', 'State & Task & Pollutant & \\texttt{SEP} & \\texttt{DIAG} & \\texttt{JOINT} & \\texttt{MIX} & LMC \\\\', '\\midrule']
    for state, abbreviation in STATES.items():
        for task in 'ABC':
            for parameter, label in POLLUTANTS.items():
                if parameter == 44201:
                    label = 'Ozone'
                values = [rows[state, task, parameter, method] for method in GP_METHODS]
                cells = [f"{100 * row['coverage95']:.1f}/{row['width95']:.3g}/{row['mlpd']:.2f}" for row in values]
                lines.append(f'{abbreviation} & {task} & {label} & ' + ' & '.join(cells) + ' \\\\')
            if task != 'C':
                lines.append('\\addlinespace[2pt]')
        if state != 48:
            lines.append('\\midrule')
    return lines + ['\\bottomrule', '\\end{tabular}', '\\end{table}']

def main():
    summary = json.loads((ROOT / 'results/AQS_task_v1/summary.json').read_text())
    selected = [row for row in summary['results'] if row['variant'] == 'primary' and row['buffer_multiplier'] == 0 and (row['method'] in BASE_METHODS)]
    rows = {(row['state'], row['task'], row['parameter'], row['method']): row for row in selected}
    lmc = json.loads((ROOT / 'results/AQS_LMC_v1/summary.json').read_text())
    for row in lmc['results']:
        prefix = (row['state'], row['task'], row['parameter'])
        if row['method'] in MIXTURE_METHODS:
            key = (*prefix, row['method'])
            rows[key] = row
    outputs = {'figures/table_aqs_baselines_all.tex': baseline_lines(rows), 'tables/experiment_aqs_calibration_all.tex': calibration_lines(rows)}
    header = ''
    for name, lines in outputs.items():
        (ROOT / name).write_text(header + '\n'.join(lines) + '\n')
    print('Wrote AQS baseline and calibration tables including LMC and its independent mixture (27 rows each).')
if __name__ == '__main__':
    main()
