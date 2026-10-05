"""Generate manuscript tables, figures and CSV files from experiment summaries."""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

PACKAGE = Path(__file__).resolve().parent


def leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, (*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, (*path, str(index)))
    else:
        yield ('/'.join(path), value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--output', type=Path, help='Create figures and tables from the supplied compact results.')
    source.add_argument('--runs', type=Path, help='Use the results of a complete run_experiments.py --study all run.')
    args = parser.parse_args()
    root = (args.runs or args.output or PACKAGE / 'paper_outputs').resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.runs is None:
        for source_name, target_name in [('reference_results', 'results'), ('configuration', 'configuration'), ('data', 'data')]:
            shutil.copytree(PACKAGE / source_name, root / target_name, dirs_exist_ok=True)
    for name in ('figures', 'tables', 'csv'):
        (root / name).mkdir(exist_ok=True)
    os.environ['GP_OUTPUT_DIR'] = str(root)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    sys.dont_write_bytecode = True
    from experiments.geometry_experiments import main as geometry
    geometry()
    if args.runs is not None:
        from experiments.application_geometry import main as application_geometry
        application_geometry()
    reports = ['report_penalties', 'report_e5_extension', 'report_robustness', 'report_opportunity',
               'report_task_screen', 'report_transfer', 'report_queue', 'report_aqs_matched',
               'report_aqs_consolidated', 'report_matched_risk']
    for name in reports:
        report = importlib.import_module('experiments.' + name)
        if callable(getattr(report, 'main', None)):
            report.main()
        print(name.replace('report_', '') + ': tables generated', flush=True)
    from experiments.report_geometry_figure import geometry_figure
    geometry_figure()
    for path in sorted((root / 'results').rglob('*.json')):
        target = root / 'csv' / path.relative_to(root / 'results').with_suffix('.csv')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['statistic', 'value'])
            writer.writerows(leaves(json.loads(path.read_text())))
    tables = sorted(str(p.relative_to(root)) for folder in ('figures', 'tables') for p in (root / folder).glob('*.tex'))
    (root / 'TABLES.txt').write_text('\n'.join(tables) + '\n')
    print(f'Figures: {root / "figures"}\nTables: {root / "tables"}\nNumerical CSV files: {root / "csv"}')


if __name__ == '__main__':
    main()
