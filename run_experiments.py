"""Run the experiments for the manuscript and supplementary information."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

PACKAGE = Path(__file__).resolve().parent
STUDIES = {
    'geometry': (None, 'I-A/I-B/I-C: geometry, residual channels and oracle gains'),
    'II-A': ('E4C_v1', 'Matched designs and fitted prediction risk'),
    'II-B': ('E5_extension_v1', 'Separable and LMC covariance comparisons'),
    'II-C': ('E7_v1', 'Information and bootstrap penalty estimates'),
    'III': ('E6_task_v2', 'Parameter sharing, response coupling and screening'),
    'IV-A': ('E8_v1', 'Covariance misspecification'),
    'IV-B': ('E8_opportunity_v2', 'Target scarcity and auxiliary allocation'),
    'V': ('section7_alignment_v1', 'Transfer, dimension and lengthscale sensitivity'),
    'TQA': ('Q2_calibrated_v3', 'Tandem queueing application'),
    'AQS': ('AQS_task_v1', 'Air-quality monitoring application'),
    'AQS-LMC': ('AQS_LMC_v1', 'Air-quality covariance sensitivity'),
    'original-II-B': (None, 'Original 80-dataset comparisons used in SI E'),
}


def save(path, value):
    from experiments.common import clean_json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), indent=2) + '\n')


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    for name in ('configuration', 'data'):
        for source in (PACKAGE / name).rglob('*'):
            if source.is_file():
                target = output / source.relative_to(PACKAGE)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(source, target)
    for name in ('results', 'runs', 'figures', 'tables'):
        (output / name).mkdir(exist_ok=True)


def dispatch(function, tasks, workers, label):
    started = perf_counter()
    if workers == 1:
        for i, task in enumerate(tasks, 1):
            function(task)
            if i == len(tasks) or i % 25 == 0:
                print(f'{label}: {i}/{len(tasks)} datasets ({perf_counter()-started:.1f} s)', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(function, task) for task in tasks]
            for i, future in enumerate(as_completed(pending), 1):
                future.result()
                if i == len(tasks) or i % 25 == 0:
                    print(f'{label}: {i}/{len(tasks)} datasets ({perf_counter()-started:.1f} s)', flush=True)


def module(name, arguments=()):
    subprocess.run([sys.executable, '-B', '-m', 'experiments.' + name, *map(str, arguments)],
                   cwd=PACKAGE, check=True)


def fit(study, args):
    from experiments.common import ROOT
    if study == 'geometry':
        from experiments.geometry_experiments import main
        main()
        return
    if study == 'original-II-B':
        from experiments import original_simulations as original
        destination = ROOT / 'results/original_simulations'
        destination.mkdir(parents=True, exist_ok=True)
        reps = args.replicates or 80
        for filename, frame in [('s5_kriging_revisited.csv', original.run_s5(reps=reps)),
                                ('s5_lmc_misspec.csv', original.run_s5_misspecified(reps=reps))]:
            frame.to_csv(destination / filename, index=False)
            if reps == 80:
                frame.to_csv(ROOT / 'data/original_simulations' / filename, index=False)
        return
    identifier = STUDIES[study][0]
    spec = json.loads((ROOT / 'configuration' / identifier / 'method.json').read_text())
    phase = args.phase
    directory = ROOT / 'runs' / identifier
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {**spec, 'phase': phase}

    def selected(items):
        return items[:args.limit_scenarios] if args.limit_scenarios else items

    if study in ('II-A', 'II-C'):
        from experiments.penalty_study import run_dataset
        scenarios = selected(spec['scenarios'])
        if args.replicates:
            scenarios = [{**s, 'evaluation_replicates': args.replicates} for s in scenarios]
        manifest['scenarios'] = scenarios
        function = run_dataset
        tasks = [(str(directory), manifest, s, r) for s in scenarios for r in range(s['evaluation_replicates'])]
    elif study == 'II-B':
        from experiments.e5_extension import run_dataset
        scenarios = selected(spec['scenarios'])
        repetitions = args.replicates or spec['evaluation_datasets_per_cell']
        manifest.update(scenarios=scenarios, replicates=repetitions)
        function = run_dataset
        tasks = [(manifest, s, r) for s in scenarios for r in range(repetitions)]
    elif study == 'III':
        from experiments.task_screen_study import run_dataset
        scenarios = selected(spec['scenarios'])
        repetitions = args.replicates or spec['replicates'][phase]
        manifest.update(scenarios=scenarios, replicates=repetitions)
        function = run_dataset
        tasks = [(str(directory), manifest, s, r) for s in scenarios for r in range(repetitions)]
    elif study == 'IV-A':
        from experiments.robustness import run_dataset
        scenarios = selected(spec['scenarios'])
        repetitions = args.replicates or spec['evaluation_replicates']
        manifest.update(scenarios=scenarios, replicates_per_scenario=repetitions)
        function = run_dataset
        tasks = [(str(directory), manifest, s, r) for s in scenarios for r in range(repetitions)]
    elif study == 'IV-B':
        from experiments.borrow_opportunity_study import run_one
        cases = selected(spec['cases'])
        repetitions = args.replicates or spec['replicates']
        manifest = dict(spec={**spec, 'cases': cases, 'replicates': repetitions}, phase=phase)
        function = run_one
        tasks = [('e8', phase, s, r, manifest, None) for s in cases for r in range(repetitions)]
    elif study == 'V':
        from experiments.transfer_study import run_one
        function = run_one
        tasks = [(part, s, phase, r, manifest) for part in args.transfer_studies
                 for s in selected(spec['scenarios'][part])
                 for r in range(args.replicates or spec['replicates'][part])]
    elif study == 'TQA':
        from experiments.queue_study import run_one, load_reference
        function = run_one
        reference = load_reference(phase)
        repetitions = args.replicates or spec['replicates'][phase]
        tasks = [(phase, s, r, manifest, reference, None) for s in selected(spec['cases']) for r in range(repetitions)]
    else:
        import pandas as pd
        data = pd.read_csv(ROOT / 'data/aqs/analysis.csv').to_dict('list')
        folds = json.loads((ROOT / 'data/aqs/folds.json').read_text())['folds']
        indexed = list(enumerate(folds))
        if study == 'AQS-LMC':
            indexed = [(i, f) for i, f in indexed if f['buffer_multiplier'] == 0]
            from experiments.aqs_lmc_study import run_fold
            tasks = [(str(directory), manifest, i, f, data) for i, f in selected(indexed)]
        else:
            from experiments.aqs_study import run_fold
            tasks = [(str(directory), manifest, i, f, data, variant) for i, f in selected(indexed)
                     for variant in ('primary', 'instrument_mean')]
        function = run_fold
    save(directory / f'{phase}_manifest.json', manifest)
    dispatch(function, tasks, args.workers, study)
    save(directory / f'{phase}_completion.json', dict(completed=len(tasks), attempted=len(tasks), task_exceptions=[]))
    if study == 'III' and phase == 'evaluation':
        from experiments.pair_control import run_one
        pair_dir = ROOT / 'runs/E6_task_pair_control_v1'
        tasks = [(str(directory), str(pair_dir), s, r, manifest) for s in scenarios for r in range(repetitions)]
        dispatch(run_one, tasks, args.workers, 'III two-candidate validation')
        save(pair_dir / 'completion.json', dict(completed=len(tasks), attempted=len(tasks), task_exceptions=[]))


def analyze(study, args):
    from experiments.common import ROOT
    identifier = STUDIES[study][0]
    if identifier is None:
        return
    directory = ROOT / 'runs' / identifier
    destination = ROOT / 'results' / identifier
    destination.mkdir(parents=True, exist_ok=True)
    if study in ('II-A', 'II-C'):
        module('analyze_penalties', ['--run-dir', directory, '--output-dir', destination])
    elif study == 'II-B':
        module('analyze_e5_extension')
    elif study == 'III':
        module('analyze_task_screen', ['calibrate' if args.phase == 'calibration' else 'evaluate'])
        if args.phase == 'evaluation':
            module('pair_control', ['analyze'])
    elif study == 'IV-A':
        module('analyze_robustness', ['--run-dir', directory, '--output-dir', destination])
    elif study == 'IV-B':
        module('analyze_borrow_opportunity', ['--study', 'e8'])
    elif study == 'V':
        module('analyze_transfer', ['--studies', *args.transfer_studies])
    elif study == 'TQA':
        module('analyze_queue', ['--phase', args.phase])
    elif study == 'AQS':
        module('analyze_aqs')
    elif study == 'AQS-LMC':
        module('analyze_aqs_lmc')


def timing(study, args):
    if study == 'III':
        module('time_task_screen')
        module('pair_control', ['time'])
    elif study == 'V':
        module('time_transfer', ['--studies', *args.transfer_studies])
        from experiments.summarize_timing import main
        main(args.transfer_studies)
    elif study == 'TQA':
        module('analyze_queue', ['--phase', 'timing'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List manuscript experiments.')
    parser.add_argument('--study', choices=['all', *STUDIES], default='geometry')
    parser.add_argument('--output', type=Path, default=PACKAGE / 'output')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--stage', choices=['all', 'fit', 'analyze', 'timing'], default='all')
    parser.add_argument('--phase', choices=['evaluation', 'calibration'], default='evaluation')
    parser.add_argument('--replicates', type=int, help='Use fewer replications with --stage fit.')
    parser.add_argument('--limit-scenarios', type=int, help='Run the first N scenarios or AQS folds with --stage fit.')
    parser.add_argument('--transfer-studies', nargs='+', choices=['balanced', 'robustness', 'scales', 'q1', 'q2'],
                        default=['balanced', 'robustness', 'scales', 'q1', 'q2'])
    args = parser.parse_args()
    if args.list:
        for name, (_, description) in STUDIES.items():
            print(f'{name:16} {description}')
        return
    if args.workers < 1 or any(v is not None and v < 1 for v in (args.replicates, args.limit_scenarios)):
        parser.error('Worker, scenario and replication counts must be positive.')
    if (args.replicates or args.limit_scenarios) and args.stage != 'fit':
        parser.error('Partial runs use --stage fit; paper summaries require complete designs.')
    if args.phase == 'calibration' and args.study not in ('III', 'TQA'):
        parser.error('Independent calibration is available for III and TQA.')
    if args.phase == 'calibration' and args.stage == 'timing':
        parser.error('Paired timings use evaluation data.')
    os.environ['GP_OUTPUT_DIR'] = str(args.output.resolve())
    os.environ['PYTHONPATH'] = str(PACKAGE)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    sys.dont_write_bytecode = True
    prepare(args.output.resolve())
    chosen = ['geometry', 'original-II-B', *[name for name in STUDIES if name not in ('geometry', 'original-II-B')]] if args.study == 'all' else [args.study]
    for study in chosen:
        print(f'Running {study}: {STUDIES[study][1]}', flush=True)
        if args.stage in ('all', 'fit'):
            fit(study, args)
        if args.stage in ('all', 'analyze'):
            analyze(study, args)
        if args.phase == 'evaluation' and args.stage in ('all', 'timing'):
            timing(study, args)
    print(f'Outputs: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
