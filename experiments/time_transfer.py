"""Time transfer for the multivariate kriging experiments."""
from experiments.common import ROOT as PACKAGE_ROOT
import argparse
import gzip
import json
ROOT = PACKAGE_ROOT
from experiments.pipeline import execute
from experiments.transfer_study import RUN, METHOD, STUDIES, branches, fit_config, load_data, scenarios, write_record

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--studies', nargs='+', choices=STUDIES, default=STUDIES)
    args = parser.parse_args()
    completed = 0
    for study in args.studies:
        for si, s in enumerate(scenarios(study)):
            for rep in range(2):
                path = RUN / 'evaluation' / study / s['id'] / f'dataset_{rep:04d}.json.gz'
                out = RUN / 'timing' / study / s['id'] / f'dataset_{rep:04d}.json.gz'
                if out.exists():
                    continue
                data = load_data(study, s, 'evaluation', rep)
                records = {}
                for bi, branch in enumerate(branches(study)):
                    order = ('screen', 'direct') if (si + rep + bi) % 2 == 0 else ('direct', 'screen')
                    records[branch] = dict(order=order)
                    for mode in order:
                        row, predictions = execute(data['xs'], data['ys'], data['evaluation'], data['fold_seed'], config=fit_config(study, branch), noise=data['noise'], linear=branch == 'linear', membership=data['membership'], future_noise=study == 'q1', mode=mode)
                        records[branch][mode] = dict(runtime_seconds=row['runtime_seconds'], counts=row['counts'], selected_model=row['selected_model'])
                write_record(out, dict(study=study, scenario=s, replication=rep, branches=records))
                completed += 1
                print(study, s['id'], rep, 'complete', flush=True)
    print('New timing pairs:', completed, flush=True)
if __name__ == '__main__':
    main()
