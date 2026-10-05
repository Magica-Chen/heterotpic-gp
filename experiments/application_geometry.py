"""Descriptive geometry for the queueing and monitoring applications."""
import gzip
import json
import numpy as np
import pandas as pd
from experiments.common import ROOT
from experiments.analyze_queue import decision


def nearest(points, design):
    return np.sqrt(np.sum((points[:, None, :] - design[None, :, :]) ** 2, axis=2)).min(axis=1)


def distribution(values):
    a = np.asarray(values)
    return dict(mean=float(a.mean()), median=float(np.median(a)), q25=float(np.quantile(a, .25)),
                q75=float(np.quantile(a, .75)), minimum=float(a.min()), maximum=float(a.max()))


def main():
    selection = json.loads((ROOT / 'configuration/Q2_calibrated_v3/selection.json').read_text())
    branch = selection['primary_mean']
    threshold = selection['chosen'][branch]['threshold']
    queue = []
    for directory in sorted((ROOT / 'runs/Q2_calibrated_v3/evaluation').iterdir()):
        scores, chosen, excluded = [], [], []
        for path in sorted(directory.glob('*.json.gz')):
            with gzip.open(path, 'rt') as stream:
                row = json.load(stream)
            local = row['branches'][branch]
            outcome = decision(local, 'calibrated', threshold)
            scores.append(local['scores']['opportunity'])
            chosen.append(outcome['model'])
            excluded.append(outcome['full_excluded'])
        queue.append(dict(scenario=row['scenario'], datasets=len(chosen),
                          conditional_opportunity=distribution(scores),
                          joint_selected_percent=100 * chosen.count('joint') / len(chosen),
                          exclusion_percent=100 * float(np.mean(excluded))))
    data = pd.read_csv(ROOT / 'data/aqs/analysis.csv')
    folds = json.loads((ROOT / 'data/aqs/folds.json').read_text())['folds']
    distances = []
    for fold in folds:
        if fold['buffer_multiplier'] != 0:
            continue
        train, test = data.iloc[fold['train_indices']], data.iloc[fold['test_indices']]
        for parameter, target in test.groupby('Parameter Code'):
            own = train[train['Parameter Code'] == parameter][['x_km', 'y_km']].to_numpy()
            auxiliary = train[train['Parameter Code'] != parameter][['x_km', 'y_km']].to_numpy()
            x = target[['x_km', 'y_km']].to_numpy()
            dt, da = nearest(x, own), nearest(x, auxiliary)
            distances.extend(dict(state=fold['state'], task=fold['task'], parameter=int(parameter),
                                  site_id=site, target_distance_km=float(t), auxiliary_distance_km=float(a))
                             for site, t, a in zip(target.site_id, dt, da))
    columns = ['target_distance_km', 'auxiliary_distance_km']
    site = pd.DataFrame(distances).groupby(['state', 'task', 'parameter', 'site_id'])[columns].mean()
    pollutant = site.groupby(['state', 'task', 'parameter'])[columns].mean()
    aggregate = pollutant.groupby(['state', 'task'])[columns].mean().reset_index()
    result = dict(tqa=dict(primary_mean=branch, threshold=threshold, rows=queue),
                  aqs=dict(state_task=aggregate.to_dict('records'), pollutant=pollutant.reset_index().to_dict('records')))
    destination = ROOT / 'results/application_geometry_v1/summary.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
