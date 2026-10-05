"""Aqs study for the multivariate kriging experiments."""
import gzip
import os
from time import perf_counter
import numpy as np
import pandas as pd
from experiments.common import ROOT, canonical, clean_json
from experiments.exact_gp import FitConfig
from experiments.screening import fit_marginals, fit_pair
from experiments.aqs_models import fit_shared, point_baselines
from experiments.aqs_folds import PARAMETERS
MODELS = ('mean', 'nearest', 'idw5', 'linear_trend', 'independent', 'shared', 'joint')

def run_fold(task):
    directory, manifest, index, fold, records, variant = task
    phase = manifest['phase']
    path = ROOT / directory / phase / variant / f'fold_{index:04d}.json.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    started = perf_counter()
    data = pd.DataFrame(records)
    train = data.loc[fold['train_indices']]
    test = data.loc[fold['test_indices']]
    column = 'response' if variant == 'primary' else 'instrument_mean_sensitivity'
    xs = [train.loc[train['Parameter Code'] == p, ['x_model', 'y_model']].to_numpy() for p in PARAMETERS]
    ys = [train.loc[train['Parameter Code'] == p, column].to_numpy() for p in PARAMETERS]
    centre = np.array([y.mean() for y in ys])
    scale = np.array([max(y.std(ddof=1), 1e-08) for y in ys])
    z = [(y - c) / s for y, c, s in zip(ys, centre, scale)]
    config = FitConfig(**manifest['fit_config'])
    marginals = fit_marginals(xs, z, config)
    joint = fit_pair(xs, z, marginals, config)
    shared = fit_shared(xs, z, marginals, config)
    prediction = np.full((len(MODELS), 3, len(test)), np.nan)
    details = {}
    distance = []
    for target, p in enumerate(PARAMETERS):
        mask = (test['Parameter Code'] == p).to_numpy()
        evaluation = test.loc[mask, ['x_model', 'y_model']].to_numpy()
        if not len(evaluation):
            continue
        points, metadata = point_baselines(xs[target], ys[target], evaluation)
        details[str(p)] = metadata
        for name, values in points.items():
            prediction[MODELS.index(name), 0, mask] = values
        for name, fit, j in [('independent', marginals[target], 0), ('shared', shared, target), ('joint', joint, target)]:
            if not fit.success and name != 'independent':
                fit, j = (marginals[target], 0)
            mean, latent, observed = fit.predict(evaluation, j)
            prediction[MODELS.index(name)][:, mask] = np.array([mean * scale[target] + centre[target], latent * scale[target] ** 2, observed * scale[target] ** 2])
        from scipy.spatial.distance import cdist
        own = cdist(evaluation, xs[target]).min(axis=1)
        aux = cdist(evaluation, np.vstack([x for j, x in enumerate(xs) if j != target])).min(axis=1)
        distance.extend(({'analysis_row': int(row), 'target_nearest_model_distance': float(a), 'auxiliary_nearest_model_distance': float(b)} for row, a, b in zip(test.index[mask], own, aux)))
    record = {'experiment_id': 'AQS_task_reanalysis', 'phase': phase, 'variant': variant, 'fold_index': index, 'fold': fold, 'training_centres': centre.tolist(), 'training_scales': scale.tolist(), 'marginals': [f.record() for f in marginals], 'joint': joint.record(), 'shared': shared.record(), 'baseline_details': details, 'test_distances': distance, 'research_wall_seconds': perf_counter() - started}
    arrays = path.with_suffix('').with_suffix('.npz')
    temporary = arrays.with_name(arrays.name + f'.tmp.{os.getpid()}')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, train_indices=np.array(fold['train_indices']), test_indices=np.array(fold['test_indices']), train_x=train[['x_model', 'y_model']].to_numpy(), train_y=train[column].to_numpy(), train_parameters=train['Parameter Code'].to_numpy(), test_x=test[['x_model', 'y_model']].to_numpy(), test_y=test[column].to_numpy(), test_parameters=test['Parameter Code'].to_numpy(), method_names=np.array(MODELS), predictions=prediction)
    temporary.replace(arrays)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with gzip.open(temporary, 'wt') as stream:
        stream.write(canonical(clean_json(record)))
    temporary.replace(path)
