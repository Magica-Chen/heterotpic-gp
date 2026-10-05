"""Aqs lmc study for the multivariate kriging experiments."""
from dataclasses import replace
import gzip
import json
import os
from time import perf_counter
import numpy as np
import pandas as pd
from experiments.common import ROOT, canonical, clean_json
from experiments.aqs_study import MODELS as PRIMARY_MODELS
from experiments.aqs_folds import PARAMETERS
from experiments.aqs_lmc import fit_lmc
from experiments.rich_gp import fit_rich
from experiments.exact_gp import FitConfig
MODELS = ('independent', 'mixture_independent', 'separable', 'lmc')

def run_fold(task):
    directory, manifest, index, fold, records = task
    phase = manifest['phase']
    path = ROOT / directory / phase / f'fold_{index:04d}.json.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    started = perf_counter()
    data = pd.DataFrame(records)
    train = data.loc[fold['train_indices']]
    test = data.loc[fold['test_indices']]
    xs = [train.loc[train['Parameter Code'] == p, ['x_model', 'y_model']].to_numpy() for p in PARAMETERS]
    ys = [train.loc[train['Parameter Code'] == p, 'response'].to_numpy() for p in PARAMETERS]
    c = np.array([v.mean() for v in ys])
    s = np.array([max(v.std(ddof=1), 1e-08) for v in ys])
    z = [(v - a) / b for v, a, b in zip(ys, c, s)]
    primary_path = ROOT / 'runs/AQS_task_v1/evaluation/primary' / f'fold_{index:04d}.json.gz'
    ap = primary_path.with_suffix('').with_suffix('.npz')
    if not ap.exists():
        from experiments.aqs_study import run_fold as fit_primary
        settings = json.loads((ROOT / 'configuration/AQS_task_v1/method.json').read_text())
        fit_primary((str(ROOT / 'runs/AQS_task_v1'), {**settings, 'phase': 'evaluation'}, index, fold, records, 'primary'))
    with np.load(ap) as arrays:
        raw = {key: arrays[key] for key in arrays.files}
    config = FitConfig(**manifest['fit_config'])
    marginal_config = replace(config, signal_variance_bounds=tuple(np.array(config.signal_variance_bounds) / 2))
    mixture = [fit_rich([x], [y], 'mixture', marginal_config) for x, y in zip(xs, z)]
    lmc = fit_lmc(xs, z, config)
    prediction = np.zeros((4, 3, len(test)))
    prediction[0] = raw['predictions'][PRIMARY_MODELS.index('independent')]
    prediction[2] = raw['predictions'][PRIMARY_MODELS.index('joint')]
    for target, p in enumerate(PARAMETERS):
        mask = (test['Parameter Code'] == p).to_numpy()
        evaluation = test.loc[mask, ['x_model', 'y_model']].to_numpy()
        if not len(evaluation):
            continue
        if mixture[target].success:
            mean, latent, observed = mixture[target].predict(evaluation)
            prediction[1][:, mask] = np.array([mean * s[target] + c[target], latent * s[target] ** 2, observed * s[target] ** 2])
        else:
            prediction[1][:, mask] = prediction[0][:, mask]
        if lmc.success:
            mean, latent, observed = lmc.predict(evaluation, target)
            prediction[3][:, mask] = np.array([mean * s[target] + c[target], latent * s[target] ** 2, observed * s[target] ** 2])
        else:
            prediction[3][:, mask] = prediction[1][:, mask]
    record = {'experiment_id': 'AQS_LMC_sensitivity', 'phase': phase, 'fold_index': index, 'fold': fold, 'training_centres': c.tolist(), 'training_scales': s.tolist(), 'mixture_marginals': [f.record() for f in mixture], 'lmc': lmc.record(), 'research_wall_seconds': perf_counter() - started}
    raw.update(predictions=prediction, method_names=np.array(MODELS))
    arrays = path.with_suffix('').with_suffix('.npz')
    temporary = arrays.with_name(arrays.name + f'.tmp.{os.getpid()}')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **raw)
    temporary.replace(arrays)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with gzip.open(temporary, 'wt') as stream:
        stream.write(canonical(clean_json(record)))
    temporary.replace(path)
