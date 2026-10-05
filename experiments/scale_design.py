"""Scale study for the multivariate kriging experiments."""
from __future__ import annotations
from functools import lru_cache
from itertools import product
import json
import numpy as np
from scipy.stats import qmc
from experiments.synthetic import GaussianDGP, seed_for

def scenarios():
    return [{'id': f'A{i + 1:02d}', 'dimension': dimension, 'n1': n, 'n2': 2 * n, 'geometry': geometry, 'anisotropy_ratio': ratio, 'rho': 0.7, 'noise_variance': 0.05, 'replicates': 100, 'lengthscale': [0.2 * np.sqrt(dimension) / ratio] + [0.2 * np.sqrt(dimension)] * (dimension - 1)} for i, (dimension, n, geometry, ratio) in enumerate(product((2, 5, 10), (20, 50), ('distributed', 'half', 'separated'), (1, 5)))]

def designs(s):
    dimension = s['dimension']
    design_id = f"T{dimension}_N{s['n1']}"
    xs = [qmc.LatinHypercube(dimension, seed=seed_for('E9', 'design', design_id, 0, str(j))).random(n) for j, n in enumerate((s['n1'], s['n2']))]
    if s['geometry'] == 'half':
        xs[1][:, 0] *= 0.5
    elif s['geometry'] == 'separated':
        xs[1][:, 0] += 1.25
    evaluation = qmc.Sobol(dimension, scramble=True, seed=seed_for('E9', 'evaluation_design', f'T{dimension}', 0)).random_base2(10)
    return (xs, evaluation)

@lru_cache(maxsize=1)
def generating_model(scenario_json):
    s = json.loads(scenario_json)
    xs, evaluation = designs(s)
    return GaussianDGP(xs, evaluation, [(np.array([[1.0, s['rho']], [s['rho'], 1.0]]), s['lengthscale'], 'matern32')], [s['noise_variance']] * 2)
