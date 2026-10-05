"""Auxiliary design for the multivariate kriging experiments."""
from __future__ import annotations
import numpy as np
from experiments.synthetic import GaussianDGP, pair_design

def e8_dgp(case):
    xs = pair_design(case['geometry'], case['n1'], case['n2'])
    evaluation = ((np.arange(201) + 0.5) / 201)[:, None]
    if case['family'] == 'lmc_positive':
        loading = np.array([[1.0, 0.25], [0.2, 1.0]])
        loading /= np.linalg.norm(loading, axis=1)[:, None]
        components = [(np.outer(loading[:, k], loading[:, k]), ell, 'matern32') for k, ell in enumerate((0.06, 0.35))]
    else:
        components = [(np.array([[0.49, 0.7], [0.7, 1.0]]), 0.3, 'matern32'), (np.diag([0.51, 0.0]), 0.06, 'matern32')]
    return GaussianDGP(xs, evaluation, components, [0.05, 0.05])
