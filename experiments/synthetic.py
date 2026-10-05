"""Synthetic for the multivariate kriging experiments."""
from __future__ import annotations
import hashlib
from itertools import product
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from experiments.exact_gp import kernel_and_logscale_derivatives

def seed_for(experiment, namespace, scenario_id, replication, purpose='dataset', master=2602738):
    text = f'{master}|{experiment}|{namespace}|{scenario_id}|{replication}|{purpose}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')

def pair_design(geometry, n1, n2=None):
    n2 = n1 if n2 is None else n2
    x = np.linspace(0, 1, n1)[:, None]
    if geometry == 'isotopic':
        if n1 != n2:
            raise ValueError('Isotopic designs require equal sample sizes.')
        y = x.copy()
    elif geometry in ('interleaved', 'moderate', 'separated'):
        offset = {'interleaved': 0, 'moderate': 0.5, 'separated': 1.25}[geometry]
        y = (offset + (np.arange(n2) + 0.5) / n2)[:, None]
    else:
        raise ValueError(f'Unknown geometry: {geometry}')
    return [x, y]

def core_scenarios():
    scenarios = []
    for index, (geometry, n, rho, noise) in enumerate(product(('isotopic', 'interleaved', 'moderate', 'separated'), (12, 30), (0.0, 0.4, 0.8), (0.01, 0.1))):
        scenarios.append({'id': f'C{index + 1:02d}', 'geometry': geometry, 'n1': n, 'n2': n, 'rho': rho, 'noise_variance': [noise, noise], 'kernel': 'matern32', 'lengthscale': 0.15, 'signal_variance': [1.0, 1.0], 'dimension': 1})
    return scenarios

class GaussianDGP:
    """Exact generator and conditional risk reference, never a fitting input.

    Components are (output covariance, lengthscale, kernel family). Arbitrary
    output-specific Gaussian noise and deterministic means can be supplied.
    Training observations and target latent values are sampled jointly. Only
    the training observation blocks contain measurement noise.
    """

    def __init__(self, xs, evaluation, components, noise, means=None, evaluation_mean=None):
        self.xs = [np.asarray(x, float) for x in xs]
        self.evaluation = np.asarray(evaluation, float)
        self.x = np.vstack([*self.xs, self.evaluation])
        self.ids = np.r_[np.repeat(np.arange(len(xs)), [len(x) for x in xs]), np.zeros(len(evaluation), dtype=int)]
        self.ntrain = sum((len(x) for x in xs))
        covariance = np.zeros((len(self.x), len(self.x)))
        for output_covariance, ell, family in components:
            k = kernel_and_logscale_derivatives(self.x, self.x, np.atleast_1d(ell), family, True)[0]
            covariance += np.asarray(output_covariance)[self.ids[:, None], self.ids[None, :]] * k
        self.noise = [np.broadcast_to(n, (len(x),)).copy() for n, x in zip(noise, xs)]
        covariance[:self.ntrain, :self.ntrain] += np.diag(np.concatenate(self.noise))
        self.covariance = covariance
        self.means = [np.zeros(len(x)) for x in xs] if means is None else means
        self.evaluation_mean = np.zeros(len(evaluation)) if evaluation_mean is None else evaluation_mean
        self.mean = np.r_[np.concatenate(self.means), self.evaluation_mean]
        self.cholesky = np.linalg.cholesky(covariance)
        train = covariance[:self.ntrain, :self.ntrain]
        cross = covariance[self.ntrain:, :self.ntrain]
        factor = cho_factor(train, lower=True)
        self.joint_weights = cho_solve(factor, cross.T).T
        self.joint_variance = np.diag(covariance[self.ntrain:, self.ntrain:]) - np.sum(self.joint_weights * cross, axis=1)
        n0 = len(xs[0])
        self.ind_weights = cho_solve(cho_factor(train[:n0, :n0], lower=True), cross[:, :n0].T).T
        self.ind_variance = np.diag(covariance[self.ntrain:, self.ntrain:]) - np.sum(self.ind_weights * cross[:, :n0], axis=1)

    def draw(self, seed):
        rng = np.random.default_rng(seed)
        draw = self.mean + self.cholesky @ rng.standard_normal(len(self.x))
        ys = np.split(draw[:self.ntrain], np.cumsum([len(x) for x in self.xs])[:-1])
        f_eval = draw[self.ntrain:]
        centred = draw[:self.ntrain] - self.mean[:self.ntrain]
        oracle_joint = self.evaluation_mean + self.joint_weights @ centred
        oracle_ind = self.evaluation_mean + self.ind_weights @ centred[:len(self.xs[0])]
        return (ys, f_eval, oracle_joint, oracle_ind)

def core_dgp(scenario):
    xs = pair_design(scenario['geometry'], scenario['n1'], scenario['n2'])
    evaluation = ((np.arange(201) + 0.5) / 201)[:, None]
    rho = scenario['rho']
    v = np.asarray(scenario['signal_variance'])
    lam = np.sqrt(v[:, None] * v[None, :]) * np.array([[1.0, rho], [rho, 1.0]])
    return GaussianDGP(xs, evaluation, [(lam, scenario['lengthscale'], scenario['kernel'])], scenario['noise_variance'])
