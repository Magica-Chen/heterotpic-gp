"""Section7 pipeline for the multivariate kriging experiments."""
from dataclasses import replace
from time import perf_counter
import numpy as np
from scipy.stats import qmc
from experiments.exact_gp import FitConfig, fit_gp
from experiments.noise_aware_gp import fit_noise_gp
from experiments.shared_scale_gp import fit_shared_scale, shared_marginals
from experiments.simulation_noise import fit_noise_diagonal, fit_noise_coupled, noise_opportunity
from experiments.task_screening import TaskScreenConfig, coupled_fit, exclusion, opportunity_scores, target_folds
MODELS = ('independent', 'diagonal', 'joint')
SCREEN = TaskScreenConfig(threshold=0.2)

class Stage:
    """All response transformations use only this training stage."""

    def __init__(self, xs, ys, noise, config, linear):
        self.xs, self.ys, self.noise, self.config, self.linear = (xs, ys, noise, config, linear)
        if noise is None:
            self.centres, self.scales = (np.zeros(2), np.ones(2))
            self.z, self.nv = (ys, None)
        else:
            self.centres = np.array([y.mean() for y in ys])
            self.scales = np.array([max(y.std(ddof=1), 1e-08) for y in ys])
            self.z = [(y - c) / s for y, c, s in zip(ys, self.centres, self.scales)]
            self.nv = [v / s ** 2 for v, s in zip(noise, self.scales)]

    def marginal(self, index):
        if self.nv is None:
            return fit_gp([self.xs[index]], [self.z[index]], self.config)
        return fit_noise_gp([self.xs[index]], [self.z[index]], [self.nv[index]], config=self.config, linear_mean=self.linear)

    def diagonal(self, marginals):
        if self.nv is None:
            return fit_shared_scale(self.xs, self.z, marginals, self.config)
        return fit_noise_diagonal(self.xs, self.z, self.nv, marginals, self.config, self.linear)

    def joint(self, marginals, diagonal):
        if self.nv is None:
            return coupled_fit(self.xs, self.z, marginals, diagonal, self.config)
        return fit_noise_coupled(self.xs, self.z, self.nv, marginals, diagonal, self.config, self.linear)

    def scores(self, diagonal, evaluation, screen):
        if self.nv is None:
            return opportunity_scores(shared_marginals(diagonal), evaluation, screen)
        return noise_opportunity(diagonal, evaluation, screen)

    def predict(self, fit, x, future_noise=False):
        started = perf_counter()
        mean, latent, observed = fit.predict(x)
        prediction = np.array([mean * self.scales[0] + self.centres[0], latent * self.scales[0] ** 2, observed * self.scales[0] ** 2])
        if future_noise and self.noise is not None:
            if x.shape[1] != 1:
                raise ValueError('The interpolation convention is for Q1 only.')
            order = np.argsort(self.xs[0][:, 0])
            prediction[2] += np.exp(np.interp(x[:, 0], self.xs[0][order, 0], np.log(np.maximum(self.noise[0][order], 1e-14))))
        return (prediction, perf_counter() - started)

    def transformation(self):
        return dict(centres=self.centres.tolist(), scales=self.scales.tolist(), linear_mean=self.linear)

def execute(xs, ys, evaluation, fold_seed, config=FitConfig(), screen=SCREEN, noise=None, linear=False, mode='shadow', membership=None, future_noise=False, return_diagonal=False):
    """Shadow fits every candidate; screen/direct perform only required work.

    Direct validation uses the same three models, starts, transformations and
    target folds. The full-data joint fit is deferred until selection in real
    executions. A zero-threshold screen bypasses scores; shadow execution still
    records them. Generating values are not accepted by this function.
    """
    if mode not in ('shadow', 'screen', 'direct'):
        raise ValueError(mode)
    started = perf_counter()
    shadow = mode == 'shadow'
    direct = mode == 'direct' or (mode == 'screen' and screen.threshold == 0)
    counts = dict(marginal_fits=0, diagonal_fits=0, joint_fits=0, scores=0)
    full = Stage(xs, ys, noise, config, linear)
    marginals = [full.marginal(i) for i in range(2)]
    counts['marginal_fits'] += 2
    diagonal = full.diagonal(marginals)
    counts['diagonal_fits'] += 1
    score = None if direct else full.scores(diagonal, evaluation, screen)
    counts['scores'] += int(not direct)
    full_excluded = not direct and exclusion(score, screen.threshold, screen.retain_boundary)
    joint = full.joint(marginals, diagonal) if shadow else None
    counts['joint_fits'] += int(shadow)
    row = dict(marginals=[m.record() for m in marginals], diagonal=diagonal.record(), scores=score, transformation=full.transformation(), folds=[])
    if membership is None:
        membership = target_folds(xs[0], fold_seed, screen.folds)
    for index in range(screen.folds):
        keep = membership != index
        tx, ty = ([xs[0][keep], xs[1]], [ys[0][keep], ys[1]])
        nv = None if noise is None else [noise[0][keep], noise[1]]
        stage = Stage(tx, ty, nv, config, linear)
        target = stage.marginal(0)
        pilots = [target, marginals[1]]
        counts['marginal_fits'] += 1
        diag = stage.diagonal(pilots)
        counts['diagonal_fits'] += 1
        if shadow or (not direct and (not full_excluded)):
            local_score = stage.scores(diag, evaluation, screen)
            counts['scores'] += 1
        else:
            local_score = None
        skip = full_excluded or (not direct and local_score is not None and exclusion(local_score, screen.threshold, screen.retain_boundary))
        coupled = stage.joint(pilots, diag) if shadow or not skip else None
        counts['joint_fits'] += int(coupled is not None)
        fits = dict(independent=target, diagonal=diag if diag.success else target)
        fits['joint'] = coupled if coupled is not None and coupled.success else fits['diagonal']
        fold = dict(fold=index, target_train_indices=np.flatnonzero(keep).tolist(), target_validation_indices=np.flatnonzero(~keep).tolist(), auxiliary_train_indices=list(range(len(xs[1]))), target_marginal=target.record(), diagonal=diag.record(), joint=coupled.record() if coupled is not None else None, scores=local_score, transformation=stage.transformation(), excluded=bool(skip))
        for name in MODELS:
            if name == 'joint' and coupled is None:
                fold['joint_sse'] = fold['diagonal_sse']
                fold['joint_prediction_seconds'] = 0.0
                continue
            pred, seconds = stage.predict(fits[name], xs[0][~keep])
            fold[f'{name}_sse'] = float(np.sum((ys[0][~keep] - pred[0]) ** 2))
            fold[f'{name}_prediction_seconds'] = seconds
        row['folds'].append(fold)
    own_sse = sum((f['independent_sse'] for f in row['folds']))
    diag_sse = sum((f['diagonal_sse'] for f in row['folds']))
    base = 'diagonal' if diag_sse < (1 - screen.material_fraction) * own_sse else 'independent'
    adaptive = sum((f['diagonal_sse'] if f['excluded'] else f['joint_sse'] for f in row['folds']))
    selected = not full_excluded and adaptive < (1 - screen.material_fraction) * (diag_sse if base == 'diagonal' else own_sse)
    if selected and (not shadow):
        joint = full.joint(marginals, diagonal)
        counts['joint_fits'] += 1
    name = 'joint' if selected and joint is not None and joint.success else base
    if name == 'diagonal' and (not diagonal.success):
        name = 'independent'
    fits = dict(independent=marginals[0], diagonal=diagonal if diagonal.success else marginals[0])
    fits['joint'] = joint if joint is not None and joint.success else fits['diagonal']
    predictions = {}
    for key in MODELS if shadow else (name,):
        predictions[key], row[f'{key}_prediction_seconds'] = full.predict(fits[key], evaluation, future_noise)
    row.update(joint=joint.record() if joint is not None else None, selected_model=name, full_excluded=bool(full_excluded), counts=counts, runtime_seconds=perf_counter() - started)
    result = (row, predictions)
    return (*result, diagonal) if return_diagonal else result

def coordinate_sensitivity(diagonal, evaluation, seed, dense=False):
    """Finite covariance scenarios, conditional on full-data DIAG estimates."""
    started = perf_counter()
    dimension = diagonal.model.dimension
    box = 2.0 ** (2 * qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(5) - 1)
    multipliers = np.vstack([np.full(dimension, s) for s in (0.5, 1.0, 2.0)] + [box])
    scores = opportunity_scores(shared_marginals(diagonal), evaluation, replace(SCREEN, scale_multipliers=tuple(multipliers)))
    values = np.array([r['opportunity'] for r in scores['scenarios']])
    result = dict(scores=scores, sampled_scenario_count=len(multipliers), pointwise_exclusion_changes=bool(np.any(values < SCREEN.threshold) and np.any(values >= SCREEN.threshold)))
    if dense:
        if dimension != 2:
            raise ValueError('Dense reference is specified only for two dimensions.')
        mesh = np.stack(np.meshgrid(*[np.linspace(-1, 1, 21)] * 2), axis=-1).reshape(-1, 2)
        result['dense'] = opportunity_scores(shared_marginals(diagonal), evaluation, replace(SCREEN, scale_multipliers=tuple(2.0 ** mesh)))
    result['runtime_seconds'] = perf_counter() - started
    return result
