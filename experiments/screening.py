"""Screening for the multivariate kriging experiments."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from experiments.geometry import SITE_TOLERANCE, overlap_index, unique_sites
from experiments.diagnostics import correlation_information
from experiments.exact_gp import FitConfig, fit_gp, kernel_and_logscale_derivatives

@dataclass(frozen=True)
class ScreenConfig:
    scale_multipliers: tuple = (0.5, 1.0, 2.0)
    rho_max: float = 0.95
    variance_floor: float = 1e-10
    marginal_scale_ratio_limit: float = 4.0
    opportunity_threshold: float = 0.0
    information_threshold: float = 0.0
    material_fraction: float = 0.02
    folds: int = 4

def pair_bpi(x, y, evaluation):
    x, y = (unique_sites(x), unique_sites(y))
    if len(x) == 0 or len(y) == 0:
        return float('nan')
    own, aux = (cdist(evaluation, x).min(axis=1), cdist(evaluation, y).min(axis=1))
    return float(np.mean(np.divide(np.maximum(own - aux, 0), own, out=np.zeros_like(own), where=own > 0)))

def pair_scores(marginals, evaluation, screen=ScreenConfig()):
    start = perf_counter()
    if len(marginals) != 2:
        raise ValueError('The binary gate applies only to target-plus-one-auxiliary models.')
    x, y = [f.model.xs[0] for f in marginals]
    evaluation = np.asarray(evaluation, float)
    flags = []
    for i, fit in enumerate(marginals):
        if not fit.success:
            flags.append(f'marginal_{i}_fit_failed')
        if fit.active_bounds:
            flags.append(f'marginal_{i}_active_bound')
        if len(unique_sites(fit.model.xs[0])) < 2:
            flags.append(f'marginal_{i}_fewer_than_two_sites')
        if fit.jitter > fit.model.config.jitter_ladder[0] * fit.model.config.jitter_scale:
            flags.append(f'marginal_{i}_jitter_escalated')
    parameters = [f.model.unpack(f.theta) for f in marginals]
    ells = np.vstack([v[2] for v in parameters])
    if np.any(ells.max(axis=0) / ells.min(axis=0) > screen.marginal_scale_ratio_limit):
        flags.append('marginal_lengthscale_ratio_above_limit')
    ell = np.exp(np.average(np.log(ells), axis=0, weights=[len(x), len(y)]))
    v = np.array([p[0][0] for p in parameters])
    noise = np.array([p[1][0] for p in parameters])
    config = marginals[0].model.config
    scenarios = []
    for multiplier in screen.scale_multipliers:
        scale = ell * multiplier

        def k(a, b):
            return kernel_and_logscale_derivatives(a, b, scale, config.kernel, config.ard)[0]
        try:
            k11, k22, k12 = (k(x, x), k(y, y), k(x, y))
            own, auxiliary = (k(evaluation, x), k(evaluation, y))
            target_factor = cho_factor(k11 + noise[0] / v[0] * np.eye(len(x)), lower=True)
            residual = auxiliary - own @ cho_solve(target_factor, k12)
            vind = v[0] * (1 - np.sum(own * cho_solve(target_factor, own.T).T, axis=1))
            if np.min(vind) < -1e-07 * max(v[0], 1):
                raise np.linalg.LinAlgError('Negative marginal variance')
            vind = np.maximum(vind, 0)
            upper = v[0] * v[1] * screen.rho_max ** 2 * np.sum(residual ** 2, axis=1) / noise[1]
            opportunity = np.mean(np.minimum(vind, upper) / np.maximum(vind, screen.variance_floor))
            information = correlation_information(k12, v[0] * k11 + noise[0] * np.eye(len(x)), v[1] * k22 + noise[1] * np.eye(len(y)), *v)
            scenarios.append({'multiplier': multiplier, 'lengthscale': scale.tolist(), 'opportunity': float(opportunity), 'information': information, 'mass': float(np.sum(k12 * k12)), 'average_mass': float(np.mean(k12 * k12)), 'information_per_min_n': information / min(len(x), len(y)), 'variance_floor_count': int(np.sum(vind < screen.variance_floor))})
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            flags.append(f'scale_{multiplier}_solve_failed')
    return {'opportunity': max((s['opportunity'] for s in scenarios), default=float('nan')), 'information': max((s['information'] for s in scenarios), default=float('nan')), 'pilot_lengthscale': ell.tolist(), 'scenarios': scenarios, 'flags': sorted(set(flags)), 'bpi': pair_bpi(x, y, evaluation), 'overlap': overlap_index(x, y), 'minimum_sample_size': min(len(x), len(y)), 'runtime_seconds': perf_counter() - start, 'settings': asdict(screen)}

def gate(scores, opportunity_threshold, information_threshold):
    if opportunity_threshold < 0 or information_threshold < 0:
        raise ValueError('Gate thresholds must be nonnegative.')
    if scores['flags'] or not np.isfinite(scores['opportunity'] + scores['information']):
        return 'unresolved'
    return 'exclude' if scores['opportunity'] < opportunity_threshold and scores['information'] < information_threshold else 'retain'

def site_folds(xs, seed, folds=4):
    """Assign common site IDs without outcomes; balance target sites first.

    Other unique sites are independently shuffled and assigned round-robin.
    Every observation, from every output, at a held-out site has the same fold.
    """
    quantised = [np.rint(np.asarray(x) / SITE_TOLERANCE).astype(np.int64) for x in xs]
    target_sites = sorted(set(map(tuple, quantised[0])))
    if len(target_sites) < folds:
        raise ValueError('Fewer target sites than validation folds.')
    other_sites = sorted(set().union(*(set(map(tuple, x)) for x in quantised)) - set(target_sites))
    rng = np.random.default_rng(seed)
    mapping = {}
    for sites in (target_sites, other_sites):
        for index, position in enumerate(rng.permutation(len(sites))):
            mapping[sites[position]] = index % folds
    return [np.array([mapping[tuple(row)] for row in x]) for x in quantised]

def fit_marginals(xs, ys, fit_config):
    return [fit_gp([x], [y], fit_config) for x, y in zip(xs, ys)]

def joint_initial(marginals, joint_model):
    natural = [m.model.unpack(m.theta) for m in marginals]
    ell = np.exp(np.average(np.log(np.vstack([p[2] for p in natural])), axis=0, weights=[len(m.model.y) for m in marginals]))
    return joint_model.pack([p[0][0] for p in natural], [p[1][0] for p in natural], ell, np.eye(len(marginals)))

def fit_pair(xs, ys, marginals, fit_config):
    from experiments.exact_gp import ExactGP
    model = ExactGP(xs, ys, fit_config)
    return fit_gp(xs, ys, fit_config, initial=joint_initial(marginals, model))

def validate_pair(xs, ys, evaluation, seed, fit_config=FitConfig(), screen=ScreenConfig(), shadow=False):
    """Executable fold-adaptive validation. Shadow fits are assessment-only.

    With shadow=False, excluded candidates are never optimised. The returned
    losses and timings distinguish the adaptive and unscreened comparisons.
    """
    folds = site_folds(xs, seed, screen.folds)
    records = []
    for fold in range(screen.folds):
        masks = [membership != fold for membership in folds]
        train_x = [x[mask] for x, mask in zip(xs, masks)]
        train_y = [y[mask] for y, mask in zip(ys, masks)]
        if any((len(x) == 0 for x in train_x)):
            raise ValueError('Site holdout leaves an output without training observations.')
        target_validation = ~masks[0]
        validation_x, validation_y = (xs[0][target_validation], ys[0][target_validation])
        marginals = fit_marginals(train_x, train_y, fit_config)
        scores = pair_scores(marginals, evaluation, screen)
        status = gate(scores, screen.opportunity_threshold, screen.information_threshold)
        ind_mean = marginals[0].predict(validation_x)[0]
        joint = fit_pair(train_x, train_y, marginals, fit_config) if shadow or status != 'exclude' else None
        joint_mean = joint.predict(validation_x)[0] if joint is not None and joint.success else ind_mean
        records.append({'fold': fold, 'train_indices': [np.flatnonzero(m).tolist() for m in masks], 'validation_target_indices': np.flatnonzero(target_validation).tolist(), 'count': len(validation_y), 'independent_sse': float(np.sum((ind_mean - validation_y) ** 2)), 'joint_sse': float(np.sum((joint_mean - validation_y) ** 2)), 'scores': scores, 'gate': status, 'marginals': [m.record() for m in marginals], 'joint': joint.record() if joint is not None else None})
    return records

def validation_selection(fold_records, screen=ScreenConfig(), unscreened=False):
    independent = sum((f['independent_sse'] for f in fold_records))
    adaptive = 0.0
    for fold in fold_records:
        status = gate(fold['scores'], screen.opportunity_threshold, screen.information_threshold)
        retained = unscreened or status != 'exclude'
        if retained and fold['joint'] is None:
            raise ValueError('Cannot assess a retained fold without its joint fit; use shadow traces for calibration.')
        adaptive += fold['joint_sse'] if retained else fold['independent_sse']
    return bool(adaptive < (1 - screen.material_fraction) * independent)

def screen_validate(xs, ys, evaluation, seed, fit_config=FitConfig(), screen=ScreenConfig()):
    """Deployable pipeline using no generator parameters or latent test values."""
    start = perf_counter()
    folds = validate_pair(xs, ys, evaluation, seed, fit_config, screen, shadow=False)
    selected = validation_selection(folds, screen)
    marginals = fit_marginals(xs, ys, fit_config)
    scores = pair_scores(marginals, evaluation, screen)
    status = gate(scores, screen.opportunity_threshold, screen.information_threshold)
    joint = fit_pair(xs, ys, marginals, fit_config) if selected and status != 'exclude' else None
    use_joint = joint is not None and joint.success
    prediction = (joint if use_joint else marginals[0]).predict(evaluation)
    return {'selected_joint': use_joint, 'adaptive_selected_in_validation': selected, 'full_gate': status, 'scores': scores, 'folds': folds, 'marginals': [m.record() for m in marginals], 'joint': joint.record() if joint else None, 'prediction': prediction, 'runtime_seconds': perf_counter() - start}
