"""Task screening for the multivariate kriging experiments."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from experiments.geometry import unique_sites
from experiments.diagnostics import correlation_information
from experiments.exact_gp import FitConfig, fit_gp, kernel_and_logscale_derivatives
from experiments.screening import fit_marginals, fit_pair, pair_bpi
from experiments.shared_scale_gp import fit_shared_scale, shared_marginals

@dataclass(frozen=True)
class TaskScreenConfig:
    scale_multipliers: tuple = (0.5, 1.0, 2.0)
    rho_max: float = 0.995
    threshold: float = 0.0
    material_fraction: float = 0.02
    folds: int = 4
    retain_boundary: bool = False

def conditional_gain(k11, k22, k12, own, auxiliary, variance, noise, rho):
    """Exact pointwise gain at rho, without a joint likelihood optimisation.

    With t=rho^2, B=K21 (K11+noise1/variance1 I)^-1 K12,
    D=K22+noise2/variance2 I and R the residual cross kernel, the gain
    is variance1 * t * diag(R (D-tB)^-1 R'). The derivative of
    t(D-tB)^-1 is (D-tB)^-1 D (D-tB)^-1, which is positive definite.
    """
    if not 0 <= abs(rho) < 1:
        raise ValueError('The dependence envelope must be strictly below one.')
    variance, noise = (np.asarray(variance, float), np.asarray(noise, float))
    if np.any(variance <= 0) or np.any(noise <= 0):
        raise ValueError('Positive finite signal and noise variances are required.')
    target = cho_factor(k11 + noise[0] / variance[0] * np.eye(len(k11)), lower=True)
    target_cross = cho_solve(target, k12)
    residual = auxiliary - own @ target_cross
    vind = variance[0] * (1 - np.einsum('ij,ji->i', own, cho_solve(target, own.T)))
    schur = k22 + noise[1] / variance[1] * np.eye(len(k22)) - rho ** 2 * k12.T @ target_cross
    factor = cho_factor((schur + schur.T) / 2, lower=True)
    gain = variance[0] * rho ** 2 * np.einsum('ij,ji->i', residual, cho_solve(factor, residual.T))
    tolerance = 1e-07 * max(variance[0], 1.0)
    if not np.all(np.isfinite(np.r_[vind, gain])) or np.min(vind - gain) < -tolerance:
        raise FloatingPointError('Invalid conditional variance or opportunity gain.')
    if np.min(vind) < -tolerance or np.min(gain) < -tolerance:
        raise FloatingPointError('Negative conditional variance or opportunity gain.')
    return (np.maximum(vind, 0), np.minimum(np.maximum(gain, 0), np.maximum(vind, 0)), residual)

def opportunity_scores(marginals, evaluation, config=TaskScreenConfig()):
    started = perf_counter()
    if len(marginals) != 2:
        raise ValueError('This screen is for one target and one auxiliary.')
    xs = [m.model.xs[0] for m in marginals]
    flags, warnings = ([], [])
    for i, fit in enumerate(marginals):
        if not fit.success:
            flags.append(f'marginal_{i}_fit_failed')
        if len(unique_sites(xs[i])) < 2:
            flags.append(f'marginal_{i}_fewer_than_two_sites')
        if fit.active_bounds:
            warnings.append(f'marginal_{i}_active_bound')
            if config.retain_boundary:
                flags.append(f'marginal_{i}_active_bound')
        if not np.all(np.isfinite(fit.theta)):
            flags.append(f'marginal_{i}_nonfinite_parameters')
        if fit.jitter > fit.model.config.jitter_ladder[0] * fit.model.config.jitter_scale:
            flags.append(f'marginal_{i}_jitter_escalated')
    natural = [m.model.unpack(m.theta) for m in marginals]
    variance = np.array([v[0][0] for v in natural])
    noise = np.array([v[1][0] for v in natural])
    scales = np.vstack([v[2] for v in natural])
    pilot = np.exp(np.average(np.log(scales), axis=0, weights=list(map(len, xs))))
    ratio = float(np.max(scales.max(axis=0) / scales.min(axis=0)))
    if ratio > 4:
        warnings.append('marginal_lengthscale_ratio_above_four')
    fit_config = marginals[0].model.config
    scenarios = []
    for multiplier in config.scale_multipliers:
        ell = multiplier * pilot

        def k(a, b):
            return kernel_and_logscale_derivatives(a, b, ell, fit_config.kernel, fit_config.ard)[0]
        try:
            k11, k22, k12 = (k(xs[0], xs[0]), k(xs[1], xs[1]), k(xs[0], xs[1]))
            vind, gain, residual = conditional_gain(k11, k22, k12, k(evaluation, xs[0]), k(evaluation, xs[1]), variance, noise, config.rho_max)
            if vind.mean() <= 1e-12 * variance[0]:
                raise FloatingPointError('Marginal variance too small for a relative score')
            loose = np.minimum(vind, variance[0] * variance[1] * config.rho_max ** 2 * np.sum(residual ** 2, axis=1) / noise[1])
            information = correlation_information(k12, variance[0] * k11 + noise[0] * np.eye(len(k11)), variance[1] * k22 + noise[1] * np.eye(len(k22)), *variance)
            scenarios.append({'multiplier': multiplier, 'lengthscale': ell.tolist(), 'opportunity': float(gain.mean() / vind.mean()), 'integrated_gain': float(gain.mean()), 'integrated_variance': float(vind.mean()), 'loose_opportunity': float(loose.mean() / vind.mean()), 'information': information, 'information_per_min_n': information / min(map(len, xs))})
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            flags.append(f'scale_{multiplier}_unavailable')
    return {'opportunity': max((s['opportunity'] for s in scenarios), default=float('nan')), 'loose_opportunity': max((s['loose_opportunity'] for s in scenarios), default=float('nan')), 'information': max((s['information'] for s in scenarios), default=float('nan')), 'flags': sorted(set(flags)), 'warnings': sorted(set(warnings)), 'scenarios': scenarios, 'pilot_lengthscale': pilot.tolist(), 'marginal_scale_ratio': ratio, 'bpi': pair_bpi(xs[0], xs[1], evaluation), 'settings': asdict(config), 'runtime_seconds': perf_counter() - started}

def exclusion(scores, threshold, retain_boundary=False, loose=False):
    if threshold < 0 or not np.isfinite(threshold):
        raise ValueError('The threshold must be finite and nonnegative.')
    value = scores['loose_opportunity' if loose else 'opportunity']
    unavailable = bool(scores['flags']) or not np.isfinite(value)
    unavailable |= retain_boundary and any(('active_bound' in w for w in scores['warnings']))
    return bool(not unavailable and value < threshold)

def target_folds(x, seed, count=4):
    sites = np.rint(np.asarray(x) / 1e-10).astype(np.int64)
    unique, membership = np.unique(sites, axis=0, return_inverse=True)
    if len(unique) < count:
        raise ValueError('Fewer target sites than folds.')
    order = np.random.default_rng(seed).permutation(len(unique))
    fold = np.empty(len(unique), int)
    fold[order] = np.arange(len(unique)) % count
    return fold[membership]

def fold_data(xs, ys, membership, index):
    keep = membership != index
    return ([xs[0][keep], xs[1]], [ys[0][keep], ys[1]], xs[0][~keep], ys[0][~keep])

def choose(folds, threshold, fraction=0.02, direct=False, retain_boundary=False, loose=False):
    direct = direct or threshold == 0
    own = sum((f['independent_sse'] for f in folds))
    diagonal = sum((f['diagonal_sse'] for f in folds))
    baseline = 'diagonal' if diagonal < (1 - fraction) * own else 'independent'
    adaptive = sum((f['diagonal_sse'] if not direct and exclusion(f['scores'], threshold, retain_boundary, loose) else f['joint_sse'] for f in folds))
    return 'joint' if adaptive < (1 - fraction) * (diagonal if baseline == 'diagonal' else own) else baseline

def coupled_fit(xs, ys, marginals, diagonal, fit_config):
    return fit_gp(xs, ys, fit_config, initial=diagonal.theta) if diagonal.success else fit_pair(xs, ys, marginals, fit_config)

def run_pipeline(xs, ys, evaluation, seed, fit_config=FitConfig(), config=TaskScreenConfig(), direct=False, *, retain_all=False):
    """Real execution, including every required score, fit and prediction.

    Independent auxiliary pilots are reusable across target folds because all
    auxiliary observations remain available. Both procedures use this cache.
    A zero threshold bypasses scoring. The explicit retain-all ablation computes
    scores while suppressing exclusions, including at a zero threshold.
    """
    if direct and retain_all:
        raise ValueError('Direct validation and score-inclusive retain-all are distinct controls.')
    direct = direct or (config.threshold == 0 and (not retain_all))
    started = perf_counter()
    counts = {'marginal_fits': 0, 'diagonal_fits': 0, 'joint_fits': 0, 'scores': 0}
    full = fit_marginals(xs, ys, fit_config)
    counts['marginal_fits'] += 2
    full_diagonal = fit_shared_scale(xs, ys, full, fit_config)
    counts['diagonal_fits'] += 1
    full_excluded = False
    if not direct:
        scores = opportunity_scores(shared_marginals(full_diagonal), evaluation, config)
        counts['scores'] += 1
        full_excluded = not retain_all and exclusion(scores, config.threshold, config.retain_boundary)
    aux = full[1]
    membership = target_folds(xs[0], seed, config.folds)
    records = []
    for index in range(config.folds):
        tx, ty, vx, vy = fold_data(xs, ys, membership, index)
        own = fit_gp([tx[0]], [ty[0]], fit_config)
        counts['marginal_fits'] += 1
        pilots = [own, aux]
        own_mean = own.predict(vx)[0]
        diagonal = fit_shared_scale(tx, ty, pilots, fit_config)
        counts['diagonal_fits'] += 1
        diagonal_mean = (diagonal if diagonal.success else own).predict(vx)[0]
        if direct or full_excluded:
            skip = full_excluded
        else:
            score = opportunity_scores(shared_marginals(diagonal), evaluation, config)
            counts['scores'] += 1
            skip = not retain_all and exclusion(score, config.threshold, config.retain_boundary)
        joint = None if skip else coupled_fit(tx, ty, pilots, diagonal, fit_config)
        counts['joint_fits'] += int(joint is not None)
        joint_mean = joint.predict(vx)[0] if joint is not None and joint.success else diagonal_mean
        records.append({'independent_sse': float(np.sum((vy - own_mean) ** 2)), 'diagonal_sse': float(np.sum((vy - diagonal_mean) ** 2)), 'joint_sse': float(np.sum((vy - joint_mean) ** 2))})
    own_sse = sum((f['independent_sse'] for f in records))
    diag_sse = sum((f['diagonal_sse'] for f in records))
    baseline = 'diagonal' if diag_sse < (1 - config.material_fraction) * own_sse else 'independent'
    selected = not full_excluded and sum((f['joint_sse'] for f in records)) < (1 - config.material_fraction) * (diag_sse if baseline == 'diagonal' else own_sse)
    joint = coupled_fit(xs, ys, full, full_diagonal, fit_config) if selected else None
    counts['joint_fits'] += int(joint is not None)
    selected = bool(joint is not None and joint.success)
    name = 'joint' if selected else baseline
    if name == 'diagonal' and (not full_diagonal.success):
        name = 'independent'
    prediction = {'joint': joint, 'diagonal': full_diagonal, 'independent': full[0]}[name].predict(evaluation)
    return {'selected_joint': selected, 'selected_model': name, 'prediction': prediction, 'full_excluded': full_excluded, 'counts': counts, 'runtime_seconds': perf_counter() - started}
