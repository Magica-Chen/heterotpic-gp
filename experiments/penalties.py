"""Penalties for the multivariate kriging experiments."""
from __future__ import annotations
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from experiments.exact_gp import ExactGP
from experiments.diagnostics import expected_information
from experiments.paired_models import fit_models
from experiments.synthetic import seed_for

def free_indices(fit):
    m = fit.model
    if fit.mode == 'fixed':
        return np.array([], dtype=int)
    return np.arange(2 * m.outputs + m.scales, m.parameter_count) if fit.mode == 'correlation_only' else np.arange(m.parameter_count)

def observed_information(model, theta, free, step=0.0001):
    columns = []
    for index in free:
        h = step * max(1.0, abs(theta[index]))
        shift = np.eye(len(theta))[index] * h
        columns.append((model.nll_gradient(theta + shift)[1][free] - model.nll_gradient(theta - shift)[1][free]) / (2 * h))
    matrix = np.column_stack(columns)
    return (matrix + matrix.T) / 2

def quadratic_penalty(information, gradient, relative_floor=1e-10):
    """Unavailable for ill-conditioned/indefinite information; never pseudoinvert."""
    eig = np.linalg.eigvalsh(information)
    valid = bool(np.all(np.isfinite(eig)) and eig[0] > relative_floor * max(1.0, eig[-1]))
    result = {'eigenvalues': eig.tolist(), 'condition_number': float(eig[-1] / eig[0]) if eig[0] > 0 else None, 'available': valid, 'penalty': None}
    if valid:
        solved = cho_solve(cho_factor(information, lower=True), gradient.T).T
        result['penalty'] = float(np.mean(np.sum(gradient * solved, axis=1)))
    return result

def information_penalties(fit, evaluation):
    start = perf_counter()
    free = free_indices(fit)
    if not fit.success:
        return {'available': False, 'reason': 'fit_failed', 'runtime_seconds': perf_counter() - start}
    if not len(free):
        fixed = {'available': True, 'penalty': 0.0, 'eigenvalues': [], 'condition_number': None}
        return {'available': True, 'fixed': True, 'free_parameter_names': [], 'expected': fixed, 'observed': fixed, 'regular_interior': True, 'runtime_seconds': perf_counter() - start}
    model, theta = (fit.model, fit.theta)
    covariance, derivatives = model.covariance(theta, derivatives=True)
    _, jitter = model.factor(covariance)
    expected = expected_information(covariance + jitter * np.eye(len(covariance)), derivatives[free])
    observed = observed_information(model, theta, free)
    observed_half = observed_information(model, theta, free, step=5e-05)
    relative_discrepancy = float(np.linalg.norm(observed - observed_half) / max(1.0, np.linalg.norm(observed_half)))
    gradient = model.prediction_gradient(theta, evaluation)[:, free]
    result = {'available': True, 'fixed': False, 'free_parameter_names': [model.names[i] for i in free], 'expected': quadratic_penalty(expected, gradient), 'observed': quadratic_penalty(observed_half, gradient), 'expected_matrix': expected.tolist(), 'observed_matrix': observed_half.tolist(), 'observed_step_relative_discrepancy': relative_discrepancy, 'regular_interior': not fit.active_bounds and fit.gradient_norm <= model.config.gtol, 'active_bounds': fit.active_bounds}
    if relative_discrepancy > 0.001:
        result['observed'].update(available=False, penalty=None, reason='hessian_step_instability')
    p = model.correlations
    if fit.mode == 'full' and p and result['expected']['available']:
        nuisance = expected[:-p, :-p]
        cross = expected[-p:, :-p]
        nu_factor = cho_factor(nuisance, lower=True)
        efficient = expected[-p:, -p:] - cross @ cho_solve(nu_factor, cross.T)
        adjusted = gradient[:, -p:] - gradient[:, :-p] @ cho_solve(nu_factor, cross.T)
        result['block_decomposition'] = {'nuisance': quadratic_penalty(nuisance, gradient[:, :-p])['penalty'], 'adjusted_dependence': quadratic_penalty(efficient, adjusted)['penalty'], 'eta_only_hypothetical': quadratic_penalty(expected[-p:, -p:], gradient[:, -p:])['penalty'], 'efficient_information_eigenvalues': np.linalg.eigvalsh(efficient).tolist()}
    result['runtime_seconds'] = perf_counter() - start
    return result

def reference_marginal(joint):
    model = joint.model
    v, noise, ell, _ = model.unpack(joint.theta)
    marginal = ExactGP([model.xs[0]], [model.ys[0]], model.config)
    theta = marginal.pack(v[:1], noise[:1], ell, np.eye(1))
    return (marginal, theta)

def fitted_oracle_gain(joint, evaluation):
    if not joint.success:
        return None
    marginal, theta = reference_marginal(joint)
    own = marginal.predict(theta, evaluation)[1]
    combined = joint.predict(evaluation)[1]
    return {'gain': float(np.mean(own - combined)), 'marginal_variance': float(np.mean(own)), 'joint_variance': float(np.mean(combined))}

def bootstrap_draw(joint, evaluation, seed, correlation_only=False):
    """Refitted and reference predictors see the identical newly generated Y_b."""
    if not joint.success:
        raise ValueError('No fitted joint bootstrap reference is available.')
    model = joint.model
    y = np.linalg.cholesky(model.covariance(joint.theta)) @ np.random.default_rng(seed).standard_normal(len(model.y))
    ys = np.split(y, np.cumsum([len(x) for x in model.xs])[:-1])
    v, noise, ell, _ = model.unpack(joint.theta)
    known = {'signal_variance': v, 'noise_variance': noise, 'lengthscale': ell} if correlation_only else None
    marginals, refit = fit_models(model.xs, ys, model.config, known)
    reference = ExactGP(model.xs, ys, model.config)
    reference_joint = reference.predict(joint.theta, evaluation)[0]
    marginal = ExactGP([model.xs[0]], [ys[0]], model.config)
    marginal_theta = marginal.pack(v[:1], noise[:1], ell, np.eye(1))
    reference_ind = marginal.predict(marginal_theta, evaluation)[0]
    success = bool(refit.success and marginals[0].success)
    joint_penalty = float(np.mean((refit.predict(evaluation)[0] - reference_joint) ** 2)) if refit.success else None
    ind_penalty = float(np.mean((marginals[0].predict(evaluation)[0] - reference_ind) ** 2)) if marginals[0].success else None
    return {'seed': seed, 'success': success, 'joint_penalty': joint_penalty, 'independent_penalty': ind_penalty, 'difference': joint_penalty - ind_penalty if success else None, 'joint': refit.record(), 'marginals': [f.record() for f in marginals], 'train_y': y.tolist()}

def bootstrap_penalties(joint, evaluation, scenario_id, outer_rep, phase, draws=200):
    start = perf_counter()
    if not joint.success:
        return {'available': False, 'reason': 'reference_joint_fit_failed', 'draws': [], 'runtime_seconds': perf_counter() - start}
    records = [bootstrap_draw(joint, evaluation, seed_for('E7', phase, scenario_id, outer_rep, f'bootstrap_{b}'), correlation_only=joint.mode == 'correlation_only') for b in range(draws)]
    good = [r for r in records if r['success']]
    failure_rate = 1 - len(good) / draws
    summary = {name: float(np.mean([r[name] for r in good])) if good else None for name in ('joint_penalty', 'independent_penalty', 'difference')}
    return {'available': bool(good) and failure_rate <= 0.05, 'successful_draw_summary': summary, 'failure_rate': failure_rate, 'attempted_draws': draws, 'successful_draws': len(good), 'active_bound_draws': sum((bool(r['joint']['active_bounds'] or r['marginals'][0]['active_bounds']) for r in records)), 'draws': records, 'runtime_seconds': perf_counter() - start}
