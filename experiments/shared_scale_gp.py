"""Shared scale gp for the multivariate kriging experiments."""
from __future__ import annotations
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from experiments.exact_gp import ExactGP, FitConfig, FitResult
from experiments.screening import joint_initial

class SharedScaleGP(ExactGP):

    def __init__(self, xs, ys, config=FitConfig()):
        super().__init__(xs, ys, config)
        if self.outputs != 2:
            raise ValueError('The replacement study uses a target and one auxiliary.')
        self.marginal_models = [ExactGP([x], [y], config) for x, y in zip(xs, ys)]

    def marginal_theta(self, theta, index):
        return np.r_[theta[index], theta[self.outputs + index], theta[2 * self.outputs:2 * self.outputs + self.scales]]

    def nll_gradient(self, theta):
        if np.any(theta[2 * self.outputs + self.scales:] != 0):
            raise ValueError('The independent shared-scale model fixes all correlations at zero.')
        value, gradient = (0.0, np.zeros_like(theta))
        for index, marginal in enumerate(self.marginal_models):
            v, g = marginal.nll_gradient(self.marginal_theta(theta, index))
            value += v
            gradient[index] = g[0]
            gradient[self.outputs + index] = g[1]
            gradient[2 * self.outputs:2 * self.outputs + self.scales] += g[2:]
        return (float(value), gradient)

    def predict(self, theta, x, target=0, full_covariance=False):
        return self.marginal_models[target].predict(self.marginal_theta(theta, target), x, target=0, full_covariance=full_covariance)

def fit_shared_scale(xs, ys, marginals, config=FitConfig()):
    started = perf_counter()
    model = SharedScaleGP(xs, ys, config)
    initial = joint_initial(marginals, model)
    bounds = np.asarray(model.bounds())
    free = np.arange(2 * model.outputs + model.scales)
    attempts, results = ([], [])
    for index in range(config.starts):
        start = initial.copy()
        start[4:4 + model.scales] += np.log((0.5, 1.0, 1.0, 2.0)[index])
        if index:
            start[2:4] = np.log(np.exp(initial[:2]) * (0.0, 0.03, 0.1, 0.3)[index])
        start[free] = np.clip(start[free], bounds[free, 0] + 1e-07, bounds[free, 1] - 1e-07)
        nonfinite = 0

        def objective(value):
            nonlocal nonfinite
            theta = start.copy()
            theta[free] = value
            try:
                v, g = model.nll_gradient(theta)
                return (v, g[free])
            except (FloatingPointError, np.linalg.LinAlgError):
                nonfinite += 1
                return (1e+100, np.zeros(len(free)))
        result = minimize(objective, start[free], jac=True, method='L-BFGS-B', bounds=bounds[free], options={'maxiter': config.maxiter, 'ftol': config.ftol, 'gtol': config.gtol, 'maxls': 40})
        theta = start.copy()
        theta[free] = result.x
        success = bool(result.success and np.isfinite(result.fun) and (result.fun < 1e+99))
        attempts.append({'index': index, 'initial_theta': start.tolist(), 'success': success, 'message': str(result.message), 'nll': float(result.fun), 'iterations': int(result.nit), 'evaluations': int(result.nfev), 'nonfinite_evaluations': nonfinite})
        results.append((success, float(result.fun), theta))
    success, nll, theta = min([r for r in results if r[0]] or results, key=lambda r: r[1])
    active, gradient_norm, jitter = ([], float('nan'), float('nan'))
    try:
        _, grad = model.nll_gradient(theta)
        projected = grad[free].copy()
        for i, p in enumerate(free):
            low, high = (theta[p] - bounds[p, 0] <= 1e-05, bounds[p, 1] - theta[p] <= 1e-05)
            if low or high:
                active.append(model.names[p])
            if low and grad[p] > 0 or (high and grad[p] < 0):
                projected[i] = 0
        gradient_norm = float(np.max(np.abs(projected)))
        jitter = max((m.factor(m.covariance(model.marginal_theta(theta, i)))[1] for i, m in enumerate(model.marginal_models)))
    except (np.linalg.LinAlgError, FloatingPointError):
        success = False
    return FitResult(model, theta, success, nll, perf_counter() - started, jitter, active, gradient_norm, attempts, 'shared_scale_independent')

def shared_marginals(fit):
    """Views of fitted diagonal blocks for common-covariance opportunity scores."""
    result = []
    for i, model in enumerate(fit.model.marginal_models):
        active = [name for name in fit.active_bounds if name.startswith('log_lengthscale') or name in (f'log_signal_variance_{i}', f'log_noise_variance_{i}')]
        result.append(FitResult(model, fit.model.marginal_theta(fit.theta, i), fit.success, fit.nll, 0.0, fit.jitter, active, fit.gradient_norm, [], 'shared_scale_block'))
    return result
