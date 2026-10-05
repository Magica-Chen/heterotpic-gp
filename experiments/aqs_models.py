"""Aqs models for the multivariate kriging experiments."""
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from experiments.exact_gp import ExactGP, FitResult
from experiments.screening import joint_initial

def fit_shared(xs, ys, marginals, config):
    started = perf_counter()
    model = ExactGP(xs, ys, config)
    d = model.outputs
    bounds = np.array(model.bounds())
    free = np.arange(2 * d + model.scales)
    initial = joint_initial(marginals, model)
    initial[len(free):] = 0.0
    attempts = []
    results = []
    for index, multiplier in enumerate((0.5, 1.0, 1.0, 2.0)):
        start = initial.copy()
        start[2 * d:2 * d + model.scales] += np.log(multiplier)
        if index:
            start[d:2 * d] = np.log(np.exp(initial[:d]) * (0.03, 0.1, 0.3)[index - 1])
        start[free] = np.clip(start[free], bounds[free, 0] + 1e-07, bounds[free, 1] - 1e-07)
        bad = 0

        def objective(value):
            nonlocal bad
            theta = start.copy()
            theta[free] = value
            try:
                f, g = model.nll_gradient(theta)
                return (f, g[free])
            except (np.linalg.LinAlgError, FloatingPointError):
                bad += 1
                return (1e+100, np.zeros(len(free)))
        result = minimize(objective, start[free], jac=True, method='L-BFGS-B', bounds=bounds[free], options={'maxiter': config.maxiter, 'ftol': config.ftol, 'gtol': config.gtol, 'maxls': 40})
        theta = start.copy()
        theta[free] = result.x
        success = bool(result.success and np.isfinite(result.fun) and (result.fun < 1e+99))
        attempts.append({'index': index, 'initial_theta': start.tolist(), 'success': success, 'message': str(result.message), 'nll': float(result.fun), 'iterations': int(result.nit), 'evaluations': int(result.nfev), 'nonfinite_evaluations': bad})
        results.append((success, float(result.fun), theta))
    success, nll, theta = min([r for r in results if r[0]] or results, key=lambda r: r[1])
    norm = jitter = float('nan')
    active = []
    try:
        g = model.nll_gradient(theta)[1][free]
        lo = theta[free] - bounds[free, 0] <= 1e-05
        hi = bounds[free, 1] - theta[free] <= 1e-05
        active = [model.names[i] for i in free[lo | hi]]
        g[lo & (g > 0) | hi & (g < 0)] = 0.0
        norm = float(np.max(np.abs(g)))
        _, jitter = model.factor(model.covariance(theta))
    except (np.linalg.LinAlgError, FloatingPointError):
        success = False
    return FitResult(model, theta, success, nll, perf_counter() - started, jitter, active, norm, attempts, 'diagonal_shared_nuisance')

def point_baselines(x, y, evaluation):
    distance = cdist(evaluation, x)
    nearest = np.argmin(distance, axis=1)
    idw = []
    for row in distance:
        zero = np.flatnonzero(row <= 1e-12)
        if len(zero):
            idw.append(float(y[zero].mean()))
        else:
            order = np.argsort(row, kind='stable')[:min(5, len(y))]
            weight = row[order] ** (-2)
            idw.append(float(weight @ y[order] / weight.sum()))
    h = np.c_[np.ones(len(x)), x]
    coefficients = np.linalg.lstsq(h, y, rcond=None)[0]
    predictions = {'mean': np.full(len(evaluation), np.mean(y)), 'nearest': y[nearest], 'idw5': np.array(idw), 'linear_trend': np.c_[np.ones(len(evaluation)), evaluation] @ coefficients}
    return (predictions, {'linear_coefficients': coefficients.tolist(), 'linear_design_rank': int(np.linalg.matrix_rank(h)), 'nearest_training_indices': nearest.tolist(), 'idw_power': 2, 'idw_neighbours': min(5, len(y)), 'zero_distance_tolerance_model_units': 1e-12})
