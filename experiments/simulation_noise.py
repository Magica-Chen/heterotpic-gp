"""Section7 noise for the multivariate kriging experiments."""
from dataclasses import asdict
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from experiments.geometry import unique_sites
from experiments.exact_gp import FitConfig, kernel_and_logscale_derivatives
from experiments.noise_aware_gp import NoiseAwareGP, NoiseFit
from experiments.screening import pair_bpi
from experiments.task_screening import TaskScreenConfig

class NoiseSharedGP(NoiseAwareGP):
    """Sum of two marginal likelihoods, including separately profiled means."""

    def __init__(self, xs, ys, noise, config=FitConfig(), linear_mean=False):
        super().__init__(xs, ys, noise, config=config, linear_mean=linear_mean)
        if self.outputs != 2:
            raise ValueError('One target and one auxiliary are required.')
        self.marginal_models = [NoiseAwareGP([x], [y], [v], config=config, linear_mean=linear_mean) for x, y, v in zip(xs, ys, noise)]

    def marginal_theta(self, theta, index):
        return np.r_[theta[index], theta[2 + index], theta[4:4 + self.base.scales]]

    def nll_gradient(self, theta):
        if np.any(theta[4 + self.base.scales:] != 0):
            raise ValueError('DIAG correlations must be zero.')
        value, gradient = (0.0, np.zeros_like(theta))
        for i, model in enumerate(self.marginal_models):
            v, g = model.nll_gradient(self.marginal_theta(theta, i))
            value += v
            gradient[i], gradient[2 + i] = g[:2]
            gradient[4:4 + self.base.scales] += g[2:]
        return (float(value), gradient)

    def predict(self, theta, x, target=0):
        return self.marginal_models[target].predict(self.marginal_theta(theta, target), x)

def initial_from_marginals(marginals, model):
    natural = [f.model.base.unpack(f.theta) for f in marginals]
    ell = np.exp(np.average(np.log([p[2] for p in natural]), axis=0, weights=[len(f.model.y) for f in marginals]))
    return model.base.pack([p[0][0] for p in natural], [p[1][0] for p in natural], ell, np.eye(2))

def optimise(model, starts, free):
    started = perf_counter()
    config, bounds = (model.config, np.asarray(model._bounds))
    attempts, results = ([], [])
    for index, initial in enumerate(starts):
        start = np.clip(initial, bounds[:, 0] + 1e-07, bounds[:, 1] - 1e-07)
        bad = 0

        def objective(value):
            nonlocal bad
            theta = start.copy()
            theta[free] = value
            try:
                v, g = model.nll_gradient(theta)
                return (v, g[free])
            except (FloatingPointError, np.linalg.LinAlgError):
                bad += 1
                return (1e+100, np.zeros(len(free)))
        r = minimize(objective, start[free], jac=True, method='L-BFGS-B', bounds=bounds[free], options=dict(maxiter=config.maxiter, ftol=config.ftol, gtol=config.gtol, maxls=40))
        theta = start.copy()
        theta[free] = r.x
        success = bool(r.success and np.isfinite(r.fun) and (r.fun < 1e+99))
        attempts.append(dict(index=index, initial_theta=start.tolist(), success=success, nll=float(r.fun), iterations=int(r.nit), evaluations=int(r.nfev), nonfinite_evaluations=bad, message=str(r.message)))
        results.append((success, float(r.fun), theta))
    success, nll, theta = min([r for r in results if r[0]] or results, key=lambda r: r[1])
    active, norm = ([], float('nan'))
    try:
        gradient = model.nll_gradient(theta)[1][free]
        low = theta[free] - bounds[free, 0] <= 1e-05
        high = bounds[free, 1] - theta[free] <= 1e-05
        active = [model.names[j] for j in free[low | high]]
        gradient[low & (gradient > 0) | high & (gradient < 0)] = 0
        norm = float(np.max(np.abs(gradient)))
    except (FloatingPointError, np.linalg.LinAlgError):
        success = False
    return NoiseFit(model, theta, success, nll, perf_counter() - started, attempts, active, norm, False)

def fit_noise_diagonal(xs, ys, noise, marginals, config=FitConfig(), linear_mean=False):
    model = NoiseSharedGP(xs, ys, noise, config, linear_mean)
    initial = initial_from_marginals(marginals, model)
    starts = []
    for i in range(config.starts):
        start = initial.copy()
        start[4:4 + model.base.scales] += np.log((0.5, 1.0, 1.0, 2.0)[i])
        if i:
            start[2:4] = initial[:2] + np.log((0.0, 0.03, 0.1, 0.3)[i])
        starts.append(start)
    return optimise(model, starts, np.arange(4 + model.base.scales))

def fit_noise_coupled(xs, ys, noise, marginals, diagonal, config=FitConfig(), linear_mean=False):
    model = NoiseAwareGP(xs, ys, noise, config=config, linear_mean=linear_mean)
    initial = diagonal.theta if diagonal.success else initial_from_marginals(marginals, model)
    starts = []
    for i in range(config.starts):
        start = initial.copy()
        start[4:4 + model.base.scales] += np.log((0.5, 1.0, 1.0, 2.0)[i])
        if i:
            start[2:4] = initial[:2] + np.log((0.0, 0.03, 0.1, 0.3)[i])
        start[4 + model.base.scales:] = np.arctanh((-0.7, -0.2, 0.2, 0.7)[i])
        starts.append(start)
    return optimise(model, starts, np.arange(model.parameter_count))

def heterogeneous_gain(k11, k22, k12, own, auxiliary, variance, noise, rho):
    if not 0 <= abs(rho) < 1:
        raise ValueError('Correlation must lie strictly between -1 and 1.')
    variance = np.asarray(variance, float)
    if np.any(variance <= 0) or any((np.any(np.asarray(n) <= 0) for n in noise)):
        raise ValueError('Positive signal and total observation variances are required.')
    a = cho_factor(k11 + np.diag(noise[0] / variance[0]), lower=True)
    cross = cho_solve(a, k12)
    residual = auxiliary - own @ cross
    vind = variance[0] * (1 - np.einsum('ij,ji->i', own, cho_solve(a, own.T)))
    schur = k22 + np.diag(noise[1] / variance[1]) - rho ** 2 * k12.T @ cross
    factor = cho_factor((schur + schur.T) / 2, lower=True)
    gain = variance[0] * rho ** 2 * np.einsum('ij,ji->i', residual, cho_solve(factor, residual.T))
    tolerance = 1e-07 * max(variance[0], 1.0)
    if not np.all(np.isfinite(np.r_[vind, gain])) or min(vind.min(), gain.min(), (vind - gain).min()) < -tolerance:
        raise FloatingPointError('Invalid conditional variance or gain.')
    return (np.maximum(vind, 0), np.clip(gain, 0, np.maximum(vind, 0)))

def noise_opportunity(diagonal, evaluation, config=TaskScreenConfig()):
    started = perf_counter()
    model = diagonal.model
    signal, nugget, pilot, _ = model.base.unpack(diagonal.theta)
    xs, fit_config = (model.xs, model.config)
    flags, warnings, scenarios = ([], [], [])
    if not diagonal.success:
        flags.append('diagonal_fit_failed')
    if not np.all(np.isfinite(diagonal.theta)):
        flags.append('diagonal_nonfinite_parameters')
    if diagonal.active_bounds:
        warnings.append('diagonal_active_bound')
        if config.retain_boundary:
            flags.append('diagonal_active_bound')
    for i, x in enumerate(xs):
        if len(unique_sites(x)) < 2:
            flags.append(f'marginal_{i}_fewer_than_two_sites')
        try:
            marginal = model.marginal_models[i]
            _, jitter, _, _ = marginal.factor_and_mean(model.marginal_theta(diagonal.theta, i))
            if jitter > fit_config.jitter_ladder[0] * fit_config.jitter_scale:
                flags.append(f'marginal_{i}_jitter_escalated')
        except (FloatingPointError, np.linalg.LinAlgError):
            flags.append(f'marginal_{i}_factor_failed')
    noise = [n + v for n, v in zip(nugget, model.simulation_variance)]
    for multiplier in config.scale_multipliers:
        ell = multiplier * pilot

        def k(x, y):
            return kernel_and_logscale_derivatives(x, y, ell, fit_config.kernel, fit_config.ard)[0]
        try:
            vind, gain = heterogeneous_gain(k(xs[0], xs[0]), k(xs[1], xs[1]), k(xs[0], xs[1]), k(evaluation, xs[0]), k(evaluation, xs[1]), signal, noise, config.rho_max)
            if vind.mean() <= 1e-12 * signal[0]:
                raise FloatingPointError('Marginal variance denominator is too small.')
            scenarios.append(dict(multiplier=multiplier, lengthscale=ell.tolist(), opportunity=float(gain.mean() / vind.mean()), integrated_gain=float(gain.mean()), integrated_variance=float(vind.mean())))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            flags.append(f'scale_{multiplier}_unavailable')
    return dict(opportunity=max((s['opportunity'] for s in scenarios), default=float('nan')), flags=sorted(set(flags)), warnings=warnings, scenarios=scenarios, pilot_lengthscale=pilot.tolist(), bpi=pair_bpi(*xs, evaluation), settings=asdict(config), runtime_seconds=perf_counter() - started, convention='Exact conditional covariance gain with supplied site variances plus fitted nuggets; means held fixed.')
