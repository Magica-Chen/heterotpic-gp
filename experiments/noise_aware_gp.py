"""Noise aware gp for the multivariate kriging experiments."""
from dataclasses import asdict, dataclass
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from experiments.exact_gp import FitConfig, kernel_and_logscale_derivatives
from experiments.rich_gp import RichGP
from experiments.screening import ScreenConfig, pair_bpi
from experiments.diagnostics import correlation_information

class NoiseAwareGP(RichGP):

    def __init__(self, xs, ys, simulation_variance, kind='separable', config=FitConfig(), linear_mean=False):
        super().__init__(xs, ys, kind, config, linear_mean)
        self.simulation_variance = [np.asarray(v, float) for v in simulation_variance]
        if len(simulation_variance) != len(xs) or any((v.shape != (len(x),) for v, x in zip(self.simulation_variance, xs))):
            raise ValueError('One simulation variance per training response is required.')
        self.known_noise = np.concatenate(self.simulation_variance)
        if not np.all(np.isfinite(self.known_noise)) or np.min(self.known_noise) < 0:
            raise ValueError('Finite nonnegative simulation variances required.')

    def covariance(self, theta, derivatives=False):
        value = super().covariance(theta, derivatives)
        if derivatives:
            return (value[0] + np.diag(self.known_noise), value[1])
        return value + np.diag(self.known_noise)

    def mean_basis(self, x, ids):
        columns = x.shape[1] + 1
        h = np.zeros((len(x), columns * self.outputs))
        for j in range(self.outputs):
            h[:, j * columns:(j + 1) * columns] = (ids == j)[:, None] * np.c_[np.ones(len(x)), x]
        return h

@dataclass
class NoiseFit:
    model: NoiseAwareGP
    theta: np.ndarray
    success: bool
    nll: float
    runtime_seconds: float
    starts: list
    active_bounds: list
    gradient_norm: float
    fixed_working: bool

    def predict(self, x, target=0, simulation_variance=None):
        if self.success:
            mean, latent, observed = self.model.predict(self.theta, x, target)
        else:
            y = self.model.ys[target]
            v = max(float(np.var(y)), 1e-10)
            mean, latent, observed = (np.full(len(x), np.mean(y)), np.full(len(x), v), np.full(len(x), v))
        if simulation_variance is not None:
            observed = observed + np.asarray(simulation_variance)
        return (mean, latent, observed)

    def record(self):
        jitter = None
        beta = None
        try:
            _, jitter, _, terms = self.model.factor_and_mean(self.theta)
            if terms is not None:
                beta = terms[0].tolist()
        except np.linalg.LinAlgError:
            pass
        return {'kind': self.model.kind, 'config': asdict(self.model.config), 'linear_mean': self.model.linear_mean, 'simulation_variance': [v.tolist() for v in self.model.simulation_variance], 'theta': self.theta.tolist(), 'parameter_names': self.model.names, 'success': self.success, 'nll': self.nll, 'runtime_seconds': self.runtime_seconds, 'starts': self.starts, 'active_bounds': self.active_bounds, 'projected_gradient_norm': self.gradient_norm, 'jitter': jitter, 'mean_coefficients': beta, 'fixed_working_latent_covariance': self.fixed_working}

def fit_noise_gp(xs, ys, simulation_variance, kind='separable', config=FitConfig(), linear_mean=False, fixed_working=False):
    start_time = perf_counter()
    model = NoiseAwareGP(xs, ys, simulation_variance, kind, config, linear_mean)
    starts = model.starts()
    free = np.arange(model.parameter_count)
    if fixed_working:
        if kind != 'separable':
            raise ValueError('Fixed working covariance is separable.')
        d = model.outputs
        free = np.r_[np.arange(d, 2 * d), np.arange(2 * d + model.base.scales, model.parameter_count)]
        for start in starts:
            start[:d] = 0.0
            start[2 * d:2 * d + model.base.scales] = np.log(0.11)
    bounds = np.array(model._bounds)
    attempts = []
    results = []
    for i, start in enumerate(starts):
        bad = 0

        def objective(t):
            nonlocal bad
            theta = start.copy()
            theta[free] = t
            try:
                value, gradient = model.nll_gradient(theta)
                return (value, gradient[free])
            except (FloatingPointError, np.linalg.LinAlgError):
                bad += 1
                return (1e+100, np.zeros(len(free)))
        r = minimize(objective, start[free], jac=True, method='L-BFGS-B', bounds=bounds[free], options={'maxiter': config.maxiter, 'ftol': config.ftol, 'gtol': config.gtol, 'maxls': 40})
        theta = start.copy()
        theta[free] = r.x
        good = bool(r.success and np.isfinite(r.fun) and (r.fun < 1e+99))
        attempts.append({'index': i, 'initial_theta': start.tolist(), 'success': good, 'nll': float(r.fun), 'iterations': int(r.nit), 'evaluations': int(r.nfev), 'nonfinite_evaluations': bad, 'message': str(r.message)})
        results.append((good, float(r.fun), theta))
    success, nll, theta = min([r for r in results if r[0]] or results, key=lambda r: r[1])
    active = []
    norm = float('nan')
    try:
        gradient = model.nll_gradient(theta)[1][free]
        low = theta[free] - bounds[free, 0] <= 1e-05
        high = bounds[free, 1] - theta[free] <= 1e-05
        active = [model.names[j] for j in free[low | high]]
        gradient[low & (gradient > 0) | high & (gradient < 0)] = 0
        norm = float(np.max(np.abs(gradient)))
    except (FloatingPointError, np.linalg.LinAlgError):
        success = False
    return NoiseFit(model, theta, success, nll, perf_counter() - start_time, attempts, active, norm, fixed_working)

def pair_noise_scores(marginals, evaluation, screen=ScreenConfig()):
    started = perf_counter()
    flags = []
    if len(marginals) != 2 or any((m.model.kind != 'separable' for m in marginals)):
        raise ValueError('Scores require two separable marginal pilots.')
    for j, fit in enumerate(marginals):
        if not fit.success:
            flags.append(f'marginal_{j}_fit_failed')
        if fit.active_bounds:
            flags.append(f'marginal_{j}_active_bound')
    natural = [f.model.base.unpack(f.theta) for f in marginals]
    xs = [f.model.xs[0] for f in marginals]
    scales = np.array([p[2] for p in natural])
    ell = np.exp(np.average(np.log(scales), axis=0, weights=[len(x) for x in xs]))
    if np.any(scales.max(axis=0) / scales.min(axis=0) > screen.marginal_scale_ratio_limit):
        flags.append('marginal_lengthscale_ratio_above_limit')
    signal = np.array([p[0][0] for p in natural])
    noise = [p[1][0] + f.model.simulation_variance[0] for p, f in zip(natural, marginals)]
    config = marginals[0].model.config
    scenarios = []
    for multiplier in screen.scale_multipliers:
        try:

            def k(x, y):
                return kernel_and_logscale_derivatives(x, y, ell * multiplier, config.kernel, config.ard)[0]
            x, y = xs
            k11, k22, k12 = (k(x, x), k(y, y), k(x, y))
            own, aux = (k(evaluation, x), k(evaluation, y))
            factor = cho_factor(k11 + np.diag(noise[0] / signal[0]), lower=True)
            residual = aux - own @ cho_solve(factor, k12)
            vind = np.maximum(signal[0] * (1 - np.sum(own * cho_solve(factor, own.T).T, axis=1)), 0)
            upper = signal[0] * signal[1] * screen.rho_max ** 2 * np.sum(residual ** 2, axis=1) / np.min(noise[1])
            opportunity = float(np.mean(np.minimum(vind, upper) / np.maximum(vind, screen.variance_floor)))
            info = correlation_information(k12, signal[0] * k11 + np.diag(noise[0]), signal[1] * k22 + np.diag(noise[1]), *signal)
            scenarios.append({'multiplier': multiplier, 'lengthscale': (ell * multiplier).tolist(), 'opportunity': opportunity, 'information': info, 'mass': float(np.sum(k12 * k12)), 'average_mass': float(np.mean(k12 * k12))})
        except (FloatingPointError, np.linalg.LinAlgError):
            flags.append(f'scale_{multiplier}_solve_failed')
    return {'opportunity': max((s['opportunity'] for s in scenarios), default=float('nan')), 'information': max((s['information'] for s in scenarios), default=float('nan')), 'flags': flags, 'scenarios': scenarios, 'pilot_lengthscale': ell.tolist(), 'bpi': pair_bpi(*xs, evaluation), 'runtime_seconds': perf_counter() - started, 'noise_convention': 'Estimated site-mean simulation variance plus fitted additional nugget; residual upper bound uses minimum auxiliary diagonal noise.', 'settings': asdict(screen)}
