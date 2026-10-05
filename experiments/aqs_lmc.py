"""Aqs lmc for the multivariate kriging experiments."""
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from experiments.exact_gp import FitConfig, correlation_cholesky, correlation_coordinates
from experiments.rich_gp import RichGP, RichFit

class TwoComponentLMC(RichGP):

    def __init__(self, xs, ys, config=FitConfig()):
        super().__init__(xs, ys, 'separable', config)
        self.kind = 'two_full_component_lmc'
        d = self.outputs
        self.nc = d * (d - 1) // 2
        self.component_width = d + self.nc
        zmax = np.arctanh(config.partial_correlation_limit)
        vb = tuple(np.log(np.array(config.signal_variance_bounds) / 2))
        self.names = []
        self._bounds = []
        for q in range(2):
            self.names.extend([f'component{q}_log_variance{j}' for j in range(d)] + [f'component{q}_partial_corr{j}' for j in range(self.nc)])
            self._bounds.extend([vb] * d + [(-zmax, zmax)] * self.nc)
        self.names.extend(['log_scale0', 'log_scale1'] + [f'log_noise{j}' for j in range(d)])
        self._bounds.extend([tuple(np.log(config.lengthscale_bounds))] * 2 + [tuple(np.log(config.noise_variance_bounds))] * d)
        self.noise_start = 2 * self.component_width + 2
        self.parameter_count = len(self.names)

    def components(self, theta):
        result = []
        d = self.outputs
        for q in range(2):
            offset = q * self.component_width
            v = np.exp(theta[offset:offset + d])
            sd = np.sqrt(v[:, None] * v[None, :])
            r, dr = correlation_cholesky(theta[offset + d:offset + self.component_width], d, True)
            b = sd * r
            derivatives = {}
            for j in range(d):
                indicator = (np.arange(d) == j).astype(float)
                derivatives[offset + j] = b * (indicator[:, None] + indicator[None, :]) / 2
            for j, value in enumerate(dr):
                derivatives[offset + d + j] = sd * value
            index = 2 * self.component_width + q
            result.append((b, np.exp(theta[index]), derivatives, index))
        return result

    def starts(self):
        d = self.outputs
        v = np.clip([np.mean(y * y) for y in self.ys], 0.05, 5.0)
        starts = []
        for i, (short, long) in enumerate(((0.04, 0.2), (0.08, 0.4), (0.15, 0.8), (0.3, 1.5))):
            values = []
            for q, rho in enumerate(((-0.7, -0.2, 0.2, 0.7)[i], (0.2, 0.7, -0.7, -0.2)[i])):
                loading = np.full(d, np.sqrt(abs(rho)))
                loading[0] *= np.sign(rho)
                r = np.outer(loading, loading) + np.diag(1 - loading ** 2)
                values.extend([*np.log(v / 2), *correlation_coordinates(r)])
            noise = np.clip(v * (0.03, 0.1, 0.3, 0.1)[i], 0.001, 0.5)
            values.extend([np.log(short), np.log(long), *np.log(noise)])
            bounds = np.array(self._bounds)
            starts.append(np.clip(values, bounds[:, 0] + 1e-07, bounds[:, 1] - 1e-07))
        return starts

def fit_lmc(xs, ys, config=FitConfig()):
    started = perf_counter()
    model = TwoComponentLMC(xs, ys, config)
    results = []
    attempts = []
    for i, start in enumerate(model.starts()):
        bad = 0

        def objective(theta):
            nonlocal bad
            try:
                return model.nll_gradient(theta)
            except (np.linalg.LinAlgError, FloatingPointError):
                bad += 1
                return (1e+100, np.zeros(len(theta)))
        r = minimize(objective, start, jac=True, method='L-BFGS-B', bounds=model._bounds, options={'maxiter': config.maxiter, 'ftol': config.ftol, 'gtol': config.gtol, 'maxls': 40})
        success = bool(r.success and np.isfinite(r.fun) and (r.fun < 1e+99))
        attempts.append({'index': i, 'initial_theta': start.tolist(), 'success': success, 'nll': float(r.fun), 'message': str(r.message), 'iterations': int(r.nit), 'evaluations': int(r.nfev), 'nonfinite_evaluations': bad})
        results.append((success, float(r.fun), r.x))
    success, nll, theta = min([r for r in results if r[0]] or results, key=lambda r: r[1])
    active = []
    norm = float('nan')
    try:
        g = model.nll_gradient(theta)[1]
        bounds = np.array(model._bounds)
        lo = theta - bounds[:, 0] <= 1e-05
        hi = bounds[:, 1] - theta <= 1e-05
        active = [model.names[i] for i in np.flatnonzero(lo | hi)]
        g[lo & (g > 0) | hi & (g < 0)] = 0.0
        norm = float(np.max(abs(g)))
    except (np.linalg.LinAlgError, FloatingPointError):
        success = False
    return RichFit(model, theta, success, nll, perf_counter() - started, attempts, active, norm)
