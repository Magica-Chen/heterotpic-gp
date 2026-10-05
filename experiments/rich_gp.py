"""Rich gp for the multivariate kriging experiments."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from experiments.exact_gp import ExactGP, FitConfig, kernel_and_logscale_derivatives

class RichGP:

    def __init__(self, xs, ys, kind, config=FitConfig(), linear_mean=False, noise_shape=None):
        self.base = ExactGP(xs, ys, config)
        self.xs, self.ys, self.x, self.y, self.output_ids = (self.base.xs, self.base.ys, self.base.x, self.base.y, self.base.output_ids)
        self.outputs, self.config, self.kind = (self.base.outputs, config, kind)
        self.linear_mean = linear_mean
        self.noise_shape = noise_shape
        if kind == 'lmc':
            self.names = ['log_target_loading_0', 'aux_loading_0', 'log_target_loading_1', 'aux_loading_1', 'log_lengthscale_0', 'log_lengthscale_1', 'log_noise_0', 'log_noise_1']
            self.noise_start = 6
            self._bounds = [(np.log(0.001), np.log(5)), (-5.0, 5.0)] * 2 + [tuple(np.log(config.lengthscale_bounds))] * 2 + [tuple(np.log(config.noise_variance_bounds))] * 2
        elif kind == 'ar':
            self.names = ['log_low_variance', 'log_discrepancy_variance', 'beta', 'log_low_scale', 'log_discrepancy_scale', 'log_noise_high', 'log_noise_low']
            self.noise_start = 5
            self._bounds = [tuple(np.log(config.signal_variance_bounds))] * 2 + [(-2.0, 2.0)] + [tuple(np.log(config.lengthscale_bounds))] * 2 + [tuple(np.log(config.noise_variance_bounds))] * 2
        elif kind == 'mixture':
            self.names = ['log_variance_0', 'log_variance_1', 'log_scale_0', 'log_scale_1', 'log_noise']
            self.noise_start = 4
            self._bounds = [tuple(np.log(config.signal_variance_bounds))] * 2 + [tuple(np.log(config.lengthscale_bounds))] * 2 + [tuple(np.log(config.noise_variance_bounds))]
        elif kind == 'separable':
            self.names = self.base.names
            self._bounds = self.base.bounds()
            self.noise_start = self.outputs
        else:
            raise ValueError(kind)
        self.parameter_count = len(self.names)

    def components(self, theta):
        if self.kind == 'lmc':
            result = []
            for k in range(2):
                loading = np.array([np.exp(theta[2 * k]), theta[2 * k + 1]])
                b = np.outer(loading, loading)
                a = np.array([loading[0], 0.0])
                c = np.array([0.0, 1.0])
                result.append((b, np.exp(theta[4 + k]), {2 * k: np.outer(a, loading) + np.outer(loading, a), 2 * k + 1: np.outer(c, loading) + np.outer(loading, c)}, 4 + k))
            return result
        if self.kind == 'ar':
            v, delta = np.exp(theta[:2])
            beta = theta[2]
            b = v * np.array([[beta ** 2, beta], [beta, 1.0]])
            db = v * np.array([[2 * beta, 1.0], [1.0, 0.0]])
            c = np.array([[delta, 0.0], [0.0, 0.0]])
            return [(b, np.exp(theta[3]), {0: b, 2: db}, 3), (c, np.exp(theta[4]), {1: c}, 4)]
        if self.kind == 'mixture':
            return [(np.array([[np.exp(theta[k])]]), np.exp(theta[2 + k]), {k: np.array([[np.exp(theta[k])]])}, 2 + k) for k in range(2)]
        raise ValueError('Separable covariance uses the base implementation.')

    def noise_profile(self, x, ids):
        if self.noise_shape is None:
            return np.ones(len(x))
        return np.array([0.01 + 0.19 * point[0] ** 2 if self.noise_shape[int(j)] == 'quadratic' else 0.02 for point, j in zip(x, ids)])

    def covariance(self, theta, derivatives=False):
        ids = self.output_ids
        if self.kind == 'separable':
            value = self.base.covariance(theta, derivatives)
            covariance, dc = value if derivatives else (value, None)
            noise = np.exp(theta[self.outputs:2 * self.outputs])
            profile = self.noise_profile(self.x, ids)
            covariance = covariance + np.diag(noise[ids] * (profile - 1))
            if derivatives:
                for j in range(self.outputs):
                    dc[self.outputs + j] = np.diag(noise[ids] * profile * (ids == j))
        else:
            covariance = np.zeros((len(ids), len(ids)))
            dc = np.zeros((self.parameter_count, len(ids), len(ids))) if derivatives else None
            for b, ell, db, index in self.components(theta):
                k, dk = kernel_and_logscale_derivatives(self.x, self.x, np.array([ell]), self.config.kernel, False)
                block = b[ids[:, None], ids[None, :]]
                covariance += block * k
                if derivatives:
                    for parameter, derivative in db.items():
                        dc[parameter] += derivative[ids[:, None], ids[None, :]] * k
                    dc[index] += block * dk[0]
            noise = np.exp(theta[self.noise_start:self.noise_start + self.outputs])
            covariance += np.diag(noise[ids])
            if derivatives:
                for j in range(self.outputs):
                    dc[self.noise_start + j] = np.diag(noise[ids] * (ids == j))
        return (covariance, dc) if derivatives else covariance

    def mean_basis(self, x, ids):
        h = np.zeros((len(x), 2 * self.outputs))
        for j in range(self.outputs):
            h[:, 2 * j] = ids == j
            h[:, 2 * j + 1] = (ids == j) * x[:, 0]
        return h

    def factor_and_mean(self, theta, covariance=None):
        covariance = self.covariance(theta) if covariance is None else covariance
        factor, jitter = self.base.factor(covariance)
        if self.linear_mean:
            h = self.mean_basis(self.x, self.output_ids)
            ih = cho_solve(factor, h)
            normal = h.T @ ih
            beta = cho_solve(cho_factor(normal, lower=True), h.T @ cho_solve(factor, self.y))
            residual = self.y - h @ beta
            return (factor, jitter, residual, (beta, h, ih, normal))
        return (factor, jitter, self.y, None)

    def nll_gradient(self, theta):
        covariance, dc = self.covariance(theta, True)
        factor, _, residual, _ = self.factor_and_mean(theta, covariance)
        alpha = cho_solve(factor, residual)
        value = 0.5 * (residual @ alpha + 2 * np.log(np.diag(factor[0])).sum() + len(residual) * np.log(2 * np.pi))
        score = cho_solve(factor, np.eye(len(residual))) - np.outer(alpha, alpha)
        gradient = 0.5 * np.einsum('ij,pij->p', score, dc, optimize=False)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError('Non-finite likelihood')
        return (float(value), gradient)

    def predict(self, theta, x, target=0):
        x = np.asarray(x, float)
        if self.kind == 'separable':
            v, noise, ell, r = self.base.unpack(theta)
            lam = np.sqrt(v[:, None] * v[None, :]) * r
            k = kernel_and_logscale_derivatives(x, self.x, ell, self.config.kernel, self.config.ard)[0]
            cross = k * lam[target, self.output_ids]
            prior = v[target]
        else:
            cross = np.zeros((len(x), len(self.y)))
            prior = 0.0
            for b, ell, _, _ in self.components(theta):
                cross += b[target, self.output_ids] * kernel_and_logscale_derivatives(x, self.x, np.array([ell]), self.config.kernel, False)[0]
                prior += b[target, target]
            noise = np.exp(theta[self.noise_start:self.noise_start + self.outputs])
        factor, _, residual, mean_terms = self.factor_and_mean(theta)
        weights = cho_solve(factor, cross.T).T
        mean = weights @ residual
        variance = prior - np.sum(weights * cross, axis=1)
        if mean_terms is not None:
            beta, h, ih, normal = mean_terms
            hx = self.mean_basis(x, np.full(len(x), target))
            mean += hx @ beta
            q = hx - cross @ ih
            variance += np.sum(q * cho_solve(cho_factor(normal, lower=True), q.T).T, axis=1)
        if variance.min() < -1e-07 * max(1, prior):
            raise FloatingPointError('Negative predictive variance')
        variance = np.maximum(variance, 0)
        observation_noise = noise[target] * self.noise_profile(x, np.full(len(x), target))
        return (mean, variance, variance + observation_noise)

    def starts(self):
        centred = [y - np.mean(y) if self.linear_mean else y for y in self.ys]
        v = np.clip([np.mean(y * y) for y in centred], 0.05, 5.0)
        starts = []
        for i, (short, long) in enumerate(((0.04, 0.2), (0.08, 0.4), (0.15, 0.8), (0.3, 1.5))):
            noise = np.clip(v * (0.03, 0.1, 0.3, 0.1)[i], 0.001, 0.5)
            if self.kind == 'lmc':
                signs = ((-1, 1), (1, -1), (1, 1), (1, 1))[i]
                values = [np.log(np.sqrt(0.5 * v[0])), signs[0] * np.sqrt(0.5 * v[1]), np.log(np.sqrt(0.5 * v[0])), signs[1] * np.sqrt(0.5 * v[1]), np.log(short), np.log(long), *np.log(noise)]
            elif self.kind == 'ar':
                values = [np.log(v[1]), np.log(0.5 * v[0]), (-0.7, -0.2, 0.2, 0.7)[i], np.log(long), np.log(short), *np.log(noise)]
            elif self.kind == 'mixture':
                values = [np.log(0.5 * v[0]), np.log(0.5 * v[0]), np.log(short), np.log(long), np.log(noise[0])]
            else:
                r = np.eye(self.outputs)
                if self.outputs == 2:
                    r[0, 1] = r[1, 0] = (-0.7, -0.2, 0.2, 0.7)[i]
                if self.noise_shape is not None:
                    profile = self.noise_profile(self.x, self.output_ids)
                    noise = np.array([noise[j] / profile[self.output_ids == j].mean() for j in range(self.outputs)])
                values = self.base.pack(v, noise, (0.05, 0.15, 0.5, 1.5)[i], r)
            bounds = np.array(self._bounds)
            starts.append(np.clip(values, bounds[:, 0] + 1e-07, bounds[:, 1] - 1e-07))
        return starts

@dataclass
class RichFit:
    model: RichGP
    theta: np.ndarray
    success: bool
    nll: float
    runtime_seconds: float
    starts: list
    active_bounds: list
    gradient_norm: float

    def predict(self, x, target=0):
        if self.success:
            return self.model.predict(self.theta, x, target)
        y = self.model.ys[target]
        variance = max(float(np.var(y)), 1e-10)
        return (np.full(len(x), np.mean(y)), np.full(len(x), variance), np.full(len(x), variance))

    def record(self):
        factor, jitter, residual, terms = self.model.factor_and_mean(self.theta)
        return {'kind': self.model.kind, 'theta': self.theta.tolist(), 'parameter_names': self.model.names, 'success': self.success, 'nll': self.nll, 'runtime_seconds': self.runtime_seconds, 'starts': self.starts, 'active_bounds': self.active_bounds, 'projected_gradient_norm': self.gradient_norm, 'jitter': jitter, 'linear_mean': self.model.linear_mean, 'mean_coefficients': terms[0].tolist() if terms else None, 'noise_shape': self.model.noise_shape, 'config': asdict(self.model.config)}

def fit_rich(xs, ys, kind, config=FitConfig(), linear_mean=False, noise_shape=None):
    started = perf_counter()
    model = RichGP(xs, ys, kind, config, linear_mean, noise_shape)
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
        result = minimize(objective, start, jac=True, method='L-BFGS-B', bounds=model._bounds, options={'maxiter': config.maxiter, 'ftol': config.ftol, 'gtol': config.gtol, 'maxls': 40})
        success = bool(result.success and np.isfinite(result.fun) and (result.fun < 1e+99))
        attempts.append({'index': i, 'initial_theta': start.tolist(), 'success': success, 'nll': float(result.fun), 'message': str(result.message), 'iterations': int(result.nit), 'nonfinite_evaluations': bad})
        results.append((success, float(result.fun), result.x))
    best = min([r for r in results if r[0]] or results, key=lambda r: r[1])
    success, nll, theta = best
    bounds = np.array(model._bounds)
    gradient = model.nll_gradient(theta)[1]
    low = theta - bounds[:, 0] <= 1e-05
    high = bounds[:, 1] - theta <= 1e-05
    gradient[low & (gradient > 0) | high & (gradient < 0)] = 0
    active = [model.names[i] for i in np.flatnonzero(low | high)]
    return RichFit(model, theta, success, nll, perf_counter() - started, attempts, active, float(abs(gradient).max()))
