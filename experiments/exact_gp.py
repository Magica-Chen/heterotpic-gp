"""Exact gp for the multivariate kriging experiments."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

@dataclass(frozen=True)
class FitConfig:
    kernel: str = 'matern32'
    ard: bool = False
    maxiter: int = 1000
    ftol: float = 1e-08
    gtol: float = 1e-05
    lengthscale_bounds: tuple = (0.01, 3.0)
    signal_variance_bounds: tuple = (0.0025, 25.0)
    noise_variance_bounds: tuple = (1e-06, 2.0)
    partial_correlation_limit: float = 0.995
    jitter_scale: float = 1.0
    jitter_ladder: tuple = (1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05)
    starts: int = 4

def correlation_cholesky(z, outputs, derivatives=False):
    """Map row-wise partial correlations to a positive-definite correlation."""
    z = np.asarray(z, dtype=float)
    if len(z) != outputs * (outputs - 1) // 2:
        raise ValueError('Incorrect partial-correlation coordinate count.')
    l = np.zeros((outputs, outputs))
    dl = []
    cursor = 0
    for i in range(outputs):
        partial = np.tanh(z[cursor:cursor + i])
        product = 1.0
        prefixes = []
        for j in range(i):
            prefixes.append(product)
            l[i, j] = partial[j] * product
            product *= np.sqrt(1 - partial[j] ** 2)
        l[i, i] = product
        if derivatives:
            for j in range(i):
                d = np.zeros_like(l)
                d[i, j] = (1 - partial[j] ** 2) * prefixes[j]
                d[i, j + 1:i + 1] = -partial[j] * l[i, j + 1:i + 1]
                dl.append(d)
        cursor += i
    r = l @ l.T
    return (r, [d @ l.T + l @ d.T for d in dl]) if derivatives else r

def correlation_coordinates(r):
    l = np.linalg.cholesky(r)
    coords = []
    for i in range(len(r)):
        product = 1.0
        for j in range(i):
            p = l[i, j] / product
            if abs(p) >= 1:
                raise ValueError('Correlation matrix must be interior positive definite.')
            coords.append(np.arctanh(p))
            product *= np.sqrt(1 - p * p)
    return np.array(coords)

def kernel_and_logscale_derivatives(x, y, ell, family, ard):
    scaled2 = ((x[:, None, :] - y[None, :, :]) / ell) ** 2
    r2 = np.sum(scaled2, axis=-1)
    r = np.sqrt(r2)
    if family == 'rbf':
        k = np.exp(-r2 / 2)
        coefficient = k
    elif family == 'matern32':
        z = np.sqrt(3) * r
        k = (1 + z) * np.exp(-z)
        coefficient = 3 * np.exp(-z)
    elif family == 'matern52':
        z = np.sqrt(5) * r
        k = (1 + z + z * z / 3) * np.exp(-z)
        coefficient = 5 / 3 * (1 + z) * np.exp(-z)
    else:
        raise ValueError(f'Unknown kernel: {family}')
    d = coefficient[None, :, :] * (scaled2.transpose(2, 0, 1) if ard else r2[None, :, :])
    return (k, d)

class ExactGP:

    def __init__(self, xs, ys, config=FitConfig()):
        self.xs = [np.asarray(x, dtype=float).reshape(len(x), -1) for x in xs]
        self.ys = [np.asarray(y, dtype=float).reshape(-1) for y in ys]
        if not self.xs or any((len(x) == 0 for x in self.xs)):
            raise ValueError('Each output requires at least one observation.')
        if any((len(x) != len(y) for x, y in zip(self.xs, self.ys))) or len(xs) != len(ys):
            raise ValueError('Output observation dimensions disagree.')
        self.x, self.y = (np.concatenate(self.xs), np.concatenate(self.ys))
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.y)):
            raise ValueError('Inputs and responses must be finite.')
        self.config = config
        self.outputs, self.dimension = (len(xs), self.x.shape[1])
        self.output_ids = np.repeat(np.arange(self.outputs), [len(x) for x in self.xs])
        self.scales = self.dimension if config.ard else 1
        self.correlations = self.outputs * (self.outputs - 1) // 2
        self.parameter_count = 2 * self.outputs + self.scales + self.correlations
        self.names = [f'log_signal_variance_{j}' for j in range(self.outputs)] + [f'log_noise_variance_{j}' for j in range(self.outputs)] + [f'log_lengthscale_{t}' for t in range(self.scales)] + [f'partial_z_{i}_{j}' for i in range(self.outputs) for j in range(i)]

    def pack(self, signal_variance, noise_variance, lengthscale, correlation):
        ell = np.broadcast_to(np.atleast_1d(lengthscale), (self.scales,))
        return np.r_[np.log(signal_variance), np.log(noise_variance), np.log(ell), correlation_coordinates(correlation)]

    def unpack(self, theta):
        d = self.outputs
        variance, noise = (np.exp(theta[:d]), np.exp(theta[d:2 * d]))
        ell = np.exp(theta[2 * d:2 * d + self.scales])
        r = correlation_cholesky(theta[2 * d + self.scales:], d)
        return (variance, noise, ell, r)

    def covariance(self, theta, derivatives=False):
        d, ids = (self.outputs, self.output_ids)
        v, noise, ell, r = self.unpack(theta)
        k, dk = kernel_and_logscale_derivatives(self.x, self.x, ell, self.config.kernel, self.config.ard)
        lam = np.sqrt(v[:, None] * v[None, :]) * r
        observation_lam = lam[ids[:, None], ids[None, :]]
        signal = observation_lam * k
        sigma = signal + np.diag(noise[ids])
        if not derivatives:
            return sigma
        dc = []
        for j in range(d):
            mask = (ids == j).astype(float)
            dc.append(signal * (mask[:, None] + mask[None, :]) / 2)
        for j in range(d):
            dc.append(np.diag(noise[ids] * (ids == j)))
        dc.extend((observation_lam * deriv for deriv in dk))
        _, drs = correlation_cholesky(theta[2 * d + self.scales:], d, derivatives=True)
        for dr in drs:
            dlambda = np.sqrt(v[:, None] * v[None, :]) * dr
            dc.append(dlambda[ids[:, None], ids[None, :]] * k)
        return (sigma, np.asarray(dc))

    def factor(self, sigma):
        for relative in self.config.jitter_ladder:
            jitter = relative * self.config.jitter_scale
            try:
                return (cho_factor(sigma + jitter * np.eye(len(sigma)), lower=True, check_finite=False), jitter)
            except np.linalg.LinAlgError:
                pass
        raise np.linalg.LinAlgError('Cholesky failed across the declared jitter ladder.')

    def nll_gradient(self, theta):
        sigma, dc = self.covariance(theta, derivatives=True)
        factor, _ = self.factor(sigma)
        alpha = cho_solve(factor, self.y, check_finite=False)
        value = 0.5 * (self.y @ alpha + 2 * np.log(np.diag(factor[0])).sum() + len(self.y) * np.log(2 * np.pi))
        q = cho_solve(factor, np.eye(len(sigma)), check_finite=False) - np.outer(alpha, alpha)
        grad = 0.5 * np.einsum('ij,pij->p', q, dc, optimize=False)
        if not np.isfinite(value) or not np.all(np.isfinite(grad)):
            raise FloatingPointError('Non-finite exact likelihood or gradient.')
        return (float(value), grad)

    def predict(self, theta, x, target=0, full_covariance=False):
        x = np.asarray(x, dtype=float).reshape(len(x), self.dimension)
        v, noise, ell, r = self.unpack(theta)
        lam = np.sqrt(v[:, None] * v[None, :]) * r
        k, _ = kernel_and_logscale_derivatives(x, self.x, ell, self.config.kernel, self.config.ard)
        c = k * lam[target, self.output_ids][None, :]
        factor, _ = self.factor(self.covariance(theta))
        solved = cho_solve(factor, c.T, check_finite=False)
        mean = c @ cho_solve(factor, self.y, check_finite=False)
        variance = v[target] - np.sum(c * solved.T, axis=1)
        if np.min(variance) < -1e-07 * max(v[target], 1):
            raise FloatingPointError('Materially negative posterior variance.')
        variance = np.maximum(variance, 0)
        if full_covariance:
            kxx, _ = kernel_and_logscale_derivatives(x, x, ell, self.config.kernel, self.config.ard)
            return (mean, (v[target] * kxx - c @ solved + (v[target] * kxx - c @ solved).T) / 2)
        return (mean, variance, variance + noise[target])

    def prediction_gradient(self, theta, x, target=0):
        x = np.asarray(x, float).reshape(len(x), self.dimension)
        v, _, ell, r = self.unpack(theta)
        lam = np.sqrt(v[:, None] * v[None, :]) * r
        k, dk = kernel_and_logscale_derivatives(x, self.x, ell, self.config.kernel, self.config.ard)
        c = k * lam[target, self.output_ids]
        dcross = []
        for j in range(self.outputs):
            dcross.append(c * ((target == j) + (self.output_ids == j).astype(float))[None, :] / 2)
        dcross.extend((np.zeros_like(c) for _ in range(self.outputs)))
        dcross.extend((a * lam[target, self.output_ids] for a in dk))
        _, drs = correlation_cholesky(theta[2 * self.outputs + self.scales:], self.outputs, derivatives=True)
        for dr in drs:
            dl = np.sqrt(v[:, None] * v[None, :]) * dr
            dcross.append(k * dl[target, self.output_ids])
        sigma, derivatives = self.covariance(theta, derivatives=True)
        factor, _ = self.factor(sigma)
        alpha = cho_solve(factor, self.y, check_finite=False)
        weights = cho_solve(factor, c.T, check_finite=False).T
        return np.asarray([a @ alpha - weights @ (b @ alpha) for a, b in zip(dcross, derivatives)]).T

    def bounds(self):
        c = self.config
        zmax = np.arctanh(c.partial_correlation_limit)
        return [tuple(np.log(c.signal_variance_bounds))] * self.outputs + [tuple(np.log(c.noise_variance_bounds))] * self.outputs + [tuple(np.log(c.lengthscale_bounds))] * self.scales + [(-zmax, zmax)] * self.correlations

@dataclass
class FitResult:
    model: ExactGP
    theta: np.ndarray
    success: bool
    nll: float
    runtime_seconds: float
    jitter: float
    active_bounds: list
    gradient_norm: float
    starts: list = field(default_factory=list)
    mode: str = 'full'

    def predict(self, x, target=0):
        if self.success:
            return self.model.predict(self.theta, x, target)
        y = self.model.ys[target]
        variance = max(float(np.var(y)), 1e-10)
        return (np.full(len(x), y.mean()), np.full(len(x), variance), np.full(len(x), variance))

    def record(self):
        v, noise, ell, r = self.model.unpack(self.theta)
        return {'success': self.success, 'nll': self.nll, 'runtime_seconds': self.runtime_seconds, 'jitter': self.jitter, 'active_bounds': self.active_bounds, 'projected_gradient_norm': self.gradient_norm, 'starts': self.starts, 'theta': self.theta.tolist(), 'parameter_names': self.model.names, 'signal_variance': v.tolist(), 'noise_variance': noise.tolist(), 'lengthscale': ell.tolist(), 'correlation': r.tolist(), 'mode': self.mode, 'fitting': 'exact_float64_known_zero_mean', 'config': asdict(self.model.config)}

def fit_gp(xs, ys, config=FitConfig(), initial=None, fixed_theta=None):
    """Fit from a fixed list of complete starts; fixed_theta enables correlation-only fits."""
    started = perf_counter()
    if not 1 <= config.starts <= 4:
        raise ValueError('Choose one through four entries from the fixed complete-start list.')
    model = ExactGP(xs, ys, config)
    d = model.outputs
    bounds = np.array(model.bounds())
    if fixed_theta is None:
        v = np.clip([np.mean(y * y) for y in model.ys], 0.05, 5)
        initial = model.pack(v, np.clip(0.1 * v, 0.001, 0.5), 0.2, np.eye(d)) if initial is None else np.array(initial)
        free = np.arange(model.parameter_count)
    else:
        initial = np.asarray(fixed_theta).copy()
        free = np.arange(2 * d + model.scales, model.parameter_count)
    if len(free) == 0:
        value, _ = model.nll_gradient(initial)
        _, jitter = model.factor(model.covariance(initial))
        return FitResult(model, initial, True, value, perf_counter() - started, jitter, [], 0, mode='fixed')
    start_thetas = []
    correlations = [-0.7, -0.2, 0.2, 0.7]
    multipliers = [0.5, 1.0, 1.0, 2.0] if d > 1 else [0.25, 0.75, 2.5, 7.5]
    for i in range(config.starts):
        if i >= 4:
            raise ValueError('The primary complete-start list contains exactly four entries.')
        t = initial.copy()
        if fixed_theta is None:
            t[2 * d:2 * d + model.scales] += np.log(multipliers[i])
            if d > 1 and i > 0:
                noise_fraction = (0.0, 0.03, 0.1, 0.3)[i]
                t[d:2 * d] = np.log(np.exp(initial[:d]) * noise_fraction)
        if d > 1:
            rho = correlations[i]
            loading = np.full(d, np.sqrt(abs(rho)))
            loading[0] *= np.sign(rho)
            r = np.outer(loading, loading) + np.diag(1 - loading ** 2)
            t[2 * d + model.scales:] = correlation_coordinates(r)
        t[free] = np.clip(t[free], bounds[free, 0] + 1e-07, bounds[free, 1] - 1e-07)
        start_thetas.append(t)
    results, attempts = ([], [])
    for index, start in enumerate(start_thetas):
        count_bad = 0

        def objective(value):
            nonlocal count_bad
            t = start.copy()
            t[free] = value
            try:
                f, gradient = model.nll_gradient(t)
                return (f, gradient[free])
            except (np.linalg.LinAlgError, FloatingPointError):
                count_bad += 1
                return (1e+100, np.zeros(len(free)))
        r = minimize(objective, start[free], jac=True, method='L-BFGS-B', bounds=bounds[free], options={'maxiter': config.maxiter, 'ftol': config.ftol, 'gtol': config.gtol, 'maxls': 40})
        t = start.copy()
        t[free] = r.x
        good = bool(r.success and np.isfinite(r.fun) and (r.fun < 1e+99))
        attempts.append({'index': index, 'initial_theta': start.tolist(), 'success': good, 'message': str(r.message), 'nll': float(r.fun), 'iterations': int(r.nit), 'evaluations': int(r.nfev), 'nonfinite_evaluations': count_bad})
        results.append((good, float(r.fun), t))
    eligible = [r for r in results if r[0]]
    best = min(eligible or results, key=lambda r: r[1])
    success, nll, theta = best
    gradient_norm, jitter = (float('nan'), float('nan'))
    active = []
    try:
        _, grad = model.nll_gradient(theta)
        projected = grad[free].copy()
        for i, p in enumerate(free):
            near_low = theta[p] - bounds[p, 0] <= 1e-05
            near_high = bounds[p, 1] - theta[p] <= 1e-05
            if near_low or near_high:
                active.append(model.names[p])
            if near_low and grad[p] > 0 or (near_high and grad[p] < 0):
                projected[i] = 0
        gradient_norm = float(np.max(np.abs(projected)))
        _, jitter = model.factor(model.covariance(theta))
    except (np.linalg.LinAlgError, FloatingPointError):
        success = False
    return FitResult(model, theta, success, nll, perf_counter() - started, jitter, active, gradient_norm, attempts, 'correlation_only' if fixed_theta is not None else 'full')
