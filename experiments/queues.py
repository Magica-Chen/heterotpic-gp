"""Queues for the multivariate kriging experiments."""
import numpy as np

def departures(arrivals, service):
    arrivals, service = (np.asarray(arrivals, float), np.asarray(service, float))
    if arrivals.ndim != 1 or arrivals.shape != service.shape or len(arrivals) == 0:
        raise ValueError('Nonempty equal-length arrival/service vectors required.')
    if np.any(np.diff(arrivals) < 0) or np.any(arrivals < 0) or np.any(service <= 0):
        raise ValueError('Ordered nonnegative arrivals and positive service times required.')
    cumulative = np.cumsum(service)
    return cumulative + np.maximum.accumulate(arrivals - np.r_[0.0, cumulative[:-1]])

def mm1_run(rho, rng, customers=10000, warmup=1000):
    if not 0 < rho < 1:
        raise ValueError('Stable M/M/1 traffic must lie in (0,1).')
    n = customers + warmup
    arrival = np.cumsum(rng.exponential(1 / rho, size=n))
    service = rng.exponential(1.0, size=n)
    departure = departures(arrival, service)
    wait = departure - service - arrival
    utilization = service[warmup:].sum() / (departure[-1] - arrival[warmup])
    return np.array([wait[warmup:].mean(), utilization])

def tandem_run(arrival_rate, scv, rng, customers=20000, warmup=2000, high_fidelity=True):
    if not 0 < arrival_rate < 1 or scv <= 0:
        raise ValueError('Stable arrival rate and positive service SCV required.')
    n = customers + warmup
    arrival = np.cumsum(rng.exponential(1 / arrival_rate, size=n))
    variance = np.log1p(scv)
    sigma = np.sqrt(variance)
    service1 = rng.lognormal(-variance / 2, sigma, size=n)
    d1 = departures(arrival, service1)
    if not high_fidelity:
        return float(np.mean((d1 - arrival)[warmup:]))
    service2 = rng.lognormal(np.log(1 / 1.1) - variance / 2, sigma, size=n)
    d2 = departures(d1, service2)
    return float(np.mean((d2 - arrival)[warmup:]))

def q1_design(geometry):
    x = np.linspace(0.35, 0.85, 14)[:, None]
    if geometry == 'isotopic':
        y = x.copy()
    elif geometry == 'interleaved':
        y = (np.linspace(0.35, 0.85, 14, endpoint=False) + 0.5 * (0.5 / 14))[:, None]
    elif geometry == 'separated':
        y = np.linspace(0.05, 0.25, 14)[:, None]
    else:
        raise ValueError(geometry)
    return [x, y]
