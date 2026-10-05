"""Geometry for the multivariate kriging experiments."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable
import numpy as np
SITE_TOLERANCE = 1e-10

def _to_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[:, None]
    return x

def unique_sites(x: np.ndarray, tol: float=SITE_TOLERANCE) -> np.ndarray:
    x = _to_2d(x)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError('Sites must be a finite two-dimensional coordinate array.')
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError('Coordinate-grid tolerance must be finite and positive.')
    return np.unique(np.rint(x / tol) * tol, axis=0)

def pairwise_distances(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    x1 = _to_2d(x1)
    x2 = _to_2d(x2)
    diff = x1[:, None, :] - x2[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))

def directed_nearest_distances(x_from: np.ndarray, x_to: np.ndarray) -> np.ndarray:
    dmat = pairwise_distances(x_from, x_to)
    return np.min(dmat, axis=1)

def within_design_spacing(x: np.ndarray) -> float:
    x = unique_sites(x)
    if len(x) <= 1:
        return float('nan')
    dmat = pairwise_distances(x, x)
    np.fill_diagonal(dmat, np.inf)
    return float(np.mean(np.min(dmat, axis=1)))

def directed_coverage(x_from: np.ndarray, x_to: np.ndarray, radii: Iterable[float]) -> np.ndarray:
    x_from, x_to = (unique_sites(x_from), unique_sites(x_to))
    radii = np.asarray(list(radii), dtype=float)
    if len(x_from) == 0 or len(x_to) == 0:
        return np.full(radii.shape, np.nan)
    d = directed_nearest_distances(x_from, x_to)
    return np.array([(d <= r).mean() for r in radii], dtype=float)

def directed_proximity(x_from: np.ndarray, x_to: np.ndarray) -> float:
    x_from, x_to = (unique_sites(x_from), unique_sites(x_to))
    if len(x_from) == 0 or len(x_to) == 0:
        return float('nan')
    return float(np.mean(directed_nearest_distances(x_from, x_to)))

def normalized_directed_proximity(x_from: np.ndarray, x_to: np.ndarray) -> float:
    spacing = within_design_spacing(x_to)
    if not np.isfinite(spacing) or spacing <= 0:
        return float('nan')
    return directed_proximity(x_from, x_to) / spacing

def overlap_index(x1: np.ndarray, x2: np.ndarray, tol: float=SITE_TOLERANCE) -> float:
    x1 = unique_sites(x1, tol)
    x2 = unique_sites(x2, tol)
    s1 = set(map(tuple, x1))
    s2 = set(map(tuple, x2))
    if not s1 or not s2:
        return float('nan')
    return len(s1 & s2) / min(len(s1), len(s2))

def proximity_matrix(xs: list[np.ndarray]) -> np.ndarray:
    d = len(xs)
    out = np.zeros((d, d), dtype=float)
    for p in range(d):
        for q in range(d):
            if p == q:
                continue
            out[p, q] = normalized_directed_proximity(xs[p], xs[q])
    return out

def local_bpi(x_star: np.ndarray, x_target: np.ndarray, x_aux: list[np.ndarray]) -> float:
    x_star = unique_sites(x_star)
    x_target = unique_sites(x_target)
    x_aux = [unique_sites(xa) for xa in x_aux if len(xa) > 0]
    if len(x_star) != 1:
        raise ValueError('local_bpi requires exactly one prediction location.')
    if len(x_target) == 0 or not x_aux:
        return float('nan')
    delta_target = directed_nearest_distances(x_star, x_target)[0]
    if delta_target <= 0:
        return 0.0
    delta_aux = min((directed_nearest_distances(x_star, xa)[0] for xa in x_aux if len(xa) > 0))
    if delta_aux >= delta_target:
        return 0.0
    return 1.0 - delta_aux / delta_target

def global_bpi(xs: list[np.ndarray], eval_grid: np.ndarray) -> np.ndarray:
    eval_grid = _to_2d(eval_grid)
    d = len(xs)
    out = np.zeros(d, dtype=float)
    for j in range(d):
        aux = [xs[q] for q in range(d) if q != j]
        out[j] = np.mean([local_bpi(x[None, :], xs[j], aux) for x in eval_grid])
    return out

@dataclass
class PairGeometry:
    omega: float
    pi_pq: float
    pi_qp: float
    tpi_pq: float
    tpi_qp: float
    coverage_radii: np.ndarray
    dc_pq: np.ndarray
    dc_qp: np.ndarray

def summarise_pair_geometry(x_p: np.ndarray, x_q: np.ndarray, radii: Iterable[float]) -> PairGeometry:
    radii = np.asarray(list(radii), dtype=float)
    return PairGeometry(omega=overlap_index(x_p, x_q), pi_pq=directed_proximity(x_p, x_q), pi_qp=directed_proximity(x_q, x_p), tpi_pq=normalized_directed_proximity(x_p, x_q), tpi_qp=normalized_directed_proximity(x_q, x_p), coverage_radii=radii, dc_pq=directed_coverage(x_p, x_q, radii), dc_qp=directed_coverage(x_q, x_p, radii))

def pairwise_overlap_matrix(xs: list[np.ndarray]) -> np.ndarray:
    d = len(xs)
    out = np.eye(d, dtype=float)
    for p, q in combinations(range(d), 2):
        val = overlap_index(xs[p], xs[q])
        out[p, q] = val
        out[q, p] = val
    return out
