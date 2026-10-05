"""Heterotopic dataset: per-output (X, y), observation index, union grid."""

import jax.numpy as jnp
from jax import Array
from typing import NamedTuple
import itertools


class HeterotopicDataset(NamedTuple):
    """Stores heterotopic multi-output data.

    Fields
    ------
    xs : list[Array]
        xs[j] has shape (N_j, T) — inputs for output j.
    ys : list[Array]
        ys[j] has shape (N_j,) — observations for output j.
    obs_idx : Array
        (N_obs, 2) int array.  Each row is (grid_index, output_index).
    grid : Array
        (N, T) union grid of all unique input locations.
    D : int
        Number of outputs.
    """
    xs: list
    ys: list
    obs_idx: Array
    grid: Array
    D: int


def build_dataset(xs: list, ys: list) -> HeterotopicDataset:
    """Build a HeterotopicDataset from per-output inputs/observations.

    Parameters
    ----------
    xs : list of arrays, xs[j] shape (N_j, T)
    ys : list of arrays, ys[j] shape (N_j,)
    """
    D = len(xs)
    T = xs[0].shape[1] if xs[0].ndim == 2 else 1

    # Ensure 2D
    xs_2d = [x.reshape(-1, T) for x in xs]

    # Build union grid (unique input locations across all outputs)
    all_x = jnp.concatenate(xs_2d, axis=0)
    # Round to avoid floating-point duplicates
    grid, _ = _unique_rows(all_x)

    # Build obs_idx: for each output j, find grid index of each x in xs[j]
    obs_pairs = []
    for j in range(D):
        for n in range(xs_2d[j].shape[0]):
            grid_idx = _find_row(grid, xs_2d[j][n])
            obs_pairs.append([grid_idx, j])

    obs_idx = jnp.array(obs_pairs, dtype=jnp.int32)

    return HeterotopicDataset(
        xs=[jnp.array(x) for x in xs_2d],
        ys=[jnp.array(y) for y in ys],
        obs_idx=obs_idx,
        grid=grid,
        D=D,
    )


def _unique_rows(X: Array, tol: float = 1e-10) -> tuple:
    """Return unique rows of X (sorted) and their indices."""
    import numpy as np
    X_np = np.array(X)
    X_rounded = np.round(X_np / tol) * tol
    _, idx = np.unique(X_rounded, axis=0, return_index=True)
    idx = np.sort(idx)
    return jnp.array(X_np[idx]), idx


def _find_row(grid: Array, row: Array, tol: float = 1e-8) -> int:
    """Find the index of `row` in `grid`."""
    dists = jnp.sum((grid - row[None, :]) ** 2, axis=1)
    return int(jnp.argmin(dists))


def get_stacked_y(dataset: HeterotopicDataset) -> Array:
    """Stack all observations into a single 1D array matching obs_idx order."""
    ys_stacked = []
    for j in range(dataset.D):
        ys_stacked.append(dataset.ys[j])
    return jnp.concatenate(ys_stacked)


def compute_omega(dataset: HeterotopicDataset) -> float:
    """Overlap index: average pairwise fraction of shared locations.

    omega = [1/C(D,2)] * sum_{p<q} |X_p ∩ X_q| / min(|X_p|, |X_q|)
    """
    import numpy as np
    D = dataset.D
    if D < 2:
        return 1.0

    xs_sets = []
    tol = 1e-8
    for j in range(D):
        xj = np.array(dataset.xs[j])
        rounded = np.round(xj / tol) * tol
        xs_sets.append(set(map(tuple, rounded)))

    total = 0.0
    count = 0
    for p, q in itertools.combinations(range(D), 2):
        shared = len(xs_sets[p] & xs_sets[q])
        min_size = min(len(xs_sets[p]), len(xs_sets[q]))
        if min_size > 0:
            total += shared / min_size
        count += 1

    return float(total / count) if count > 0 else 1.0


def compute_beta(dataset: HeterotopicDataset, noise_fns: list) -> float:
    """Heteroscedasticity index.

    beta_j = 1 - min_x v_j(x) / max_x v_j(x)
    beta = max_j beta_j

    Parameters
    ----------
    noise_fns : list of callables
        noise_fns[j](x) returns noise variance at x for output j.
    """
    beta = 0.0
    for j in range(dataset.D):
        xj = dataset.xs[j]
        v = noise_fns[j](xj)
        v_min = float(jnp.min(v))
        v_max = float(jnp.max(v))
        if v_max > 0:
            beta_j = 1.0 - v_min / v_max
        else:
            beta_j = 0.0
        beta = max(beta, beta_j)
    return beta
