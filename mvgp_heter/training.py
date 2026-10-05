"""Training loop: mini-batch Adam optimisation of the ELBO."""

import jax
import jax.numpy as jnp
from jax import Array
import optax
from typing import NamedTuple

from mvgp_heter.config import MVGPConfig
from mvgp_heter.data import HeterotopicDataset, get_stacked_y
from mvgp_heter.model import MVGPParams, init_model_params
from mvgp_heter.variational import VariationalParams, init_variational_params
from mvgp_heter.noise import (
    init_noise_params, make_basis_centres,
    NoiseParams, ConstantNoiseParams, GPNoiseParams,
)
from mvgp_heter.elbo import elbo


class AllParams(NamedTuple):
    """Bundle all learnable parameters."""
    model: MVGPParams
    variational: VariationalParams
    noise: NoiseParams | ConstantNoiseParams | GPNoiseParams


class TrainResult(NamedTuple):
    """Training result."""
    params: AllParams
    elbo_history: Array
    noise_centres: Array | None


def train(
    config: MVGPConfig,
    dataset: HeterotopicDataset,
    key: Array,
    verbose: bool = True,
    print_every: int = 100,
    *,
    initial_params: AllParams | None = None,
    train_model: bool = True,
    train_noise: bool = True,
) -> TrainResult:
    """Full training loop.

    1. Initialise model, noise, variational parameters.
    2. JIT-compile ELBO + gradient.
    3. Loop: sample mini-batch, compute gradient, Adam step.
    4. Return fitted parameters and ELBO history.

    The revised default fixes the base-kernel variance at one, with signal
    variances represented by Lambda. Set config.unit_kernel_variance=False
    only for explicitly labelled legacy fits. initial_params together with
    train_model=False and train_noise=False permits a fixed-covariance
    inference audit while still optimising the submitted matrix-normal q(U).
    """
    key, k1, k2, k3 = jax.random.split(key, 4)

    # Data bounds
    all_x = jnp.concatenate(dataset.xs, axis=0)
    x_min = jnp.min(all_x, axis=0)
    x_max = jnp.max(all_x, axis=0)

    # Initialise parameters
    model_params = init_model_params(
        k1, config.D, config.M, config.T, config.kernel,
        x_min, x_max,
    )
    var_params = init_variational_params(k2, config.M, config.D)

    # Noise
    x_min_scalar = float(jnp.squeeze(x_min)) if config.T == 1 else float(x_min[0])
    x_max_scalar = float(jnp.squeeze(x_max)) if config.T == 1 else float(x_max[0])

    noise_centres = None
    if config.noise_type == "gp":
        noise_params = init_noise_params(
            k3, config.D, config.n_basis, None, "gp",
            M_noise=config.M_noise, T=config.T,
            x_min=x_min_scalar, x_max=x_max_scalar,
            noise_kernel=config.noise_kernel,
        )
    else:
        noise_centres = make_basis_centres(
            x_min_scalar, x_max_scalar,
            config.n_basis, config.T,
        )
        noise_params = init_noise_params(
            k3, config.D, config.n_basis, noise_centres, config.noise_type,
        )

    params = AllParams(model=model_params, variational=var_params, noise=noise_params)
    if initial_params is not None:
        params = initial_params
    if config.unit_kernel_variance and float(params.model.kernel_params.log_variance) != 0.0:
        raise ValueError("Unit-kernel fitting requires log_variance=0 in initial_params; absorb the signal scale into Lambda first.")

    # Stacked observations and index arrays for mini-batching
    obs_idx = dataset.obs_idx  # (N_obs, 2)
    n_obs = obs_idx.shape[0]
    y_stacked = get_stacked_y(dataset)  # (N_obs,)

    # Precompute: for each observed pair (grid_idx, output_j),
    # get the actual x location from the grid
    obs_x = dataset.grid[obs_idx[:, 0]]  # (N_obs, T)
    obs_j = obs_idx[:, 1]                 # (N_obs,)

    # Capture config values as locals for the JIT closure
    likelihood = config.likelihood
    nu = config.nu

    # Optimiser
    optimizer = optax.adam(config.lr)
    opt_state = optimizer.init(params)

    # Define loss (negative ELBO) and its gradient
    def neg_elbo(params, key, batch_indices):
        bx = obs_x[batch_indices]
        bj = obs_j[batch_indices]
        by = y_stacked[batch_indices]
        return -elbo(
            key, params.variational, params.model, params.noise,
            config.kernel, bx, bj, by, n_obs, config.mc_samples,
            config.jitter, noise_centres, config.log_var_clip,
            likelihood, nu,
            config.gaussian_ell,
        )

    @jax.jit
    def step(params, opt_state, key):
        k1, k2 = jax.random.split(key)
        # Sample mini-batch indices
        batch_indices = jax.random.choice(
            k1, n_obs, shape=(config.batch_size,), replace=False,
        )
        loss, grads = jax.value_and_grad(neg_elbo)(params, k2, batch_indices)
        if not train_model:
            grads = grads._replace(model=jax.tree.map(jnp.zeros_like, grads.model))
        else:
            model_grads = grads.model
            if config.unit_kernel_variance:
                kernel_grads = model_grads.kernel_params._replace(
                    log_variance=jnp.zeros_like(model_grads.kernel_params.log_variance))
                model_grads = model_grads._replace(kernel_params=kernel_grads)
            if not config.learn_inducing_locations:
                model_grads = model_grads._replace(Z=jnp.zeros_like(model_grads.Z))
            grads = grads._replace(model=model_grads)
        if not train_noise:
            grads = grads._replace(noise=jax.tree.map(jnp.zeros_like, grads.noise))
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    # Training loop
    elbo_history = []
    for i in range(config.n_iters):
        key, step_key = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, step_key)

        elbo_val = -float(loss)
        elbo_history.append(elbo_val)

        if verbose and (i % print_every == 0 or i == config.n_iters - 1):
            print(f"Iter {i:5d}/{config.n_iters}  ELBO = {elbo_val:.4f}")

    return TrainResult(
        params=params,
        elbo_history=jnp.array(elbo_history),
        noise_centres=noise_centres,
    )
