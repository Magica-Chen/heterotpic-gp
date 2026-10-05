"""Configuration dataclass for MV-GP model."""

from dataclasses import dataclass, field


@dataclass
class MVGPConfig:
    D: int                          # number of outputs
    M: int                          # number of inducing points
    W: int = 1                      # mixture components (start with 1)
    T: int = 1                      # input dimension
    kernel: str = "rbf"             # "rbf", "matern32", or "matern52"
    unit_kernel_variance: bool = True  # signal scales belong to Lambda; False is legacy fitting
    learn_inducing_locations: bool = True
    noise_type: str = "basis"       # "constant", "basis", "gp"
    n_basis: int = 8                # RBF basis functions for noise
    M_noise: int = 15               # inducing points per noise GP (noise_type="gp")
    noise_kernel: str = "rbf"       # kernel for noise GPs
    likelihood: str = "gaussian"    # "gaussian" or "student_t"
    gaussian_ell: str = "analytic"   # "mc" reproduces the historical local expectation estimator
    nu: float = 5.0                 # degrees of freedom for Student-t (nu > 2)
    learn_nu: bool = False          # whether to optimise nu jointly
    jitter: float = 1e-6
    mc_samples: int = 10            # MC samples for ELL
    batch_size: int = 64            # mini-batch size over observed pairs
    lr: float = 5e-3
    n_iters: int = 3000
    log_var_clip: tuple = (-10.0, 5.0)  # clip log-variance for stability
    seed: int = 0
