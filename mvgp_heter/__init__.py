"""HeterMV-GP: Heterotopic & Heteroscedastic Multi-output Variational GP in JAX."""

from mvgp_heter.config import MVGPConfig
from mvgp_heter.data import HeterotopicDataset, build_dataset
from mvgp_heter.model import MVGPParams, init_model_params, get_Lambda
from mvgp_heter.noise import NoiseParams, ConstantNoiseParams, GPNoiseParams
from mvgp_heter.variational import VariationalParams, init_variational_params
from mvgp_heter.training import train
from mvgp_heter.prediction import predict

__all__ = [
    "MVGPConfig",
    "HeterotopicDataset",
    "build_dataset",
    "MVGPParams",
    "init_model_params",
    "get_Lambda",
    "NoiseParams",
    "ConstantNoiseParams",
    "GPNoiseParams",
    "VariationalParams",
    "init_variational_params",
    "train",
    "predict",
]
