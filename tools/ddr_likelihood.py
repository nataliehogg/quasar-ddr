"""Dependency-free log-distance likelihood used by Cobaya and tests."""

from __future__ import annotations

import numpy as np

from .ddr_statistics import (
    gaussian_log_likelihood,
    log_eta_covariance,
    observed_log_eta,
)
from .forecast import design_matrix


def predict_log_eta(redshift, model, alpha=0.0, beta=0.0):
    """Predict natural-log DDR ratio for legacy and log-linear models."""
    z = np.asarray(redshift, dtype=float)
    if model == "M1":
        eta = 1.0 + alpha
        if eta <= 0:
            raise ValueError("M1 requires 1 + alpha > 0")
        return np.full(z.shape, np.log(eta))
    if model == "M2":
        eta = 1.0 + alpha + beta * z
        if np.any(eta <= 0):
            raise ValueError("M2 requires positive eta at every redshift")
        return np.log(eta)
    if model == "M3":
        return alpha * np.log1p(z)
    if model in {"constant", "linear_z", "linear_z_over_1pz", "power_law"}:
        parameters = {
            "constant": np.array([alpha]),
            "linear_z": np.array([alpha, beta]),
            "linear_z_over_1pz": np.array([alpha, beta]),
            "power_law": np.array([alpha]),
        }[model]
        return design_matrix(z, model) @ parameters
    raise ValueError(f"unknown DDR model {model!r}")


def paired_distance_log_likelihood(
    dl,
    da,
    redshift,
    sigma_ln_dl,
    sigma_ln_da,
    model,
    alpha=0.0,
    beta=0.0,
    rho=0.0,
    calibration_offset=0.0,
    calibration_sigma=0.0,
    sigma_int_ln_eta=0.0,
):
    """Normalized Gaussian likelihood for paired observations in ``ln(eta)``."""
    if sigma_int_ln_eta < 0:
        raise ValueError("sigma_int_ln_eta must be non-negative")
    observed = observed_log_eta(dl, da, redshift)
    sigma_ln_dl = np.broadcast_to(np.asarray(sigma_ln_dl, dtype=float), observed.shape)
    sigma_ln_da = np.broadcast_to(np.asarray(sigma_ln_da, dtype=float), observed.shape)
    rho = np.broadcast_to(np.asarray(rho, dtype=float), observed.shape)
    covariance = log_eta_covariance(
        sigma_ln_dl,
        sigma_ln_da,
        rho=rho,
        calibration_sigma=calibration_sigma,
    )
    covariance = covariance + sigma_int_ln_eta**2 * np.eye(observed.size)
    predicted = predict_log_eta(
        redshift, model=model, alpha=alpha, beta=beta
    ) + calibration_offset
    return gaussian_log_likelihood(observed, predicted, covariance)
