"""Fast analytic forecasts for linear models of ``ln(eta)``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .error_models import (
    UVX_EFFECTIVE_OPTIONS,
    resolve_sarm_error,
    resolve_uvx_error,
)


@dataclass(frozen=True)
class LinearDDRForecast:
    model: str
    redshift: np.ndarray
    injected_parameters: np.ndarray
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray

    @property
    def posterior_sigma(self):
        return np.sqrt(np.diag(self.posterior_covariance))


def required_sample_grid(
    eta,
    sigma_ln_dl,
    sigma_ln_da,
    significance=3.0,
    calibration_sigma=0.0,
    rho=0.0,
):
    """Return iid sample requirements over a distance-error grid.

    Rows correspond to ``sigma_ln_dl`` and columns to ``sigma_ln_da``.  Cells
    are infinite when the requested detection is below the shared calibration
    floor and therefore cannot be reached by increasing the sample size.
    """
    eta = float(eta)
    significance = float(significance)
    calibration_sigma = float(calibration_sigma)
    rho = float(rho)
    sigma_ln_dl = np.asarray(sigma_ln_dl, dtype=float)
    sigma_ln_da = np.asarray(sigma_ln_da, dtype=float)

    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and positive")
    if not np.isfinite(significance) or significance <= 0:
        raise ValueError("significance must be finite and positive")
    if not np.isfinite(calibration_sigma) or calibration_sigma < 0:
        raise ValueError("calibration_sigma must be finite and non-negative")
    if not np.isfinite(rho) or abs(rho) > 1:
        raise ValueError("rho must be finite and lie in [-1, 1]")
    if sigma_ln_dl.ndim != 1 or sigma_ln_da.ndim != 1:
        raise ValueError("distance-scatter grids must be one-dimensional")
    if (
        np.any(~np.isfinite(sigma_ln_dl))
        or np.any(~np.isfinite(sigma_ln_da))
        or np.any(sigma_ln_dl < 0)
        or np.any(sigma_ln_da < 0)
    ):
        raise ValueError("distance scatters must be finite and non-negative")

    random_variance = (
        sigma_ln_dl[:, None] ** 2
        + sigma_ln_da[None, :] ** 2
        - 2.0 * rho * sigma_ln_dl[:, None] * sigma_ln_da[None, :]
    )
    remaining_variance = (
        abs(np.log(eta)) / significance
    ) ** 2 - calibration_sigma**2
    if remaining_variance <= 0:
        return np.full(random_variance.shape, np.inf)
    return np.ceil(random_variance / remaining_variance)


def design_matrix(redshift, model):
    """Return the basis matrix for a model defined directly in ``ln(eta)``."""
    z = np.asarray(redshift, dtype=float)
    if z.ndim != 1 or not np.all(np.isfinite(z)) or np.any(z <= -1):
        raise ValueError(
            "redshift must be a finite one-dimensional array above -1"
        )
    if model == "constant":
        return np.ones((z.size, 1))
    if model == "linear_z":
        return np.column_stack([np.ones(z.size), z])
    if model == "linear_z_over_1pz":
        return np.column_stack([np.ones(z.size), z / (1.0 + z)])
    if model == "power_law":
        return np.log1p(z)[:, None]
    raise ValueError(
        "unknown DDR model {!r}; choose constant, linear_z, "
        "linear_z_over_1pz, or power_law".format(model)
    )


def _solve_diagonal_plus_calibration(diagonal, calibration_sigma, rhs):
    """Solve ``(diag(diagonal) + s^2 11^T) x = rhs`` efficiently."""
    diagonal = np.asarray(diagonal, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if np.any(diagonal <= 0):
        raise ValueError("random variances must be positive")
    inverse_diagonal = 1.0 / diagonal
    base = inverse_diagonal[:, None] * rhs if rhs.ndim == 2 else inverse_diagonal * rhs
    if calibration_sigma == 0:
        return base
    u = np.full(diagonal.size, float(calibration_sigma))
    inverse_u = inverse_diagonal * u
    denominator = 1.0 + u @ inverse_u
    projection = u @ base
    if rhs.ndim == 2:
        return base - inverse_u[:, None] * (projection / denominator)
    return base - inverse_u * (projection / denominator)


def forecast_linear_ddr(
    redshift,
    model,
    injected_parameters,
    sigma_ln_dl,
    sigma_ln_da,
    rho=0.0,
    calibration_sigma=0.0,
    prior_mean=None,
    prior_sigma=None,
):
    """Calculate the exact Gaussian Asimov posterior for a linear DDR model."""
    z = np.asarray(redshift, dtype=float)
    design = design_matrix(z, model)
    injected = np.asarray(injected_parameters, dtype=float)
    if injected.shape != (design.shape[1],):
        raise ValueError("injected_parameters has the wrong length for the model")
    if not np.all(np.isfinite(injected)):
        raise ValueError("injected_parameters must be finite")

    sigma_ln_dl = np.broadcast_to(np.asarray(sigma_ln_dl, dtype=float), z.shape)
    sigma_ln_da = np.broadcast_to(np.asarray(sigma_ln_da, dtype=float), z.shape)
    rho = np.broadcast_to(np.asarray(rho, dtype=float), z.shape)
    if not (
        np.all(np.isfinite(sigma_ln_dl))
        and np.all(np.isfinite(sigma_ln_da))
        and np.all(np.isfinite(rho))
        and np.isfinite(calibration_sigma)
    ):
        raise ValueError("distance scatters, rho, and calibration_sigma must be finite")
    if np.any(sigma_ln_dl < 0) or np.any(sigma_ln_da < 0):
        raise ValueError("distance scatters must be non-negative")
    if np.any(np.abs(rho) > 1):
        raise ValueError("rho must lie in [-1, 1]")
    if calibration_sigma < 0:
        raise ValueError("calibration_sigma must be non-negative")
    random_variance = (
        sigma_ln_dl**2
        + sigma_ln_da**2
        - 2.0 * rho * sigma_ln_dl * sigma_ln_da
    )

    observed = design @ injected
    solved_design = _solve_diagonal_plus_calibration(
        random_variance, calibration_sigma, design
    )
    solved_observed = _solve_diagonal_plus_calibration(
        random_variance, calibration_sigma, observed
    )
    precision = design.T @ solved_design
    information = design.T @ solved_observed

    if (prior_mean is None) != (prior_sigma is None):
        raise ValueError("prior_mean and prior_sigma must be supplied together")
    if prior_mean is not None:
        prior_mean = np.asarray(prior_mean, dtype=float)
        prior_sigma = np.asarray(prior_sigma, dtype=float)
        if prior_mean.shape != injected.shape or prior_sigma.shape != injected.shape:
            raise ValueError("prior arrays have the wrong shape")
        if np.any(prior_sigma <= 0):
            raise ValueError("prior_sigma must be positive")
        prior_precision = np.diag(1.0 / prior_sigma**2)
        precision = precision + prior_precision
        information = information + prior_precision @ prior_mean

    posterior_covariance = np.linalg.inv(precision)
    posterior_mean = posterior_covariance @ information
    return LinearDDRForecast(
        model=model,
        redshift=z.copy(),
        injected_parameters=injected,
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
    )


def scenario_detection_requirements(
    eta=0.925, significances=(3, 5), calibration_sigma=0.0, gamma=0.60
):
    """Return ideal iid sample requirements for every named error scenario."""
    if eta <= 0:
        raise ValueError("eta must be positive")
    if not np.isfinite(calibration_sigma) or calibration_sigma < 0:
        raise ValueError("calibration_sigma must be finite and non-negative")
    signal = abs(float(np.log(eta)))
    rows = []
    for name in UVX_EFFECTIVE_OPTIONS:
        uvx = resolve_uvx_error(name, gamma=gamma)
        sarm = resolve_sarm_error(name)
        row = {
            "scenario": name,
            "uvx_residual_dex": uvx.sigma_uvx_dex,
            "uvx_gamma": uvx.gamma,
            "sigma_ln_dl": uvx.sigma_ln_dl,
            "sigma_ln_da": sarm.sigma_ln_da,
            "sigma_ln_eta": float(np.hypot(uvx.sigma_ln_dl, sarm.sigma_ln_da)),
            "calibration_sigma_ln": float(calibration_sigma),
            "assumes_known_relative_calibration": calibration_sigma == 0,
        }
        for significance in significances:
            remaining_variance = (signal / significance) ** 2 - calibration_sigma**2
            row[f"n_{significance}sigma"] = (
                None
                if remaining_variance <= 0
                else int(np.ceil(row["sigma_ln_eta"]**2 / remaining_variance))
            )
        rows.append(row)
    return rows
