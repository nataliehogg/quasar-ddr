"""Statistical primitives for cosmology-independent DDR forecasts.

The public API uses natural logarithms for distance-space inference.  Helpers
whose names contain ``log10`` are explicit conversions from literature values
quoted in dex.
"""

from __future__ import annotations

import math

import numpy as np


def _checked_cholesky(matrix, name):
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-14):
        raise ValueError(f"{name} must be symmetric")
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc


def _cholesky_solve(cholesky, rhs):
    return np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, rhs))


def uvx_residual_dex_to_log10_distance(delta_uvx_dex, gamma=0.6):
    """Convert UV--X residual dispersion in dex to distance dispersion in dex."""
    delta = np.asarray(delta_uvx_dex, dtype=float)
    if np.any(~np.isfinite(delta)) or not np.isfinite(gamma):
        raise ValueError("delta_uvx_dex and gamma must be finite")
    if np.any(delta < 0):
        raise ValueError("delta_uvx_dex must be non-negative")
    if gamma >= 1:
        raise ValueError("gamma must be smaller than 1")
    return delta / (2.0 * (1.0 - gamma))


def log10_scatter_to_fractional_linearized(sigma_log10_distance):
    """Return the small-scatter fractional approximation for a dex scatter."""
    sigma = np.asarray(sigma_log10_distance, dtype=float)
    if np.any(~np.isfinite(sigma)):
        raise ValueError("sigma_log10_distance must be finite")
    if np.any(sigma < 0):
        raise ValueError("sigma_log10_distance must be non-negative")
    return np.log(10.0) * sigma


def observed_log_eta(dl, da, redshift):
    """Compute ``ln(DL / ((1+z)^2 DA))`` for paired positive distances."""
    dl = np.asarray(dl, dtype=float)
    da = np.asarray(da, dtype=float)
    redshift = np.asarray(redshift, dtype=float)
    if dl.shape != da.shape or dl.shape != redshift.shape:
        raise ValueError("dl, da, and redshift must have matching shapes")
    if not (
        np.all(np.isfinite(dl))
        and np.all(np.isfinite(da))
        and np.all(np.isfinite(redshift))
    ):
        raise ValueError("dl, da, and redshift must contain only finite values")
    if np.any(dl <= 0) or np.any(da <= 0):
        raise ValueError("dl and da must be strictly positive")
    if np.any(redshift <= -1):
        raise ValueError("redshift must be greater than -1")
    return np.log(dl) - np.log(da) - 2.0 * np.log1p(redshift)


def log_eta_covariance(
    sigma_ln_dl,
    sigma_ln_da,
    rho=0.0,
    calibration_sigma=0.0,
    additional_covariance=None,
):
    """Build the covariance of paired ``ln(eta)`` measurements.

    ``calibration_sigma`` is a shared one-sigma offset in natural-log space,
    represented as a fully correlated rank-one covariance.
    """
    sigma_ln_dl = np.atleast_1d(np.asarray(sigma_ln_dl, dtype=float))
    sigma_ln_da = np.atleast_1d(np.asarray(sigma_ln_da, dtype=float))
    if sigma_ln_dl.shape != sigma_ln_da.shape:
        raise ValueError("sigma_ln_dl and sigma_ln_da must have matching shapes")
    rho = np.broadcast_to(np.asarray(rho, dtype=float), sigma_ln_dl.shape)
    if np.any(sigma_ln_dl < 0) or np.any(sigma_ln_da < 0):
        raise ValueError("distance scatters must be non-negative")
    if np.any(np.abs(rho) > 1):
        raise ValueError("rho must lie in [-1, 1]")
    if calibration_sigma < 0:
        raise ValueError("calibration_sigma must be non-negative")

    variance = (
        sigma_ln_dl**2
        + sigma_ln_da**2
        - 2.0 * rho * sigma_ln_dl * sigma_ln_da
    )
    if np.any(variance < -1e-15):
        raise ValueError("correlation produced a negative variance")
    covariance = np.diag(np.maximum(variance, 0.0))
    covariance += calibration_sigma**2 * np.ones_like(covariance)
    if additional_covariance is not None:
        extra = np.asarray(additional_covariance, dtype=float)
        if extra.shape != covariance.shape:
            raise ValueError("additional_covariance has the wrong shape")
        covariance = covariance + extra
    return covariance


def gaussian_log_likelihood(observed, predicted, covariance):
    """Normalized multivariate Gaussian log likelihood."""
    observed = np.atleast_1d(np.asarray(observed, dtype=float))
    predicted = np.broadcast_to(np.asarray(predicted, dtype=float), observed.shape)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (observed.size, observed.size):
        raise ValueError("covariance has the wrong shape")
    cholesky = _checked_cholesky(covariance, "covariance")
    logdet = 2.0 * np.sum(np.log(np.diag(cholesky)))
    residual = observed - predicted
    chi2 = residual @ _cholesky_solve(cholesky, residual)
    return -0.5 * (chi2 + logdet + observed.size * np.log(2.0 * np.pi))


def linear_gaussian_posterior(
    design,
    observed,
    covariance,
    prior_mean=None,
    prior_covariance=None,
):
    """Return posterior mean and covariance for ``y = X theta``."""
    design = np.asarray(design, dtype=float)
    observed = np.asarray(observed, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if design.ndim != 2 or observed.shape != (design.shape[0],):
        raise ValueError("design and observed have incompatible shapes")
    if covariance.shape != (design.shape[0], design.shape[0]):
        raise ValueError("covariance has the wrong shape")

    covariance_cholesky = _checked_cholesky(covariance, "covariance")
    solved_design = _cholesky_solve(covariance_cholesky, design)
    precision = design.T @ solved_design
    information = design.T @ _cholesky_solve(covariance_cholesky, observed)

    if (prior_mean is None) != (prior_covariance is None):
        raise ValueError("prior_mean and prior_covariance must be supplied together")
    if prior_mean is not None:
        prior_mean = np.asarray(prior_mean, dtype=float)
        prior_covariance = np.asarray(prior_covariance, dtype=float)
        if prior_mean.shape != (design.shape[1],):
            raise ValueError("prior_mean has the wrong shape")
        if prior_covariance.shape != (design.shape[1], design.shape[1]):
            raise ValueError("prior_covariance has the wrong shape")
        prior_cholesky = _checked_cholesky(prior_covariance, "prior_covariance")
        prior_precision = _cholesky_solve(
            prior_cholesky, np.eye(prior_covariance.shape[0])
        )
        precision = precision + prior_precision
        information = information + prior_precision @ prior_mean

    precision_cholesky = _checked_cholesky(precision, "posterior precision")
    posterior_covariance = _cholesky_solve(
        precision_cholesky, np.eye(precision.shape[0])
    )
    posterior_mean = posterior_covariance @ information
    return posterior_mean, posterior_covariance


def required_sample_size(signal_ln_eta, sigma_ln_eta, significance=3.0):
    """Minimum iid sample size required for a target detection significance."""
    signal = abs(float(signal_ln_eta))
    sigma = float(sigma_ln_eta)
    significance = float(significance)
    if signal <= 0 or sigma <= 0 or significance <= 0:
        raise ValueError("signal, sigma, and significance must be positive")
    return math.ceil((significance * sigma / signal) ** 2)


def equal_distance_precision_for_detection(eta, n_objects, significance=3.0):
    """Per-distance natural-log scatter when DL and DA contribute equally."""
    eta = float(eta)
    n_objects = int(n_objects)
    significance = float(significance)
    if eta <= 0 or n_objects <= 0 or significance <= 0:
        raise ValueError("eta, n_objects, and significance must be positive")
    return abs(math.log(eta)) * math.sqrt(n_objects) / (
        significance * math.sqrt(2.0)
    )
