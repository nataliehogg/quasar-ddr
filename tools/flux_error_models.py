"""Runtime loading, interpolation, and sampling of tabulated flux errors."""

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np


_TABLE_COLUMNS = (
    "z",
    "median_sigma_log_fuv_dex",
    "scatter_ln_sigma_log_fuv",
    "median_sigma_log_fx_dex",
    "scatter_ln_sigma_log_fx",
    "corr_ln_uncertainty_sizes",
    "effective_reference_count",
    "smoothing_width_z",
)
_MEDIAN_COLUMNS = (
    "median_sigma_log_fuv_dex",
    "median_sigma_log_fx_dex",
)
_SCATTER_COLUMNS = (
    "scatter_ln_sigma_log_fuv",
    "scatter_ln_sigma_log_fx",
)


@dataclass(frozen=True)
class FluxErrorTable:
    z: np.ndarray
    median_sigma_log_fuv_dex: np.ndarray
    scatter_ln_sigma_log_fuv: np.ndarray
    median_sigma_log_fx_dex: np.ndarray
    scatter_ln_sigma_log_fx: np.ndarray
    corr_ln_uncertainty_sizes: np.ndarray
    effective_reference_count: np.ndarray
    smoothing_width_z: np.ndarray
    schema_version: int
    csv_path: Path
    metadata_path: Path
    csv_sha256: str
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class InterpolatedFluxErrorDistribution:
    redshift: np.ndarray
    median_sigma_log_fuv_dex: np.ndarray
    scatter_ln_sigma_log_fuv: np.ndarray
    median_sigma_log_fx_dex: np.ndarray
    scatter_ln_sigma_log_fx: np.ndarray
    corr_ln_uncertainty_sizes: np.ndarray
    effective_reference_count: np.ndarray
    smoothing_width_z: np.ndarray


@dataclass(frozen=True)
class FluxPropagation:
    sigma_uvx_measurement_dex: np.ndarray
    sigma_uvx_intrinsic_dex: np.ndarray
    sigma_uvx_total_dex: np.ndarray
    sigma_ln_dl: np.ndarray


@dataclass(frozen=True)
class FluxUncertaintyRealization:
    sigma_log_fuv_dex: np.ndarray
    sigma_log_fx_dex: np.ndarray
    sigma_uvx_measurement_dex: np.ndarray
    sigma_uvx_intrinsic_dex: np.ndarray
    sigma_uvx_total_dex: np.ndarray
    sigma_ln_dl: np.ndarray
    corr_ln_uncertainty_sizes: np.ndarray
    effective_reference_count: np.ndarray
    smoothing_width_z: np.ndarray
    gamma: float
    delta_dex: float
    flux_measurement_error_correlation: float
    model_name: str = "lusso2020_flux_table"


def _readonly(values):
    result = np.asarray(values, dtype=float).copy()
    result.setflags(write=False)
    return result


def _validated_metadata(metadata_path, csv_sha256):
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read flux-error metadata {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError("flux-error metadata must be a JSON object")
    if metadata.get("table_schema_version") != 1:
        raise ValueError("unsupported flux-error table schema version")
    if metadata.get("csv_sha256") != csv_sha256:
        raise ValueError("flux-error table CSV digest does not match metadata")
    if not isinstance(metadata.get("reference_identifier"), str) or not metadata[
        "reference_identifier"
    ].strip():
        raise ValueError("flux-error metadata requires a reference identifier")
    for key in ("supported_redshift_min", "supported_redshift_max"):
        try:
            value = float(metadata[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"flux-error metadata requires finite {key}") from error
        if not np.isfinite(value):
            raise ValueError(f"flux-error metadata requires finite {key}")
    if float(metadata["supported_redshift_min"]) > float(
        metadata["supported_redshift_max"]
    ):
        raise ValueError("invalid flux-error metadata redshift support")
    return metadata


def load_flux_error_table(csv_path) -> FluxErrorTable:
    """Load and validate a schema-version-1 derived flux-error table."""
    csv_path = Path(csv_path)
    metadata_path = csv_path.with_name(f"{csv_path.stem}_metadata.json")
    try:
        csv_bytes = csv_path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read flux-error table {csv_path}") from error
    csv_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    metadata = _validated_metadata(metadata_path, csv_sha256)

    try:
        with io.StringIO(csv_bytes.decode(), newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(_TABLE_COLUMNS):
                raise ValueError("flux-error CSV columns must match the schema exactly")
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"cannot read flux-error table {csv_path}") from error
    if len(rows) < 2:
        raise ValueError("flux-error table requires at least two redshift nodes")
    if any(set(row) != set(_TABLE_COLUMNS) for row in rows):
        raise ValueError("flux-error CSV rows must match the schema exactly")

    columns = {}
    for name in _TABLE_COLUMNS:
        try:
            values = np.asarray([float(row[name]) for row in rows], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid values in flux-error CSV column {name}") from error
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite values in flux-error CSV column {name}")
        columns[name] = values

    if not np.all(np.diff(columns["z"]) > 0):
        raise ValueError("flux-error table redshift nodes must be strictly increasing")
    for name in _MEDIAN_COLUMNS:
        if np.any(columns[name] <= 0):
            raise ValueError(f"flux-error table requires positive {name}")
    for name in _SCATTER_COLUMNS:
        if np.any(columns[name] < 0):
            raise ValueError(f"flux-error table requires non-negative {name}")
    if np.any(np.abs(columns["corr_ln_uncertainty_sizes"]) > 1):
        raise ValueError("flux-error table correlation must lie in [-1, 1]")
    for name in ("effective_reference_count", "smoothing_width_z"):
        if np.any(columns[name] <= 0):
            raise ValueError(f"flux-error table requires positive {name}")
    if (
        float(metadata["supported_redshift_min"]) != columns["z"][0]
        or float(metadata["supported_redshift_max"]) != columns["z"][-1]
    ):
        raise ValueError("metadata redshift support does not match table endpoints")

    return FluxErrorTable(
        **{name: _readonly(columns[name]) for name in _TABLE_COLUMNS},
        schema_version=1,
        csv_path=csv_path,
        metadata_path=metadata_path,
        csv_sha256=csv_sha256,
        provenance=MappingProxyType(metadata),
    )


def interpolate_flux_error_table(
    table: FluxErrorTable, redshift
) -> InterpolatedFluxErrorDistribution:
    """Interpolate a table at redshift values without extrapolation."""
    redshift = np.asarray(redshift, dtype=float)
    if not np.all(np.isfinite(redshift)):
        raise ValueError("mock redshifts must be finite")
    if np.any(redshift < table.z[0]) or np.any(redshift > table.z[-1]):
        raise ValueError("mock redshifts lie outside flux-error table support")

    def interpolate(values):
        return _readonly(np.interp(redshift, table.z, values))

    def interpolate_median(values):
        result = np.exp(np.interp(redshift, table.z, np.log(values)))
        for node, value in zip(table.z, values):
            result = np.where(redshift == node, value, result)
        return _readonly(result)

    return InterpolatedFluxErrorDistribution(
        redshift=_readonly(redshift),
        median_sigma_log_fuv_dex=interpolate_median(
            table.median_sigma_log_fuv_dex
        ),
        scatter_ln_sigma_log_fuv=interpolate(table.scatter_ln_sigma_log_fuv),
        median_sigma_log_fx_dex=interpolate_median(table.median_sigma_log_fx_dex),
        scatter_ln_sigma_log_fx=interpolate(table.scatter_ln_sigma_log_fx),
        corr_ln_uncertainty_sizes=interpolate(table.corr_ln_uncertainty_sizes),
        effective_reference_count=interpolate(table.effective_reference_count),
        smoothing_width_z=interpolate(table.smoothing_width_z),
    )


def _finite_float(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def propagate_flux_uncertainties(
    *,
    sigma_log_fuv_dex,
    sigma_log_fx_dex,
    gamma,
    delta_dex,
    flux_measurement_error_correlation,
) -> FluxPropagation:
    """Propagate signed flux-measurement errors into distance uncertainty."""
    sigma_log_fuv_dex = np.asarray(sigma_log_fuv_dex, dtype=float)
    sigma_log_fx_dex = np.asarray(sigma_log_fx_dex, dtype=float)
    if sigma_log_fuv_dex.shape != sigma_log_fx_dex.shape:
        raise ValueError("flux measurement uncertainty arrays must have matching shapes")
    if not (
        np.all(np.isfinite(sigma_log_fuv_dex))
        and np.all(np.isfinite(sigma_log_fx_dex))
    ):
        raise ValueError("flux measurement uncertainties must be finite")
    if np.any(sigma_log_fuv_dex < 0) or np.any(sigma_log_fx_dex < 0):
        raise ValueError("flux measurement uncertainties must be non-negative")

    gamma = _finite_float(gamma, "gamma")
    delta_dex = _finite_float(delta_dex, "delta_dex")
    flux_measurement_error_correlation = _finite_float(
        flux_measurement_error_correlation, "flux_measurement_error_correlation"
    )
    if not 0 <= gamma < 1:
        raise ValueError("gamma must satisfy 0 <= gamma < 1")
    if delta_dex < 0:
        raise ValueError("delta_dex must be non-negative")
    if not -1 <= flux_measurement_error_correlation <= 1:
        raise ValueError("flux_measurement_error_correlation must lie in [-1, 1]")

    measurement_variance = (
        sigma_log_fx_dex**2
        + (gamma * sigma_log_fuv_dex) ** 2
        - 2
        * gamma
        * flux_measurement_error_correlation
        * sigma_log_fx_dex
        * sigma_log_fuv_dex
    )
    scale = (
        sigma_log_fx_dex**2
        + (gamma * sigma_log_fuv_dex) ** 2
        + 2
        * abs(gamma * flux_measurement_error_correlation)
        * sigma_log_fx_dex
        * sigma_log_fuv_dex
    )
    roundoff_tolerance = 64 * np.finfo(float).eps * scale
    if np.any(measurement_variance < -roundoff_tolerance):
        raise ValueError("invalid negative flux measurement variance")
    sigma_uvx_measurement_dex = np.sqrt(np.maximum(measurement_variance, 0.0))
    sigma_uvx_intrinsic_dex = np.full(
        sigma_uvx_measurement_dex.shape, delta_dex, dtype=float
    )
    sigma_uvx_total_dex = np.hypot(
        sigma_uvx_measurement_dex, sigma_uvx_intrinsic_dex
    )
    sigma_ln_dl = (
        np.log(10.0) * sigma_uvx_total_dex / (2.0 * (1.0 - gamma))
    )
    return FluxPropagation(
        sigma_uvx_measurement_dex=_readonly(sigma_uvx_measurement_dex),
        sigma_uvx_intrinsic_dex=_readonly(sigma_uvx_intrinsic_dex),
        sigma_uvx_total_dex=_readonly(sigma_uvx_total_dex),
        sigma_ln_dl=_readonly(sigma_ln_dl),
    )


def realize_flux_uncertainties(
    table: FluxErrorTable,
    redshift,
    *,
    gamma,
    delta_dex,
    flux_measurement_error_correlation,
    mode,
    rng=None,
) -> FluxUncertaintyRealization:
    """Sample or deterministically realize flux uncertainties at mock redshifts."""
    distribution = interpolate_flux_error_table(table, redshift)
    if mode == "asimov":
        sigma_log_fuv_dex = distribution.median_sigma_log_fuv_dex
        sigma_log_fx_dex = distribution.median_sigma_log_fx_dex
    elif mode == "random":
        if not isinstance(rng, np.random.Generator):
            raise ValueError("random flux-error realization requires an np.random.Generator")
        a = rng.standard_normal(distribution.redshift.shape)
        b = rng.standard_normal(distribution.redshift.shape)
        size_correlation = distribution.corr_ln_uncertainty_sizes
        tolerance = 64 * np.finfo(float).eps
        if np.any(size_correlation < -1 - tolerance) or np.any(
            size_correlation > 1 + tolerance
        ):
            raise ValueError("invalid interpolated uncertainty-size correlation")
        size_correlation = np.clip(size_correlation, -1.0, 1.0)
        sigma_log_fuv_dex = _readonly(
            np.exp(
                np.log(distribution.median_sigma_log_fuv_dex)
                + distribution.scatter_ln_sigma_log_fuv * a
            )
        )
        sigma_log_fx_dex = _readonly(
            np.exp(
                np.log(distribution.median_sigma_log_fx_dex)
                + distribution.scatter_ln_sigma_log_fx
                * (
                    size_correlation * a
                    + np.sqrt(1.0 - size_correlation**2) * b
                )
            )
        )
    else:
        raise ValueError("flux-error realization mode must be 'random' or 'asimov'")

    propagation = propagate_flux_uncertainties(
        sigma_log_fuv_dex=sigma_log_fuv_dex,
        sigma_log_fx_dex=sigma_log_fx_dex,
        gamma=gamma,
        delta_dex=delta_dex,
        flux_measurement_error_correlation=flux_measurement_error_correlation,
    )
    return FluxUncertaintyRealization(
        sigma_log_fuv_dex=sigma_log_fuv_dex,
        sigma_log_fx_dex=sigma_log_fx_dex,
        sigma_uvx_measurement_dex=propagation.sigma_uvx_measurement_dex,
        sigma_uvx_intrinsic_dex=propagation.sigma_uvx_intrinsic_dex,
        sigma_uvx_total_dex=propagation.sigma_uvx_total_dex,
        sigma_ln_dl=propagation.sigma_ln_dl,
        corr_ln_uncertainty_sizes=distribution.corr_ln_uncertainty_sizes,
        effective_reference_count=distribution.effective_reference_count,
        smoothing_width_z=distribution.smoothing_width_z,
        gamma=float(gamma),
        delta_dex=float(delta_dex),
        flux_measurement_error_correlation=float(flux_measurement_error_correlation),
    )
