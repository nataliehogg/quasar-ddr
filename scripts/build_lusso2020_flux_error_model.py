#!/usr/bin/env python3
"""Build the offline Lusso et al. (2020) flux-uncertainty runtime table."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np


TABLE_COLUMNS = (
    "z",
    "median_sigma_log_fuv_dex",
    "scatter_ln_sigma_log_fuv",
    "median_sigma_log_fx_dex",
    "scatter_ln_sigma_log_fx",
    "corr_ln_uncertainty_sizes",
    "effective_reference_count",
    "smoothing_width_z",
)
SOURCE_COLUMN_NAMES = ("z", "e_logFUV", "e_logFX")
SOURCE_COLUMN_MAPPING = {
    "z": {
        "byte_range_1_indexed_inclusive": "52-57",
        "meaning": "redshift",
        "unit": "dimensionless",
    },
    "e_logFUV": {
        "byte_range_1_indexed_inclusive": "73-79",
        "meaning": "uncertainty in log10 F_UV",
        "unit": "dex",
    },
    "e_logFX": {
        "byte_range_1_indexed_inclusive": "95-101",
        "meaning": "uncertainty in log10 F_X",
        "unit": "dex",
    },
}
REFERENCE_IDENTIFIER = "J/A+A/642/A150"
REFERENCE_URL = "https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/642/A150"
REFERENCE_CITATION = (
    "Lusso et al. (2020), Quasars as standard candles III, "
    "Astronomy & Astrophysics, 642, A150"
)
CSV_FILENAME = "lusso2020_flux_error_model.csv"
METADATA_FILENAME = "lusso2020_flux_error_model_metadata.json"
REFERENCE_FIGURE_FILENAME = "lusso2020_flux_error_reference.pdf"
PROPAGATED_FIGURE_FILENAME = "lusso2020_propagated_distance_uncertainty.pdf"
_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class InvalidSourceRow:
    line_number: int
    reason: str


@dataclass(frozen=True)
class SourceFluxErrors:
    redshift: np.ndarray
    sigma_log_fuv_dex: np.ndarray
    sigma_log_fx_dex: np.ndarray
    source_sha256: str
    total_row_count: int
    invalid_rows: tuple[InvalidSourceRow, ...]

    @property
    def valid_row_count(self) -> int:
        return int(np.asarray(self.redshift).size)

    @property
    def source_column_names(self) -> tuple[str, str, str]:
        return SOURCE_COLUMN_NAMES


@dataclass(frozen=True)
class BuilderSettings:
    retrieval_date: str
    grid_spacing_z: float = 0.05
    bandwidth_min_z: float = 0.05
    bandwidth_step_z: float = 0.05
    bandwidth_max_z: float = 1.00
    target_effective_count: float = 40.0
    minimum_effective_count: float = 10.0
    lower_quantile: float = 0.16
    upper_quantile: float = 0.84
    winsor_lower_quantile: float = 0.01
    winsor_upper_quantile: float = 0.99
    float_decimal_places: int = 10
    overwrite: bool = False

    def __post_init__(self) -> None:
        try:
            date.fromisoformat(self.retrieval_date)
        except (TypeError, ValueError) as error:
            raise ValueError("retrieval_date must use ISO format YYYY-MM-DD") from error
        positive_names = (
            "grid_spacing_z",
            "bandwidth_min_z",
            "bandwidth_step_z",
            "bandwidth_max_z",
            "target_effective_count",
            "minimum_effective_count",
        )
        for name in positive_names:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.bandwidth_min_z > self.bandwidth_max_z:
            raise ValueError("minimum bandwidth cannot exceed maximum bandwidth")
        if not (
            0 <= self.winsor_lower_quantile < self.lower_quantile < 0.5
            < self.upper_quantile < self.winsor_upper_quantile <= 1
        ):
            raise ValueError("builder quantiles must be strictly ordered")
        if self.float_decimal_places < 1:
            raise ValueError("float_decimal_places must be positive")


@dataclass(frozen=True)
class FluxErrorRow:
    z: float
    median_sigma_log_fuv_dex: float
    scatter_ln_sigma_log_fuv: float
    median_sigma_log_fx_dex: float
    scatter_ln_sigma_log_fx: float
    corr_ln_uncertainty_sizes: float
    effective_reference_count: float
    smoothing_width_z: float


@dataclass(frozen=True)
class RuntimeArtifacts:
    csv_path: Path
    metadata_path: Path
    source_sha256: str
    csv_sha256: str


@dataclass(frozen=True)
class BuildOutputs:
    source: SourceFluxErrors
    rows: tuple[FluxErrorRow, ...]
    runtime_artifacts: RuntimeArtifacts
    figure_paths: tuple[Path, Path]


def _readonly(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    result.setflags(write=False)
    return result


def read_lusso_table3(path: Path) -> SourceFluxErrors:
    """Read the three required fields from a local fixed-width VizieR table."""
    path = Path(path)
    try:
        source_bytes = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read local Lusso Table 3 source {path}") from error
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    redshift = []
    sigma_fuv = []
    sigma_fx = []
    invalid_rows = []
    total_row_count = 0
    for line_number, raw_line in enumerate(source_bytes.splitlines(), start=1):
        if not raw_line.strip():
            continue
        total_row_count += 1
        try:
            line = raw_line.decode("ascii")
        except UnicodeDecodeError:
            invalid_rows.append(
                InvalidSourceRow(line_number, "row is not ASCII fixed-width data")
            )
            continue

        values = []
        malformed_name = None
        for name, field_slice in zip(
            SOURCE_COLUMN_NAMES, (slice(51, 57), slice(72, 79), slice(94, 101))
        ):
            field = line[field_slice]
            try:
                values.append(float(field))
            except ValueError:
                malformed_name = name
                break
        if malformed_name is not None:
            invalid_rows.append(
                InvalidSourceRow(
                    line_number,
                    f"invalid or missing numeric field {malformed_name}",
                )
            )
            continue

        z_value, fuv_value, fx_value = values
        if not all(np.isfinite(values)):
            invalid_rows.append(
                InvalidSourceRow(line_number, "non-finite required source value")
            )
            continue
        if fuv_value <= 0 or fx_value <= 0:
            invalid_rows.append(
                InvalidSourceRow(
                    line_number,
                    "non-positive e_logFUV or e_logFX uncertainty",
                )
            )
            continue
        redshift.append(z_value)
        sigma_fuv.append(fuv_value)
        sigma_fx.append(fx_value)

    return SourceFluxErrors(
        redshift=_readonly(redshift),
        sigma_log_fuv_dex=_readonly(sigma_fuv),
        sigma_log_fx_dex=_readonly(sigma_fx),
        source_sha256=source_sha256,
        total_row_count=total_row_count,
        invalid_rows=tuple(invalid_rows),
    )


def _validated_weighted_inputs(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("values and weights must be matching one-dimensional arrays")
    if values.size == 0:
        raise ValueError("weighted statistic requires at least one value")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("weighted statistic inputs must be finite")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("weights must be non-negative with a positive sum")
    selected = weights > 0
    return values[selected], weights[selected]


def weighted_quantile(values, weights, probabilities) -> np.ndarray:
    """Return midpoint-CDF weighted quantiles with deterministic stable sorting."""
    values, weights = _validated_weighted_inputs(values, weights)
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(
        probabilities > 1
    ):
        raise ValueError("quantile probabilities must lie in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative_midpoints = (
        np.cumsum(sorted_weights) - 0.5 * sorted_weights
    ) / np.sum(sorted_weights)
    return np.asarray(
        np.interp(probabilities, cumulative_midpoints, sorted_values), dtype=float
    )


def effective_sample_size(weights) -> float:
    """Return (sum w)^2 / sum(w^2) for non-negative finite weights."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")
    weight_sum = float(np.sum(weights))
    squared_sum = float(np.dot(weights, weights))
    if weight_sum <= 0 or squared_sum <= 0:
        raise ValueError("weights must have a positive sum")
    return weight_sum * weight_sum / squared_sum


def _weighted_winsorized_correlation(x, y, weights, lower, upper) -> float:
    x_limits = weighted_quantile(x, weights, [lower, upper])
    y_limits = weighted_quantile(y, weights, [lower, upper])
    x = np.clip(x, x_limits[0], x_limits[1])
    y = np.clip(y, y_limits[0], y_limits[1])
    normalized_weights = weights / np.sum(weights)
    x_centered = x - np.sum(normalized_weights * x)
    y_centered = y - np.sum(normalized_weights * y)
    covariance = float(np.sum(normalized_weights * x_centered * y_centered))
    x_variance = float(np.sum(normalized_weights * x_centered**2))
    y_variance = float(np.sum(normalized_weights * y_centered**2))
    if x_variance <= 0 or y_variance <= 0:
        return 0.0
    correlation = covariance / math.sqrt(x_variance * y_variance)
    return float(np.clip(correlation, -1.0, 1.0))


def _validate_source(source: SourceFluxErrors) -> tuple[np.ndarray, ...]:
    redshift = np.asarray(source.redshift, dtype=float)
    sigma_fuv = np.asarray(source.sigma_log_fuv_dex, dtype=float)
    sigma_fx = np.asarray(source.sigma_log_fx_dex, dtype=float)
    if (
        redshift.ndim != 1
        or sigma_fuv.ndim != 1
        or sigma_fx.ndim != 1
        or not (redshift.shape == sigma_fuv.shape == sigma_fx.shape)
        or redshift.size == 0
    ):
        raise ValueError("source arrays must be non-empty, matching one-dimensional arrays")
    if not (
        np.all(np.isfinite(redshift))
        and np.all(np.isfinite(sigma_fuv))
        and np.all(np.isfinite(sigma_fx))
    ):
        raise ValueError("source arrays must contain only finite values")
    if np.any(sigma_fuv <= 0) or np.any(sigma_fx <= 0):
        raise ValueError("source flux uncertainties must be strictly positive")
    return redshift, sigma_fuv, sigma_fx


def _candidate_grid(redshift: np.ndarray, spacing: float) -> np.ndarray:
    tolerance = 16 * np.finfo(float).eps
    first_tick = math.ceil(float(np.min(redshift)) / spacing - tolerance)
    last_tick = math.floor(float(np.max(redshift)) / spacing + tolerance)
    if first_tick > last_tick:
        raise ValueError("source redshift range contains no requested grid node")
    return np.arange(first_tick, last_tick + 1, dtype=float) * spacing


def _bandwidths(settings: BuilderSettings) -> tuple[float, ...]:
    count = int(
        math.floor(
            (settings.bandwidth_max_z - settings.bandwidth_min_z)
            / settings.bandwidth_step_z
            + 1e-12
        )
    )
    values = [
        settings.bandwidth_min_z + index * settings.bandwidth_step_z
        for index in range(count + 1)
    ]
    if values[-1] < settings.bandwidth_max_z - 1e-12:
        values.append(settings.bandwidth_max_z)
    return tuple(round(value, 12) for value in values)


def build_flux_error_rows(
    source: SourceFluxErrors, settings: BuilderSettings
) -> tuple[FluxErrorRow, ...]:
    """Construct deterministic adaptive-kernel rows on a zero-anchored grid."""
    redshift, sigma_fuv, sigma_fx = _validate_source(source)
    log_fuv = np.log(sigma_fuv)
    log_fx = np.log(sigma_fx)
    rows = []
    for node in _candidate_grid(redshift, settings.grid_spacing_z):
        selected_width = settings.bandwidth_max_z
        selected_weights = None
        selected_effective_count = None
        for width in _bandwidths(settings):
            log_weights = -0.5 * ((redshift - node) / width) ** 2
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            effective_count = effective_sample_size(weights)
            selected_width = width
            selected_weights = weights
            selected_effective_count = effective_count
            if effective_count >= settings.target_effective_count:
                break

        fuv_quantiles = weighted_quantile(
            sigma_fuv,
            selected_weights,
            [0.5],
        )
        fx_quantiles = weighted_quantile(
            sigma_fx,
            selected_weights,
            [0.5],
        )
        log_fuv_quantiles = weighted_quantile(
            log_fuv,
            selected_weights,
            [settings.lower_quantile, settings.upper_quantile],
        )
        log_fx_quantiles = weighted_quantile(
            log_fx,
            selected_weights,
            [settings.lower_quantile, settings.upper_quantile],
        )
        rows.append(
            FluxErrorRow(
                z=float(round(node, 12)),
                median_sigma_log_fuv_dex=float(fuv_quantiles[0]),
                scatter_ln_sigma_log_fuv=float(
                    (log_fuv_quantiles[1] - log_fuv_quantiles[0]) / 2
                ),
                median_sigma_log_fx_dex=float(fx_quantiles[0]),
                scatter_ln_sigma_log_fx=float(
                    (log_fx_quantiles[1] - log_fx_quantiles[0]) / 2
                ),
                corr_ln_uncertainty_sizes=_weighted_winsorized_correlation(
                    log_fuv,
                    log_fx,
                    selected_weights,
                    settings.winsor_lower_quantile,
                    settings.winsor_upper_quantile,
                ),
                effective_reference_count=float(selected_effective_count),
                smoothing_width_z=float(selected_width),
            )
        )

    first = 0
    last = len(rows)
    while (
        first < last
        and rows[first].effective_reference_count < settings.minimum_effective_count
    ):
        first += 1
    while (
        last > first
        and rows[last - 1].effective_reference_count
        < settings.minimum_effective_count
    ):
        last -= 1
    retained = tuple(rows[first:last])
    if not retained:
        raise ValueError("no grid nodes have sufficient effective reference support")
    return retained


def _settings_provenance(settings: BuilderSettings) -> dict[str, object]:
    return {
        "grid_spacing_z": settings.grid_spacing_z,
        "bandwidth_min_z": settings.bandwidth_min_z,
        "bandwidth_step_z": settings.bandwidth_step_z,
        "bandwidth_max_z": settings.bandwidth_max_z,
        "target_effective_count": settings.target_effective_count,
        "minimum_effective_count": settings.minimum_effective_count,
        "spread_quantiles": [
            settings.lower_quantile,
            settings.upper_quantile,
        ],
        "winsor_quantiles": [
            settings.winsor_lower_quantile,
            settings.winsor_upper_quantile,
        ],
        "float_decimal_places": settings.float_decimal_places,
    }


def _csv_bytes(rows: Sequence[FluxErrorRow], decimal_places: int) -> bytes:
    if not rows:
        raise ValueError("cannot write an empty flux-error table")
    lines = [",".join(TABLE_COLUMNS)]
    number_format = f".{decimal_places}f"
    for row in rows:
        lines.append(
            ",".join(format(float(getattr(row, name)), number_format) for name in TABLE_COLUMNS)
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def _metadata_bytes(
    rows: Sequence[FluxErrorRow],
    source: SourceFluxErrors,
    settings: BuilderSettings,
    csv_sha256: str,
) -> bytes:
    redshift, _, _ = _validate_source(source)
    metadata = {
        "builder_settings": _settings_provenance(settings),
        "csv_columns": list(TABLE_COLUMNS),
        "csv_sha256": csv_sha256,
        "invalid_row_count": len(source.invalid_rows),
        "invalid_row_rule": (
            "Reject rows with missing, malformed, or non-finite required fields, "
            "or non-positive e_logFUV/e_logFX."
        ),
        "model_version": "lusso2020_flux_error_model_v1",
        "reference_citation": REFERENCE_CITATION,
        "reference_identifier": REFERENCE_IDENTIFIER,
        "reference_url": REFERENCE_URL,
        "retrieval_date": settings.retrieval_date,
        "source_column_mapping": SOURCE_COLUMN_MAPPING,
        "source_redshift_max": float(np.max(redshift)),
        "source_redshift_min": float(np.min(redshift)),
        "source_row_count": int(source.total_row_count),
        "source_sha256": source.source_sha256,
        "supported_redshift_max": float(rows[-1].z),
        "supported_redshift_min": float(rows[0].z),
        "table_schema_version": 1,
        "valid_reference_row_count": int(source.valid_row_count),
    }
    return (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_runtime_artifacts(
    rows: Sequence[FluxErrorRow],
    source: SourceFluxErrors,
    settings: BuilderSettings,
    output_dir: Path,
) -> RuntimeArtifacts:
    """Write deterministic schema-v1 CSV and provenance metadata."""
    output_dir = Path(output_dir)
    csv_path = output_dir / CSV_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    existing = [path for path in (csv_path, metadata_path) if path.exists()]
    if existing and not settings.overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing output: {', '.join(map(str, existing))}"
        )

    csv_bytes = _csv_bytes(rows, settings.float_decimal_places)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(csv_bytes)
    csv_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    metadata_bytes = _metadata_bytes(rows, source, settings, csv_sha256)
    metadata_path.write_bytes(metadata_bytes)
    return RuntimeArtifacts(
        csv_path=csv_path,
        metadata_path=metadata_path,
        source_sha256=source.source_sha256,
        csv_sha256=csv_sha256,
    )


def _row_arrays(rows: Sequence[FluxErrorRow]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("reference figures require at least one table row")
    return {
        name: np.asarray([getattr(row, name) for row in rows], dtype=float)
        for name in TABLE_COLUMNS
    }


def _propagated_sigma_ln_dl(sigma_fuv, sigma_fx, gamma=0.60, delta_dex=0.21):
    total_dex = np.sqrt(sigma_fx**2 + (gamma * sigma_fuv) ** 2 + delta_dex**2)
    return np.log(10.0) * total_dex / (2.0 * (1.0 - gamma))


def make_reference_figures(
    source: SourceFluxErrors,
    rows: Sequence[FluxErrorRow],
    figures_dir: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Create the two deterministic publication reference PDFs."""
    figures_dir = Path(figures_dir)
    reference_path = figures_dir / REFERENCE_FIGURE_FILENAME
    propagated_path = figures_dir / PROPAGATED_FIGURE_FILENAME
    existing = [
        path for path in (reference_path, propagated_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing output: {', '.join(map(str, existing))}"
        )
    redshift, sigma_fuv, sigma_fx = _validate_source(source)
    table = _row_arrays(rows)

    import matplotlib.pyplot as plt

    plt.style.use(str(_ROOT / "tools" / "qso_gp_mock_matplotlibrc"))
    plt.rcParams["text.usetex"] = False
    figures_dir.mkdir(parents=True, exist_ok=True)

    z = table["z"]
    fuv_median = table["median_sigma_log_fuv_dex"]
    fuv_scatter = table["scatter_ln_sigma_log_fuv"]
    fx_median = table["median_sigma_log_fx_dex"]
    fx_scatter = table["scatter_ln_sigma_log_fx"]

    source_distance = _propagated_sigma_ln_dl(sigma_fuv, sigma_fx)
    distance_lower = _propagated_sigma_ln_dl(
        fuv_median * np.exp(-fuv_scatter), fx_median * np.exp(-fx_scatter)
    )
    distance_median = _propagated_sigma_ln_dl(fuv_median, fx_median)
    distance_upper = _propagated_sigma_ln_dl(
        fuv_median * np.exp(fuv_scatter), fx_median * np.exp(fx_scatter)
    )

    def plot_propagated_table_panel(axis):
        axis.scatter(
            redshift,
            source_distance,
            s=8,
            alpha=0.18,
            color="black",
            linewidths=0,
            label="Lusso et al. (2020) catalogue",
        )
        axis.fill_between(
            z,
            distance_lower,
            distance_upper,
            color="tab:blue",
            alpha=0.20,
            label="table-based conditional 16th--84th range",
        )
        axis.plot(
            z, distance_median, color="tab:blue", lw=2,
            label="table-based conditional median",
        )
        gamma = 0.60
        for label, residual_dex, linestyle, color in (
            ("Current", 0.24, "--", "tab:red"),
            ("Future", 0.17, "-.", "tab:green"),
            ("Targeted", 0.14, ":", "tab:purple"),
        ):
            reference = np.log(10.0) * residual_dex / (2.0 * (1.0 - gamma))
            axis.axhline(
                reference, color=color, linestyle=linestyle, lw=1.3, label=label
            )
        axis.set_ylabel(r"$\sigma_{\ln D_L}$")
        axis.set_title(
            r"Table-based propagated uncertainty ($\gamma=0.60$, $\delta=0.21$ dex)"
        )
        axis.legend(loc="best", fontsize="small")

    figure, axes = plt.subplots(3, 1, figsize=(7.0, 9.0), sharex=True)
    panels = (
        (
            axes[0],
            sigma_fuv,
            fuv_median,
            fuv_scatter,
            r"$\sigma_{\log F_{\rm UV}}$ [dex]",
            "tab:blue",
        ),
        (
            axes[1],
            sigma_fx,
            fx_median,
            fx_scatter,
            r"$\sigma_{\log F_X}$ [dex]",
            "tab:orange",
        ),
    )
    for axis, source_values, median, scatter, label, color in panels:
        axis.scatter(
            redshift,
            source_values,
            s=8,
            alpha=0.22,
            color=color,
            linewidths=0,
            label="Lusso et al. (2020) catalogue",
        )
        axis.fill_between(
            z,
            median * np.exp(-scatter),
            median * np.exp(scatter),
            color=color,
            alpha=0.20,
            label="conditional 16th--84th range",
        )
        axis.plot(z, median, color=color, lw=1.8, label="conditional median")
        axis.set_yscale("log")
        axis.set_ylabel(label)
        axis.legend(loc="best", fontsize="small")

    plot_propagated_table_panel(axes[2])
    axes[2].set_xlabel("redshift")
    figure.suptitle("Lusso et al. (2020) flux-uncertainty reference")
    figure.tight_layout()
    pdf_metadata = {
        "Creator": "Horizon deterministic Lusso table builder",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(reference_path, metadata=pdf_metadata)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    plot_propagated_table_panel(axis)
    axis.set_xlabel("redshift")
    axis.text(
        0.01,
        0.01,
        "Effective tiers and the table model are alternative prescriptions.",
        transform=axis.transAxes,
        fontsize="small",
        va="bottom",
    )
    figure.tight_layout()
    figure.savefig(propagated_path, metadata=pdf_metadata)
    plt.close(figure)
    return reference_path, propagated_path


def _broad_bin_checks(source: SourceFluxErrors) -> list[str]:
    redshift, sigma_fuv, sigma_fx = _validate_source(source)
    lines = []
    for lower, upper in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 8)):
        selected = (redshift > lower) & (redshift < upper)
        if np.any(selected):
            lines.append(
                f"  {lower} < z < {upper}: "
                f"median e_logFUV={np.median(sigma_fuv[selected]):.6f} dex, "
                f"median e_logFX={np.median(sigma_fx[selected]):.6f} dex "
                f"(N={np.count_nonzero(selected)})"
            )
        else:
            lines.append(f"  {lower} < z < {upper}: no valid rows")
    return lines


def _all_output_paths(output_directory: Path, figures_directory: Path):
    return (
        output_directory / CSV_FILENAME,
        output_directory / METADATA_FILENAME,
        figures_directory / REFERENCE_FIGURE_FILENAME,
        figures_directory / PROPAGATED_FIGURE_FILENAME,
    )


def _promote_staged_outputs(
    staged_and_final_paths, *, overwrite: bool = True
) -> None:
    """Promote a complete staged build, restoring prior files on an error."""
    staged_and_final_paths = tuple(staged_and_final_paths)
    if not overwrite:
        created_paths = []
        try:
            for staged_path, final_path in staged_and_final_paths:
                os.link(staged_path, final_path)
                created_paths.append(final_path)
        except BaseException:
            for final_path in reversed(created_paths):
                if final_path.exists():
                    final_path.unlink()
            raise
        return

    promotions = []
    try:
        for staged_path, final_path in staged_and_final_paths:
            backup_path = None
            if final_path.exists():
                backup_path = staged_path.with_name(f"{staged_path.name}.previous")
                os.replace(final_path, backup_path)
            promotions.append((final_path, backup_path, False))
            os.replace(staged_path, final_path)
            promotions[-1] = (final_path, backup_path, True)
    except BaseException:
        for final_path, backup_path, promoted in reversed(promotions):
            if promoted and final_path.exists():
                final_path.unlink()
            if backup_path is not None and backup_path.exists():
                os.replace(backup_path, final_path)
        raise


@contextmanager
def _destination_locks(directories):
    """Serialize builders sharing either destination directory."""
    identities = sorted(
        {str(Path(directory).resolve()) for directory in directories}
    )
    lock_root = Path(tempfile.gettempdir()) / "horizon-lusso-builder-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for identity in identities:
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            handle = (lock_root / f"{digest}.lock").open("a+b")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def build_all_outputs(
    source_path: Path,
    settings: BuilderSettings,
    output_directory: Path,
    figures_directory: Path,
) -> BuildOutputs:
    """Stage all four outputs and promote them only after a complete build."""
    source_path = Path(source_path)
    output_directory = Path(output_directory)
    figures_directory = Path(figures_directory)
    final_paths = _all_output_paths(output_directory, figures_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    figures_directory.mkdir(parents=True, exist_ok=True)
    with _destination_locks((output_directory, figures_directory)):
        existing = [path for path in final_paths if path.exists()]
        if existing and not settings.overwrite:
            raise FileExistsError(
                f"existing output(s): {', '.join(map(str, existing))}; "
                "pass --overwrite to replace them"
            )

        source = read_lusso_table3(source_path)
        rows = build_flux_error_rows(source, settings)
        staging_settings = replace(settings, overwrite=False)
        with tempfile.TemporaryDirectory(
            prefix=".lusso-runtime-stage-", dir=output_directory
        ) as runtime_temporary, tempfile.TemporaryDirectory(
            prefix=".lusso-figures-stage-", dir=figures_directory
        ) as figures_temporary:
            staged_artifacts = write_runtime_artifacts(
                rows, source, staging_settings, Path(runtime_temporary)
            )
            staged_figures = make_reference_figures(
                source, rows, Path(figures_temporary)
            )
            staged_paths = (
                staged_artifacts.csv_path,
                staged_artifacts.metadata_path,
                *staged_figures,
            )
            try:
                _promote_staged_outputs(
                    zip(staged_paths, final_paths),
                    overwrite=settings.overwrite,
                )
            except FileExistsError as error:
                raise FileExistsError(
                    f"an output appeared during the build; pass --overwrite "
                    f"to replace it: {error.filename}"
                ) from error

    artifacts = RuntimeArtifacts(
        csv_path=final_paths[0],
        metadata_path=final_paths[1],
        source_sha256=staged_artifacts.source_sha256,
        csv_sha256=staged_artifacts.csv_sha256,
    )
    return BuildOutputs(
        source=source,
        rows=rows,
        runtime_artifacts=artifacts,
        figure_paths=(final_paths[2], final_paths[3]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the offline Lusso et al. (2020) flux-error model."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--figures-directory", required=True, type=Path)
    parser.add_argument("--retrieval-date", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing derived artifacts and reference figures",
    )
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        settings = BuilderSettings(
            retrieval_date=args.retrieval_date, overwrite=args.overwrite
        )
        outputs = build_all_outputs(
            args.source,
            settings,
            args.output_directory,
            args.figures_directory,
        )
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))

    source = outputs.source
    rows = outputs.rows
    artifacts = outputs.runtime_artifacts
    figure_paths = outputs.figure_paths
    print(f"valid rows: {source.valid_row_count}")
    print(f"rejected rows: {len(source.invalid_rows)}")
    print(
        "source redshift range: "
        f"{np.min(source.redshift):.6f} to {np.max(source.redshift):.6f}"
    )
    print(
        "supported redshift range: "
        f"{rows[0].z:.6f} to {rows[-1].z:.6f}"
    )
    print("broad-bin median checks:")
    print("\n".join(_broad_bin_checks(source)))
    print(f"source SHA-256: {artifacts.source_sha256}")
    print(f"CSV SHA-256: {artifacts.csv_sha256}")
    print(f"runtime CSV: {artifacts.csv_path}")
    print(f"provenance metadata: {artifacts.metadata_path}")
    for path in figure_paths:
        print(f"reference figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
