"""Generate a paired Horizon quasar mock catalogue from editable settings."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from tools.QSO_specifications import QSOSpecs
from tools.error_models import (
    ResolvedDistanceErrors,
    ResolvedSARMError,
    ResolvedUVXError,
    resolve_sarm_error,
    resolve_uvx_error,
)
from tools.flux_error_models import (
    FluxErrorTable,
    FluxUncertaintyRealization,
    load_flux_error_table,
    realize_flux_uncertainties,
)
from tools.mockchecker import save_paired_mock_diagnostic
from tools.qso_mock import (
    PairedQSOMock,
    generate_paired_log_mock,
    save_paired_log_mock,
)
from tools.theory import angular_diameter_distance, ddr_eta


# =============================================================================
# EDIT THIS CONFIGURATION
# =============================================================================

# Catalogue
# Redshift sampling: tools/QSO_specifications.py
N_QSO = 200
Z_MIN = 0.2
Z_MAX = 2.0

# Cosmological distances & DDR equations: tools/theory.py
# Options: "flat_lcdm", "lcdm", or "flat_wcdm"
COSMOLOGY_MODEL = "flat_lcdm"
H0 = 67.36
OMEGA_M = 0.314
OMEGA_K = 0.0  # Used only for "lcdm".
W0 = -1.0      # Used only for "flat_wcdm".

# Distance-duality ratio:
#     eta(z) = DL(z) / [(1 + z)^2 DA(z)]
#
#     M1: eta(z) = 1 + alpha
#     M2: eta(z) = 1 + alpha + beta*z
#     M3: eta(z) = (1 + z)^alpha
DDR_MODEL = "M1"
ALPHA = -0.075
BETA = 0.0  # Used only for "M2".

# Observational uncertainties:
# Named UV--X residuals and SARM errors: tools/error_models.py
# Tabulated flux-error sampling:        tools/flux_error_models.py

# UV--X relation slope. Used by current, future, targeted, and
# lusso2020_flux_table. It does not modify direct custom sigma_ln_dl.
UVX_GAMMA = 0.60

# UV--X luminosity-distance uncertainty:
# "current", "future", "targeted", "custom", or "lusso2020_flux_table"
UVX_ERROR_OPTION = "current"
CUSTOM_SIGMA_LN_DL = 0.30  # Used only for "custom".

# Used only for "lusso2020_flux_table".
UVX_DELTA_DEX = 0.21
FLUX_MEASUREMENT_ERROR_CORRELATION = 0.0
FLUX_ERROR_TABLE = "data/lusso2020_flux_error_model.csv"

# SARM angular-distance uncertainty:
# "current", "future", "targeted", or "custom"
SARM_ERROR_OPTION = "current"
CUSTOM_SIGMA_LN_DA = 0.20  # Used only for "custom".

# Correlation between final same-object ln(DL) and ln(DA) residuals.
RHO = 0.0

# Random/Asimov draws: tools/qso_mock.py
# "random": one reproducible noisy realization.
# "asimov": observed equals truth at deterministic redshift quantiles.
MOCK_MODE = "random"
RANDOM_SEED = 27  # Ignored for Asimov mocks.

# Output and overwrite safety: generate_mock.py
OUTPUT_DIRECTORY = "mock_data"
OVERWRITE = False

# Diagnostics: tools/mockchecker.py
MAKE_DIAGNOSTIC_PLOT = True

# =============================================================================
# You should not normally need to edit anything below this line.
# =============================================================================


@dataclass(frozen=True)
class MockConfiguration:
    n_qso: int = 200
    z_min: float = 0.2
    z_max: float = 2.0
    cosmology_model: str = "flat_lcdm"
    h0: float = 67.36
    omega_m: float = 0.314
    omega_k: float = 0.0
    w0: float = -1.0
    ddr_model: str = "M1"
    alpha: float = -0.075
    beta: float = 0.0
    uvx_gamma: float = 0.60
    uvx_error_option: str = "current"
    custom_sigma_ln_dl: float = 0.30
    uvx_delta_dex: float = 0.21
    flux_measurement_error_correlation: float = 0.0
    flux_error_table: str = "data/lusso2020_flux_error_model.csv"
    sarm_error_option: str = "current"
    custom_sigma_ln_da: float = 0.20
    rho: float = 0.0
    mock_mode: str = "random"
    random_seed: int = 27
    output_directory: str = "mock_data"
    make_diagnostic_plot: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class ValidatedErrorConfiguration:
    uvx: ResolvedUVXError | None
    sarm: ResolvedSARMError
    flux_table: FluxErrorTable | None


@dataclass(frozen=True)
class GeneratedCatalogue:
    mock: PairedQSOMock
    eta_true: np.ndarray
    uvx_errors: ResolvedUVXError | None
    sarm_errors: ResolvedSARMError
    flux_uncertainties: FluxUncertaintyRealization | None
    flux_table: FluxErrorTable | None

    @property
    def resolved_errors(self) -> ResolvedDistanceErrors:
        """Provide the effective-error view consumed by legacy output code."""
        if self.uvx_errors is None:
            raise NotImplementedError(
                "tabulated flux uncertainties require object-level output "
                "support"
            )
        name = (
            self.uvx_errors.option
            if self.uvx_errors.option == self.sarm_errors.option
            else (
                f"{self.uvx_errors.option}_uvx_"
                f"{self.sarm_errors.option}_sarm"
            )
        )
        return ResolvedDistanceErrors(
            name=name,
            sigma_ln_dl=self.uvx_errors.sigma_ln_dl,
            sigma_ln_da=self.sarm_errors.sigma_ln_da,
            uvx_residual_dex=self.uvx_errors.sigma_uvx_dex,
            errors_include_intrinsic=self.mock.errors_include_intrinsic,
        )


def configuration_from_constants() -> MockConfiguration:
    """Build an immutable configuration from the editable module constants."""
    return MockConfiguration(
        n_qso=N_QSO,
        z_min=Z_MIN,
        z_max=Z_MAX,
        cosmology_model=COSMOLOGY_MODEL,
        h0=H0,
        omega_m=OMEGA_M,
        omega_k=OMEGA_K,
        w0=W0,
        ddr_model=DDR_MODEL,
        alpha=ALPHA,
        beta=BETA,
        uvx_gamma=UVX_GAMMA,
        uvx_error_option=UVX_ERROR_OPTION,
        custom_sigma_ln_dl=CUSTOM_SIGMA_LN_DL,
        uvx_delta_dex=UVX_DELTA_DEX,
        flux_measurement_error_correlation=(
            FLUX_MEASUREMENT_ERROR_CORRELATION
        ),
        flux_error_table=FLUX_ERROR_TABLE,
        sarm_error_option=SARM_ERROR_OPTION,
        custom_sigma_ln_da=CUSTOM_SIGMA_LN_DA,
        rho=RHO,
        mock_mode=MOCK_MODE,
        random_seed=RANDOM_SEED,
        output_directory=OUTPUT_DIRECTORY,
        make_diagnostic_plot=MAKE_DIAGNOSTIC_PLOT,
        overwrite=OVERWRITE,
    )


def validate_configuration(
    config: MockConfiguration,
) -> ValidatedErrorConfiguration:
    """Validate every setting needed to build a catalogue."""
    if isinstance(config.n_qso, (bool, np.bool_)) or not isinstance(
        config.n_qso, (int, np.integer)
    ):
        raise ValueError("N_QSO must be a positive integer")
    if int(config.n_qso) <= 0:
        raise ValueError("N_QSO must be a positive integer")

    finite_settings = (
        ("Z_MIN", config.z_min),
        ("Z_MAX", config.z_max),
        ("H0", config.h0),
        ("OMEGA_M", config.omega_m),
        ("OMEGA_K", config.omega_k),
        ("W0", config.w0),
        ("ALPHA", config.alpha),
        ("BETA", config.beta),
        ("UVX_GAMMA", config.uvx_gamma),
        ("CUSTOM_SIGMA_LN_DL", config.custom_sigma_ln_dl),
        ("UVX_DELTA_DEX", config.uvx_delta_dex),
        (
            "FLUX_MEASUREMENT_ERROR_CORRELATION",
            config.flux_measurement_error_correlation,
        ),
        ("CUSTOM_SIGMA_LN_DA", config.custom_sigma_ln_da),
        ("RHO", config.rho),
    )
    for name, value in finite_settings:
        try:
            is_finite = bool(np.isfinite(value))
        except (TypeError, ValueError):
            is_finite = False
        if not is_finite:
            raise ValueError(f"{name} must be finite")

    if not (
        0 <= config.z_min < config.z_max
    ):
        raise ValueError("require 0 <= Z_MIN < Z_MAX")
    if not 0 <= config.uvx_gamma < 1:
        raise ValueError("UVX_GAMMA must satisfy 0 <= UVX_GAMMA < 1")
    if config.uvx_delta_dex < 0:
        raise ValueError("UVX_DELTA_DEX must be finite and non-negative")
    if config.custom_sigma_ln_dl < 0:
        raise ValueError(
            "CUSTOM_SIGMA_LN_DL must be finite and non-negative"
        )
    if config.custom_sigma_ln_da < 0:
        raise ValueError(
            "CUSTOM_SIGMA_LN_DA must be finite and non-negative"
        )
    if (
        np.iscomplexobj(config.flux_measurement_error_correlation)
        or abs(config.flux_measurement_error_correlation) > 1
    ):
        raise ValueError(
            "FLUX_MEASUREMENT_ERROR_CORRELATION must lie in [-1, 1]"
        )
    if config.mock_mode not in {"random", "asimov"}:
        raise ValueError("MOCK_MODE must be 'random' or 'asimov'")
    if (
        isinstance(config.random_seed, (bool, np.bool_))
        or not isinstance(config.random_seed, (int, np.integer))
        or int(config.random_seed) < 0
        or int(config.random_seed) > np.iinfo(np.int64).max
    ):
        raise ValueError(
            "RANDOM_SEED must be a non-negative signed 64-bit integer"
        )
    try:
        np.random.SeedSequence(int(config.random_seed))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "RANDOM_SEED must be a NumPy-compatible non-negative integer"
        ) from exc
    if np.iscomplexobj(config.rho) or abs(config.rho) > 1:
        raise ValueError("RHO must lie in [-1, 1]")

    uvx_options = (
        "current",
        "future",
        "targeted",
        "custom",
        "lusso2020_flux_table",
    )
    if config.uvx_error_option not in uvx_options:
        valid = ", ".join(uvx_options)
        raise KeyError(
            f"unknown UV--X option {config.uvx_error_option!r}; "
            f"choose one of: {valid}"
        )
    sarm_options = ("current", "future", "targeted", "custom")
    if config.sarm_error_option not in sarm_options:
        valid = ", ".join(sarm_options)
        raise KeyError(
            f"unknown SARM option {config.sarm_error_option!r}; "
            f"choose one of: {valid}"
        )

    sarm = resolve_sarm_error(
        config.sarm_error_option,
        custom_sigma_ln_da=config.custom_sigma_ln_da,
    )
    if config.uvx_error_option == "lusso2020_flux_table":
        try:
            table_path = Path(config.flux_error_table)
        except TypeError as error:
            raise ValueError("FLUX_ERROR_TABLE must be a path") from error
        if not table_path.is_absolute():
            table_path = Path(__file__).resolve().parent / table_path
        flux_table = load_flux_error_table(table_path)
        if (
            config.z_min < flux_table.z[0]
            or config.z_max > flux_table.z[-1]
        ):
            raise ValueError(
                "requested mock redshift interval "
                f"[{config.z_min}, {config.z_max}] lies outside supported "
                "flux-error table interval "
                f"[{flux_table.z[0]}, {flux_table.z[-1]}]"
            )
        uvx = None
    else:
        uvx = resolve_uvx_error(
            config.uvx_error_option,
            gamma=config.uvx_gamma,
            custom_sigma_ln_dl=config.custom_sigma_ln_dl,
        )
        flux_table = None

    angular_diameter_distance(
        np.array([config.z_max]),
        config.h0,
        config.omega_m,
        config.cosmology_model,
        omega_k=config.omega_k,
        w0=config.w0,
    )
    ddr_eta(
        np.array([config.z_min, config.z_max]),
        config.ddr_model,
        config.alpha,
        config.beta,
    )
    return ValidatedErrorConfiguration(uvx, sarm, flux_table)


def dataset_stem(config: MockConfiguration) -> str:
    """Return the legacy-compatible stem for one generated catalogue."""
    return (
        f"{config.ddr_model}alpha{config.alpha}"
        f"beta{config.beta}nQSO{config.n_qso}"
    )


def build_catalogue(config: MockConfiguration) -> GeneratedCatalogue:
    """Construct an in-memory paired catalogue without writing any files."""
    validated = validate_configuration(config)
    normalized_seed = int(config.random_seed)
    if config.mock_mode == "random":
        children = np.random.SeedSequence(normalized_seed).spawn(3)
        redshift_rng = np.random.default_rng(children[0])
        noise_rng = np.random.default_rng(children[1])
        flux_rng = (
            np.random.default_rng(children[2])
            if validated.flux_table is not None
            else None
        )
    else:
        redshift_rng = None
        noise_rng = None
        flux_rng = None
    redshift = QSOSpecs.redshift_distribution(
        {
            "n_QSO": config.n_qso,
            "z_min": config.z_min,
            "z_max": config.z_max,
        },
        rng=redshift_rng,
        mode=config.mock_mode,
    )
    da_true = angular_diameter_distance(
        redshift,
        config.h0,
        config.omega_m,
        config.cosmology_model,
        omega_k=config.omega_k,
        w0=config.w0,
    )
    eta_true = ddr_eta(
        redshift,
        config.ddr_model,
        config.alpha,
        config.beta,
    )
    if validated.flux_table is None:
        flux_uncertainties = None
        sigma_ln_dl = validated.uvx.sigma_ln_dl
        errors_include_intrinsic = validated.uvx.errors_include_intrinsic
    else:
        flux_uncertainties = realize_flux_uncertainties(
            validated.flux_table,
            redshift,
            gamma=config.uvx_gamma,
            delta_dex=config.uvx_delta_dex,
            flux_measurement_error_correlation=(
                config.flux_measurement_error_correlation
            ),
            mode=config.mock_mode,
            rng=flux_rng,
        )
        sigma_ln_dl = flux_uncertainties.sigma_ln_dl
        errors_include_intrinsic = True
    mock = generate_paired_log_mock(
        redshift,
        da_true,
        eta_true,
        sigma_ln_dl,
        validated.sarm.sigma_ln_da,
        rho=config.rho,
        mode=config.mock_mode,
        seed=normalized_seed if config.mock_mode == "random" else None,
        rng=noise_rng,
        errors_include_intrinsic=errors_include_intrinsic,
    )
    return GeneratedCatalogue(
        mock,
        eta_true,
        validated.uvx,
        validated.sarm,
        flux_uncertainties,
        validated.flux_table,
    )


def _target_paths(config: MockConfiguration) -> dict[str, Path]:
    root = Path(config.output_directory)
    stem = dataset_stem(config)
    paths = {
        "csv": root / f"{stem}.csv",
        "metadata": root / f"{stem}_metadata.json",
        "npz": root / f"{stem}_paired_dataset.npz",
        "dl_text": root / f"{stem}_dl_dataset.txt",
        "da_text": root / f"{stem}_da_dataset.txt",
    }
    if config.make_diagnostic_plot:
        paths["diagnostic"] = _diagnostic_path(config)
    return paths


def _diagnostic_path(config: MockConfiguration) -> Path:
    return (
        Path(config.output_directory)
        / f"{dataset_stem(config)}_diagnostic.png"
    )


def _configuration_metadata(config: MockConfiguration) -> dict:
    values = asdict(config)
    return {
        name: value.item() if isinstance(value, np.generic) else value
        for name, value in values.items()
    }


def _flux_component_columns(generated: GeneratedCatalogue) -> dict[str, object]:
    """Return schema-v2 UV--X component columns for one catalogue."""
    size = generated.mock.redshift.size
    empty = [""] * size
    names = (
        "sigma_log_fuv_dex",
        "sigma_log_fx_dex",
        "sigma_uvx_measurement_dex",
        "sigma_uvx_intrinsic_dex",
        "sigma_uvx_total_dex",
        "corr_ln_uncertainty_sizes",
        "flux_model_effective_count",
        "flux_model_smoothing_width_z",
    )
    if generated.flux_uncertainties is not None:
        flux = generated.flux_uncertainties
        return {
            "sigma_log_fuv_dex": flux.sigma_log_fuv_dex,
            "sigma_log_fx_dex": flux.sigma_log_fx_dex,
            "sigma_uvx_measurement_dex": flux.sigma_uvx_measurement_dex,
            "sigma_uvx_intrinsic_dex": flux.sigma_uvx_intrinsic_dex,
            "sigma_uvx_total_dex": flux.sigma_uvx_total_dex,
            "corr_ln_uncertainty_sizes": flux.corr_ln_uncertainty_sizes,
            "flux_model_effective_count": flux.effective_reference_count,
            "flux_model_smoothing_width_z": flux.smoothing_width_z,
        }
    columns = {name: empty for name in names}
    if generated.uvx_errors.sigma_uvx_dex is not None:
        columns["sigma_uvx_total_dex"] = np.full(
            size,
            generated.uvx_errors.sigma_uvx_dex,
        )
    return columns


def _uvx_error_metadata(
    config: MockConfiguration,
    generated: GeneratedCatalogue,
) -> dict[str, object]:
    """Describe the active UV--X model with inactive fields set to null."""
    metadata = {
        "option": config.uvx_error_option,
        "gamma": float(config.uvx_gamma),
        "gamma_used": bool(
            generated.flux_uncertainties is not None
            or generated.uvx_errors.gamma_used
        ),
        "sigma_uvx_dex": None,
        "custom_sigma_ln_dl": None,
        "delta_dex": None,
        "flux_measurement_error_correlation": None,
        "distribution": None,
        "interpolation": None,
        "table_path": None,
        "table_sha256": None,
        "table_schema_version": None,
        "reference": None,
        "supported_redshift_interval": None,
        "redshifts_generated_independently": None,
    }
    if generated.flux_uncertainties is None:
        if generated.uvx_errors.option == "custom":
            metadata["custom_sigma_ln_dl"] = float(
                generated.uvx_errors.sigma_ln_dl
            )
        else:
            metadata["sigma_uvx_dex"] = float(
                generated.uvx_errors.sigma_uvx_dex
            )
        return metadata

    table = generated.flux_table
    if table is None:
        raise ValueError("table-mode catalogue is missing retained provenance")
    metadata.update(
        {
            "delta_dex": float(config.uvx_delta_dex),
            "flux_measurement_error_correlation": float(
                config.flux_measurement_error_correlation
            ),
            "distribution": (
                "correlated lognormal per-object flux uncertainties"
            ),
            "interpolation": (
                "log-linear medians and linear distribution parameters "
                "in redshift"
            ),
            "table_path": str(table.csv_path.resolve()),
            "table_sha256": table.csv_sha256,
            "table_schema_version": int(table.schema_version),
            "reference": table.provenance["reference_identifier"],
            "supported_redshift_interval": [
                float(table.z[0]),
                float(table.z[-1]),
            ],
            "redshifts_generated_independently": True,
        }
    )
    return metadata


def _diagnostic_metadata(
    metadata: dict[str, object],
    generated: GeneratedCatalogue,
) -> dict[str, object]:
    """Add plot-only uncertainty summaries without changing output schema."""
    sigma_ln_dl = generated.mock.sigma_ln_dl
    diagnostic_uncertainty_summary = {
        "sigma_ln_dl": {
            "min": float(np.min(sigma_ln_dl)),
            "median": float(np.median(sigma_ln_dl)),
            "max": float(np.max(sigma_ln_dl)),
        }
    }
    return {
        **metadata,
        "diagnostic_uncertainty_summary": diagnostic_uncertainty_summary,
    }


class _OutputRecoveryError(RuntimeError):
    """Signal that output backups must be recovered manually."""


def _promote_staged_outputs(
    *,
    config: MockConfiguration,
    paths: dict[str, Path],
    staged_paths: dict[str, Path],
    conflict_candidates: list[Path],
    staging_root: Path,
) -> None:
    """Promote one complete staged set, rolling back every recoverable path."""
    backup_directory = staging_root / "previous"
    backup_directory.mkdir()
    backups = {}
    promoted = []
    linked_promotions = []
    try:
        if not config.overwrite:
            late_conflicts = [
                path for path in conflict_candidates if path.exists()
            ]
            if late_conflicts:
                joined = "\n".join(
                    f"  - {path}" for path in late_conflicts
                )
                raise FileExistsError(
                    "output files appeared during staging; "
                    "set OVERWRITE = True to replace them:\n"
                    + joined
                )
        else:
            for final_path in conflict_candidates:
                if final_path.exists():
                    backup_path = backup_directory / final_path.name
                    final_path.replace(backup_path)
                    backups[final_path] = backup_path

        for name, final_path in paths.items():
            if config.overwrite:
                staged_paths[name].replace(final_path)
                promoted.append(final_path)
            else:
                staged_stat = staged_paths[name].lstat()
                staged_identity = (staged_stat.st_dev, staged_stat.st_ino)
                try:
                    os.link(staged_paths[name], final_path)
                except FileExistsError as error:
                    raise FileExistsError(
                        "output files appeared during staging; "
                        "set OVERWRITE = True to replace them:\n"
                        f"  - {final_path}"
                    ) from error
                linked_promotions.append((final_path, staged_identity))
    except BaseException as promotion_error:
        recovery_errors = []
        for final_path in promoted:
            if final_path.exists():
                try:
                    final_path.unlink()
                except OSError as error:
                    recovery_errors.append((final_path, error))
        for final_path, owned_identity in linked_promotions:
            try:
                current_stat = final_path.lstat()
            except FileNotFoundError:
                continue
            except OSError as error:
                recovery_errors.append((final_path, error))
                continue
            if (current_stat.st_dev, current_stat.st_ino) != owned_identity:
                continue
            try:
                final_path.unlink()
            except OSError as error:
                recovery_errors.append((final_path, error))
        for final_path, backup_path in backups.items():
            if backup_path.exists():
                try:
                    backup_path.replace(final_path)
                except OSError as error:
                    recovery_errors.append((final_path, error))
        if recovery_errors:
            details = "; ".join(
                f"{path}: {error}" for path, error in recovery_errors
            )
            raise _OutputRecoveryError(
                f"output recovery was incomplete ({details}); "
                f"recovery files retained at {staging_root}"
            ) from promotion_error
        raise


def write_catalogue(
    config: MockConfiguration, generated: GeneratedCatalogue
) -> dict[str, Path]:
    """Write configured outputs after a same-stem, all-file preflight."""
    mock = generated.mock
    paths = _target_paths(config)
    diagnostic_path = _diagnostic_path(config)
    conflict_candidates = list(paths.values())
    if diagnostic_path not in conflict_candidates:
        conflict_candidates.append(diagnostic_path)
    conflicts = [path for path in conflict_candidates if path.exists()]
    if conflicts and not config.overwrite:
        joined = "\n".join(f"  - {path}" for path in conflicts)
        raise FileExistsError(
            "output files already exist; set OVERWRITE = True to replace them:\n"
            + joined
        )

    fieldnames = [
        "z",
        "eta_true",
        "dl_true_mpc",
        "dl_observed_mpc",
        "sigma_ln_dl",
        "sigma_log_fuv_dex",
        "sigma_log_fx_dex",
        "sigma_uvx_measurement_dex",
        "sigma_uvx_intrinsic_dex",
        "sigma_uvx_total_dex",
        "corr_ln_uncertainty_sizes",
        "flux_model_effective_count",
        "flux_model_smoothing_width_z",
        "da_true_mpc",
        "da_observed_mpc",
        "sigma_ln_da",
        "rho",
    ]
    flux_columns = _flux_component_columns(generated)
    rows = zip(
        mock.redshift,
        generated.eta_true,
        mock.dl_true,
        mock.dl_observed,
        mock.sigma_ln_dl,
        *(flux_columns[name] for name in fieldnames[5:13]),
        mock.da_true,
        mock.da_observed,
        mock.sigma_ln_da,
        mock.rho,
    )
    effective_omega_k = (
        float(config.omega_k) if config.cosmology_model == "lcdm" else 0.0
    )
    effective_w0 = (
        float(config.w0) if config.cosmology_model == "flat_wcdm" else -1.0
    )
    metadata = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "distance_unit": "Mpc",
        "error_unit": "natural-log distance standard deviation",
        "errors_include_intrinsic": mock.errors_include_intrinsic,
        "configuration": _configuration_metadata(config),
        "uvx_errors": _uvx_error_metadata(config, generated),
        "sarm_errors": asdict(generated.sarm_errors),
        "cosmology": {
            "model": config.cosmology_model,
            "H0_km_s_Mpc": float(config.h0),
            "omega_m": float(config.omega_m),
            "omega_k": effective_omega_k,
            "w0": effective_w0,
            "radiation_included": False,
        },
        "ddr": {
            "definition": "eta(z) = DL(z) / ((1 + z)^2 DA(z))",
            "model": config.ddr_model,
            "model_equation": {
                "M1": "eta(z) = 1 + alpha",
                "M2": "eta(z) = 1 + alpha + beta*z",
                "M3": "eta(z) = (1 + z)^alpha",
            }[config.ddr_model],
            "alpha": float(config.alpha),
            "beta": float(config.beta),
        },
        "mock_generation": {
            "mode": config.mock_mode,
            "master_seed": (
                int(config.random_seed) if config.mock_mode == "random" else None
            ),
            "seed_semantics": (
                (
                    "independent redshift, distance-noise, and "
                    "flux-uncertainty streams derived from master_seed"
                )
                if (
                    config.mock_mode == "random"
                    and generated.flux_uncertainties is not None
                )
                else (
                    "independent redshift and noise streams derived "
                    "from master_seed"
                )
                if config.mock_mode == "random"
                else "deterministic midpoint redshifts; no random noise"
            ),
        },
        "source_files": {
            "redshifts": "tools/QSO_specifications.py",
            "theory": "tools/theory.py",
            "errors": "tools/error_models.py",
            "draws": "tools/qso_mock.py",
            "diagnostics": "tools/mockchecker.py",
        },
        "columns": {
            "z": "QSO redshift",
            "eta_true": "injected DL / ((1+z)^2 DA)",
            "dl_true_mpc": "true luminosity distance in Mpc",
            "dl_observed_mpc": "mock observed luminosity distance in Mpc",
            "sigma_ln_dl": "standard deviation in natural-log luminosity distance",
            "sigma_log_fuv_dex": (
                "one-object F_UV measurement uncertainty in log10 flux"
            ),
            "sigma_log_fx_dex": (
                "one-object F_X measurement uncertainty in log10 flux"
            ),
            "sigma_uvx_measurement_dex": (
                "propagated UV--X measurement scatter in dex"
            ),
            "sigma_uvx_intrinsic_dex": (
                "intrinsic UV--X relation scatter in dex"
            ),
            "sigma_uvx_total_dex": "total UV--X relation scatter in dex",
            "corr_ln_uncertainty_sizes": (
                "correlation of sampled log flux-uncertainty sizes"
            ),
            "flux_model_effective_count": (
                "effective reference-object count in the flux-error model"
            ),
            "flux_model_smoothing_width_z": (
                "redshift smoothing width of the flux-error model"
            ),
            "da_true_mpc": "true angular-diameter distance in Mpc",
            "da_observed_mpc": "mock observed angular-diameter distance in Mpc",
            "sigma_ln_da": "standard deviation in natural-log angular distance",
            "rho": "same-object correlation of the two log-distance errors",
        },
    }
    metadata_json = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    with np.errstate(over="ignore", invalid="ignore"):
        sigma_dl_mpc = mock.dl_true * np.sqrt(np.expm1(mock.sigma_ln_dl**2))
        sigma_da_mpc = mock.da_true * np.sqrt(np.expm1(mock.sigma_ln_da**2))
    if not (
        np.all(np.isfinite(sigma_dl_mpc))
        and np.all(np.isfinite(sigma_da_mpc))
        and np.all(sigma_dl_mpc >= 0)
        and np.all(sigma_da_mpc >= 0)
    ):
        raise ValueError(
            "legacy absolute distance errors must be finite and non-negative"
        )

    output_directory = Path(config.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(
        prefix=f".{dataset_stem(config)}.staging-",
        dir=output_directory,
    ))
    retain_recovery = False
    try:
        staged_paths = {
            name: staging_root / path.name
            for name, path in paths.items()
        }
        with staged_paths["csv"].open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fieldnames)
            writer.writerows(rows)
        staged_paths["metadata"].write_text(metadata_json)
        save_paired_log_mock(
            staged_paths["npz"],
            mock,
            flux_uncertainties=generated.flux_uncertainties,
        )
        np.savetxt(
            staged_paths["dl_text"],
            np.column_stack([mock.redshift, mock.dl_observed, sigma_dl_mpc]),
            header="z dl_mpc sigma_dl_mpc",
        )
        np.savetxt(
            staged_paths["da_text"],
            np.column_stack([mock.redshift, mock.da_observed, sigma_da_mpc]),
            header="z da_mpc sigma_da_mpc",
        )
        if config.make_diagnostic_plot:
            save_paired_mock_diagnostic(
                staged_paths["diagnostic"],
                mock,
                metadata=_diagnostic_metadata(metadata, generated),
                title=dataset_stem(config),
            )

        _promote_staged_outputs(
            config=config,
            paths=paths,
            staged_paths=staged_paths,
            conflict_candidates=conflict_candidates,
            staging_root=staging_root,
        )
    except _OutputRecoveryError:
        retain_recovery = True
        raise
    finally:
        if not retain_recovery:
            shutil.rmtree(staging_root)
    return paths


def print_summary(
    config: MockConfiguration,
    generated: GeneratedCatalogue,
    paths: dict[str, Path],
) -> None:
    """Print the resolved catalogue settings and saved paths."""
    mock = generated.mock
    print("Horizon paired quasar mock generated")
    print(f"  objects: {mock.redshift.size}")
    print(f"  redshift range: {mock.redshift.min():.4f} to {mock.redshift.max():.4f}")
    print(f"  cosmology: {config.cosmology_model}")
    print(f"  DDR: {config.ddr_model} (alpha={config.alpha}, beta={config.beta})")
    if generated.flux_uncertainties is None:
        uvx = generated.uvx_errors
        gamma_text = (
            f"{uvx.gamma:.6g} (used)"
            if uvx.gamma_used
            else "not used"
        )
        uvx_text = (
            f"  UV-X: {uvx.option} | gamma: {gamma_text}"
        )
        if uvx.sigma_uvx_dex is not None:
            uvx_text += f" | residual: {uvx.sigma_uvx_dex:.6g} dex"
        uvx_text += f" | sigma_ln_dl: {uvx.sigma_ln_dl:.6g}"
        print(uvx_text)
    else:
        flux = generated.flux_uncertainties
        print(
            f"  UV-X: {flux.model_name}"
            f" | gamma: {flux.gamma:.6g} (used)"
            f" | delta: {flux.delta_dex:.6g} dex"
            " | flux-error correlation: "
            f"{flux.flux_measurement_error_correlation:.6g}"
        )
        for label, values in (
            ("UV uncertainty (dex)", flux.sigma_log_fuv_dex),
            ("X-ray uncertainty (dex)", flux.sigma_log_fx_dex),
            ("sigma_ln_dl", mock.sigma_ln_dl),
        ):
            print(
                f"    {label} min/median/max: "
                f"{np.min(values):.6g} / "
                f"{np.median(values):.6g} / "
                f"{np.max(values):.6g}"
            )
        table = generated.flux_table
        print(f"    table support: [{table.z[0]:.6g}, {table.z[-1]:.6g}]")
        print(f"    table: {table.csv_path.resolve()}")
    print(
        f"  SARM: {generated.sarm_errors.option}"
        f" | sigma_ln_da: {generated.sarm_errors.sigma_ln_da:.6g}"
        f" | rho: {config.rho:.6g}"
    )
    mode = config.mock_mode
    seed_text = f", seed={config.random_seed}" if mode == "random" else ""
    print(f"  mode: {mode}{seed_text}")
    print("  outputs:")
    for path in paths.values():
        print(f"    {path}")


def main() -> None:
    """Generate and save the configured catalogue."""
    config = configuration_from_constants()
    generated = build_catalogue(config)
    paths = write_catalogue(config, generated)
    print_summary(config, generated, paths)


if __name__ == "__main__":
    main()
