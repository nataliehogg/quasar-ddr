import numpy as np
from scipy.interpolate import interp1d
import os
from dataclasses import dataclass
import warnings

from .QSO_specifications import QSOSpecs
from .constants import SPEED_OF_LIGHT_KMS
from .error_models import get_error_scenario


@dataclass(frozen=True)
class PairedQSOMock:
    """Paired, positive luminosity- and angular-distance mock observations."""

    redshift: np.ndarray
    dl_true: np.ndarray
    da_true: np.ndarray
    dl_observed: np.ndarray
    da_observed: np.ndarray
    sigma_ln_dl: np.ndarray
    sigma_ln_da: np.ndarray
    rho: np.ndarray
    mode: str
    seed: int | None
    errors_include_intrinsic: bool


def generate_paired_log_mock(
    redshift,
    da_true,
    eta,
    sigma_ln_dl,
    sigma_ln_da,
    rho=0.0,
    mode="asimov",
    seed=None,
    rng=None,
    errors_include_intrinsic=True,
):
    """Generate synchronized DDR mocks with Gaussian noise in log distance.

    ``eta`` may be a scalar, an array matching ``redshift``, or a callable.
    Random catalogues are unconditional draws; no residual-moment selection is
    performed.
    """
    redshift = np.asarray(redshift, dtype=float)
    da_true = np.asarray(da_true, dtype=float)
    if redshift.ndim != 1 or da_true.shape != redshift.shape:
        raise ValueError("redshift and da_true must be matching one-dimensional arrays")
    if not np.all(np.isfinite(redshift)) or not np.all(np.isfinite(da_true)):
        raise ValueError("redshift and da_true must contain only finite values")
    if np.any(redshift <= -1) or np.any(da_true <= 0):
        raise ValueError("redshift must exceed -1 and da_true must be positive")

    eta_values = eta(redshift) if callable(eta) else eta
    eta_values = np.broadcast_to(np.asarray(eta_values, dtype=float), redshift.shape)
    if not np.all(np.isfinite(eta_values)):
        raise ValueError("eta must contain only finite values")
    if np.any(eta_values <= 0):
        raise ValueError("eta must be strictly positive")

    sigma_ln_dl = np.broadcast_to(
        np.asarray(sigma_ln_dl, dtype=float), redshift.shape
    ).copy()
    sigma_ln_da = np.broadcast_to(
        np.asarray(sigma_ln_da, dtype=float), redshift.shape
    ).copy()
    rho = np.broadcast_to(np.asarray(rho, dtype=float), redshift.shape).copy()
    if not (
        np.all(np.isfinite(sigma_ln_dl))
        and np.all(np.isfinite(sigma_ln_da))
        and np.all(np.isfinite(rho))
    ):
        raise ValueError("log-distance scatters and rho must be finite")
    if np.any(sigma_ln_dl < 0) or np.any(sigma_ln_da < 0):
        raise ValueError("log-distance scatters must be non-negative")
    if np.any(np.abs(rho) > 1):
        raise ValueError("rho must lie in [-1, 1]")
    if mode not in {"asimov", "random"}:
        raise ValueError("mode must be 'asimov' or 'random'")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    dl_true = eta_values * (1.0 + redshift) ** 2 * da_true
    if mode == "asimov":
        dl_observed = dl_true.copy()
        da_observed = da_true.copy()
    else:
        rng = np.random.default_rng(seed) if rng is None else rng
        common = rng.standard_normal(redshift.size)
        independent = rng.standard_normal(redshift.size)
        dl_noise = sigma_ln_dl * common
        da_noise = sigma_ln_da * (
            rho * common + np.sqrt(np.maximum(1.0 - rho**2, 0.0)) * independent
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            dl_observed = dl_true * np.exp(dl_noise)
            da_observed = da_true * np.exp(da_noise)

    distances = (dl_true, da_true, dl_observed, da_observed)
    if any(
        not np.all(np.isfinite(distance)) or np.any(distance <= 0)
        for distance in distances
    ):
        raise ValueError(
            "true and observed distances must be finite and strictly positive"
        )

    return PairedQSOMock(
        redshift=redshift.copy(),
        dl_true=dl_true,
        da_true=da_true.copy(),
        dl_observed=dl_observed,
        da_observed=da_observed,
        sigma_ln_dl=sigma_ln_dl,
        sigma_ln_da=sigma_ln_da,
        rho=rho,
        mode=mode,
        seed=seed,
        errors_include_intrinsic=bool(errors_include_intrinsic),
    )


def save_paired_log_mock(path, mock, *, flux_uncertainties=None):
    """Save a paired mock with unit-explicit log-error fields."""
    arrays = {
        "redshift": mock.redshift,
        "dl_true": mock.dl_true,
        "da_true": mock.da_true,
        "dl_observed": mock.dl_observed,
        "da_observed": mock.da_observed,
        "sigma_ln_dl": mock.sigma_ln_dl,
        "sigma_ln_da": mock.sigma_ln_da,
        "rho": mock.rho,
        "mode": np.asarray(mock.mode),
        "seed": np.asarray(-1 if mock.seed is None else mock.seed, dtype=int),
        "errors_include_intrinsic": np.asarray(
            mock.errors_include_intrinsic,
            dtype=bool,
        ),
    }
    if flux_uncertainties is not None:
        component_attributes = {
            "sigma_log_fuv_dex": "sigma_log_fuv_dex",
            "sigma_log_fx_dex": "sigma_log_fx_dex",
            "sigma_uvx_measurement_dex": "sigma_uvx_measurement_dex",
            "sigma_uvx_intrinsic_dex": "sigma_uvx_intrinsic_dex",
            "sigma_uvx_total_dex": "sigma_uvx_total_dex",
            "corr_ln_uncertainty_sizes": "corr_ln_uncertainty_sizes",
            "flux_model_effective_count": "effective_reference_count",
            "flux_model_smoothing_width_z": "smoothing_width_z",
        }
        arrays.update(
            {
                name: np.asarray(
                    getattr(flux_uncertainties, attribute),
                    dtype=float,
                )
                for name, attribute in component_attributes.items()
            }
        )
    np.savez_compressed(path, **arrays)


def load_paired_log_mock(path):
    """Load a :class:`PairedQSOMock` written by ``save_paired_log_mock``."""
    with np.load(path, allow_pickle=False) as data:
        seed = int(data["seed"])
        return PairedQSOMock(
            redshift=data["redshift"].copy(),
            dl_true=data["dl_true"].copy(),
            da_true=data["da_true"].copy(),
            dl_observed=data["dl_observed"].copy(),
            da_observed=data["da_observed"].copy(),
            sigma_ln_dl=data["sigma_ln_dl"].copy(),
            sigma_ln_da=data["sigma_ln_da"].copy(),
            rho=data["rho"].copy(),
            mode=str(data["mode"]),
            seed=None if seed == -1 else seed,
            errors_include_intrinsic=(
                bool(data["errors_include_intrinsic"])
                if "errors_include_intrinsic" in data.files
                else True
            ),
        )

class QSOMockGenerator:
    """
    Generate synchronized mock luminosity and angular diameter distance data.
    """

    def __init__(
        self,
        cosmoparams,
        specifications,
        z_in,
        outpath,
        dl_func,
        da_func,
        sigma_uvx_fid=None,
        beta_dl_meas=None,
        sigma_sarm_fid=None,
        beta_da_meas=None,
        *,
        error_scenario="current",
        sigma_ln_dl=None,
        sigma_ln_da=None,
        rho=0.0,
        mode="asimov",
        seed=None,
        errors_include_intrinsic=True,
    ):
        """Generate one paired catalogue using explicit log-error semantics.

        The four legacy fractional arguments remain as a compatibility bridge.
        New analyses should use ``error_scenario`` or the explicit
        ``sigma_ln_dl``/``sigma_ln_da`` arguments.
        """
        print('Generating synchronized QSO mock datasets in log-distance space.')
        redshift_stream, noise_stream = np.random.SeedSequence(seed).spawn(2)
        self.redshifts = QSOSpecs.redshift_distribution(
            specifications,
            rng=np.random.default_rng(redshift_stream),
            mode=mode,
        )
        hubble_dist = SPEED_OF_LIGHT_KMS / cosmoparams['H0']

        z_in = np.asarray(z_in, dtype=float)
        dl_grid_mpc = hubble_dist * np.asarray(dl_func(z_in), dtype=float)
        self.dl_theoretical = interp1d(z_in, dl_grid_mpc)
        self.da_theoretical = da_func
        dl_true = np.asarray(self.dl_theoretical(self.redshifts), dtype=float)
        da_true = np.asarray(da_func(self.redshifts), dtype=float)
        eta = dl_true / ((1.0 + self.redshifts) ** 2 * da_true)

        legacy = any(
            value is not None
            for value in (
                sigma_uvx_fid, beta_dl_meas, sigma_sarm_fid, beta_da_meas
            )
        )
        if legacy:
            warnings.warn(
                "legacy fractional mock errors are deprecated; use explicit "
                "sigma_ln_dl/sigma_ln_da or error_scenario",
                DeprecationWarning,
                stacklevel=2,
            )
            sigma_ln_dl = np.hypot(
                0.0 if sigma_uvx_fid is None else sigma_uvx_fid,
                0.0 if beta_dl_meas is None else beta_dl_meas,
            )
            sigma_ln_da = np.hypot(
                0.0 if sigma_sarm_fid is None else sigma_sarm_fid,
                0.0 if beta_da_meas is None else beta_da_meas,
            )
        else:
            scenario = get_error_scenario(error_scenario)
            if sigma_ln_dl is None:
                sigma_ln_dl = scenario.sigma_ln_dl
            if sigma_ln_da is None:
                sigma_ln_da = scenario.sarm_sigma_ln

        self.paired_mock = generate_paired_log_mock(
            self.redshifts,
            da_true,
            eta,
            sigma_ln_dl=sigma_ln_dl,
            sigma_ln_da=sigma_ln_da,
            rho=rho,
            mode=mode,
            seed=seed,
            rng=np.random.default_rng(noise_stream),
            errors_include_intrinsic=errors_include_intrinsic,
        )
        self.dl_data = self.paired_mock.dl_observed
        self.da_data = self.paired_mock.da_observed
        self.sigma_ln_dl = self.paired_mock.sigma_ln_dl
        self.sigma_ln_da = self.paired_mock.sigma_ln_da
        self.dl_error = self.paired_mock.dl_true * np.sqrt(
            np.expm1(self.sigma_ln_dl**2)
        )
        self.da_error = self.paired_mock.da_true * np.sqrt(
            np.expm1(self.sigma_ln_da**2)
        )
        self._save_data(outpath)

    def _save_data(self, outpath):
        # 1. Extract the directory from the outpath
        directory = os.path.dirname(outpath)
        
        # 2. Create the directory if it doesn't exist (and it's not the current dir)
        if directory and not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)

        save_paired_log_mock(f"{outpath}_paired_dataset.npz", self.paired_mock)

        # Legacy three-column files retain absolute plotting errors. New
        # likelihoods should load the paired NPZ file above.
        np.savetxt(f"{outpath}_dl_dataset.txt", 
                   np.column_stack([self.redshifts, self.dl_data, self.dl_error]), 
                   delimiter=' ', header='z dl_mpc sigma_dl_mpc')
        np.savetxt(f"{outpath}_da_dataset.txt", 
                   np.column_stack([self.redshifts, self.da_data, self.da_error]), 
                   delimiter=' ', header='z da_mpc sigma_da_mpc')
        print(f"Datasets saved to {outpath}_paired_dataset.npz and legacy text files")

    def mock(self):
        return {
            'z': self.redshifts,
            'dl_value': self.dl_data,
            'dl_error': self.dl_error,
            'da_value': self.da_data,
            'da_error': self.da_error,
            'sigma_ln_dl': self.sigma_ln_dl,
            'sigma_ln_da': self.sigma_ln_da,
            'rho': self.paired_mock.rho,
            'mode': self.paired_mock.mode,
            'seed': self.paired_mock.seed,
        }
