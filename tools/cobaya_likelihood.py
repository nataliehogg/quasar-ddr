"""Cobaya wrapper for the cosmology-independent paired-distance likelihood."""

from pathlib import Path
import warnings

import numpy as np
from cobaya.likelihood import Likelihood

from .ddr_likelihood import paired_distance_log_likelihood
from .qso_mock import load_paired_log_mock


class DDRlike(Likelihood):
    """DDR likelihood in natural-log distance space.

    Paired NPZ datasets store unit-explicit ``sigma_ln_dl`` and
    ``sigma_ln_da`` arrays. Legacy three-column text pairs remain readable, but
    their absolute errors are converted to approximate log errors and emit a
    deprecation warning.
    """

    datapath: str = './mock_data/test'
    model: int | str = 1
    calibration_sigma: float = 0.0

    def _model_name(self):
        if isinstance(self.model, str):
            if self.model not in {
                'M1', 'M2', 'M3', 'constant', 'linear_z',
                'linear_z_over_1pz', 'power_law'
            }:
                raise ValueError(f"unknown DDR model {self.model!r}")
            return self.model
        model_map = {1: 'M1', 2: 'M2', 3: 'M3'}
        try:
            return model_map[self.model]
        except KeyError as exc:
            raise ValueError("model must be 1, 2, 3, or a supported model name") from exc

    def initialize(self):
        self.EMmodel = self._model_name()
        paired_path = Path(self.datapath + '_paired_dataset.npz')
        if paired_path.exists():
            mock = load_paired_log_mock(paired_path)
            self.z = mock.redshift
            self.dl = mock.dl_observed
            self.da = mock.da_observed
            self.sigma_ln_dl = mock.sigma_ln_dl
            self.sigma_ln_da = mock.sigma_ln_da
            self.rho = mock.rho
            self.errors_include_intrinsic = mock.errors_include_intrinsic
            return

        warnings.warn(
            "loading legacy absolute-error text datasets; regenerate paired NPZ "
            "data for explicit log-error semantics",
            DeprecationWarning,
            stacklevel=2,
        )
        dl_data = np.atleast_2d(np.loadtxt(self.datapath + '_dl_dataset.txt'))
        da_data = np.atleast_2d(np.loadtxt(self.datapath + '_da_dataset.txt'))
        if dl_data.shape[0] != da_data.shape[0] or not np.allclose(
            dl_data[:, 0], da_data[:, 0]
        ):
            raise ValueError("DL and DA datasets must contain identical redshifts")
        self.z = dl_data[:, 0]
        self.dl = dl_data[:, 1]
        self.da = da_data[:, 1]
        if np.any(self.dl <= 0) or np.any(self.da <= 0):
            raise ValueError("legacy distances must be positive")
        self.sigma_ln_dl = dl_data[:, 2] / self.dl
        self.sigma_ln_da = da_data[:, 2] / self.da
        self.rho = np.zeros_like(self.z)
        self.errors_include_intrinsic = True

    def get_can_support_params(self):
        """Declare direct likelihood inputs understood by real Cobaya."""
        return ("alpha", "beta", "calibration_offset", "sigma_int")

    def get_requirements(self):
        # alpha, beta, calibration_offset, and sigma_int are likelihood
        # parameters supplied by Cobaya, not theory-provider requirements.
        return {}

    def logp(
        self,
        alpha=0.0,
        beta=0.0,
        calibration_offset=0.0,
        sigma_int=0.0,
        **params_values,
    ):
        if self.errors_include_intrinsic and sigma_int > 0:
            raise ValueError(
                "dataset errors already include intrinsic scatter; regenerate "
                "with errors_include_intrinsic=False before sampling sigma_int"
            )
        return paired_distance_log_likelihood(
            self.dl,
            self.da,
            self.z,
            self.sigma_ln_dl,
            self.sigma_ln_da,
            model=self.EMmodel,
            alpha=alpha,
            beta=beta,
            rho=self.rho,
            calibration_offset=calibration_offset,
            calibration_sigma=self.calibration_sigma,
            sigma_int_ln_eta=sigma_int,
        )
